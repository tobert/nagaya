use std::os::unix::net::UnixDatagram;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use nagaya::admission::{Factory, OpenError, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::lifecycle::{self, LifecycleError};
use nagaya::notify::Notifier;
use nagaya::process::Event;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{DeviceId, Engine, Limits, Tier};

fn limits_on(devices: &[DeviceId]) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.devices = devices.to_vec();
    limits
}

fn engine(name: &str, kind: &str, devices: &[DeviceId]) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: devices.to_vec(),
        config: toml::Table::new(),
        share: 1,
        pin_identity: false,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    let bytes = engines.len() as u64 * 4096;
    PlacementProfile {
        engines,
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes,
            },
        ],
    }
}

fn fake_profile() -> PlacementProfile {
    profile(vec![engine("fake-one", "fake", &[0])])
}

fn inventory(ids: &[DeviceId]) -> Inventory {
    Inventory {
        devices: ids
            .iter()
            .map(|id| Device {
                id: *id,
                name: format!("device-{id}"),
                memory_bytes: 1024,
                unified: false,
            })
            .collect(),
        host_bytes: 1024,
    }
}

fn bound(directory: &Path) -> (UnixDatagram, std::path::PathBuf) {
    let path = directory.join("notify");
    let socket = UnixDatagram::bind(&path).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    (socket, path)
}

fn receive(socket: &UnixDatagram) -> String {
    let mut buffer = [0u8; 512];
    let count = socket.recv(&mut buffer).unwrap();
    std::str::from_utf8(&buffer[..count]).unwrap().to_owned()
}

#[test]
fn a_missing_profile_is_empty() {
    let directory = tempfile::tempdir().unwrap();
    let profile = lifecycle::active_profile(directory.path()).unwrap();
    assert!(profile.engines.is_empty());
    assert!(profile.budgets.is_empty());
}

#[test]
fn an_accepted_profile_is_read() {
    let directory = tempfile::tempdir().unwrap();
    let written = profile(vec![engine("a", "fake", &[0])]);
    written.write(directory.path()).unwrap();
    assert_eq!(lifecycle::active_profile(directory.path()).unwrap(), written);
}

#[test]
fn a_profile_that_does_not_parse_is_an_error() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join(nagaya::profile::PROFILE_FILE),
        "this is not TOML =\n",
    )
    .unwrap();
    let error = lifecycle::active_profile(directory.path()).unwrap_err();
    assert!(error.to_string().contains("not valid"), "{error}");
}

#[test]
fn ready_reaches_the_socket() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let mut ready = lifecycle::start(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(ready.opened.len(), 1);
    assert_eq!(receive(&socket), "READY=1\nSTATUS=engines opened: fake-one");
    let report = ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean);
}

#[test]
fn an_empty_profile_is_ready() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let profile = PlacementProfile {
        engines: Vec::new(),
        budgets: Vec::new(),
    };
    let mut ready = lifecycle::start(
        profile,
        Registry::new(),
        inventory(&[]),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap();
    assert!(ready.opened.is_empty());
    assert_eq!(receive(&socket), "READY=1\nSTATUS=no engines in the profile");
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

struct FailingFactory {
    limits: Limits,
}

impl Factory for FailingFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        Err(OpenError {
            message: "the weights are absent".to_owned(),
        })
    }
}

#[test]
fn an_open_failure_gets_status_and_exit_three() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let mut registry = Registry::new();
    registry.register(
        "broken",
        Box::new(FailingFactory {
            limits: limits_on(&[0]),
        }),
    );
    let error = lifecycle::start(
        profile(vec![engine("broken-one", "broken", &[0])]),
        registry,
        inventory(&[0]),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
    assert_eq!(
        receive(&socket),
        "STATUS=engine broken-one failed to open: the weights are absent"
    );
}

struct PanicFactory {
    limits: Limits,
}

impl Factory for PanicFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        panic!("the engine cannot open")
    }
}

#[test]
fn a_group_panic_exits_one() {
    let mut registry = Registry::new();
    registry.register(
        "panic",
        Box::new(PanicFactory {
            limits: limits_on(&[0]),
        }),
    );
    let error = lifecycle::start(
        profile(vec![engine("panic-one", "panic", &[0])]),
        registry,
        inventory(&[0]),
        None,
        Duration::from_secs(2),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::Crash);
    match error {
        LifecycleError::Event(Event::Panicked { thread, message }) => {
            assert_eq!(thread, "group");
            assert_eq!(message, "the engine cannot open");
        }
        other => panic!("expected a group panic, got {other}"),
    }
}

#[test]
fn a_signal_ends_the_wait_with_exit_zero() {
    let mut ready = lifecycle::start(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        None,
        Duration::from_secs(2),
    )
    .unwrap();
    let flag = Arc::new(SignalFlag::new());
    let raiser = Arc::clone(&flag);
    let tick = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        raiser.raise();
    });
    let code = lifecycle::wait(&ready.process, &flag, None, Duration::from_millis(5));
    tick.join().unwrap();
    assert_eq!(code, ExitCode::Clean);
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}
