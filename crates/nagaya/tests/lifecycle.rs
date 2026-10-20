use std::os::unix::net::UnixDatagram;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use nagaya::admission::{Factory, OpenError, OpenErrorKind, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::lifecycle::{self, LifecycleError};
use nagaya::notify::Notifier;
use nagaya::process::Event;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
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
        store: Some(StoreSpec {
            nvme: 65536,
            host: None,
        }),
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
fn the_open_wait_outlasts_the_watchdog() {
    assert!(
        lifecycle::DEFAULT_OPEN_TIMEOUT > nagaya::process::DEFAULT_WATCHDOG_TIMEOUT,
        "the watchdog must report a stalled open before the wait gives up"
    );
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
        directory.path(),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(ready.opened.len(), 1);
    assert_eq!(receive(&socket), "STATUS=engine fake-one loading");
    assert_eq!(receive(&socket), "READY=1\nSTATUS=engine fake-one ready");
    let report = ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
}

#[test]
fn the_stop_reports_draining_then_unloaded() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let mut ready = lifecycle::start(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        directory.path(),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(receive(&socket), "STATUS=engine fake-one loading");
    assert_eq!(receive(&socket), "READY=1\nSTATUS=engine fake-one ready");
    let report = lifecycle::stop(&mut ready, Some(&notifier), Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean);
    assert_eq!(receive(&socket), "STATUS=engine fake-one draining");
    assert_eq!(receive(&socket), "STATUS=engine fake-one unloaded");
}

#[test]
fn an_empty_profile_is_ready() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let profile = PlacementProfile {
        engines: Vec::new(),
        budgets: Vec::new(),
        store: None,
    };
    let mut ready = lifecycle::start(
        profile,
        Registry::new(),
        inventory(&[]),
        directory.path(),
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
    kind: OpenErrorKind,
}

impl Factory for FailingFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        Err(OpenError {
            kind: self.kind,
            message: "the weights are absent".to_owned(),
        })
    }
}

#[test]
fn a_plain_open_failure_is_a_crash() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let mut registry = Registry::new();
    registry.register(
        "broken",
        Box::new(FailingFactory {
            limits: limits_on(&[0]),
            kind: OpenErrorKind::Other,
        }),
    );
    let error = lifecycle::start(
        profile(vec![engine("broken-one", "broken", &[0])]),
        registry,
        inventory(&[0]),
        directory.path(),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::Crash);
    assert_eq!(receive(&socket), "STATUS=engine broken-one loading");
    assert_eq!(
        receive(&socket),
        "STATUS=engine broken-one fatal: the weights are absent"
    );
}

#[test]
fn an_identity_pin_or_part_size_mismatch_exits_three() {
    for kind in [OpenErrorKind::Identity, OpenErrorKind::Pin, OpenErrorKind::PartSize] {
        let directory = tempfile::tempdir().unwrap();
        let mut registry = Registry::new();
        registry.register(
            "broken",
            Box::new(FailingFactory {
                limits: limits_on(&[0]),
                kind,
            }),
        );
        let error = lifecycle::start(
            profile(vec![engine("broken-one", "broken", &[0])]),
            registry,
            inventory(&[0]),
            directory.path(),
            None,
            Duration::from_secs(2),
        )
        .unwrap_err();
        assert_eq!(error.exit_code(), ExitCode::RefusedStart, "{kind:?}");
    }
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
    let directory = tempfile::tempdir().unwrap();
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
        directory.path(),
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
    let directory = tempfile::tempdir().unwrap();
    let mut ready = lifecycle::start(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        directory.path(),
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

/// Start the fake profile and give a wait with a tick that returns `code` after it raises the signal when asked.
fn wait_with_tick(raise: bool, code: ExitCode) -> ExitCode {
    let directory = tempfile::tempdir().unwrap();
    let mut ready = lifecycle::start(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        directory.path(),
        None,
        Duration::from_secs(2),
    )
    .unwrap();
    let flag = SignalFlag::new();
    let mut tick = || {
        if raise {
            flag.raise();
        }
        Some(code)
    };
    let result = lifecycle::wait_with(&ready.process, &flag, None, Duration::from_millis(5), &mut tick);
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
    result
}

#[test]
fn a_planned_restart_from_the_tick_gives_exit_four() {
    assert_eq!(
        wait_with_tick(false, ExitCode::PlannedRestart),
        ExitCode::PlannedRestart
    );
}

#[test]
fn a_signal_during_the_tick_beats_a_planned_restart() {
    // An operator stop is not a planned restart, also when a profile was persisted in the same tick.
    assert_eq!(wait_with_tick(true, ExitCode::PlannedRestart), ExitCode::Clean);
}

#[test]
fn a_signal_during_the_tick_does_not_hide_a_crash() {
    assert_eq!(wait_with_tick(true, ExitCode::Crash), ExitCode::Crash);
}

struct SlowFactory {
    limits: Limits,
}

impl Factory for SlowFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        thread::sleep(Duration::from_millis(400));
        Ok(Box::new(FakeAlways::new()))
    }
}

#[test]
fn an_open_that_outlasts_the_wait_is_a_crash_that_names_the_engine() {
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(
        "slow",
        Box::new(SlowFactory {
            limits: limits_on(&[0]),
        }),
    );
    let error = lifecycle::start(
        profile(vec![engine("slow-one", "slow", &[0])]),
        registry,
        inventory(&[0]),
        directory.path(),
        None,
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::Crash);
    assert!(error.to_string().contains("slow-one"), "{error}");
}

#[test]
fn closed_group_channels_are_a_crash() {
    assert_eq!(lifecycle::LifecycleError::Disconnected.exit_code(), ExitCode::Crash);
}
