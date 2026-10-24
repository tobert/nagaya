use std::os::unix::net::UnixDatagram;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError, Registry};
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
    // The engine declares group indexes, one for each machine device of the profile (spec 2.4).
    limits.devices = (0..devices.len() as DeviceId).collect();
    limits
}

fn engine(name: &str, kind: &str, devices: &[DeviceId]) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: devices.to_vec(),
        config: toml::Table::new(),
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency: Default::default(),
        idle_unload_s: None,
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
            interval_copies: 0,
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
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 1024,
                unified: false,
                ignored: false,
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
fn the_open_wait_is_no_shorter_than_the_unit_start_timeout() {
    // A group beats while its engine opens, so the watchdog cannot end a hung open. The wait is the backstop (7.4).
    let unit =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/nagaya.service")).unwrap();
    let seconds: u64 = unit
        .lines()
        .find_map(|line| line.strip_prefix("TimeoutStartSec="))
        .expect("the unit sets TimeoutStartSec")
        .parse()
        .unwrap();
    assert!(
        lifecycle::DEFAULT_OPEN_TIMEOUT >= Duration::from_secs(seconds),
        "the wait for the opens ends before the unit's start timeout"
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
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
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
fn an_identity_or_part_size_mismatch_exits_three() {
    for kind in [OpenErrorKind::Identity, OpenErrorKind::PartSize] {
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
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
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
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
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

#[test]
fn an_engine_with_no_segments_exits_three_at_cold_start() {
    // Review finding 8.
    struct Zero;
    impl Factory for Zero {
        fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            let mut limits = limits_on(&[0]);
            limits.max_segments = 0;
            Ok(limits)
        }
        fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            panic!("an engine that admission refused is never opened")
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("zero", Box::new(Zero));
    let error = lifecycle::start(
        profile(vec![engine("no-passes", "zero", &[0])]),
        registry,
        inventory(&[0]),
        directory.path(),
        None,
        Duration::from_secs(2),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
    let text = error.to_string();
    assert!(
        text.contains("`no-passes`") && text.contains("`max_segments`"),
        "{text}"
    );
}

fn slow_registry(open_time: Duration) -> Registry {
    let mut registry = Registry::new();
    registry.register(
        "slow",
        Box::new(OpenTimeFactory {
            limits: limits_on(&[0]),
            open_time,
        }),
    );
    registry
}

struct OpenTimeFactory {
    limits: Limits,
    open_time: Duration,
}

impl Factory for OpenTimeFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        thread::sleep(self.open_time);
        Ok(Box::new(FakeAlways::new()))
    }
}

#[test]
fn a_slow_open_extends_the_start_timeout_with_a_status() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let options = lifecycle::Options::new(Duration::from_secs(5))
        .tick(Duration::from_millis(40))
        .extend(Duration::from_secs(30));
    let mut ready = lifecycle::start_with(
        profile(vec![engine("slow-one", "slow", &[0])]),
        slow_registry(Duration::from_millis(300)),
        inventory(&[0]),
        directory.path(),
        Some(&notifier),
        options,
    )
    .unwrap();
    assert_eq!(receive(&socket), "STATUS=engine slow-one loading");
    let mut extensions = 0;
    loop {
        let text = receive(&socket);
        if text.starts_with("READY=1") {
            assert_eq!(text, "READY=1\nSTATUS=engine slow-one ready");
            break;
        }
        assert_eq!(text, "EXTEND_TIMEOUT_USEC=30000000\nSTATUS=engine slow-one loading");
        extensions += 1;
    }
    assert!(
        extensions >= 3,
        "a 300 ms open with a 40 ms tick extended {extensions} times"
    );
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_fast_open_sends_no_extension() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let options = lifecycle::Options::new(Duration::from_secs(5)).tick(Duration::from_secs(1));
    let mut ready = lifecycle::start_with(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        directory.path(),
        Some(&notifier),
        options,
    )
    .unwrap();
    assert_eq!(receive(&socket), "STATUS=engine fake-one loading");
    assert_eq!(receive(&socket), "READY=1\nSTATUS=engine fake-one ready");
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_start_timeout_sends_a_status_that_names_the_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let options = lifecycle::Options::new(Duration::from_millis(150)).tick(Duration::from_secs(1));
    let error = lifecycle::start_with(
        profile(vec![engine("slow-one", "slow", &[0])]),
        slow_registry(Duration::from_millis(600)),
        inventory(&[0]),
        directory.path(),
        Some(&notifier),
        options,
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::Crash);
    assert_eq!(receive(&socket), "STATUS=engine slow-one loading");
    assert_eq!(receive(&socket), "STATUS=engine slow-one did not open in time");
}

#[test]
fn a_held_ready_waits_for_the_caller() {
    // Spec 10.1: the process sends READY=1 after the council routes attach, not when the engines open.
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    let options = lifecycle::Options::new(Duration::from_secs(2)).hold_ready();
    let mut ready = lifecycle::start_with(
        fake_profile(),
        Registry::with_fake(),
        inventory(&[0]),
        directory.path(),
        Some(&notifier),
        options,
    )
    .unwrap();
    assert_eq!(receive(&socket), "STATUS=engine fake-one loading");
    socket.set_nonblocking(true).unwrap();
    let mut buffer = [0u8; 512];
    let early = socket.recv(&mut buffer);
    assert!(
        early.is_err(),
        "nothing else before the caller sends READY: {:?}",
        early.map(|count| String::from_utf8_lossy(&buffer[..count]).into_owned())
    );
    socket.set_nonblocking(false).unwrap();
    lifecycle::notify_ready(Some(&notifier), &ready.opened);
    assert_eq!(receive(&socket), "READY=1\nSTATUS=engine fake-one ready");
    let report = ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
}
