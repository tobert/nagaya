//! The process lifecycle records its engine states and its spans from `lifecycle::start` and `lifecycle::stop`
//! (spec 9.2, 9.4, 9.5).

use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::lifecycle;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Coverage, Engine, Identity, Limits, OutputsWanted, PartError, Parts, PassError, PassOut, Seal, Segment,
    SegmentInputs, Slots, State, Tap, Tier, WriteResult,
};
use nagaya_telemetry::testing::{Capture, SpanData};

fn engine(name: &str, kind: &str) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: vec![0],
        config: toml::Table::new(),
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        residency: Default::default(),
        pinned: false,
    }
}

fn profile(engines: Vec<EnginePlacement>, nvme: u64) -> PlacementProfile {
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
            nvme,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1024,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1024,
    }
}

/// A fake whose slot 0 holds one token. A stop parks only a state with tokens (spec 10.3).
fn fake_with_a_token() -> FakeAlways {
    let mut fake = FakeAlways::new();
    let segment = Segment {
        control: Vec::new(),
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(vec![7]),
        positions: vec![0],
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    };
    fake.pass(&[segment], &[]).expect("the fill pass runs");
    fake
}

/// A factory whose engine holds one token, so a stop has a state to park.
struct TokenFactory;

impl Factory for TokenFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(fake_with_a_token()))
    }
}

struct FailingFactory;

impl Factory for FailingFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Err(OpenError {
            kind: OpenErrorKind::Other,
            message: "the weights are absent".to_owned(),
        })
    }
}

/// A factory whose engine opens when the test says so.
struct GatedFactory {
    gate: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}

impl Factory for GatedFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        self.gate.lock().unwrap().recv().expect("the test opens the gate");
        Ok(Box::new(FakeAlways::new()))
    }
}

/// The value of `nagaya.engine.state` for one engine and one state.
fn state(capture: &Capture, engine: &str, state: &str) -> Option<f64> {
    capture
        .metric("nagaya.engine.state")
        .into_iter()
        .find(|point| point.attributes["engine"] == engine && point.attributes["state"] == state)
        .map(|point| point.value)
}

fn span(capture: &Capture, name: &str) -> SpanData {
    capture
        .spans()
        .into_iter()
        .find(|span| span.name == name)
        .unwrap_or_else(|| panic!("no span named {name}"))
}

fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|item| item.key.as_str() == key)
        .map(|item| item.value.to_string())
}

fn start_fake(directory: &std::path::Path, nvme: u64) -> lifecycle::Ready {
    lifecycle::start(
        profile(vec![engine("fake-one", "fake")], nvme),
        Registry::with_fake(),
        inventory(),
        directory,
        None,
        Duration::from_secs(2),
    )
    .unwrap()
}

#[test]
fn start_records_loading_and_then_ready_for_each_engine() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start_fake(directory.path(), 65536);

    assert_eq!(state(&capture, "fake-one", "ready"), Some(1.0));
    assert_eq!(state(&capture, "fake-one", "loading"), Some(0.0));
    assert_eq!(state(&capture, "fake-one", "fatal"), Some(0.0));
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn start_makes_the_span_nagaya_start() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start_fake(directory.path(), 65536);

    let start = span(&capture, "nagaya.start");
    assert_eq!(attribute(&start, "engines").as_deref(), Some("1"));
    assert_eq!(attribute(&start, "outcome").as_deref(), Some("ready"));
    assert!(!matches!(start.status, opentelemetry::trace::Status::Error { .. }));
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_failed_open_records_fatal_and_marks_the_span_as_an_error() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("broken", Box::new(FailingFactory));
    let error = lifecycle::start(
        profile(vec![engine("broken-one", "broken")], 65536),
        registry,
        inventory(),
        directory.path(),
        None,
        Duration::from_secs(2),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), nagaya::exit::ExitCode::Crash);

    assert_eq!(state(&capture, "broken-one", "fatal"), Some(1.0));
    assert_eq!(state(&capture, "broken-one", "ready"), Some(0.0));
    let start = span(&capture, "nagaya.start");
    assert_eq!(attribute(&start, "outcome").as_deref(), Some("failed"));
    assert!(
        matches!(start.status, opentelemetry::trace::Status::Error { .. }),
        "{:?}",
        start.status
    );
}

#[test]
fn a_profile_the_ledger_refuses_is_a_failed_start() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut profile = profile(vec![engine("fake-one", "fake")], 65536);
    // The device budget is too small for the fake engine.
    profile.budgets[0].bytes = 1;
    let error = lifecycle::start(
        profile,
        Registry::with_fake(),
        inventory(),
        directory.path(),
        None,
        Duration::from_secs(2),
    )
    .unwrap_err();
    assert_eq!(error.exit_code(), nagaya::exit::ExitCode::RefusedStart);
    let start = span(&capture, "nagaya.start");
    assert_eq!(attribute(&start, "outcome").as_deref(), Some("failed"));
}

#[test]
fn a_clean_stop_records_unloaded_and_makes_the_span_nagaya_stop() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start_fake(directory.path(), 65536);
    let report = lifecycle::stop(&mut ready, None, Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), nagaya::exit::ExitCode::Clean, "{report:?}");

    assert_eq!(state(&capture, "fake-one", "unloaded"), Some(1.0));
    assert_eq!(state(&capture, "fake-one", "draining"), Some(0.0));
    assert_eq!(state(&capture, "fake-one", "ready"), Some(0.0));
    let stop = span(&capture, "nagaya.stop");
    assert_eq!(attribute(&stop, "engines").as_deref(), Some("1"));
    assert_eq!(attribute(&stop, "outcome").as_deref(), Some("clean"));
}

#[test]
fn a_stop_that_cannot_park_records_fatal_and_marks_the_span_as_an_error() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    // The store has no room for the parked state.
    let mut registry = Registry::new();
    registry.register("token", Box::new(TokenFactory));
    let mut ready = lifecycle::start(
        profile(vec![engine("fake-one", "token")], 1),
        registry,
        inventory(),
        directory.path(),
        None,
        Duration::from_secs(2),
    )
    .unwrap();
    let report = lifecycle::stop(&mut ready, None, Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), nagaya::exit::ExitCode::Crash, "{report:?}");

    assert_eq!(state(&capture, "fake-one", "fatal"), Some(1.0));
    assert_eq!(state(&capture, "fake-one", "unloaded"), Some(0.0));
    let stop = span(&capture, "nagaya.stop");
    assert_eq!(attribute(&stop, "outcome").as_deref(), Some("failed"));
    assert!(
        matches!(stop.status, opentelemetry::trace::Status::Error { .. }),
        "{:?}",
        stop.status
    );
}

#[test]
fn an_engine_that_is_still_opening_is_loading() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let (open, gate) = std::sync::mpsc::channel();
    let mut registry = Registry::new();
    registry.register(
        "gated",
        Box::new(GatedFactory {
            gate: std::sync::Mutex::new(gate),
        }),
    );
    let instruments = capture.instruments();
    let path = directory.path().to_owned();
    let starter = std::thread::spawn(move || {
        let _scope = nagaya_telemetry::scope(instruments);
        lifecycle::start(
            profile(vec![engine("gated-one", "gated")], 65536),
            registry,
            inventory(),
            &path,
            None,
            Duration::from_secs(5),
        )
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while state(&capture, "gated-one", "loading") != Some(1.0) {
        assert!(
            std::time::Instant::now() < deadline,
            "the engine never showed as loading"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        state(&capture, "gated-one", "ready"),
        Some(0.0),
        "the engine has not opened yet"
    );

    open.send(()).unwrap();
    let mut ready = starter.join().unwrap().expect("the engine opens");
    assert_eq!(state(&capture, "gated-one", "ready"), Some(1.0));
    assert_eq!(state(&capture, "gated-one", "loading"), Some(0.0));
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

/// An engine that blocks its first part read until the test drops the sender. The stop is then in its drain.
struct ParkGate {
    inner: FakeAlways,
    gate: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<()>>>,
}

struct ParkGateFactory {
    gate: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<()>>>,
}

impl Factory for ParkGateFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(ParkGate {
            inner: fake_with_a_token(),
            gate: std::sync::Arc::clone(&self.gate),
        }))
    }
}

impl Engine for ParkGate {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Parts for ParkGate {
    fn part_read(&mut self, state: State, part: usize, out: &mut [u8]) -> Result<(), PartError> {
        // A closed channel ends the wait at once.
        let _ = self.gate.lock().unwrap().recv();
        self.inner.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: usize, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: usize, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: usize, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

#[test]
fn an_engine_that_is_still_parking_is_draining() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let (release, gate) = std::sync::mpsc::channel::<()>();
    let mut registry = Registry::new();
    registry.register(
        "parkgate",
        Box::new(ParkGateFactory {
            gate: std::sync::Arc::new(std::sync::Mutex::new(gate)),
        }),
    );
    let mut ready = lifecycle::start(
        profile(vec![engine("gate-one", "parkgate")], 65536),
        registry,
        inventory(),
        directory.path(),
        None,
        Duration::from_secs(2),
    )
    .unwrap();
    let instruments = capture.instruments();
    let stopper = std::thread::spawn(move || {
        let _scope = nagaya_telemetry::scope(instruments);
        let report = lifecycle::stop(&mut ready, None, Duration::from_secs(10), &SignalFlag::new());
        (ready, report)
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while state(&capture, "gate-one", "draining") != Some(1.0) {
        assert!(
            std::time::Instant::now() < deadline,
            "the engine never showed as draining"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        state(&capture, "gate-one", "unloaded"),
        Some(0.0),
        "the park has not ended"
    );

    drop(release);
    let (_ready, report) = stopper.join().unwrap();
    assert_eq!(report.exit_code(), nagaya::exit::ExitCode::Clean, "{report:?}");
    assert_eq!(state(&capture, "gate-one", "unloaded"), Some(1.0));
    assert_eq!(state(&capture, "gate-one", "draining"), Some(0.0));
}

/// A factory whose engine opens after a short sleep.
struct SlowFactory;

impl Factory for SlowFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        std::thread::sleep(Duration::from_millis(400));
        Ok(Box::new(FakeAlways::new()))
    }
}

#[test]
fn an_engine_that_does_not_open_in_time_is_fatal() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("slow", Box::new(SlowFactory));
    let error = lifecycle::start(
        profile(vec![engine("slow-one", "slow")], 65536),
        registry,
        inventory(),
        directory.path(),
        None,
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert!(matches!(error, lifecycle::LifecycleError::Timeout { .. }), "{error}");

    assert_eq!(state(&capture, "slow-one", "fatal"), Some(1.0));
    assert_eq!(state(&capture, "slow-one", "loading"), Some(0.0));
    assert_eq!(
        attribute(&span(&capture, "nagaya.start"), "outcome").as_deref(),
        Some("failed")
    );
}
