use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use nagaya::admission::{AdmitError, Factory, OpenError, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::process::Start;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Coverage, DeviceId, Engine, Identity, Limits, PartError, Parts, PassError, PassOut, Seal, Segment, Slots, State,
    Tap, Tier, WriteResult,
};

#[derive(Default)]
struct ParkRecord {
    reads: Vec<(String, State, usize, ThreadId)>,
}

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

struct ParkFactory {
    name: String,
    limits: Limits,
    record: Arc<Mutex<ParkRecord>>,
    delay: Duration,
    fail: bool,
}

impl ParkFactory {
    fn new(name: &str, record: Arc<Mutex<ParkRecord>>) -> Self {
        Self {
            name: name.to_owned(),
            limits: limits_on(&[0]),
            record,
            delay: Duration::ZERO,
            fail: false,
        }
    }

    fn slow(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn failing(mut self) -> Self {
        self.fail = true;
        self
    }
}

impl Factory for ParkFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(ParkEngine {
            inner: FakeAlways::new(),
            limits: self.limits.clone(),
            name: self.name.clone(),
            record: Arc::clone(&self.record),
            delay: self.delay,
            fail: self.fail,
        }))
    }
}

struct ParkEngine {
    inner: FakeAlways,
    limits: Limits,
    name: String,
    record: Arc<Mutex<ParkRecord>>,
    delay: Duration,
    fail: bool,
}

impl Engine for ParkEngine {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
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

impl Parts for ParkEngine {
    fn part_read(&mut self, state: State, part: usize, out: &mut [u8]) -> Result<(), PartError> {
        if !self.delay.is_zero() {
            thread::sleep(self.delay);
        }
        self.record
            .lock()
            .unwrap()
            .reads
            .push((self.name.clone(), state, part, thread::current().id()));
        if self.fail {
            return Err(PartError::Refused);
        }
        self.inner.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: usize, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: usize, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: usize, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

fn started(factory: ParkFactory) -> (nagaya::process::Process, Vec<nagaya::process::OpenedEngine>) {
    let mut registry = Registry::new();
    registry.register("park", Box::new(factory));
    let profile = profile(vec![engine("a", "park", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(2)).unwrap();
    (process, opened)
}

#[test]
fn a_stop_parks_every_engine() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, opened) = started(ParkFactory::new("a", Arc::clone(&record)));
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert!(report.complete, "every group answered");
    assert_eq!(report.exit_code(), ExitCode::Clean);
    assert_eq!(report.parked.engines, 1);
    assert_eq!(report.parked.states, 2, "the fake declares two slots");
    assert_eq!(report.parked.parts, 6, "three stored parts in two states");
    assert_eq!(report.parked.bytes, 32, "two fixed parts of 16 bytes");
    assert_eq!(report.parked.unmeasured_row_parts, 0, "the fake measures its row parts");
    assert!(report.parked.failures.is_empty());
    let record = record.lock().unwrap();
    assert_eq!(record.reads.len(), 6);
    assert!(
        record.reads.iter().all(|read| read.3 == opened[0].thread),
        "every park read runs on the group thread"
    );
    drop(record);
    let deadline = Instant::now() + Duration::from_secs(2);
    while !process.finished() {
        assert!(Instant::now() < deadline, "every worker ends after the stop");
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_second_signal_stops_at_once() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _) = started(ParkFactory::new("a", Arc::clone(&record)).slow(Duration::from_millis(100)));
    let flag = Arc::new(SignalFlag::new());
    let raiser = Arc::clone(&flag);
    let tick = thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        raiser.raise();
    });
    let started = Instant::now();
    let report = process.stop(Duration::from_secs(10), &flag);
    let elapsed = started.elapsed();
    tick.join().unwrap();
    assert!(report.interrupted, "the second signal stopped the park");
    assert!(!report.complete);
    assert!(
        elapsed < Duration::from_millis(200),
        "the second signal exits at once, but the stop waited {elapsed:?}"
    );
}

#[test]
fn a_stop_takes_no_more_admissions() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _) = started(ParkFactory::new("a", record));
    process.stop(Duration::from_secs(2), &SignalFlag::new());
    let error = process.admit(profile(vec![engine("a", "park", &[0])])).unwrap_err();
    assert_eq!(error.to_string(), "the process is stopping");
    assert!(matches!(error, AdmitError::Stopping));
}

#[test]
fn two_signals_before_the_stop_park_nothing() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _) = started(ParkFactory::new("a", Arc::clone(&record)).slow(Duration::from_millis(100)));
    let flag = SignalFlag::new();
    flag.raise();
    flag.raise();
    let started = Instant::now();
    let report = process.stop(Duration::from_secs(10), &flag);
    assert!(report.interrupted);
    assert!(!report.complete);
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "a second signal exits at once"
    );
    assert!(
        record.lock().unwrap().reads.is_empty(),
        "an interrupted stop parks nothing"
    );
}

#[test]
fn an_abort_parks_nothing() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _) = started(ParkFactory::new("a", Arc::clone(&record)));
    process.abort();
    let error = process.admit(profile(vec![engine("a", "park", &[0])])).unwrap_err();
    assert!(matches!(error, AdmitError::Stopping));
    // The drop joins every worker, so the record is complete here.
    drop(process);
    assert!(record.lock().unwrap().reads.is_empty(), "an abort parks nothing");
}

#[test]
fn a_stop_that_writes_nothing_says_so() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _) = started(ParkFactory::new("a", record));
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    let line = report.log_line();
    assert!(
        line.contains("parked nothing"),
        "the stop must not claim a park: {line}"
    );
}

#[test]
fn a_park_failure_is_a_crash() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _) = started(ParkFactory::new("a", record).failing());
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.parked.failures.len(), 1, "{report:?}");
    assert!(report.parked.failures[0].contains("state 0"), "{report:?}");
    assert_eq!(report.exit_code(), ExitCode::Crash);
}
