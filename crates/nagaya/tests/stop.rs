use std::os::unix::net::UnixDatagram;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use nagaya::admission::{AdmitError, Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::lifecycle;
use nagaya::notify::Notifier;
use nagaya::park::StoreBudget;
use nagaya::process::{ParkReport, Quantum, QuantumOutcome, Start, StartError, StopReport, SubmitError};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, DeviceId, Engine, Identity, Limits, OutputsWanted, PartError, Parts, PassError, PassOut,
    Seal, Segment, SegmentInputs, SlotLimit, Slots, State, Tap, Tier, WriteResult,
};
use nagaya_store::{
    EngineFacts, Put, Reason, RecipeId, Snapshot, Store, Tensor, Tier as StoreTier, recipe_root, snapshot_from_engine,
};

#[derive(Default)]
struct ParkRecord {
    reads: Vec<(String, State, usize, ThreadId)>,
    events: Vec<String>,
}

/// A fake with two slots. A `ParkFactory` fills each slot with one token, so a stop parks both.
fn two_slots() -> FakeAlways {
    FakeAlways::with_slots(vec![
        SlotLimit {
            bytes: 64,
            count: 1,
            tokens: 4096,
        },
        SlotLimit {
            bytes: 256,
            count: 1,
            tokens: 4096,
        },
    ])
}

fn limits_on(devices: &[DeviceId]) -> Limits {
    let mut limits = two_slots().limits();
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
        lab: toml::Table::new(),
        share: 1,
        residency: Default::default(),
        pinned: false,
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
            objects: None,
        }),
        cpu: None,
    }
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

fn segment(state: State, tokens: &[u32]) -> Segment {
    Segment {
        control: Vec::new(),
        state,
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (0..tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
        layer_loop: None,
        cycles: None,
    }
}

struct ParkFactory {
    name: String,
    limits: Limits,
    record: Arc<Mutex<ParkRecord>>,
    delay: Duration,
    fail: bool,
    no_slots: bool,
    misfit: bool,
    filled: bool,
}

impl ParkFactory {
    fn new(name: &str, record: Arc<Mutex<ParkRecord>>) -> Self {
        Self {
            name: name.to_owned(),
            limits: limits_on(&[0]),
            record,
            delay: Duration::ZERO,
            fail: false,
            no_slots: false,
            misfit: false,
            filled: true,
        }
    }

    /// Leave every slot empty at open. A slot with no token has nothing for the park to keep (spec 10.3).
    fn empty(mut self) -> Self {
        self.filled = false;
        self
    }

    /// Make the coverage capability refuse every span with `fits`.
    fn misfit(mut self) -> Self {
        self.misfit = true;
        self
    }

    fn slow(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn failing(mut self) -> Self {
        self.fail = true;
        self
    }

    /// Remove the slots capability, so a row part has no row count.
    fn without_slots(mut self) -> Self {
        self.no_slots = true;
        self.limits
            .capabilities
            .retain(|capability| *capability != Capability::Slots);
        self
    }
}

impl Factory for ParkFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let mut inner = two_slots();
        if self.filled {
            for slot in 0..2 {
                inner
                    .pass(&[segment(State::Slot(slot), &[slot + 1])], &[])
                    .expect("the fill pass runs");
            }
        }
        Ok(Box::new(ParkEngine {
            inner,
            limits: self.limits.clone(),
            name: self.name.clone(),
            record: Arc::clone(&self.record),
            delay: self.delay,
            fail: self.fail,
            no_slots: self.no_slots,
            misfit: self.misfit.then_some(MisfitCoverage),
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
    no_slots: bool,
    misfit: Option<MisfitCoverage>,
}

/// A coverage capability for which no span fits any slot.
struct MisfitCoverage;

impl Coverage for MisfitCoverage {
    fn coverage(&self, _state: State, part: usize, _bytes: &[u8]) -> nagaya_abi::CoverageSpan {
        nagaya_abi::CoverageSpan { part, start: 0, end: 0 }
    }

    fn fits(&self, _coverage: &nagaya_abi::CoverageSpan, _slot: u32) -> Result<(), String> {
        Err("the span is too large for the slot".to_owned())
    }
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
        match self.misfit.as_mut() {
            Some(misfit) => Some(misfit),
            None => self.inner.coverage(),
        }
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        if self.no_slots {
            return None;
        }
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
        self.record.lock().unwrap().events.push("read".to_owned());
        if self.fail {
            return Err(PartError::Refused);
        }
        self.inner.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: usize, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: usize, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.record.lock().unwrap().events.push("cells_read".to_owned());
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: usize, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.record.lock().unwrap().events.push("cells_write".to_owned());
        self.inner.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

fn started(
    factory: ParkFactory,
) -> (
    nagaya::process::Process,
    Vec<nagaya::process::OpenedEngine>,
    tempfile::TempDir,
) {
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("park", Box::new(factory));
    let profile = profile(vec![engine("a", "park", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(2)).unwrap();
    (process, opened, directory)
}

#[test]
fn a_stop_parks_every_engine() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, opened, _directory) = started(ParkFactory::new("a", Arc::clone(&record)));
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert!(report.complete, "every group answered");
    assert_eq!(report.exit_code(), ExitCode::Clean);
    assert_eq!(report.parked.engines, 1);
    assert_eq!(
        report.parked.states, 2,
        "the fake declares two slots, and each holds a token"
    );
    assert_eq!(report.parked.parts, 6, "three stored parts in two states");
    assert_eq!(
        report.parked.bytes, 64,
        "in each state, a fixed part of 16 bytes, one cells row and one ring row of 8 bytes"
    );
    assert!(report.parked.failures.is_empty());
    let record = record.lock().unwrap();
    assert_eq!(record.reads.len(), 2, "one fixed part is read in each state");
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
fn a_stop_parks_only_the_states_that_hold_tokens() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, directory) = started(ParkFactory::new("a", Arc::clone(&record)).empty());
    let (quantum, outcome) = Quantum::with_outcome(|engines| {
        engines[0]
            .pass(&[segment(State::Slot(1), &[5, 6])], &[])
            .expect("the quantum runs its pass");
    });
    process.submit(0, quantum).unwrap();
    assert_eq!(outcome.recv_timeout(Duration::from_secs(2)), Ok(QuantumOutcome::Ran));
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.states, 1, "slot 0 holds no token: {report:?}");
    assert_eq!(report.parked.entries, 1, "{report:?}");
    assert!(report.parked.failures.is_empty(), "{report:?}");
    let reads: Vec<State> = record.lock().unwrap().reads.iter().map(|read| read.1).collect();
    assert_eq!(reads, vec![State::Slot(1)], "the park reads only the slot with tokens");

    let mut seeded = two_slots();
    seeded
        .pass(&[segment(State::Slot(1), &[5, 6])], &[])
        .expect("the seeded pass runs");
    let snapshot = snapshot_from_engine(&mut seeded, State::Slot(1), &[(1, 0, 2), (2, 0, 2)]).unwrap();
    let id = snapshot.content_id().unwrap();
    drop(process);
    let store = Store::open(directory.path()).unwrap();
    assert_eq!(store.entries().len(), 1, "the store holds one entry");
    assert!(store.entries().contains_key(&id), "the entry is the state of slot 1");
}

#[test]
fn a_stop_with_only_empty_states_parks_nothing_and_is_clean() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", Arc::clone(&record)).empty());
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.states, 0, "{report:?}");
    assert_eq!(report.parked.entries, 0, "{report:?}");
    assert!(record.lock().unwrap().reads.is_empty(), "the park reads no empty slot");
    assert!(report.log_line().contains("parked nothing"), "{}", report.log_line());
}

#[test]
fn a_second_signal_stops_at_once() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) =
        started(ParkFactory::new("a", Arc::clone(&record)).slow(Duration::from_millis(100)));
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
    assert_eq!(
        report.exit_code(),
        ExitCode::Crash,
        "an interrupted park is a crash: {report:?}"
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "the second signal exits at once, but the stop waited {elapsed:?}"
    );
}

#[test]
fn a_stop_takes_no_more_admissions() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record));
    process.stop(Duration::from_secs(2), &SignalFlag::new());
    let error = process.admit(profile(vec![engine("a", "park", &[0])])).unwrap_err();
    assert_eq!(error.to_string(), "the process is stopping");
    assert!(matches!(error, AdmitError::Stopping));
}

#[test]
fn two_signals_before_the_stop_park_nothing() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) =
        started(ParkFactory::new("a", Arc::clone(&record)).slow(Duration::from_millis(100)));
    let flag = SignalFlag::new();
    flag.raise();
    flag.raise();
    let started = Instant::now();
    let report = process.stop(Duration::from_secs(10), &flag);
    assert!(report.interrupted);
    assert!(!report.complete);
    assert_eq!(report.exit_code(), ExitCode::Crash, "nothing was parked: {report:?}");
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
    let (mut process, _, _directory) = started(ParkFactory::new("a", Arc::clone(&record)));
    process.abort();
    let error = process.admit(profile(vec![engine("a", "park", &[0])])).unwrap_err();
    assert!(matches!(error, AdmitError::Stopping));
    // The drop joins every worker, so the record is complete here.
    drop(process);
    assert!(record.lock().unwrap().reads.is_empty(), "an abort parks nothing");
}

#[test]
fn a_stop_with_no_state_says_it_parked_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let profile = PlacementProfile {
        engines: Vec::new(),
        budgets: Vec::new(),
        store: None,
        cpu: None,
    };
    let mut process = Start::new(profile, Registry::new(), inventory(&[]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    let line = report.log_line();
    assert!(
        line.contains("parked nothing"),
        "the stop must not claim a park: {line}"
    );
}

#[test]
fn a_stop_that_no_group_answered_does_not_say_it_parked_nothing() {
    let report = StopReport {
        parked: ParkReport::default(),
        complete: false,
        interrupted: false,
    };
    let line = report.log_line();
    assert!(line.contains("did not finish"), "the park did not finish: {line}");
    assert!(!line.contains("parked nothing"), "{line}");
    assert_eq!(report.exit_code(), ExitCode::Crash);
}

#[test]
fn a_stop_that_wrote_a_part_and_did_not_finish_says_so() {
    let report = StopReport {
        parked: ParkReport {
            engines: 1,
            states: 1,
            parts: 3,
            bytes: 16,
            entries: 1,
            ..ParkReport::default()
        },
        complete: false,
        interrupted: false,
    };
    assert!(report.log_line().contains("did not finish"), "{}", report.log_line());
}

#[test]
fn a_stop_that_wrote_nothing_and_failed_says_the_failure() {
    let report = StopReport {
        parked: ParkReport {
            failures: vec!["engine `a` state 0 failed".to_owned()],
            ..ParkReport::default()
        },
        complete: true,
        interrupted: false,
    };
    let line = report.log_line();
    assert!(line.contains("failed 1 part"), "{line}");
}

#[test]
fn a_stop_that_writes_says_what_it_wrote() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record));
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    let line = report.log_line();
    assert_eq!(report.parked.entries, 2, "two slots give two writes");
    assert_eq!(report.parked.states, 2, "every state is parked");
    assert!(
        line.contains("wrote 2 states and 6 parts"),
        "the stop must say what it wrote: {line}"
    );
}

#[test]
fn a_store_refusal_is_a_crash_with_the_names() {
    let directory = tempfile::tempdir().unwrap();
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let mut registry = Registry::new();
    registry.register("park", Box::new(ParkFactory::new("a", record)));
    let profile = profile(vec![engine("a", "park", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .store(directory.path())
        .store_budget(StoreBudget {
            host: 0,
            nvme: 8,
            interval_copies: 0,
        })
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Crash);
    assert_eq!(report.parked.entries, 0);
    let text = report.parked.failures.join(" | ");
    assert!(text.contains("engine `a`"), "{text}");
    assert!(text.contains("state 0"), "{text}");
    assert!(text.contains("`state`"), "{text}");
    assert!(text.contains("budget cannot hold the entry"), "{text}");
}

#[test]
fn an_unmeasured_row_part_is_a_crash_with_the_names() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record).without_slots());
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Crash);
    assert_eq!(report.parked.failures.len(), 2, "{report:?}");
    let text = report.parked.failures.join(" | ");
    assert!(text.contains("engine `a`"), "{text}");
    assert!(text.contains("state 0"), "{text}");
    assert!(text.contains("part `cells`"), "{text}");
    assert!(text.contains("no slots capability"), "{text}");
}

#[test]
fn a_span_that_does_not_fit_its_slot_is_a_crash_with_the_names() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record).misfit());
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Crash, "{report:?}");
    assert_eq!(report.parked.entries, 0, "a misfit state is not stored: {report:?}");
    assert_eq!(
        report.parked.failures.len(),
        2,
        "one failure for each state: {report:?}"
    );
    let text = report.parked.failures.join(" | ");
    assert!(text.contains("engine `a`"), "{text}");
    assert!(text.contains("state 0"), "{text}");
    assert!(text.contains("part `cells`"), "{text}");
    assert!(text.contains("does not fit"), "{text}");
}

#[test]
fn a_park_failure_is_a_crash() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record).failing());
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.parked.failures.len(), 2, "{report:?}");
    assert!(report.parked.failures[0].contains("engine `a`"), "{report:?}");
    assert!(report.parked.failures[0].contains("state 0"), "{report:?}");
    assert!(report.parked.failures[0].contains("part `state`"), "{report:?}");
    assert_eq!(report.exit_code(), ExitCode::Crash);
}

#[test]
fn a_running_quantum_ends_before_the_park() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("park", Box::new(ParkFactory::new("a", Arc::clone(&record)).empty()));
    let mut process = Start::new(profile(vec![engine("a", "park", &[0])]), registry, inventory(&[0]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();

    let events = Arc::clone(&record);
    process
        .submit(
            0,
            Quantum::new(move |engines| {
                events.lock().unwrap().events.push("quantum".to_owned());
                thread::sleep(Duration::from_millis(60));
                engines[0]
                    .pass(&[segment(State::Slot(0), &[7, 8, 9])], &[])
                    .expect("the quantum runs its pass");
                events.lock().unwrap().events.push("quantum-end".to_owned());
            }),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !record.lock().unwrap().events.iter().any(|event| event == "quantum") {
        assert!(Instant::now() < deadline, "the quantum did not start");
        thread::sleep(Duration::from_millis(2));
    }
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert!(report.complete);

    let events = record.lock().unwrap().events.clone();
    let end = events
        .iter()
        .position(|event| event == "quantum-end")
        .expect("the quantum ends");
    let is_access = |event: &String| matches!(event.as_str(), "read" | "cells_read" | "cells_write");
    assert!(
        !events[..end].iter().any(is_access),
        "the park touches a part during the quantum: {events:?}"
    );
    assert!(
        events[end + 1..].iter().all(is_access),
        "the park reads only after the quantum: {events:?}"
    );
    assert!(
        events[end + 1..].iter().any(|event| event == "cells_read"),
        "the park reads the cells parts: {events:?}"
    );
    assert!(
        events[end + 1..].len() >= 2,
        "the park reads the fixed and the row parts of slot 0: {events:?}"
    );
    drop(record);

    let mut seeded = two_slots();
    seeded
        .pass(&[segment(State::Slot(0), &[7, 8, 9])], &[])
        .expect("the seeded pass runs");
    let snapshot = snapshot_from_engine(&mut seeded, State::Slot(0), &[(1, 0, 3), (2, 0, 3)]).unwrap();
    let id = snapshot.content_id().unwrap();
    drop(process);
    let store = Store::open(directory.path()).unwrap();
    assert!(store.entries().contains_key(&id), "the quantum's result is parked");
}

/// Start one `park` engine with a short heartbeat and the given watchdog timeout.
fn started_with_timeout(factory: ParkFactory, timeout: Duration) -> (nagaya::process::Process, tempfile::TempDir) {
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("park", Box::new(factory));
    let mut process = Start::new(profile(vec![engine("a", "park", &[0])]), registry, inventory(&[0]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(timeout)
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    (process, directory)
}

/// Submit a quantum that sleeps, and wait until it runs.
fn submit_sleeper(process: &nagaya::process::Process, sleep: Duration, ran: &Arc<Mutex<Vec<String>>>) {
    let log = Arc::clone(ran);
    process
        .submit(
            0,
            Quantum::new(move |_| {
                log.lock().unwrap().push("long".to_owned());
                thread::sleep(sleep);
            }),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !ran.lock().unwrap().iter().any(|entry| entry == "long") {
        assert!(Instant::now() < deadline, "the long quantum did not start");
        thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_stop_during_a_long_quantum_still_parks_every_state() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _directory) = started_with_timeout(ParkFactory::new("a", record), Duration::from_secs(2));
    let ran = Arc::new(Mutex::new(Vec::new()));
    submit_sleeper(&process, Duration::from_millis(400), &ran);
    // The park budget is shorter than the quantum. The budget starts when the drain ends (spec 10.3).
    let report = process.stop(Duration::from_millis(150), &SignalFlag::new());
    assert!(report.complete, "the drain does not use the park budget: {report:?}");
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.states, 2, "every state is parked: {report:?}");
    assert!(report.parked.failures.is_empty(), "{report:?}");
}

#[test]
fn a_stop_cancels_the_queued_quanta_and_runs_none_of_them() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _directory) = started_with_timeout(ParkFactory::new("a", record), Duration::from_secs(2));
    let ran = Arc::new(Mutex::new(Vec::new()));
    submit_sleeper(&process, Duration::from_millis(200), &ran);
    let mut outcomes = Vec::new();
    for name in ["second", "third"] {
        let log = Arc::clone(&ran);
        let (quantum, outcome) = Quantum::with_outcome(move |_| log.lock().unwrap().push(name.to_owned()));
        process.submit(0, quantum).unwrap();
        outcomes.push(outcome);
    }
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(
        report.parked.cancelled, 2,
        "the stop counts the cancelled quanta: {report:?}"
    );
    assert_eq!(report.parked.states, 2, "{report:?}");
    assert_eq!(*ran.lock().unwrap(), vec!["long".to_owned()], "no queued quantum runs");
    for outcome in outcomes {
        assert_eq!(outcome.try_recv(), Ok(QuantumOutcome::Cancelled));
    }
    assert!(report.log_line().contains("cancelled 2"), "{}", report.log_line());
}

#[test]
fn a_quantum_that_ran_reports_its_outcome() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _directory) = started_with_timeout(ParkFactory::new("a", record), Duration::from_secs(2));
    let (quantum, outcome) = Quantum::with_outcome(|_| {});
    process.submit(0, quantum).unwrap();
    assert_eq!(outcome.recv_timeout(Duration::from_secs(2)), Ok(QuantumOutcome::Ran));
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.parked.cancelled, 0);
}

#[test]
fn a_drain_that_never_ends_is_a_crash() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    // The quantum outlasts the watchdog timeout, so the watchdog reports a stall (spec 7.4).
    let (mut process, _directory) = started_with_timeout(ParkFactory::new("a", record), Duration::from_millis(150));
    let ran = Arc::new(Mutex::new(Vec::new()));
    submit_sleeper(&process, Duration::from_millis(900), &ran);
    let started = Instant::now();
    let report = process.stop(Duration::from_secs(10), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Crash, "{report:?}");
    assert!(!report.complete, "{report:?}");
    assert!(
        started.elapsed() < Duration::from_millis(700),
        "the stall ends the stop; it took {:?}",
        started.elapsed()
    );
}

#[test]
fn a_park_that_outlasts_its_budget_after_the_drain_is_a_crash() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _directory) = started_with_timeout(
        ParkFactory::new("a", record).slow(Duration::from_millis(300)),
        Duration::from_secs(5),
    );
    let started = Instant::now();
    let report = process.stop(Duration::from_millis(100), &SignalFlag::new());
    assert!(!report.complete, "{report:?}");
    assert_eq!(report.exit_code(), ExitCode::Crash);
    assert!(
        started.elapsed() < Duration::from_millis(280),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn a_park_that_finished_while_an_earlier_group_used_the_budget_is_not_dropped() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    // The first group parks slowly and uses the budget. The second group parks at once and has its reply waiting.
    registry.register(
        "slow",
        Box::new(ParkFactory::new("a", Arc::clone(&record)).slow(Duration::from_millis(300))),
    );
    let mut fast = ParkFactory::new("b", Arc::clone(&record));
    fast.limits = limits_on(&[1]);
    registry.register("fast", Box::new(fast));
    let mut profile = profile(vec![engine("a", "slow", &[0]), engine("b", "fast", &[1])]);
    profile.budgets.push(BudgetSpec {
        tiers: vec![TierSpec::from(Tier::Device(1))],
        bytes: 4096,
    });
    let mut process = Start::new(profile, registry, inventory(&[0, 1]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(2)).unwrap();
    let names: Vec<String> = opened.iter().map(|engine| engine.name.clone()).collect();
    let report = process.stop(Duration::from_millis(100), &SignalFlag::new());
    assert!(!report.complete, "the slow park outlasts its budget: {report:?}");
    assert_eq!(report.exit_code(), ExitCode::Crash);
    let (parked, unparked) = report.split_parked(&names);
    assert_eq!(parked, ["b"], "the second group parked in time: {report:?}");
    assert_eq!(unparked, ["a"], "{report:?}");
}

#[test]
fn a_job_submitted_after_the_stop_is_refused() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record));
    let flag = SignalFlag::new();
    flag.raise();
    process.stop(Duration::from_secs(2), &flag);
    assert_eq!(process.submit(0, Quantum::new(|_| {})), Err(SubmitError::Stopping));
}

#[test]
fn a_quantum_for_an_unknown_group_is_refused() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record));
    assert_eq!(process.submit(3, Quantum::new(|_| {})), Err(SubmitError::UnknownGroup));
    process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_profile_without_a_store_budget_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let mut profile = profile(vec![engine("fake-one", "fake", &[0])]);
    profile.store = None;
    let error = Start::new(profile, Registry::with_fake(), inventory(&[0]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap_err();
    assert!(matches!(error, StartError::NoStoreBudget), "{error}");
    assert!(error.to_string().contains("store.nvme"), "{error}");
}

/// Start two engines on one group with a notifier: `good` parks, `bad` fails to read.
fn started_pair(directory: &std::path::Path) -> (lifecycle::Ready, UnixDatagram, Notifier, tempfile::TempDir) {
    let socket = UnixDatagram::bind(directory.join("notify")).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    let notifier = Notifier::at(&directory.join("notify")).unwrap();
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let mut registry = Registry::new();
    registry.register("good", Box::new(ParkFactory::new("good", Arc::clone(&record))));
    registry.register("bad", Box::new(ParkFactory::new("bad", record).failing()));
    let store = tempfile::tempdir().unwrap();
    let ready = lifecycle::start(
        profile(vec![engine("good", "good", &[0]), engine("bad", "bad", &[0])]),
        registry,
        inventory(&[0]),
        store.path(),
        Some(&notifier),
        Duration::from_secs(2),
    )
    .unwrap();
    (ready, socket, notifier, store)
}

fn receive(socket: &UnixDatagram) -> String {
    let mut buffer = [0u8; 512];
    let count = socket.recv(&mut buffer).unwrap();
    std::str::from_utf8(&buffer[..count]).unwrap().to_owned()
}

#[test]
fn a_failed_park_reports_fatal_for_the_engine_that_did_not_park() {
    let directory = tempfile::tempdir().unwrap();
    let (mut ready, socket, notifier, _store) = started_pair(directory.path());
    assert_eq!(receive(&socket), "STATUS=engine good loading; engine bad loading");
    assert_eq!(receive(&socket), "READY=1\nSTATUS=engine bad ready; engine good ready");
    let report = lifecycle::stop(&mut ready, Some(&notifier), Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Crash, "{report:?}");
    assert_eq!(receive(&socket), "STATUS=engine good draining; engine bad draining");
    assert_eq!(receive(&socket), "STATUS=engine good unloaded");
    assert_eq!(receive(&socket), "STATUS=engine bad fatal");
}

#[test]
fn an_interrupted_stop_reports_fatal_for_every_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (mut ready, socket, notifier, _store) = started_pair(directory.path());
    receive(&socket);
    receive(&socket);
    let flag = SignalFlag::new();
    flag.raise();
    flag.raise();
    let report = lifecycle::stop(&mut ready, Some(&notifier), Duration::from_secs(2), &flag);
    assert!(report.interrupted);
    assert_eq!(receive(&socket), "STATUS=engine good draining; engine bad draining");
    assert_eq!(receive(&socket), "STATUS=engine good fatal; engine bad fatal");
}

// The stop parks the pinned entries of the host tier, so a pin stays through a restart (spec 6.11, Decided
// 2026-10-23).

fn wall_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Put one entry in the host tier of the process's store, and pin it for an hour. Give its content id.
fn pin_a_host_entry(process: &nagaya::process::Process) -> nagaya_store::ContentId {
    let snapshot = Snapshot {
        tensors: vec![Tensor {
            name: "state".to_owned(),
            kind: nagaya_abi::PartKind::Fixed,
            dtype: nagaya_abi::Dtype::U8,
            shape: vec![16],
            data: vec![7; 16],
            coverage: None,
            block_format: None,
        }],
        parent: None,
    };
    let recipe: RecipeId = recipe_root(&[1u8; 32], &[], &[], &[]);
    let now = wall_ms();
    let ports = process.ports();
    let mut store = ports.store().unwrap().lock().unwrap();
    // The test profile assigns all of its host bytes to the engine, so the store gets a host budget of its own.
    store.set_budget(StoreTier::Host, 1 << 20);
    let id = store
        .put(Put {
            snapshot: &snapshot,
            recipe,
            previous_recipe: None,
            chains_from: None,
            engine: EngineFacts::of("a", &limits_on(&[0])),
            identity: [1u8; 32],
            owner: 9,
            tier: StoreTier::Host,
            reason: Reason::Anchor,
            sealed_recipe: None,
            provenance: None,
            now_ms: now,
        })
        .unwrap();
    store.pin(id, 9, now + 3_600_000, now).unwrap();
    id
}

#[test]
fn a_stop_parks_a_pinned_host_entry_and_its_pin_stays() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, directory) = started(ParkFactory::new("a", record).empty());
    let id = pin_a_host_entry(&process);
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.pinned, 1, "{report:?}");
    let line = report.log_line();
    assert!(line.contains("parked 1 pinned entries"), "{line}");
    assert!(!line.contains("parked nothing"), "{line}");
    drop(process);

    let store = Store::open(directory.path()).unwrap();
    assert_eq!(store.entries()[&id].tier, StoreTier::Nvme, "6.11: the entry is on NVMe");
    assert_eq!(store.pins()[&id].len(), 1, "6.10: the pin stays through a restart");
}

#[test]
fn a_stop_whose_park_of_a_pin_fails_is_a_crash() {
    let record = Arc::new(Mutex::new(ParkRecord::default()));
    let (mut process, _, _directory) = started(ParkFactory::new("a", record).empty());
    let id = pin_a_host_entry(&process);
    process
        .ports()
        .store()
        .unwrap()
        .lock()
        .unwrap()
        .set_budget(StoreTier::Nvme, 1);
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Crash, "{report:?}");
    assert_eq!(report.parked.failures.len(), 1, "{report:?}");
    assert!(report.parked.failures[0].contains(&id.hex()), "{report:?}");
}
