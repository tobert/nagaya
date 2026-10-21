use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Event, OwnedLedger, Start, WaitError};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Coverage, DeviceId, Engine, Identity, Limits, Parts, PassError, PassOut, Seal, Segment, Slots, Tap, Tier,
};

type Record = Arc<Mutex<Vec<(String, ThreadId)>>>;

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
                tiers: vec![TierSpec::from(Tier::Device(1))],
                bytes,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes,
            },
        ],
        store: None,
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
            })
            .collect(),
        host_bytes: 1024,
    }
}

struct RecordingFactory {
    name: String,
    planned: Limits,
    opened: Limits,
    record: Record,
}

impl RecordingFactory {
    fn new(name: &str, devices: &[DeviceId], record: Record) -> Self {
        let limits = limits_on(devices);
        Self {
            name: name.to_owned(),
            planned: limits.clone(),
            opened: limits,
            record,
        }
    }

    fn opens_with(name: &str, devices: &[DeviceId], opened: Limits, record: Record) -> Self {
        Self {
            name: name.to_owned(),
            planned: limits_on(devices),
            opened,
            record,
        }
    }
}

impl Factory for RecordingFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.planned.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(RecordingEngine {
            inner: FakeAlways::new(),
            name: self.name.clone(),
            limits: self.opened.clone(),
            record: Arc::clone(&self.record),
        }))
    }
}

struct RecordingEngine {
    inner: FakeAlways,
    name: String,
    limits: Limits,
    record: Record,
}

impl Engine for RecordingEngine {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.record
            .lock()
            .unwrap()
            .push((self.name.clone(), thread::current().id()));
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
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

fn thread_of(record: &Record, name: &str) -> ThreadId {
    record
        .lock()
        .unwrap()
        .iter()
        .find(|(candidate, _)| candidate == name)
        .unwrap_or_else(|| panic!("engine {name} did not open"))
        .1
}

#[test]
fn each_engine_opens_on_its_group_thread() {
    let record: Record = Arc::new(Mutex::new(Vec::new()));
    let mut registry = Registry::new();
    registry.register("a", Box::new(RecordingFactory::new("a", &[0], Arc::clone(&record))));
    registry.register("b", Box::new(RecordingFactory::new("b", &[1], Arc::clone(&record))));
    let profile = profile(vec![engine("a", "a", &[0]), engine("b", "b", &[1])]);
    let mut process = Start::new(profile, registry, inventory(&[0, 1]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(2)).unwrap();
    assert_eq!(opened.len(), 2);
    let main = thread::current().id();
    assert_ne!(thread_of(&record, "a"), main);
    assert_ne!(thread_of(&record, "b"), main);
    assert_ne!(thread_of(&record, "a"), thread_of(&record, "b"));
    let a = opened.iter().find(|engine| engine.name == "a").unwrap();
    let b = opened.iter().find(|engine| engine.name == "b").unwrap();
    assert_eq!(thread_of(&record, "a"), a.thread);
    assert_eq!(thread_of(&record, "b"), b.thread);
}

#[test]
fn two_engines_on_one_device_share_a_thread() {
    let record: Record = Arc::new(Mutex::new(Vec::new()));
    let mut registry = Registry::new();
    registry.register("a", Box::new(RecordingFactory::new("a", &[0], Arc::clone(&record))));
    registry.register("b", Box::new(RecordingFactory::new("b", &[0], Arc::clone(&record))));
    let profile = profile(vec![engine("a", "a", &[0]), engine("b", "b", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(2)).unwrap();
    assert_eq!(opened.len(), 2);
    assert_eq!(thread_of(&record, "a"), thread_of(&record, "b"));
    assert_eq!(thread_of(&record, "a"), opened[0].thread);
}
#[test]
fn an_engine_whose_limits_change_after_open_is_refused() {
    let record: Record = Arc::new(Mutex::new(Vec::new()));
    let mut opened = limits_on(&[0]);
    opened.max_quantum_ms += 1;
    let mut registry = Registry::new();
    registry.register(
        "a",
        Box::new(RecordingFactory::opens_with("a", &[0], opened, Arc::clone(&record))),
    );
    let profile = profile(vec![engine("a", "a", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    match process.wait_opens(Duration::from_secs(2)).unwrap_err() {
        WaitError::Event(Event::OpenFailed {
            engine, message, kind, ..
        }) => {
            assert_eq!(engine, "a");
            assert!(message.contains("limits"), "{message}");
            assert_eq!(kind, OpenErrorKind::Other);
        }
        other => panic!("expected an open failure, got {other:?}"),
    }
}

#[test]
fn an_engine_whose_part_size_changes_after_open_refuses_the_start() {
    let record: Record = Arc::new(Mutex::new(Vec::new()));
    let mut opened = limits_on(&[0]);
    assert!(!opened.parts.is_empty(), "the fixture needs a part");
    opened.parts[0].shape[0] += 1;
    let mut registry = Registry::new();
    registry.register(
        "a",
        Box::new(RecordingFactory::opens_with("a", &[0], opened, Arc::clone(&record))),
    );
    let profile = profile(vec![engine("a", "a", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    match process.wait_opens(Duration::from_secs(2)).unwrap_err() {
        WaitError::Event(event @ Event::OpenFailed { .. }) => {
            let Event::OpenFailed {
                engine, message, kind, ..
            } = &event
            else {
                unreachable!()
            };
            assert_eq!(engine, "a");
            assert!(message.contains("part"), "{message}");
            assert_eq!(*kind, OpenErrorKind::PartSize);
            assert_eq!(event.exit_code(), Some(nagaya::exit::ExitCode::RefusedStart));
        }
        other => panic!("expected an open failure, got {other:?}"),
    }
}

struct FailingOpenFactory {
    limits: Limits,
}

impl Factory for FailingOpenFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Err(OpenError {
            kind: OpenErrorKind::Other,
            message: "the weights are absent".to_owned(),
        })
    }
}

#[test]
fn an_open_error_reaches_the_caller() {
    let mut registry = Registry::new();
    registry.register(
        "broken",
        Box::new(FailingOpenFactory {
            limits: limits_on(&[0]),
        }),
    );
    let profile = profile(vec![engine("broken-one", "broken", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    match process.wait_opens(Duration::from_secs(2)).unwrap_err() {
        WaitError::Event(Event::OpenFailed { engine, message, .. }) => {
            assert_eq!(engine, "broken-one");
            assert!(message.contains("the weights are absent"), "{message}");
        }
        other => panic!("expected an open failure, got {other:?}"),
    }
}

#[test]
fn only_the_ledger_thread_writes_the_ledger() {
    let ledger = OwnedLedger::new();
    let profile = profile(vec![engine("fake-one", "fake", &[0])]);
    let registry = Registry::with_fake();
    let inventory = inventory(&[0]);
    let handle = thread::spawn(move || {
        let mut ledger = ledger;
        ledger.admit(&profile, &registry, &inventory)
    });
    let error = handle.join().unwrap_err();
    let message = match error.downcast_ref::<String>() {
        Some(message) => message.clone(),
        None => panic!("the write did not panic with a message"),
    };
    assert!(message.contains("the ledger is written off its thread"), "{message}");
}

struct Release(Arc<AtomicBool>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A group stalls in a pass, not in an open: a group beats while its engine opens (spec 7.4).
#[test]
fn a_stalled_group_reaches_the_caller() {
    let release = Arc::new(AtomicBool::new(false));
    let record: Record = Arc::new(Mutex::new(Vec::new()));
    let mut registry = Registry::new();
    registry.register("a", Box::new(RecordingFactory::new("a", &[0], record)));
    let profile = profile(vec![engine("stall-one", "a", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_millis(100))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    // The guard drops before the process, so a failed assert cannot block the join.
    let guard = Release(Arc::clone(&release));
    let hold = Arc::clone(&release);
    let quantum = nagaya::process::Quantum::new(move |_| {
        while !hold.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(2));
        }
    });
    process.submit(0, quantum).unwrap();
    let event = process
        .recv(Duration::from_secs(2))
        .expect("the watchdog reports the stall");
    assert_eq!(
        event,
        Event::Stalled {
            group: 0,
            devices: vec![0],
        }
    );
    assert_eq!(event.exit_code(), Some(nagaya::exit::ExitCode::Crash));
    drop(guard);
}

fn started_one_group(timeout: Duration) -> nagaya::process::Process {
    let mut registry = Registry::new();
    registry.register(
        "a",
        Box::new(RecordingFactory::new("a", &[0], Arc::new(Mutex::new(Vec::new())))),
    );
    let profile = profile(vec![engine("a", "a", &[0])]);
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(timeout)
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    process
}

#[test]
fn a_busy_group_with_short_quanta_does_not_stall() {
    let process = started_one_group(Duration::from_millis(100));
    // The queue never empties for 400 ms, so the idle beat never runs. Each quantum is shorter than the timeout.
    for _ in 0..20 {
        let quantum = nagaya::process::Quantum::new(|_| thread::sleep(Duration::from_millis(20)));
        process.submit(0, quantum).unwrap();
    }
    assert_eq!(
        process.recv(Duration::from_millis(450)),
        None,
        "a busy group is not stalled"
    );
}

#[test]
fn a_quantum_that_blocks_past_the_timeout_stalls_the_group() {
    let process = started_one_group(Duration::from_millis(100));
    let quantum = nagaya::process::Quantum::new(|_| thread::sleep(Duration::from_millis(600)));
    process.submit(0, quantum).unwrap();
    let event = process
        .recv(Duration::from_secs(2))
        .expect("the watchdog reports the stall");
    assert_eq!(
        event,
        Event::Stalled {
            group: 0,
            devices: vec![0]
        }
    );
}

/// An engine whose open takes `open_time`, as mk's open does (spec 7.4).
struct SlowOpenFactory {
    limits: Limits,
    open_time: Duration,
}

impl Factory for SlowOpenFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        thread::sleep(self.open_time);
        Ok(Box::new(FakeAlways::new()))
    }
}

fn started_slow_open(open_time: Duration, timeout: Duration) -> nagaya::process::Process {
    let mut registry = Registry::new();
    registry.register(
        "slow",
        Box::new(SlowOpenFactory {
            limits: limits_on(&[0]),
            open_time,
        }),
    );
    let profile = profile(vec![engine("slow-one", "slow", &[0])]);
    Start::new(profile, registry, inventory(&[0]))
        .heartbeat(Duration::from_millis(10))
        .timeout(timeout)
        .start()
        .unwrap()
}

#[test]
fn an_open_that_outlasts_the_watchdog_timeout_is_not_a_stall() {
    // The open takes 600 ms. The watchdog timeout is 100 ms. The group beats while it opens (spec 7.4).
    let mut process = started_slow_open(Duration::from_millis(600), Duration::from_millis(100));
    let opened = process
        .wait_opens(Duration::from_secs(5))
        .expect("a slow open is not a stall");
    assert_eq!(opened.len(), 1);
}

#[test]
fn a_group_keeps_its_beat_after_a_slow_open() {
    let mut process = started_slow_open(Duration::from_millis(300), Duration::from_millis(100));
    process.wait_opens(Duration::from_secs(5)).unwrap();
    // An idle group beats on its interval, so the long open leaves no stall behind.
    assert_eq!(process.recv(Duration::from_millis(600)), None);
}

#[test]
fn an_open_that_never_ends_reaches_the_wait_timeout() {
    // The beat during an open hides a hung open from the watchdog. The wait for the opens is the backstop.
    let mut process = started_slow_open(Duration::from_secs(3), Duration::from_millis(100));
    match process.wait_opens(Duration::from_millis(400)).unwrap_err() {
        WaitError::Timeout { waiting } => assert_eq!(waiting, vec!["slow-one".to_owned()]),
        other => panic!("expected a timeout, got {other:?}"),
    }
}

/// A factory that keeps the devices that `plan` and `open` get.
struct DevicesFactory {
    seen: Arc<Mutex<Vec<(&'static str, Vec<GroupDevice>)>>>,
}

impl Factory for DevicesFactory {
    fn plan(&self, _config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        self.seen.lock().unwrap().push(("plan", devices.to_vec()));
        Ok(limits_on(&[1, 0]))
    }

    fn open(&self, _config: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        self.seen.lock().unwrap().push(("open", devices.to_vec()));
        Ok(Box::new(RecordingEngine {
            inner: FakeAlways::new(),
            name: "a".to_owned(),
            limits: limits_on(&[1, 0]),
            record: Arc::new(Mutex::new(Vec::new())),
        }))
    }
}

#[test]
fn plan_and_open_get_the_machine_devices_of_the_engine_in_profile_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut registry = Registry::new();
    registry.register(
        "pair",
        Box::new(DevicesFactory {
            seen: Arc::clone(&seen),
        }),
    );
    let mut inventory = inventory(&[0, 1]);
    inventory.devices[0].pci_address = "0000:03:00.0".to_owned();
    inventory.devices[1].pci_address = "0000:08:00.0".to_owned();
    let profile = profile(vec![engine("a", "pair", &[1, 0])]);
    let mut process = Start::new(profile, registry, inventory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    let expected = vec![
        GroupDevice {
            id: 1,
            name: "device-1".to_owned(),
            pci_address: "0000:08:00.0".to_owned(),
            memory_bytes: 1024,
        },
        GroupDevice {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:03:00.0".to_owned(),
            memory_bytes: 1024,
        },
    ];
    assert_eq!(
        *seen.lock().unwrap(),
        vec![("plan", expected.clone()), ("open", expected)]
    );
}
