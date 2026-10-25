//! The swap in place on a group thread (spec 3.3a): a job for an engine that is not loaded starts the swap. The
//! group unloads the engines that must give their memory, closes them, and opens the engine of the job.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::{Event, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Kind, Lifetime, Limits, MandatoryTerm, OutputsWanted, Parts, PassError,
    PassOut, Seal, Segment, SegmentInputs, Slots, State, Tap, Tier,
};
use nagaya_sched::Class;

type Log = Arc<Mutex<Vec<String>>>;

/// A fake engine that writes its open and its close to a log, and declares `close`.
struct Counted {
    inner: FakeAlways,
    name: String,
    limits: Limits,
    log: Log,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.log.lock().unwrap().push(format!("close:{}", self.name));
    }
}

impl Engine for Counted {
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
        self.inner.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.inner.seal()
    }
}

fn counted_limits(config: &toml::Table) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes: config.get("bytes").and_then(toml::Value::as_integer).unwrap_or(600) as u64,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    limits
}

struct CountedFactory(Log);

impl Factory for CountedFactory {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(counted_limits(config))
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let name = config["name"].as_str().unwrap().to_owned();
        self.0.lock().unwrap().push(format!("open:{name}"));
        Ok(Box::new(Counted {
            inner: FakeAlways::new(),
            name,
            limits: counted_limits(config),
            log: Arc::clone(&self.0),
        }))
    }
}

fn engine(name: &str, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("name".to_owned(), toml::Value::String(name.to_owned()));
    config.insert("bytes".to_owned(), toml::Value::Integer(600));
    EnginePlacement {
        name: name.to_owned(),
        kind: "counted".to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
    }
}

/// A profile of a resident engine `chat` and an on-demand engine `voice` on one device. With a device budget of
/// 1000 bytes they do not fit together; with 2000 they do.
fn profile(device_bytes: u64) -> PlacementProfile {
    PlacementProfile {
        engines: vec![
            engine("chat", Residency::Resident),
            engine("voice", Residency::OnDemand),
        ],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: device_bytes,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 4096,
            },
        ],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn start(directory: &std::path::Path, device_bytes: u64) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("counted", Box::new(CountedFactory(Arc::clone(&log))));
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 4096,
            unified: false,
            ignored: false,
        }],
        host_bytes: 4096,
    };
    let mut process = Start::new(profile(device_bytes), registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    let names: Vec<&str> = opened.iter().map(|engine| engine.name.as_str()).collect();
    assert_eq!(names, ["chat"], "only the resident engine opens at start");
    (process, log)
}

/// A job that writes `run:<engine>` to the log. It reads the identity of its engine, which panics on an engine that
/// is not loaded. It can wait for a gate before it ends.
struct Probe {
    engine: usize,
    name: &'static str,
    log: Log,
    gate: Option<Receiver<()>>,
    done: Option<Sender<()>>,
}

impl JobBody for Probe {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        assert_eq!(engines.len(), 2, "the group holds both engines of the profile");
        let _ = engines[self.engine].identity();
        self.log.lock().unwrap().push(format!("run:{}", self.name));
        if let Some(gate) = self.gate.take() {
            let _ = gate.recv_timeout(Duration::from_secs(10));
        }
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
        Ran::Done
    }
}

const CHAT: usize = 0;
const VOICE: usize = 1;

fn probe(engine: usize, log: &Log) -> (JobSpec, Receiver<()>) {
    let (done, finished) = mpsc::channel();
    let name = if engine == CHAT { "chat" } else { "voice" };
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine,
        body: Box::new(Probe {
            engine,
            name,
            log: Arc::clone(log),
            gate: None,
            done: Some(done),
        }),
    };
    (spec, finished)
}

fn log_of(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// The next `Swapped` event of the process, skipping none.
fn swapped(process: &Process) -> (Vec<String>, Vec<String>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match process.recv(Duration::from_millis(50)) {
            Some(Event::Swapped { unloaded, loaded, .. }) => {
                return (unloaded, loaded.into_iter().map(|engine| engine.name).collect());
            }
            Some(other) => panic!("expected a swap, got {other:?}"),
            None => {}
        }
    }
    panic!("no swap came in 5 s");
}

#[test]
fn a_job_for_an_engine_on_demand_loads_it_in_place_of_the_resident_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "run:voice"]);
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
}

#[test]
fn a_job_for_the_resident_engine_swaps_it_back() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    let (job, finished) = probe(CHAT, &log);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).expect("the chat job ran");
    assert_eq!(
        log_of(&log),
        [
            "open:chat",
            "close:chat",
            "open:voice",
            "run:voice",
            "close:voice",
            "open:chat",
            "run:chat"
        ]
    );
}

#[test]
fn an_engine_that_fits_beside_the_loaded_engines_loads_with_no_unload() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 2000);
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    let (job, finished) = probe(CHAT, &log);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).expect("the chat job ran");
    assert_eq!(log_of(&log), ["open:chat", "open:voice", "run:voice", "run:chat"]);
    assert_eq!(swapped(&process), (Vec::<String>::new(), vec!["voice".to_owned()]));
}

#[test]
fn the_swap_waits_for_the_running_job_of_the_engine_that_unloads() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let (gate_tx, gate_rx) = mpsc::channel();
    let (done, chat_finished) = mpsc::channel();
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine: CHAT,
                body: Box::new(Probe {
                    engine: CHAT,
                    name: "chat",
                    log: Arc::clone(&log),
                    gate: Some(gate_rx),
                    done: Some(done),
                }),
            },
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(&log).contains(&"run:chat".to_owned()) {
        assert!(
            Instant::now() < deadline,
            "the chat job did not start: {:?}",
            log_of(&log)
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let (job, voice_finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(log_of(&log), ["open:chat", "run:chat"], "the chat job holds its engine");
    gate_tx.send(()).unwrap();
    chat_finished.recv_timeout(Duration::from_secs(5)).unwrap();
    voice_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(
        log_of(&log),
        ["open:chat", "run:chat", "close:chat", "open:voice", "run:voice"]
    );
}

#[test]
fn a_stop_cancels_a_job_that_waits_for_a_swap() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start(directory.path(), 1000);
    let (gate_tx, gate_rx) = mpsc::channel();
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine: CHAT,
                body: Box::new(Probe {
                    engine: CHAT,
                    name: "chat",
                    log: Arc::clone(&log),
                    gate: Some(gate_rx),
                    done: None,
                }),
            },
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(&log).contains(&"run:chat".to_owned()) {
        assert!(Instant::now() < deadline, "the chat job did not start");
        std::thread::sleep(Duration::from_millis(5));
    }
    let (job, voice_finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let stopper = std::thread::spawn(move || {
        let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
        (process, report)
    });
    std::thread::sleep(Duration::from_millis(50));
    gate_tx.send(()).unwrap();
    let (_process, _report) = stopper.join().unwrap();
    assert!(
        voice_finished.recv_timeout(Duration::from_secs(1)).is_err(),
        "the waiting job was cancelled, so it sent nothing"
    );
    assert!(!log_of(&log).contains(&"open:voice".to_owned()), "{:?}", log_of(&log));
    assert!(!log_of(&log).contains(&"run:voice".to_owned()), "{:?}", log_of(&log));
}

/// A job that feeds four tokens into slot 0 of `chat`, so that the slot holds a state.
struct Fill(Sender<()>);

impl JobBody for Fill {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let segment = Segment {
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(vec![1, 2, 3, 4]),
            positions: (0..4).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted { logit_rows: vec![3] },
        };
        engines[CHAT].pass(&[segment], &[]).expect("a pass of the fake");
        let _ = self.0.send(());
        Ran::Done
    }
}

#[test]
fn the_unload_parks_each_state_that_holds_tokens() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let (done, filled) = mpsc::channel();
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Prefill,
                engine: CHAT,
                body: Box::new(Fill(done)),
            },
        )
        .unwrap();
    filled.recv_timeout(Duration::from_secs(5)).expect("the fill ran");
    let store = process.ports().store().cloned().expect("the process has a store");
    assert!(
        store.lock().unwrap().entries().is_empty(),
        "nothing is parked before the swap"
    );
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(
        store.lock().unwrap().entries().len(),
        1,
        "the unload of `chat` parked the state of slot 0"
    );
}

#[test]
fn the_ledger_charges_the_engine_that_loaded() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let device = |process: &Process| {
        process
            .ports()
            .unassigned()
            .into_iter()
            .find(|(tiers, _)| tiers == &vec![Tier::Device(0)])
            .unwrap()
            .1
    };
    assert_eq!(device(&process), 400, "`chat` holds 600 of 1000 bytes");
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(
        device(&process),
        400,
        "`voice` holds the 600 bytes that `chat` gave back"
    );
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 2000);
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(device(&process), 800, "both engines hold 600 bytes each of 2000");
}

#[test]
fn the_swap_waits_for_the_queued_jobs_of_the_engine_that_unloads() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let (gate_tx, gate_rx) = mpsc::channel();
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine: CHAT,
                body: Box::new(Probe {
                    engine: CHAT,
                    name: "chat",
                    log: Arc::clone(&log),
                    gate: Some(gate_rx),
                    done: None,
                }),
            },
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(&log).contains(&"run:chat".to_owned()) {
        assert!(Instant::now() < deadline, "the chat job did not start");
        std::thread::sleep(Duration::from_millis(5));
    }
    // A second chat job queues behind the first, and then the voice job comes.
    let (second, second_finished) = probe(CHAT, &log);
    process.submit_job(0, second).unwrap();
    let (voice, voice_finished) = probe(VOICE, &log);
    process.submit_job(0, voice).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    gate_tx.send(()).unwrap();
    second_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the queued chat job ran");
    voice_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(
        log_of(&log),
        [
            "open:chat",
            "run:chat",
            "run:chat",
            "close:chat",
            "open:voice",
            "run:voice"
        ]
    );
}

#[test]
fn a_job_for_the_engine_that_unloads_waits_for_its_next_load() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let (gate_tx, gate_rx) = mpsc::channel();
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine: CHAT,
                body: Box::new(Probe {
                    engine: CHAT,
                    name: "chat",
                    log: Arc::clone(&log),
                    gate: Some(gate_rx),
                    done: None,
                }),
            },
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(&log).contains(&"run:chat".to_owned()) {
        assert!(Instant::now() < deadline, "the chat job did not start");
        std::thread::sleep(Duration::from_millis(5));
    }
    let (voice, voice_finished) = probe(VOICE, &log);
    process.submit_job(0, voice).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let (late, late_finished) = probe(CHAT, &log);
    process.submit_job(0, late).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    gate_tx.send(()).unwrap();
    voice_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    late_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the late chat job ran");
    assert_eq!(
        log_of(&log),
        [
            "open:chat",
            "run:chat",
            "close:chat",
            "open:voice",
            "run:voice",
            "close:voice",
            "open:chat",
            "run:chat"
        ]
    );
}

#[test]
fn an_idle_engine_on_demand_unloads_and_the_resident_engine_loads_again() {
    let directory = tempfile::tempdir().unwrap();
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("counted", Box::new(CountedFactory(Arc::clone(&log))));
    let mut profile = profile(1000);
    profile.engines[VOICE].idle_unload_s = Some(1);
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 4096,
            unified: false,
            ignored: false,
        }],
        host_bytes: 4096,
    };
    let mut process = Start::new(profile, registry, inventory)
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    let (job, finished) = probe(VOICE, &log);
    let submitted = Instant::now();
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
    // No job comes. After its idle time, `voice` unloads and `chat` loads again.
    assert_eq!(swapped(&process), (vec!["voice".to_owned()], vec!["chat".to_owned()]));
    let after = submitted.elapsed();
    assert!(after >= Duration::from_secs(1), "the swap back came after {after:?}");
    assert_eq!(
        log_of(&log),
        [
            "open:chat",
            "close:chat",
            "open:voice",
            "run:voice",
            "close:voice",
            "open:chat"
        ]
    );
}

/// Start with `voice` unloading after `idle_s` seconds with no job.
fn start_idle(directory: &std::path::Path, idle_s: u64) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("counted", Box::new(CountedFactory(Arc::clone(&log))));
    let mut profile = profile(1000);
    profile.engines[VOICE].idle_unload_s = Some(idle_s);
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 4096,
            unified: false,
            ignored: false,
        }],
        host_bytes: 4096,
    };
    let mut process = Start::new(profile, registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, log)
}

#[test]
fn an_idle_resident_engine_stays_loaded() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_idle(directory.path(), 1);
    std::thread::sleep(Duration::from_millis(1500));
    assert!(process.recv(Duration::from_millis(10)).is_none(), "no swap came");
    assert_eq!(log_of(&log), ["open:chat"]);
}

#[test]
fn a_job_in_the_idle_time_starts_the_idle_time_again() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_idle(directory.path(), 1);
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the first voice job ran");
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
    std::thread::sleep(Duration::from_millis(700));
    let (job, finished) = probe(VOICE, &log);
    let second = Instant::now();
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the second voice job ran");
    assert_eq!(swapped(&process), (vec!["voice".to_owned()], vec!["chat".to_owned()]));
    let after = second.elapsed();
    assert!(
        after >= Duration::from_millis(950),
        "the swap back came {after:?} after the second job; the idle time counts from it"
    );
}

/// A council request charges its context before its job waits for a swap (spec 3.3a, 4.2), so the ledger knows the
/// context terms of an engine that is not loaded.
#[test]
fn a_context_of_an_engine_that_is_not_loaded_can_be_charged() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    let ports = process.ports();
    ports.admit_context("voice").expect("a context of the unloaded `voice`");
    ports.release_context("voice").unwrap();
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    ports.admit_context("chat").expect("a context of the unloaded `chat`");
    ports.release_context("chat").unwrap();
}
