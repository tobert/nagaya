//! The swap in place on a group thread (spec 3.3a): a load of an engine that is not loaded starts the swap (spec
//! 3.3b). The group unloads the engines that must give their memory, closes them, and opens the engine.

mod telemetry_support;

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::{Event, LoadError, Process, Start};
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
        lab: toml::Table::new(),
        share: 1,
        residency,
        pinned: false,
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
        cpu: None,
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
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

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

/// Load an engine, and wait for the end of its swap. A job for the engine runs only after that (spec 3.3).
/// Load an engine, and give the swap. The load names the engines in its way in `unload`, as a client does after a
/// refusal (spec 3.3a, "Explicit layout").
fn load(process: &Process, engine: &str) -> (Vec<String>, Vec<String>) {
    match process.load(engine) {
        Err(nagaya::process::LoadError::InTheWay { in_the_way, .. }) => {
            process.load_with(engine, &in_the_way).unwrap();
        }
        answer => {
            answer.unwrap();
        }
    }
    swapped(process)
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
fn a_load_of_an_engine_on_demand_loads_it_in_place_of_the_resident_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    assert_eq!(
        load(&process, "voice"),
        (vec!["chat".to_owned()], vec!["voice".to_owned()])
    );
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "run:voice"]);
}

#[test]
fn a_load_of_the_resident_engine_swaps_it_back() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    load(&process, "voice");
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    load(&process, "chat");
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
    assert_eq!(
        load(&process, "voice"),
        (Vec::<String>::new(), vec!["voice".to_owned()])
    );
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    let (job, finished) = probe(CHAT, &log);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).expect("the chat job ran");
    assert_eq!(log_of(&log), ["open:chat", "open:voice", "run:voice", "run:chat"]);
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
    // The gate blocks the group thread in the chat quantum, so the load comes from another thread.
    let ports = process.ports();
    let loader = std::thread::spawn(move || ports.load_with("voice", &["chat".to_owned()]));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(log_of(&log), ["open:chat", "run:chat"], "the chat job holds its engine");
    gate_tx.send(()).unwrap();
    chat_finished.recv_timeout(Duration::from_secs(5)).unwrap();
    loader.join().unwrap().unwrap();
    swapped(&process);
    assert_eq!(log_of(&log), ["open:chat", "run:chat", "close:chat", "open:voice"]);
}

#[test]
fn a_stop_drops_a_load_that_waits_for_the_group() {
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
    // The gate blocks the group thread, so the load waits in the channel of the group.
    let ports = process.ports();
    let loader = std::thread::spawn(move || ports.load_with("voice", &["chat".to_owned()]));
    std::thread::sleep(Duration::from_millis(50));
    let stopper = std::thread::spawn(move || {
        let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
        (process, report)
    });
    std::thread::sleep(Duration::from_millis(50));
    gate_tx.send(()).unwrap();
    let (_process, _report) = stopper.join().unwrap();
    assert_eq!(
        loader.join().unwrap(),
        Err(LoadError::Stopped),
        "the stop dropped the load"
    );
    assert!(!log_of(&log).contains(&"open:voice".to_owned()), "{:?}", log_of(&log));
}

/// A job that feeds four tokens into slot 0 of `chat`, so that the slot holds a state.
struct Fill(Sender<()>);

impl JobBody for Fill {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let segment = Segment {
            control: Vec::new(),
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(vec![1, 2, 3, 4]),
            positions: (0..4).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted { logit_rows: vec![3] },
            layer_loop: None,
            cycles: None,
        };
        engines[CHAT].pass(&[segment], &[]).expect("a pass of the fake");
        let _ = self.0.send(());
        Ran::Done
    }
}

#[test]
fn the_unload_parks_each_state_that_holds_tokens() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path(), 1000);
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
    load(&process, "voice");
    assert_eq!(
        store.lock().unwrap().entries().len(),
        1,
        "the unload of `chat` parked the state of slot 0"
    );
}

#[test]
fn the_ledger_charges_the_engine_that_loaded() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path(), 1000);
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
    load(&process, "voice");
    assert_eq!(
        device(&process),
        400,
        "`voice` holds the 600 bytes that `chat` gave back"
    );
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path(), 2000);
    load(&process, "voice");
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
    // A second chat job queues behind the first, and then the load of voice comes.
    let (second, second_finished) = probe(CHAT, &log);
    process.submit_job(0, second).unwrap();
    let ports = process.ports();
    let loader = std::thread::spawn(move || ports.load_with("voice", &["chat".to_owned()]));
    std::thread::sleep(Duration::from_millis(50));
    gate_tx.send(()).unwrap();
    second_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the queued chat job ran");
    loader.join().unwrap().unwrap();
    swapped(&process);
    assert_eq!(
        log_of(&log),
        ["open:chat", "run:chat", "run:chat", "close:chat", "open:voice"]
    );
}

/// No idle time unloads an engine (spec 3.3a, "Explicit layout"). An on-demand engine with no job stays loaded, and
/// the resident engine that it displaced does not load again by itself.
#[test]
fn an_idle_engine_on_demand_stays_loaded_and_nothing_loads_again() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 1000);
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
    let (job, finished) = probe(VOICE, &log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    std::thread::sleep(Duration::from_millis(1500));
    assert!(process.recv(Duration::from_millis(10)).is_none(), "no swap came");
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "run:voice"]);
}

/// The terms of the contexts of an engine stay admitted while it is not loaded (spec 3.3a, "The ledger of a swap"),
/// so a context of an unloaded engine can be charged.
#[test]
fn a_context_of_an_engine_that_is_not_loaded_can_be_charged() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path(), 1000);
    let ports = process.ports();
    ports.admit_context("voice").expect("a context of the unloaded `voice`");
    ports.release_context("voice").unwrap();
    load(&process, "voice");
    ports.admit_context("chat").expect("a context of the unloaded `chat`");
    ports.release_context("chat").unwrap();
}

/// In a group of two engines on one device, the telemetry of a job names the engine of the job, not the joined name
/// `chat+voice` (spec 9.2, "The engine of a job").
#[test]
fn the_telemetry_of_a_job_names_the_engine_of_the_job() {
    let capture = nagaya_telemetry::testing::Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), 2000);
    assert_eq!(load(&process, "voice"), (Vec::new(), vec!["voice".to_owned()]));
    for engine in [CHAT, VOICE] {
        let (job, finished) = probe(engine, &log);
        process.submit_job(0, job).unwrap();
        finished.recv_timeout(Duration::from_secs(5)).expect("the job ran");
    }
    telemetry_support::wait_for_jobs(&capture, 2);
    telemetry_support::assert_own_engines(&capture, &["chat", "voice"], &["chat", "voice"]);
}
