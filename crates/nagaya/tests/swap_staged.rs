//! The swap in place on a group whose resident engine has stages (spec 3.3a, engine-abi D1): Gemma 4 on two cards,
//! and an on-demand engine on one of them. The two do not fit on card 1 together, so a job for the other engine
//! unloads the engine with stages, and a job for it unloads the other. `swap_beside.rs` has the two when they fit.

mod feed_support;

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::{Event, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Identity, Kind, Limits, PassError, PassOut, Segment, Stages, Tap, Tier};
use nagaya_sched::Class;

type Log = Arc<Mutex<Vec<String>>>;

/// An engine that writes its close to a log, declares `close`, and forwards the rest.
struct Closing {
    inner: Box<dyn Engine + Send>,
    name: String,
    log: Log,
}

impl Drop for Closing {
    fn drop(&mut self) {
        self.log.lock().unwrap().push(format!("close:{}", self.name));
    }
}

fn with_close(mut limits: Limits) -> Limits {
    limits.capabilities.push(Capability::Close);
    limits
}

impl Engine for Closing {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        with_close(self.inner.limits())
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.inner.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.inner.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.inner.coverage()
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        self.inner.stages()
    }
}

/// `chat` is the fake of two stages on devices 0 and 1. `voice` is the plain fake on device 1.
struct Pair(Log);

fn fake(name: &str) -> Box<dyn Engine + Send> {
    match name {
        "chat" => Box::new(feed_support::engine(2)),
        "voice" => Box::new(FakeAlways::new()),
        other => panic!("no fake named {other}"),
    }
}

impl Factory for Pair {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(with_close(fake(config["name"].as_str().unwrap()).limits()))
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let name = config["name"].as_str().unwrap().to_owned();
        self.0.lock().unwrap().push(format!("open:{name}"));
        Ok(Box::new(Closing {
            inner: fake(&name),
            name,
            log: Arc::clone(&self.0),
        }))
    }
}

fn engine(name: &str, devices: Vec<u32>, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("name".to_owned(), toml::Value::String(name.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: "pair".to_owned(),
        devices,
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
    }
}

/// Each engine takes 4,096 bytes of card 1, so a budget of 6,000 bytes holds one of them at a time.
fn profile() -> PlacementProfile {
    let budget = |tier: Tier| BudgetSpec {
        tiers: vec![TierSpec::from(tier)],
        bytes: if tier == Tier::Device(1) { 6000 } else { 1 << 20 },
    };
    PlacementProfile {
        engines: vec![
            engine("chat", vec![0, 1], Residency::Resident),
            engine("voice", vec![1], Residency::OnDemand),
        ],
        budgets: vec![budget(Tier::Device(0)), budget(Tier::Device(1)), budget(Tier::Host)],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn start(directory: &std::path::Path) -> (Process, Log) {
    start_watched(directory, Duration::from_secs(5))
}

/// The process of `start`, with a watchdog timeout of `timeout`.
fn start_watched(directory: &std::path::Path, timeout: Duration) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("pair", Box::new(Pair(Arc::clone(&log))));
    let inventory = Inventory {
        devices: (0..2)
            .map(|id| Device {
                id,
                name: format!("device-{id}"),
                pci_address: format!("0000:0{id}:00.0"),
                memory_bytes: 1 << 20,
                unified: false,
                ignored: false,
            })
            .collect(),
        host_bytes: 1 << 20,
    };
    let mut process = Start::new(profile(), registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(timeout)
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    let names: Vec<&str> = opened.iter().map(|engine| engine.name.as_str()).collect();
    assert_eq!(names, ["chat"], "only the resident engine opens at start");
    (process, log)
}

/// A job that reads the identity of its engine, which panics on an engine that is not loaded, and writes
/// `run:<engine>` to the log. It can wait for a gate first.
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

fn probe(engine: usize, log: &Log, gate: Option<Receiver<()>>) -> (JobSpec, Receiver<()>) {
    let (done, finished) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine,
        body: Box::new(Probe {
            engine,
            name: if engine == CHAT { "chat" } else { "voice" },
            log: Arc::clone(log),
            gate,
            done: Some(done),
        }),
    };
    (spec, finished)
}

fn log_of(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// Wait until an engine is ready. A job for the engine runs only after that (spec 3.3).
fn ready(process: &Process, engine: &str) {
    let ports = process.ports();
    let deadline = Instant::now() + Duration::from_secs(5);
    while ports.engine_state(engine) != Some(nagaya::notify::EngineState::Ready) {
        assert!(Instant::now() < deadline, "engine `{engine}` did not load in 5 s");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Load an engine, and wait until it is ready.
fn load(process: &Process, engine: &str) {
    process.load(engine).unwrap();
    ready(process, engine);
}

#[test]
fn a_load_of_an_on_demand_engine_unloads_the_engine_with_stages() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    load(&process, "voice");
    let (job, finished) = probe(VOICE, &log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "run:voice"]);
}

#[test]
fn a_load_of_the_engine_with_stages_swaps_it_back() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    load(&process, "voice");
    let (job, finished) = probe(VOICE, &log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    load(&process, "chat");
    let (job, finished) = probe(CHAT, &log, None);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).expect("the chat job ran");
    let (job, finished) = probe(CHAT, &log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("a second chat job ran");
    assert_eq!(
        log_of(&log),
        [
            "open:chat",
            "close:chat",
            "open:voice",
            "run:voice",
            "close:voice",
            "open:chat",
            "run:chat",
            "run:chat"
        ]
    );
}

#[test]
fn the_swap_waits_for_the_jobs_of_the_engine_with_stages() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let (gate_tx, gate_rx) = mpsc::channel();
    let (first, first_finished) = probe(CHAT, &log, Some(gate_rx));
    process.submit_job(0, first).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(&log).contains(&"run:chat".to_owned()) {
        assert!(Instant::now() < deadline, "the chat job did not start");
        std::thread::sleep(Duration::from_millis(5));
    }
    let (second, second_finished) = probe(CHAT, &log, None);
    process.submit_job(0, second).unwrap();
    // The gate can block the group thread, so the load comes from another thread.
    let ports = process.ports();
    let loader = std::thread::spawn(move || ports.load("voice"));
    std::thread::sleep(Duration::from_millis(50));
    gate_tx.send(()).unwrap();
    first_finished.recv_timeout(Duration::from_secs(5)).unwrap();
    second_finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the queued chat job ran");
    loader.join().unwrap().unwrap();
    ready(&process, "voice");
    assert_eq!(
        log_of(&log),
        ["open:chat", "run:chat", "run:chat", "close:chat", "open:voice"]
    );
}

/// `chat`, with stages, is on demand with an idle time of one second, and `voice` is resident.
fn start_with_stages_on_demand(directory: &std::path::Path) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("pair", Box::new(Pair(Arc::clone(&log))));
    let mut profile = profile();
    profile.engines[CHAT].residency = Residency::OnDemand;
    profile.engines[CHAT].idle_unload_s = Some(1);
    profile.engines[VOICE].residency = Residency::Resident;
    let inventory = Inventory {
        devices: (0..2)
            .map(|id| Device {
                id,
                name: format!("device-{id}"),
                pci_address: format!("0000:0{id}:00.0"),
                memory_bytes: 1 << 20,
                unified: false,
                ignored: false,
            })
            .collect(),
        host_bytes: 1 << 20,
    };
    let mut process = Start::new(profile, registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    let names: Vec<&str> = opened.iter().map(|engine| engine.name.as_str()).collect();
    assert_eq!(names, ["voice"]);
    (process, log)
}

#[test]
fn an_idle_engine_with_stages_on_demand_unloads_through_its_stage_threads() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with_stages_on_demand(directory.path());
    load(&process, "chat");
    let (job, finished) = probe(CHAT, &log, None);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).expect("the chat job ran");
    let deadline = Instant::now() + Duration::from_secs(5);
    while log_of(&log).len() < 6 {
        assert!(Instant::now() < deadline, "no swap back: {:?}", log_of(&log));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        log_of(&log),
        [
            "open:voice",
            "close:voice",
            "open:chat",
            "run:chat",
            "close:chat",
            "open:voice"
        ]
    );
}

/// The stage threads of an engine that unloads end, and the watchdog stops watching them (spec 7.4). On b70host on
/// 2026-10-24, a swap from gemma to the speech engine ended in a stall 30 s later, the watchdog timeout, and exit 1.
#[test]
fn a_swap_that_ends_the_stage_threads_is_not_a_stall() {
    let directory = tempfile::tempdir().unwrap();
    let timeout = Duration::from_millis(300);
    let (process, log) = start_watched(directory.path(), timeout);
    load(&process, "voice");
    let (job, finished) = probe(VOICE, &log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    let deadline = Instant::now() + timeout * 4;
    while Instant::now() < deadline {
        if let Some(event) = process.recv(Duration::from_millis(20)) {
            assert!(!matches!(event, Event::Stalled { .. }), "{event:?}");
        }
    }
}
