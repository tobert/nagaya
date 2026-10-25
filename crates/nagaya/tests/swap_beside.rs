//! An engine with one device beside a loaded engine with stages (engine-abi "How a group loads both"): Gemma 4 on two
//! cards, and an on-demand engine on card 1 that fits beside it. The engine opens on the stage thread of card 1, its
//! quanta run there, and a stage of the engine with stages runs on card 0 at that time. `swap_staged.rs` has the two
//! when they do not fit.

mod feed_support;

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
    Capability, Engine, Identity, Kind, Limits, OutputsWanted, PassError, PassOut, Segment, SegmentInputs, Stages,
    State, Tap, Tier,
};
use nagaya_sched::{Class, Item};

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
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        self.0.lock().unwrap().push(format!("open:{name}@{thread}"));
        Ok(Box::new(Closing {
            inner: fake(&name),
            name,
            log: Arc::clone(&self.0),
        }))
    }
}

fn engine(name: &str, devices: Vec<u32>, residency: Residency, idle_unload_s: Option<u64>) -> EnginePlacement {
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
        idle_unload_s,
    }
}

/// Both engines fit on card 1 together. `voice` unloads after `idle_unload_s` with no job.
fn start(directory: &std::path::Path, idle_unload_s: Option<u64>) -> (Process, Log) {
    start_with(
        directory,
        [
            engine("chat", vec![0, 1], Residency::Resident, None),
            engine("voice", vec![1], Residency::OnDemand, idle_unload_s),
        ],
    )
}

fn start_with(directory: &std::path::Path, engines: [EnginePlacement; 2]) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("pair", Box::new(Pair(Arc::clone(&log))));
    let budget = |tier: Tier| BudgetSpec {
        tiers: vec![TierSpec::from(tier)],
        bytes: 1 << 20,
    };
    let profile = PlacementProfile {
        engines: engines.to_vec(),
        budgets: vec![budget(Tier::Device(0)), budget(Tier::Device(1)), budget(Tier::Host)],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    };
    let inventory = Inventory {
        devices: (0..2)
            .map(|id| Device {
                id,
                name: format!("card-{id}"),
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
    process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, log)
}

const CHAT: usize = 0;
const VOICE: usize = 1;

/// A job of `voice` that writes `run:voice@<thread>` to the log. It can wait for a gate first.
struct Probe {
    log: Log,
    gate: Option<Receiver<()>>,
    done: Option<Sender<()>>,
}

impl JobBody for Probe {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = engines[VOICE].identity();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        self.log.lock().unwrap().push(format!("run:voice@{thread}"));
        if let Some(gate) = self.gate.take() {
            let _ = gate.recv_timeout(Duration::from_secs(10));
        }
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
        Ran::Done
    }
}

fn voice(log: &Log, gate: Option<Receiver<()>>) -> (JobSpec, Receiver<()>) {
    let (done, finished) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Decode,
        engine: VOICE,
        body: Box::new(Probe {
            log: Arc::clone(log),
            gate,
            done: Some(done),
        }),
    };
    (spec, finished)
}

/// A job of `chat` with one pass, which runs stage by stage on the two cards. It writes `begin:chat` when the pass
/// begins and `end:chat` when its last stage ends.
struct Pass {
    log: Log,
    done: Option<Sender<()>>,
}

impl JobBody for Pass {
    fn run(&mut self, kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        panic!("a {kind:?} quantum of a pass runs stage by stage")
    }

    fn item(&self, kind: Kind) -> Item {
        Item::Block(kind)
    }

    fn begin_pass(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        self.log.lock().unwrap().push("begin:chat".to_owned());
        Ok(vec![Segment {
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(vec![1, 2, 3]),
            positions: vec![0, 1, 2],
            meta: Vec::new(),
            outputs: OutputsWanted { logit_rows: vec![2] },
        }])
    }

    fn end_pass(&mut self, _kind: Kind, result: Result<PassOut, PassError>, _engines: &mut [Box<dyn Engine>]) -> Ran {
        result.expect("the pass of chat");
        self.log.lock().unwrap().push("end:chat".to_owned());
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
        Ran::Done
    }
}

fn chat_pass(log: &Log) -> (JobSpec, Receiver<()>) {
    let (done, finished) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Prefill,
        engine: CHAT,
        body: Box::new(Pass {
            log: Arc::clone(log),
            done: Some(done),
        }),
    };
    (spec, finished)
}

fn log_of(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn wait_for(log: &Log, line: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(log).iter().any(|entry| entry == line) {
        assert!(Instant::now() < deadline, "no `{line}` in 5 s: {:?}", log_of(log));
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The next `Swapped` event of the process.
fn swapped(process: &Process, within: Duration) -> (Vec<String>, Vec<String>) {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if let Some(Event::Swapped { unloaded, loaded, .. }) = process.recv(Duration::from_millis(50)) {
            return (unloaded, loaded.into_iter().map(|engine| engine.name).collect());
        }
    }
    panic!("no swap came in {within:?}");
}

#[test]
fn a_job_for_an_engine_with_one_device_loads_it_beside_the_engine_with_stages() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), None);
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    // `voice` opens and runs on the stage thread of card 1, and `chat` stays loaded.
    assert_eq!(
        log_of(&log),
        [
            "open:chat@nagaya-group-0",
            "open:voice@nagaya-stg-0-1",
            "run:voice@nagaya-stg-0-1"
        ]
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (Vec::new(), vec!["voice".to_owned()])
    );
    // A job that arrives while `voice` is loaded runs on the same stage thread.
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the second voice job ran");
    assert_eq!(log_of(&log).last().unwrap(), "run:voice@nagaya-stg-0-1");
}

#[test]
fn a_stage_on_card_zero_runs_while_a_quantum_holds_card_one() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), None);
    let (gate, held) = mpsc::channel();
    let (job, voice_done) = voice(&log, Some(held));
    process.submit_job(0, job).unwrap();
    wait_for(&log, "run:voice@nagaya-stg-0-1");
    // The quantum of `voice` holds card 1. The pass of `chat` begins, and its stage 0 runs on card 0.
    let (job, chat_done) = chat_pass(&log);
    process.submit_job(0, job).unwrap();
    wait_for(&log, "begin:chat");
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        !log_of(&log).contains(&"end:chat".to_owned()),
        "stage 1 of the pass waits for card 1"
    );
    gate.send(()).unwrap();
    voice_done.recv_timeout(Duration::from_secs(5)).unwrap();
    chat_done
        .recv_timeout(Duration::from_secs(5))
        .expect("the pass ended after card 1 came free");
    let log = log_of(&log);
    assert!(!log.iter().any(|line| line.starts_with("close:")), "{log:?}");
}

#[test]
fn an_idle_engine_beside_unloads_on_its_stage_thread_and_the_engine_with_stages_stays() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path(), Some(1));
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (Vec::new(), vec!["voice".to_owned()])
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["voice".to_owned()], Vec::new())
    );
    assert!(log_of(&log).contains(&"close:voice".to_owned()));
    // `chat` stays loaded and runs; a new job of `voice` loads it beside `chat` again.
    let (job, finished) = chat_pass(&log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the pass of chat ran");
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran again");
    let opens: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("open:"))
        .collect();
    assert_eq!(
        opens,
        [
            "open:chat@nagaya-group-0",
            "open:voice@nagaya-stg-0-1",
            "open:voice@nagaya-stg-0-1"
        ]
    );
    assert!(!log_of(&log).contains(&"close:chat".to_owned()));
}

#[test]
fn a_stop_parks_and_closes_the_engine_beside() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start(directory.path(), None);
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
    assert!(
        report.complete && report.parked.failures.is_empty(),
        "{:?}",
        report.parked
    );
    assert!(
        report.parked.parked_engines.contains(&"voice".to_owned()),
        "{:?}",
        report.parked.parked_engines
    );
    let log = log_of(&log);
    assert!(log.contains(&"close:voice".to_owned()), "{log:?}");
}

#[test]
fn an_idle_engine_with_stages_unloads_the_engine_beside_it_first() {
    let directory = tempfile::tempdir().unwrap();
    // `voice` is resident on card 1, and `chat` with stages is on demand with an idle time of one second.
    let (process, log) = start_with(
        directory.path(),
        [
            engine("chat", vec![0, 1], Residency::OnDemand, Some(1)),
            engine("voice", vec![1], Residency::Resident, None),
        ],
    );
    let (job, finished) = chat_pass(&log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the pass of chat ran");
    // A load of the engine with stages unloads the others first (engine-abi "How a group loads both").
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["voice".to_owned()], vec!["chat".to_owned()])
    );
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran beside chat");
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (Vec::new(), vec!["voice".to_owned()])
    );
    // `chat` goes idle. The loop unloads `voice` on its stage thread, then `chat`, and the ledger hears of both.
    let (mut unloaded, _) = swapped(&process, Duration::from_secs(5));
    unloaded.sort();
    assert_eq!(unloaded, ["chat", "voice"]);
    let closes: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("close:"))
        .collect();
    assert_eq!(closes, ["close:voice", "close:voice", "close:chat"]);
}
