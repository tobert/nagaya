//! An engine with one device beside a loaded engine with stages (engine-abi "How a group loads both"): Gemma 4 on two
//! cards, and an on-demand engine on card 1 that fits beside it. The engine opens on the device thread of card 1, which
//! is also the stage thread of card 1 (spec 3.2). Its quanta run there, and a stage of the engine with stages runs on
//! card 0 at that time. A swap of the engine with stages never moves it (spec 3.3a, "No moves"). `swap_staged.rs` has the two
//! when they do not fit.

mod feed_support;
mod telemetry_support;

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::{Event, LoadError, LoadReply, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Engine, Identity, Kind, Limits, OutputsWanted, PassError, PassOut, Residual, Segment, SegmentInputs,
    StageHandle, StageOut, Stages, State, Tap, Tier,
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
        if self.name == "slow" || self.name == "slower" {
            self.inner.stages()?;
            return Some(self);
        }
        self.inner.stages()
    }
}

/// The engines `slow` and `slower` give stage handles that take 200 ms and 400 ms to drop, and write
/// `drop:<engine>:<stage>` after it.
impl Stages for Closing {
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>> {
        let handles = self.inner.stages().expect("the fake has stages").take_handles();
        let wait = if self.name == "slower" { 400 } else { 200 };
        handles
            .into_iter()
            .enumerate()
            .map(|(stage, inner)| {
                Box::new(SlowDrop {
                    inner,
                    label: format!("drop:{}:{stage}", self.name),
                    log: Arc::clone(&self.log),
                    wait: Duration::from_millis(wait),
                }) as Box<dyn StageHandle + Send>
            })
            .collect()
    }
}

/// A stage handle whose drop is slow, as a drop that frees device memory is.
struct SlowDrop {
    inner: Box<dyn StageHandle + Send>,
    label: String,
    log: Log,
    wait: Duration,
}

impl Drop for SlowDrop {
    fn drop(&mut self) {
        std::thread::sleep(self.wait);
        self.log.lock().unwrap().push(self.label.clone());
    }
}

impl StageHandle for SlowDrop {
    fn stage_pass(&mut self, segments: &[Segment], residual: &mut Residual) -> Result<StageOut, PassError> {
        self.inner.stage_pass(segments, residual)
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
}

/// `chat`, `paint` and `slow` are fakes of two stages on devices 0 and 1. `voice` and `draw` are plain fakes on device
/// 1.
struct Pair(Log);

fn fake(name: &str) -> Box<dyn Engine + Send> {
    match name {
        "chat" | "paint" | "slow" | "slower" => Box::new(feed_support::engine(2)),
        "voice" | "draw" | "late" => Box::new(FakeAlways::new()),
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
        // The open of `late` takes 600 ms, longer than the watch of a test with a short timeout.
        if name == "late" {
            std::thread::sleep(Duration::from_millis(600));
        }
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
        pinned: false,
    }
}

/// A pinned engine (spec 3.3a, "Explicit layout").
fn pinned(mut placement: EnginePlacement) -> EnginePlacement {
    placement.pinned = true;
    placement
}

/// Both engines fit on card 1 together.
fn start(directory: &std::path::Path) -> (Process, Log) {
    start_with(
        directory,
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            engine("voice", vec![1], Residency::OnDemand),
        ],
    )
}

fn start_with(directory: &std::path::Path, engines: &[EnginePlacement]) -> (Process, Log) {
    start_watched(directory, engines, Duration::from_secs(5))
}

/// As [`start_with`], with a watch that stops the process after `timeout` with no beat (spec 7.4).
fn start_watched(directory: &std::path::Path, engines: &[EnginePlacement], timeout: Duration) -> (Process, Log) {
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
        .timeout(timeout)
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
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

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
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

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

/// Load an engine, and wait for the end of its swap. A job for the engine runs only after that (spec 3.3).
fn load(process: &Process, engine: &str) -> (Vec<String>, Vec<String>) {
    process.load(engine).unwrap();
    swapped(process, Duration::from_secs(5))
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
fn a_load_of_an_engine_with_one_device_loads_it_beside_the_engine_with_stages() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    assert!(process.ports().engine_identity("voice").is_none(), "voice never opened");
    assert_eq!(
        process.load("voice").unwrap(),
        LoadReply::Started {
            unload: Vec::new(),
            load: vec!["voice".to_owned()],
        }
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (Vec::new(), vec!["voice".to_owned()])
    );
    // The process keeps the identity of an engine that loads beside the stages, as of any engine (spec 3.3a).
    assert_eq!(
        process.ports().engine_identity("voice").map(|identity| identity.model),
        Some(fake("voice").identity().model)
    );
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
            "open:voice@nagaya-dev-0-1",
            "run:voice@nagaya-dev-0-1"
        ]
    );
    // A job that arrives while `voice` is loaded runs on the same stage thread.
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the second voice job ran");
    assert_eq!(log_of(&log).last().unwrap(), "run:voice@nagaya-dev-0-1");
}

#[test]
fn a_stage_on_card_zero_runs_while_a_quantum_holds_card_one() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    load(&process, "voice");
    let (gate, held) = mpsc::channel();
    let (job, voice_done) = voice(&log, Some(held));
    process.submit_job(0, job).unwrap();
    wait_for(&log, "run:voice@nagaya-dev-0-1");
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
fn an_unload_of_an_engine_beside_runs_on_its_stage_thread_and_the_engine_with_stages_stays() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    assert_eq!(load(&process, "voice"), (Vec::new(), vec!["voice".to_owned()]));
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    process.unload("voice").unwrap();
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["voice".to_owned()], Vec::new())
    );
    assert!(log_of(&log).contains(&"close:voice".to_owned()));
    // `chat` stays loaded and runs; a new load of `voice` loads it beside `chat` again.
    let (job, finished) = chat_pass(&log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the pass of chat ran");
    load(&process, "voice");
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
            "open:voice@nagaya-dev-0-1",
            "open:voice@nagaya-dev-0-1"
        ]
    );
    assert!(!log_of(&log).contains(&"close:chat".to_owned()));
}

#[test]
fn a_stop_parks_and_closes_the_engine_beside() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start(directory.path());
    load(&process, "voice");
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

/// A load of an engine with stages unloads only the engines that it names. An engine beside it that fits stays loaded
/// on its device thread (spec 3.3a, "No moves").
#[test]
fn a_load_of_an_engine_with_stages_keeps_an_engine_that_fits_beside_it() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::OnDemand),
            engine("voice", vec![1], Residency::Resident),
        ],
    );
    assert_eq!(
        process.load("chat").unwrap(),
        LoadReply::Started {
            unload: Vec::new(),
            load: vec!["chat".to_owned()],
        }
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (Vec::new(), vec!["chat".to_owned()])
    );
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran beside chat");
    assert_eq!(
        log_of(&log),
        [
            "open:voice@nagaya-dev-0-1",
            "open:chat@nagaya-group-0",
            "run:voice@nagaya-dev-0-1"
        ]
    );
}

/// Each engine that a swap of the engine with stages does not name stays loaded on its device thread. It is ready
/// through the swap, and it never closes (spec 3.3a, "No moves").
fn assert_kept(process: &Process, log: &Log) {
    assert_eq!(
        process.ports().engine_state("voice"),
        Some(nagaya::notify::EngineState::Ready)
    );
    let (job, finished) = voice(log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    let opens = log_of(log)
        .iter()
        .filter(|line| *line == "open:voice@nagaya-dev-0-1")
        .count();
    assert_eq!(opens, 1, "voice opened once: {:?}", log_of(log));
    assert!(!log_of(log).contains(&"close:voice".to_owned()), "{:?}", log_of(log));
    assert_eq!(log_of(log).last().unwrap(), "run:voice@nagaya-dev-0-1");
}

/// An unload of the engine with stages and its load again keep the engine beside it loaded (spec 3.3a, "No moves").
#[test]
fn an_unload_and_a_load_of_the_engine_with_stages_keep_the_engine_beside_it() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            pinned(engine("voice", vec![1], Residency::Resident)),
        ],
    );
    assert_eq!(
        process.unload("chat").unwrap(),
        LoadReply::Started {
            unload: vec!["chat".to_owned()],
            load: Vec::new(),
        }
    );
    // A job that arrives during the swap waits for its end, and runs on the device thread of `voice`.
    let (job, during) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["chat".to_owned()], Vec::new())
    );
    during
        .recv_timeout(Duration::from_secs(5))
        .expect("the job of the swap ran");
    assert_kept(&process, &log);
    assert_eq!(load(&process, "chat"), (Vec::new(), vec!["chat".to_owned()]));
    assert_kept(&process, &log);
    let (job, finished) = chat_pass(&log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the pass of chat ran after its load again");
    let closes: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("close:"))
        .collect();
    assert_eq!(closes, ["close:chat"]);
}

/// The swaps of b70host on 2026-10-30: `chat` on both cards, `voice` pinned on card 1, `draw` on demand on card 1.
/// A load of `draw` that names `chat`, then a load of `chat` that names `draw`. `voice` stays loaded through both, and
/// each reply names only the engines of the load (spec 3.3a, "No moves").
#[test]
fn a_pinned_engine_stays_loaded_through_a_swap_out_and_back_of_the_engine_with_stages() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            pinned(engine("voice", vec![1], Residency::Resident)),
            engine("draw", vec![1], Residency::OnDemand),
        ],
    );
    assert_eq!(
        process.load_with("draw", &["chat".to_owned()]).unwrap(),
        LoadReply::Started {
            unload: vec!["chat".to_owned()],
            load: vec!["draw".to_owned()],
        }
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["chat".to_owned()], vec!["draw".to_owned()])
    );
    assert_kept(&process, &log);
    assert_eq!(
        process.load_with("chat", &["draw".to_owned()]).unwrap(),
        LoadReply::Started {
            unload: vec!["draw".to_owned()],
            load: vec!["chat".to_owned()],
        }
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["draw".to_owned()], vec!["chat".to_owned()])
    );
    assert_kept(&process, &log);
    assert_eq!(process.unload("voice"), Err(LoadError::Pinned("voice".to_owned())));
    let opens: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("open:"))
        .collect();
    assert_eq!(
        opens,
        [
            "open:chat@nagaya-group-0",
            "open:voice@nagaya-dev-0-1",
            "open:draw@nagaya-dev-0-1",
            "open:chat@nagaya-group-0"
        ]
    );
}

/// A resident engine shares its group with a resident engine with stages. At start it opens on the stage thread of its
/// device, after the engine with stages (spec 3.3a, "Explicit layout").
#[test]
fn a_resident_engine_opens_beside_a_resident_engine_with_stages_at_start() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            pinned(engine("voice", vec![1], Residency::Resident)),
        ],
    );
    assert_eq!(log_of(&log), ["open:chat@nagaya-group-0", "open:voice@nagaya-dev-0-1"]);
    let (job, finished) = voice(&log, None);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    assert_eq!(log_of(&log).last().unwrap(), "run:voice@nagaya-dev-0-1");
}

/// A job that tells its caller the state of its engine when the group ends it as unavailable.
struct Refused(Sender<String>);

impl JobBody for Refused {
    fn run(&mut self, kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        panic!("a {kind:?} quantum ran on an engine that is not ready")
    }

    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        let _ = self.0.send(format!("{engine}:{}", state.as_str()));
    }
}

#[test]
fn a_job_for_an_unloaded_engine_beside_the_stages_ends_as_unavailable() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let (tx, rx) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine: VOICE,
        body: Box::new(Refused(tx)),
    };
    process.submit_job(0, spec).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), "voice:unloaded");
    assert_eq!(log_of(&log), ["open:chat@nagaya-group-0"], "nothing loaded");
}

/// A load of a second engine with stages that names only the first in `unload` keeps the resident engine beside them
/// (spec 3.3a, "No moves").
#[test]
fn a_load_of_another_engine_with_stages_keeps_the_resident_engine_beside() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::OnDemand),
            engine("voice", vec![1], Residency::Resident),
            engine("paint", vec![0, 1], Residency::OnDemand),
        ],
    );
    assert_eq!(load(&process, "chat"), (Vec::new(), vec!["chat".to_owned()]));
    assert_eq!(
        process.load_with("paint", &["chat".to_owned()]).unwrap(),
        LoadReply::Started {
            unload: vec!["chat".to_owned()],
            load: vec!["paint".to_owned()],
        }
    );
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["chat".to_owned()], vec!["paint".to_owned()])
    );
    assert_kept(&process, &log);
}

/// A group holds one engine with stages at a time. A load of a second one that does not name the first is a refusal
/// that names it, and nothing changes (spec 3.2, "The device threads live as long as the group").
#[test]
fn a_load_of_a_second_engine_with_stages_names_the_first_in_the_way() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            engine("paint", vec![0, 1], Residency::OnDemand),
        ],
    );
    let error = process.load("paint").unwrap_err();
    assert!(
        matches!(&error, LoadError::InTheWay { engine, in_the_way, .. }
            if engine == "paint" && in_the_way == &["chat"]),
        "{error:?}"
    );
    assert_eq!(log_of(&log), ["open:chat@nagaya-group-0"], "nothing changed");
}

/// A job of `voice` that holds its device thread for a while, as steady embedding traffic does.
struct Busy {
    done: Sender<()>,
}

impl JobBody for Busy {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!(
            "engine `{engine}` is {}, and a job of a staying engine waits, never fails",
            state.as_str()
        );
    }

    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = engines[VOICE].identity();
        std::thread::sleep(Duration::from_millis(20));
        let _ = self.done.send(());
        Ran::Done
    }
}

/// A steady stream of jobs on a pinned engine beside the engine with stages does not hold off a swap of the engine with
/// stages, and each job of the stream runs (spec 3.3a, "The wait before an unload"). The swap waited for an idle
/// group before, so steady embeddings held it off with no end.
#[test]
fn a_steady_stream_on_a_pinned_engine_does_not_hold_off_a_swap_of_the_engine_with_stages() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start_with(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            pinned(engine("voice", vec![1], Residency::Resident)),
        ],
    );
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (done, finished) = mpsc::channel();
    let feeder = {
        let ports = process.ports();
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut sent = 0usize;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                // Two jobs wait at each moment, so the group is never idle.
                for _ in 0..2 {
                    let spec = JobSpec {
                        class: Class::Interactive,
                        priority: 0,
                        deadline: None,
                        first: Kind::Decode,
                        engine: VOICE,
                        body: Box::new(Busy { done: done.clone() }),
                    };
                    ports.submit_job(0, spec).unwrap();
                    sent += 1;
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            sent
        })
    };
    std::thread::sleep(Duration::from_millis(100));
    process.unload("chat").unwrap();
    let swap = swapped(&process, Duration::from_secs(5));
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    let sent = feeder.join().unwrap();
    assert_eq!(swap, (vec!["chat".to_owned()], Vec::new()));
    for job in 0..sent {
        finished
            .recv_timeout(Duration::from_secs(5))
            .unwrap_or_else(|_| panic!("job {job} of {sent} of the stream did not run"));
    }
}

/// The detach of an engine with stages waits until each stage handle drops, so its device memory is free before the
/// next open (spec 3.3a). The handles of `slow` take 200 ms each to drop.
#[test]
fn the_next_open_waits_for_the_drop_of_each_stage_handle() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_with(
        directory.path(),
        &[
            engine("slow", vec![0, 1], Residency::Resident),
            engine("paint", vec![0, 1], Residency::OnDemand),
        ],
    );
    process.load_with("paint", &["slow".to_owned()]).unwrap();
    assert_eq!(
        swapped(&process, Duration::from_secs(5)),
        (vec!["slow".to_owned()], vec!["paint".to_owned()])
    );
    let lines = log_of(&log);
    let at = |line: &str| {
        lines
            .iter()
            .position(|entry| entry == line)
            .unwrap_or_else(|| panic!("no `{line}`: {lines:?}"))
    };
    let open = at("open:paint@nagaya-group-0");
    assert!(at("drop:slow:0") < open, "{lines:?}");
    assert!(at("drop:slow:1") < open, "{lines:?}");
}

/// A slow detach does not stop the process: the stage thread beats while its handle drops, and the group thread beats
/// while it waits for the detach (spec 7.4). Each handle of `slower` takes 400 ms to drop, and the watch allows 300 ms.
#[test]
fn a_slow_detach_beats_for_the_stage_thread_and_the_group_thread() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_watched(
        directory.path(),
        &[
            engine("slower", vec![0, 1], Residency::Resident),
            engine("paint", vec![0, 1], Residency::OnDemand),
        ],
        Duration::from_millis(300),
    );
    process.load_with("paint", &["slower".to_owned()]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut swapped = false;
    // The watch may report a stall after the swap too, so the test watches a while longer.
    let mut after: Option<Instant> = None;
    while Instant::now() < deadline && after.is_none_or(|at| at.elapsed() < Duration::from_millis(500)) {
        match process.recv(Duration::from_millis(20)) {
            Some(Event::Stalled { .. }) => panic!("the watch saw a stall: {:?}", log_of(&log)),
            Some(Event::Swapped { .. }) => {
                swapped = true;
                after = Some(Instant::now());
            }
            _ => {}
        }
    }
    assert!(swapped, "no swap came in 5 s: {:?}", log_of(&log));
}

/// A stop that comes while a single-card swap opens an engine waits for the open, and the group thread beats while it
/// waits (spec 7.4). The open of `late` takes 600 ms, and the watch allows 300 ms.
#[test]
fn a_stop_during_a_single_card_swap_beats_while_it_waits() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start_watched(
        directory.path(),
        &[
            engine("chat", vec![0, 1], Residency::Resident),
            engine("late", vec![1], Residency::OnDemand),
        ],
        Duration::from_millis(300),
    );
    process.load("late").unwrap();
    wait_for(&log, "open:late@nagaya-dev-0-1");
    let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
    assert!(report.complete, "{:?}", report.parked);
    while let Some(event) = process.recv(Duration::from_millis(20)) {
        assert!(
            !matches!(event, Event::Stalled { .. }),
            "the watch saw a stall: {event:?}"
        );
    }
    assert!(log_of(&log).contains(&"close:late".to_owned()), "{:?}", log_of(&log));
}

/// A pass of `chat` runs stage by stage, and a job of `voice` runs on the device thread of card 1. The telemetry of
/// each job names the engine of the job: `voice` is not the engine with stages (spec 9.2, "The engine of a job").
#[test]
fn the_telemetry_of_a_job_beside_the_stages_names_the_engine_of_the_job() {
    let capture = nagaya_telemetry::testing::Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    load(&process, "voice");
    let (job, finished) = chat_pass(&log);
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the chat pass ran");
    let (mut job, finished) = voice(&log, None);
    job.first = Kind::Read;
    process.submit_job(0, job).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the voice job ran");
    telemetry_support::wait_for_jobs(&capture, 2);
    telemetry_support::assert_own_engines(&capture, &["chat", "voice"], &["chat", "voice"]);
}
