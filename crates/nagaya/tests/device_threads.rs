//! A group of two devices with no engine with stages loaded (spec 3.2, "One device thread per device", Decided
//! 2026-10-28). `chat` with stages spans both cards, so the cards are one group. `left` on card 0 and `right` on
//! card 1 each open on the device thread of their card, and their quanta run there at once. A swap of an engine on one
//! card runs on the thread of that card, so the other card serves meanwhile (spec 3.3a, "Single-card swaps").

mod feed_support;
mod telemetry_support;

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::notify::EngineState;
use nagaya::process::{Event, Process, Start, WaitError};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Identity, Kind, Limits, PassError, PassOut, Segment, Stages, Tap, Tier};
use nagaya_sched::{Class, Item};

type Log = Arc<Mutex<Vec<String>>>;

/// The gates that hold a plan, an open or a close until the test opens them, by `plan:<name>`, `open:<name>` or
/// `close:<name>`.
type Gates = Arc<Mutex<BTreeMap<String, Receiver<()>>>>;

/// Wait at the gate of `key`, when the test set one. Before the wait, write `waits:<key>` to the log.
fn wait_at(gates: &Gates, log: &Log, key: &str) {
    let gate = gates.lock().unwrap().remove(key);
    if let Some(gate) = gate {
        log.lock().unwrap().push(format!("waits:{key}"));
        let _ = gate.recv_timeout(Duration::from_secs(10));
    }
}

/// An engine that writes its close to a log, declares `close`, and forwards the rest.
struct Closing {
    inner: Box<dyn Engine + Send>,
    name: String,
    log: Log,
    gates: Gates,
}

impl Drop for Closing {
    fn drop(&mut self) {
        wait_at(&self.gates, &self.log, &format!("close:{}", self.name));
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

/// An engine of two devices with no stages: the plain fake, with its device terms on each card.
struct Wide(FakeAlways);

impl Engine for Wide {
    fn identity(&self) -> Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        let mut limits = self.0.limits();
        let second: Vec<_> = limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.tier == Tier::Device(0))
            .map(|term| {
                let mut term = term.clone();
                term.tier = Tier::Device(1);
                term.name = format!("{}-1", term.name);
                term
            })
            .collect();
        limits.memory.mandatory.extend(second);
        limits.devices = vec![0, 1];
        limits
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }
}

fn fake(name: &str) -> Box<dyn Engine + Send> {
    match name {
        "chat" => Box::new(feed_support::engine(2)),
        "left" | "right" => Box::new(FakeAlways::new()),
        "wide" => Box::new(Wide(FakeAlways::new())),
        other => panic!("no fake named {other}"),
    }
}

/// The factory of the fakes. It writes each open with the name of the thread that opens. An open of `left` fails
/// with this kind. An open waits at its gate.
struct Fakes(Log, Option<OpenErrorKind>, Gates);

impl Factory for Fakes {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        let name = config["name"].as_str().unwrap();
        wait_at(&self.2, &self.0, &format!("plan:{name}"));
        Ok(with_close(fake(name).limits()))
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let name = config["name"].as_str().unwrap().to_owned();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        self.0.lock().unwrap().push(format!("open:{name}@{thread}"));
        wait_at(&self.2, &self.0, &format!("open:{name}"));
        if let (Some(kind), "left") = (self.1, name.as_str()) {
            return Err(OpenError {
                kind,
                message: "a part size differs from the plan".to_owned(),
            });
        }
        Ok(Box::new(Closing {
            inner: fake(&name),
            name,
            log: Arc::clone(&self.0),
            gates: Arc::clone(&self.2),
        }))
    }
}

fn engine(name: &str, devices: Vec<u32>, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("name".to_owned(), toml::Value::String(name.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: "fakes".to_owned(),
        devices,
        config,
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        residency,
        pinned: false,
    }
}

const CHAT: usize = 0;
const LEFT: usize = 1;
const RIGHT: usize = 2;

/// The mandatory bytes of a fake on its first device.
fn device_bytes(name: &str) -> u64 {
    with_close(fake(name).limits())
        .memory
        .mandatory
        .iter()
        .filter(|term| term.tier == Tier::Device(0))
        .map(|term| term.bytes)
        .sum()
}

/// Start `chat`, `left` and `right` with these residencies. `card0` is the budget of card 0. Give the process, the
/// log, and the names of the engines that opened at the start.
fn start(residency: [Residency; 3], card0: u64, directory: &std::path::Path) -> (Process, Log, Vec<String>) {
    let (mut process, log) = start_failing(residency, card0, directory, None, false);
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    let names = opened.into_iter().map(|engine| engine.name).collect();
    (process, log, names)
}

/// Start the engines, and do not wait for the opens. With `fail`, an open of `left` fails with that kind.
fn start_failing(
    residency: [Residency; 3],
    card0: u64,
    directory: &std::path::Path,
    fail: Option<OpenErrorKind>,
    wide: bool,
) -> (Process, Log) {
    start_gated(residency, card0, directory, fail, wide, Gates::default())
}

/// Start the engines with these gates on their opens and closes, and do not wait for the opens.
fn start_gated(
    residency: [Residency; 3],
    card0: u64,
    directory: &std::path::Path,
    fail: Option<OpenErrorKind>,
    wide: bool,
    gates: Gates,
) -> (Process, Log) {
    start_on(residency, card0, directory, fail, wide, gates, vec![0, 1])
}

/// Start the engines with `chat` (or `wide`) on the machine devices `first`, in this order.
fn start_on(
    residency: [Residency; 3],
    card0: u64,
    directory: &std::path::Path,
    fail: Option<OpenErrorKind>,
    wide: bool,
    gates: Gates,
    first: Vec<u32>,
) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("fakes", Box::new(Fakes(Arc::clone(&log), fail, gates)));
    let budget = |tier: Tier, bytes| BudgetSpec {
        tiers: vec![TierSpec::from(tier)],
        bytes,
    };
    // `wide` takes the place of `chat`: an engine of both cards with no stages.
    let first = match wide {
        true => engine("wide", first, residency[CHAT]),
        false => engine("chat", first, residency[CHAT]),
    };
    let profile = PlacementProfile {
        engines: vec![
            first,
            engine("left", vec![0], residency[LEFT]),
            engine("right", vec![1], residency[RIGHT]),
        ],
        budgets: vec![
            budget(Tier::Device(0), card0),
            budget(Tier::Device(1), 1 << 20),
            budget(Tier::Host, 1 << 20),
        ],
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
    let process = Start::new(profile, registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    (process, log)
}

/// A job that writes `run:<engine>@<thread>` to the log, tells `started`, and can wait for a gate before it ends.
struct Probe {
    engine: usize,
    log: Log,
    started: Sender<()>,
    gate: Option<Receiver<()>>,
    done: Sender<()>,
}

impl JobBody for Probe {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        // The engine is on this thread: its identity answers here.
        let _ = engines[self.engine].identity();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        let engine = ["chat", "left", "right"][self.engine];
        self.log.lock().unwrap().push(format!("run:{engine}@{thread}"));
        let _ = self.started.send(());
        if let Some(gate) = self.gate.take() {
            let _ = gate.recv_timeout(Duration::from_secs(10));
        }
        let _ = self.done.send(());
        Ran::Done
    }
}

/// A job of `engine`, with the receivers of its start and its end.
fn probe(engine: usize, log: &Log, gate: Option<Receiver<()>>) -> (JobSpec, Receiver<()>, Receiver<()>) {
    let (started, started_rx) = mpsc::channel();
    let (done, done_rx) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Decode,
        engine,
        body: Box::new(Probe {
            engine,
            log: Arc::clone(log),
            started,
            gate,
            done,
        }),
    };
    (spec, started_rx, done_rx)
}

fn log_of(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// The next `Swapped` event of the process.
fn swapped(process: &Process) -> (Vec<String>, Vec<String>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Some(Event::Swapped { unloaded, loaded, .. }) = process.recv(Duration::from_millis(50)) {
            return (unloaded, loaded.into_iter().map(|engine| engine.name).collect());
        }
    }
    panic!("no swap in 5 s");
}

fn load(process: &Process, engine: &str) -> (Vec<String>, Vec<String>) {
    process.load(engine).unwrap();
    swapped(process)
}

#[test]
fn engines_on_two_cards_open_on_their_device_threads_and_run_at_once() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log, opened) = start([Residency::OnDemand; 3], 1 << 20, directory.path());
    assert!(opened.is_empty(), "{opened:?}");
    assert_eq!(load(&process, "left"), (Vec::new(), vec!["left".to_owned()]));
    assert_eq!(load(&process, "right"), (Vec::new(), vec!["right".to_owned()]));
    assert_eq!(log_of(&log), ["open:left@nagaya-dev-0-0", "open:right@nagaya-dev-0-1"]);
    // A job of `left` holds card 0 until its gate opens. A job of `right` runs on card 1 meanwhile.
    let (gate_tx, gate) = mpsc::channel();
    let (job, left_started, left_done) = probe(LEFT, &log, Some(gate));
    process.submit_job(0, job).unwrap();
    left_started
        .recv_timeout(Duration::from_secs(5))
        .expect("the left job started");
    let (job, _, right_done) = probe(RIGHT, &log, None);
    process.submit_job(0, job).unwrap();
    right_done
        .recv_timeout(Duration::from_secs(5))
        .expect("the right job ran on card 1 while the left job held card 0");
    assert!(left_done.try_recv().is_err(), "the left job waits for its gate");
    gate_tx.send(()).unwrap();
    left_done
        .recv_timeout(Duration::from_secs(5))
        .expect("the left job ended");
    let runs: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("run:"))
        .collect();
    assert_eq!(runs, ["run:left@nagaya-dev-0-0", "run:right@nagaya-dev-0-1"]);
}

/// With `chat` on `devices = [1, 0]`, its stage 0 is on card 1 (spec 2.4). The device threads stay in machine order,
/// so `left` on card 0 still opens and runs on the thread of card 0, and `right` on the thread of card 1.
#[test]
fn engines_on_one_card_keep_the_thread_of_their_card_beside_an_engine_on_devices_one_then_zero() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start_on(
        [Residency::Resident, Residency::Resident, Residency::Resident],
        1 << 20,
        directory.path(),
        None,
        false,
        Gates::default(),
        vec![1, 0],
    );
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    let mut names: Vec<String> = opened.into_iter().map(|engine| engine.name).collect();
    names.sort();
    assert_eq!(names, ["chat", "left", "right"]);
    let opens: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| !line.starts_with("open:chat"))
        .collect();
    assert_eq!(opens, ["open:left@nagaya-dev-0-0", "open:right@nagaya-dev-0-1"]);
    for engine in [LEFT, RIGHT] {
        let (job, _, done) = probe(engine, &log, None);
        process.submit_job(0, job).unwrap();
        done.recv_timeout(Duration::from_secs(5)).expect("the job ran");
    }
    let runs: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("run:"))
        .collect();
    assert_eq!(runs, ["run:left@nagaya-dev-0-0", "run:right@nagaya-dev-0-1"]);
}

#[test]
fn a_resident_engine_with_one_device_opens_on_its_device_thread_at_the_start() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log, opened) = start(
        [Residency::OnDemand, Residency::Resident, Residency::OnDemand],
        1 << 20,
        directory.path(),
    );
    assert_eq!(
        opened,
        ["left"],
        "the start reports the engine that opened on its device thread"
    );
    assert_eq!(log_of(&log), ["open:left@nagaya-dev-0-0"]);
    let (job, _, done) = probe(LEFT, &log, None);
    process.submit_job(0, job).unwrap();
    done.recv_timeout(Duration::from_secs(5)).expect("the left job ran");
    assert_eq!(log_of(&log).last().unwrap(), "run:left@nagaya-dev-0-0");
}

#[test]
fn a_swap_out_of_the_engine_with_stages_opens_the_engine_on_its_device_thread() {
    let directory = tempfile::tempdir().unwrap();
    // Card 0 holds `chat` or `left`, not both, so a load of `left` unloads `chat`.
    let (chat, left) = (device_bytes("chat"), device_bytes("left"));
    let card0 = chat.max(left) + chat.min(left) / 2;
    let (process, log, opened) = start(
        [Residency::Resident, Residency::OnDemand, Residency::OnDemand],
        card0,
        directory.path(),
    );
    assert_eq!(opened, ["chat"]);
    process.load_with("left", &["chat".to_owned()]).unwrap();
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["left".to_owned()]));
    let (job, _, done) = probe(LEFT, &log, None);
    process.submit_job(0, job).unwrap();
    done.recv_timeout(Duration::from_secs(5)).expect("the left job ran");
    assert_eq!(
        log_of(&log),
        [
            "open:chat@nagaya-group-0",
            "close:chat",
            "open:left@nagaya-dev-0-0",
            "run:left@nagaya-dev-0-0"
        ]
    );
    // An unload of `left` loads nothing (spec 3.3a, "Explicit layout"). A load of `chat` opens it on the group thread.
    process.unload("left").unwrap();
    assert_eq!(swapped(&process), (vec!["left".to_owned()], Vec::new()));
    assert_eq!(load(&process, "chat"), (Vec::new(), vec!["chat".to_owned()]));
    assert_eq!(log_of(&log)[4..], ["close:left", "open:chat@nagaya-group-0"]);
}

/// A resident engine whose open on its device thread fails keeps the kind of its failure, so a part size that differs
/// from the plan refuses the start (spec 10.4 row 3), as on the group thread.
#[test]
fn a_failed_open_on_a_device_thread_keeps_its_kind() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start_failing(
        [Residency::OnDemand, Residency::Resident, Residency::OnDemand],
        1 << 20,
        directory.path(),
        Some(OpenErrorKind::PartSize),
        false,
    );
    let error = process.wait_opens(Duration::from_secs(5)).unwrap_err();
    let WaitError::Event(event) = error else {
        panic!("the start ends with an event, not {error:?}");
    };
    assert_eq!(
        event.exit_code(),
        Some(nagaya::exit::ExitCode::RefusedStart),
        "{event:?}"
    );
    assert_eq!(log_of(&log), ["open:left@nagaya-dev-0-0"]);
    process.abort();
}

/// A job of a council text engine wants a block for its pass and can ask for a copy first. On an engine that is not
/// the engine with stages, such as an engine of two devices with no stages, its quanta hold every device and run on
/// the group thread, where the engine is.
struct Text {
    log: Log,
    first: bool,
    done: Sender<()>,
}

impl JobBody for Text {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    /// The prefill adds two decodes. The first decode asks for a copy first, so the copy runs before the second.
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = engines[CHAT].identity();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        self.log.lock().unwrap().push(format!("run:{kind:?}@{thread}"));
        match kind {
            Kind::Prefill => Ran::More(vec![Kind::Decode, Kind::Decode]),
            Kind::Decode if self.first => {
                self.first = false;
                Ran::First(vec![Kind::Copy])
            }
            Kind::Decode => {
                let _ = self.done.send(());
                Ran::Done
            }
            _ => Ran::Done,
        }
    }

    fn item(&self, kind: Kind) -> Item {
        match kind {
            Kind::Copy => Item::Whole(kind),
            _ => Item::Block(kind),
        }
    }
}

#[test]
fn a_text_job_of_an_engine_on_the_group_thread_runs_there_with_its_copy_first() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start_failing(
        [Residency::Resident, Residency::OnDemand, Residency::OnDemand],
        1 << 20,
        directory.path(),
        None,
        true,
    );
    let opened: Vec<String> = process
        .wait_opens(Duration::from_secs(5))
        .unwrap()
        .into_iter()
        .map(|engine| engine.name)
        .collect();
    assert_eq!(opened, ["wide"]);
    let (done, finished) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Prefill,
        engine: CHAT,
        body: Box::new(Text {
            log: Arc::clone(&log),
            first: true,
            done,
        }),
    };
    process.submit_job(0, spec).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the text job of wide ran");
    let runs: Vec<String> = log_of(&log)
        .into_iter()
        .filter(|line| line.starts_with("run:"))
        .collect();
    assert_eq!(
        runs,
        [
            "run:Prefill@nagaya-group-0",
            "run:Decode@nagaya-group-0",
            "run:Copy@nagaya-group-0",
            "run:Decode@nagaya-group-0"
        ]
    );
}

/// A stop counts a job whose quantum runs on a device thread as cancelled, as a job that waits.
#[test]
fn a_stop_cancels_a_job_whose_quantum_runs_on_a_device_thread() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log, _) = start([Residency::OnDemand; 3], 1 << 20, directory.path());
    load(&process, "left");
    let (gate_tx, gate) = mpsc::channel();
    let (job, started, _done) = probe(LEFT, &log, Some(gate));
    process.submit_job(0, job).unwrap();
    started
        .recv_timeout(Duration::from_secs(5))
        .expect("the left job started");
    // The quantum ends a moment after the stop begins, so the stop waits for it.
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        let _ = gate_tx.send(());
    });
    let report = process.stop(Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
    release.join().unwrap();
    assert_eq!(report.parked.cancelled, 1, "{report:?}");
}

/// Wait until the log holds `line`.
fn wait_for_line(log: &Log, line: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if log_of(log).iter().any(|held| held == line) {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("the log has no line `{line}` in 5 s: {:?}", log_of(log));
}

/// A gate on the open or the close with this key, and the sender that opens it.
fn gate_on(key: &str) -> (Gates, Sender<()>) {
    let gates = Gates::default();
    let (open, gate) = mpsc::channel();
    gates.lock().unwrap().insert(key.to_owned(), gate);
    (gates, open)
}

/// A job of `right` on card 1 runs to its end while an engine of card 0 waits in a swap.
fn right_runs_while_card_0_swaps(process: &Process, log: &Log) {
    let (job, _, done) = probe(RIGHT, log, None);
    process.submit_job(0, job).unwrap();
    done.recv_timeout(Duration::from_secs(2))
        .expect("a job of right on card 1 ran while the swap of card 0 waited");
}

/// A load of `left` runs on the thread of card 0. A job of `right` that holds card 1 ends, and the next job of
/// `right` runs, while the open of `left` waits (spec 3.3a, "Single-card swaps").
#[test]
fn a_load_on_card_0_does_not_delay_the_next_quantum_on_card_1() {
    let directory = tempfile::tempdir().unwrap();
    let (gates, open_left) = gate_on("open:left");
    let (mut process, log) = start_gated([Residency::OnDemand; 3], 1 << 20, directory.path(), None, false, gates);
    assert!(process.wait_opens(Duration::from_secs(5)).unwrap().is_empty());
    assert_eq!(load(&process, "right"), (Vec::new(), vec!["right".to_owned()]));
    // A job of `right` holds card 1 when the swap of card 0 begins.
    let (gate_tx, gate) = mpsc::channel();
    let (job, started, held) = probe(RIGHT, &log, Some(gate));
    process.submit_job(0, job).unwrap();
    started
        .recv_timeout(Duration::from_secs(5))
        .expect("the right job started");
    process.load("left").unwrap();
    wait_for_line(&log, "waits:open:left");
    gate_tx.send(()).unwrap();
    held.recv_timeout(Duration::from_secs(5))
        .expect("the held right job ended");
    right_runs_while_card_0_swaps(&process, &log);
    assert_eq!(process.ports().engine_state("left"), Some(EngineState::Loading));
    open_left.send(()).unwrap();
    assert_eq!(swapped(&process), (Vec::new(), vec!["left".to_owned()]));
    assert_eq!(process.ports().engine_state("left"), Some(EngineState::Ready));
    let (job, _, done) = probe(LEFT, &log, None);
    process.submit_job(0, job).unwrap();
    done.recv_timeout(Duration::from_secs(5)).expect("the left job ran");
    assert_eq!(log_of(&log).last().unwrap(), "run:left@nagaya-dev-0-0");
}

/// A swap admits from the plans of the start, so a slow plan does not hold the group thread (spec 3.3, 3.3a). A plan
/// of a real checkpoint takes about a second, and each swap check planned every engine of the profile.
#[test]
fn a_swap_on_card_0_plans_no_engine_and_card_1_runs_on() {
    let directory = tempfile::tempdir().unwrap();
    let gates = Gates::default();
    let (mut process, log) = start_gated(
        [Residency::OnDemand; 3],
        1 << 20,
        directory.path(),
        None,
        false,
        Arc::clone(&gates),
    );
    assert!(process.wait_opens(Duration::from_secs(5)).unwrap().is_empty());
    assert_eq!(load(&process, "right"), (Vec::new(), vec!["right".to_owned()]));
    // From here, a plan of `chat` waits at its gate for 10 s. The test never opens it.
    let (_never, gate) = mpsc::channel();
    gates.lock().unwrap().insert("plan:chat".to_owned(), gate);
    let (gate_tx, gate) = mpsc::channel();
    let (job, started, held) = probe(RIGHT, &log, Some(gate));
    process.submit_job(0, job).unwrap();
    started
        .recv_timeout(Duration::from_secs(5))
        .expect("the right job started");
    let ports = process.ports();
    std::thread::scope(|scope| {
        let loading = scope.spawn(move || ports.load("left").unwrap());
        std::thread::sleep(Duration::from_millis(200));
        gate_tx.send(()).unwrap();
        held.recv_timeout(Duration::from_secs(2))
            .expect("the held right job ended while card 0 swapped");
        right_runs_while_card_0_swaps(&process, &log);
        loading.join().unwrap();
    });
    assert_eq!(swapped(&process), (Vec::new(), vec!["left".to_owned()]));
    assert!(
        !log_of(&log).iter().any(|line| line.starts_with("waits:plan")),
        "a swap planned an engine: {:?}",
        log_of(&log)
    );
}

/// An unload of `left` closes it on the thread of card 0. A job of `right` runs while the close waits.
#[test]
fn an_unload_on_card_0_does_not_delay_a_job_on_card_1() {
    let directory = tempfile::tempdir().unwrap();
    let (gates, close_left) = gate_on("close:left");
    let (mut process, log) = start_gated([Residency::OnDemand; 3], 1 << 20, directory.path(), None, false, gates);
    assert!(process.wait_opens(Duration::from_secs(5)).unwrap().is_empty());
    load(&process, "left");
    load(&process, "right");
    process.unload("left").unwrap();
    wait_for_line(&log, "waits:close:left");
    right_runs_while_card_0_swaps(&process, &log);
    assert_eq!(process.ports().engine_state("left"), Some(EngineState::Draining));
    close_left.send(()).unwrap();
    assert_eq!(swapped(&process), (vec!["left".to_owned()], Vec::new()));
    assert!(log_of(&log).contains(&"close:left".to_owned()));
    assert_eq!(process.ports().engine_state("left"), Some(EngineState::Unloaded));
}

/// An engine of both cards opens on the group thread, not on a device thread (spec 3.3a). The other jobs of the group
/// wait. The engine of a device thread stays loaded there, because the load does not name it (spec 3.3a, "No moves").
#[test]
fn an_engine_of_both_cards_swaps_on_the_group_thread() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start_failing([Residency::OnDemand; 3], 1 << 20, directory.path(), None, true);
    assert!(process.wait_opens(Duration::from_secs(5)).unwrap().is_empty());
    assert_eq!(load(&process, "left"), (Vec::new(), vec!["left".to_owned()]));
    assert_eq!(load(&process, "wide"), (Vec::new(), vec!["wide".to_owned()]));
    assert_eq!(log_of(&log), ["open:left@nagaya-dev-0-0", "open:wide@nagaya-group-0"]);
    assert_eq!(process.ports().engine_state("left"), Some(EngineState::Ready));
}

/// A load that arrives while a swap runs on a device thread starts its swap after that swap ends, so the ledger sees
/// the swaps in order (spec 3.3a, "Single-card swaps").
#[test]
fn a_second_swap_waits_for_the_swap_that_runs() {
    let directory = tempfile::tempdir().unwrap();
    let (gates, open_left) = gate_on("open:left");
    let (mut process, log) = start_gated([Residency::OnDemand; 3], 1 << 20, directory.path(), None, false, gates);
    assert!(process.wait_opens(Duration::from_secs(5)).unwrap().is_empty());
    process.load("left").unwrap();
    wait_for_line(&log, "waits:open:left");
    // The ask waits for its group to plan it, so it runs on its own thread.
    let ports = process.ports();
    let right = std::thread::spawn(move || ports.load("right"));
    std::thread::sleep(Duration::from_millis(100));
    assert!(!log_of(&log).iter().any(|line| line.starts_with("open:right")));
    open_left.send(()).unwrap();
    assert_eq!(swapped(&process), (Vec::new(), vec!["left".to_owned()]));
    assert_eq!(swapped(&process), (Vec::new(), vec!["right".to_owned()]));
    right.join().unwrap().unwrap();
    assert_eq!(
        log_of(&log),
        [
            "open:left@nagaya-dev-0-0",
            "waits:open:left",
            "open:right@nagaya-dev-0-1"
        ]
    );
}

/// A stop during a swap on a device thread waits for the open, and then parks the engine that opened.
#[test]
fn a_stop_during_a_swap_parks_the_engine_that_opens() {
    let directory = tempfile::tempdir().unwrap();
    let (gates, open_left) = gate_on("open:left");
    let (mut process, log) = start_gated([Residency::OnDemand; 3], 1 << 20, directory.path(), None, false, gates);
    assert!(process.wait_opens(Duration::from_secs(5)).unwrap().is_empty());
    load(&process, "right");
    process.load("left").unwrap();
    wait_for_line(&log, "waits:open:left");
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        let _ = open_left.send(());
    });
    let report = process.stop(Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
    release.join().unwrap();
    assert!(report.complete, "{report:?}");
    let mut parked = report.parked.parked_engines.clone();
    parked.sort();
    assert_eq!(parked, ["left", "right"], "{report:?}");
}

/// A job of `engine` whose first quantum is a read, so that it has a quantum span.
fn read_probe(engine: usize, log: &Log) -> (JobSpec, Receiver<()>) {
    let (mut spec, _, done) = probe(engine, log, None);
    spec.first = Kind::Read;
    (spec, done)
}

/// Engines on the device threads of two cards, each loaded by a single-card swap. The telemetry of each job names the
/// engine of the job, not the joined name `chat+left+right` (spec 9.2, "The engine of a job").
#[test]
fn the_telemetry_of_a_job_on_a_device_thread_names_the_engine_of_the_job() {
    let capture = nagaya_telemetry::testing::Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let (process, log, _) = start([Residency::OnDemand; 3], 1 << 20, directory.path());
    load(&process, "left");
    load(&process, "right");
    for engine in [LEFT, RIGHT] {
        let (job, done) = read_probe(engine, &log);
        process.submit_job(0, job).unwrap();
        done.recv_timeout(Duration::from_secs(5)).expect("the job ran");
    }
    // A decode quantum makes metrics only (spec 9.2).
    let (job, _, done) = probe(LEFT, &log, None);
    process.submit_job(0, job).unwrap();
    done.recv_timeout(Duration::from_secs(5)).expect("the decode job ran");
    telemetry_support::wait_for_jobs(&capture, 3);
    telemetry_support::assert_own_engines(&capture, &["left", "right"], &["left", "right"]);
}

/// A job of an engine of two cards with no stages runs on the group thread. Its telemetry names `wide`, and a job of
/// `left` on its device thread names `left` (spec 9.2, "The engine of a job").
#[test]
fn the_telemetry_of_a_job_on_the_group_thread_names_the_engine_of_the_job() {
    let capture = nagaya_telemetry::testing::Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let (mut process, log) = start_failing(
        [Residency::Resident, Residency::OnDemand, Residency::OnDemand],
        1 << 20,
        directory.path(),
        None,
        true,
    );
    process.wait_opens(Duration::from_secs(5)).unwrap();
    load(&process, "left");
    let (done, finished) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Prefill,
        engine: CHAT,
        body: Box::new(Text {
            log: Arc::clone(&log),
            first: true,
            done,
        }),
    };
    process.submit_job(0, spec).unwrap();
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("the text job of wide ran");
    let (job, done) = read_probe(LEFT, &log);
    process.submit_job(0, job).unwrap();
    done.recv_timeout(Duration::from_secs(5)).expect("the left job ran");
    telemetry_support::wait_for_jobs(&capture, 2);
    telemetry_support::assert_own_engines(&capture, &["wide", "left"], &["wide", "left"]);
}

/// A body that ends its job in its first quantum with `outcome`.
struct Ends {
    outcome: Ran,
    done: Sender<()>,
}

impl JobBody for Ends {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = self.done.send(());
        self.outcome.clone()
    }
}

/// The failure lines of the log for one job of `left`, whose quantum runs on the device thread of card 0 and ends
/// with `outcome`.
fn failure_lines_on_a_device_thread(outcome: Ran) -> Vec<String> {
    let directory = tempfile::tempdir().unwrap();
    let lines = nagaya_telemetry::testing::log_lines(|| {
        let (mut process, _log, _) = start([Residency::OnDemand; 3], 1 << 20, directory.path());
        load(&process, "left");
        let (done, done_rx) = mpsc::channel();
        let spec = JobSpec {
            class: Class::Interactive,
            priority: 0,
            deadline: None,
            first: Kind::Copy,
            engine: LEFT,
            body: Box::new(Ends { outcome, done }),
        };
        process.submit_job(0, spec).unwrap();
        done_rx.recv_timeout(Duration::from_secs(5)).expect("the job ran");
        // The stop joins the threads, so each line is written when it returns.
        process.stop(Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
    });
    lines
        .into_iter()
        .filter(|line| line.starts_with("quantum failed") || line.starts_with("job refused"))
        .collect()
}

/// A refused job on a device thread logs no failure line (spec 9.6, "A refused job is not a failed quantum"), and a
/// failed one logs its line. The live service of 2026-11-15 logged a line for each fork that waited for a slot.
///
/// Falsified by a driver that logs each job that ends as failed: the refused job has one line.
#[test]
fn a_refused_job_on_a_device_thread_logs_no_failure_and_a_failed_job_logs_one() {
    let failed = failure_lines_on_a_device_thread(Ran::Failed);
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert!(
        failed[0].contains("copy quantum") && failed[0].contains("engine left"),
        "{}",
        failed[0]
    );
    let refused = failure_lines_on_a_device_thread(Ran::Refused);
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert!(refused[0].starts_with("job refused: copy quantum"), "{}", refused[0]);
    assert!(
        refused[0].contains("engine left") && !refused[0].ends_with("level=error"),
        "{}",
        refused[0]
    );
}
