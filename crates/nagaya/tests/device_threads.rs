//! A group of two devices with no engine with stages loaded (spec 3.2, "One device thread per device", Decided
//! 2026-10-28). `chat` with stages spans both cards, so the cards are one group. `left` on card 0 and `right` on
//! card 1 each open on the device thread of their card, and their quanta run there at once.

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

fn fake(name: &str) -> Box<dyn Engine + Send> {
    match name {
        "chat" => Box::new(feed_support::engine(2)),
        "left" | "right" => Box::new(FakeAlways::new()),
        other => panic!("no fake named {other}"),
    }
}

/// The factory of the fakes. It writes each open with the name of the thread that opens.
struct Fakes(Log);

impl Factory for Fakes {
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

fn engine(name: &str, devices: Vec<u32>, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("name".to_owned(), toml::Value::String(name.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: "fakes".to_owned(),
        devices,
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
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
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("fakes", Box::new(Fakes(Arc::clone(&log))));
    let budget = |tier: Tier, bytes| BudgetSpec {
        tiers: vec![TierSpec::from(tier)],
        bytes,
    };
    let profile = PlacementProfile {
        engines: vec![
            engine("chat", vec![0, 1], residency[CHAT]),
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
    let mut process = Start::new(profile, registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    let names = opened.into_iter().map(|engine| engine.name).collect();
    (process, log, names)
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
    assert_eq!(
        load(&process, "left"),
        (vec!["chat".to_owned()], vec!["left".to_owned()])
    );
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
    // An unload of `left` loads the resident `chat` again, on the group thread.
    process.unload("left").unwrap();
    assert_eq!(swapped(&process), (vec!["left".to_owned()], vec!["chat".to_owned()]));
    assert_eq!(log_of(&log)[4..], ["close:left", "open:chat@nagaya-group-0"]);
}
