//! Explicit loads (spec 3.3, 3.3a, 3.3b): a request never starts a load. A job for an engine that is not `ready` ends
//! at once as unavailable. A load or an unload of the process starts a swap on the group thread.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::notify::EngineState;
use nagaya::process::{DeviceReply, Event, LoadError, LoadReply, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Kind, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal,
    Segment, Slots, Tap, Tier,
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

/// The limits of an engine whose weights take `bytes` of the config, or 600 bytes.
fn counted_limits(config: &toml::Table) -> Limits {
    let bytes = config.get("bytes").and_then(toml::Value::as_integer).unwrap_or(600) as u64;
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes,
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

/// A resident engine `chat` and an on-demand engine `voice` on one device. With 1000 device bytes they do not fit
/// together, so a load of one unloads the other.
fn start(directory: &std::path::Path) -> (Process, Log) {
    start_sized(directory, 600)
}

/// As [`start`], with weights of `voice_bytes` for `voice`.
fn start_sized(directory: &std::path::Path, voice_bytes: i64) -> (Process, Log) {
    start_profile(directory, profile_sized(voice_bytes))
}

/// The profile of [`start_sized`].
fn profile_sized(voice_bytes: i64) -> PlacementProfile {
    let mut voice = engine("voice", Residency::OnDemand);
    voice
        .config
        .insert("bytes".to_owned(), toml::Value::Integer(voice_bytes));
    PlacementProfile {
        engines: vec![engine("chat", Residency::Resident), voice],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1000,
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

/// Start a process of `profile` on one device, and wait for its opens.
fn start_profile(directory: &std::path::Path, profile: PlacementProfile) -> (Process, Log) {
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
    let mut process = Start::new(profile, registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, log)
}

/// A job that writes `run:<engine>` to the log, or `unavailable:<engine>:<state>` when its engine is not ready.
///
/// A job with a gate holds its engine until the gate opens. It runs one short quantum after another, so the group
/// thread still takes messages.
struct Probe {
    name: &'static str,
    log: Log,
    gate: Option<Receiver<()>>,
    started: bool,
    done: Sender<String>,
}

impl JobBody for Probe {
    fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        if !self.started {
            self.started = true;
            self.log.lock().unwrap().push(format!("run:{}", self.name));
        }
        if let Some(gate) = &self.gate
            && gate.try_recv().is_err()
        {
            std::thread::sleep(Duration::from_millis(1));
            return Ran::More(vec![Kind::Read]);
        }
        let _ = self.done.send("ran".to_owned());
        Ran::Done
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        let _ = self.done.send(format!("unavailable:{engine}:{}", state.as_str()));
    }
}

const CHAT: usize = 0;
const VOICE: usize = 1;

fn probe(engine: usize, log: &Log, gate: Option<Receiver<()>>) -> (JobSpec, Receiver<String>) {
    let (done, finished) = mpsc::channel();
    let name = if engine == CHAT { "chat" } else { "voice" };
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine,
        body: Box::new(Probe {
            name,
            log: Arc::clone(log),
            gate,
            started: false,
            done,
        }),
    };
    (spec, finished)
}

fn log_of(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// The next `Swapped` event of the process.
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

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// Run a chat job that holds its engine until the gate opens, and wait until it runs.
fn hold_chat(process: &Process, log: &Log) -> (Sender<()>, Receiver<String>) {
    let (gate_tx, gate_rx) = mpsc::channel();
    let (job, finished) = probe(CHAT, log, Some(gate_rx));
    process.submit_job(0, job).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log_of(log).contains(&"run:chat".to_owned()) {
        assert!(Instant::now() < deadline, "the chat job did not start");
        std::thread::sleep(Duration::from_millis(5));
    }
    (gate_tx, finished)
}

#[test]
fn a_job_for_an_unloaded_engine_ends_as_unavailable_and_loads_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let (job, finished) = probe(VOICE, &log, None);
    process.submit_job(0, job).unwrap();
    assert_eq!(
        finished.recv_timeout(Duration::from_secs(5)).unwrap(),
        "unavailable:voice:unloaded"
    );
    std::thread::sleep(Duration::from_millis(100));
    assert!(process.recv(Duration::from_millis(10)).is_none(), "no swap came");
    assert_eq!(log_of(&log), ["open:chat"]);
}

#[test]
fn the_ports_give_the_state_of_each_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let ports = process.ports();
    assert_eq!(ports.engine_state("chat"), Some(EngineState::Ready));
    assert_eq!(ports.engine_state("voice"), Some(EngineState::Unloaded));
    assert_eq!(ports.engine_state("nobody"), None);
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    swapped(&process);
    assert_eq!(ports.engine_state("chat"), Some(EngineState::Unloaded));
    assert_eq!(ports.engine_state("voice"), Some(EngineState::Ready));
}

#[test]
fn a_load_swaps_the_engine_in_and_its_jobs_run() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    assert_eq!(
        process.load_with("voice", &["chat".to_owned()]).unwrap(),
        LoadReply::Started {
            unload: names(&["chat"]),
            load: names(&["voice"]),
        }
    );
    assert_eq!(swapped(&process), (names(&["chat"]), names(&["voice"])));
    let (job, finished) = probe(VOICE, &log, None);
    process.submit_job(0, job).unwrap();
    assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "ran");
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "run:voice"]);
}

#[test]
fn a_load_of_a_ready_engine_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    assert_eq!(process.load("chat").unwrap(), LoadReply::Already(EngineState::Ready));
    std::thread::sleep(Duration::from_millis(100));
    assert!(process.recv(Duration::from_millis(10)).is_none(), "no swap came");
    assert_eq!(log_of(&log), ["open:chat"]);
}

#[test]
fn a_load_or_an_unload_of_an_unknown_engine_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    assert_eq!(process.load("nobody"), Err(LoadError::Unknown("nobody".to_owned())));
    assert_eq!(process.unload("nobody"), Err(LoadError::Unknown("nobody".to_owned())));
}

/// A resident engine that is not pinned unloads by the route, so an agent can make room (spec 3.3b). No other engine
/// loads.
#[test]
fn an_unload_of_a_resident_engine_unloads_it_and_loads_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    assert_eq!(
        process.unload("chat").unwrap(),
        LoadReply::Started {
            unload: names(&["chat"]),
            load: Vec::new(),
        }
    );
    assert_eq!(swapped(&process), (names(&["chat"]), Vec::new()));
    assert_eq!(log_of(&log), ["open:chat", "close:chat"]);
    assert_eq!(process.ports().engine_state("chat"), Some(EngineState::Unloaded));
}

/// A load that does not fit beside the loaded engines is a refusal that names the shortfall and the engines in the
/// way, and nothing changes (spec 3.3a, "Explicit layout").
#[test]
fn a_load_that_does_not_fit_names_the_engine_in_the_way_and_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let error = process.load("voice").unwrap_err();
    assert!(
        matches!(&error, LoadError::InTheWay { engine, in_the_way, pinned, shortfall }
            if engine == "voice" && in_the_way == &names(&["chat"]) && pinned.is_empty()
                && shortfall.contains("bytes")),
        "{error:?}"
    );
    std::thread::sleep(Duration::from_millis(100));
    assert!(process.recv(Duration::from_millis(10)).is_none(), "no swap came");
    assert_eq!(log_of(&log), ["open:chat"]);
    assert_eq!(process.ports().engine_state("chat"), Some(EngineState::Ready));
}

/// An `unload` list must name engines of the group of the load, and not the engine itself.
#[test]
fn a_load_that_names_itself_or_an_unknown_engine_in_unload_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    assert!(matches!(
        process.load_with("voice", &names(&["voice"])),
        Err(LoadError::BadUnload(_))
    ));
    assert_eq!(
        process.load_with("voice", &names(&["nobody"])),
        Err(LoadError::Unknown("nobody".to_owned()))
    );
    assert_eq!(log_of(&log), ["open:chat"]);
}

#[test]
fn an_unload_of_an_unloaded_engine_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    assert_eq!(
        process.unload("voice").unwrap(),
        LoadReply::Already(EngineState::Unloaded)
    );
}

/// No unload loads another engine (spec 3.3a, "Explicit layout"). The client loads the resident engine again.
#[test]
fn an_unload_of_an_on_demand_engine_loads_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    swapped(&process);
    assert_eq!(
        process.unload("voice").unwrap(),
        LoadReply::Started {
            unload: names(&["voice"]),
            load: Vec::new(),
        }
    );
    assert_eq!(swapped(&process), (names(&["voice"]), Vec::new()));
    assert_eq!(process.ports().engine_state("chat"), Some(EngineState::Unloaded));
    assert_eq!(
        process.load("chat").unwrap(),
        LoadReply::Started {
            unload: Vec::new(),
            load: names(&["chat"]),
        }
    );
    assert_eq!(swapped(&process), (Vec::new(), names(&["chat"])));
    assert_eq!(
        log_of(&log),
        ["open:chat", "close:chat", "open:voice", "close:voice", "open:chat"]
    );
}

#[test]
fn a_job_for_an_engine_that_a_planned_swap_unloads_ends_as_unavailable() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let (gate, held) = hold_chat(&process, &log);
    assert_eq!(
        process.load_with("voice", &["chat".to_owned()]).unwrap(),
        LoadReply::Started {
            unload: names(&["chat"]),
            load: names(&["voice"]),
        }
    );
    assert_eq!(process.ports().engine_state("chat"), Some(EngineState::Draining));
    assert_eq!(process.ports().engine_state("voice"), Some(EngineState::Loading));
    let (late, late_finished) = probe(CHAT, &log, None);
    process.submit_job(0, late).unwrap();
    assert_eq!(
        late_finished.recv_timeout(Duration::from_secs(5)).unwrap(),
        "unavailable:chat:draining"
    );
    let (early, early_finished) = probe(VOICE, &log, None);
    process.submit_job(0, early).unwrap();
    assert_eq!(
        early_finished.recv_timeout(Duration::from_secs(5)).unwrap(),
        "unavailable:voice:loading"
    );
    // The held job runs to its end, and then the swap runs.
    gate.send(()).unwrap();
    assert_eq!(held.recv_timeout(Duration::from_secs(5)).unwrap(), "ran");
    assert_eq!(swapped(&process), (names(&["chat"]), names(&["voice"])));
    assert_eq!(log_of(&log), ["open:chat", "run:chat", "close:chat", "open:voice"]);
}

/// The ledger takes a swap when the group plans it, before any unload (spec 3.3a). A charge between the plan and the
/// swap is checked against the engines after the swap, so it cannot make the swap fail after its unload. Before, the
/// swap asked the ledger after the unload, and a refusal there stopped the process.
#[test]
fn a_charge_while_a_swap_waits_cannot_refuse_the_swap_after_its_unload() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_sized(directory.path(), 700);
    let (gate, held) = hold_chat(&process, &log);
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    // 600 bytes of chat and 350 of staging fit in 1000 now, and 700 bytes of voice and 350 of staging do not.
    let ports = process.ports();
    assert!(
        ports.admit_staging("chat", Tier::Device(0), 350).is_err(),
        "the ledger counts the planned swap, so the staging does not fit"
    );
    gate.send(()).unwrap();
    assert_eq!(held.recv_timeout(Duration::from_secs(5)).unwrap(), "ran");
    assert_eq!(swapped(&process), (names(&["chat"]), names(&["voice"])));
    assert_eq!(ports.engine_state("voice"), Some(EngineState::Ready));
}

/// The engine that unloads keeps its bytes in the ledger until its drop, so a charge while it drains must fit beside
/// it too (spec 3.3a). voice takes 300 bytes and chat 600, in 1000. Staging of 450 bytes fits beside voice, and not
/// beside chat, so it is refused until chat drops.
#[test]
fn an_engine_that_drains_keeps_its_bytes_until_its_drop() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_sized(directory.path(), 300);
    let (gate, held) = hold_chat(&process, &log);
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    let ports = process.ports();
    assert!(
        ports.admit_staging("chat", Tier::Device(0), 450).is_err(),
        "chat still holds 600 bytes, so 450 bytes of staging do not fit"
    );
    ports
        .admit_staging("chat", Tier::Device(0), 350)
        .expect("350 bytes fit beside chat and beside voice");
    gate.send(()).unwrap();
    assert_eq!(held.recv_timeout(Duration::from_secs(5)).unwrap(), "ran");
    assert_eq!(swapped(&process), (names(&["chat"]), names(&["voice"])));
    ports.release_staging(Tier::Device(0), 350).unwrap();
    ports
        .admit_staging("voice", Tier::Device(0), 450)
        .expect("after the drop of chat, 450 bytes fit beside voice");
}

/// A retire is not done in the ledger until the engine drops (spec 3.3). A profile change waits for that before the
/// ledger takes the new profile, so a context that the engine still holds is released first.
#[test]
fn a_retired_engine_stays_loaded_in_the_ledger_until_its_drop() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let (gate, held) = hold_chat(&process, &log);
    let ports = process.ports();
    ports.retire("chat").unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        ports.loaded_engines().unwrap().contains("chat"),
        "chat holds a job, so it has not dropped"
    );
    gate.send(()).unwrap();
    assert_eq!(held.recv_timeout(Duration::from_secs(5)).unwrap(), "ran");
    assert_eq!(swapped(&process), (names(&["chat"]), Vec::new()));
    assert!(!ports.loaded_engines().unwrap().contains("chat"));
}

#[test]
fn a_load_during_a_planned_swap_waits_for_that_swap() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let (gate, held) = hold_chat(&process, &log);
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    let ports = process.ports();
    let unloader = std::thread::spawn(move || ports.unload("voice"));
    std::thread::sleep(Duration::from_millis(100));
    assert!(!unloader.is_finished(), "the unload waits for the planned swap");
    gate.send(()).unwrap();
    held.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        unloader.join().unwrap().unwrap(),
        LoadReply::Started {
            unload: names(&["voice"]),
            load: Vec::new(),
        }
    );
    assert_eq!(swapped(&process), (names(&["chat"]), names(&["voice"])));
    assert_eq!(swapped(&process), (names(&["voice"]), Vec::new()));
}

/// A release unloads each loaded engine of the device, and the device takes no load until a reclaim (spec 3.3b, "The
/// release of a device"). A reclaim loads the resident engines, and it loads no engine on demand.
///
/// Falsified by a release that keeps no mark: the load of `voice` starts. Falsified by a reclaim that loads each
/// engine of the device: `voice` loads too.
#[test]
fn a_release_unloads_the_engines_of_a_device_and_a_reclaim_loads_the_resident_ones() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_sized(directory.path(), 100);
    let profile = profile_sized(100);
    assert_eq!(
        process.release_device(0, &profile),
        Ok(DeviceReply::Started(names(&["chat"])))
    );
    assert_eq!(swapped(&process), (names(&["chat"]), Vec::new()));
    let ports = process.ports();
    assert_eq!(ports.engine_state("chat"), Some(EngineState::Unloaded));
    for name in ["voice", "chat"] {
        assert_eq!(
            process.load(name),
            Err(LoadError::Released {
                engine: name.to_owned(),
                device: 0
            }),
            "{name}"
        );
    }
    // A second release has nothing to unload.
    assert_eq!(process.release_device(0, &profile), Ok(DeviceReply::Already));
    assert_eq!(
        process.reclaim_device(0, &profile),
        Ok(DeviceReply::Started(names(&["chat"])))
    );
    assert_eq!(swapped(&process), (Vec::new(), names(&["chat"])));
    assert_eq!(ports.engine_state("chat"), Some(EngineState::Ready));
    assert_eq!(ports.engine_state("voice"), Some(EngineState::Unloaded));
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:chat"]);
    // The device takes a load again, and a second reclaim changes nothing.
    assert_eq!(process.reclaim_device(0, &profile), Ok(DeviceReply::Already));
    process.load("voice").unwrap();
    assert_eq!(swapped(&process), (Vec::new(), names(&["voice"])));
}

/// A release unloads an engine on demand that is loaded, and a reclaim does not load it again.
#[test]
fn a_release_unloads_a_loaded_engine_on_demand_and_a_reclaim_leaves_it_unloaded() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start_sized(directory.path(), 100);
    let profile = profile_sized(100);
    process.load("voice").unwrap();
    swapped(&process);
    assert_eq!(
        process.release_device(0, &profile),
        Ok(DeviceReply::Started(names(&["chat", "voice"])))
    );
    swapped(&process);
    swapped(&process);
    assert_eq!(
        process.reclaim_device(0, &profile),
        Ok(DeviceReply::Started(names(&["chat"])))
    );
    swapped(&process);
    assert_eq!(process.ports().engine_state("voice"), Some(EngineState::Unloaded));
}

/// A pinned engine leaves only by a profile change, so a release refuses its device, and nothing changes.
///
/// Falsified by a release that unloads first: `voice` is unloaded when the refusal comes.
#[test]
fn a_release_refuses_a_device_with_a_loaded_pinned_engine_and_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    // The pinned engine is the last of the profile, so a release that unloads in order meets `voice` first.
    let mut profile = profile_sized(100);
    profile.engines.swap(0, 1);
    profile.engines[1].pinned = true;
    let (process, _log) = start_profile(directory.path(), profile.clone());
    process.load("voice").unwrap();
    swapped(&process);
    assert_eq!(
        process.release_device(0, &profile),
        Err(LoadError::Pinned("chat".to_owned()))
    );
    let ports = process.ports();
    assert_eq!(ports.engine_state("chat"), Some(EngineState::Ready));
    assert_eq!(ports.engine_state("voice"), Some(EngineState::Ready));
    // The device is not released: a load of a loaded engine is a 200 as before.
    assert!(matches!(process.load("voice"), Ok(LoadReply::Already(_))));
}

/// A device that no engine of the profile names has nothing to release.
#[test]
fn a_release_or_a_reclaim_of_a_device_that_no_engine_names_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let profile = profile_sized(600);
    assert_eq!(process.release_device(7, &profile), Err(LoadError::NoDevice(7)));
    assert_eq!(process.reclaim_device(7, &profile), Err(LoadError::NoDevice(7)));
}

/// A reclaim whose load is refused leaves the device released, so the next reclaim tries again and does not answer
/// that nothing is left to do (kaibo review of 2026-11-15).
///
/// Falsified by a reclaim that ends the release before its loads: the second reclaim is `Already`.
#[test]
fn a_reclaim_whose_load_is_refused_keeps_the_device_released() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let profile = profile_sized(600);
    assert_eq!(
        process.release_device(0, &profile),
        Ok(DeviceReply::Started(names(&["chat"])))
    );
    swapped(&process);
    // A profile in which `voice` is resident too. The two engines do not fit the device together.
    let mut both = profile.clone();
    both.engines[1].residency = Residency::Resident;
    for attempt in 0..2 {
        let refused = process.reclaim_device(0, &both);
        assert!(
            matches!(&refused, Err(LoadError::InTheWay { engine, .. }) if engine == "voice"),
            "attempt {attempt}: {refused:?}"
        );
        if attempt == 0 {
            swapped(&process);
        }
    }
    // The reclaim by the active profile ends the release: `chat` is loaded, and `voice` stays on demand.
    assert_eq!(process.reclaim_device(0, &profile), Ok(DeviceReply::Already));
    assert!(matches!(process.load("chat"), Ok(LoadReply::Already(_))));
}
