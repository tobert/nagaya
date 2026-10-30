//! The profile change of the main loop (spec 3.3): each `PUT` gets an answer, the profile persists, and a change in
//! place runs on its own thread.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::notify::EngineState;
use nagaya::process::{Event, LoadError, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::reprofile::{Action, Changer, ProfileCall, PutAnswer};
use nagaya::swap::Swap;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Kind, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal,
    Segment, Slots, Tap, Tier,
};
use nagaya_sched::Class;
use std::sync::mpsc::{self, Receiver, Sender};

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

fn counted_limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes: 600,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    limits
}

struct CountedFactory(Log);

impl Factory for CountedFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(counted_limits())
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let name = config["name"].as_str().unwrap().to_owned();
        self.0.lock().unwrap().push(format!("open:{name}"));
        Ok(Box::new(Counted {
            inner: FakeAlways::new(),
            name,
            limits: counted_limits(),
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
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

/// A resident engine `chat` and an on-demand engine `voice` on one device of `device_bytes`. With 1000 bytes they do not
/// fit together; with 2000 they do.
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

fn start(directory: &std::path::Path) -> (Process, Log) {
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
    let mut process = Start::new(profile(1000), registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, log)
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
    panic!("no swap came in 5 s");
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// A main loop of a test: the process, its swap state, the changer, and the sender of the routes.
struct Main {
    process: Process,
    swap: Swap,
    changer: Changer,
    calls: Sender<ProfileCall>,
    log: Log,
    directory: tempfile::TempDir,
}

impl Main {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let (process, log) = start(directory.path());
        profile(1000).write(directory.path()).unwrap();
        let swap = Swap::open(directory.path(), "host", profile(1000)).unwrap();
        let (changer, calls) = Changer::new();
        Self {
            process,
            swap,
            changer,
            calls,
            log,
            directory,
        }
    }

    /// Send a `PUT` and run one tick of the main loop. Give the answer, and true when the process must restart.
    fn put(&mut self, profile: &PlacementProfile) -> (PutAnswer, bool) {
        let (reply, answer) = mpsc::channel();
        self.calls.send(ProfileCall::Put(profile.clone(), reply)).unwrap();
        let restart = self.changer.tick(&mut self.swap, &self.process).unwrap();
        (answer.recv_timeout(Duration::from_secs(5)).unwrap(), restart)
    }

    fn get(&mut self) -> PlacementProfile {
        let (reply, answer) = mpsc::channel();
        self.calls.send(ProfileCall::Get(reply)).unwrap();
        self.changer.tick(&mut self.swap, &self.process).unwrap();
        answer.recv_timeout(Duration::from_secs(5)).unwrap()
    }

    fn persisted(&self) -> PlacementProfile {
        PlacementProfile::read_active(self.directory.path()).unwrap()
    }

    /// Wait until the change in place ends.
    fn applied(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.changer.applying() {
            assert!(Instant::now() < deadline, "the change in place did not end in 5 s");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

fn actions(answer: &PutAnswer) -> Vec<(String, Action)> {
    match answer {
        PutAnswer::Unchanged(change) | PutAnswer::Accepted(change) => change
            .actions
            .iter()
            .map(|entry| (entry.engine.clone(), entry.action))
            .collect(),
        other => panic!("expected actions, got {other:?}"),
    }
}

fn pairs(list: &[(&str, Action)]) -> Vec<(String, Action)> {
    list.iter()
        .map(|(name, action)| ((*name).to_owned(), *action))
        .collect()
}

#[test]
fn an_equal_profile_is_unchanged() {
    let mut main = Main::new();
    let (answer, restart) = main.put(&profile(1000));
    assert!(matches!(answer, PutAnswer::Unchanged(_)), "{answer:?}");
    assert_eq!(
        actions(&answer),
        pairs(&[("chat", Action::Keep), ("voice", Action::Keep)])
    );
    assert!(!restart);
}

#[test]
fn a_profile_that_does_not_fit_is_refused_with_its_shortfall_and_nothing_persists() {
    let mut main = Main::new();
    let mut both = profile(1000);
    both.engines[1].residency = Residency::Resident;
    let (answer, restart) = main.put(&both);
    match answer {
        PutAnswer::Refused { shortfall, .. } => {
            assert_eq!(shortfall.len(), 1);
            assert_eq!(shortfall[0].bytes, 200);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(!restart);
    assert_eq!(main.persisted(), profile(1000));
    assert_eq!(main.get(), profile(1000));
}

#[test]
fn a_profile_that_breaks_a_rule_is_invalid() {
    let mut main = Main::new();
    let mut bad = profile(1000);
    // A pin on an on-demand engine breaks a rule of 3.3a.
    bad.engines[1].pinned = true;
    let (answer, _) = main.put(&bad);
    assert!(matches!(answer, PutAnswer::Invalid(_)), "{answer:?}");
    let mut storeless = profile(1000);
    storeless.store = None;
    let (answer, _) = main.put(&storeless);
    assert!(matches!(answer, PutAnswer::Invalid(_)), "{answer:?}");
    assert_eq!(main.persisted(), profile(1000));
}

#[test]
fn a_new_engine_persists_and_restarts() {
    let mut main = Main::new();
    let mut new = profile(1000);
    let mut paint = new.engines[1].clone();
    paint.name = "paint".to_owned();
    new.engines.push(paint);
    let (answer, restart) = main.put(&new);
    assert!(
        matches!(&answer, PutAnswer::Accepted(change) if change.restart),
        "{answer:?}"
    );
    assert!(restart, "the main loop stops with exit 4");
    assert_eq!(main.persisted(), new);
}

#[test]
fn a_gone_engine_unloads_in_place_and_the_ledger_takes_the_profile() {
    let mut main = Main::new();
    main.process.load_with("voice", &["chat".to_owned()]).unwrap();
    swapped(&main.process);
    let mut new = profile(2000);
    new.engines.truncate(1);
    let (answer, restart) = main.put(&new);
    assert_eq!(
        actions(&answer),
        pairs(&[("chat", Action::Keep), ("voice", Action::Unload)])
    );
    assert!(!restart);
    // The retire only unloads. The place of `chat`, resident and not loaded, then loads it beside nothing (spec 3.3).
    assert_eq!(swapped(&main.process), (names(&["voice"]), Vec::new()));
    assert_eq!(swapped(&main.process), (Vec::new(), names(&["chat"])));
    main.applied();
    assert_eq!(main.persisted(), new);
    assert_eq!(main.get(), new);
    assert_eq!(main.process.load("voice"), Err(LoadError::Unknown("voice".to_owned())));
    let device = main
        .process
        .ports()
        .unassigned()
        .into_iter()
        .find(|(tiers, _)| tiers == &vec![Tier::Device(0)])
        .unwrap()
        .1;
    assert_eq!(device, 1400, "the new budget of 2000 bytes, with `chat` loaded");
    assert_eq!(
        log_of(&main.log),
        ["open:chat", "close:chat", "open:voice", "close:voice", "open:chat"]
    );
}

#[test]
fn an_engine_that_becomes_resident_loads_in_place() {
    let mut main = Main::new();
    let mut new = profile(2000);
    new.engines[1].residency = Residency::Resident;
    new.engines[1].pinned = true;
    let (answer, restart) = main.put(&new);
    assert_eq!(
        actions(&answer),
        pairs(&[("chat", Action::Keep), ("voice", Action::Residency)])
    );
    assert!(!restart);
    assert_eq!(swapped(&main.process), (Vec::new(), names(&["voice"])));
    main.applied();
    assert_eq!(main.process.unload("voice"), Err(LoadError::Pinned("voice".to_owned())));
}

/// A job that holds its engine until the gate opens. It runs one short quantum after another.
struct Hold(Receiver<()>);

impl JobBody for Hold {
    fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        if self.0.try_recv().is_ok() {
            return Ran::Done;
        }
        std::thread::sleep(Duration::from_millis(1));
        Ran::More(vec![Kind::Read])
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }
}

#[test]
fn a_put_while_a_change_in_place_runs_is_busy() {
    let mut main = Main::new();
    let (gate, held) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine: 0,
        body: Box::new(Hold(held)),
    };
    main.process.submit_job(0, spec).unwrap();
    // `chat` holds a job, so its retire waits, and the change runs on.
    let mut without_chat = profile(1000);
    without_chat.engines.remove(0);
    let (answer, _) = main.put(&without_chat);
    assert!(matches!(answer, PutAnswer::Accepted(_)), "{answer:?}");
    assert!(main.changer.applying());
    let (answer, _) = main.put(&profile(1000));
    assert!(matches!(answer, PutAnswer::Busy), "{answer:?}");
    gate.send(()).unwrap();
    main.applied();
    assert_eq!(main.persisted(), without_chat);
}

/// Submit a job of `chat` that holds it until the gate gets a message.
fn hold_chat(main: &Main) -> Sender<()> {
    let (gate, held) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine: 0,
        body: Box::new(Hold(held)),
    };
    main.process.submit_job(0, spec).unwrap();
    gate
}

/// A PUT while a swap waits is a 409 busy, and nothing changes (spec 3.3a, Decided 2026-10-30). A budget checked
/// beside the waiting swap could refuse the engines after it.
#[test]
fn a_put_while_a_swap_waits_is_busy() {
    let mut main = Main::new();
    let gate = hold_chat(&main);
    main.process.load_with("voice", &["chat".to_owned()]).unwrap();
    let (answer, _) = main.put(&profile(2000));
    assert!(matches!(answer, PutAnswer::Busy), "{answer:?}");
    assert_eq!(main.persisted(), profile(1000), "nothing changed");
    gate.send(()).unwrap();
    assert_eq!(swapped(&main.process), (names(&["chat"]), names(&["voice"])));
    let (answer, _) = main.put(&profile(2000));
    assert!(matches!(answer, PutAnswer::Accepted(_)), "{answer:?}");
    main.applied();
}

/// After a change is accepted, a load is busy until the ledger takes the new profile (spec 3.3a).
#[test]
fn a_load_while_a_change_waits_for_the_ledger_is_busy() {
    let mut main = Main::new();
    let gate = hold_chat(&main);
    let mut without_chat = profile(2000);
    without_chat.engines.remove(0);
    let (answer, _) = main.put(&without_chat);
    assert!(matches!(answer, PutAnswer::Accepted(_)), "{answer:?}");
    assert!(
        matches!(main.process.load("voice"), Err(LoadError::Busy(_))),
        "a load during the change"
    );
    gate.send(()).unwrap();
    main.applied();
    assert_eq!(main.persisted(), without_chat);
    assert!(main.process.load("voice").is_ok(), "a load after the change");
}

/// The ledger takes the new profile only when the engines that it removes hold no context. A council request can end
/// its context after the drop of its engine, and the ledger refused the profile then, which stopped the process.
#[test]
fn a_change_waits_for_the_contexts_of_a_removed_engine() {
    let mut main = Main::new();
    let ports = main.process.ports();
    ports.admit_context("chat").unwrap();
    let mut without_chat = profile(1000);
    without_chat.engines.remove(0);
    let (answer, _) = main.put(&without_chat);
    assert!(matches!(answer, PutAnswer::Accepted(_)), "{answer:?}");
    assert_eq!(swapped(&main.process), (names(&["chat"]), Vec::new()));
    std::thread::sleep(Duration::from_millis(100));
    assert!(main.changer.applying(), "the change waits for the context of chat");
    ports.release_context("chat").unwrap();
    main.applied();
    assert_eq!(main.persisted(), without_chat);
    assert!(ports.loaded_engines().is_some(), "the process runs on");
}

/// Start a change that removes `chat` and lowers the budget to 800, and keep it waiting on a context of `chat`.
/// `voice` (600 bytes) is loaded, and `chat` is not.
fn waiting_change(main: &mut Main) -> nagaya::process::Ports {
    main.process.load_with("voice", &["chat".to_owned()]).unwrap();
    swapped(&main.process);
    let ports = main.process.ports();
    ports.admit_context("chat").unwrap();
    let mut new = profile(800);
    new.engines.remove(0);
    let (answer, _) = main.put(&new);
    assert!(matches!(answer, PutAnswer::Accepted(_)), "{answer:?}");
    std::thread::sleep(Duration::from_millis(50));
    assert!(main.changer.applying(), "the change waits for the context of chat");
    ports
}

/// While a change waits, a charge must also fit the new profile (spec 3.3a). 300 bytes of staging fit beside `voice`
/// in 1000 bytes, not in 800, so they are refused, and the change ends when the context of `chat` ends. Before, the
/// staging was admitted, and the ledger then refused the new profile, which stopped the process.
#[test]
fn a_charge_during_a_change_must_fit_the_new_profile() {
    let mut main = Main::new();
    let ports = waiting_change(&mut main);
    assert!(
        ports.admit_staging("voice", Tier::Device(0), 300).is_err(),
        "300 bytes beside voice do not fit the new budget of 800"
    );
    ports.admit_staging("voice", Tier::Device(0), 100).unwrap();
    ports.release_context("chat").unwrap();
    main.applied();
    assert!(ports.failure().is_none(), "{:?}", ports.failure());
    ports.release_staging(Tier::Device(0), 100).unwrap();
}

/// While a change waits, a context of an engine that it removes is refused as not ready, so steady traffic to that
/// engine cannot hold the change off (spec 3.3a).
#[test]
fn a_context_of_a_removed_engine_is_refused_during_a_change() {
    let mut main = Main::new();
    let ports = waiting_change(&mut main);
    assert_eq!(
        ports.admit_context("chat"),
        Err(nagaya::process::ContextChargeError::Leaving("chat".to_owned()))
    );
    ports.release_context("chat").unwrap();
    main.applied();
    assert!(ports.failure().is_none(), "{:?}", ports.failure());
}

/// A PUT while a swap that only unloads drains is busy too (spec 3.3a, Decided 2026-10-30).
#[test]
fn a_put_during_an_unload_drain_is_busy() {
    let mut main = Main::new();
    let gate = hold_chat(&main);
    main.process.unload("chat").unwrap();
    let (answer, _) = main.put(&profile(2000));
    assert!(matches!(answer, PutAnswer::Busy), "{answer:?}");
    gate.send(()).unwrap();
    assert_eq!(swapped(&main.process), (names(&["chat"]), Vec::new()));
}

/// A change whose profile cannot be kept ends, so a later load is not busy.
#[test]
fn a_change_that_cannot_be_kept_ends() {
    use std::os::unix::fs::PermissionsExt;
    let mut main = Main::new();
    let path = main.directory.path().to_owned();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555)).unwrap();
    let (reply, _answer) = mpsc::channel();
    main.calls.send(ProfileCall::Put(profile(2000), reply)).unwrap();
    let kept = main.changer.tick(&mut main.swap, &main.process);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(kept.is_err(), "a read-only store keeps no profile");
    assert!(
        !matches!(main.process.load("voice"), Err(LoadError::Busy(_))),
        "the change ended"
    );
}

/// A change that waits names what it waits for in the log, once at the start and then every 10 s (spec 9.1). The
/// change waits about 100 ms here, so the log holds one line of the wait.
#[test]
fn a_change_that_waits_names_the_wait_once_in_the_log() {
    let mut main = Main::new();
    let ports = main.process.ports();
    ports.admit_context("chat").unwrap();
    let mut without_chat = profile(1000);
    without_chat.engines.remove(0);
    let lines = nagaya_telemetry::testing::log_lines(|| {
        let (answer, _) = main.put(&without_chat);
        assert!(matches!(answer, PutAnswer::Accepted(_)), "{answer:?}");
        assert_eq!(swapped(&main.process), (names(&["chat"]), Vec::new()));
        std::thread::sleep(Duration::from_millis(100));
        assert!(main.changer.applying(), "the change waits for the context of chat");
        ports.release_context("chat").unwrap();
        main.applied();
    });
    let waits: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("the profile change waits for"))
        .collect();
    assert_eq!(waits.len(), 1, "{lines:#?}");
    assert!(waits[0].contains("before the ledger takes the profile"), "{lines:#?}");
    assert!(
        lines
            .iter()
            .any(|line| line.contains("the profile change in place ended")),
        "{lines:#?}"
    );
}
