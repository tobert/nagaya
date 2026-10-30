//! A profile change in place on a running process (spec 3.3): a check of the whole profile, the ledger of the new
//! profile, a retired engine, and a new residency.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::AdmitError;
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::notify::EngineState;
use nagaya::process::{Event, LoadError, LoadReply, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal, Segment,
    Slots, Tap, Tier,
};

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

fn device_bytes(process: &Process) -> u64 {
    process
        .ports()
        .unassigned()
        .into_iter()
        .find(|(tiers, _)| tiers == &vec![Tier::Device(0)])
        .unwrap()
        .1
}

#[test]
fn a_check_of_a_profile_that_does_not_fit_refuses_it_and_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let mut both = profile(1000);
    both.engines[1].residency = Residency::Resident;
    match process.check_profile(&both) {
        Err(AdmitError::Ledger(nagaya_ledger::LedgerError::Refused(refusal))) => {
            assert_eq!(refusal.shortfall, vec![(vec![Tier::Device(0)], 200)]);
        }
        other => panic!("two resident engines of 600 bytes fit in 1000: {other:?}"),
    }
    assert!(process.check_profile(&profile(1000)).is_ok());
    assert_eq!(device_bytes(&process), 400, "`chat` still holds 600 of 1000 bytes");
}

#[test]
fn a_new_profile_admits_the_loaded_engines_with_its_budgets() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    process.ports().reprofile(&profile(2000)).unwrap();
    assert_eq!(device_bytes(&process), 1400, "`chat` holds 600 of 2000 bytes");
    // `voice` now fits beside `chat`, so its load unloads nothing.
    assert_eq!(
        process.load("voice").unwrap(),
        LoadReply::Started {
            unload: Vec::new(),
            load: names(&["voice"]),
        }
    );
}

#[test]
fn a_new_profile_without_an_engine_that_holds_a_context_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let ports = process.ports();
    let mut without = profile(1000);
    without.engines.truncate(1);
    // A context charge of `voice` is live, so the ledger keeps the engine (spec 4.2).
    ports.admit_context("voice").unwrap();
    assert!(ports.reprofile(&without).is_err());
    ports.release_context("voice").unwrap();
    ports.reprofile(&without).unwrap();
}

#[test]
fn a_retired_engine_unloads_and_takes_no_load() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    swapped(&process);
    let ports = process.ports();
    assert_eq!(
        ports.retire("voice").unwrap(),
        LoadReply::Started {
            unload: names(&["voice"]),
            load: Vec::new(),
        }
    );
    assert_eq!(swapped(&process), (names(&["voice"]), Vec::new()));
    assert_eq!(process.load("voice"), Err(LoadError::Unknown("voice".to_owned())));
    assert_eq!(
        ports.retire("voice").unwrap(),
        LoadReply::Already(EngineState::Unloaded)
    );
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "close:voice"]);
}

#[test]
fn a_retired_resident_engine_does_not_load_again() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let ports = process.ports();
    assert_eq!(
        ports.retire("chat").unwrap(),
        LoadReply::Started {
            unload: names(&["chat"]),
            load: Vec::new(),
        }
    );
    swapped(&process);
    // A load of `voice` and its unload leave the group with no engine: `chat` is retired.
    process.load("voice").unwrap();
    assert_eq!(swapped(&process), (Vec::new(), names(&["voice"])));
    process.unload("voice").unwrap();
    assert_eq!(swapped(&process), (names(&["voice"]), Vec::new()));
    assert_eq!(ports.engine_state("chat"), Some(EngineState::Unloaded));
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice", "close:voice"]);
}

#[test]
fn an_engine_that_becomes_resident_loads() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let ports = process.ports();
    ports.retire("chat").unwrap();
    swapped(&process);
    let mut voice = profile(1000).engines[1].clone();
    voice.residency = Residency::Resident;
    voice.pinned = true;
    assert_eq!(
        ports.place(&voice).unwrap(),
        LoadReply::Started {
            unload: Vec::new(),
            load: names(&["voice"]),
        }
    );
    assert_eq!(swapped(&process), (Vec::new(), names(&["voice"])));
    // A pinned engine leaves only by a profile change.
    assert_eq!(process.unload("voice"), Err(LoadError::Pinned("voice".to_owned())));
}

/// No idle time unloads an engine (spec 3.3a, "Explicit layout"): an engine that becomes on demand stays loaded until
/// an unload.
#[test]
fn an_engine_that_becomes_on_demand_stays_loaded_until_an_unload() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let mut chat = profile(1000).engines[0].clone();
    chat.residency = Residency::OnDemand;
    assert_eq!(
        process.ports().place(&chat).unwrap(),
        LoadReply::Already(EngineState::Ready)
    );
    let quiet = Instant::now() + Duration::from_millis(1500);
    while Instant::now() < quiet {
        if let Some(event @ Event::Swapped { .. }) = process.recv(Duration::from_millis(50)) {
            panic!("no swap may come without an unload: {event:?}");
        }
    }
    assert_eq!(process.ports().engine_state("chat"), Some(EngineState::Ready));
    assert_eq!(
        process.unload("chat").unwrap(),
        LoadReply::Started {
            unload: names(&["chat"]),
            load: Vec::new(),
        }
    );
    assert_eq!(swapped(&process), (names(&["chat"]), Vec::new()));
}

#[test]
fn a_place_or_a_retire_of_an_unknown_engine_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let ports = process.ports();
    assert_eq!(ports.retire("nobody"), Err(LoadError::Unknown("nobody".to_owned())));
    let mut nobody = profile(1000).engines[1].clone();
    nobody.name = "nobody".to_owned();
    assert_eq!(ports.place(&nobody), Err(LoadError::Unknown("nobody".to_owned())));
}

/// The change retires an engine before the ledger takes the new profile. A loaded engine that the profile lacks would
/// hold memory that the ledger does not count, so the ledger thread stops (spec 8.5).
#[test]
fn a_new_profile_without_a_loaded_engine_stops_the_ledger_thread() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let mut without = profile(1000);
    without.engines.truncate(1);
    without.engines[0].name = "voice".to_owned();
    let ports = process.ports();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ports.reprofile(&without)));
    assert!(result.is_err(), "the ledger thread gave no answer");
    match process.recv(Duration::from_secs(5)) {
        Some(Event::Panicked { thread, message }) => {
            assert_eq!(thread, "ledger");
            assert!(message.contains("engine `chat` is loaded"), "{message}");
        }
        other => panic!("expected the panic of the ledger thread, got {other:?}"),
    }
}

/// The check of a profile counts what the live ledger counts, so a change that fits an empty machine and not the
/// staging that runs now is refused before it is kept (spec 3.3). A refusal after the keep stopped the process.
#[test]
fn a_check_of_a_profile_counts_the_live_staging() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let ports = process.ports();
    let mut small = profile(800);
    small.engines.truncate(1);
    assert!(process.check_profile(&small).is_ok(), "`chat` alone fits in 800 bytes");
    ports.admit_staging("chat", Tier::Device(0), 300).unwrap();
    match process.check_profile(&small) {
        Err(AdmitError::Ledger(nagaya_ledger::LedgerError::Refused(refusal))) => {
            assert_eq!(refusal.shortfall, vec![(vec![Tier::Device(0)], 100)]);
        }
        other => panic!("600 bytes of `chat` and 300 of staging do not fit in 800: {other:?}"),
    }
    ports.release_staging(Tier::Device(0), 300).unwrap();
    assert!(process.check_profile(&small).is_ok());
    assert_eq!(device_bytes(&process), 400, "the checks changed nothing");
}
