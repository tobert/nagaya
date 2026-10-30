//! The arena fds through a swap in place (spec 3.3a, 10.2): an unload removes the fds of the engine from systemd's
//! store, a load stores the fds of its cold load, and the stored fds of an on-demand engine are dropped at start.

mod fd_support;

use std::os::fd::OwnedFd;
use std::os::unix::net::UnixDatagram;
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use fd_support::{memfd, recv_with_fds};
use nagaya::admission::{AttachError, Attached, Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::arena::{ArenaSet, FdStore};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::lifecycle::{self, Options};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ArenaFd, ArenaPart, Capability, Engine, Kind, Lifetime, Limits, MandatoryTerm, Tier};
use nagaya_sched::Class;

/// The fake engine with an arena, with `close` and 600 bytes of weights on device 0.
struct ArenaClose;

fn bytes(config: &toml::Table) -> Vec<u8> {
    config["arena"].as_str().unwrap().as_bytes().to_vec()
}

fn limits(config: &toml::Table) -> Limits {
    let mut limits = FakeAlways::with_arena(&bytes(config)).limits();
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

/// The fake with the limits of `ArenaClose`.
struct Closing(FakeAlways, Limits);

impl Engine for Closing {
    fn identity(&self) -> nagaya_abi::Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        self.1.clone()
    }

    fn pass(
        &mut self,
        segments: &[nagaya_abi::Segment],
        taps: &[nagaya_abi::Tap],
    ) -> Result<nagaya_abi::PassOut, nagaya_abi::PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.0.coverage()
    }

    fn arena(&mut self) -> Option<&mut dyn nagaya_abi::Arena> {
        self.0.arena()
    }
}

impl Factory for ArenaClose {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits(config))
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Closing(
            FakeAlways::with_arena(&bytes(config)),
            limits(config),
        )))
    }

    fn arena_parts(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
        Ok(FakeAlways::arena_parts())
    }

    fn attach(
        &self,
        config: &toml::Table,
        _devices: &[GroupDevice],
        arenas: Vec<ArenaFd>,
    ) -> Result<Attached, AttachError> {
        let (engine, check) = FakeAlways::attach_arena(arenas, &bytes(config)).map_err(AttachError::Mismatch)?;
        Ok(Attached {
            engine: Box::new(Closing(engine, limits(config))),
            check,
        })
    }
}

fn engine(name: &str, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("arena".to_owned(), toml::Value::String(format!("the arena of {name}")));
    EnginePlacement {
        name: name.to_owned(),
        kind: "arena-close".to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

/// `chat` is resident and `voice` is on demand. They do not fit together in the 1000 bytes of device 0.
fn profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![
            engine("chat", Residency::Resident),
            engine("voice", Residency::OnDemand),
        ],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1000,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 1 << 20,
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

fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1 << 20,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 20,
    }
}

/// The emulated notify socket of systemd.
struct Systemd {
    _directory: tempfile::TempDir,
    socket: UnixDatagram,
    store: FdStore,
}

impl Systemd {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notify");
        let socket = UnixDatagram::bind(&path).unwrap();
        socket.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let store = FdStore::at(&path, 8);
        Self {
            _directory: directory,
            socket,
            store,
        }
    }

    /// The text of every datagram until the socket is quiet for 300 ms.
    fn messages(&self) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok((text, _fds)) = recv_with_fds(&self.socket) {
            out.push(text);
        }
        out
    }
}

fn start(stored: ArenaSet, systemd: &Systemd, store_dir: &Path) -> lifecycle::Ready {
    let mut registry = Registry::new();
    registry.register("arena-close", Box::new(ArenaClose));
    let options = Options::new(Duration::from_secs(5)).arenas(stored, Some(systemd.store.clone()));
    lifecycle::start_with(profile(), registry, inventory(), store_dir, None, options).unwrap()
}

struct Done(mpsc::Sender<()>);

impl JobBody for Done {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = self.0.send(());
        Ran::Done
    }
}

/// Load an engine of the profile by its place, and run a job on it.
fn run_on(ready: &lifecycle::Ready, engine: usize) {
    load(&ready.process, ["chat", "voice"][engine]);
    let (done, finished) = mpsc::channel();
    ready
        .process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine,
                body: Box::new(Done(done)),
            },
        )
        .unwrap();
    finished.recv_timeout(Duration::from_secs(5)).expect("the job ran");
}

/// Load an engine, and wait until it is ready. A job for the engine runs only after that (spec 3.3). The load names
/// the engines in its way in `unload`, as a client does after a refusal (spec 3.3a, "Explicit layout").
fn load(process: &nagaya::process::Process, engine: &str) {
    match process.load(engine) {
        Err(nagaya::process::LoadError::InTheWay { in_the_way, .. }) => {
            process.load_with(engine, &in_the_way).unwrap();
        }
        answer => {
            answer.unwrap();
        }
    }
    let ports = process.ports();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while ports.engine_state(engine) != Some(nagaya::notify::EngineState::Ready) {
        assert!(
            std::time::Instant::now() < deadline,
            "engine `{engine}` did not load in 5 s"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

const CHAT: &str = "chat/device0/w";
const VOICE: &str = "voice/device0/w";

#[test]
fn an_unload_removes_the_arena_fds_of_the_engine_and_a_load_stores_its_own() {
    let systemd = Systemd::new();
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start(ArenaSet::empty(), &systemd, directory.path());
    assert_eq!(systemd.messages(), [format!("FDSTORE=1\nFDNAME={CHAT}")]);
    run_on(&ready, 1);
    assert_eq!(
        systemd.messages(),
        [
            format!("FDSTOREREMOVE=1\nFDNAME={CHAT}"),
            format!("FDSTORE=1\nFDNAME={VOICE}")
        ]
    );
    run_on(&ready, 0);
    assert_eq!(
        systemd.messages(),
        [
            format!("FDSTOREREMOVE=1\nFDNAME={VOICE}"),
            format!("FDSTORE=1\nFDNAME={CHAT}")
        ]
    );
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn the_stored_fds_of_an_on_demand_engine_are_dropped_at_start() {
    let systemd = Systemd::new();
    let directory = tempfile::tempdir().unwrap();
    let stored: Vec<(String, OwnedFd)> = vec![
        (CHAT.to_owned(), memfd(b"the arena of chat")),
        (VOICE.to_owned(), memfd(b"the arena of voice")),
    ];
    let mut ready = start(ArenaSet::from_fds(stored), &systemd, directory.path());
    // `chat` attaches its stored arena, and `voice` stays unloaded, so its fd goes.
    assert_eq!(systemd.messages(), [format!("FDSTOREREMOVE=1\nFDNAME={VOICE}")]);
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}
