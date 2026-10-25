//! The start steps of spec 10.2 with the fake arena: a cold load stores its arena, a stored arena that matches
//! attaches, and a stored arena that does not match is dropped before the cold load.

mod fd_support;

use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::net::UnixDatagram;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fd_support::{bytes_of, inode, memfd, recv_with_fds};
use nagaya::admission::{AttachError, Attached, FAKE_ARENA_KIND, Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::arena::{ArenaSet, FdStore};
use nagaya::inventory::{Device, Inventory};
use nagaya::lifecycle::{self, Options};
use nagaya::process::Handoff;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ArenaFd, ArenaPart, CheckStrength, Engine, Limits, Tier};
use nagaya_telemetry::testing::Capture;

const NAME: &str = "fake-one/device0/w";

fn engine(name: &str, kind: &str, arena: &str) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("arena".to_owned(), toml::Value::String(arena.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency: Default::default(),
        idle_unload_s: None,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    let count = engines.len() as u64;
    PlacementProfile {
        engines,
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 4096 * count,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 4096 * count,
            },
        ],
        store: Some(StoreSpec {
            nvme: 65536,
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
    fn new(ceiling: usize) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notify");
        let socket = UnixDatagram::bind(&path).unwrap();
        socket.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let store = FdStore::at(&path, ceiling);
        Self {
            _directory: directory,
            socket,
            store,
        }
    }

    /// Every datagram until the socket is quiet for 300 ms.
    fn messages(&self) -> Vec<(String, Vec<OwnedFd>)> {
        let mut out = Vec::new();
        while let Ok(message) = recv_with_fds(&self.socket) {
            out.push(message);
        }
        out
    }
}

fn start(
    profile: PlacementProfile,
    registry: Registry,
    stored: ArenaSet,
    systemd: &Systemd,
    store_dir: &Path,
) -> Result<lifecycle::Ready, lifecycle::LifecycleError> {
    let options = Options::new(Duration::from_secs(5)).arenas(stored, Some(systemd.store.clone()));
    lifecycle::start_with(profile, registry, inventory(), store_dir, None, options)
}

fn handoff(capture: &Capture, engine: &str, status: &str) -> Option<f64> {
    capture
        .metric("nagaya.arena.handoff")
        .into_iter()
        .find(|point| point.attributes["engine"] == engine && point.attributes["status"] == status)
        .map(|point| point.value)
}

#[test]
fn a_cold_load_stores_each_arena_fd_under_its_name() {
    let capture = Capture::start(Vec::new());
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let profile = profile(vec![engine("fake-one", FAKE_ARENA_KIND, "build one")]);
    let mut ready = start(
        profile,
        Registry::with_fake(),
        ArenaSet::empty(),
        &systemd,
        directory.path(),
    )
    .unwrap();
    assert_eq!(ready.opened[0].handoff, Some(Handoff::Cold));
    let messages = systemd.messages();
    assert_eq!(
        messages.len(),
        1,
        "{:?}",
        messages.iter().map(|m| &m.0).collect::<Vec<_>>()
    );
    assert_eq!(messages[0].0, format!("FDSTORE=1\nFDNAME={NAME}"));
    assert_eq!(bytes_of(&messages[0].1[0]), b"build one");
    assert_eq!(handoff(&capture, "fake-one", "cold"), Some(1.0));
    assert_eq!(handoff(&capture, "fake-one", "hot"), None);
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_stored_arena_that_matches_attaches_and_stores_nothing() {
    let capture = Capture::start(Vec::new());
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![(NAME.to_owned(), memfd(b"build one"))]);
    let profile = profile(vec![engine("fake-one", FAKE_ARENA_KIND, "build one")]);
    let mut ready = start(profile, Registry::with_fake(), stored, &systemd, directory.path()).unwrap();
    assert_eq!(ready.opened[0].handoff, Some(Handoff::Hot(CheckStrength::Full)));
    let messages = systemd.messages();
    assert!(
        messages.is_empty(),
        "{:?}",
        messages.iter().map(|m| &m.0).collect::<Vec<_>>()
    );
    assert_eq!(handoff(&capture, "fake-one", "hot"), Some(1.0));
    assert_eq!(handoff(&capture, "fake-one", "cold"), None);
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_stored_arena_that_does_not_match_is_dropped_and_the_cold_load_stores_its_own() {
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![(NAME.to_owned(), memfd(b"build one"))]);
    let profile = profile(vec![engine("fake-one", FAKE_ARENA_KIND, "build two")]);
    let mut ready = start(profile, Registry::with_fake(), stored, &systemd, directory.path()).unwrap();
    assert_eq!(ready.opened[0].handoff, Some(Handoff::Cold));
    let messages = systemd.messages();
    let texts: Vec<&str> = messages.iter().map(|m| m.0.as_str()).collect();
    assert_eq!(
        texts,
        [
            format!("FDSTOREREMOVE=1\nFDNAME={NAME}"),
            format!("FDSTORE=1\nFDNAME={NAME}")
        ]
    );
    assert_eq!(bytes_of(&messages[1].1[0]), b"build two");
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn stored_names_that_differ_from_the_arena_parts_are_dropped() {
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![("fake-one/device0/x".to_owned(), memfd(b"build one"))]);
    let profile = profile(vec![engine("fake-one", FAKE_ARENA_KIND, "build one")]);
    let mut ready = start(profile, Registry::with_fake(), stored, &systemd, directory.path()).unwrap();
    assert_eq!(ready.opened[0].handoff, Some(Handoff::Cold));
    let texts: Vec<String> = systemd.messages().into_iter().map(|m| m.0).collect();
    assert_eq!(
        texts,
        [
            "FDSTOREREMOVE=1\nFDNAME=fake-one/device0/x".to_owned(),
            format!("FDSTORE=1\nFDNAME={NAME}")
        ]
    );
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_stored_fd_of_no_engine_in_the_profile_is_dropped_after_admission() {
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![
        ("old/device0/w".to_owned(), memfd(b"old")),
        ("stored".to_owned(), memfd(b"x")),
    ]);
    let profile = profile(vec![engine("fake-one", "fake", "")]);
    let mut ready = start(profile, Registry::with_fake(), stored, &systemd, directory.path()).unwrap();
    assert_eq!(ready.opened[0].handoff, None, "a kind with no arena has a handoff");
    let texts: Vec<String> = systemd.messages().into_iter().map(|m| m.0).collect();
    assert_eq!(
        texts,
        [
            "FDSTOREREMOVE=1\nFDNAME=old/device0/w".to_owned(),
            "FDSTOREREMOVE=1\nFDNAME=stored".to_owned()
        ]
    );
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_profile_that_does_not_fit_keeps_each_stored_fd_and_exits_3() {
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![
        (NAME.to_owned(), memfd(b"build two")),
        ("old/device0/w".to_owned(), memfd(b"old")),
    ]);
    let mut too_big = profile(vec![engine("fake-one", FAKE_ARENA_KIND, "build one")]);
    too_big.budgets[0].bytes = 1;
    let error = start(too_big, Registry::with_fake(), stored, &systemd, directory.path()).unwrap_err();
    assert_eq!(error.exit_code(), nagaya::exit::ExitCode::RefusedStart);
    assert!(systemd.messages().is_empty(), "a refused profile dropped a stored fd");
}

#[test]
fn a_profile_with_more_arena_fds_than_the_ceiling_exits_3_and_keeps_each_fd() {
    let systemd = Systemd::new(1);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![(NAME.to_owned(), memfd(b"build two"))]);
    let two = profile(vec![
        engine("fake-one", FAKE_ARENA_KIND, "build one"),
        engine("fake-two", FAKE_ARENA_KIND, "build one"),
    ]);
    let error = start(two, Registry::with_fake(), stored, &systemd, directory.path()).unwrap_err();
    assert_eq!(error.exit_code(), nagaya::exit::ExitCode::RefusedStart);
    assert!(error.to_string().contains("arena fds"), "{error}");
    assert!(systemd.messages().is_empty());
}

/// A factory that records the fds that `attach` gets, and then refuses them.
struct Recording {
    got: Arc<Mutex<Vec<(i32, u64)>>>,
}

impl Factory for Recording {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::with_arena(b"cold").limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(FakeAlways::with_arena(b"cold")))
    }

    fn arena_parts(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
        Ok(FakeAlways::arena_parts())
    }

    fn attach(
        &self,
        _config: &toml::Table,
        _devices: &[GroupDevice],
        arenas: Vec<ArenaFd>,
    ) -> Result<Attached, AttachError> {
        let mut got = self.got.lock().unwrap();
        for arena in &arenas {
            got.push((arena.fd.as_raw_fd(), inode(&arena.fd)));
        }
        Err(AttachError::Mismatch("the test refuses each arena".to_owned()))
    }
}

#[test]
fn attach_gets_a_duplicate_of_each_stored_fd() {
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let fd = memfd(b"build one");
    let (raw, object) = (fd.as_raw_fd(), inode(&fd));
    let stored = ArenaSet::from_fds(vec![(NAME.to_owned(), fd)]);
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut registry = Registry::new();
    registry.register("recording", Box::new(Recording { got: Arc::clone(&got) }));
    let profile = profile(vec![engine("fake-one", "recording", "")]);
    let mut ready = start(profile, registry, stored, &systemd, directory.path()).unwrap();
    let got = got.lock().unwrap().clone();
    assert_eq!(got.len(), 1);
    assert_ne!(got[0].0, raw, "attach got the stored fd, not a duplicate");
    assert_eq!(got[0].1, object, "the duplicate names another object");
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn stored_names_that_differ_from_the_arena_parts_never_reach_attach() {
    let systemd = Systemd::new(8);
    let directory = tempfile::tempdir().unwrap();
    let stored = ArenaSet::from_fds(vec![("fake-one/device0/x".to_owned(), memfd(b"build one"))]);
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut registry = Registry::new();
    registry.register("recording", Box::new(Recording { got: Arc::clone(&got) }));
    let profile = profile(vec![engine("fake-one", "recording", "")]);
    let mut ready = start(profile, registry, stored, &systemd, directory.path()).unwrap();
    assert!(
        got.lock().unwrap().is_empty(),
        "attach got fds whose names are not the arena parts"
    );
    assert_eq!(ready.opened[0].handoff, Some(Handoff::Cold));
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}
