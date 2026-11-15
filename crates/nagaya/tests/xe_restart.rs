//! Arenas through a restart on the two B70 cards (spec 10.2). The test plays systemd, as `process_arena.rs` does with
//! the fake arena: it keeps the slab fds of the first process, stops it with SIGTERM, and starts a second process with
//! the fds at 3 and up. It prints the time from the start of each process to `READY=1`.
//!
//! The process is this test binary: the test build of the `nagaya` binary has the fake inventory, so it cannot name
//! the cards. The child test opens the engine as the binary does. It needs both cards and the checkpoint in
//! `GEMMA4_MODEL_DIR`. Run it alone: `cargo test --release -p nagaya --features xe --test xe_restart --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod fd_support;

mod card_support;

use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixDatagram;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use fd_support::{inode, recv_with_fds, systemd_command};
use nagaya::admission::Registry;
use nagaya::arena::{ArenaSet, FdStore, Listen};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options};
use nagaya::notify::Notifier;
use nagaya::probe::FileProbe;
use nagaya::profile::PlacementProfile;

/// The variable that makes the child test run the process.
const CHILD: &str = "NAGAYA_XE_RESTART_CHILD";

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

fn checkpoint() -> Option<PathBuf> {
    match std::env::var_os("GEMMA4_MODEL_DIR") {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("GEMMA4_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the Gemma 4 checkpoint")
        }
        None => {
            eprintln!("skipped: GEMMA4_MODEL_DIR is not set; set it to the Gemma 4 checkpoint directory");
            None
        }
    }
}

fn profile(dir: &std::path::Path) -> PlacementProfile {
    card_support::need_devices_0_and_1();
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"gemma\"\nkind = \"mk-gemma4-bmg\"\ndevices = [0, 1]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = [4096]\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\n"
    ))
    .expect("the profile parses")
}

/// The process of one start: read the stored fds, open the engine, send READY, and wait for SIGTERM.
///
/// The child leaves the environment as it is, because the test harness runs threads. It reads the fds by the rules of
/// sd_listen_fds(3), with no `$LISTEN_PIDFDID`.
#[test]
fn child_process_for_the_restart_test() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let dir = checkpoint().expect("the parent sets the checkpoint");
    let value = |name: &str| std::env::var(name).ok();
    let listen = Listen {
        pid: value("LISTEN_PID"),
        fds: value("LISTEN_FDS"),
        names: value("LISTEN_FDNAMES"),
        pidfd_id: None,
    };
    // SAFETY: the shell of the parent put the stored fds at 3 and up, and nothing else in this process owns them.
    let stored = unsafe { ArenaSet::from_listen_at(&listen, std::process::id(), 3, || Ok(0)) }.expect("the fds");
    let fd_store = FdStore::from_env().expect("the fd store");
    let notifier = Notifier::from_env().expect("the notify socket");
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    inventory
        .ignore(&[IGNORE.to_owned()])
        .expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let ready = lifecycle::start_with(
        profile(&dir),
        Registry::compiled_in(),
        inventory,
        store.path(),
        notifier.as_ref(),
        Options::new(Duration::from_secs(600)).arenas(stored, fd_store),
    )
    .expect("the process opens the engine");
    // The parent waits for this line before SIGTERM, so the line cannot be lost.
    let line = format!("STATUS=handoff {:?}", ready.opened[0].handoff);
    notifier
        .as_ref()
        .expect("the parent sets NOTIFY_SOCKET")
        .notify(&line)
        .expect("the notify socket");
    // SIGTERM ends the process: the test installs no handler.
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}

/// The test's part of systemd for one unit.
struct Systemd {
    directory: tempfile::TempDir,
    socket: UnixDatagram,
    fds: BTreeMap<String, OwnedFd>,
}

/// One start: the messages, the stderr of the process and the time to READY.
struct Run {
    texts: Vec<String>,
    stderr: String,
    ready_ms: u128,
}

impl Systemd {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = UnixDatagram::bind(directory.path().join("notify")).unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        Self {
            directory,
            socket,
            fds: BTreeMap::new(),
        }
    }

    fn run(&mut self, dir: &std::path::Path) -> Run {
        let fds: Vec<(&str, &OwnedFd)> = self.fds.iter().map(|(name, fd)| (name.as_str(), fd)).collect();
        let (mut command, high) = systemd_command(&std::env::current_exe().unwrap(), &fds);
        command
            .args([
                "--exact",
                "child_process_for_the_restart_test",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("GEMMA4_MODEL_DIR", dir)
            .env("NOTIFY_SOCKET", self.directory.path().join("notify"))
            .env("FDSTORE", "64")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let started = Instant::now();
        let mut child = command.spawn().unwrap();
        drop(high);
        let mut texts = Vec::new();
        let mut ready_ms = None;
        let ready_ms = loop {
            let Ok((text, fds)) = recv_with_fds(&self.socket) else {
                let exited = child.try_wait().unwrap();
                if exited.is_some() || started.elapsed() > Duration::from_secs(600) {
                    let _ = child.kill();
                    let output = child.wait_with_output().unwrap();
                    panic!(
                        "the process sent no READY ({exited:?}): {texts:?}\n{}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                continue;
            };
            let name = text
                .lines()
                .find_map(|line| line.strip_prefix("FDNAME="))
                .map(str::to_owned);
            if text.lines().any(|line| line == "FDSTORE=1") {
                let [fd] = <[OwnedFd; 1]>::try_from(fds).expect("one fd with FDSTORE=1");
                self.fds.insert(name.unwrap(), fd);
            } else if text.lines().any(|line| line == "FDSTOREREMOVE=1") {
                self.fds.remove(&name.unwrap());
            }
            if text.starts_with("READY=1") {
                ready_ms = Some(started.elapsed().as_millis());
            }
            let handoff = text.starts_with("STATUS=handoff");
            texts.push(text);
            if handoff {
                break ready_ms.expect("READY comes before the handoff line");
            }
        };
        // SAFETY: the child pid is live.
        assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
        let output = child.wait_with_output().unwrap();
        Run {
            texts,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            ready_ms,
        }
    }
}

#[test]
fn a_restart_on_the_cards_attaches_the_stored_slabs() {
    if std::env::var_os(CHILD).is_some() {
        return;
    }
    let Some(dir) = checkpoint() else {
        return;
    };
    let mut systemd = Systemd::new();
    let cold = systemd.run(&dir);
    let stored = cold.texts.iter().filter(|text| text.starts_with("FDSTORE=1")).count();
    assert_eq!(stored, 31, "{:?}\n{}", cold.texts, cold.stderr);
    assert!(
        cold.texts.iter().any(|text| text == "STATUS=handoff Some(Cold)"),
        "{:?}",
        cold.texts
    );
    let head = inode(&systemd.fds["gemma/device1/head"]);

    let hot = systemd.run(&dir);
    assert!(
        hot.texts.iter().all(|text| !text.starts_with("FDSTORE")),
        "a hot start changed the store: {:?}",
        hot.texts
    );
    assert!(
        hot.texts.iter().any(|text| text == "STATUS=handoff Some(Hot(Spot))"),
        "{:?}",
        hot.texts
    );
    assert_eq!(inode(&systemd.fds["gemma/device1/head"]), head);
    eprintln!(
        "start to READY on the cards: cold {} ms, hot {} ms",
        cold.ready_ms, hot.ready_ms
    );
}
