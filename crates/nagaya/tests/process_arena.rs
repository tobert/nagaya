//! Arenas through a restart with the process binary and the fake arena (spec 10.2). The test plays systemd: it owns
//! the notify socket, keeps the fds of `FDSTORE=1`, stops the first process with SIGTERM, and starts the next one
//! with the fds at 3 and up and `$LISTEN_PID`, `$LISTEN_FDS` and `$LISTEN_FDNAMES` set.

mod fd_support;

use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixDatagram;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use fd_support::{bytes_of, dup_at, inode, recv_with_fds, systemd_command};

const NAME: &str = "fake-one/device0/w";

/// The test's part of systemd: the notify socket and the fd store of one unit.
struct Systemd {
    directory: tempfile::TempDir,
    socket: UnixDatagram,
    config: PathBuf,
    store_dir: PathBuf,
    /// The fd store, by name.
    fds: BTreeMap<String, OwnedFd>,
}

/// What one run of the process sent on the notify socket.
#[derive(Default)]
struct Run {
    texts: Vec<String>,
    stderr: String,
    code: Option<i32>,
    open_ms: u128,
}

impl Systemd {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = UnixDatagram::bind(directory.path().join("notify")).unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let store_dir = directory.path().join("store");
        std::fs::create_dir_all(&store_dir).unwrap();
        let config = directory.path().join("nagaya.toml");
        std::fs::write(
            &config,
            format!(
                "http_bind = \"127.0.0.1:0\"\ngossip_bind = \"127.0.0.1:0\"\nstore_dir = \"{}\"\n\
                 park_budget_secs = 1\n",
                store_dir.display()
            ),
        )
        .unwrap();
        Self {
            directory,
            socket,
            config,
            store_dir,
            fds: BTreeMap::new(),
        }
    }

    fn profile(&self, arena: &str) {
        std::fs::write(
            self.store_dir.join("profile.toml"),
            format!(
                "[[engines]]\nname = \"fake-one\"\nkind = \"fake-arena\"\ndevices = [0]\nshare = 1\n\
                 [engines.config]\narena = \"{arena}\"\n\
                 [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
                 [[budgets]]\ntiers = [\"host\"]\nbytes = 256\n\
                 [store]\nnvme = 65536\n"
            ),
        )
        .unwrap();
    }

    /// Start the process with the fd store, as systemd starts a unit with `FileDescriptorStoreMax=8`.
    fn spawn(&self) -> Child {
        let fds: Vec<(&str, &OwnedFd)> = self.fds.iter().map(|(name, fd)| (name.as_str(), fd)).collect();
        let (mut command, high) = systemd_command(std::path::Path::new(env!("CARGO_BIN_EXE_nagaya")), &fds);
        command
            .env("NAGAYA_CONFIG", &self.config)
            .env("NOTIFY_SOCKET", self.directory.path().join("notify"))
            .env("FDSTORE", "8")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let child = command.spawn().unwrap();
        drop(high);
        child
    }

    /// Run the process until READY, keep the fd store as systemd does, then stop it with SIGTERM.
    fn run(&mut self) -> Run {
        let started = Instant::now();
        let mut child = self.spawn();
        let mut run = Run::default();
        loop {
            let Ok((text, fds)) = recv_with_fds(&self.socket) else {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "the process sent no READY: {:?}\n{}",
                    run.texts,
                    String::from_utf8_lossy(&output.stderr)
                );
            };
            self.keep(&text, fds);
            let ready = text.starts_with("READY=1");
            run.texts.push(text);
            if ready {
                run.open_ms = started.elapsed().as_millis();
                break;
            }
        }
        // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
        assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
        let output = child.wait_with_output().unwrap();
        run.code = output.status.code();
        run.stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        run
    }

    /// Change the fd store by one message, as sd_notify(3) says.
    fn keep(&mut self, text: &str, fds: Vec<OwnedFd>) {
        let name = text
            .lines()
            .find_map(|line| line.strip_prefix("FDNAME="))
            .map(str::to_owned);
        if text.lines().any(|line| line == "FDSTORE=1") {
            let [fd] = <[OwnedFd; 1]>::try_from(fds).expect("one fd with each FDSTORE=1");
            self.fds.insert(name.expect("FDSTORE=1 with a name"), fd);
        } else if text.lines().any(|line| line == "FDSTOREREMOVE=1") {
            self.fds.remove(&name.expect("FDSTOREREMOVE=1 with a name"));
        }
    }
}

fn stores(run: &Run) -> Vec<&str> {
    run.texts
        .iter()
        .filter(|text| text.starts_with("FDSTORE"))
        .map(String::as_str)
        .collect()
}

#[test]
fn a_restart_attaches_the_arena_that_the_first_process_stored() {
    let mut systemd = Systemd::new();
    systemd.profile("build one");
    let first = systemd.run();
    assert_eq!(first.code, Some(0), "{}", first.stderr);
    assert_eq!(stores(&first), [format!("FDSTORE=1\nFDNAME={NAME}")]);
    assert_eq!(bytes_of(&systemd.fds[NAME]), b"build one");
    let object = inode(&systemd.fds[NAME]);

    let second = systemd.run();
    assert_eq!(second.code, Some(0), "{}", second.stderr);
    assert!(stores(&second).is_empty(), "a hot start stored fds: {:?}", second.texts);
    assert!(
        second
            .stderr
            .contains("engine `fake-one` attached its stored arena: 1 fds, check full"),
        "{}",
        second.stderr
    );
    assert_eq!(
        inode(&systemd.fds[NAME]),
        object,
        "the store holds the arena of the first process"
    );
    eprintln!("open to READY: cold {} ms, hot {} ms", first.open_ms, second.open_ms);
}

#[test]
fn a_restart_with_another_build_drops_the_stored_arena_and_stores_its_own() {
    let mut systemd = Systemd::new();
    systemd.profile("build one");
    let first = systemd.run();
    assert_eq!(first.code, Some(0), "{}", first.stderr);
    systemd.profile("build two");
    let second = systemd.run();
    assert_eq!(second.code, Some(0), "{}", second.stderr);
    assert_eq!(
        stores(&second),
        [
            format!("FDSTOREREMOVE=1\nFDNAME={NAME}"),
            format!("FDSTORE=1\nFDNAME={NAME}")
        ]
    );
    assert!(
        second.stderr.contains("does not use its stored arena"),
        "{}",
        second.stderr
    );
    assert_eq!(bytes_of(&systemd.fds[NAME]), b"build two");
}

#[test]
fn listen_fds_for_another_pid_are_not_used() {
    let mut systemd = Systemd::new();
    systemd.profile("build one");
    systemd.run();
    // A LISTEN_PID of another process: the fds at 3 and up are not ours, so the start is cold (sd_listen_fds(3)).
    let fd = &systemd.fds[NAME];
    let high = dup_at(fd, 300);
    let mut command = Command::new(env!("CARGO_BIN_EXE_nagaya"));
    command
        .env("NAGAYA_CONFIG", &systemd.config)
        .env("NOTIFY_SOCKET", systemd.directory.path().join("notify"))
        .env("FDSTORE", "8")
        .env("LISTEN_PID", "1")
        .env("LISTEN_FDS", "1")
        .env("LISTEN_FDNAMES", NAME)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    // SAFETY: the parent made the duplicate and owns it.
    unsafe { libc::close(high) };
    let mut texts = Vec::new();
    loop {
        let (text, _) = recv_with_fds(&systemd.socket).expect("the process sends READY");
        let ready = text.starts_with("READY=1");
        texts.push(text);
        if ready {
            break;
        }
    }
    // SAFETY: the child pid is live.
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    assert_eq!(child.wait().unwrap().code(), Some(0));
    assert!(
        texts.iter().any(|text| text == &format!("FDSTORE=1\nFDNAME={NAME}")),
        "{texts:?}"
    );
}
