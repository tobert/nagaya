//! Helpers for the gossip tests. A wait is bounded and fails with the cluster view that it saw.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use nagaya::gossip::{ClusterView, Gossip};

/// How long a test waits for gossip. Loopback gossip with a short interval takes well under a second.
pub const WAIT: Duration = Duration::from_secs(30);

/// How often a wait reads the view.
const POLL: Duration = Duration::from_millis(20);

/// Wait until `ready` accepts the view of `gossip`. Panic with the last view when the wait ends.
pub fn wait_for_view(gossip: &Gossip, what: &str, ready: impl Fn(&ClusterView) -> bool) -> ClusterView {
    wait_for_view_within(gossip, what, WAIT, ready)
}

/// Like `wait_for_view`, with a time bound.
pub fn wait_for_view_within(
    gossip: &Gossip,
    what: &str,
    timeout: Duration,
    ready: impl Fn(&ClusterView) -> bool,
) -> ClusterView {
    let deadline = Instant::now() + timeout;
    loop {
        let view = gossip.view();
        if ready(&view) {
            return view;
        }
        if Instant::now() > deadline {
            panic!(
                "no view with {what} in {timeout:?}; the last view of {}:\n{view}",
                gossip.node_name()
            );
        }
        std::thread::sleep(POLL);
    }
}

/// The lines that a child process writes to stderr. A reader thread forwards them.
pub struct Lines {
    receiver: std::sync::mpsc::Receiver<String>,
    seen: Vec<String>,
}

impl Lines {
    /// Read the lines of `stderr` on a new thread.
    pub fn new(stderr: std::process::ChildStderr) -> Self {
        use std::io::BufRead;
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stderr).lines() {
                let Ok(line) = line else { return };
                if sender.send(line).is_err() {
                    return;
                }
            }
        });
        Self {
            receiver,
            seen: Vec::new(),
        }
    }

    /// Wait for a line that starts with `prefix`. Give the rest of the line. Panic with the lines seen.
    pub fn wait_for(&mut self, prefix: &str, timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(line) = self.seen.iter().find(|line| line.starts_with(prefix)) {
                return line[prefix.len()..].to_owned();
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match self.receiver.recv_timeout(left) {
                Ok(line) => self.seen.push(line),
                Err(_) => panic!(
                    "no stderr line with `{prefix}` in {timeout:?}; the lines seen:\n{}",
                    self.seen.join("\n")
                ),
            }
        }
    }

    /// Wait for a line that starts with `prefix` and ends with `suffix`. Give the whole line. Panic with the lines seen.
    pub fn wait_for_line(&mut self, prefix: &str, suffix: &str, timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(line) = self
                .seen
                .iter()
                .find(|line| line.starts_with(prefix) && line.ends_with(suffix))
            {
                return line.clone();
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match self.receiver.recv_timeout(left) {
                Ok(line) => self.seen.push(line),
                Err(_) => panic!(
                    "no stderr line with `{prefix}...{suffix}` in {timeout:?}; the lines seen:\n{}",
                    self.seen.join("\n")
                ),
            }
        }
    }

    /// Read the lines that are waiting, without a wait.
    pub fn drain(&mut self) -> &[String] {
        while let Ok(line) = self.receiver.try_recv() {
            self.seen.push(line);
        }
        &self.seen
    }

    /// The lines read so far.
    pub fn seen(&self) -> &[String] {
        &self.seen
    }
}

const SIGTERM: i32 = 15;

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

/// The profile of a new process: one fake engine.
pub const ACTIVE_PROFILE: &str = "[[engines]]\nname = \"fake-one\"\nkind = \"fake\"\ndevices = [0]\nshare = 1\n\
     [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
     [[budgets]]\ntiers = [\"host\"]\nbytes = 256\n\
     [store]\nnvme = 65536\n";

/// A nagaya process with its own directory. The process is killed when the value drops.
pub struct Proc {
    _directory: tempfile::TempDir,
    child: Option<Child>,
    /// The lines that the process writes to stderr.
    pub lines: Lines,
    /// The gossip address of the process.
    pub gossip_addr: SocketAddr,
    /// The store directory of the process.
    pub store: PathBuf,
}

impl Proc {
    /// Start a process and wait for its gossip address.
    pub fn spawn(name: &str, http: &str, seeds: &[SocketAddr]) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store = directory.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        let seeds: Vec<String> = seeds.iter().map(|seed| format!("\"{seed}\"")).collect();
        let config = directory.path().join("nagaya.toml");
        std::fs::write(
            &config,
            format!(
                "http_bind = \"{http}\"\n\
                 gossip_bind = \"127.0.0.1:0\"\n\
                 store_dir = \"{}\"\n\
                 node_name = \"{name}\"\n\
                 seed_peers = [{}]\n\
                 park_budget_secs = 1\n",
                store.display(),
                seeds.join(", ")
            ),
        )
        .unwrap();
        std::fs::write(store.join("profile.toml"), ACTIVE_PROFILE).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_nagaya"))
            .env("NAGAYA_CONFIG", &config)
            .env_remove("NOTIFY_SOCKET")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = Lines::new(child.stderr.take().unwrap());
        let gossip_addr = lines.wait_for("gossip listening on ", WAIT).parse().unwrap();
        Self {
            _directory: directory,
            child: Some(child),
            lines,
            gossip_addr,
            store,
        }
    }

    /// Send SIGTERM and wait for the exit.
    pub fn terminate(&mut self) -> ExitStatus {
        let mut child = self.child.take().unwrap();
        // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
        assert_eq!(unsafe { kill(child.id() as i32, SIGTERM) }, 0);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit after SIGTERM");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Proc {
    /// The exit status, when the process ended.
    pub fn try_exit(&mut self) -> Option<ExitStatus> {
        self.child.as_mut().and_then(|child| child.try_wait().unwrap())
    }

    /// Wait for the process to end on its own. Panic with the stderr lines when the wait ends.
    pub fn wait_exit(&mut self, timeout: Duration) -> ExitStatus {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.try_exit() {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "the process did not exit in {timeout:?}; its stderr:\n{}",
                self.lines.drain().join("\n")
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
