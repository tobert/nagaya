//! Helpers for the gossip tests. A wait is bounded and fails with the cluster view that it saw.

#![allow(dead_code)]

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

    /// The lines read so far.
    pub fn seen(&self) -> &[String] {
        &self.seen
    }
}
