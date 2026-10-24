//! gogi's conformance suite against the nagaya process with the fake engine (milestone 12 step 6, spec 11.1).
//!
//! The process is the binary, built with the feature `fake`. Its profile has one engine of the kind `fake-toy`: the
//! fake engine over the toy vocabulary, which the process serves with the toy family (spec 11.3b).

mod support;

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::time::{Duration, Instant};

use gogi_conformance::{Opts, Report, Restart, Status, run};
use support::Lines;

const PROFILE: &str = "[[engines]]\nname = \"toy\"\nkind = \"fake-toy\"\ndevices = [0]\nshare = 1\n\
     [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
     [[budgets]]\ntiers = [\"host\"]\nbytes = 33554432\n\
     [store]\nnvme = 67108864\nhost = 16777216\n";

/// A free TCP port on loopback. The port stays the same through a restart.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A nagaya process with the fake-toy engine, and its directory.
struct Nagaya {
    directory: tempfile::TempDir,
    port: u16,
    child: Option<Child>,
    lines: Option<Lines>,
}

impl Nagaya {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store = directory.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(store.join("profile.toml"), PROFILE).unwrap();
        let mut nagaya = Self {
            directory,
            port: free_port(),
            child: None,
            lines: None,
        };
        nagaya.spawn();
        nagaya
    }

    fn config(&self) -> PathBuf {
        let path = self.directory.path().join("nagaya.toml");
        let store: &Path = &self.directory.path().join("store");
        std::fs::write(
            &path,
            format!(
                "http_bind = \"127.0.0.1:{}\"\ngossip_bind = \"127.0.0.1:0\"\nstore_dir = \"{}\"\n\
                 node_name = \"conformance\"\npark_budget_secs = 1\n",
                self.port,
                store.display()
            ),
        )
        .unwrap();
        path
    }

    fn spawn(&mut self) {
        let mut child = Command::new(env!("CARGO_BIN_EXE_nagaya"))
            .env("NAGAYA_CONFIG", self.config())
            .env_remove("NOTIFY_SOCKET")
            .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = Lines::new(child.stderr.take().unwrap());
        let addr = lines.wait_for("http listening on ", Duration::from_secs(60));
        assert_eq!(addr, format!("127.0.0.1:{}", self.port));
        // The server listens before the engines open, and serves the routes from READY on (spec 10.1).
        lines.wait_for("ready with ", Duration::from_secs(60));
        self.child = Some(child);
        self.lines = Some(lines);
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Send SIGTERM and wait for the exit. Give the exit code.
    fn terminate(&mut self) -> Option<i32> {
        let mut child = self.child.take().unwrap();
        // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
        assert_eq!(unsafe { kill(child.id() as i32, 15) }, 0);
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status.code();
            }
            if Instant::now() > deadline {
                let seen = self
                    .lines
                    .as_mut()
                    .map(|lines| lines.drain().join("\n"))
                    .unwrap_or_default();
                panic!("the process did not exit after SIGTERM; its stderr:\n{seen}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Nagaya {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

fn failures(report: &Report) -> Vec<String> {
    report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| format!("{}: {}", check.name, check.detail))
        .collect()
}

#[test]
fn gogi_conformance_passes_against_the_process_with_the_fake_engine() {
    let nagaya = Rc::new(RefCell::new(Nagaya::start()));
    let url = nagaya.borrow().url();
    // The restart check stops the process with SIGTERM and starts it again on the same store and port (spec 6.11).
    let held = Rc::clone(&nagaya);
    let restart = Restart::Hook(Box::new(move || {
        let mut nagaya = held.borrow_mut();
        match nagaya.terminate() {
            Some(0) => {}
            code => return Err(format!("the stop before the restart exited with {code:?}, not 0")),
        }
        nagaya.spawn();
        Ok(())
    }));
    let report = run(
        Opts {
            url,
            restart: Some(restart),
            model: None,
        },
        false,
    );
    let mut nagaya = nagaya.borrow_mut();
    let failed = failures(&report);
    let skipped: Vec<String> = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Skip)
        .map(|check| format!("{}: {}", check.name, check.detail))
        .collect();
    let stderr = nagaya.lines.as_mut().unwrap().drain().join("\n");
    assert!(
        failed.is_empty(),
        "{} failed:\n{:#?}\nskipped: {skipped:#?}\nstderr:\n{stderr}",
        failed.len(),
        failed
    );
    assert!(report.checks.len() > 150, "{}", report.summary());
    eprintln!("{}\nskipped: {skipped:#?}", report.summary());
    let ok = |name: &str| {
        report
            .checks
            .iter()
            .any(|check| check.name == name && check.status == Status::Ok)
    };
    // The generation and continuation checks ran: none of them is a skip.
    for name in [
        "the server answers again after the restart",
        "a held spec survives a restart",
        "a pinned snapshot survives a restart",
        "a context whose head is pinned feeds 0 tokens after a restart",
        "a decision replays bit for bit across a restart",
        "generation holds its head",
        "stream reasoning and content pieces join to done's text",
        "a build without dry_run refuses an open message, naming continue",
        "an empty content after reasoning closes the thinking",
        "a continuation of open content the client wrote starts with it",
        "a dry run of an open message the client wrote reports what it would feed",
        "an open message with reasoning and thinking absent is a 400 naming continue",
        "continue that is a string is a 400 naming continue",
        "a decision with an open context is a 400 naming continue",
        "raw reads of an open context are a 400 naming continue",
        "continue: false on the last message is a closed message",
    ] {
        assert!(ok(name), "{name}");
    }
    // The checks of a cut reply need a whole reply of two tokens or more. The suite's context holds a random nonce,
    // and about one run in twenty, the fake engine's greedy reply is one token. Then the suite skips them.
    let short = report.checks.iter().any(|check| {
        check.name == "a cut reply resumes" && check.status == Status::Skip && check.detail.contains("too few")
    });
    if !short {
        for name in [
            "a cut reply names the phase of its next token",
            "a cut reply's head is listed last in snapshots",
            "a dry run of a cut reply sent back open keeps its head",
            "a continuation resumed from the head feeds at most the last sampled token",
            "a dry run of a cut reply sent back open reports a dry run",
            "a continuation's done starts with the open text",
            "a continuation's completion counts only the sampled tokens",
            "a continuation's stream counts n from 1 over the sampled tokens",
            "a continuation's pieces join to done's text after the open text",
        ] {
            assert!(ok(name), "{name}");
        }
    }
    assert_eq!(
        nagaya.terminate(),
        Some(0),
        "a SIGTERM stops the process cleanly (spec 10.3)"
    );
}

#[test]
fn an_http_bind_that_fails_refuses_the_start() {
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let directory = tempfile::tempdir().unwrap();
    let store = directory.path().join("store");
    std::fs::create_dir_all(&store).unwrap();
    std::fs::write(store.join("profile.toml"), PROFILE).unwrap();
    let config = directory.path().join("nagaya.toml");
    std::fs::write(
        &config,
        format!(
            "http_bind = \"{}\"\ngossip_bind = \"127.0.0.1:0\"\nstore_dir = \"{}\"\nnode_name = \"taken\"\n",
            taken.local_addr().unwrap(),
            store.display()
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_nagaya"))
        .env("NAGAYA_CONFIG", &config)
        .env_remove("NOTIFY_SOCKET")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("cannot bind the HTTP address"), "{stderr}");
}
