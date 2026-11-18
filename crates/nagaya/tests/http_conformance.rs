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

/// The engine has a control table of 4 entries, so the suite runs its control checks (spec 11.12). The table is a
/// device term of 4 entries of 32 bytes.
const PROFILE: &str = "[[engines]]\nname = \"toy\"\nkind = \"fake-toy\"\ndevices = [0]\nshare = 1\n\
     config = { control = 4 }\n\
     [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4224\n\
     [[budgets]]\ntiers = [\"host\"]\nbytes = 33554432\n\
     [store]\nnvme = 67108864\nhost = 16777216\nobjects = 25165824\n";

/// The checks of gogi's suite for context records that do not depend on the reply (gogi `checks/records.rs`). Each
/// must run and pass, not skip.
const RECORD_CHECKS: [&str; 24] = [
    "a build names the record of its context",
    "the context id is the content id of the record's text",
    "identity names context_records, at least 1",
    "a record reads back with its id and no parent",
    "a record reads back as its context",
    "an unknown record is a 404",
    "a record id that is not whole names id",
    "a parent alone is the same context",
    "a build from a parent starts from the parent's snapshots",
    "a context sent with a parent and sent whole is one context",
    "a member beside parent replaces the parent's",
    "a dry run names no record",
    "a dry run holds no record",
    "a parent the server does not hold names context.parent and quotes the id",
    "a parent that is not a whole content id is a 400 naming context.parent",
    "a member that does not stand beside parent is a 400 naming it",
    "a bad new message names its place in the whole context",
    "a decision names the parent under contexts",
    "a decision from a parent gives the numbers of the whole context",
    "a decision holds the record of its context",
    "a pin reports each record with whether it is held",
    "a pin list names the held records only",
    "a pin of a record id that is not whole names contexts",
    "a generation names the record of its context and its reply",
];

/// The checks of gogi's suite for several replies from one generation (gogi `checks/replies.rs`, spec 11.15). Each must
/// run and pass, not skip.
const REPLY_CHECKS: [&str; 26] = [
    "a served model with replies names replies_per_generation, at least 1",
    "forks give one reply for each fork, in order",
    "the seed of reply i is the seed of the request plus i",
    "a response with replies has no reply of its own",
    "a fork is the reply that the same request gives alone",
    "usage counts the context of forks once and the replies together",
    "a fork's usage has its own completion and no prompt",
    "replies ids leave the text out",
    "replies ids name the records of the same replies",
    "a reply reads back later as the last message of its record",
    "a fork of a list takes the sample of its entry",
    "a fork of a list keeps a seed of its own",
    "one fork still has replies",
    "contexts give one reply after each, and each is the reply of its context alone",
    "a reply of contexts has the usage of its own context",
    "usage counts each context of contexts",
    "an unheld parent of contexts names its context",
    "a bad message of contexts names its context and its place",
    "context beside contexts names contexts",
    "forks beside contexts names forks",
    "forks of 0 is a 400 naming forks",
    "forks past replies_per_generation is a 400 naming forks",
    "forks that is not a count or a list is a 400 naming forks",
    "replies with no forks is a 400 naming replies",
    "a stream with forks is a 400 naming stream",
    "replies that is not full or ids is a 400 naming replies",
];

/// The checks of gogi's suite for a served model with `control_vectors` (gogi `checks/control.rs`). Each must run and
/// pass, not skip.
const CONTROL_CHECKS: [&str; 19] = [
    "a control vector response names its served model and counts each side",
    "a control vector response carries the served model's identity",
    "a control vector request writes no tokens",
    "the control vector names its served model and weights",
    "raw reads with an unknown op are a 400 naming contexts[1].control[0].op",
    "a build with a strength that is not a number in control is a 400 naming control[0].strength",
    "a control vector request with a context with a control of its own is a 400 naming negative[1].control",
    "the same contexts give the same vector",
    "the control vector reads back with its media type",
    "the control vector's bytes hash to its id",
    "the control vector has the layers and width the response names",
    "add and ablate at strength 0 give the numbers of no control",
    "a list that changes nothing still has model state of its own",
    "a list that changes the model never reuses the plain model state",
    "a control vector the server does not hold is a 404 naming control[0].vector",
    "a control vector for other weights is a 400 naming control[0].vector",
    "a control vector of another width is a 400 naming control[0].vector",
    "a side past control_vector_contexts is a 400 naming it",
    "a dry run of a control vector request makes no vector",
];

/// A free TCP port on loopback. The port stays the same through a restart.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A profile with two fake-toy engines, each its own text served model (spec 11.7, "Several text models in one
/// process", Proposed). `toy-a` has a control table and `toy-b` has none, so the suite checks both paths.
const TWO_MODELS: &str = "[[engines]]\nname = \"toy-a\"\nkind = \"fake-toy\"\ndevices = [0]\nshare = 1\n\
     config = { model = \"toy-a\", control = 4 }\n\
     [[engines]]\nname = \"toy-b\"\nkind = \"fake-toy\"\ndevices = [0]\nshare = 1\n\
     config = { model = \"toy-b\" }\n\
     [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 8320\n\
     [[budgets]]\ntiers = [\"host\"]\nbytes = 33554432\n\
     [store]\nnvme = 67108864\nhost = 16777216\nobjects = 25165824\n";

/// A nagaya process with the fake-toy engine, and its directory.
struct Nagaya {
    directory: tempfile::TempDir,
    port: u16,
    child: Option<Child>,
    lines: Option<Lines>,
}

impl Nagaya {
    fn start() -> Self {
        Self::start_with(PROFILE)
    }

    fn start_with(profile: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store = directory.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(store.join("profile.toml"), profile).unwrap();
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
        // No engine of the fake declares layer loops yet, so the suite checks the refusal (gogi "Layer loops").
        "a build with a loop to a model without layer_loops is a 400 naming loop",
        "a held spec survives a restart",
        "a pinned snapshot survives a restart",
        "a context whose head is pinned feeds 0 tokens after a restart",
        "a decision replays bit for bit across a restart",
        "generation holds its head",
        "stream reasoning and content pieces join to done's text",
        "an empty content after reasoning closes the thinking",
        "a continuation of open content the client wrote starts with it",
        "a dry run of an open message the client wrote reports what it would feed",
        "an open message with reasoning and thinking false is a 400 naming continue",
        "a build refuses a context without thinking with a 400 naming thinking",
        "a build refuses a context whose thinking is not a boolean with a 400 naming thinking",
        "a generation refuses a context without thinking with a 400 naming thinking",
        "a context with thinking true builds on a model that declares thinking",
        "continue that is a string is a 400 naming continue",
        "a decision with an open context is a 400 naming continue",
        "raw reads of an open context are a 400 naming continue",
        "continue: false on the last message is a closed message",
        // Objects, with a budget of 24 MiB: object_bytes and keep_bytes are each 12 MiB, so the suite probes both.
        "an object reads back bit for bit",
        "a range reads its bytes",
        "an object past object_bytes is a 413",
        "a keep set past keep_bytes is a 413",
        "a PUT replaces the owner's whole keep set",
        "a kept object survives a restart bit for bit",
        "an owner's keeps survive a restart",
    ] {
        assert!(ok(name), "{name}");
    }
    // The control checks ran: the engine has a control table, and the process has an object store (spec 11.12).
    for name in CONTROL_CHECKS {
        assert!(ok(name), "{name}");
    }
    // The record checks ran (gogi "Context records", spec 6.17, 11.13).
    for name in RECORD_CHECKS {
        assert!(ok(name), "{name}");
    }
    assert!(ok("a context record survives a restart"));
    // The reply checks ran: the served model declares `replies` (gogi "Several replies", spec 11.15).
    for name in REPLY_CHECKS {
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
            "a build without dry_run refuses an open message, naming continue",
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

/// The suite against each of two text served models of one process. Both share the specs, the pins and the snapshots
/// of the server, and each restart check stops the whole process.
#[test]
fn gogi_conformance_passes_on_each_of_two_text_models_of_one_process() {
    let nagaya = Rc::new(RefCell::new(Nagaya::start_with(TWO_MODELS)));
    let url = nagaya.borrow().url();
    for model in ["toy-a", "toy-b"] {
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
                url: url.clone(),
                restart: Some(restart),
                model: Some(model.to_owned()),
            },
            false,
        );
        let failed = failures(&report);
        let stderr = nagaya.borrow_mut().lines.as_mut().unwrap().drain().join("\n");
        assert!(
            failed.is_empty(),
            "{model}: {} failed:\n{failed:#?}\nstderr:\n{stderr}",
            failed.len()
        );
        assert!(report.checks.len() > 150, "{model}: {}", report.summary());
        eprintln!("{model}: {}", report.summary());
        let ok = |name: &str| {
            report
                .checks
                .iter()
                .any(|check| check.name == name && check.status == Status::Ok)
        };
        let other = if model == "toy-a" { "toy-b" } else { "toy-a" };
        for name in [
            format!("{other} builds the same context into a snapshot of its own"),
            "a held spec survives a restart".to_owned(),
            "a pinned snapshot survives a restart".to_owned(),
            "generation holds its head".to_owned(),
            "a build with a loop to a model without layer_loops is a 400 naming loop".to_owned(),
        ] {
            assert!(ok(&name), "{model}: {name}");
        }
        // toy-a runs the control checks, and toy-b, with no table, runs the refusals of a model without the capability.
        let control: &[&str] = match model {
            "toy-a" => &CONTROL_CHECKS,
            _ => &[
                "a build with control to a model without control_vectors is a 400 naming control",
                "a generation with control to a model without control_vectors is a 400 naming control",
                "a control vector from a model without control_vectors is a 400 naming model",
            ],
        };
        for name in control {
            assert!(ok(name), "{model}: {name}");
        }
        // Both served models share the records of the server.
        for name in RECORD_CHECKS {
            assert!(ok(name), "{model}: {name}");
        }
        for name in REPLY_CHECKS {
            assert!(ok(name), "{model}: {name}");
        }
    }
    assert_eq!(
        nagaya.borrow_mut().terminate(),
        Some(0),
        "a SIGTERM stops the process cleanly"
    );
}
