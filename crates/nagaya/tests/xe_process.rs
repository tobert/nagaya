//! The nagaya process with the Gemma 4 engine on the two B70 cards: a decision, a generation, and gogi's conformance
//! suite (milestone 15s step 6, spec 11.1).
//!
//! The test starts the process in this test process: the test build of the binary has the fake inventory (`fake`
//! feature of the dev-dependency), so it cannot name the cards. The test probes the machine, as the binary does without
//! `fake`. It needs both cards and the checkpoint in `GEMMA4_MODEL_DIR`. Run it alone, with `--test-threads=1`:
//! `cargo test --release -p nagaya --features xe --test xe_process -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gogi_conformance::{Opts, Status, run};
use nagaya::admission::Registry;
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, Server, router};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options};
use nagaya::probe::FileProbe;
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use nagaya::signal::SignalFlag;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "gemma-4-26b-a4b";

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

/// The profile of b70host: the engine on machine devices 0 and 1, the B70 cards, and a budget for each card.
fn profile(dir: &std::path::Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"gemma\"\nkind = \"mk-gemma4-bmg\"\ndevices = [0, 1]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = [4096, 16384]\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path);
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn spec() -> Value {
    json!({"name": "g", "instructions": "Judge the command.", "input_label": "Command", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "What happens?", "criteria": [
            {"option": "allow", "means": "fine"}, {"option": "ask", "means": "not clearly fine"},
            {"option": "report", "means": "harmful"}]}]})
}

fn context(content: &str) -> Value {
    json!({"messages": [
        {"role": "system", "content": "You judge shell commands."},
        {"role": "user", "content": content, "snap": true}]})
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The process opens the engine on the cards by their PCI addresses, serves a decision and a generation, and passes
/// gogi's conformance suite.
#[test]
fn the_process_serves_gemma4_on_both_cards() {
    let Some(dir) = checkpoint() else {
        return;
    };
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    for id in [0, 1] {
        let device = inventory.device(id).expect("devices 0 and 1");
        println!("device {id}: {} at {}", device.name, device.pci_address);
        assert!(!device.ignored, "device {id} is a B70 card, not the ignored iGPU");
    }
    let store = tempfile::tempdir().unwrap();
    let profile = profile(&dir);
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let start = Instant::now();
    let mut ready = lifecycle::start_with(
        profile.clone(),
        Registry::compiled_in(),
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens the engine");
    println!("opened in {} ms", start.elapsed().as_millis());
    let council = Arc::new(
        council(
            &FamilyRegistry::compiled_in(),
            &profile,
            &ready.opened,
            &ready.process.ports(),
            store.path(),
            None,
        )
        .expect("the process serves the engine"),
    );

    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK, "{identity}");
    assert!(identity.to_string().contains(MODEL), "{identity}");

    let (status, held) = send(&council, "POST", "/council/v2/specs", Some(spec()));
    assert_eq!(status, StatusCode::OK, "{held}");
    let spec_id = held["spec_id"].as_str().expect("a spec id").to_owned();
    let start = Instant::now();
    let body = json!({"model": MODEL, "spec_id": spec_id, "case": "rm -rf /", "contexts": [context("Clean up.")]});
    let (status, decision) = send(&council, "POST", "/council/v2/decisions", Some(body));
    assert_eq!(status, StatusCode::OK, "{decision}");
    println!(
        "decision in {} ms: {}",
        start.elapsed().as_millis(),
        decision["answers"]
    );
    assert!(
        decision["answers"].is_object() || decision["answers"].is_array(),
        "{decision}"
    );

    let start = Instant::now();
    let body = json!({"model": MODEL, "context": context("Say hello in five words."), "max_tokens": 48,
        "sample": {"temperature": 0}});
    let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
    assert_eq!(status, StatusCode::OK, "{generation}");
    let elapsed = start.elapsed().as_millis();
    println!("generation in {elapsed} ms: {generation}");
    let usage = &generation["usage"];
    println!("usage {usage}");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let server = Server::start(
        runtime.handle(),
        format!("127.0.0.1:{}", free_port()).parse().unwrap(),
        Arc::clone(&council),
    )
    .expect("the HTTP server binds a loopback port");
    let report = run(
        Opts {
            url: format!("http://{}", server.local_addr()),
            restart: None,
            model: None,
        },
        false,
    );
    let count = |status: Status| report.checks.iter().filter(|check| check.status == status).count();
    println!(
        "gogi-conformance: {} ok, {} FAIL, {} skip",
        count(Status::Ok),
        count(Status::Fail),
        count(Status::Skip)
    );
    for check in report.checks.iter().filter(|check| check.status != Status::Ok) {
        println!("{:?} {}: {}", check.status, check.name, check.detail);
    }
    server.stop(Duration::from_secs(2));
    council.drain();
    let signals = SignalFlag::new();
    let stopped = lifecycle::stop(&mut ready, None, Duration::from_secs(30), &signals);
    println!("stop: {}", stopped.log_line());
    // Spec 6.14 (decided 2026-10-21): a generation on an engine with a feed rule holds no head. Since gogi a07d85e,
    // gogi's check of the head skips a null head, so no check may fail.
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| check.name.as_str())
        .collect();
    assert_eq!(
        failed,
        Vec::<&str>::new(),
        "gogi-conformance failed; the lines above name each check"
    );
    assert!(generation["head"].is_null(), "{generation}");
}
