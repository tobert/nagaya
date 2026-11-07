//! The nagaya process with `LFM3-1.7B` or `LFM3-Omni-2B` resident on card 0 (milestone 24 step 9, spec 11.11): the
//! identity, a generation, and gogi's suite for the served model. This test covers text only; the media and the speech
//! of Omni have their own gates.
//!
//! It needs card 0, `LFM3_MODEL_DIR` and `LFM3_OMNI_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lfm3 -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::path::{Path, PathBuf};
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
use serde_json::{Value, json};
use tower::ServiceExt;

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

fn directory(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the model folder");
            None
        }
    }
}

/// One LFM3 model resident on device 0, with one slot, as step 9 puts it on b70host. The process serves one text
/// model now (spec 11.7), so each model gets its own process.
fn profile(name: &str, dir: &Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"{name}\"\nkind = \"mk-lfm3-bmg\"\ndevices = [0]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
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

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Run gogi's suite on one served model, and give its counts and the names of the failed checks.
fn suite(url: &str, model: &str) -> (usize, usize, usize, Vec<String>) {
    let report = run(
        Opts {
            url: url.to_owned(),
            restart: None,
            model: Some(model.to_owned()),
        },
        false,
    );
    let count = |status: Status| report.checks.iter().filter(|check| check.status == status).count();
    for check in report.checks.iter().filter(|check| check.status != Status::Ok) {
        println!("{model}: {:?} {}: {}", check.status, check.name, check.detail);
    }
    let failed = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| check.name.clone())
        .collect();
    (count(Status::Ok), count(Status::Fail), count(Status::Skip), failed)
}

#[test]
fn the_process_serves_lfm3_on_card_0() {
    if let Some(dir) = directory("LFM3_MODEL_DIR") {
        serve("lfm3", &dir);
    }
}

#[test]
fn the_process_serves_lfm3_omni_on_card_0() {
    if let Some(dir) = directory("LFM3_OMNI_MODEL_DIR") {
        serve("lfm3-omni", &dir);
    }
}

/// Start a process with the model `name` from `dir`, then check its identity, a generation and gogi's suite.
fn serve(name: &str, dir: &Path) {
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let profile = profile(name, dir);
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let start = Instant::now();
    let ready = lifecycle::start_with(
        profile.clone(),
        Registry::compiled_in(),
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens the engine");
    println!("opened in {} ms", start.elapsed().as_millis());
    let ports = ready.process.ports();
    let council = Arc::new(
        council(
            &FamilyRegistry::compiled_in(),
            &profile,
            &ready.opened,
            &ports,
            store.path(),
            &[],
            None,
        )
        .expect("the process serves the model")
        .with_loads(ports.clone()),
    );

    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK, "{identity}");
    let models: Vec<(String, String)> = identity["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| {
            (
                model["model"].as_str().unwrap().to_owned(),
                model["status"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    println!("served models: {models:?}");
    let names: Vec<String> = models.iter().map(|(name, _)| name.clone()).collect();
    assert_eq!(names, [name], "{identity}");
    assert!(models.iter().all(|(_, status)| status == "ready"), "{identity}");

    for model in &names {
        let started = Instant::now();
        let body = json!({"model": model, "context": {"messages": [{"role": "user", "content": "Name three colors."}]},
            "max_tokens": 48, "sample": {"temperature": 0}});
        let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
        assert_eq!(status, StatusCode::OK, "{model}: {generation}");
        println!(
            "{model}: generation in {} ms: {generation}",
            started.elapsed().as_millis()
        );
    }

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
    let url = format!("http://{}", server.local_addr());
    let mut failed = Vec::new();
    for model in &names {
        let (ok, fail, skip, names) = suite(&url, model);
        println!("gogi-conformance on {model}: {ok} ok, {fail} FAIL, {skip} skip");
        failed.extend(names.into_iter().map(|name| format!("{model}: {name}")));
    }
    assert!(failed.is_empty(), "gogi checks failed: {failed:?}");
}
