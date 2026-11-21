//! The nagaya process with an Inkling 2 model resident on one card (milestone 35 step 6, spec 11.18): the description
//! of the factory against the open, the identity, a generation at the default count and at another count, and gogi's
//! suite for the served model with its checks of cycles.
//!
//! It needs one card of `NAGAYA_TEST_CARDS`, and `INKLING2_MODEL_DIR` or `INKLING2_VARLOOP_MODEL_DIR`. The card must be
//! free: release it from the service first. Run it alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test --release -p nagaya --features xe --test xe_inkling2 -- --test-threads=1
//! --nocapture`.
#![cfg(feature = "xe")]

mod card_support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gogi_conformance::{Opts, Status, run};
use nagaya::admission::{Registry, group_devices};
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, Server, router};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options};
use nagaya::probe::FileProbe;
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use serde_json::{Value, json};
use tower::ServiceExt;

/// The iGPU of the B70 machine drives the display. Its config ignores it (spec 2.1, `ignore_devices`).
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

/// The config of the engine: one slot of 32,768 positions and the reserve of 5 cycles, so the suite runs each count.
fn engine_config(dir: &Path) -> String {
    format!("model_dir = \"{}\"\nslots = 1\nextra_runs = 48\n", dir.display())
}

/// One Inkling 2 model resident on the card of the test, with its family table and an object store.
fn profile(name: &str, dir: &Path) -> PlacementProfile {
    let card = card_support::device();
    let config = engine_config(dir);
    let family = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"{name}\"\nkind = \"mk-inkling2-bmg\"\ndevices = [{card}]\nshare = 1\n\
         [engines.config]\n{config}\
         [engines.family]\nmodel_dir = \"{family}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 4294967296\n"
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

/// Run gogi's suite on one served model, and give its counts and the names of the failed checks. Each check of
/// cycles prints with its status.
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
    let cycles: Vec<_> = report
        .checks
        .iter()
        .filter(|check| check.name.contains("cycles") || check.name.contains("count"))
        .collect();
    for check in &cycles {
        println!("{model}: cycles check {:?}: {}", check.status, check.name);
    }
    let cycles_ok = cycles.iter().filter(|check| check.status == Status::Ok).count();
    println!("{model}: {cycles_ok} of {} checks of cycles passed", cycles.len());
    let failed = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| check.name.clone())
        .collect();
    (count(Status::Ok), count(Status::Fail), count(Status::Skip), failed)
}

#[test]
fn the_process_serves_inkling2_mini_on_its_card() {
    if let Some(dir) = directory("INKLING2_MODEL_DIR") {
        serve("inkling2-mini", &dir);
    }
}

#[test]
fn the_process_serves_the_research_model_on_its_card() {
    if let Some(dir) = directory("INKLING2_VARLOOP_MODEL_DIR") {
        serve("inkling2-varloop", &dir);
    }
}

/// Start a process with the model `name` from `dir`, then check the description, the identity, two generations and
/// gogi's suite.
fn serve(name: &str, dir: &Path) {
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let profile = profile(name, dir);
    let devices = group_devices(&profile.engines[0], &inventory).expect("the card of the test");
    // The description of an engine on demand gives the identity and the limits of its open, with no load (spec 11.1).
    let registry = Registry::compiled_in();
    let started = Instant::now();
    let config: toml::Table = engine_config(dir).parse().unwrap();
    let (described, described_limits) = registry
        .factory(nagaya::inkling2::KIND)
        .expect("the kind of the xe build")
        .describe(&config, &devices)
        .expect("a description")
        .expect("the kind describes");
    println!("described in {} ms", started.elapsed().as_millis());

    let store = tempfile::tempdir().unwrap();
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let start = Instant::now();
    let ready = lifecycle::start_with(
        profile.clone(),
        registry,
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens the engine");
    println!("opened in {} ms", start.elapsed().as_millis());
    assert_eq!(
        ready.opened[0].identity, described,
        "the open gives the identity of the description"
    );
    assert_eq!(
        ready.opened[0].limits, described_limits,
        "the open gives the limits of the description"
    );
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
    let model = &identity["models"][0];
    assert_eq!(identity["models"].as_array().unwrap().len(), 1, "{identity}");
    assert_eq!(model["model"], json!(name), "the served model is the engine name");
    assert_eq!(model["status"], json!("ready"), "{identity}");
    println!("{name}: cycles {}", model["cycles"]);
    assert_eq!(model["cycles"]["min"], json!(1), "{identity}");
    assert_eq!(model["cycles"]["max"], json!(5), "{identity}");
    assert_eq!(model["cycles"]["default"], json!(3), "{identity}");

    for cycles in [None, Some(5)] {
        let started = Instant::now();
        let mut context = json!({"thinking": false, "messages": [{"role": "user", "content": "Name three colors."}]});
        if let Some(count) = cycles {
            context["cycles"] = json!(count);
        }
        let body = json!({"model": name, "context": context, "max_tokens": 48, "sample": {"temperature": 0}});
        let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
        assert_eq!(status, StatusCode::OK, "{name}: {generation}");
        println!(
            "{name} at {cycles:?} cycles: generation in {} ms: {generation}",
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
    let (ok, fail, skip, failed) = suite(&url, name);
    println!("gogi-conformance on {name}: {ok} ok, {fail} FAIL, {skip} skip");
    assert!(failed.is_empty(), "gogi checks failed: {failed:?}");
}
