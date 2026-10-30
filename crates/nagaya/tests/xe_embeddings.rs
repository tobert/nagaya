//! The nagaya process with Gemma 4 on the two B70 cards and EmbeddingGemma 2 pinned beside it on card 1 (milestone 19
//! step 6, spec 11.10): the embedding route, gogi's suite for both served models, and a generation of gemma beside.
//!
//! It needs both cards, `GEMMA4_MODEL_DIR` and `EMBEDDINGGEMMA2_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_embeddings -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gogi_conformance::{Opts, Status, run};
use gogi_proto::embeddings::norm;
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

const GEMMA: &str = "gemma-4-26b-a4b";
const EMBEDDER: &str = "embeddinggemma-2";

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

fn directory(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the checkpoint")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the checkpoint directory");
            None
        }
    }
}

/// Gemma 4 on devices 0 and 1, and EmbeddingGemma 2 resident and pinned on device 1, as on b70host.
fn profile(gemma: &std::path::Path, embedder: &std::path::Path) -> PlacementProfile {
    let (gemma, embedder) = (gemma.display(), embedder.display());
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"gemma\"\nkind = \"mk-gemma4-bmg\"\ndevices = [0, 1]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{gemma}\"\nslots = [4096, 16384]\n\
         [engines.family]\nmodel_dir = \"{gemma}\"\n\
         [[engines]]\nname = \"{EMBEDDER}\"\nkind = \"mk-embeddinggemma2-bmg\"\ndevices = [1]\nshare = 1\n\
         pinned = true\n\
         [engines.config]\nmodel_dir = \"{embedder}\"\n\
         [engines.family]\nmodel_dir = \"{embedder}\"\n\
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

fn vector(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
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
fn the_process_serves_embeddings_beside_gemma4() {
    let (Some(gemma), Some(embedder)) = (directory("GEMMA4_MODEL_DIR"), directory("EMBEDDINGGEMMA2_MODEL_DIR")) else {
        return;
    };
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let profile = profile(&gemma, &embedder);
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
    .expect("the process opens both engines");
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
        .expect("the process serves gemma")
        .with_embeddings(nagaya::embeddinggemma2::embedding_services(&profile, &ports).expect("the family opens"))
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
    assert!(
        models.contains(&(EMBEDDER.to_owned(), "ready".to_owned())),
        "{identity}"
    );

    let texts = [
        "Which city is the capital of France?",
        "Paris is the capital and largest city of France.",
        "The mitochondria is the powerhouse of the cell.",
    ];
    let started = Instant::now();
    let (status, query) = send(
        &council,
        "POST",
        "/council/v2/embeddings",
        Some(json!({"model": EMBEDDER, "task": "query", "inputs": [texts[0]]})),
    );
    assert_eq!(status, StatusCode::OK, "{query}");
    let (status, documents) = send(
        &council,
        "POST",
        "/council/v2/embeddings",
        Some(json!({"model": EMBEDDER, "task": "document", "inputs": [texts[1], texts[2]]})),
    );
    assert_eq!(status, StatusCode::OK, "{documents}");
    println!("three embeddings in {} ms", started.elapsed().as_millis());
    let q = vector(&query["embeddings"][0]["vector"]);
    let near = vector(&documents["embeddings"][0]["vector"]);
    let far = vector(&documents["embeddings"][1]["vector"]);
    assert_eq!(q.len(), 768);
    assert!((norm(&q) - 1.0).abs() < 1e-5, "{}", norm(&q));
    let (near, far) = (dot(&q, &near), dot(&q, &far));
    println!("cosine of the query: {near:.4} to the answer, {far:.4} to the other text");
    assert!(near > far + 0.1, "the answer is nearer the query: {near} and {far}");

    // Gemma 4 runs beside the pinned embedding engine.
    let started = Instant::now();
    let body = json!({"model": GEMMA, "context": {"messages": [{"role": "user", "content": "Say hello."}]},
        "max_tokens": 16, "sample": {"temperature": 0}});
    let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
    assert_eq!(status, StatusCode::OK, "{generation}");
    println!("generation in {} ms", started.elapsed().as_millis());

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
    let (ok, fail, skip, embed_failed) = suite(&url, EMBEDDER);
    println!("gogi-conformance on {EMBEDDER}: {ok} ok, {fail} FAIL, {skip} skip");
    let (ok, fail, skip, gemma_failed) = suite(&url, GEMMA);
    println!("gogi-conformance on {GEMMA}: {ok} ok, {fail} FAIL, {skip} skip");
    server.stop(Duration::from_secs(2));
    council.drain();
    let stopped = lifecycle::stop(&mut ready, None, Duration::from_secs(30), &SignalFlag::new());
    println!("stop: {}", stopped.log_line());
    assert!(embed_failed.is_empty(), "{EMBEDDER}: {embed_failed:?}");
    assert!(gemma_failed.is_empty(), "{GEMMA}: {gemma_failed:?}");
}
