//! The nagaya process with Gemma 4 on the two B70 cards and EmbeddingGemma 2 pinned beside it on card 1 (milestone 19
//! step 6, spec 11.10): the embedding route, gogi's suite for both served models, and a generation of gemma beside.
//! The second test swaps Gemma 4 out for an engine on card 1 and back. The pinned engine stays loaded through both
//! swaps, and it gives the same vector (spec 3.3a, "No moves").
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
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, Server, router};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options};
use nagaya::notify::EngineState;
use nagaya::probe::FileProbe;
use nagaya::process::{Event, LoadReply};
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Limits};
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

/// A fake engine with `close` on card 1. It declares 30 GiB of its card in a budget of 32 GiB, so it fits beside
/// EmbeddingGemma 2 and not beside Gemma 4 (as `xe_swap.rs`).
struct Closing;

fn closing_limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    limits.memory.mandatory = vec![nagaya_abi::MandatoryTerm {
        name: "weights".to_owned(),
        tier: nagaya_abi::Tier::Device(0),
        bytes: 30 << 30,
        lifetime: nagaya_abi::Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic.clear();
    limits
}

struct Fake(FakeAlways);

impl Engine for Fake {
    fn identity(&self) -> nagaya_abi::Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        closing_limits()
    }

    fn pass(
        &mut self,
        segments: &[nagaya_abi::Segment],
        taps: &[nagaya_abi::Tap],
    ) -> Result<nagaya_abi::PassOut, nagaya_abi::PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    // A swap parks the states of the fake before its unload, so it forwards the calls that a park makes.
    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.0.coverage()
    }
}

impl Factory for Closing {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(closing_limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Fake(FakeAlways::new())))
    }
}

/// The next swap of the process, with its two lists of engine names.
fn swapped(ready: &lifecycle::Ready) -> (Vec<String>, Vec<String>) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Some(Event::Swapped { unloaded, loaded, .. }) = ready.process.recv(Duration::from_millis(50)) {
            return (unloaded, loaded.into_iter().map(|engine| engine.name).collect());
        }
    }
    panic!("no swap came in 120 s");
}

/// The vector of one query, as JSON numbers.
fn embed(council: &Arc<Council>, text: &str) -> Vec<f32> {
    let (status, body) = send(
        council,
        "POST",
        "/council/v2/embeddings",
        Some(json!({"model": EMBEDDER, "task": "query", "inputs": [text]})),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    vector(&body["embeddings"][0]["vector"])
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

/// The swaps of b70host on 2026-10-30: a load of an engine on card 1 that names Gemma 4, then a load of Gemma 4
/// that names that engine. The pinned EmbeddingGemma 2 stays ready, never reloads, and gives the same bits.
#[test]
fn a_swap_of_gemma4_out_and_back_keeps_the_pinned_embedder() {
    let (Some(gemma), Some(embedder)) = (directory("GEMMA4_MODEL_DIR"), directory("EMBEDDINGGEMMA2_MODEL_DIR")) else {
        return;
    };
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let mut text = profile(&gemma, &embedder).to_toml().expect("the profile writes");
    text.push_str(
        "\n[[engines]]\nname = \"draw\"\nkind = \"closing\"\ndevices = [1]\nshare = 1\nresidency = \"on_demand\"\n",
    );
    let profile = PlacementProfile::parse(&text).expect("the profile with `draw` parses");
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let mut registry = Registry::compiled_in();
    registry.register("closing", Box::new(Closing));
    let mut ready = lifecycle::start_with(
        profile.clone(),
        registry,
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens both engines");
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
    let query = "Which city is the capital of France?";
    let before = embed(&council, query);
    let ready_now = || ports.engine_state(EMBEDDER);

    // A load of `draw` that names gemma: the reply and the swap name only those two (spec 3.3a, "No moves").
    let started = Instant::now();
    let reply = ready
        .process
        .load_with("draw", &["gemma".to_owned()])
        .expect("a load of `draw`");
    assert_eq!(
        reply,
        LoadReply::Started {
            unload: vec!["gemma".to_owned()],
            load: vec!["draw".to_owned()],
        }
    );
    assert_eq!(
        ready_now(),
        Some(EngineState::Ready),
        "the pinned engine is ready during the swap"
    );
    assert_eq!(swapped(&ready), (vec!["gemma".to_owned()], vec!["draw".to_owned()]));
    println!("swap to draw in {} ms", started.elapsed().as_millis());
    assert_eq!(ready_now(), Some(EngineState::Ready));
    assert_eq!(embed(&council, query), before, "the same bits beside draw");

    // A load of gemma that names `draw`. An embedding request during the swap waits for the open, and runs.
    let started = Instant::now();
    let reply = ready
        .process
        .load_with("gemma", &["draw".to_owned()])
        .expect("a load of `gemma`");
    assert_eq!(
        reply,
        LoadReply::Started {
            unload: vec!["draw".to_owned()],
            load: vec!["gemma".to_owned()],
        }
    );
    let during = {
        let council = Arc::clone(&council);
        std::thread::spawn(move || {
            let started = Instant::now();
            (embed(&council, query), started.elapsed().as_millis())
        })
    };
    assert_eq!(
        ready_now(),
        Some(EngineState::Ready),
        "the pinned engine is ready during the swap"
    );
    assert_eq!(swapped(&ready), (vec!["draw".to_owned()], vec!["gemma".to_owned()]));
    println!("swap back to gemma in {} ms", started.elapsed().as_millis());
    let (vector_during, ms) = during.join().expect("the request during the swap");
    println!("an embedding request during the swap back took {ms} ms");
    assert_eq!(vector_during, before, "the same bits during the swap");
    assert_eq!(ready_now(), Some(EngineState::Ready));
    assert_eq!(embed(&council, query), before, "the same bits beside gemma again");
    let body = json!({"model": GEMMA, "context": {"messages": [{"role": "user", "content": "Say hello."}]},
        "max_tokens": 16, "sample": {"temperature": 0}});
    let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
    assert_eq!(status, StatusCode::OK, "{generation}");

    council.drain();
    let stopped = lifecycle::stop(&mut ready, None, Duration::from_secs(30), &SignalFlag::new());
    println!("stop: {}", stopped.log_line());
}
