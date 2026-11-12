//! Embeddings of a served embedding model in the process (spec 11.10, gogi "Embeddings"), with the fake embedding
//! engine of the ABI.
//!
//! The family of these tests reads bytes: the ids of an input are `<bos>`, each byte of the prompt and the text plus 3,
//! and `<eos>`. Thus a test knows the ids of each input, and the fake gives the vector of those ids.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gogi_proto::embeddings::{cut, norm};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::embedding::{EmbedFamily, EmbeddingConfig, EmbeddingService};
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeEmbedder;
use nagaya_abi::{Embedding, Engine, Identity, Limits, PassError, PassOut, Segment, Tap, Tier};
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "embedder";

/// The length of a vector of the fake, and the cuts that the service offers.
const DIMENSIONS: [u32; 3] = [8, 4, 2];

/// The most tokens of one input.
const CONTEXT_TOKENS: u64 = 64;

/// The fake embedding engine, which counts its calls of `embed`.
struct Counted {
    inner: FakeEmbedder,
    calls: Arc<AtomicUsize>,
}

impl Embedding for Counted {
    fn embedding_max_tokens(&self) -> u32 {
        self.inner.embedding_max_tokens()
    }

    fn embedding_dimension(&self) -> u32 {
        self.inner.embedding_dimension()
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, PassError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // An input with `~` gives a zero vector, which no cut can normalize. The family gives each byte its value
        // plus 3.
        if ids.contains(&(u32::from(b'~') + 3)) {
            return Ok(vec![0.0; DIMENSIONS[0] as usize]);
        }
        self.inner.embed(ids)
    }
}

impl Engine for Counted {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn embedding(&mut self) -> Option<&mut dyn Embedding> {
        Some(self)
    }
}

fn fake() -> FakeEmbedder {
    FakeEmbedder::new(DIMENSIONS[0], CONTEXT_TOKENS as u32, 100)
}

struct Embedders(Arc<AtomicUsize>);

impl Factory for Embedders {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(fake().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Counted {
            inner: fake(),
            calls: Arc::clone(&self.0),
        }))
    }
}

/// The family of bytes, with the prompts `q: ` and `d: `.
struct Bytes;

impl Bytes {
    fn prompt(task: &str) -> Result<&'static str, String> {
        match task {
            "query" => Ok("q: "),
            "document" => Ok("d: "),
            other => Err(format!("no task {other}")),
        }
    }
}

impl EmbedFamily for Bytes {
    fn tasks(&self) -> Vec<String> {
        vec!["query".to_owned(), "document".to_owned()]
    }

    fn ids(&self, task: &str, text: &str) -> Result<Vec<u32>, String> {
        let joined = format!("{}{text}", Self::prompt(task)?);
        let mut ids = vec![2];
        ids.extend(joined.bytes().map(|byte| u32::from(byte) + 3));
        ids.push(1);
        Ok(ids)
    }

    fn template(&self) -> String {
        "bytes-tasks:1".to_owned()
    }
}

struct Fixture {
    process: Process,
    calls: Arc<AtomicUsize>,
    _directory: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self::with(Residency::Resident)
    }

    fn with(residency: Residency) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = Registry::new();
        registry.register("embedders", Box::new(Embedders(Arc::clone(&calls))));
        let profile = PlacementProfile {
            engines: vec![EnginePlacement {
                name: MODEL.to_owned(),
                kind: "embedders".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                family: toml::Table::new(),
                lab: toml::Table::new(),
                share: 1,
                residency,
                pinned: residency == Residency::Resident,
            }],
            budgets: vec![BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1000,
            }],
            store: Some(StoreSpec {
                nvme: 1 << 20,
                host: None,
                interval_copies: 0,
                objects: None,
            }),
        };
        let inventory = Inventory {
            devices: vec![Device {
                id: 0,
                name: "device-0".to_owned(),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 4096,
                unified: false,
                ignored: false,
            }],
            host_bytes: 1 << 20,
        };
        let mut process = Start::new(profile, registry, inventory)
            .store(directory.path())
            .heartbeat(Duration::from_millis(10))
            .timeout(Duration::from_secs(5))
            .start()
            .unwrap();
        process.wait_opens(Duration::from_secs(5)).unwrap();
        Self {
            process,
            calls,
            _directory: directory,
        }
    }

    fn service(&self) -> EmbeddingService {
        let mut config = EmbeddingConfig::new(MODEL, MODEL);
        config.dimensions = DIMENSIONS.to_vec();
        config.context_tokens = CONTEXT_TOKENS;
        config.inputs_per_request = 4;
        config.timeout = Duration::from_secs(5);
        EmbeddingService::new(config, Arc::new(Bytes), self.process.ports()).unwrap()
    }

    fn council(&self) -> Arc<Council> {
        Arc::new(
            Council::new(Vec::new(), None)
                .unwrap()
                .with_embeddings(vec![self.service()])
                .with_loads(self.process.ports()),
        )
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        let body = match body {
            Some(body) => {
                request = request.header("content-type", "application/json");
                Body::from(body.to_string())
            }
            None => Body::empty(),
        };
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    })
}

fn embed(council: &Arc<Council>, body: Value) -> (StatusCode, Value) {
    send(council, "POST", "/council/v2/embeddings", Some(body))
}

fn vector(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// The vector of the fake for an input, cut to `d` and normalized as spec 11.10 says.
fn expected(task: &str, text: &str, d: usize) -> Vec<f32> {
    let ids = Bytes.ids(task, text).unwrap();
    cut(&fake().embed(&ids).unwrap(), d)
}

#[test]
fn identity_lists_the_served_embedding_model() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    let model = identity["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["model"] == json!(MODEL))
        .unwrap();
    assert_eq!(model["status"], json!("ready"), "{model}");
    assert_eq!(model["capabilities"], json!(["embed", "dry_run"]), "{model}");
    assert_eq!(
        model["embeddings"],
        json!({"tasks": ["query", "document"], "dimensions": [8, 4, 2], "inputs_per_request": 4}),
        "{model}"
    );
    assert_eq!(model["media"]["text_replies"], json!(false), "{model}");
    assert_eq!(model["template"], json!("bytes-tasks:1"), "{model}");
    assert_eq!(model["weight_hash"], json!("fake:embedder"), "{model}");
    assert_eq!(model["limits"]["context_tokens"], json!(CONTEXT_TOKENS), "{model}");
}

#[test]
fn each_input_gets_its_vector_cut_and_normalized_in_input_order() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, body) = embed(
        &council,
        json!({"model": MODEL, "task": "query", "inputs": ["pin a snapshot", "evict"], "dimensions": 4}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dimensions"], json!(4), "{body}");
    for (index, text) in ["pin a snapshot", "evict"].iter().enumerate() {
        let embedding = &body["embeddings"][index];
        assert_eq!(embedding["index"], json!(index), "{body}");
        let tokens = Bytes.ids("query", text).unwrap().len();
        assert_eq!(embedding["tokens"], json!(tokens), "{body}");
        let got = vector(&embedding["vector"]);
        assert_eq!(got, expected("query", text, 4), "input {index}");
        assert!((norm(&got) - 1.0).abs() < 1e-6, "{got:?}");
    }
    let tokens: u64 = ["pin a snapshot", "evict"]
        .iter()
        .map(|text| Bytes.ids("query", text).unwrap().len() as u64)
        .sum();
    assert_eq!(
        body["usage"],
        json!({"prompt": tokens, "kept": 0, "fed": tokens, "completion": 0, "reasoning": 0}),
        "{body}"
    );
    assert_eq!(
        body["identity"],
        json!({"model": MODEL, "weight_hash": "fake:embedder", "tokenizer_hash": "fake:bytes",
               "template": "bytes-tasks:1", "engine": MODEL}),
        "{body}"
    );
    assert_eq!(fixture.calls(), 2, "one quantum for each input");
}

/// The nagaya normalizes the vector at its full length too (spec 11.10), so a vector of the engine with no norm of 1
/// comes out with one.
#[test]
fn a_vector_at_its_full_length_is_normalized() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, body) = embed(&council, json!({"model": MODEL, "task": "document", "inputs": ["x"]}));
    assert_eq!(status, StatusCode::OK, "{body}");
    let raw = fake().embed(&Bytes.ids("document", "x").unwrap()).unwrap();
    assert!(
        (norm(&raw) - 1.0).abs() > 1e-3,
        "the fake gives no unit vector: {}",
        norm(&raw)
    );
    let got = vector(&body["embeddings"][0]["vector"]);
    assert!(
        (norm(&got) - 1.0).abs() < 1e-6,
        "the vector has a norm of {}",
        norm(&got)
    );
    assert_eq!(got, expected("document", "x", 8));
}

/// A zero vector has no direction, so the cut cannot normalize it. The request fails with a message that names the
/// input, and the process goes on (spec 11.10).
#[test]
fn a_zero_vector_fails_the_request_and_names_the_input() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, body) = embed(&council, json!({"model": MODEL, "task": "query", "inputs": ["x", "~"]}));
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("inputs[1]"), "{body}");
    assert!(council.failure().is_none(), "a zero vector does not stop the process");
    let (status, body) = embed(&council, json!({"model": MODEL, "task": "query", "inputs": ["x"]}));
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// The checks of gogi run before the 503 of a model that is not ready (spec 11.10), so a request that can never pass
/// gets its 400 at once.
#[test]
fn a_bad_request_to_an_unloaded_embedding_model_is_a_400() {
    let fixture = Fixture::with(Residency::OnDemand);
    let council = fixture.council();
    for (body, param) in [
        (json!({"model": MODEL, "task": "bogus", "inputs": ["x"]}), "task"),
        (
            json!({"model": MODEL, "task": "query", "inputs": ["x"], "dimensions": 3}),
            "dimensions",
        ),
    ] {
        let (status, answer) = embed(&council, body);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
        assert_eq!(answer["error"]["param"], json!(param), "{answer}");
    }
}

#[test]
fn a_dry_run_counts_the_tokens_and_runs_no_quantum() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let long = "y".repeat(CONTEXT_TOKENS as usize * 2);
    let (status, body) = embed(
        &council,
        json!({"model": MODEL, "task": "query", "inputs": ["x", long], "dry_run": true}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["embeddings"][0], json!({"index": 0, "tokens": 6}), "{body}");
    assert_eq!(
        body["embeddings"][1]["tokens"],
        json!(CONTEXT_TOKENS * 2 + 5),
        "a dry run counts an input past the limit"
    );
    assert_eq!(body["usage"]["fed"], json!(0), "{body}");
    assert_eq!(fixture.calls(), 0);
}

#[test]
fn an_input_past_context_tokens_is_a_413_and_nothing_is_embedded() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let long = "y".repeat(CONTEXT_TOKENS as usize);
    let (status, body) = embed(
        &council,
        json!({"model": MODEL, "task": "query", "inputs": ["x", long]}),
    );
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["type"], json!("too_large"), "{body}");
    assert_eq!(body["error"]["param"], json!("context_tokens"), "{body}");
    assert_eq!(body["error"]["context_tokens"], json!(CONTEXT_TOKENS), "{body}");
    assert_eq!(body["error"]["prompt_tokens"], json!(CONTEXT_TOKENS + 5), "{body}");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("inputs[1]"),
        "{body}"
    );
    assert_eq!(fixture.calls(), 0);
}

/// A model that only embeds takes no text route (gogi "Embeddings").
#[test]
fn a_build_that_names_the_embedding_model_is_a_400_that_names_model() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, body) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(
            json!({"model": MODEL, "context": {"thinking": false, "messages": [{"role": "user", "content": "hello"}]}}),
        ),
    );
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], json!("model"), "{body}");
}

#[test]
fn an_unloaded_embedding_model_is_a_503_and_a_load_serves_it() {
    let fixture = Fixture::with(Residency::OnDemand);
    let council = fixture.council();
    let (status, body) = embed(&council, json!({"model": MODEL, "task": "query", "inputs": ["x"]}));
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], json!("model_unavailable"), "{body}");
    assert_eq!(body["error"]["status"], json!("unloaded"), "{body}");
    fixture.process.load(MODEL).unwrap();
    let ports = fixture.process.ports();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while ports.engine_state(MODEL) != Some(nagaya::notify::EngineState::Ready) {
        assert!(std::time::Instant::now() < deadline, "the embedder did not load");
        std::thread::sleep(Duration::from_millis(5));
    }
    let (status, body) = embed(&council, json!({"model": MODEL, "task": "query", "inputs": ["x"]}));
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// A client that goes away drops its cancel token, and the job ends at its next quantum with no more `embed`.
#[test]
fn a_cancelled_job_runs_no_more_quanta() {
    let fixture = Fixture::new();
    let service = fixture.service();
    let flag = Arc::new(AtomicBool::new(true));
    let made = service
        .embed(vec![vec![2, 5, 1], vec![2, 6, 1]], Default::default(), flag)
        .unwrap();
    assert!(
        made.recv_timeout(Duration::from_secs(5)).is_err(),
        "a cancelled job tells nothing, and its sender drops"
    );
    assert_eq!(fixture.calls(), 0);
}

#[test]
fn the_gogi_suite_passes_for_the_embedding_model() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = runtime.spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(listener, router(council)).await.unwrap();
    });
    let url = format!("http://{address}");
    let report = std::thread::spawn(move || {
        gogi_conformance::run(
            gogi_conformance::Opts {
                url,
                restart: None,
                model: Some(MODEL.to_owned()),
            },
            true,
        )
    })
    .join()
    .unwrap();
    server.abort();
    assert_eq!(report.failures(), 0, "{}", report.summary());
}

/// The vectors of the job do not depend on the other inputs of the request, so a client can cache them by input.
#[test]
fn an_input_alone_gives_the_vector_that_it_gives_beside_others() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let pair = Mutex::new(Vec::new());
    for inputs in [json!(["a", "b"]), json!(["b"])] {
        let (status, body) = embed(&council, json!({"model": MODEL, "task": "query", "inputs": inputs}));
        assert_eq!(status, StatusCode::OK, "{body}");
        pair.lock()
            .unwrap()
            .push(body["embeddings"].as_array().unwrap().last().unwrap()["vector"].clone());
    }
    let pair = pair.into_inner().unwrap();
    assert_eq!(pair[0], pair[1]);
}
