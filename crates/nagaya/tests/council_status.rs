//! Each served model has the status of its engine (spec 3.3, 11.1). A request for a served model that is not `ready`
//! is a 503 `model_unavailable` that names `model` and `status`, and nothing runs.

mod council_support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, TermFactory, inventory, profile};
use nagaya::admission::Registry;
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya::notify::EngineState;
use nagaya::process::{Process, Start};
use nagaya::profile::Residency;
use nagaya_family_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

const SPARE: &str = "spare";

/// A process with the toy engine, resident, and an on-demand engine `spare` that does not fit beside it. A council
/// serves the toy engine as the model `toy`.
fn start(directory: &std::path::Path) -> (Process, Arc<Council>) {
    // The fake engine holds all 4,096 bytes of the device budget, so two engines do not fit together.
    let factory = || TermFactory {
        close: true,
        ..TermFactory::toy()
    };
    let mut registry = Registry::new();
    registry.register("term", Box::new(factory()));
    let mut profile = profile(1 << 20, 1 << 24, None);
    let mut spare = profile.engines[0].clone();
    spare.name = SPARE.to_owned();
    spare.residency = Residency::OnDemand;
    profile.engines.push(spare);
    let mut process = Start::new(profile, registry, inventory())
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(10))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(10)).unwrap();
    let service = Service::attach(
        process.ports(),
        Arc::new(ToyFamily::new()),
        directory,
        ServiceConfig::new("toy", ENGINE),
    )
    .unwrap();
    let council = Council::new(
        vec![Served {
            service,
            aliases: Vec::new(),
            device: None,
        }],
        None,
    )
    .unwrap();
    (process, Arc::new(council))
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        if body.is_some() {
            request = request.header("content-type", "application/json");
        }
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    })
}

fn status_of(council: &Arc<Council>) -> Value {
    let (status, identity) = send(council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    identity["models"][0]["status"].clone()
}

fn wait_for(process: &Process, engine: &str, state: EngineState) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while process.ports().engine_state(engine) != Some(state) {
        assert!(
            Instant::now() < deadline,
            "engine `{engine}` did not become {state:?} in 5 s"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn build() -> Value {
    json!({"model": "toy", "context": {"thinking": true, "messages": [{"role": "user", "content": "hello"}]}})
}

#[test]
fn a_text_model_whose_engine_unloads_is_unloaded_and_its_requests_are_a_503() {
    let directory = tempfile::tempdir().unwrap();
    let (process, council) = start(directory.path());
    assert_eq!(status_of(&council), json!("ready"));
    let (status, body) = send(&council, "POST", "/council/v2/contexts", Some(build()));
    assert_eq!(status, StatusCode::OK, "{body}");

    process.load_with(SPARE, &[ENGINE.to_owned()]).unwrap();
    wait_for(&process, ENGINE, EngineState::Unloaded);
    assert_eq!(status_of(&council), json!("unloaded"));
    let (status, body) = send(&council, "POST", "/council/v2/contexts", Some(build()));
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], json!("model_unavailable"), "{body}");
    assert_eq!(body["error"]["model"], json!("toy"), "{body}");
    assert_eq!(body["error"]["status"], json!("unloaded"), "{body}");

    // No unload loads another engine (spec 3.3a, "Explicit layout"), so the client loads the text engine again.
    process.unload(SPARE).unwrap();
    wait_for(&process, SPARE, EngineState::Unloaded);
    process.load(ENGINE).unwrap();
    wait_for(&process, ENGINE, EngineState::Ready);
    assert_eq!(status_of(&council), json!("ready"));
    let (status, body) = send(&council, "POST", "/council/v2/contexts", Some(build()));
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[test]
fn a_drain_makes_every_model_draining_whatever_its_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (_process, council) = start(directory.path());
    council.drain();
    assert_eq!(status_of(&council), json!("draining"));
}
