//! Layer loops through the council (spec 11.16, gogi "Layer loops"): the loop of a context reaches each segment of its
//! job, it changes the numbers, and a speech reply refuses it until a frame carries it.

mod council_support;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, Fixture, Seen, TermFactory};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya_abi::LayerLoop;
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// A council over an engine with a reserve of `runs` extra runs, or none, and the log of the segments it saw.
fn council(runs: Option<u32>) -> (Fixture, Arc<Council>, Seen) {
    let seen = Seen::default();
    let factory = TermFactory {
        layer_loops: runs,
        seen: Some(seen.clone()),
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    let service = Service::attach_with_clock(
        f.process.ports(),
        Arc::new(ToyFamily::new()),
        f.directory.path(),
        ServiceConfig::new("toy", ENGINE),
        f.reader(),
    )
    .unwrap();
    let served = Served {
        service,
        aliases: Vec::new(),
        device: Some("fake-0".to_owned()),
    };
    let council = Arc::new(Council::new(vec![served], None).unwrap());
    (f, council, seen)
}

fn send(council: &Arc<Council>, path: &str, body: Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn context(layer_loop: Value) -> Value {
    let mut c = json!({"thinking": false, "messages": [{"role": "user", "content": "Plan the heist.", "snap": true}]});
    if !layer_loop.is_null() {
        c["loop"] = layer_loop;
    }
    c
}

/// The letter log probabilities of a raw read of the context.
fn letters(council: &Arc<Council>, layer_loop: Value) -> Value {
    let body =
        json!({"model": "toy", "contexts": [context(layer_loop)], "suffixes": ["Answer: "], "letters": ["A", "B"]});
    let (status, r) = send(council, "/council/v2/reads", body);
    assert_eq!(status, StatusCode::OK, "{r}");
    r["reads"][0]["logprobs"].clone()
}

#[test]
fn the_served_model_declares_the_loops_of_its_engine() {
    let (_f, looped, _) = council(Some(4));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let identity: Value = runtime.block_on(async {
        let request = Request::builder()
            .uri("/council/v2/identity")
            .body(Body::empty())
            .unwrap();
        let response = router(Arc::clone(&looped)).oneshot(request).await.unwrap();
        serde_json::from_slice(&axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
    });
    let model = &identity["models"][0];
    assert!(
        model["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("layer_loops")),
        "{model}"
    );
    assert_eq!(
        model["layer_loops"],
        json!({"layers": 2, "bounds": [0, 1, 2], "count": 4, "extra_runs": 4})
    );
    let (_g, plain, _) = council(None);
    let (status, e) = send(
        &plain,
        "/council/v2/contexts",
        json!({"model": "toy", "context": context(json!({"start": 0, "end": 1, "count": 2}))}),
    );
    assert_eq!(
        (status, e["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("loop")),
        "{e}"
    );
}

#[test]
fn each_segment_of_a_looped_job_carries_the_loop_and_the_loop_changes_the_numbers() {
    let (_f, council, seen) = council(Some(4));
    let plain = letters(&council, Value::Null);
    seen.take();
    let twice = letters(&council, json!({"start": 0, "end": 2, "count": 2}));
    let segments = seen.take();
    assert!(!segments.is_empty());
    let want = Some(LayerLoop {
        start: 0,
        end: 2,
        count: 2,
    });
    assert!(
        segments.iter().all(|s| s.layer_loop == want),
        "{:?}",
        segments.iter().map(|s| s.layer_loop).collect::<Vec<_>>()
    );
    assert_ne!(twice, plain, "the fake's numbers change under a loop");
    assert_eq!(
        letters(&council, json!({"start": 0, "end": 2, "count": 1})),
        plain,
        "count 1 gives the plain numbers"
    );
    seen.take();
    let (status, r) = send(
        &council,
        "/council/v2/generations",
        json!({"model": "toy", "context": context(json!({"start": 0, "end": 2, "count": 2})), "max_tokens": 4}),
    );
    assert_eq!(status, StatusCode::OK, "{r}");
    let decode = seen.take();
    assert!(
        decode.iter().all(|s| s.layer_loop == want),
        "a generation's passes carry the loop: {:?}",
        decode
            .iter()
            .map(|s| (s.positions.clone(), s.layer_loop))
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_loop_past_the_reserve_is_refused_before_the_job() {
    let (_f, council, seen) = council(Some(2));
    seen.take();
    let (status, e) = send(
        &council,
        "/council/v2/contexts",
        json!({"model": "toy", "context": context(json!({"start": 0, "end": 2, "count": 3}))}),
    );
    assert_eq!(
        (status, e["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("loop")),
        "{e}"
    );
    assert!(seen.take().is_empty(), "nothing ran");
}
