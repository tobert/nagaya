//! Cycles through the council (spec 11.17, gogi "Cycles"): the served model declares the cycles of its engine, the
//! count that runs reaches each segment of a job, it changes the numbers, and the default count shares model state
//! with no count.

mod council_support;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, Fixture, Seen, TermFactory};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya_abi::CyclesLimit;
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// The cycles of the fake: 1 to 5 over layer 1 of its 2, 3 by default.
const LIMIT: CyclesLimit = CyclesLimit {
    min: 1,
    max: 5,
    default: 3,
    start: 1,
    end: 2,
};

/// A council over an engine with the cycles of `limit`, or none, and the log of the segments it saw.
fn council(limit: Option<CyclesLimit>) -> (Fixture, Arc<Council>, Seen) {
    let seen = Seen::default();
    let factory = TermFactory {
        cycles: limit,
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

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn context(cycles: Value) -> Value {
    let mut c = json!({"thinking": false, "messages": [{"role": "user", "content": "Plan the heist.", "snap": true}]});
    if !cycles.is_null() {
        c["cycles"] = cycles;
    }
    c
}

/// The letter log probabilities of a raw read of the context.
fn letters(council: &Arc<Council>, cycles: Value) -> Value {
    let body = json!({"model": "toy", "contexts": [context(cycles)], "suffixes": ["Answer: "], "letters": ["A", "B"]});
    let (status, r) = send(council, "POST", "/council/v2/reads", Some(body));
    assert_eq!(status, StatusCode::OK, "{r}");
    r["reads"][0]["logprobs"].clone()
}

/// The build of the context.
fn build(council: &Arc<Council>, cycles: Value) -> Value {
    let (status, r) = send(
        council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context(cycles)})),
    );
    assert_eq!(status, StatusCode::OK, "{r}");
    r
}

/// The counts of cycles of the segments that the engine saw since the last call.
fn counts(seen: &Seen) -> Vec<Option<u8>> {
    seen.take().iter().map(|s| s.cycles).collect()
}

/// Falsified by a served model that does not declare `cycles`, or whose member is not the engine's limit.
#[test]
fn the_served_model_declares_the_cycles_of_its_engine() {
    let (_f, cycled, _) = council(Some(LIMIT));
    let (status, identity) = send(&cycled, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    let model = &identity["models"][0];
    assert!(
        model["capabilities"].as_array().unwrap().contains(&json!("cycles")),
        "{model}"
    );
    assert_eq!(
        model["cycles"],
        json!({"min": 1, "max": 5, "default": 3, "start": 1, "end": 2})
    );
    let (_g, plain, _) = council(None);
    let (_, identity) = send(&plain, "GET", "/council/v2/identity", None);
    let model = &identity["models"][0];
    assert!(model.get("cycles").is_none(), "{model}");
    assert!(
        !model["capabilities"].as_array().unwrap().contains(&json!("cycles")),
        "{model}"
    );
    let (status, e) = send(
        &plain,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context(json!(3))})),
    );
    assert_eq!(
        (status, e["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("cycles")),
        "{e}"
    );
}

/// Falsified by a job that puts no count on a segment, a read or a generation that passes no count to its job, and a
/// segment that carries a count on an engine with no cycles.
#[test]
fn each_segment_carries_the_count_that_runs_and_the_count_changes_the_numbers() {
    let (_f, cycled, seen) = council(Some(LIMIT));
    let plain = letters(&cycled, Value::Null);
    let none = counts(&seen);
    assert!(!none.is_empty());
    assert!(
        none.iter().all(|c| *c == Some(3)),
        "no count runs the default: {none:?}"
    );
    let five = letters(&cycled, json!(5));
    let ran = counts(&seen);
    assert!(ran.iter().all(|c| *c == Some(5)), "{ran:?}");
    assert_ne!(five, plain, "the fake's numbers change at another count");
    assert_ne!(letters(&cycled, json!(1)), five, "each count is its own");
    assert_eq!(
        letters(&cycled, json!(3)),
        plain,
        "the default count gives the numbers of no count"
    );
    seen.take();
    let (status, r) = send(
        &cycled,
        "POST",
        "/council/v2/generations",
        Some(json!({"model": "toy", "context": context(json!(5)), "max_tokens": 4})),
    );
    assert_eq!(status, StatusCode::OK, "{r}");
    let decode = counts(&seen);
    assert!(
        decode.iter().all(|c| *c == Some(5)),
        "a generation's passes carry the count: {decode:?}"
    );
    let (_g, plain_council, plain_seen) = council(None);
    letters(&plain_council, Value::Null);
    let none = counts(&plain_seen);
    assert!(!none.is_empty() && none.iter().all(Option::is_none), "{none:?}");
}

/// Falsified by a recipe that leaves out the count, or one that hashes the default count, so that `cycles: 3` and no
/// `cycles` have two roots.
#[test]
fn the_default_count_shares_model_state_with_no_count_and_another_count_never_does() {
    let (_f, council, _) = council(Some(LIMIT));
    let plain = build(&council, Value::Null);
    let named = build(&council, json!(3));
    assert_eq!(named["head"], plain["head"], "{named}");
    assert_eq!(named["usage"]["fed"], json!(0), "{named}");
    assert_ne!(named["context"], plain["context"], "the records differ (gogi)");
    let five = build(&council, json!(5));
    assert_ne!(five["head"], plain["head"], "{five}");
    assert_eq!(five["usage"]["kept"], json!(0), "{five}");
    let again = build(&council, json!(5));
    assert_eq!(
        (&again["head"], &again["usage"]["fed"]),
        (&five["head"], &json!(0)),
        "{again}"
    );
}

/// Falsified by a service that runs a count out of the range: the engine sees a segment.
#[test]
fn a_count_out_of_the_range_is_refused_before_the_job() {
    let (_f, cycled, seen) = council(Some(LIMIT));
    seen.take();
    for count in [0, 6] {
        let (status, e) = send(
            &cycled,
            "POST",
            "/council/v2/contexts",
            Some(json!({"model": "toy", "context": context(json!(count))})),
        );
        assert_eq!(
            (status, e["error"]["param"].as_str()),
            (StatusCode::BAD_REQUEST, Some("cycles")),
            "{e}"
        );
        assert!(e["error"]["message"].as_str().unwrap().contains("1 to 5"), "{e}");
    }
    assert!(seen.take().is_empty(), "nothing ran");
}
