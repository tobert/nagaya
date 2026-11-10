//! The lab route `ensemble` (`docs/lab.md`, "`ensemble`: joined forks") with the fake engine and the toy family. One
//! context gives the response of the same generation, K contexts decode in passes of K segments, and each fault is a
//! 400 that names its member.

mod council_support;
mod feed_support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ControlLog, ENGINE, Fixture, TermFactory};
use feed_support::{FeedFactory, FeedFixture};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya_store::objects::ObjectStore;
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

const PATH: &str = "/nagaya/v1/lab/ensemble";

struct Lab {
    _fixture: Fixture,
    council: Arc<Council>,
    log: ControlLog,
    _folder: tempfile::TempDir,
}

fn lab() -> Lab {
    lab_of(1)
}

/// A council on the toy with `segments` segments in a pass. With more than one, the toy joins its decode rows, as mk
/// does: it declares `segments_exact` and no feed rule.
fn lab_of(segments: u32) -> Lab {
    let log = ControlLog::default();
    let factory = TermFactory {
        seal: false,
        log: log.clone(),
        segments,
        ..TermFactory::toy()
    };
    let fixture = Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    let service = Service::attach_with_clock(
        fixture.process.ports(),
        Arc::new(ToyFamily::new()),
        fixture.directory.path(),
        ServiceConfig::new("toy", ENGINE),
        fixture.reader(),
    )
    .unwrap();
    let served = Served {
        service,
        aliases: Vec::new(),
        device: Some("fake-0".to_owned()),
    };
    let dir = tempfile::tempdir().unwrap();
    let store = ObjectStore::open(&fixture.directory.path().join("objects"), 1 << 24).unwrap();
    let council = Council::new(vec![served], None)
        .unwrap()
        .with_objects(store)
        .with_lab(BTreeMap::from([(ENGINE.to_owned(), dir.path().to_path_buf())]));
    Lab {
        _fixture: fixture,
        council: Arc::new(council),
        log,
        _folder: dir,
    }
}

/// A council on the fake with a feed rule and 8 segments, whose decode rows share passes (milestone 27).
struct FeedLab {
    council: Arc<Council>,
    sizes: Arc<Mutex<Vec<usize>>>,
    _process: nagaya::process::Process,
    _directory: tempfile::TempDir,
    _folder: tempfile::TempDir,
}

fn feed_lab(stages: usize) -> FeedLab {
    let factory = FeedFactory::new(stages);
    let sizes = Arc::clone(&factory.sizes);
    let FeedFixture {
        service,
        process,
        directory,
        ..
    } = FeedFixture::with(factory, Duration::from_secs(10));
    let served = Served {
        service,
        aliases: Vec::new(),
        device: Some("fake-0".to_owned()),
    };
    let dir = tempfile::tempdir().unwrap();
    let council = Council::new(vec![served], None).unwrap().with_lab(BTreeMap::from([(
        feed_support::ENGINE.to_owned(),
        dir.path().to_path_buf(),
    )]));
    FeedLab {
        council: Arc::new(council),
        sizes,
        _process: process,
        _directory: directory,
        _folder: dir,
    }
}

fn send(council: &Arc<Council>, path: &str, body: &Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn context(text: &str) -> Value {
    json!({"thinking": false, "messages": [{"role": "user", "content": text}]})
}

const SAMPLE: &str = r#"{"temperature": 0.8, "seed": 7}"#;

fn sample() -> Value {
    serde_json::from_str(SAMPLE).unwrap()
}

fn ensemble(contexts: Vec<Value>, combine: &str) -> Value {
    json!({"model": "toy", "contexts": contexts, "combine": combine, "max_tokens": 16, "sample": sample()})
}

#[test]
fn one_context_gives_the_response_and_the_passes_of_the_same_generation() {
    let plain = lab();
    let generation = json!({"model": "toy", "context": context("Tell the tale of the dragon."), "max_tokens": 16,
                            "sample": sample()});
    let (status, want) = send(&plain.council, "/council/v2/generations", &generation);
    assert_eq!(status, StatusCode::OK, "{want}");
    let generated = plain.log.take();

    let joined = lab();
    let body = ensemble(vec![context("Tell the tale of the dragon.")], "logprob_sum");
    let (status, got) = send(&joined.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "{got}");
    assert_eq!(got["content"], want["content"]);
    assert_eq!(got["finish"], want["finish"]);
    for member in ["prompt", "fed", "completion"] {
        assert_eq!(got["usage"][member], want["usage"][member], "usage.{member}");
    }
    assert_eq!(joined.log.take().sizes, generated.sizes, "the same passes");
}

#[test]
fn k_contexts_decode_in_passes_of_k_segments() {
    for stages in [1, 2] {
        let lab = feed_lab(stages);
        let texts = [
            "Tell the tale of the dragon.",
            "Tell the tale of the moon.",
            "Tell the tale of the sea.",
        ];
        let body = ensemble(texts.iter().map(|t| context(t)).collect(), "prob_mean");
        let (status, got) = send(&lab.council, PATH, &body);
        assert_eq!(status, StatusCode::OK, "{stages} stages: {got}");
        let completion = got["usage"]["completion"].as_u64().unwrap() as usize;
        assert!(completion >= 2, "{got}");
        let sizes = std::mem::take(&mut *lab.sizes.lock().unwrap());
        // Each prompt feeds alone. Each step after the first row is one pass of the three rows.
        let joined = sizes.iter().filter(|&&n| n == 3).count();
        assert_eq!(joined, completion - 1, "{stages} stages, passes: {sizes:?}");
        assert!(
            sizes.iter().all(|&n| n == 1 || n == 3),
            "{stages} stages, passes: {sizes:?}"
        );
    }
}

/// Forks on an engine that declares `segments_exact` and no feed rule, as mk does, join their decode rows (spec 5.7,
/// "The first joins of an engine with `segments_exact`").
#[test]
fn k_contexts_on_an_engine_with_segments_exact_decode_in_passes_of_k_segments() {
    let lab = lab_of(8);
    let texts = [
        "Tell the tale of the dragon.",
        "Tell the tale of the moon.",
        "Tell the tale of the sea.",
    ];
    let body = ensemble(texts.iter().map(|t| context(t)).collect(), "prob_mean");
    let (status, got) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "{got}");
    let completion = got["usage"]["completion"].as_u64().unwrap() as usize;
    assert!(completion >= 2, "{got}");
    let sizes = lab.log.take().sizes;
    // Each prompt feeds alone. Each step after the first row is one pass of the three rows.
    let joined = sizes.iter().filter(|&&n| n == 3).count();
    assert_eq!(joined, completion - 1, "passes: {sizes:?}");
    assert!(sizes.iter().all(|&n| n == 1 || n == 3), "passes: {sizes:?}");
}

/// Equal contexts give equal rows, and each rule then gives a monotone map of that row: `2 l`, `l` and `l`. Thus the
/// greedy replies agree. The reply of one context alone is another: the fake's bits depend on the token count of a
/// pass (D11), so a pass of two rows gives other rows than a pass of one. The card gate compares with one context,
/// because an LFM3 row does not depend on its pass (milestone 27).
#[test]
fn equal_contexts_give_one_greedy_reply_by_each_rule() {
    let lab = feed_lab(1);
    let greedy = |combine: &str| {
        let contexts = vec![
            context("Tell the tale of the dragon."),
            context("Tell the tale of the dragon."),
        ];
        let mut body = ensemble(contexts, combine);
        body["sample"] = json!({"temperature": 0, "seed": 1});
        let (status, got) = send(&lab.council, PATH, &body);
        assert_eq!(status, StatusCode::OK, "{combine}: {got}");
        got["content"].clone()
    };
    let sum = greedy("logprob_sum");
    assert!(sum.as_str().unwrap().chars().count() >= 2, "{sum}");
    assert_eq!(greedy("prob_mean"), sum);
    assert_eq!(greedy("contrast"), sum, "a contrast of two equal rows is the row");
}

#[test]
fn several_contexts_on_a_model_with_no_batched_passes_are_a_400_that_names_model() {
    let lab = lab();
    let two = vec![
        context("Tell the tale of the dragon."),
        context("Tell the tale of the moon."),
    ];
    let (status, got) = send(&lab.council, PATH, &ensemble(two, "prob_mean"));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    assert_eq!(got["error"]["param"], "model", "{got}");
    assert!(lab.log.take().passes.is_empty(), "a refused ensemble runs no pass");
}

#[test]
fn each_fault_is_a_400_that_names_its_member() {
    let lab = lab();
    let one = || context("Tell the tale of the dragon.");
    let cases = [
        ("contexts", ensemble(vec![], "logprob_sum")),
        ("contexts", ensemble((0..9).map(|_| one()).collect(), "logprob_sum")),
        ("combine", ensemble(vec![one(), one(), one()], "contrast")),
        ("combine", ensemble(vec![one()], "vote")),
        ("weights[1]", {
            let mut b = ensemble(vec![one(), one()], "prob_mean");
            b["weights"] = json!([1.0, 0.0]);
            b
        }),
        ("weights", {
            let mut b = ensemble(vec![one(), one()], "prob_mean");
            b["weights"] = json!([1.0]);
            b
        }),
        ("model", {
            let mut b = ensemble(vec![one()], "logprob_sum");
            b["model"] = json!("nope");
            b
        }),
    ];
    for (member, body) in cases {
        let (status, got) = send(&lab.council, PATH, &body);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{member}: {got}");
        assert_eq!(got["error"]["param"], member, "{member}: {got}");
    }
}
