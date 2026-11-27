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
use nagaya_family_toy::ToyFamily;
use nagaya_store::objects::ObjectStore;
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

/// A council on the toy with `segments` segments in a pass. With more than one, the toy joins its decode rows, as the
/// Qwen3.8 engine does: it declares `segments_exact` and no feed rule. The toy runs layer loops with a reserve of 4,
/// and 1 to 5 cycles.
fn lab_of(segments: u32) -> Lab {
    let log = ControlLog::default();
    let factory = TermFactory {
        seal: false,
        log: log.clone(),
        segments,
        layer_loops: Some(4),
        cycles: Some(nagaya_abi::CyclesLimit {
            min: 1,
            max: 5,
            default: 3,
            start: 1,
            end: 2,
        }),
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

/// An ensemble takes a context that names a `parent`, and it holds the record of each context that it reads
/// (`docs/lab.md`, "Context records in the lab routes"). It names no record.
///
/// Falsified by a route with no `whole`: the 400 of the unknown member `parent`. Also by an ensemble that holds no
/// record: the second request is a 404.
#[test]
fn an_ensemble_takes_a_parent_and_holds_the_record_of_each_context() {
    let lab = lab();
    let whole = context("Tell the tale of the dragon.");
    let (status, first) = send(&lab.council, PATH, &ensemble(vec![whole.clone()], "logprob_sum"));
    assert_eq!(status, StatusCode::OK, "{first}");
    assert!(first.get("context").is_none(), "an ensemble names no record: {first}");
    let record: gogi_proto::Context = serde_json::from_value(whole).unwrap();
    let parent = gogi_proto::context_id(&record).to_string();
    let (status, again) = send(
        &lab.council,
        PATH,
        &ensemble(vec![json!({"parent": parent})], "logprob_sum"),
    );
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(again["content"], first["content"]);
    assert_eq!(again["usage"]["prompt"], first["usage"]["prompt"]);
    let gone = format!("blake3:{}", "0".repeat(64));
    let (status, e) = send(
        &lab.council,
        PATH,
        &ensemble(vec![json!({"parent": gone})], "logprob_sum"),
    );
    assert_eq!(
        (status, e["error"]["param"].as_str()),
        (StatusCode::NOT_FOUND, Some("contexts[0].parent")),
        "{e}"
    );
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
        // The rows of the lead shared each pass of a step, so the reply holds no decoded head (spec 6.14). The store
        // holds the boundary at the end of each context, and nothing more.
        let ports = lab._process.ports();
        let entries = ports.store().unwrap().lock().unwrap().entries().len();
        assert_eq!(entries, texts.len(), "{stages} stages");
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

/// Forks on an engine that declares `segments_exact` and no feed rule, as the Qwen3.8 engine does, join their decode
/// rows (spec 5.7, "The first joins of an engine with `segments_exact`").
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

/// On a model with no batched passes, each row of a step runs in a pass of its own (docs/lab.md, "The passes of a
/// step").
#[test]
fn several_contexts_on_a_model_with_no_batched_passes_run_a_pass_for_each_row() {
    let lab = lab();
    let two = vec![
        context("Tell the tale of the dragon."),
        context("Tell the tale of the moon."),
    ];
    let (status, got) = send(&lab.council, PATH, &ensemble(two, "prob_mean"));
    assert_eq!(status, StatusCode::OK, "{got}");
    let completion = got["usage"]["completion"].as_u64().unwrap() as usize;
    assert!(completion >= 2, "{got}");
    let sizes = lab.log.take().sizes;
    assert!(sizes.iter().all(|&n| n == 1), "passes: {sizes:?}");
}

/// An ensemble over counts of cycles: the contexts take other counts, and at each step the rows of each count run in
/// a pass of their own (spec 11.17, docs/lab.md "Counts of cycles"). The toy joins its decode rows, so the two rows of
/// count 1 share a pass, and the row of count 5 has its own.
#[test]
fn an_ensemble_over_counts_of_cycles_runs_each_count_in_a_pass_of_its_own() {
    let lab = lab_of(8);
    let counted = |cycles: u8| {
        let mut c = context("Tell the tale of the dragon.");
        c["cycles"] = json!(cycles);
        c
    };
    let body = ensemble(vec![counted(1), counted(5), counted(1)], "prob_mean");
    let (status, got) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "{got}");
    let completion = got["usage"]["completion"].as_u64().unwrap() as usize;
    assert!(completion >= 2, "{got}");
    let log = lab.log.take();
    // Each step after the first row: one pass of the two rows of count 1, and one pass of the row of count 5.
    let ones = log.cycles.iter().filter(|c| *c == &vec![Some(1), Some(1)]).count();
    assert_eq!(ones, completion - 1, "passes: {:?}", log.cycles);
    for pass in &log.cycles {
        assert!(
            pass.windows(2).all(|w| w[0] == w[1]),
            "a pass runs one count: {:?}",
            log.cycles
        );
    }
    // The prompt of count 5 feeds alone too, so its passes are one more than the steps.
    let fives = log.cycles.iter().filter(|c| *c == &vec![Some(5)]).count();
    assert!(fives > completion - 1, "passes: {:?}", log.cycles);
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
        // The contexts of an ensemble share a pass, and a pass runs one loop (spec 11.16).
        ("contexts[1].loop", {
            let mut looped = one();
            looped["loop"] = json!({"start": 0, "end": 2, "count": 2});
            ensemble(vec![one(), looped], "logprob_sum")
        }),
        // A count past the limit of the engine (spec 11.17).
        ("contexts[1].cycles", {
            let mut counted = one();
            counted["cycles"] = json!(6);
            ensemble(vec![one(), counted], "logprob_sum")
        }),
    ];
    for (member, body) in cases {
        let (status, got) = send(&lab.council, PATH, &body);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{member}: {got}");
        assert_eq!(got["error"]["param"], member, "{member}: {got}");
    }
}
