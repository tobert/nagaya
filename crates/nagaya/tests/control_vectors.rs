//! Control vectors through council v2 with the fake engine and the toy family (spec 11.12, engine-abi "Control
//! vectors", gogi "Control vectors"): the capability, the route that makes a vector, the control list of a context, its
//! root recipe, and the worker of the control table.

mod council_support;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ControlLog, ENGINE, Fixture, TermFactory};
use gogi_proto::control::{ControlVector, MEDIA_TYPE};
use gogi_proto::identity::{Capability, ServerIdentity};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{CONTROL_VECTOR_CONTEXTS, Council, Served, router};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ControlOp, Engine, OutputsWanted, Segment, SegmentInputs, State, Tap};
use nagaya_council::family::Family;
use nagaya_store::objects::ObjectStore;
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// A fixture whose engine has a control table of `tables` entries, or none, and the log of what the engine saw.
fn fixture(tables: Option<u32>) -> (Fixture, ControlLog) {
    let log = ControlLog::default();
    let factory = TermFactory {
        control: tables,
        log: log.clone(),
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    (f, log)
}

/// A council with a second service on the fixture's process, with `config`, and an object store when `objects`.
fn council_with(f: &Fixture, config: ServiceConfig, objects: bool) -> Arc<Council> {
    let service = Service::attach_with_clock(
        f.process.ports(),
        Arc::new(ToyFamily::new()),
        f.directory.path(),
        config,
        f.reader(),
    )
    .unwrap();
    let served = Served {
        service,
        aliases: Vec::new(),
        device: Some("fake-0".to_owned()),
    };
    let council = Council::new(vec![served], None).unwrap();
    let council = match objects {
        true => council.with_objects(ObjectStore::open(&f.directory.path().join("objects"), 1 << 24).unwrap()),
        false => council,
    };
    Arc::new(council)
}

fn council(f: &Fixture) -> Arc<Council> {
    council_with(f, ServiceConfig::new("toy", ENGINE), true)
}

/// Send one request through the router. Give the status and the body, as JSON when it is JSON.
fn send_raw(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: Vec<u8>,
    mime: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        if let Some(mime) = mime {
            request = request.header("content-type", mime);
        }
        let response = router(Arc::clone(council))
            .oneshot(request.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn send(council: &Arc<Council>, path: &str, body: Value) -> (StatusCode, Value) {
    let (status, bytes) = send_raw(council, "POST", path, body.to_string().into_bytes(), None);
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Put an object, and give its id.
fn put(council: &Arc<Council>, bytes: &[u8], mime: &str) -> String {
    let id = gogi_proto::content_id(bytes);
    let (status, body) = send_raw(
        council,
        "PUT",
        &format!("/council/v2/objects/{id}"),
        bytes.to_vec(),
        Some(mime),
    );
    assert!(status.is_success(), "{status}: {}", String::from_utf8_lossy(&body));
    id
}

/// A vector of the fake's weights: 2 layers of 4 values.
fn vector(weight_hash: &str, layers: &[(u32, Vec<f32>)]) -> Vec<u8> {
    ControlVector {
        model: "toy".to_owned(),
        weight_hash: weight_hash.to_owned(),
        layers: layers.iter().cloned().collect(),
    }
    .to_bytes()
}

fn said(reply: &str) -> Value {
    json!({"thinking": false, "messages": [
        {"role": "user", "content": "Write the scene where the dragon wins.", "snap": true},
        {"role": "assistant", "content": reply}]})
}

fn steered(text: &str, control: Value) -> Value {
    json!({"thinking": false, "control": control,
           "messages": [{"role": "user", "content": text, "snap": true}]})
}

fn item(vector: &str, op: &str, strength: f64) -> Value {
    json!({"vector": vector, "op": op, "strength": strength})
}

/// The residual row of the last token of `context` on a fresh fake, rendered with the generation prompt.
fn direct_row(context: &Value) -> Vec<f32> {
    let context: gogi_proto::context::Context = serde_json::from_value(context.clone()).unwrap();
    let tokens = ToyFamily::new().render(&context, true).unwrap().tokens;
    let mut engine = FakeAlways::with_slots(council_support::roomy_slots(1))
        .with_vocab(gogi_toy::VOCAB)
        .with_control(4);
    let n = tokens.len() as u64;
    // The fake takes 64 tokens in a pass, and its bits do not depend on the split.
    let mut out = None;
    for from in (0..n).step_by(64) {
        let to = n.min(from + 64);
        let last = to == n;
        let taps = match last {
            true => vec![Tap {
                name: "residual".to_owned(),
            }],
            false => Vec::new(),
        };
        let segment = Segment {
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(tokens[from as usize..to as usize].to_vec()),
            positions: (from..to).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if last { vec![n - 1] } else { Vec::new() },
            },
            control: Vec::new(),
        };
        out = Some(engine.pass(&[segment], &taps).unwrap());
    }
    let out = out.unwrap();
    let bytes = &out.outputs.iter().find(|o| o.name == "residual").unwrap().bytes;
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn tokens_of(context: &Value) -> u64 {
    let context: gogi_proto::context::Context = serde_json::from_value(context.clone()).unwrap();
    ToyFamily::new().render(&context, true).unwrap().tokens.len() as u64
}

#[test]
fn a_served_model_declares_control_vectors_with_a_table_and_an_object_store() {
    for (tables, objects, declares) in [(Some(4), true, true), (Some(4), false, false), (None, true, false)] {
        let (f, _) = fixture(tables);
        let council = council_with(&f, ServiceConfig::new("toy", ENGINE), objects);
        let (status, bytes) = send_raw(&council, "GET", "/council/v2/identity", Vec::new(), None);
        assert_eq!(status, StatusCode::OK);
        let identity: ServerIdentity = serde_json::from_slice(&bytes).unwrap();
        let model = &identity.models[0];
        assert_eq!(model.has(Capability::ControlVectors), declares, "{tables:?} {objects}");
        let limit = declares.then_some(CONTROL_VECTOR_CONTEXTS);
        assert_eq!(model.limits.control_vector_contexts, limit, "{tables:?} {objects}");
    }
}

#[test]
fn the_route_puts_the_difference_of_the_means_as_an_object_and_leaves_no_state() {
    let (f, log) = fixture(Some(4));
    let council = council(&f);
    let positive = [said("I can't write that."), said("I won't help with that.")];
    let negative = [said("The dragon lands on the tower.")];
    let body = json!({"model": "toy", "positive": positive, "negative": negative});
    let entries = f.entries();
    log.take();
    let (status, r) = send(&council, "/council/v2/control-vectors", body.clone());
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_eq!(f.entries(), entries, "the job holds no snapshot and no interval copy");
    let logged = log.take();
    let lens: Vec<u64> = positive.iter().chain(&negative).map(tokens_of).collect();
    // The fake takes 64 tokens in a pass. Only the last pass of each context names the tap.
    let mut want: Vec<bool> = Vec::new();
    for len in &lens {
        let passes = len.div_ceil(64) as usize;
        want.extend((0..passes).map(|pass| pass + 1 == passes));
    }
    let tapped: Vec<bool> = logged.passes.iter().map(|(_, taps)| taps == &["residual"]).collect();
    assert_eq!(tapped, want, "{lens:?} {logged:?}");
    assert!(
        logged
            .passes
            .iter()
            .all(|(items, taps)| items.is_empty() && taps.len() <= 1),
        "a vector reads the plain model"
    );
    assert_eq!(
        (r["model"].as_str(), r["layers"].as_u64(), r["width"].as_u64()),
        (Some("toy"), Some(2), Some(4))
    );
    assert_eq!(r["contexts"], json!({"positive": 2, "negative": 1}));
    let prompt: u64 = positive.iter().chain(&negative).map(tokens_of).sum();
    let usage = &r["usage"];
    assert_eq!(
        (usage["prompt"].as_u64(), usage["kept"].as_u64(), usage["fed"].as_u64()),
        (Some(prompt), Some(0), Some(prompt))
    );
    assert_eq!(
        (usage["completion"].as_u64(), usage["reasoning"].as_u64()),
        (Some(0), Some(0))
    );
    let id = r["vector"].as_str().unwrap().to_owned();
    let (status, bytes) = send_raw(&council, "GET", &format!("/council/v2/objects/{id}"), Vec::new(), None);
    assert_eq!(status, StatusCode::OK);
    assert_eq!(gogi_proto::content_id(&bytes), id);
    assert_eq!(usage["media"]["written"], json!({"objects": 1, "bytes": bytes.len()}));
    let made = ControlVector::parse(&bytes).unwrap();
    assert_eq!((made.model.as_str(), made.weight_hash.as_str()), ("toy", "fake:0"));
    let rows: Vec<Vec<f32>> = positive.iter().chain(&negative).map(direct_row).collect();
    let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    let want = nagaya_council::control::vector_of_means(&refs[..2], &refs[2..]).unwrap();
    let got: Vec<f32> = made.layers.values().flatten().copied().collect();
    assert_eq!(made.layers.keys().copied().collect::<Vec<_>>(), [0, 1]);
    assert_eq!(
        got, want,
        "each layer is the mean of positive less the mean of negative"
    );
    assert!(
        want.iter().any(|v| *v != 0.0),
        "the sets differ, so the vector is not zero"
    );
    let (status, again) = send(&council, "/council/v2/control-vectors", body);
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        again["vector"],
        json!(id),
        "the same request gives the same bytes and the same id"
    );
    // A batch job unless the request names a class; an interactive request gives the same bytes.
    let interactive = json!({"model": "toy", "positive": positive, "negative": negative, "class": "interactive"});
    assert_eq!(
        send(&council, "/council/v2/control-vectors", interactive).1["vector"],
        json!(id)
    );
}

/// An engine with a checkpoint grain makes interval copies in a long feed (spec 6.13). The job of a vector makes none.
#[test]
fn the_route_makes_no_interval_copy_on_an_engine_with_a_grain() {
    let log = ControlLog::default();
    let factory = TermFactory {
        control: Some(4),
        log: log.clone(),
        grain: Some(16),
        interval_copies: 8,
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    let council = council(&f);
    let body =
        json!({"model": "toy", "positive": [said("I can't write that.")], "negative": [said("The dragon lands.")]});
    let (status, r) = send(&council, "/council/v2/control-vectors", body);
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_eq!(f.entries(), 0, "the job leaves no entry in the store");
    // A build of the same context does make copies, so the grain is live.
    build(&council, said("I can't write that."));
    assert!(f.entries() > 1, "a build holds its boundaries and its interval copies");
}

#[test]
fn a_dry_run_counts_the_contexts_and_makes_no_vector() {
    let (f, log) = fixture(Some(4));
    let council = council(&f);
    log.take();
    let body = json!({"model": "toy", "positive": [said("No.")], "negative": [said("Yes."), said("Sure.")],
                      "dry_run": true});
    let (status, r) = send(&council, "/council/v2/control-vectors", body);
    assert_eq!(status, StatusCode::OK, "{r}");
    assert!(r["vector"].is_null());
    assert_eq!((r["layers"].as_u64(), r["width"].as_u64()), (Some(2), Some(4)));
    let prompt = tokens_of(&said("No.")) + tokens_of(&said("Yes.")) + tokens_of(&said("Sure."));
    assert_eq!(
        (r["usage"]["prompt"].as_u64(), r["usage"]["fed"].as_u64()),
        (Some(prompt), Some(prompt))
    );
    assert!(log.take().passes.is_empty(), "a dry run runs no pass");
}

#[test]
fn the_route_refuses_what_gogi_and_the_limits_refuse() {
    let (f, _) = fixture(Some(4));
    let council = council(&f);
    let past = vec![said("No."); CONTROL_VECTOR_CONTEXTS as usize + 1];
    let any = format!("blake3:{}", "0".repeat(64));
    for (what, body, status, param) in [
        (
            "a side past the limit",
            json!({"model": "toy", "positive": past, "negative": [said("Yes.")]}),
            400,
            "positive",
        ),
        (
            "no negative",
            json!({"model": "toy", "positive": [said("No.")]}),
            400,
            "negative",
        ),
        (
            "a context with a control list",
            json!({"model": "toy", "positive": [said("No.")], "negative": [steered("x", json!([item(&any, "add", 1.0)]))]}),
            400,
            "negative[0].control",
        ),
        (
            "an unknown member",
            json!({"model": "toy", "positive": [said("No.")], "negative": [said("Yes.")], "normalize": true}),
            400,
            "normalize",
        ),
    ] {
        let (got, e) = send(&council, "/council/v2/control-vectors", body);
        assert_eq!(
            (got.as_u16(), e["error"]["param"].as_str()),
            (status, Some(param)),
            "{what}: {e}"
        );
    }
    // A model with no table refuses the route with a 400 that names `model`.
    let (plain, _) = fixture(None);
    let council = council_with(&plain, ServiceConfig::new("toy", ENGINE), true);
    let (got, e) = send(
        &council,
        "/council/v2/control-vectors",
        json!({"model": "toy", "positive": [said("No.")], "negative": [said("Yes.")]}),
    );
    assert_eq!(
        (got, e["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("model")),
        "{e}"
    );
    // A context past context_tokens with its generation prompt is a 413, and nothing runs.
    let (f, log) = fixture(Some(4));
    let mut config = ServiceConfig::new("toy", ENGINE);
    config.context_tokens = tokens_of(&said("Yes.")) as usize;
    let council = council_with(&f, config, true);
    log.take();
    let long = said("The dragon lands on the tower, and the knight runs.");
    let (got, e) = send(
        &council,
        "/council/v2/control-vectors",
        json!({"model": "toy", "positive": [said("Yes.")], "negative": [long]}),
    );
    assert_eq!(
        (got, e["error"]["param"].as_str()),
        (StatusCode::PAYLOAD_TOO_LARGE, Some("context_tokens")),
        "{e}"
    );
    assert!(log.take().passes.is_empty(), "a refused request runs no pass");
}

/// The fixture's vectors: `a` and `b` fit the fake, and the others break one rule each.
struct Vectors {
    a: String,
    b: String,
    c: String,
}

fn vectors(council: &Arc<Council>) -> Vectors {
    let layers = |base: f32| vec![(0, vec![base, 0.5, -1.0, 2.0]), (1, vec![0.25, base, 0.0, -0.5])];
    Vectors {
        a: put(council, &vector("fake:0", &layers(1.0)), MEDIA_TYPE),
        b: put(council, &vector("fake:0", &layers(3.0)), MEDIA_TYPE),
        c: put(council, &vector("fake:0", &[(1, vec![1.0, 1.0, 1.0, 1.0])]), MEDIA_TYPE),
    }
}

fn build(council: &Arc<Council>, context: Value) -> Value {
    let (status, built) = send(
        council,
        "/council/v2/contexts",
        json!({"model": "toy", "context": context}),
    );
    assert_eq!(status, StatusCode::OK, "{built}");
    built
}

fn snapshot_ids(built: &Value) -> Vec<Value> {
    built["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].clone())
        .collect()
}

fn logprobs(council: &Arc<Council>, context: Value) -> Value {
    let body = json!({"model": "toy", "contexts": [context], "suffixes": ["\nAnswer: "], "letters": ["A", "B", "C"]});
    let (status, read) = send(council, "/council/v2/reads", body);
    assert_eq!(status, StatusCode::OK, "{read}");
    read["reads"][0]["logprobs"].clone()
}

#[test]
fn a_control_list_has_its_own_root_and_a_list_that_changes_nothing_gives_the_plain_numbers() {
    let (f, _) = fixture(Some(4));
    let council = council(&f);
    let v = vectors(&council);
    let text = "Tell the story of the dragon.";
    let plain = build(&council, steered(text, json!([])));
    let zero = build(
        &council,
        steered(text, json!([item(&v.a, "add", 0.0), item(&v.a, "ablate", 0.0)])),
    );
    assert_ne!(
        zero["head"], plain["head"],
        "a list that changes nothing still has its own state"
    );
    assert!(
        zero["from"].is_null(),
        "the list never starts from the plain snapshot: {zero}"
    );
    assert!(snapshot_ids(&zero).iter().all(|id| !snapshot_ids(&plain).contains(id)));
    // The store keeps one recipe for each content id. A list that changes nothing gives the bytes of the plain state,
    // so its build holds no snapshot, and its head is the id of its recipe.
    assert!(snapshot_ids(&zero).is_empty(), "{zero}");
    // Two contexts with equal lists that change the bits share their head.
    let one = build(&council, steered(text, json!([item(&v.a, "add", 1.0)])));
    let again = build(&council, steered(text, json!([item(&v.a, "add", 1.0)])));
    assert_eq!(
        again["head"], one["head"],
        "two contexts with equal lists share their head"
    );
    assert_eq!(again["from"], one["head"], "{again}");
    assert_eq!(
        again["usage"]["kept"], again["usage"]["prompt"],
        "the second build keeps the whole context"
    );
    for list in [
        json!([item(&v.b, "add", 1.0)]),
        json!([item(&v.a, "add", 0.5)]),
        json!([item(&v.a, "clamp", 1.0)]),
    ] {
        let other = build(&council, steered(text, list.clone()));
        for seen in [&plain, &zero, &one] {
            assert_ne!(other["head"], seen["head"], "{list}");
            assert!(
                snapshot_ids(&other).iter().all(|id| !snapshot_ids(seen).contains(id)),
                "{list}"
            );
        }
    }
    let plain = logprobs(&council, steered(text, json!([])));
    let zero = logprobs(
        &council,
        steered(text, json!([item(&v.a, "add", 0.0), item(&v.b, "ablate", 0.0)])),
    );
    assert_eq!(zero, plain, "an add and an ablate of 0 give the numbers of no list");
    let clamp = logprobs(&council, steered(text, json!([item(&v.a, "clamp", 0.0)])));
    assert_ne!(clamp, plain, "a clamp always changes the numbers");
}

#[test]
fn each_pass_of_a_steered_context_carries_its_items_and_a_plain_one_carries_none() {
    let (f, log) = fixture(Some(4));
    let council = council(&f);
    let v = vectors(&council);
    log.take();
    let context = steered(
        "Tell the story of the knight.",
        json!([item(&v.a, "add", 2.0), item(&v.c, "clamp", -1.5)]),
    );
    let body = json!({"model": "toy", "context": context, "max_tokens": 6, "sample": {"temperature": 0}});
    let (status, generation) = send(&council, "/council/v2/generations", body);
    assert_eq!(status, StatusCode::OK, "{generation}");
    let logged = log.take();
    assert_eq!(
        logged.writes.len(),
        2,
        "two vectors go into the table once each: {:?}",
        logged.writes
    );
    assert!(
        logged.passes.len() > 2,
        "the prompt pass and the decode steps: {}",
        logged.passes.len()
    );
    let a = logged.writes.iter().find(|(_, first)| *first == 1.0).unwrap().0;
    let c = logged.writes.iter().find(|(_, first)| *first == 0.0).unwrap().0;
    for (items, _) in &logged.passes {
        let got: Vec<(u32, ControlOp, f32)> = items.iter().map(|i| (i.index, i.op, i.strength)).collect();
        assert_eq!(got, [(a, ControlOp::Add, 2.0), (c, ControlOp::Clamp, -1.5)]);
    }
    let body = json!({"model": "toy", "context": steered("Tell the story of the knight.", json!([])), "max_tokens": 6,
                      "sample": {"temperature": 0}});
    let (status, plain) = send(&council, "/council/v2/generations", body);
    assert_eq!(status, StatusCode::OK, "{plain}");
    let logged = log.take();
    assert!(logged.writes.is_empty());
    assert!(logged.passes.iter().all(|(items, _)| items.is_empty()));
}

#[test]
fn the_refusals_of_a_control_list_name_their_member() {
    let (f, log) = fixture(Some(4));
    let council = council(&f);
    let v = vectors(&council);
    let unheld = gogi_proto::content_id(b"no such vector");
    let text = put(&council, b"a text, not a vector", "text/plain");
    let theirs = put(
        &council,
        &vector("other:0", &[(0, vec![1.0, 0.0, 0.0, 0.0])]),
        MEDIA_TYPE,
    );
    let wide = put(
        &council,
        &vector("fake:0", &[(0, vec![1.0, 0.0, 0.0, 0.0, 1.0])]),
        MEDIA_TYPE,
    );
    let deep = put(
        &council,
        &vector("fake:0", &[(2, vec![1.0, 0.0, 0.0, 0.0])]),
        MEDIA_TYPE,
    );
    log.take();
    for (what, list, status, param) in [
        (
            "an unheld vector",
            json!([item(&v.a, "add", 1.0), item(&unheld, "add", 1.0)]),
            404,
            "control[1].vector",
        ),
        (
            "a text object",
            json!([item(&text, "add", 1.0)]),
            400,
            "control[0].vector",
        ),
        (
            "other weights",
            json!([item(&theirs, "add", 1.0)]),
            400,
            "control[0].vector",
        ),
        (
            "another width",
            json!([item(&wide, "add", 1.0)]),
            400,
            "control[0].vector",
        ),
        (
            "a layer past the model",
            json!([item(&deep, "add", 1.0)]),
            400,
            "control[0].vector",
        ),
        (
            "a strength past f32",
            json!([item(&v.a, "add", 1e300)]),
            400,
            "control[0].strength",
        ),
        ("an unknown op", json!([item(&v.a, "boost", 1.0)]), 400, "control[0].op"),
    ] {
        for path in ["/council/v2/contexts", "/council/v2/generations"] {
            let mut body = json!({"model": "toy", "context": steered("Tell the story.", list.clone())});
            if path.ends_with("generations") {
                body["max_tokens"] = json!(4);
            }
            let (got, e) = send(&council, path, body);
            assert_eq!(
                (got.as_u16(), e["error"]["param"].as_str()),
                (status, Some(param)),
                "{what} {path}: {e}"
            );
            if status == 404 {
                assert!(e["error"]["message"].as_str().unwrap().contains(&unheld), "{e}");
            }
        }
    }
    let body = json!({"model": "toy", "contexts": [steered("A.", json!([])), steered("B.", json!([item(&unheld, "add", 1.0)]))],
                      "suffixes": ["x"], "letters": ["A", "B"]});
    let (got, e) = send(&council, "/council/v2/reads", body);
    assert_eq!(
        (got, e["error"]["param"].as_str()),
        (StatusCode::NOT_FOUND, Some("contexts[1].control[0].vector")),
        "{e}"
    );
    let logged = log.take();
    assert!(
        logged.passes.is_empty() && logged.writes.is_empty(),
        "a refused list runs nothing: {logged:?}"
    );
}

#[test]
fn the_worker_writes_each_vector_once_replaces_the_least_recent_and_refuses_a_list_past_the_table() {
    let (f, log) = fixture(Some(2));
    let council = council(&f);
    let v = vectors(&council);
    log.take();
    build(&council, steered("One.", json!([item(&v.a, "add", 1.0)])));
    build(&council, steered("Two.", json!([item(&v.a, "ablate", 0.5)])));
    assert_eq!(log.take().writes, [(0, 1.0)], "a held vector is not written again");
    build(&council, steered("Three.", json!([item(&v.b, "add", 1.0)])));
    assert_eq!(log.take().writes, [(1, 3.0)], "an empty entry takes the next vector");
    build(&council, steered("Four.", json!([item(&v.b, "add", 2.0)])));
    build(&council, steered("Five.", json!([item(&v.c, "add", 1.0)])));
    // Vector c holds no layer 0, so its first value is 0.
    assert_eq!(
        log.take().writes,
        [(0, 0.0)],
        "the entry of least recent use takes the new vector"
    );
    let (status, e) = send(
        &council,
        "/council/v2/contexts",
        json!({"model": "toy", "context": steered("Six.", json!([item(&v.a, "add", 1.0), item(&v.b, "add", 1.0), item(&v.c, "add", 1.0)]))}),
    );
    assert_eq!(
        (status, e["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("control")),
        "{e}"
    );
    assert!(log.take().passes.is_empty(), "the refusal comes before the job runs");
}
