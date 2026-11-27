//! The lab route `trace` (`docs/lab.md`, "The rules of `trace`") with the fake engine and the toy family. The fake has
//! the taps `residual` and `lens`, and a temporary model folder gives the layers, the width and the end token.

mod council_support;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ControlLog, ENGINE, Fixture, TermFactory, lens_file};
use gogi_proto::control::{ControlVector, MEDIA_TYPE};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ControlItem, ControlOp, Engine, OutputsWanted, Segment, SegmentInputs, State, Tap};
use nagaya_council::family::Family;
use nagaya_council::trace::{Plan, Trace};
use nagaya_family_toy::ToyFamily;
use nagaya_store::objects::ObjectStore;
use safetensors::SafeTensors;
use serde_json::{Value, json};
use tower::ServiceExt;

const PATH: &str = "/nagaya/v1/lab/trace";
/// The layers and the width of the fake's residual.
const LAYERS: usize = 2;
const WIDTH: usize = 4;
const VOCAB: usize = gogi_toy::VOCAB;
const END: u32 = gogi_toy::END;

struct Lab {
    fixture: Fixture,
    council: Arc<Council>,
    log: ControlLog,
    _folder: tempfile::TempDir,
}

/// The model folder of the fake: the facts that a trace reads, and no weights.
fn folder(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        json!({"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": LAYERS, "hidden_size": WIDTH}).to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        json!({"eos_token_id": [END]}).to_string(),
    )
    .unwrap();
}

/// The weight of the final norm of the small head of a J-Lens folder.
const NORM: [f32; WIDTH] = [2.0, 0.5, -1.0, 1.0];
/// The epsilon of the final norm of that folder.
const EPS: f64 = 0.25;

/// The value of the row `r` of the small head at the column `c`: small integers, which BF16 holds exactly.
fn table(r: usize, c: usize) -> f32 {
    ((r * 7 + c * 3) % 11) as f32 - 5.0
}

/// Add a small head to the model folder: a table of `VOCAB` rows, the norm weight `NORM` and the epsilon `EPS`.
fn head(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        json!({"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": LAYERS, "hidden_size": WIDTH,
               "norm_eps": EPS})
        .to_string(),
    )
    .unwrap();
    let bf16 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let flat: Vec<f32> = (0..VOCAB).flat_map(|r| (0..WIDTH).map(move |c| table(r, c))).collect();
    let (embed, norm) = (bf16(&flat), bf16(&NORM));
    let header = json!({
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [VOCAB, WIDTH], "data_offsets": [0, embed.len()]},
        "model.embedding_norm.weight": {"dtype": "BF16", "shape": [WIDTH],
                                        "data_offsets": [embed.len(), embed.len() + norm.len()]},
    })
    .to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(embed);
    bytes.extend(norm);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

/// A council whose served model `toy` speaks the toy language on the fake. `taps` gives the fake its control table
/// and the tap `residual`, and `lens` the tap `lens` too. `objects` is the budget of the object store.
fn lab_with(taps: bool, lens: bool, objects: u64) -> Lab {
    lab_full(taps, lens, objects, None)
}

/// `lab_with`, and with `jlens` the folder has a small head and the profile names a J-Lens of these weights and
/// matrices.
fn lab_full(taps: bool, lens: bool, objects: u64, jlens: Option<(&str, &[Vec<f32>])>) -> Lab {
    let log = ControlLog::default();
    let factory = TermFactory {
        control: taps.then_some(4),
        lens,
        speaking: true,
        // A seal of the fake changes its bits. With no seal, a replay on a fresh fake gives the bits of the job.
        seal: false,
        log: log.clone(),
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
    folder(dir.path());
    let store = ObjectStore::open(&fixture.directory.path().join("objects"), objects).unwrap();
    let mut council = Council::new(vec![served], None)
        .unwrap()
        .with_objects(store)
        .with_lab(BTreeMap::from([(ENGINE.to_owned(), dir.path().to_path_buf())]));
    if let Some((hash, matrices)) = jlens {
        head(dir.path());
        let path = dir.path().join("lens.safetensors");
        lens_file(&path, hash, matrices, WIDTH);
        council = council.with_lenses(BTreeMap::from([(ENGINE.to_owned(), path)]));
    }
    Lab {
        fixture,
        council: Arc::new(council),
        log,
        _folder: dir,
    }
}

fn lab() -> Lab {
    lab_with(true, true, 1 << 24)
}

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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn send(council: &Arc<Council>, path: &str, body: &Value) -> (StatusCode, Value) {
    let (status, bytes) = send_raw(council, "POST", path, body.to_string().into_bytes(), None);
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Put a control vector with `layers`, and give its id.
fn put(council: &Arc<Council>, weight_hash: &str, layers: &[(u32, Vec<f32>)]) -> String {
    let bytes = ControlVector {
        model: "toy".to_owned(),
        weight_hash: weight_hash.to_owned(),
        layers: layers.iter().cloned().collect(),
    }
    .to_bytes();
    let id = gogi_proto::content_id(&bytes);
    let (status, body) = send_raw(
        council,
        "PUT",
        &format!("/council/v2/objects/{id}"),
        bytes,
        Some(MEDIA_TYPE),
    );
    assert!(status.is_success(), "{status}: {}", String::from_utf8_lossy(&body));
    id
}

const V0: [f32; WIDTH] = [1.0, -2.0, 0.5, 3.0];
const V1: [f32; WIDTH] = [0.0, 1.0, 1.0, 0.0];

fn vector(council: &Arc<Council>) -> String {
    put(council, "fake:0", &[(0, V0.to_vec()), (1, V1.to_vec())])
}

fn context(control: Value) -> Value {
    json!({"thinking": false, "control": control,
           "messages": [{"role": "user", "content": "Tell the tale of the dragon.", "snap": true}]})
}

/// A generation request: greedy, with a fixed seed, so that two runs give the same response.
fn generation(control: Value) -> Value {
    json!({"model": "toy", "context": context(control), "max_tokens": 24, "sample": {"temperature": 0, "seed": 7}})
}

fn traced(control: Value, trace: Value) -> Value {
    let mut body = generation(control);
    body["trace"] = trace;
    body
}

fn object(council: &Arc<Council>, id: &str) -> Vec<u8> {
    let (status, bytes) = send_raw(council, "GET", &format!("/council/v2/objects/{id}"), Vec::new(), None);
    assert_eq!(status, StatusCode::OK);
    assert_eq!(gogi_proto::content_id(&bytes), id);
    bytes
}

fn f32_of(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn u32_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

/// What a fresh fake gives at each step of a reply: the logit row, the residual rows and the lens rows of the
/// position before each sampled token. Each pass feeds one token, with the taps.
struct Replayed {
    logits: Vec<Vec<f32>>,
    residual: Vec<Vec<f32>>,
    lens: Vec<Vec<f32>>,
}

fn replay(prompt: &[u32], sampled: &[u32], items: &[(ControlOp, f32)]) -> Replayed {
    let mut engine = FakeAlways::with_slots(council_support::roomy_slots(1))
        .with_vocab(VOCAB)
        .with_logit_bias(nagaya_family_toy::speaking_bias())
        .with_control(4)
        .with_lens();
    let table: Vec<u8> = V0.iter().chain(&V1).flat_map(|v| v.to_le_bytes()).collect();
    engine.control().unwrap().control_write(0, &table).unwrap();
    let control: Vec<ControlItem> = items
        .iter()
        .map(|(op, strength)| ControlItem {
            index: 0,
            op: *op,
            strength: *strength,
            layers: None,
        })
        .collect();
    let tokens: Vec<u32> = prompt.iter().chain(&sampled[..sampled.len() - 1]).copied().collect();
    let mut out = Replayed {
        logits: Vec::new(),
        residual: Vec::new(),
        lens: Vec::new(),
    };
    for (position, token) in tokens.iter().enumerate() {
        let step = position + 1 >= prompt.len();
        let segment = Segment {
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(vec![*token]),
            positions: vec![position as u64],
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if step { vec![position as u64] } else { Vec::new() },
            },
            control: control.clone(),
            layer_loop: None,
            cycles: None,
        };
        let taps: Vec<Tap> = match step {
            true => ["residual", "lens"].map(|name| Tap { name: name.to_owned() }).to_vec(),
            false => Vec::new(),
        };
        let pass = engine.pass(&[segment], &taps).unwrap();
        if step {
            let get = |name: &str| f32_of(&pass.outputs.iter().find(|o| o.name == name).unwrap().bytes);
            out.logits.push(get("logits"));
            out.residual.push(get("residual"));
            out.lens.push(get("lens"));
        }
    }
    out
}

fn prompt_of(control: Value) -> Vec<u32> {
    let context: gogi_proto::context::Context = serde_json::from_value(context(control)).unwrap();
    ToyFamily::new().render(&context, true).unwrap().tokens
}

/// The log softmax of a row in `f64`, as the trace takes it.
fn logprob(row: &[f32], id: u32) -> f32 {
    let max = row.iter().map(|x| *x as f64).fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = row.iter().map(|x| (*x as f64 - max).exp()).sum();
    (row[id as usize] as f64 - (max + sum.ln())) as f32
}

fn argsort(row: &[f32], k: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..row.len() as u32).collect();
    ids.sort_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]).then(a.cmp(b)));
    ids.truncate(k);
    ids
}

fn unit(v: &[f32]) -> Vec<f64> {
    let norm = v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    v.iter().map(|x| *x as f64 / norm).collect()
}

fn without_trace(mut response: Value) -> Value {
    response.as_object_mut().unwrap().remove("trace");
    response
}

#[test]
fn a_trace_records_each_step_of_the_reply_as_the_fake_gives_it() {
    for (what, control, items) in [
        ("plain", json!([]), Vec::new()),
        ("ablate 1", json!(null), vec![(ControlOp::Ablate, 1.0)]),
    ] {
        let lab = lab();
        let v = vector(&lab.council);
        let control = match control {
            Value::Null => json!([{"vector": v, "op": "ablate", "strength": 1.0}]),
            other => other,
        };
        let trace = json!({"layers": [1, 0], "vectors": [v], "top_k": 5, "lens": "logit", "rows": true});
        let (status, r) = send(&lab.council, PATH, &traced(control.clone(), trace));
        assert_eq!(status, StatusCode::OK, "{what}: {r}");
        let id = r["trace"].as_str().unwrap();
        let bytes = object(&lab.council, id);
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let get = |name: &str| st.tensor(name).unwrap();
        let tokens = u32_of(get("tokens").data());
        let s = tokens.len();
        assert_eq!(s as u64, r["usage"]["completion"].as_u64().unwrap(), "{what}");
        assert!(s > 1, "{what}: a reply of {s} tokens");
        assert_eq!(
            tokens.last(),
            Some(&END),
            "{what}: the greedy toy reply ends with <end>"
        );
        let prompt = prompt_of(control.clone());
        let fake = replay(&prompt, &tokens, &items);
        assert_eq!(get("top_ids").shape(), [s, 5]);
        assert_eq!(get("norms").shape(), [s, 2]);
        assert_eq!(get("along").shape(), [s, 2, 1]);
        assert_eq!(get("rows").shape(), [s, 2, WIDTH]);
        assert_eq!(get("lens_ids").shape(), [s, 2, 5]);
        assert_eq!(get("end_logprob").shape(), [s, 1]);
        assert_eq!(u32_of(get("layers").data()), [1, 0]);
        assert_eq!(u32_of(get("end_ids").data()), [END]);
        let (top, top_lp) = (u32_of(get("top_ids").data()), f32_of(get("top_logprobs").data()));
        let (norms, along, rows) = (
            f32_of(get("norms").data()),
            f32_of(get("along").data()),
            f32_of(get("rows").data()),
        );
        let (lens_ids, lens_lp) = (u32_of(get("lens_ids").data()), f32_of(get("lens_logprobs").data()));
        let end_lp = f32_of(get("end_logprob").data());
        let entropy = f32_of(get("entropy").data());
        let (u0, u1) = (unit(&V0), unit(&V1));
        for i in 0..s {
            let logits = &fake.logits[i];
            assert_eq!(tokens[i], argsort(logits, 1)[0], "{what}: step {i} is the greedy token");
            assert_eq!(&top[i * 5..(i + 1) * 5], &argsort(logits, 5)[..], "{what} step {i}");
            assert_eq!(top_lp[i * 5], logprob(logits, top[i * 5]), "{what} step {i}");
            assert_eq!(end_lp[i], logprob(logits, END), "{what} step {i}");
            let (_, h) = nagaya_council::trace::distribution(logits);
            assert_eq!(entropy[i], h as f32, "{what} step {i}");
            for (j, (layer, u)) in [(1usize, &u1), (0usize, &u0)].into_iter().enumerate() {
                let h = &fake.residual[i][layer * WIDTH..(layer + 1) * WIDTH];
                let n = h.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt() as f32;
                assert_eq!(norms[i * 2 + j], n, "{what} step {i} layer {layer}");
                let a = h.iter().zip(u.iter()).map(|(x, y)| *x as f64 * y).sum::<f64>() as f32;
                assert_eq!(along[i * 2 + j], a, "{what} step {i} layer {layer}");
                assert_eq!(&rows[(i * 2 + j) * WIDTH..(i * 2 + j + 1) * WIDTH], h);
                let lens = &fake.lens[i][layer * VOCAB..(layer + 1) * VOCAB];
                let at = (i * 2 + j) * 5;
                assert_eq!(
                    &lens_ids[at..at + 5],
                    &argsort(lens, 5)[..],
                    "{what} step {i} layer {layer}"
                );
                assert_eq!(lens_lp[at], logprob(lens, lens_ids[at]));
            }
            // The fake's lens of its last layer is its head, so its top entries are the step's own.
            assert_eq!(
                &lens_ids[i * 10..i * 10 + 5],
                &top[i * 5..(i + 1) * 5],
                "{what} step {i}"
            );
        }
        // Only the last pass of each feed names the taps, and every pass of a trace keeps the control list.
        let passes = lab.log.take().passes;
        let tapped: Vec<_> = passes.iter().filter(|(_, taps)| !taps.is_empty()).collect();
        assert_eq!(tapped.len(), s, "{what}: one tapped pass for each step");
        assert!(tapped.iter().all(|(_, taps)| taps == &["residual", "lens"]), "{what}");
        assert!(
            passes.iter().all(|(listed, _)| listed.len() == items.len()),
            "{what}: {passes:?}"
        );
    }
}

/// The response of a trace is the response of the same generation with no trace, on a fresh process. The bits of the
/// fake's logits come from its state, so the same tokens show the same bits.
#[test]
fn a_trace_gives_the_response_of_the_generation_with_no_trace() {
    for (what, ablate) in [("plain", false), ("ablate 1", true)] {
        let plain = lab();
        let traced_lab = lab();
        let control = |lab: &Lab| match ablate {
            true => json!([{"vector": vector(&lab.council), "op": "ablate", "strength": 1.0}]),
            false => json!([]),
        };
        let (status, want) = send(&plain.council, "/council/v2/generations", &generation(control(&plain)));
        assert_eq!(status, StatusCode::OK, "{what}: {want}");
        let generated = plain.log.take().passes;
        assert!(
            generated.iter().all(|(_, taps)| taps.is_empty()),
            "{what}: a generation names no tap"
        );
        let trace = json!({"lens": "logit", "vectors": [vector(&traced_lab.council)]});
        let (status, got) = send(&traced_lab.council, PATH, &traced(control(&traced_lab), trace));
        assert_eq!(status, StatusCode::OK, "{what}: {got}");
        assert!(got["trace"].as_str().unwrap().starts_with("blake3:"));
        assert_eq!(without_trace(got), want, "{what}");
        let traced_passes = traced_lab.log.take().passes;
        assert_eq!(traced_passes.len(), generated.len(), "{what}: the same passes");
        let _ = &plain.fixture;
    }
}

/// A trace takes a context that names a `parent`, and its response names the record with the reply, as a generation
/// does (`docs/lab.md`, "Context records in the lab routes").
///
/// Falsified by a route with no `whole`: the 400 of the unknown member `parent`.
#[test]
fn a_trace_takes_a_parent_and_names_the_record_with_its_reply() {
    let lab = lab();
    let (status, want) = send(&lab.council, "/council/v2/generations", &generation(json!([])));
    assert_eq!(status, StatusCode::OK, "{want}");
    let record: gogi_proto::Context = serde_json::from_value(context(json!([]))).unwrap();
    let mut body = traced(json!([]), json!({"lens": "logit"}));
    body["context"] = json!({"parent": gogi_proto::context_id(&record).as_str()});
    let (status, got) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "{got}");
    assert!(got["trace"].as_str().unwrap().starts_with("blake3:"));
    assert_eq!(got["content"], want["content"]);
    assert!(got["context"].is_string(), "{got}");
    assert_eq!(got["context"], want["context"], "the same context and the same reply");
    body["context"] = json!({"parent": format!("blake3:{}", "0".repeat(64))});
    assert_refused(send(&lab.council, PATH, &body), StatusCode::NOT_FOUND, "context.parent");
}

#[test]
fn the_same_trace_gives_the_same_object() {
    let lab = lab();
    let v = vector(&lab.council);
    let body = traced(json!([]), json!({"vectors": [v], "lens": "logit", "rows": true}));
    let (status, first) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "{first}");
    // The second request restores the held context. The trace reads the same rows.
    let (status, second) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "{second}");
    assert!(second["usage"]["kept"].as_u64().unwrap() > 0, "{second}");
    assert_eq!(first["trace"], second["trace"]);
    let other = lab_with(true, true, 1 << 24);
    let v = vector(&other.council);
    let body = traced(json!([]), json!({"vectors": [v], "lens": "logit", "rows": true}));
    assert_eq!(send(&other.council, PATH, &body).1["trace"], first["trace"]);
    // With no lens and the default members, the object has no lens and no rows, and a trace of every layer.
    let (_, r) = send(&lab.council, PATH, &traced(json!([]), json!({})));
    let bytes = object(&lab.council, r["trace"].as_str().unwrap());
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut names: Vec<&str> = st.names();
    names.sort();
    assert_eq!(
        names,
        [
            "end_ids",
            "end_logprob",
            "entropy",
            "layers",
            "norms",
            "tokens",
            "top_ids",
            "top_logprobs"
        ]
    );
    assert_eq!(u32_of(st.tensor("layers").unwrap().data()), [0, 1]);
    assert_eq!(st.tensor("top_ids").unwrap().shape()[1], 8, "the default top_k");
}

fn assert_refused(response: (StatusCode, Value), status: StatusCode, param: &str) {
    let (got, body) = response;
    assert_eq!(got, status, "{param}: {body}");
    assert_eq!(body["error"]["param"], param, "{body}");
}

#[test]
fn trace_refuses_what_breaks_its_rules_before_the_job() {
    let lab = lab();
    let v = vector(&lab.council);
    let other = put(&lab.council, "other-weights", &[(0, V0.to_vec())]);
    let missing = format!("blake3:{}", "0".repeat(64));
    lab.log.take();
    let cases = [
        ("no trace", generation(json!([])), StatusCode::BAD_REQUEST, "trace"),
        (
            "trace not an object",
            traced(json!([]), json!(3)),
            StatusCode::BAD_REQUEST,
            "trace",
        ),
        (
            "an unknown member",
            traced(json!([]), json!({"layer": [0]})),
            StatusCode::BAD_REQUEST,
            "trace.layer",
        ),
        (
            "layers past the count",
            traced(json!([]), json!({"layers": [2]})),
            StatusCode::BAD_REQUEST,
            "trace.layers",
        ),
        (
            "empty layers",
            traced(json!([]), json!({"layers": []})),
            StatusCode::BAD_REQUEST,
            "trace.layers",
        ),
        (
            "a layer twice",
            traced(json!([]), json!({"layers": [0, 0]})),
            StatusCode::BAD_REQUEST,
            "trace.layers",
        ),
        (
            "layers of text",
            traced(json!([]), json!({"layers": "some"})),
            StatusCode::BAD_REQUEST,
            "trace.layers",
        ),
        (
            "top_k 0",
            traced(json!([]), json!({"top_k": 0})),
            StatusCode::BAD_REQUEST,
            "trace.top_k",
        ),
        (
            "top_k 65",
            traced(json!([]), json!({"top_k": 65})),
            StatusCode::BAD_REQUEST,
            "trace.top_k",
        ),
        (
            "another lens",
            traced(json!([]), json!({"lens": "jlens"})),
            StatusCode::BAD_REQUEST,
            "trace.lens",
        ),
        (
            "rows of text",
            traced(json!([]), json!({"rows": "yes"})),
            StatusCode::BAD_REQUEST,
            "trace.rows",
        ),
        (
            "a vector not held",
            traced(json!([]), json!({"vectors": [missing]})),
            StatusCode::NOT_FOUND,
            "trace.vectors[0]",
        ),
        (
            "a vector of other weights",
            traced(json!([]), json!({"vectors": [other]})),
            StatusCode::BAD_REQUEST,
            "trace.vectors[0]",
        ),
        (
            "a vector twice",
            traced(json!([]), json!({"vectors": [v, v]})),
            StatusCode::BAD_REQUEST,
            "trace.vectors[1]",
        ),
        (
            "not an id",
            traced(json!([]), json!({"vectors": ["seawater"]})),
            StatusCode::BAD_REQUEST,
            "trace.vectors[0]",
        ),
    ];
    for (what, body, status, param) in cases {
        let (got, r) = send(&lab.council, PATH, &body);
        assert_eq!(got, status, "{what}: {r}");
        assert_eq!(r["error"]["param"], param, "{what}: {r}");
    }
    let mut stream = traced(json!([]), json!({}));
    stream["stream"] = json!(true);
    assert_refused(send(&lab.council, PATH, &stream), StatusCode::BAD_REQUEST, "stream");
    let mut unknown = traced(json!([]), json!({}));
    unknown["model"] = json!("nobody");
    assert_refused(send(&lab.council, PATH, &unknown), StatusCode::BAD_REQUEST, "model");
    assert!(lab.log.take().passes.is_empty(), "no refused trace runs a pass");
}

#[test]
fn a_model_without_the_taps_does_not_trace() {
    // No control table: the fake has no tap `residual`.
    let bare = lab_with(false, false, 1 << 24);
    assert_refused(
        send(&bare.council, PATH, &traced(json!([]), json!({}))),
        StatusCode::BAD_REQUEST,
        "model",
    );
    // The tap `residual` and no tap `lens`: a trace runs, and a lens is refused.
    let no_lens = lab_with(true, false, 1 << 24);
    let (status, r) = send(&no_lens.council, PATH, &traced(json!([]), json!({})));
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_refused(
        send(&no_lens.council, PATH, &traced(json!([]), json!({"lens": "logit"}))),
        StatusCode::BAD_REQUEST,
        "trace.lens",
    );
}

/// The plan that the route makes for `trace` with no vector, for the size of its object.
fn plan(rows: bool) -> Plan {
    Plan {
        model: "toy".to_owned(),
        weight_hash: "fake:0".to_owned(),
        layers: vec![0, 1],
        model_layers: LAYERS,
        width: WIDTH,
        vocab: VOCAB,
        top_k: 8,
        ends: vec![END],
        vectors: Vec::new(),
        lens: nagaya_council::trace::Lens::None,
        rows,
    }
}

#[test]
fn rows_past_the_budget_are_a_413_before_the_job() {
    // A store of 64 KiB holds an object of 32 KiB at most.
    let budget = 1u64 << 16;
    let largest = budget / 2;
    let steps = (1..4096)
        .find(|n| Trace::object_bytes(&plan(true), *n) > largest)
        .unwrap();
    assert!(
        Trace::object_bytes(&plan(false), steps) <= largest,
        "the rows decide the size"
    );
    let lab = lab_with(true, true, budget);
    lab.log.take();
    let mut body = traced(json!([]), json!({"rows": true}));
    body["max_tokens"] = json!(steps);
    assert_refused(
        send(&lab.council, PATH, &body),
        StatusCode::PAYLOAD_TOO_LARGE,
        "trace.rows",
    );
    assert!(lab.log.take().passes.is_empty(), "the 413 comes before the job");
    body["max_tokens"] = json!(steps - 1);
    let (status, r) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "one step fewer fits: {r}");
    body["trace"] = json!({});
    body["max_tokens"] = json!(steps);
    let (status, r) = send(&lab.council, PATH, &body);
    assert_eq!(status, StatusCode::OK, "with no rows it fits: {r}");
}

/// The J-Lens of layer 0 of the fake. The lens of layer 1, the last, is the identity.
fn j0() -> Vec<f32> {
    vec![
        1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0,
    ]
}

/// The logit row of the J-Lens of the residual row `h` of `layer`, by hand: `J_l h`, the final norm with its division,
/// and the small head.
fn jlens_row(layer: usize, h: &[f32]) -> Vec<f32> {
    let moved: Vec<f32> = if layer == 0 {
        let j = j0();
        (0..WIDTH)
            .map(|i| (0..WIDTH).map(|c| j[i * WIDTH + c] as f64 * h[c] as f64).sum::<f64>() as f32)
            .collect()
    } else {
        h.to_vec()
    };
    let mean = moved.iter().map(|x| *x as f64 * *x as f64).sum::<f64>() / WIDTH as f64;
    let inverse = 1.0 / (mean + EPS).sqrt();
    let scaled: Vec<f32> = moved.iter().map(|x| (*x as f64 * inverse) as f32).collect();
    (0..VOCAB)
        .map(|r| {
            (0..WIDTH)
                .map(|c| table(r, c) as f64 * (NORM[c] * scaled[c]) as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn a_jlens_trace_reads_each_layer_through_the_lens_of_the_profile_from_the_tap_residual_only() {
    let lab = lab_full(true, true, 1 << 24, Some(("fake:0", &[j0()])));
    let trace = json!({"layers": [1, 0], "top_k": 5, "lens": "jlens", "rows": true});
    let (status, r) = send(&lab.council, PATH, &traced(json!([]), trace));
    assert_eq!(status, StatusCode::OK, "{r}");
    let bytes = object(&lab.council, r["trace"].as_str().unwrap());
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let get = |name: &str| st.tensor(name).unwrap();
    let s = u32_of(get("tokens").data()).len();
    let rows = f32_of(get("rows").data());
    let (lens_ids, lens_lp) = (u32_of(get("lens_ids").data()), f32_of(get("lens_logprobs").data()));
    assert_eq!(get("lens_ids").shape(), [s, 2, 5]);
    for i in 0..s {
        for (j, layer) in [1usize, 0].into_iter().enumerate() {
            let h = &rows[(i * 2 + j) * WIDTH..(i * 2 + j + 1) * WIDTH];
            let row = jlens_row(layer, h);
            let at = (i * 2 + j) * 5;
            assert_eq!(&lens_ids[at..at + 5], &argsort(&row, 5)[..], "step {i} layer {layer}");
            assert_eq!(lens_lp[at], logprob(&row, lens_ids[at]), "step {i} layer {layer}");
        }
    }
    let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
    let meta = meta.metadata().clone().unwrap();
    assert_eq!(meta["lens"], "jlens");
    let lens = std::fs::read(lab._folder.path().join("lens.safetensors")).unwrap();
    assert_eq!(meta["lens_id"], format!("blake3:{}", blake3::hash(&lens).to_hex()));
    // The CPU reads the lens, so the passes name the tap `residual` alone.
    let passes = lab.log.take().passes;
    let tapped: Vec<_> = passes.iter().filter(|(_, taps)| !taps.is_empty()).collect();
    assert_eq!(tapped.len(), s);
    assert!(tapped.iter().all(|(_, taps)| taps == &["residual"]), "{tapped:?}");
}

#[test]
fn a_jlens_trace_refuses_a_model_with_no_lens_and_a_lens_of_other_weights() {
    let trace = json!({"layers": "all", "lens": "jlens"});
    let none = lab();
    assert_refused(
        send(&none.council, PATH, &traced(json!([]), trace.clone())),
        StatusCode::BAD_REQUEST,
        "trace.lens",
    );
    let other = lab_full(true, true, 1 << 24, Some(("blake3:00", &[j0()])));
    assert_refused(
        send(&other.council, PATH, &traced(json!([]), trace)),
        StatusCode::BAD_REQUEST,
        "trace.lens",
    );
}
