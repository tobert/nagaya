//! Latent steps in a request of `ensemble` (`docs/lab.md`, "Latent steps in a request") with the fake engine and the
//! toy family. The fake has the tap `residual`, and a temporary model folder gives the final norm and the table.

mod council_support;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ControlLog, ENGINE, Fixture, Seen, TermFactory};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya_abi::SegmentInputs;
use nagaya_text_toy::{LATENT_TOKEN, ToyFamily};
use serde_json::{Value, json};
use tower::ServiceExt;

const PATH: &str = "/nagaya/v1/lab/ensemble";
/// The layers and the width of the fake's residual.
const LAYERS: usize = 2;
const WIDTH: usize = 4;
const VOCAB: usize = gogi_toy::VOCAB;

struct Lab {
    fixture: Fixture,
    council: Arc<Council>,
    log: ControlLog,
    seen: Seen,
    folder: tempfile::TempDir,
}

/// The value of the row `r` of the small table at the column `c`: small integers, which BF16 holds exactly.
fn table(r: usize, c: usize) -> f32 {
    ((r * 7 + c * 3) % 11) as f32 - 5.0
}

/// The mean L2 norm of the rows of the small table: the length of each latent row.
fn scale() -> f64 {
    (0..VOCAB)
        .map(|r| (0..WIDTH).map(|c| f64::from(table(r, c)).powi(2)).sum::<f64>().sqrt())
        .sum::<f64>()
        / VOCAB as f64
}

/// The model folder of the fake: the facts of the model, a table of `VOCAB` rows and the weight of the final norm.
fn folder(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        json!({"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": LAYERS, "hidden_size": WIDTH,
               "norm_eps": 0.25})
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        json!({"eos_token_id": [gogi_toy::END]}).to_string(),
    )
    .unwrap();
    let bf16 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let flat: Vec<f32> = (0..VOCAB).flat_map(|r| (0..WIDTH).map(move |c| table(r, c))).collect();
    let (embed, norm) = (bf16(&flat), bf16(&[2.0, 0.5, -1.0, 1.0]));
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

/// A council whose served model `toy` runs on the fake. With `taps`, the fake has the control table and the tap
/// `residual`.
fn lab_with(taps: bool) -> Lab {
    let log = ControlLog::default();
    let seen = Seen::default();
    let factory = TermFactory {
        control: taps.then_some(4),
        speaking: true,
        seal: false,
        log: log.clone(),
        seen: Some(seen.clone()),
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
    let council = Council::new(vec![served], None)
        .unwrap()
        .with_lab(BTreeMap::from([(ENGINE.to_owned(), dir.path().to_path_buf())]));
    Lab {
        fixture,
        council: Arc::new(council),
        log,
        seen,
        folder: dir,
    }
}

fn send(council: &Arc<Council>, body: &Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri(PATH)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn request(latent: Option<Value>) -> Value {
    let mut body = json!({"model": "toy", "combine": "logprob_sum", "max_tokens": 16,
                          "sample": {"temperature": 0.8, "seed": 7},
                          "contexts": [{"thinking": false,
                                        "messages": [{"role": "user", "content": "Tell the tale of the dragon."}]}]});
    if let Some(latent) = latent {
        body["latent"] = latent;
    }
    body
}

/// The record of a latent span in the meta of a segment, or `None` for a segment with no latent rows.
fn latent_span(meta: &[u8]) -> Option<(u64, u32, u32)> {
    let word = |at: usize| u32::from_le_bytes(meta[at..at + 4].try_into().unwrap());
    (meta.len() == 32 && word(0) == 5).then(|| (u64::from_le_bytes(meta[8..16].try_into().unwrap()), word(4), word(16)))
}

#[test]
fn zero_steps_give_the_response_and_the_passes_of_no_member() {
    let plain = lab_with(true);
    let (status, want) = send(&plain.council, &request(None));
    assert_eq!(status, StatusCode::OK, "{want}");
    let passes = plain.log.take();
    assert!(want.get("latent").is_none(), "{want}");

    let zero = lab_with(true);
    let (status, got) = send(&zero.council, &request(Some(json!({"steps": 0, "read": true}))));
    assert_eq!(status, StatusCode::OK, "{got}");
    for member in ["content", "finish", "usage"] {
        assert_eq!(got[member], want[member], "{member}");
    }
    assert_eq!(got["latent"], json!({"steps": 0, "read": []}));
    let logged = zero.log.take();
    assert_eq!(
        (logged.sizes, logged.passes),
        (passes.sizes, passes.passes),
        "the same passes and taps"
    );
    assert!(zero.seen.take().iter().all(|s| s.meta.is_empty()), "no latent span");
}

#[test]
fn n_steps_add_n_passes_of_one_latent_row_each_and_the_reply_ends() {
    let lab = lab_with(true);
    let (status, got) = send(&lab.council, &request(Some(json!({"steps": 3, "read": true}))));
    assert_eq!(status, StatusCode::OK, "{got}");
    assert!(got["finish"].is_string(), "{got}");
    assert!(got["usage"]["completion"].as_u64().unwrap() >= 1, "{got}");
    let read = got["latent"]["read"].as_array().unwrap();
    assert_eq!(got["latent"]["steps"], 3);
    assert_eq!(read.len(), 3, "one top token for each step: {got}");
    for token in read {
        let id = token["id"].as_u64().unwrap() as usize;
        assert!(id < VOCAB, "{token}");
        assert!(token["text"].is_string(), "{token}");
    }
    let segments = lab.seen.take();
    let latent: Vec<_> = segments
        .iter()
        .filter_map(|s| latent_span(&s.meta).map(|span| (s, span)))
        .collect();
    assert_eq!(latent.len(), 3, "three passes of a latent row");
    let first = latent[0].1.0;
    let scale = scale();
    for (k, (segment, (at, count, width))) in latent.iter().enumerate() {
        assert_eq!((*at, *count, *width), (first + k as u64, 1, WIDTH as u32), "step {k}");
        assert_eq!(segment.positions, [first + k as u64], "step {k}");
        assert_eq!(segment.outputs.logit_rows, [first + k as u64], "step {k}");
        let SegmentInputs::TokensAndRows { tokens, rows } = &segment.inputs else {
            panic!("step {k} feeds tokens and rows");
        };
        assert_eq!(tokens, &[LATENT_TOKEN], "step {k}");
        let row: Vec<f64> = rows
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes(*b)))
            .collect();
        let length = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (length - scale).abs() < 1e-5,
            "step {k}: the length {length}, the scale {scale}"
        );
    }
    // The pass of the last prompt token and the steps before the last name the tap; the last step names none.
    let logged = lab.log.take();
    let tapped = logged.passes.iter().filter(|(_, taps)| taps == &["residual"]).count();
    assert_eq!(tapped, 3, "{:?}", logged.passes);
    // The reply's first token takes the position after the last latent row.
    let after: Vec<_> = segments
        .iter()
        .filter(|s| s.positions.first() > Some(&(first + 2)))
        .collect();
    assert!(after.iter().all(|s| s.meta.is_empty()), "the reply feeds tokens");
}

/// A head names the tokens of its slot, and a latent row is not a token. So a reply after latent steps holds no head,
/// and the store keeps only the seals of the build.
#[test]
fn latent_steps_hold_no_head_and_give_no_read_without_read() {
    let lab = lab_with(true);
    let (status, got) = send(&lab.council, &request(Some(json!({"steps": 2}))));
    assert_eq!(status, StatusCode::OK, "{got}");
    assert_eq!(got["latent"], json!({"steps": 2}));
    let plain = lab_with(true);
    let (_, want) = send(&plain.council, &request(None));
    assert_eq!(
        got["usage"]["prompt"], want["usage"]["prompt"],
        "a latent row is not a prompt token"
    );
    // The latent reply holds no more than the plain reply of the same request: the seals of the build.
    assert!(
        lab.fixture.entries() <= plain.fixture.entries(),
        "the latent reply holds {} entries, and the plain reply {}",
        lab.fixture.entries(),
        plain.fixture.entries()
    );
}

#[test]
fn each_fault_of_latent_is_a_400_that_names_its_member() {
    let lab = lab_with(true);
    let cases = [
        ("latent.steps", json!({"steps": 257})),
        ("latent.steps", json!({"steps": 1.5})),
        ("latent.steps", json!({"steps": -1})),
        ("latent.steps", json!({"steps": "4"})),
        ("latent.read", json!({"steps": 4, "read": "yes"})),
        ("latent.depth", json!({"steps": 4, "depth": 2})),
        ("latent", json!(4)),
    ];
    for (member, latent) in cases {
        let (status, got) = send(&lab.council, &request(Some(latent.clone())));
        assert_eq!(status, StatusCode::BAD_REQUEST, "{latent}: {got}");
        assert_eq!(got["error"]["param"], member, "{latent}: {got}");
    }
    assert!(lab.log.take().passes.is_empty(), "a refused request runs no pass");
    let (status, got) = send(&lab.council, &request(Some(json!({"steps": 256}))));
    assert_eq!(status, StatusCode::OK, "256 steps is the most: {got}");
}

#[test]
fn a_model_with_no_tap_residual_is_a_400_that_names_latent() {
    let lab = lab_with(false);
    // The check reads no weights: the folder of a model with no tap need not hold them.
    std::fs::remove_file(lab.folder.path().join("model.safetensors")).unwrap();
    let (status, got) = send(&lab.council, &request(Some(json!({"steps": 1}))));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    assert_eq!(got["error"]["param"], "latent", "{got}");
    assert!(lab.log.take().passes.is_empty(), "a refused request runs no pass");
    // With 0 steps, the request is the request with no member, so it runs.
    let (status, got) = send(&lab.council, &request(Some(json!({"steps": 0}))));
    assert_eq!(status, StatusCode::OK, "{got}");
}

// Soft inputs (docs/lab.md, "Soft inputs in a request"). They read the table of the same model folder.

fn soft_request(soft: Value, temperature: f64, thinking: bool) -> Value {
    let mut body = request(None);
    body["sample"]["temperature"] = json!(temperature);
    body["contexts"][0]["thinking"] = json!(thinking);
    body["soft"] = soft;
    body
}

/// The rows of each segment with a latent span, as `f32` values.
fn latent_rows(segments: &[nagaya_abi::Segment]) -> Vec<Vec<f32>> {
    segments
        .iter()
        .filter(|s| latent_span(&s.meta).is_some())
        .map(|s| {
            let SegmentInputs::TokensAndRows { rows, .. } = &s.inputs else {
                panic!("a latent span feeds tokens and rows");
            };
            rows.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
        })
        .collect()
}

/// At temperature 0 the sampler keeps one survivor, so each mix is one row and the pass feeds the token. Thus `moi`
/// gives the reply and the passes of the same request with no `soft`.
#[test]
fn moi_at_temperature_0_gives_the_reply_and_the_passes_of_no_member() {
    let plain = lab_with(true);
    let mut body = request(None);
    body["sample"]["temperature"] = json!(0.0);
    let (status, want) = send(&plain.council, &body);
    assert_eq!(status, StatusCode::OK, "{want}");
    let passes = plain.log.take();

    let lab = lab_with(true);
    let (status, got) = send(&lab.council, &soft_request(json!({"map": "moi"}), 0.0, false));
    assert_eq!(status, StatusCode::OK, "{got}");
    for member in ["content", "finish", "usage"] {
        assert_eq!(got[member], want[member], "{member}");
    }
    assert_eq!(
        got["soft"],
        json!({"map": "moi", "beta": 1.0, "top": 16, "span": "all", "mixed": 0, "mean_share": 0.0})
    );
    let logged = lab.log.take();
    assert_eq!((logged.sizes, logged.passes), (passes.sizes, passes.passes));
    assert!(lab.seen.take().iter().all(|s| s.meta.is_empty()), "no latent span");
}

/// With a spread distribution, a token after another takes a soft row: a latent span at its position, with a row that
/// is no row of the table.
#[test]
fn moi_feeds_a_mixed_row_after_an_uncertain_token() {
    let lab = lab_with(true);
    let (status, got) = send(&lab.council, &soft_request(json!({"map": "moi", "top": 4}), 1.0, false));
    assert_eq!(status, StatusCode::OK, "{got}");
    let mixed = got["soft"]["mixed"].as_u64().unwrap();
    let share = got["soft"]["mean_share"].as_f64().unwrap();
    assert!(mixed >= 1, "{got}");
    assert!(share > 0.0 && share < 1.0, "{got}");
    let rows = latent_rows(&lab.seen.take());
    assert_eq!(rows.len() as u64, mixed, "one latent span for each soft row");
    for row in &rows {
        assert_eq!(row.len(), WIDTH);
        let is_table_row = (0..VOCAB).any(|r| (0..WIDTH).all(|c| row[c] == table(r, c)));
        assert!(!is_table_row, "a soft row mixes rows: {row:?}");
    }
    // A soft row is not a token, so the reply holds no head: only the seals of the build.
    let plain = lab_with(true);
    let mut body = request(None);
    body["sample"]["temperature"] = json!(1.0);
    let (status, want) = send(&plain.council, &body);
    assert_eq!(status, StatusCode::OK, "{want}");
    assert!(
        lab.fixture.entries() < plain.fixture.entries(),
        "the soft reply holds {} entries, and the plain reply {}",
        lab.fixture.entries(),
        plain.fixture.entries()
    );
}

/// With a small `tau`, the Gumbel-softmax puts almost all the weight on its token, so each soft row is near one row
/// of the table.
#[test]
fn gumbel_with_a_small_tau_feeds_rows_near_the_table() {
    let lab = lab_with(true);
    let (status, got) = send(
        &lab.council,
        &soft_request(json!({"map": "gumbel", "tau": 0.05, "top": 4}), 1.0, false),
    );
    assert_eq!(status, StatusCode::OK, "{got}");
    assert_eq!(got["soft"]["tau"], 0.05);
    let rows = latent_rows(&lab.seen.take());
    assert_eq!(rows.len() as u64, got["soft"]["mixed"].as_u64().unwrap());
    assert!(!rows.is_empty(), "{got}");
    for row in rows {
        let near = (0..VOCAB).any(|r| (0..WIDTH).all(|c| (row[c] - table(r, c)).abs() < 0.05));
        assert!(near, "a soft row of a small tau is near a row of the table: {row:?}");
    }
}

/// With the span `think`, only a token of the reasoning takes a soft row. A reply with no reasoning takes none, and
/// the sampler picks each token, so `gumbel` gives the reply of the same request with no `soft`.
#[test]
fn the_span_think_mixes_only_the_reasoning() {
    let plain = lab_with(true);
    let mut body = request(None);
    body["sample"]["temperature"] = json!(1.0);
    let (_, want) = send(&plain.council, &body);
    let lab = lab_with(true);
    let (status, got) = send(
        &lab.council,
        &soft_request(json!({"map": "gumbel", "span": "think"}), 1.0, false),
    );
    assert_eq!(status, StatusCode::OK, "{got}");
    assert_eq!(got["soft"]["mixed"], 0, "{got}");
    assert_eq!(got["content"], want["content"]);
    assert!(lab.seen.take().iter().all(|s| s.meta.is_empty()), "no latent span");

    let thinking = lab_with(true);
    let (status, got) = send(
        &thinking.council,
        &soft_request(json!({"map": "moi", "span": "think"}), 1.0, true),
    );
    assert_eq!(status, StatusCode::OK, "{got}");
    assert!(got["soft"]["mixed"].as_u64().unwrap() >= 1, "{got}");
}

#[test]
fn each_fault_of_soft_is_a_400_that_names_its_member() {
    let lab = lab_with(true);
    let cases = [
        ("soft.map", json!({"map": "coconut"})),
        ("soft.map", json!({})),
        ("soft.tau", json!({"map": "moi", "tau": 0.5})),
        ("soft.beta", json!({"map": "gumbel", "beta": 1.0})),
        ("soft.beta", json!({"map": "moi", "beta": -1.0})),
        ("soft.tau", json!({"map": "gumbel", "tau": 0.0})),
        ("soft.top", json!({"map": "moi", "top": 0})),
        ("soft.top", json!({"map": "moi", "top": 65})),
        ("soft.span", json!({"map": "moi", "span": "reply"})),
        ("soft", json!("moi")),
    ];
    for (member, soft) in cases {
        let (status, got) = send(&lab.council, &soft_request(soft.clone(), 1.0, false));
        assert_eq!(status, StatusCode::BAD_REQUEST, "{soft}: {got}");
        assert_eq!(got["error"]["param"], member, "{soft}: {got}");
    }
    let mut two = soft_request(json!({"map": "moi"}), 1.0, false);
    let context = two["contexts"][0].clone();
    two["contexts"].as_array_mut().unwrap().push(context);
    let (status, got) = send(&lab.council, &two);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    assert_eq!(got["error"]["param"], "soft", "{got}");
    let mut weighted = soft_request(json!({"map": "moi"}), 1.0, false);
    weighted["weights"] = json!([2.0]);
    let (status, got) = send(&lab.council, &weighted);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    assert_eq!(got["error"]["param"], "weights", "{got}");
    assert!(lab.log.take().passes.is_empty(), "a refused request runs no pass");
}
