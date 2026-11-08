//! The lab route `project` (`docs/lab.md`, "The rules of `project`") with the fake engine and the toy family. The
//! council reads a small LFM3 checkpoint in a temporary folder as the weights of the fake's served model.

mod council_support;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, Fixture, TermFactory, lens_file};
use gogi_proto::control::{ControlVector, MEDIA_TYPE};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya_council::family::Family;
use nagaya_store::objects::ObjectStore;
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// The rows of the small head: the toy vocabulary and 3 rows past it.
const ROWS: usize = gogi_toy::VOCAB + 3;
/// The width and the layers of the fake's control table.
const WIDTH: usize = 4;
/// The end token of the small checkpoint.
const END: u32 = 7;
/// The weight of the final norm.
const NORM: [f32; WIDTH] = [2.0, 0.5, -1.0, 1.0];
const PATH: &str = "/nagaya/v1/lab/project";

/// The value of the row `r` at the column `c`: small integers, which BF16 holds exactly.
fn table(r: usize, c: usize) -> f32 {
    ((r * 7 + c * 3) % 11) as f32 - 5.0
}

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Write the small checkpoint of `LFM3-1.7B` into `dir`.
fn checkpoint(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        r#"{"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": 2, "hidden_size": 4, "norm_eps": 0.00001}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        format!("{{\"eos_token_id\": [{END}]}}"),
    )
    .unwrap();
    let flat: Vec<f32> = (0..ROWS).flat_map(|r| (0..WIDTH).map(move |c| table(r, c))).collect();
    let (embed, norm) = (bf16(&flat), bf16(&NORM));
    let header = json!({
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [ROWS, WIDTH], "data_offsets": [0, embed.len()]},
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

struct Lab {
    _fixture: Fixture,
    council: Arc<Council>,
    _weights: tempfile::TempDir,
}

/// A council whose served model `toy` reads the small checkpoint when `weights`, with an object store. With no
/// `weights`, the folder belongs to an engine that serves no model here. With `lens`, the profile names a J-Lens
/// file of these weights and matrices.
fn council_lab(weights: bool) -> Lab {
    council_lab_with(weights, None)
}

fn council_lab_with(weights: bool, lens: Option<(&str, &[Vec<f32>])>) -> Lab {
    let factory = TermFactory {
        control: Some(4),
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
    checkpoint(dir.path());
    let objects = ObjectStore::open(&fixture.directory.path().join("objects"), 1 << 24).unwrap();
    let council = Council::new(vec![served], None).unwrap().with_objects(objects);
    // With no weights, the folder belongs to another engine, so the served model `toy` has none.
    let engine = if weights { ENGINE } else { "another-engine" };
    let mut council = council.with_lab(BTreeMap::from([(engine.to_owned(), dir.path().to_path_buf())]));
    if let Some((hash, matrices)) = lens {
        let path = dir.path().join("lens.safetensors");
        lens_file(&path, hash, matrices, WIDTH);
        council = council.with_lenses(BTreeMap::from([(engine.to_owned(), path)]));
    }
    Lab {
        _fixture: fixture,
        council: Arc::new(council),
        _weights: dir,
    }
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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn send(council: &Arc<Council>, body: Value) -> (StatusCode, Value) {
    let (status, bytes) = send_raw(council, "POST", PATH, body.to_string().into_bytes(), None);
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Put a control vector of `weight_hash` with `layers`, and give its id.
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

/// The score of each row for the direction `v`, in `f64`: the row times the norm weight times `v`.
fn want_scores(v: &[f32]) -> Vec<f32> {
    (0..ROWS)
        .map(|r| {
            (0..WIDTH)
                .map(|c| table(r, c) as f64 * (NORM[c] * v[c]) as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// The ids of the `k` largest scores, a tie to the smaller id.
fn want_up(scores: &[f32], k: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..scores.len() as u32).collect();
    ids.sort_by(|a, b| scores[*b as usize].total_cmp(&scores[*a as usize]).then(a.cmp(b)));
    ids.truncate(k);
    ids
}

fn assert_refused(response: (StatusCode, Value), status: StatusCode, param: &str) {
    let (got, body) = response;
    assert_eq!(got, status, "{body}");
    assert_eq!(body["error"]["param"], param, "{body}");
}

#[test]
fn project_gives_the_ends_of_the_scores_of_each_layer() {
    let lab = council_lab(true);
    let (zero, one) = (vec![0.0; WIDTH], vec![1.0, -2.0, 0.5, 3.0]);
    let two = vec![-1.0, 0.25, 2.0, -0.5];
    let id = put(&lab.council, "fake:0", &[(0, zero), (1, one.clone())]);

    // With no layers, the route skips the layer of zeros.
    let (status, body) = send(
        &lab.council,
        json!({"model": "toy", "vector": id, "through": "unembed", "top_k": 5}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        (body["model"].as_str(), body["through"].as_str()),
        (Some("toy"), Some("unembed"))
    );
    let layers = body["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1, "{body}");
    assert_eq!(layers[0]["layer"], 1);
    let scores = want_scores(&one);
    let up: Vec<u64> = layers[0]["up"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["id"].as_u64().unwrap())
        .collect();
    let want: Vec<u64> = want_up(&scores, 5).into_iter().map(u64::from).collect();
    assert_eq!(up, want);
    for entry in layers[0]["up"].as_array().unwrap() {
        let id = entry["id"].as_u64().unwrap() as u32;
        assert_eq!(entry["score"].as_f64().unwrap() as f32, scores[id as usize], "{entry}");
        let token = ToyFamily::new().pieces(&[id]).unwrap().pop().unwrap();
        assert_eq!(entry["token"], token, "{entry}");
    }
    let negated: Vec<f32> = scores.iter().map(|s| -s).collect();
    let down: Vec<u64> = layers[0]["down"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["id"].as_u64().unwrap())
        .collect();
    let want: Vec<u64> = want_up(&negated, 5).into_iter().map(u64::from).collect();
    assert_eq!(down, want);
    let end = &layers[0]["end"][0];
    let rank = want_up(&scores, ROWS).iter().position(|id| *id == END).unwrap();
    assert_eq!(
        (end["id"].as_u64(), end["rank"].as_u64()),
        (Some(END as u64), Some(rank as u64)),
        "{end}"
    );
    assert_eq!(end["score"].as_f64().unwrap() as f32, scores[END as usize]);

    // `layers` names the layers in the order of the response.
    let (status, body) = send(
        &lab.council,
        json!({"model": "toy", "vector": id, "through": "unembed",
                                                   "layers": [1, 0]}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    let named: Vec<&Value> = body["layers"].as_array().unwrap().iter().map(|l| &l["layer"]).collect();
    assert_eq!(named, [&json!(1), &json!(0)]);
    assert_eq!(
        body["layers"][0]["up"].as_array().unwrap().len(),
        20,
        "the default top_k is 20"
    );

    // A list of numbers is one direction, with no layer. The two ends of 256 cover each of the 275 rows.
    let (status, body) = send(
        &lab.council,
        json!({"model": "toy", "vector": two, "through": "unembed", "top_k": 256}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    let layer = &body["layers"][0];
    assert_eq!(layer["layer"], Value::Null);
    let entries: Vec<&Value> = ["up", "down"]
        .iter()
        .flat_map(|end| layer[*end].as_array().unwrap())
        .collect();
    let mut seen: Vec<u64> = entries.iter().map(|e| e["id"].as_u64().unwrap()).collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), ROWS);
    for entry in entries {
        let past = entry["id"].as_u64().unwrap() as usize >= gogi_toy::VOCAB;
        assert_eq!(
            entry["token"].is_null(),
            past,
            "an id past the vocabulary has no token: {entry}"
        );
    }
    let (status, body) = send(
        &lab.council,
        json!({"model": "toy", "vector": two, "through": "unembed",
                                                   "layers": [1]}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["layers"][0]["layer"], 1);
}

#[test]
fn the_same_request_gives_the_same_bytes() {
    let lab = council_lab(true);
    let id = put(
        &lab.council,
        "fake:0",
        &[(0, vec![0.3, -1.7, 2.2, 0.01]), (1, vec![1.0, 1.0, 1.0, 1.0])],
    );
    let body = json!({"model": "toy", "vector": id, "through": "unembed", "top_k": 64}).to_string();
    let first = send_raw(&lab.council, "POST", PATH, body.clone().into_bytes(), None);
    let second = send_raw(&lab.council, "POST", PATH, body.into_bytes(), None);
    assert_eq!(first.0, StatusCode::OK);
    assert_eq!(first, second);
}

#[test]
fn project_refuses_what_breaks_its_rules() {
    let lab = council_lab(true);
    let c = &lab.council;
    let id = put(c, "fake:0", &[(0, vec![1.0, 0.0, 0.0, 0.0])]);
    let ok = json!({"model": "toy", "vector": id, "through": "unembed"});
    let with = |key: &str, value: Value| {
        let mut body = ok.clone();
        body[key] = value;
        body
    };
    let bad = StatusCode::BAD_REQUEST;
    assert_refused(send(c, with("model", json!("nope"))), bad, "model");
    assert_refused(send(c, with("through", json!("jlens"))), bad, "through");
    let mut body = ok.clone();
    body.as_object_mut().unwrap().remove("through");
    assert_refused(send(c, body), bad, "through");
    assert_refused(send(c, with("extra", json!(1))), bad, "extra");
    assert_refused(send(c, with("top_k", json!(257))), bad, "top_k");
    assert_refused(send(c, with("top_k", json!(0))), bad, "top_k");
    assert_refused(send(c, with("vector", json!([1.0, 2.0, 3.0]))), bad, "vector");
    assert_refused(send(c, with("vector", json!([1.0, 2.0, "x", 4.0]))), bad, "vector");
    assert_refused(send(c, with("vector", json!([1.0, 2.0, 1e39, 4.0]))), bad, "vector");
    assert_refused(send(c, with("vector", json!(7))), bad, "vector");
    assert_refused(send(c, with("vector", json!("blake3:zz"))), bad, "vector");
    let unheld = gogi_proto::content_id(b"no such vector");
    assert_refused(send(c, with("vector", json!(unheld))), StatusCode::NOT_FOUND, "vector");
    let other = put(c, "fake:other", &[(0, vec![1.0, 0.0, 0.0, 0.0])]);
    assert_refused(send(c, with("vector", json!(other))), bad, "vector");
    let wide = put(c, "fake:0", &[(0, vec![1.0; 5])]);
    assert_refused(send(c, with("vector", json!(wide))), bad, "vector");
    let text = gogi_proto::content_id(b"text");
    let (status, _) = send_raw(
        c,
        "PUT",
        &format!("/council/v2/objects/{text}"),
        b"text".to_vec(),
        Some("text/plain"),
    );
    assert!(status.is_success());
    assert_refused(send(c, with("vector", json!(text))), bad, "vector");
    assert_refused(send(c, with("layers", json!([1]))), bad, "layers");
    assert_refused(send(c, with("layers", json!([0, 0]))), bad, "layers");
    assert_refused(send(c, with("layers", json!([]))), bad, "layers");
    let list = with("vector", json!([1.0, 2.0, 3.0, 4.0]));
    let mut far = list.clone();
    far["layers"] = json!([2]);
    assert_refused(send(c, far), bad, "layers");
    let mut two = list;
    two["layers"] = json!([0, 1]);
    assert_refused(send(c, two), bad, "layers");

    // A served model whose engine has no weights for the lab is a 400 that names `model`.
    let plain = council_lab(false);
    assert_refused(send(&plain.council, ok), bad, "model");
}

/// The J-Lens of layer 0 of the small model. The lens of layer 1, the last, is the identity.
fn j0() -> Vec<f32> {
    vec![
        1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0,
    ]
}

#[test]
fn project_through_the_jlens_moves_the_vector_by_the_matrix_of_its_layer() {
    let lab = council_lab_with(true, Some(("fake:0", &[j0()])));
    let v = [1.0, -2.0, 0.5, 3.0];
    // J_0 v, by hand.
    let moved = [2.0, -2.0, -0.5, 3.5];
    let project = |vector: &[f32], layer: u32, through: &str| {
        let (status, body) = send(
            &lab.council,
            json!({"model": "toy", "vector": vector, "layers": [layer], "through": through, "top_k": 4}),
        );
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    };
    let lensed = project(&v, 0, "jlens");
    assert_eq!(lensed["through"], "jlens");
    let plain = project(&moved, 0, "unembed");
    assert_eq!(lensed["layers"], plain["layers"]);
    let up: Vec<u64> = lensed["layers"][0]["up"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["id"].as_u64().unwrap())
        .collect();
    let want: Vec<u64> = want_up(&want_scores(&moved), 4).into_iter().map(u64::from).collect();
    assert_eq!(up, want);
    // The last layer has the identity lens.
    assert_eq!(project(&v, 1, "jlens")["layers"], project(&v, 1, "unembed")["layers"]);
}

#[test]
fn project_through_the_jlens_refuses_a_model_with_no_lens_a_lens_of_other_weights_and_a_list_with_no_layer() {
    let v = json!([1.0, -2.0, 0.5, 3.0]);
    let none = council_lab(true);
    assert_refused(
        send(
            &none.council,
            json!({"model": "toy", "vector": v, "layers": [0], "through": "jlens"}),
        ),
        StatusCode::BAD_REQUEST,
        "through",
    );
    let other = council_lab_with(true, Some(("blake3:00", &[j0()])));
    assert_refused(
        send(
            &other.council,
            json!({"model": "toy", "vector": v, "layers": [0], "through": "jlens"}),
        ),
        StatusCode::BAD_REQUEST,
        "through",
    );
    let lab = council_lab_with(true, Some(("fake:0", &[j0()])));
    assert_refused(
        send(&lab.council, json!({"model": "toy", "vector": v, "through": "jlens"})),
        StatusCode::BAD_REQUEST,
        "layers",
    );
}
