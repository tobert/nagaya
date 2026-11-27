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
use nagaya_family_toy::ToyFamily;
use nagaya_store::objects::ObjectStore;
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
    // The puts of every test in this binary share the limit of puts at once, and a 429 busy says to try again.
    for _ in 0..200 {
        let (status, body) = send_raw(
            council,
            "PUT",
            &format!("/council/v2/objects/{id}"),
            bytes.clone(),
            Some(MEDIA_TYPE),
        );
        if status == StatusCode::TOO_MANY_REQUESTS {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        }
        assert!(status.is_success(), "{status}: {}", String::from_utf8_lossy(&body));
        return id;
    }
    panic!("the put of {id} stayed busy for a second")
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
    // The checks of the request come before the read of the lens, so a bad top_k names top_k.
    assert_refused(
        send(
            &none.council,
            json!({"model": "toy", "vector": v, "layers": [0], "through": "jlens", "top_k": 0}),
        ),
        StatusCode::BAD_REQUEST,
        "top_k",
    );
}

/// Send a request to `compose` (`docs/lab.md`, "compose").
fn compose(council: &Arc<Council>, body: Value) -> (StatusCode, Value) {
    let (status, bytes) = send_raw(
        council,
        "POST",
        "/nagaya/v1/lab/compose",
        body.to_string().into_bytes(),
        None,
    );
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// The layers of the control vector object `id`.
fn layers_of(council: &Arc<Council>, id: &str) -> BTreeMap<u32, Vec<f32>> {
    let (status, bytes) = send_raw(council, "GET", &format!("/council/v2/objects/{id}"), Vec::new(), None);
    assert!(status.is_success(), "{status}");
    let vector = ControlVector::parse(&bytes).unwrap();
    assert_eq!((vector.model.as_str(), vector.weight_hash.as_str()), ("toy", "fake:0"));
    vector.layers
}

fn close(got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((f64::from(*g) - w).abs() < 1e-5, "{got:?} and {want:?}");
    }
}

#[test]
fn compose_adds_weighted_vectors_at_their_layers_and_takes_a_direction_out() {
    let lab = council_lab(true);
    let a = put(
        &lab.council,
        "fake:0",
        &[(0, vec![1.0, 2.0, 0.0, 0.0]), (1, vec![0.0, 0.0, 3.0, 0.0])],
    );
    let b = put(&lab.council, "fake:0", &[(1, vec![1.0, 0.0, 0.0, 4.0])]);
    let x = put(&lab.council, "fake:0", &[(1, vec![0.0, 0.0, 1.0, 0.0])]);
    let (status, body) = compose(
        &lab.council,
        json!({"model": "toy", "parts": [
            {"vector": a, "weight": 2.0},
            {"vector": b, "weight": 0.5, "layers": [1]},
            {"vector": x, "remove": true, "layers": [1]}]}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["layers"], json!([0, 1]));
    let layers = layers_of(&lab.council, body["vector"].as_str().unwrap());
    close(&layers[&0], &[2.0, 4.0, 0.0, 0.0]);
    // Layer 1: 2 * (0, 0, 3, 0) + 0.5 * (1, 0, 0, 4), then the part along (0, 0, 1, 0) goes.
    close(&layers[&1], &[0.5, 0.0, 0.0, 2.0]);
    assert!(
        (body["norms"]["1"].as_f64().unwrap() - (0.25f64 + 4.0).sqrt()).abs() < 1e-6,
        "{body}"
    );
    // The result is a control vector of the model, so project reads it.
    let (status, _) = send(
        &lab.council,
        json!({"model": "toy", "vector": body["vector"], "through": "unembed"}),
    );
    assert_eq!(status, StatusCode::OK);
}

#[test]
fn a_words_part_adds_a_unit_direction_of_the_backward_lens_at_its_layers() {
    let lab = council_lab_with(true, Some(("fake:0", &[j0()])));
    let text = "hi";
    let ids = ToyFamily::new().tokenize(text).unwrap();
    let rows: Vec<f64> = (0..WIDTH)
        .map(|c| ids.iter().map(|id| f64::from(table(*id as usize, c))).sum::<f64>() * f64::from(NORM[c]))
        .collect();
    let unit = |v: Vec<f64>| {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.into_iter().map(|x| x / n).collect::<Vec<f64>>()
    };
    // The backward lens of layer 0 is J_0 transposed; layer 1 has the identity.
    let j = j0();
    let back: Vec<f64> = (0..WIDTH)
        .map(|c| (0..WIDTH).map(|r| f64::from(j[r * WIDTH + c]) * rows[r]).sum())
        .collect();
    for (through, want0) in [("unembed", unit(rows.clone())), ("jlens", unit(back))] {
        let (status, body) = compose(
            &lab.council,
            json!({"model": "toy", "parts": [{"words": [text], "through": through, "weight": 3.0, "layers": [0, 1]}]}),
        );
        assert_eq!(status, StatusCode::OK, "{through}: {body}");
        assert_eq!(body["words"][0]["ids"], json!(ids), "{body}");
        let layers = layers_of(&lab.council, body["vector"].as_str().unwrap());
        close(&layers[&0], &want0.iter().map(|x| 3.0 * x).collect::<Vec<_>>());
        close(
            &layers[&1],
            &unit(rows.clone()).iter().map(|x| 3.0 * x).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn compose_refuses_a_part_that_breaks_its_rules() {
    let lab = council_lab(true);
    let a = put(&lab.council, "fake:0", &[(0, vec![1.0, 2.0, 0.0, 0.0])]);
    let theirs = put(&lab.council, "fake:9", &[(0, vec![1.0, 2.0, 0.0, 0.0])]);
    for (parts, param) in [
        (json!([{"vector": a, "words": ["hi"]}]), "parts[0]"),
        (json!([{"weight": 1.0}]), "parts[0]"),
        (json!([{"vector": a, "remove": true, "weight": 1.0}]), "parts[0].weight"),
        (json!([{"words": ["hi"], "through": "unembed"}]), "parts[0].layers"),
        (
            json!([{"words": ["hi"], "through": "jlens", "layers": [0]}]),
            "parts[0].through",
        ),
        (
            json!([{"words": ["hi"], "through": "sideways", "layers": [0]}]),
            "parts[0].through",
        ),
        (json!([{"vector": a, "layers": [0, 0]}]), "parts[0].layers"),
        (json!([{"vector": a, "layers": [7]}]), "parts[0].layers"),
        (json!([{"vector": a, "strength": 1}]), "parts[0].strength"),
        (json!([{"vector": theirs}]), "parts[0].vector"),
        (json!([{"vector": a, "remove": true}]), "parts"),
        (json!([]), "parts"),
    ] {
        assert_refused(
            compose(&lab.council, json!({"model": "toy", "parts": parts})),
            StatusCode::BAD_REQUEST,
            param,
        );
    }
}

/// A council whose served model `toy` reads the model folder `folder`.
fn council_on(folder: &Path) -> (Fixture, Arc<Council>) {
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
    let council = Council::new(vec![served], None)
        .unwrap()
        .with_lab(BTreeMap::from([(ENGINE.to_owned(), folder.to_path_buf())]));
    (fixture, Arc::new(council))
}

/// `project` reads the head of Inkling 2 from its folder (`docs/lab.md`, "The rules of `project`", "Inkling 2"): the
/// first 200,058 rows of `unembed`, the weight of `norm`, and the end token 200,006. `compose` reads LFM3 only. The
/// test skips when the folder of Inkling 2 Mini is absent.
///
/// Falsified by a lab that reads every folder as LFM3: a 500.
#[test]
fn project_reads_the_head_of_inkling_2() {
    let Some(dir) = std::env::var_os("INKLING2_MODEL_DIR").map(std::path::PathBuf::from) else {
        eprintln!("skipped: project_reads_the_head_of_inkling_2: INKLING2_MODEL_DIR is not set");
        return;
    };
    let (_fixture, council) = council_on(&dir);
    let vector: Vec<f32> = (0..2304).map(|i| ((i * 13 % 17) as f32 - 8.0) / 4.0).collect();
    let body = json!({"model": "toy", "vector": vector, "through": "unembed", "top_k": 5});
    let (status, got) = send(&council, body);
    assert_eq!(status, StatusCode::OK, "{got}");
    let layer = &got["layers"][0];
    let head = nagaya_mk_inkling2_bmg::lab::Head::open(&dir).unwrap();
    let scores = head.scores(&[&vector]).remove(0);
    let top = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .unwrap();
    assert_eq!(layer["up"][0]["id"], top.0 as u64, "{layer}");
    assert_eq!(layer["up"][0]["score"].as_f64().unwrap() as f32, *top.1);
    assert_eq!(layer["end"][0]["id"], 200_006);
    assert_eq!(layer["up"].as_array().unwrap().len(), 5);
    let compose = json!({"model": "toy", "parts": [{"vector": "blake3:0000", "weight": 1.0}]});
    let (status, bytes) = send_raw(
        &council,
        "POST",
        "/nagaya/v1/lab/compose",
        compose.to_string().into_bytes(),
        None,
    );
    let got: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    assert_eq!(got["error"]["param"], "model", "{got}");
}
