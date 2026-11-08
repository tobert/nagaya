//! The lab route `trace` of a process with `LFM3-1.7B` on card 0 (`docs/lab.md`, "The rules of `trace`", milestone 26
//! step 2): a trace gives the response and the tokens of the same generation, `along` after an ablate of 1 is 0, the
//! lens of the last layer is the head, and the times of 200 steps. The oracle is `lfm3.control.safetensors` of
//! `tools/lfm3_golden.sh`, in `LFM3_GOLDEN_DIR` or `target/lfm3-golden`.
//!
//! It needs card 0 and `LFM3_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lab_trace -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::Registry;
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, router};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options};
use nagaya::probe::FileProbe;
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use safetensors::SafeTensors;
use serde_json::{Value, json};
use tower::ServiceExt;

/// The iGPU of b70host drives the display (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";
const LAYERS: usize = 24;
const WIDTH: usize = 2048;
/// The gate of a relative error and of a logit (`crates/nagaya-mk-lfm3-bmg/tests/common/mod.rs`).
const STAGE_TOL: f64 = 1e-4;
const LOGIT_TOL: f64 = 1e-3;

fn directory(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the model folder");
            None
        }
    }
}

/// The oracle of control vectors: its header, its data and its `meta`. `None` when it is absent.
fn oracle() -> Option<(serde_json::Map<String, Value>, Vec<u8>, Value)> {
    let dir = std::env::var_os("LFM3_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"));
    let path = dir.join("lfm3.control.safetensors");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") {
                panic!("{} is absent and NAGAYA_REQUIRE_MODELS is set", path.display());
            }
            eprintln!(
                "skipped: {} is absent; write it with tools/lfm3_golden.sh",
                path.display()
            );
            return None;
        }
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    };
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let meta: Value = serde_json::from_str(header["__metadata__"]["meta"].as_str().unwrap()).unwrap();
    Some((header, bytes[8 + n..].to_vec(), meta))
}

/// The values of one tensor of the oracle, as `f64`. The oracle holds F32 and I64.
fn oracle_tensor(header: &serde_json::Map<String, Value>, data: &[u8], name: &str) -> Vec<f64> {
    let t = &header[name];
    let (a, b) = (
        t["data_offsets"][0].as_u64().unwrap() as usize,
        t["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let bytes = &data[a..b];
    match t["dtype"].as_str().unwrap() {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect(),
        "I64" => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f64)
            .collect(),
        other => panic!("{name} has dtype {other}"),
    }
}

/// `LFM3-1.7B` resident on device 0, with one slot and an object store, as in `xe_lfm3.rs`.
fn profile(dir: &Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"lfm3\"\nkind = \"mk-lfm3-bmg\"\ndevices = [0]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 1073741824\n"
    ))
    .expect("the profile parses")
}

fn send_raw(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: Vec<u8>,
    mime: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 30).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn send(council: &Arc<Council>, path: &str, body: &Value) -> Value {
    let (status, bytes) = send_raw(council, "POST", path, body.to_string().into_bytes(), None);
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    assert_eq!(status, StatusCode::OK, "{path}: {value}");
    value
}

fn user(text: &str, control: Value) -> Value {
    json!({"thinking": false, "control": control, "messages": [{"role": "user", "content": text}]})
}

fn generation(context: &Value, max_tokens: u64) -> Value {
    json!({"model": "lfm3", "context": context, "max_tokens": max_tokens, "sample": {"temperature": 0, "seed": 7}})
}

/// The id and the entropy of each token of a streamed generation.
fn streamed(council: &Arc<Council>, context: &Value, max_tokens: u64) -> Vec<(u32, f64)> {
    let mut body = generation(context, max_tokens);
    body["stream"] = json!(true);
    let (status, bytes) = send_raw(
        council,
        "POST",
        "/council/v2/generations",
        body.to_string().into_bytes(),
        None,
    );
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    let mut tokens = Vec::new();
    for event in text.split("\n\n") {
        let (mut name, mut data) = ("", "");
        for line in event.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                name = rest.trim();
            } else if let Some(rest) = line.strip_prefix("data:") {
                data = rest.trim();
            }
        }
        assert_ne!(name, "error", "{data}");
        if name == "token" {
            let token: Value = serde_json::from_str(data).unwrap();
            tokens.push((token["id"].as_u64().unwrap() as u32, token["h"].as_f64().unwrap()));
        }
    }
    tokens
}

/// The tensors of a trace object, by name: the shape and the bytes.
fn object(council: &Arc<Council>, id: &str) -> BTreeMap<String, (Vec<usize>, Vec<u8>)> {
    let (status, bytes) = send_raw(council, "GET", &format!("/council/v2/objects/{id}"), Vec::new(), None);
    assert_eq!(status, StatusCode::OK);
    let st = SafeTensors::deserialize(&bytes).unwrap();
    st.tensors()
        .into_iter()
        .map(|(name, view)| (name, (view.shape().to_vec(), view.data().to_vec())))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn without_trace(mut response: Value) -> Value {
    response.as_object_mut().unwrap().remove("trace");
    response
}

#[test]
fn a_trace_on_the_card_gives_the_generation_and_its_numbers() {
    let Some(dir) = directory("LFM3_MODEL_DIR") else { return };
    let Some((header, data, meta)) = oracle() else { return };
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let profile = profile(&dir);
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let ready = lifecycle::start_with(
        profile.clone(),
        Registry::compiled_in(),
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens the engine");
    let ports = ready.process.ports();
    let council = Arc::new(
        council(
            &FamilyRegistry::compiled_in(),
            &profile,
            &ready.opened,
            &ports,
            store.path(),
            &[],
            None,
        )
        .expect("the process serves the model")
        .with_loads(ports.clone()),
    );
    let (status, bytes) = send_raw(&council, "GET", "/council/v2/identity", Vec::new(), None);
    assert_eq!(status, StatusCode::OK);
    let identity: Value = serde_json::from_slice(&bytes).unwrap();
    let weight_hash = identity["models"][0]["weight_hash"].as_str().unwrap().to_owned();
    // The oracle's vector as a control vector object of the served weights.
    let vector = oracle_tensor(&header, &data, "vector");
    let layers: BTreeMap<u32, Vec<f32>> = (0..LAYERS as u32)
        .map(|l| {
            (
                l,
                vector[l as usize * WIDTH..(l as usize + 1) * WIDTH]
                    .iter()
                    .map(|v| *v as f32)
                    .collect(),
            )
        })
        .collect();
    let bytes = gogi_proto::control::ControlVector {
        model: "lfm3".to_owned(),
        weight_hash,
        layers: layers.clone(),
    }
    .to_bytes();
    let v = gogi_proto::content_id(&bytes);
    let (status, _) = send_raw(
        &council,
        "PUT",
        &format!("/council/v2/objects/{v}"),
        bytes,
        Some(gogi_proto::control::MEDIA_TYPE),
    );
    assert!(status.is_success());
    let units: Vec<Vec<f64>> = layers
        .values()
        .map(|row| {
            let n = row.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
            row.iter().map(|x| *x as f64 / n).collect()
        })
        .collect();
    let test = meta["test"].as_str().unwrap();
    let all = json!({"layers": "all", "vectors": [v], "top_k": 8, "lens": "logit"});
    for (name, control) in [
        ("plain", json!([])),
        ("ablate 1", json!([{"vector": v, "op": "ablate", "strength": 1.0}])),
    ] {
        let context = user(test, control);
        // The first generation fills the store. The trace and the second generation start from the same state.
        send(&council, "/council/v2/generations", &generation(&context, 48));
        let mut body = generation(&context, 48);
        body["trace"] = all.clone();
        let started = Instant::now();
        let traced = send(&council, "/nagaya/v1/lab/trace", &body);
        let ms = started.elapsed().as_millis();
        let again = send(&council, "/council/v2/generations", &generation(&context, 48));
        assert_eq!(
            without_trace(traced.clone()),
            again,
            "{name}: the trace changes the response"
        );
        let stream = streamed(&council, &context, 48);
        let t = object(&council, traced["trace"].as_str().unwrap());
        let tokens = u32s(&t["tokens"].1);
        let entropy = f32s(&t["entropy"].1);
        let s = tokens.len();
        println!("{name}: a trace of {s} steps in {ms} ms");
        assert_eq!(t["along"].0, [s, LAYERS, 1]);
        assert_eq!(t["lens_ids"].0, [s, LAYERS, 8]);
        let ids: Vec<u32> = stream.iter().map(|(id, _)| *id).collect();
        assert_eq!(tokens, ids, "{name}: the trace gives the tokens of the stream");
        for (i, (_, h)) in stream.iter().enumerate() {
            assert_eq!(
                entropy[i], *h as f32,
                "{name} step {i}: the entropy is not h of the token event"
            );
        }
        let (norms, along) = (f32s(&t["norms"].1), f32s(&t["along"].1));
        let (top, top_lp) = (u32s(&t["top_ids"].1), f32s(&t["top_logprobs"].1));
        let (lens_ids, lens_lp) = (u32s(&t["lens_ids"].1), f32s(&t["lens_logprobs"].1));
        let (mut worst_along, mut worst_lens) = (0.0f64, 0.0f64);
        for i in 0..s {
            // The greedy token is the top entry of its step, and the lens of layer 23 gives it too.
            assert_eq!(top[i * 8], tokens[i], "{name} step {i}");
            let at = (i * LAYERS + LAYERS - 1) * 8;
            assert_eq!(
                lens_ids[at], tokens[i],
                "{name} step {i}: the lens of layer 23 is not the head"
            );
            worst_lens = worst_lens.max((lens_lp[at] as f64 - top_lp[i * 8] as f64).abs());
            for l in 0..LAYERS {
                worst_along = worst_along.max((along[i * LAYERS + l] as f64 / norms[i * LAYERS + l] as f64).abs());
            }
        }
        println!("{name}: worst |along|/norm {worst_along:.3e}; lens of layer 23 against the head {worst_lens:.3e}");
        assert!(
            worst_lens < LOGIT_TOL,
            "{name}: the top log probability of layer 23 differs by {worst_lens}"
        );
        if name == "plain" {
            // Step 0 is the last position of test.ids, where the oracle has its plain rows.
            let want = oracle_tensor(&header, &data, "plain.rows");
            let mut worst = 0.0f64;
            for (l, unit) in units.iter().enumerate() {
                let w = &want[l * WIDTH..(l + 1) * WIDTH];
                let wn = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                let wa: f64 = w.iter().zip(unit).map(|(a, b)| a * b).sum();
                worst = worst.max((along[l] as f64 - wa).abs() / wn);
            }
            println!("plain: along at step 0 against the oracle's rows, worst gap {worst:.3e} of the row");
            assert!(worst < STAGE_TOL, "along differs from the oracle by {worst} of the row");
        } else {
            assert!(
                worst_along < 1e-5,
                "after an ablate of 1, along is {worst_along} of the row"
            );
        }
    }
    // The times of 200 steps: a generation, a trace with no lens, and a trace of every layer with the lens.
    let story = user(
        "Write a long story about a dragon who lives under the sea. Use many details.",
        json!([]),
    );
    send(&council, "/council/v2/generations", &generation(&story, 200));
    let started = Instant::now();
    let plain = send(&council, "/council/v2/generations", &generation(&story, 200));
    let generated = started.elapsed().as_millis();
    let steps = plain["usage"]["completion"].as_u64().unwrap();
    for (what, trace) in [
        ("no lens", json!({"vectors": [v]})),
        ("the lens of 24 layers", json!({"vectors": [v], "lens": "logit"})),
        (
            "the lens of 4 layers",
            json!({"layers": [5, 11, 17, 23], "vectors": [v], "lens": "logit"}),
        ),
    ] {
        let mut body = generation(&story, 200);
        body["trace"] = trace;
        let started = Instant::now();
        let traced = send(&council, "/nagaya/v1/lab/trace", &body);
        let ms = started.elapsed().as_millis();
        assert_eq!(traced["usage"]["completion"].as_u64(), Some(steps));
        println!("{steps} steps: generation {generated} ms, trace with {what} {ms} ms");
    }
}
