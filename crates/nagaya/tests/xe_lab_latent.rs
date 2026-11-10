//! Latent steps of `LFM3-1.7B` on card 0 through the lab route `ensemble` (`docs/lab.md`, "Latent steps in a request",
//! milestone 29 step 4). The oracle is `lfm3.latent.safetensors` of `tools/lfm3_golden.sh` (`run_latent`), in
//! `LFM3_GOLDEN_DIR` or `target/lfm3-golden`: 16 latent steps after the prompt, then 8 greedy tokens.
//!
//! The gate: the top token of each step is the oracle's, and the reply is the oracle's 8 tokens. A token may differ
//! only at a near tie of the oracle's logits, within `TIE` of its top value. A latent row is not snapped to a token, so
//! a gap of the card grows over the steps. The test prints the count of steps that match.
//!
//! It needs card 0 and `LFM3_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lab_latent -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

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
use serde_json::{Value, json};
use tower::ServiceExt;

/// The iGPU of b70host drives the display (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";
/// The prompt of `run_latent` (`tools/lfm3_reference.py`, `LATENT_PROMPT`).
const PROMPT: &str = "Plan a short story about a lighthouse keeper who finds a map. Think before you write.";
const STEPS: usize = 16;
const THEN: usize = 8;
/// The room of a near tie in the oracle's logits. The logit gate of LFM3 is 1e-3 (`LOGIT_TOL` of `xe_lab_trace.rs`);
/// 16 steps with no snap to a token can grow the gap, so a tie here is 1e-2 of the oracle's top value.
const TIE: f64 = 1e-2;

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

/// The tensors of the latent oracle, by name, as `f64`. `None` when the file is absent.
fn oracle() -> Option<std::collections::BTreeMap<String, Vec<f64>>> {
    let dir = std::env::var_os("LFM3_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"));
    let path = dir.join("lfm3.latent.safetensors");
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
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    Some(
        st.tensors()
            .into_iter()
            .map(|(name, view)| {
                let values = match view.dtype() {
                    safetensors::Dtype::F32 => view
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f64::from(f32::from_le_bytes(*c)))
                        .collect(),
                    safetensors::Dtype::I64 => view
                        .data()
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| i64::from_le_bytes(*c) as f64)
                        .collect(),
                    other => panic!("{name} has dtype {other:?}"),
                };
                (name, values)
            })
            .collect(),
    )
}

/// `LFM3-1.7B` resident on device 0, with one slot, as in `xe_lab_trace.rs`.
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

fn send(council: &Arc<Council>, path: &str, body: &Value) -> Value {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 30).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        assert_eq!(status, StatusCode::OK, "{path}: {value}");
        value
    })
}

/// True when `id` is the oracle's top token of a row, or a near tie of it: an id of the oracle's top 64 whose logit is
/// within `TIE` of the top value.
fn matches(id: u32, top_ids: &[f64], top_values: &[f64]) -> (bool, bool) {
    if id as f64 == top_ids[0] {
        return (true, false);
    }
    let tie = top_ids
        .iter()
        .zip(top_values)
        .any(|(i, v)| *i == id as f64 && (top_values[0] - v).abs() <= TIE * top_values[0].abs().max(1.0));
    (tie, tie)
}

#[test]
fn latent_steps_on_the_card_read_and_reply_as_the_oracle() {
    let Some(dir) = directory("LFM3_MODEL_DIR") else { return };
    let Some(golden) = oracle() else { return };
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

    let body = json!({"model": "lfm3", "combine": "logprob_sum", "max_tokens": THEN,
                      "sample": {"temperature": 0, "seed": 7},
                      "contexts": [{"thinking": false, "messages": [{"role": "user", "content": PROMPT}]}],
                      "latent": {"steps": STEPS, "read": true}});
    let started = Instant::now();
    let got = send(&council, "/nagaya/v1/lab/ensemble", &body);
    eprintln!("latent: {STEPS} steps and {THEN} tokens in {:?}", started.elapsed());
    assert_eq!(
        got["usage"]["prompt"].as_u64().unwrap() as usize,
        golden["ids"].len(),
        "the prompt of the oracle: {got}"
    );

    // The top token of each step.
    let read = got["latent"]["read"].as_array().expect("the read of each step");
    assert_eq!(read.len(), STEPS, "{got}");
    let (top_ids, top_values) = (&golden["top_ids"], &golden["top_values"]);
    let k = top_ids.len() / STEPS;
    let (mut exact, mut ties) = (0, 0);
    for (step, token) in read.iter().enumerate() {
        let id = token["id"].as_u64().unwrap() as u32;
        let row = step * k..(step + 1) * k;
        let (ok, tie) = matches(id, &top_ids[row.clone()], &top_values[row]);
        eprintln!(
            "latent step {step}: {id} {:?}, the oracle's {}{}",
            token["text"],
            top_ids[step * k],
            if tie { " (a near tie)" } else { "" }
        );
        assert!(
            ok,
            "step {step}: the card read {id}, and the oracle's top is {}",
            top_ids[step * k]
        );
        if tie {
            ties += 1;
        } else {
            exact += 1;
        }
    }
    eprintln!("latent: {exact} of {STEPS} steps match the oracle's top token, and {ties} are near ties");

    // The reply: the oracle's greedy tokens after the steps.
    let then: Vec<u32> = golden["then_tokens"].iter().map(|v| *v as u32).collect();
    assert_eq!(got["usage"]["completion"].as_u64().unwrap() as usize, THEN, "{got}");
    let text = send(
        &council,
        "/nagaya/v1/detokenize",
        &json!({"model": "lfm3", "tokens": then}),
    );
    let want = text["text"].as_str().unwrap();
    eprintln!("latent: the reply {:?}, the oracle's {want:?}", got["content"]);
    assert_eq!(
        got["content"].as_str().unwrap().trim(),
        want.trim(),
        "the reply after the latent steps is the oracle's"
    );
}
