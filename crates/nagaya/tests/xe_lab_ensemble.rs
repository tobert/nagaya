//! The lab route `ensemble` of a process with `LFM3-1.7B` on card 0 (`docs/lab.md`, "`ensemble`: joined forks",
//! milestone 29 step 1b). One context gives the response of the same generation. An LFM3 row does not depend on the
//! other rows of its pass (milestone 27), so two equal contexts give the greedy reply of one, by each rule. Three
//! contexts decode in passes of three rows, and the time of a step is printed.
//!
//! It needs card 0 and `LFM3_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lab_ensemble -- --test-threads=1 --nocapture`.
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

const IGNORE: &str = "0000:80:00.0";
const PATH: &str = "/nagaya/v1/lab/ensemble";

fn directory(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is unset and NAGAYA_REQUIRE_MODELS is set")
        }
        None => {
            eprintln!("skipped: {variable} is unset");
            None
        }
    }
}

/// `LFM3-1.7B` resident on device 0, with four slots, as in `xe_lab_trace.rs`.
fn profile(dir: &Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"lfm3\"\nkind = \"mk-lfm3-bmg\"\ndevices = [0]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 4\n\
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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        assert_eq!(status, StatusCode::OK, "{path}: {value}");
        value
    })
}

fn context(text: &str) -> Value {
    json!({"thinking": false, "messages": [{"role": "user", "content": text}]})
}

fn ensemble(contexts: Vec<Value>, combine: &str, sample: &Value, max_tokens: u64) -> Value {
    json!({"model": "lfm3", "contexts": contexts, "combine": combine, "max_tokens": max_tokens, "sample": sample})
}

#[test]
fn an_ensemble_on_the_card_gives_the_generation_of_one_context_and_joins_its_rows() {
    let Some(dir) = directory("LFM3_MODEL_DIR") else { return };
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
    let prompt = "Plan a short story about a lighthouse keeper who finds a map.";
    let sampled = json!({"temperature": 0.8, "seed": 7});
    let greedy = json!({"temperature": 0, "seed": 1});

    // 1. One context gives the response of the same generation, with sampling.
    let generation = json!({"model": "lfm3", "context": context(prompt), "max_tokens": 48, "sample": sampled});
    let want = send(&council, "/council/v2/generations", &generation);
    let got = send(
        &council,
        PATH,
        &ensemble(vec![context(prompt)], "logprob_sum", &sampled, 48),
    );
    assert_eq!(got["content"], want["content"]);
    assert_eq!(got["finish"], want["finish"]);
    assert_eq!(got["usage"]["completion"], want["usage"]["completion"]);
    eprintln!(
        "one context: {} tokens, the same as the generation",
        want["usage"]["completion"]
    );

    // 2. Two equal contexts give the greedy reply of one, by each rule: their rows do not depend on the pass.
    let generation = json!({"model": "lfm3", "context": context(prompt), "max_tokens": 48, "sample": greedy});
    let one = send(&council, "/council/v2/generations", &generation)["content"].clone();
    for rule in ["logprob_sum", "prob_mean", "contrast"] {
        let got = send(
            &council,
            PATH,
            &ensemble(vec![context(prompt), context(prompt)], rule, &greedy, 48),
        );
        assert_eq!(got["content"], one, "{rule}");
    }
    eprintln!("two equal contexts: the greedy reply of one, by each rule");

    // 3. Three contexts: a reply, and the time of a step.
    let three = vec![
        context(prompt),
        context("Plan a short story about a lighthouse keeper who loses a map."),
        context("Plan a short story about a sailor who finds a lighthouse."),
    ];
    let started = Instant::now();
    let got = send(&council, PATH, &ensemble(three, "prob_mean", &sampled, 96));
    let completion = got["usage"]["completion"].as_u64().unwrap();
    assert!(completion > 0, "{got}");
    eprintln!(
        "three contexts: {completion} tokens in {:.2} s, with the three prompts",
        started.elapsed().as_secs_f64()
    );
}
