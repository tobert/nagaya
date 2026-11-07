//! The nagaya process with `LFM3-1.7B` or `LFM3-Omni-2B` resident on card 0 (milestone 24 step 9, spec 11.11): the
//! identity, a generation, and gogi's suite for the served model. This test covers text only; the media and the speech
//! of Omni have their own gates.
//!
//! It needs card 0, `LFM3_MODEL_DIR` and `LFM3_OMNI_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lfm3 -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gogi_conformance::{Opts, Status, run};
use nagaya::admission::Registry;
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, Server, router};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options};
use nagaya::probe::FileProbe;
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use serde_json::{Value, json};
use tower::ServiceExt;

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

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

/// One LFM3 model resident on device 0, with one slot, as step 9 puts it on b70host. The object store holds the
/// control vectors (spec 11.12). The process serves one text
/// model now (spec 11.7), so each model gets its own process.
fn profile(name: &str, dir: &Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"{name}\"\nkind = \"mk-lfm3-bmg\"\ndevices = [0]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 1073741824\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path);
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Run gogi's suite on one served model, and give its counts and the names of the failed checks.
fn suite(url: &str, model: &str) -> (usize, usize, usize, Vec<String>) {
    let report = run(
        Opts {
            url: url.to_owned(),
            restart: None,
            model: Some(model.to_owned()),
        },
        false,
    );
    let count = |status: Status| report.checks.iter().filter(|check| check.status == status).count();
    for check in report.checks.iter().filter(|check| check.status != Status::Ok) {
        println!("{model}: {:?} {}: {}", check.status, check.name, check.detail);
    }
    let failed = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| check.name.clone())
        .collect();
    // The checks of control vectors run, not skip, on a served model that declares control_vectors (spec 11.12).
    if model == "lfm3" {
        let ok = |name: &str| {
            report
                .checks
                .iter()
                .any(|check| check.name == name && check.status == Status::Ok)
        };
        let missing: Vec<&str> = CONTROL_CHECKS.iter().copied().filter(|name| !ok(name)).collect();
        assert!(
            missing.is_empty(),
            "{model}: control checks that did not pass: {missing:?}"
        );
        println!(
            "{model}: the {} control checks of the list passed",
            CONTROL_CHECKS.len()
        );
    }
    (count(Status::Ok), count(Status::Fail), count(Status::Skip), failed)
}

/// The checks of gogi's suite for a served model with `control_vectors` (gogi `checks/control.rs`), as
/// `tests/http_conformance.rs` names them.
const CONTROL_CHECKS: [&str; 19] = [
    "a control vector response names its served model and counts each side",
    "a control vector response carries the served model's identity",
    "a control vector request writes no tokens",
    "the control vector names its served model and weights",
    "raw reads with an unknown op are a 400 naming contexts[1].control[0].op",
    "a build with a strength that is not a number in control is a 400 naming control[0].strength",
    "a control vector request with a context with a control of its own is a 400 naming negative[1].control",
    "the same contexts give the same vector",
    "the control vector reads back with its media type",
    "the control vector's bytes hash to its id",
    "the control vector has the layers and width the response names",
    "add and ablate at strength 0 give the numbers of no control",
    "a list that changes nothing still has model state of its own",
    "a list that changes the model never reuses the plain model state",
    "a control vector the server does not hold is a 404 naming control[0].vector",
    "a control vector for other weights is a 400 naming control[0].vector",
    "a control vector of another width is a 400 naming control[0].vector",
    "a side past control_vector_contexts is a 400 naming it",
    "a dry run of a control vector request makes no vector",
];

#[test]
fn the_process_serves_lfm3_on_card_0() {
    if let Some(dir) = directory("LFM3_MODEL_DIR") {
        serve("lfm3", &dir);
    }
}

#[test]
fn the_process_serves_lfm3_omni_on_card_0() {
    if let Some(dir) = directory("LFM3_OMNI_MODEL_DIR") {
        serve("lfm3-omni", &dir);
    }
}

/// Start a process with the model `name` from `dir`, then check its identity, a generation and gogi's suite.
fn serve(name: &str, dir: &Path) {
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let profile = profile(name, dir);
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let start = Instant::now();
    let ready = lifecycle::start_with(
        profile.clone(),
        Registry::compiled_in(),
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens the engine");
    println!("opened in {} ms", start.elapsed().as_millis());
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

    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK, "{identity}");
    let models: Vec<(String, String)> = identity["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| {
            (
                model["model"].as_str().unwrap().to_owned(),
                model["status"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    println!("served models: {models:?}");
    let names: Vec<String> = models.iter().map(|(name, _)| name.clone()).collect();
    assert_eq!(names, [name], "{identity}");
    assert!(models.iter().all(|(_, status)| status == "ready"), "{identity}");

    for model in &names {
        let started = Instant::now();
        let body = json!({"model": model, "context": {"thinking": false, "messages": [{"role": "user", "content": "Name three colors."}]},
            "max_tokens": 48, "sample": {"temperature": 0}});
        let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
        assert_eq!(status, StatusCode::OK, "{model}: {generation}");
        println!(
            "{model}: generation in {} ms: {generation}",
            started.elapsed().as_millis()
        );
    }

    // LFM3-1.7B declares control_vectors, and Omni does not yet (spec 11.12, milestone 25 step 9).
    let declares = identity["models"][0]["capabilities"]
        .as_array()
        .unwrap()
        .contains(&json!("control_vectors"));
    assert_eq!(declares, name == "lfm3", "{identity}");
    if name == "lfm3" {
        control_gate(&council);
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let server = Server::start(
        runtime.handle(),
        format!("127.0.0.1:{}", free_port()).parse().unwrap(),
        Arc::clone(&council),
    )
    .expect("the HTTP server binds a loopback port");
    let url = format!("http://{}", server.local_addr());
    let mut failed = Vec::new();
    for model in &names {
        let (ok, fail, skip, names) = suite(&url, model);
        println!("gogi-conformance on {model}: {ok} ok, {fail} FAIL, {skip} skip");
        failed.extend(names.into_iter().map(|name| format!("{model}: {name}")));
    }
    assert!(failed.is_empty(), "gogi checks failed: {failed:?}");
}

/// The gate of a relative error: `STAGE_TOL` of `crates/nagaya-mk-lfm3-bmg/tests/common/mod.rs` (spec 11.11).
const STAGE_TOL: f64 = 1e-4;

/// The oracle of control vectors, `lfm3.control.safetensors` of `tools/lfm3_golden.sh`, from `LFM3_GOLDEN_DIR` or
/// `target/lfm3-golden`. `None` when the file is absent, and a failure then with `NAGAYA_REQUIRE_MODELS=1`.
fn control_oracle() -> Option<(serde_json::Map<String, Value>, Vec<u8>, Value)> {
    let dir = std::env::var_os("LFM3_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"));
    let path = dir.join("lfm3.control.safetensors");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") {
                panic!(
                    "{} is absent and NAGAYA_REQUIRE_MODELS is set; write it with tools/lfm3_golden.sh",
                    path.display()
                );
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

/// Send one request through the router, and give the status and the raw body.
fn send_raw(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path);
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn user(text: &str) -> Value {
    json!({"thinking": false, "messages": [{"role": "user", "content": text}]})
}

/// The ids of the tokens of a streamed generation.
fn streamed_ids(council: &Arc<Council>, context: Value, max_tokens: u64) -> Vec<u64> {
    let body = json!({"model": "lfm3", "context": context, "max_tokens": max_tokens, "sample": {"temperature": 0},
                      "stream": true});
    let (status, bytes) = send_raw(council, "POST", "/council/v2/generations", Some(body));
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    let mut ids = Vec::new();
    for event in text.split("\n\n") {
        let mut name = "";
        let mut data = "";
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
            ids.push(token["id"].as_u64().unwrap());
        }
    }
    ids
}

/// The route makes the vector of the oracle's prompt sets on the card, and the vector matches the oracle's. A
/// generation with an ablate of 1 of that vector gives the oracle's greedy ids (spec 11.12, milestone 25 step 7).
fn control_gate(council: &Arc<Council>) {
    let Some((header, data, meta)) = control_oracle() else {
        return;
    };
    let texts = |side: &str| -> Vec<String> {
        meta[side]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_owned())
            .collect()
    };
    let (positive, negative) = (texts("positive"), texts("negative"));
    // The family renders each prompt as the oracle did: one user message, thinking off, the generation prompt.
    let mut prompt = 0;
    for (side, list) in [("positive", &positive), ("negative", &negative)] {
        for (i, text) in list.iter().enumerate() {
            let body = json!({"model": "lfm3", "context": user(text), "generation_prompt": true});
            let (status, rendered) = send(council, "POST", "/nagaya/v1/render", Some(body));
            assert_eq!(status, StatusCode::OK, "{rendered}");
            let want: Vec<u64> = oracle_tensor(&header, &data, &format!("{side}{i}.ids"))
                .iter()
                .map(|v| *v as u64)
                .collect();
            let got: Vec<u64> = rendered["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(got, want, "{side}{i}: the rendering differs from the oracle's ids");
            prompt += want.len() as u64;
        }
    }
    let body = json!({"model": "lfm3", "positive": positive.iter().map(|t| user(t)).collect::<Vec<_>>(),
                      "negative": negative.iter().map(|t| user(t)).collect::<Vec<_>>()});
    let started = Instant::now();
    let (status, made) = send(council, "POST", "/council/v2/control-vectors", Some(body.clone()));
    assert_eq!(status, StatusCode::OK, "{made}");
    println!("control vector in {} ms: {made}", started.elapsed().as_millis());
    assert_eq!(made["usage"]["prompt"].as_u64(), Some(prompt), "{made}");
    assert_eq!(
        (made["layers"].as_u64(), made["width"].as_u64()),
        (Some(24), Some(2048))
    );
    let id = made["vector"].as_str().unwrap().to_owned();
    let (status, again) = send(council, "POST", "/council/v2/control-vectors", Some(body));
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(
        again["vector"].as_str(),
        Some(id.as_str()),
        "the same request gives the same bytes on the card"
    );
    let (status, bytes) = send_raw(council, "GET", &format!("/council/v2/objects/{id}"), None);
    assert_eq!(status, StatusCode::OK);
    let vector = gogi_proto::control::ControlVector::parse(&bytes).unwrap();
    assert_eq!(vector.model, "lfm3");
    let want = oracle_tensor(&header, &data, "vector");
    let rows: Vec<Vec<f64>> = (0..4)
        .map(|i| oracle_tensor(&header, &data, &format!("positive{i}.rows")))
        .collect();
    // A layer of the vector is a difference of two means, and layer 0 cancels. So the error of each layer is relative
    // to the norm of the positive mean of that layer, as in `xe_control.rs`.
    let mut worst = (0u32, 0.0f64);
    for (layer, got) in &vector.layers {
        let at = *layer as usize * 2048;
        let diff: f64 = got
            .iter()
            .zip(&want[at..at + 2048])
            .map(|(g, w)| (f64::from(*g) - w).powi(2))
            .sum();
        let mean: f64 = (0..2048)
            .map(|j| (rows.iter().map(|r| r[at + j]).sum::<f64>() / 4.0).powi(2))
            .sum();
        let gap = (diff / mean).sqrt();
        if gap > worst.1 {
            worst = (*layer, gap);
        }
    }
    println!(
        "the route's vector: worst layer {}, error {:.3e} of the norm of the positive mean",
        worst.0, worst.1
    );
    assert!(
        worst.1 < STAGE_TOL,
        "layer {}: error {} of the positive mean",
        worst.0,
        worst.1
    );
    // An ablate of 1 of the route's vector gives the oracle's greedy ids, and the plain model gives its own.
    let test = meta["test"].as_str().unwrap();
    for (name, control) in [
        ("ablate", json!([{"vector": id, "op": "ablate", "strength": 1.0}])),
        ("plain", json!([])),
    ] {
        let mut context = user(test);
        context["control"] = control;
        let want: Vec<u64> = oracle_tensor(&header, &data, &format!("{name}.decode_ids"))
            .iter()
            .map(|v| *v as u64)
            .collect();
        let got = streamed_ids(council, context, want.len() as u64);
        println!("{name}: greedy ids {got:?}, oracle {want:?}");
        assert_eq!(got, want, "{name}: the greedy ids differ from the oracle's");
    }
}
