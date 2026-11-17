//! The nagaya process with `LFM3-Omni-2B` resident on card 0, with an object store (milestone 24 step 9, spec 11.11
//! "Media through the council"). A generation with the image or the clip of the oracle gives the top token of
//! transformers at its first step, and gogi's suite passes on `lfm3-omni`.
//!
//! It needs card 0, `LFM3_OMNI_MODEL_DIR`, and the oracle of `tools/lfm3_golden.sh` in `target/lfm3-golden` or
//! `LFM3_GOLDEN_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lfm3_media -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod card_support;

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

/// The iGPU of the B70 machine drives the display. Its config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

/// The most that the probability of the first token may differ from the oracle's. The card gates of the engine
/// measured logit gaps of 2.6e-5 or less for the image passes (`signoff.md`, W10), so this bound is loose.
const P_GAP: f64 = 1e-3;

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

/// The folder of the oracle, or none when its media file is absent.
fn golden() -> Option<PathBuf> {
    let dir = std::env::var_os("LFM3_GOLDEN_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"),
        PathBuf::from,
    );
    let needed = ["lfm3-omni.media.json", "image_small.png", "chirp.wav"];
    match needed.iter().find(|file| !dir.join(file).is_file()) {
        None => Some(dir),
        Some(file) if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!(
                "{} is absent and NAGAYA_REQUIRE_MODELS is set; run tools/lfm3_golden.sh",
                dir.join(file).display()
            )
        }
        Some(file) => {
            eprintln!(
                "skipped: {} is absent; run tools/lfm3_golden.sh",
                dir.join(file).display()
            );
            None
        }
    }
}

/// Omni resident on device 0, with one slot and an object store, as step 9 puts it on the B70 machine.
fn profile(dir: &Path) -> PlacementProfile {
    let card = card_support::device();
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"lfm3-omni\"\nkind = \"mk-lfm3-bmg\"\ndevices = [{card}]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 4294967296\n"
    ))
    .expect("the profile parses")
}

/// Send a request. Give the status and the body as text.
fn send(council: &Arc<Council>, method: &str, path: &str, body: Body, mime: &str) -> (StatusCode, String) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", mime)
            .body(body)
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    })
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Put an object through the route, and give its id.
fn put(council: &Arc<Council>, bytes: Vec<u8>, mime: &str) -> String {
    let id = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let (status, body) = send(
        council,
        "PUT",
        &format!("/council/v2/objects/{id}"),
        Body::from(bytes),
        mime,
    );
    assert!(status.is_success(), "{status}: {body}");
    id
}

/// The first token event of an SSE body.
fn first_token(sse: &str) -> Value {
    let mut lines = sse.lines();
    while let Some(line) = lines.next() {
        if line == "event: token" {
            let data = lines
                .next()
                .and_then(|data| data.strip_prefix("data: "))
                .expect("a data line");
            return serde_json::from_str(data).unwrap();
        }
    }
    panic!("the stream has no token event: {sse}")
}

#[test]
fn a_generation_of_omni_reads_the_image_and_the_clip_of_the_oracle() {
    let (Some(dir), Some(golden)) = (directory("LFM3_OMNI_MODEL_DIR"), golden()) else {
        return;
    };
    let mut oracle: Value =
        serde_json::from_slice(&std::fs::read(golden.join("lfm3-omni.media.json")).unwrap()).unwrap();
    // A build with the feature video also reads the videos of the oracle (spec 11.11, "The video of a request").
    let video = cfg!(feature = "video");
    if video {
        let videos: Value =
            serde_json::from_slice(&std::fs::read(golden.join("lfm3-omni.video.json")).unwrap()).unwrap();
        let prompts = oracle["prompts"].as_array_mut().unwrap();
        prompts.extend(videos["prompts"].as_array().unwrap().iter().cloned());
    }
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
    let (status, identity) = send(
        &council,
        "GET",
        "/council/v2/identity",
        Body::empty(),
        "application/json",
    );
    assert_eq!(status, StatusCode::OK, "{identity}");
    let identity: Value = serde_json::from_str(&identity).unwrap();
    let mut inputs = json!(["image/png", "image/jpeg", "audio/wav"]);
    if video {
        inputs = json!(["image/png", "image/jpeg", "audio/wav", "video/mp4", "video/webm"]);
    }
    assert_eq!(identity["models"][0]["media"]["inputs"], inputs, "{identity}");

    let image = put(
        &council,
        std::fs::read(golden.join("image_small.png")).unwrap(),
        "image/png",
    );
    let chirp = put(&council, std::fs::read(golden.join("chirp.wav")).unwrap(), "audio/wav");
    let mut cases = vec![
        ("image_small", image, "image/png", "Describe the image."),
        ("audio_chirp", chirp, "audio/wav", "What do you hear?"),
    ];
    if video {
        for name in ["video_av", "video_v"] {
            let id = put(
                &council,
                std::fs::read(golden.join(format!("{name}.mp4"))).unwrap(),
                "video/mp4",
            );
            cases.push((name, id, "video/mp4", "Describe the video."));
        }
    }
    for (name, id, mime, text) in cases {
        let prompt = oracle["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|prompt| prompt["name"] == name)
            .unwrap_or_else(|| panic!("the oracle has the prompt {name}"));
        let context = json!({"thinking": false, "messages": [{"role": "user", "content": [
            {"type": "object", "id": id, "mime": mime}, {"type": "text", "text": text}]}]});
        // The rendering is the prompt of transformers, token for token.
        let body = json!({"model": "lfm3-omni", "context": context, "generation_prompt": true});
        let (status, rendered) = send(
            &council,
            "POST",
            "/nagaya/v1/render",
            Body::from(body.to_string()),
            "application/json",
        );
        assert_eq!(status, StatusCode::OK, "{rendered}");
        let rendered: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(rendered["tokens"], prompt["ids"], "{name}: the tokens of the prompt");
        let started = Instant::now();
        let body = json!({"model": "lfm3-omni", "context": context, "max_tokens": 1, "stream": true,
            "sample": {"temperature": 0}});
        let (status, sse) = send(
            &council,
            "POST",
            "/council/v2/generations",
            Body::from(body.to_string()),
            "application/json",
        );
        assert_eq!(status, StatusCode::OK, "{sse}");
        let token = first_token(&sse);
        let p_oracle = (prompt["top_logits"][0].as_f64().unwrap() - prompt["logsumexp"].as_f64().unwrap()).exp();
        let p = token["p"].as_f64().unwrap();
        println!(
            "{name}: first token {} (p {p:.6}, oracle {} p {p_oracle:.6}) in {} ms",
            token["id"],
            prompt["top_ids"][0],
            started.elapsed().as_millis()
        );
        assert_eq!(
            token["id"], prompt["top_ids"][0],
            "{name}: the top token of transformers"
        );
        assert!(
            (p - p_oracle).abs() < P_GAP,
            "{name}: p {p} and the oracle's {p_oracle}"
        );
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
    let report = run(
        Opts {
            url: format!("http://{}", server.local_addr()),
            restart: None,
            model: Some("lfm3-omni".to_owned()),
        },
        false,
    );
    let count = |status: Status| report.checks.iter().filter(|check| check.status == status).count();
    let failed: Vec<String> = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| format!("{}: {}", check.name, check.detail))
        .collect();
    println!(
        "gogi-conformance on lfm3-omni: {} ok, {} FAIL, {} skip",
        count(Status::Ok),
        count(Status::Fail),
        count(Status::Skip)
    );
    assert!(failed.is_empty(), "gogi checks failed: {failed:#?}");
}
