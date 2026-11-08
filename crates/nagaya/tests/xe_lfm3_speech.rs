//! The process with `LFM3-Omni-2B` resident on card 0 speaks through council v2 (milestone 24 step 9, spec 11.11): a
//! generation with an `output` of `audio/wav`, in `speech` and `interleaved` mode, with the prompts of the oracle
//! replies `tts` and `mixed` (`target/lfm3-golden`, `tools/lfm3_reference.py`). Then gogi's suite runs on the model.
//!
//! The codes of a reply are random, and the nagaya draws them with its own hash (engine-abi "The speech calls of
//! LFM3"). Thus the gate checks the shape of the reply, not the codes of the oracle: the prompt, a WAV of 1,920
//! samples for each frame, the same bytes for the same seed, and the order of the stream events.
//!
//! It needs card 0, `LFM3_OMNI_MODEL_DIR`, and `speech.safetensors` and `facts.json` in `target/lfm3-golden` or
//! `LFM3_GOLDEN_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lfm3_speech -- --test-threads=1 --nocapture`.
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

/// The samples of one frame, and the bytes of the WAV header.
const FRAME_SAMPLES: u64 = 1920;
const HEADER: u64 = 44;

fn required() -> bool {
    std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1")
}

fn skip(what: &str) {
    assert!(!required(), "{what} is absent and NAGAYA_REQUIRE_MODELS is set");
    eprintln!("skipped: {what} is absent");
}

fn golden() -> Option<PathBuf> {
    let dir = std::env::var_os("LFM3_GOLDEN_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"),
        PathBuf::from,
    );
    if dir.join("facts.json").exists() && dir.join("speech.safetensors").exists() {
        Some(dir)
    } else {
        skip(&format!("{}/facts.json", dir.display()));
        None
    }
}

/// Omni resident on device 0 with one slot, as step 9 puts it on b70host, with a budget of objects.
fn profile(dir: &Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"lfm3-omni\"\nkind = \"mk-lfm3-bmg\"\ndevices = [0]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 4294967296\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Vec<u8>) {
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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 28).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn json_of(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or(Value::Null)
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The I64 tensor `name` of a safetensors file.
fn ids(path: &Path, name: &str) -> Vec<u64> {
    let bytes = std::fs::read(path).unwrap();
    let size = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + size]).unwrap();
    let offsets = header[name]["data_offsets"].as_array().unwrap();
    let (start, end) = (
        offsets[0].as_u64().unwrap() as usize,
        offsets[1].as_u64().unwrap() as usize,
    );
    bytes[8 + size + start..8 + size + end]
        .chunks(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as u64)
        .collect()
}

/// The generation of an oracle run: its system message and its text, in `mode`.
fn request(record: &Value, mode: &str, stream: bool) -> Value {
    json!({"model": "lfm3-omni",
        "context": {"thinking": false, "messages": [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["text"]}]},
        "max_tokens": 64, "sample": {"temperature": 0, "seed": 1234},
        "output": {"mime": "audio/wav", "options": {"mode": mode}},
        "stream": stream})
}

/// Check one reply with audio, and give its object id and its count of frames.
fn check_reply(council: &Arc<Council>, run: &str, done: &Value, prompt: usize) -> (String, u64) {
    let outputs = done["outputs"].as_array().unwrap_or_else(|| panic!("{run}: {done}"));
    assert_eq!(outputs.len(), 1, "{run}: one output: {done}");
    let output = &outputs[0];
    assert_eq!(output["mime"], "audio/wav", "{run}");
    let bytes = output["bytes"].as_u64().unwrap();
    let data = bytes - HEADER;
    assert_eq!(
        data % (2 * FRAME_SAMPLES),
        0,
        "{run}: the WAV holds whole frames: {bytes} bytes"
    );
    let frames = data / (2 * FRAME_SAMPLES);
    assert!(frames > 0, "{run}: the reply speaks");
    assert_eq!(
        done["usage"]["prompt"], prompt as u64,
        "{run}: the prompt of the oracle, with the marker"
    );
    assert_eq!(done["usage"]["media"]["written"]["audio_ms"], frames * 80, "{run}");
    let id = output["id"].as_str().unwrap().to_owned();
    let (status, wav) = send(council, "GET", &format!("/council/v2/objects/{id}"), None);
    assert_eq!(status, StatusCode::OK, "{run}: the object of the output");
    assert_eq!(wav.len() as u64, bytes, "{run}");
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(
        u32::from_le_bytes(wav[40..44].try_into().unwrap()) as u64,
        data,
        "{run}: the true size"
    );
    let peak = wav[44..]
        .chunks(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(peak > 100, "{run}: the audio is not silent: peak {peak}");
    (id, frames)
}

#[test]
fn omni_speaks_the_oracle_prompts_through_council_v2() {
    let Some(dir) = std::env::var_os("LFM3_OMNI_MODEL_DIR").map(PathBuf::from) else {
        skip("LFM3_OMNI_MODEL_DIR");
        return;
    };
    let Some(golden) = golden() else {
        return;
    };
    let facts: Value = serde_json::from_str(&std::fs::read_to_string(golden.join("facts.json")).unwrap()).unwrap();
    let speech = golden.join("speech.safetensors");

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

    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    let identity = json_of(&identity);
    assert_eq!(status, StatusCode::OK, "{identity}");
    assert_eq!(
        identity["models"][0]["media"]["outputs"],
        json!(["audio/wav"]),
        "{identity}"
    );

    for (run, mode) in [("tts", "speech"), ("mixed", "interleaved")] {
        let record = &facts["speech"][run];
        let prompt = ids(&speech, &format!("{run}.prompt_ids")).len();
        let started = Instant::now();
        let (status, bytes) = send(
            &council,
            "POST",
            "/council/v2/generations",
            Some(request(record, mode, false)),
        );
        let done = json_of(&bytes);
        assert_eq!(status, StatusCode::OK, "{run}: {done}");
        let (id, frames) = check_reply(&council, run, &done, prompt);
        println!(
            "{run}: {frames} frames in {} ms, finish {}, content {:?}; the oracle made {} frames",
            started.elapsed().as_millis(),
            done["finish"],
            done["content"],
            record["frames_with_end"].as_u64().unwrap_or(0).saturating_sub(1)
        );
        if run == "mixed" {
            assert!(
                !done["content"].as_str().unwrap().is_empty(),
                "an interleaved reply has text: {done}"
            );
        } else {
            // The text after the end frame is `<|im_end|>` in the oracle. With other codes it can differ, so the gate
            // checks only that the audio ended.
            assert_eq!(done["finish"], "end_turn", "a reply of speech ends its turn: {done}");
        }
        // The same request with the same seed gives the same codes, so the same bytes (spec 11.2).
        let (_, again) = send(
            &council,
            "POST",
            "/council/v2/generations",
            Some(request(record, mode, false)),
        );
        let again = json_of(&again);
        assert_eq!(
            again["outputs"][0]["id"].as_str(),
            Some(id.as_str()),
            "{run}: the same seed gives the same WAV"
        );
    }

    // A stream: the output event, then the token events, then done.
    let record = &facts["speech"]["mixed"];
    let (status, bytes) = send(
        &council,
        "POST",
        "/council/v2/generations",
        Some(request(record, "interleaved", true)),
    );
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes).unwrap();
    let names: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("event:").map(str::trim))
        .collect();
    assert_eq!(names.first(), Some(&"output"), "{text}");
    assert_eq!(names.last(), Some(&"done"), "{text}");
    assert!(
        names[1..names.len() - 1].iter().all(|name| *name == "token"),
        "{names:?}"
    );

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
    for check in report.checks.iter().filter(|check| check.status != Status::Ok) {
        println!("lfm3-omni: {:?} {}: {}", check.status, check.name, check.detail);
    }
    println!(
        "gogi-conformance on lfm3-omni: {} ok, {} FAIL, {} skip",
        count(Status::Ok),
        count(Status::Fail),
        count(Status::Skip)
    );
    assert_eq!(count(Status::Fail), 0, "gogi checks failed");
}
