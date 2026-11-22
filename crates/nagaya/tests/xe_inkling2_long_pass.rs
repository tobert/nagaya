//! The card gate of a long pass (engine-abi D8, "A long pass with a declared bound"): the process with Inkling 2 Mini
//! on one card, the watchdog at its defaults of 30 s and 5 s, and a chat with one image of 4,096 by 4,096 px that the
//! engine does not hold. The pass of its first block runs the tower, about 57 s. The gate checks that the pass shows
//! on the route `busy` with its bound, that it runs past the timeout, that no stall comes, and that the reply is a 200.
//!
//! It needs one card of `NAGAYA_TEST_CARDS` and `INKLING2_MODEL_DIR`. The card must be free: release it from the
//! service first. Run it alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test --release -p nagaya --features xe --test xe_inkling2_long_pass --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod card_support;

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

/// The iGPU of the B70 machine drives the display. Its config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

/// The side of the largest image that the family takes (spec 11.18, "Images").
const SIDE: u32 = 4_096;

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

/// Inkling 2 Mini resident on the card of the test, with one slot of 32,768 positions and an object store.
fn profile(dir: &Path) -> PlacementProfile {
    let card = card_support::device();
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"inkling2-mini\"\nkind = \"mk-inkling2-bmg\"\ndevices = [{card}]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 4294967296\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Body, mime: &str) -> (StatusCode, Value) {
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
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

/// A PNG of `side` by `side` px of fixed noise, so that no earlier run left its rows on the card.
fn noise_png(side: u32) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1du64 ^ u64::from(std::process::id());
    let pixels: Vec<u8> = (0..side as usize * side as usize * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, side, side);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&pixels).unwrap();
    }
    out
}

#[test]
fn a_cold_image_of_4096_px_runs_its_tower_in_a_long_pass_with_no_stall() {
    let Some(dir) = directory("INKLING2_MODEL_DIR") else {
        return;
    };
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let profile = profile(&dir);
    let card = card_support::device();
    let store = tempfile::tempdir().unwrap();
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    // The process keeps the default watchdog: a timeout of 30 s and a beat of 5 s (spec 7.4).
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
    let (calls, asked) = std::sync::mpsc::channel();
    let active = profile.clone();
    std::thread::spawn(move || {
        while let Ok(call) = asked.recv() {
            if let nagaya::reprofile::ProfileCall::Get(reply) = call {
                let _ = reply.send(active.clone());
            }
        }
    });
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
        .with_loads(ports.clone())
        .with_profile(calls),
    );

    let image = noise_png(SIDE);
    let id = format!("blake3:{}", blake3::hash(&image).to_hex());
    let (status, body) = send(
        &council,
        "PUT",
        &format!("/council/v2/objects/{id}"),
        Body::from(image),
        "image/png",
    );
    assert!(status.is_success(), "{status}: {body}");

    let context = json!({"thinking": false, "messages": [{"role": "user", "content": [
        {"type": "object", "id": id, "mime": "image/png"}, {"type": "text", "text": "Describe the image."}]}]});
    let request = json!({"model": "inkling2-mini", "context": context, "max_tokens": 1, "sample": {"temperature": 0}});
    let started = Instant::now();
    let asking = Arc::clone(&council);
    let generation = std::thread::spawn(move || {
        send(
            &asking,
            "POST",
            "/council/v2/generations",
            Body::from(request.to_string()),
            "application/json",
        )
    });
    let busy = format!("/nagaya/v1/devices/{card}/busy");
    let mut seen: Option<Value> = None;
    while !generation.is_finished() {
        let (status, body) = send(&council, "GET", &busy, Body::empty(), "application/json");
        assert_eq!(status, StatusCode::OK, "{body}");
        if !body["long_pass"].is_null() {
            seen = Some(body["long_pass"].clone());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let (status, reply) = generation.join().unwrap();
    let total = started.elapsed();
    println!("the generation took {} ms: {reply}", total.as_millis());
    assert_eq!(status, StatusCode::OK, "{reply}");
    let seen = seen.expect("the route showed the long pass of the tower");
    println!("the last view of the long pass: {seen}");
    let bound = nagaya_mk_inkling2_bmg::bound::pass_bound_ms(&[10_609]).unwrap().get();
    assert_eq!(seen["bound_ms"], json!(bound), "{seen}");
    assert_eq!(seen["devices"], json!([card]), "{seen}");
    assert_eq!(seen["overdue"], json!(false), "the tower ran within its bound: {seen}");
    let elapsed = seen["elapsed_ms"].as_u64().unwrap();
    assert!(
        elapsed > 30_000,
        "the pass ran {elapsed} ms; the gate needs a pass past the timeout of 30 s"
    );
    assert_eq!(
        ready.process.recv(Duration::from_secs(1)),
        None,
        "no stall and no other event came during the long pass"
    );
}
