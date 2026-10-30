//! Gemma 4 on the two B70 cards in both orders of its profile list (spec 2.4). With `devices = [1, 0]`, stage 0 runs
//! on card 1 and stage 1 on card 0. The reply is the same as with `[0, 1]`, and the device bytes of the two cards trade
//! places.
//!
//! It needs both cards and the checkpoint in `GEMMA4_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_reversed -- --test-threads=1 --nocapture`.
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
use nagaya::signal::SignalFlag;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "gemma-4-26b-a4b";

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

fn checkpoint() -> Option<PathBuf> {
    match std::env::var_os("GEMMA4_MODEL_DIR") {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("GEMMA4_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the Gemma 4 checkpoint")
        }
        None => {
            eprintln!("skipped: GEMMA4_MODEL_DIR is not set; set it to the Gemma 4 checkpoint directory");
            None
        }
    }
}

/// The profile of b70host with Gemma 4 alone, on `devices` in this order.
fn profile(dir: &Path, devices: &str) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"gemma\"\nkind = \"mk-gemma4-bmg\"\ndevices = {devices}\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = [4096, 16384]\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
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

/// The device bytes of this process on each `xe` card, by PCI address: `drm-total-vram0` over its DRM fds.
fn vram_by_card() -> BTreeMap<String, u64> {
    let mut cards = BTreeMap::new();
    for entry in std::fs::read_dir("/proc/self/fdinfo").expect("the fdinfo of this process") {
        let entry = entry.expect("an fdinfo entry");
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        if !text
            .lines()
            .any(|line| line.starts_with("drm-driver:") && line.ends_with("xe"))
        {
            continue;
        }
        let pdev = text
            .lines()
            .find_map(|line| line.strip_prefix("drm-pdev:"))
            .expect("an xe fd names its PCI device")
            .trim()
            .to_owned();
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("drm-total-vram0:") {
                let kib = value.trim().strip_suffix(" KiB").expect("a count of KiB");
                *cards.entry(pdev.clone()).or_insert(0) += kib.trim().parse::<u64>().expect("a count of KiB") * 1024;
            }
        }
    }
    cards
}

/// Open Gemma 4 on `devices`, generate one reply, and stop. Give the reply and the device bytes of each card.
fn run(dir: &Path, devices: &str) -> (String, BTreeMap<String, u64>) {
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let profile = profile(dir, devices);
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let start = Instant::now();
    let mut ready = lifecycle::start_with(
        profile.clone(),
        Registry::compiled_in(),
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens Gemma 4");
    println!("devices = {devices}: opened in {} ms", start.elapsed().as_millis());
    let council = Arc::new(
        council(
            &FamilyRegistry::compiled_in(),
            &profile,
            &ready.opened,
            &ready.process.ports(),
            store.path(),
            &[],
            None,
        )
        .expect("the process serves Gemma 4"),
    );
    let body = json!({"model": MODEL, "max_tokens": 32, "sample": {"temperature": 0},
        "context": {"messages": [{"role": "user", "content": "Name three colors of the rainbow."}]}});
    let (status, generation) = send(&council, "POST", "/council/v2/generations", Some(body));
    assert_eq!(status, StatusCode::OK, "{generation}");
    let reply = format!(
        "{}|{}",
        generation["reasoning"].as_str().expect("a reply has reasoning"),
        generation["content"].as_str().expect("a reply has content")
    );
    let vram = vram_by_card();
    println!("devices = {devices}: {reply:?}, device bytes {vram:?}");
    council.drain();
    let stopped = lifecycle::stop(&mut ready, None, Duration::from_secs(30), &SignalFlag::new());
    println!("stop: {}", stopped.log_line());
    (reply, vram)
}

#[test]
fn gemma4_on_devices_1_0_gives_the_reply_of_0_1_with_its_stages_on_the_other_cards() {
    let Some(dir) = checkpoint() else {
        return;
    };
    let (forward, by_forward) = run(&dir, "[0, 1]");
    let (reversed, by_reversed) = run(&dir, "[1, 0]");
    assert_eq!(reversed, forward, "the order of the cards does not change the reply");
    let cards: Vec<&String> = by_forward.keys().collect();
    assert_eq!(cards.len(), 2, "Gemma 4 holds memory on both cards: {by_forward:?}");
    let (a, b) = (cards[0], cards[1]);
    // The trade shows only when the two stages hold different bytes.
    assert!(
        by_forward[a].abs_diff(by_forward[b]) >= 1 << 30,
        "the two stages hold about the same bytes, so the test cannot see a trade: {by_forward:?}"
    );
    let trade = |x: u64, y: u64| x.abs_diff(y) < 256 << 20;
    assert!(
        trade(by_reversed[a], by_forward[b]) && trade(by_reversed[b], by_forward[a]),
        "the device bytes of the cards trade places: [0, 1] gave {by_forward:?}, [1, 0] gave {by_reversed:?}"
    );
}
