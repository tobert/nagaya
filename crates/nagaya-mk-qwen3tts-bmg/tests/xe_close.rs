//! `close` of the speech engine (engine-abi "`close` in Rust", spec 3.3a): a drop frees its device memory, so a swap
//! can load another engine on the card. Run alone, with `--test-threads=1`.
//!
//! The test reads the device memory of this process from DRM fdinfo (`drm-total-vram0` of each `xe` client). The Level
//! Zero driver keeps some device memory for the process after its first init and its first launch, so the test
//! measures that first: a cluster that runs an empty ring, with no weights. Then it opens, speaks and drops three
//! times. On 2026-10-24 a baseline with no launch was 2,269,184 bytes lower, and each drop came back to the same
//! count.
#![cfg(feature = "xe")]

mod common;

use nagaya_abi::{PromptPair, Speech, SpeechSampler};
use nagaya_mk_bmg::level_zero::L0Cluster;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{KERNEL_ENTRY, QWEN3TTS_KERNEL};
use nagaya_mk_qwen3tts_bmg::{Config, Qwen3TtsEngine};

/// The old `speech_begin`: a feed of text and codec pairs into an empty slot.
fn begin(
    engine: &mut dyn nagaya_abi::Speech,
    slot: u32,
    prompt: &[PromptPair],
    sampler: &nagaya_abi::SpeechSampler,
) -> Result<(), nagaya_abi::PassError> {
    let positions: Vec<_> = prompt.iter().map(|p| nagaya_abi::SpeechPosition::Pair(*p)).collect();
    engine.speech_feed(slot, &positions, sampler)
}

/// The device bytes above the driver's own that may stay after a drop.
const SLACK: u64 = 1 << 20;

/// The device bytes of this process on every `xe` card: the sum of `drm-total-vram0` over its DRM fds.
fn process_vram() -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir("/proc/self/fdinfo").expect("the fdinfo of this process") {
        let entry = entry.expect("an fdinfo entry");
        // An fd can close between the listing and the read.
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        if !text
            .lines()
            .any(|line| line.starts_with("drm-driver:") && line.ends_with("xe"))
        {
            continue;
        }
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("drm-total-vram0:") {
                let kib = value
                    .trim()
                    .strip_suffix(" KiB")
                    .unwrap_or_else(|| panic!("`drm-total-vram0` is `{value}`, not a count of KiB"));
                total += kib.trim().parse::<u64>().expect("a count of KiB") * 1024;
            }
        }
    }
    total
}

#[test]
fn a_drop_frees_the_device_memory_of_each_open() {
    if common::absent() {
        return;
    }
    let cluster = L0Cluster::init(QWEN3TTS_KERNEL, KERNEL_ENTRY);
    let card = &cluster.cards[0];
    let mut ring = Ring::new(card, 1);
    ring.set(&[]);
    // SAFETY: the ring holds only its halt.
    unsafe { ring.run(card) };
    ring.free(card);
    drop(cluster);
    let driver = process_vram();
    let mut first = None;
    for round in 1..=3 {
        let config = Config {
            model_dir: common::model_dir(),
            slots: 1,
            max_frames: 64,
            max_prompt: 64,
            frames_per_quantum: 4,
        };
        let start = std::time::Instant::now();
        let mut engine = Qwen3TtsEngine::open(config, &common::cards()[0]).expect("open the speech engine");
        let open_ms = start.elapsed().as_millis();
        let prompt = [PromptPair {
            text: Some(151672),
            codec: Some(2154),
        }];
        let sampler = SpeechSampler {
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
            seed: round,
        };
        begin(&mut engine, 0, &prompt, &sampler).expect("begin");
        engine.speech_frames(0, 2).expect("frames");
        let loaded = process_vram();
        let start = std::time::Instant::now();
        drop(engine);
        let drop_ms = start.elapsed().as_millis();
        let left = process_vram();
        eprintln!(
            "round {round}: open {open_ms} ms, {loaded} bytes loaded, drop {drop_ms} ms, {left} bytes left, driver {driver}"
        );
        assert!(loaded > driver + (3 << 30), "the open holds {loaded} bytes");
        assert!(
            left <= driver + SLACK,
            "round {round}: {left} bytes stay after the drop"
        );
        assert_eq!(*first.get_or_insert(left), left, "round {round}: the leftover grows");
    }
}
