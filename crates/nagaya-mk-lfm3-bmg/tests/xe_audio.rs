//! The gate of the audio encoder of Omni on a B70 card (spec 8.3, 11.11 "The LFM3 oracle"): the rows after the
//! downsampler, after the layers and after the projector, against transformers in f32 on the CPU, and against the CPU
//! form of the tasks (`common/cpu.rs`) for clips that the oracle does not hold.
//!
//! The tolerance of the gate is not measured yet, so the tests print each gap. They assert only `DEFECT_BOUND`, which
//! any tolerance holds. On the host, the CPU form gave 4.9e-7, 1.3e-7 and 1.9e-6 against the oracle, and a wrong op
//! gave 1.3e-2 or more (`audio_cpu_form.rs`).
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_audio -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::cpu::{CpuEncoder, STAGES, mel_rows};
use nagaya_mk_lfm3_bmg::audio::{MAX_CLIP_FRAMES, Stage};

/// The bound of a defect: a relative error of a stage past this is a wrong op.
const DEFECT_BOUND: f64 = 0.05;

/// The feed block of the gates, the default of the engine (engine-abi D11).
const BLOCK: usize = 512;

const ORDER: [Stage; 3] = [Stage::Downsample, Stage::Encoder, Stage::Rows];

/// Mel rows `[frames][128]` in the range of the features of the family, from a seed: a slow wave in each bin, and
/// noise. The values of the family lie in about -1.0 to 1.5.
fn synthetic_mel(frames: usize, seed: u64) -> Vec<f32> {
    let mut z = seed;
    (0..frames * 128)
        .map(|i| {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (z >> 40) as f32 / 16_777_216.0 - 0.5;
            let (t, b) = ((i / 128) as f32, (i % 128) as f32);
            0.4 * (0.05 * t + 0.3 * b).sin() + 0.2 * noise
        })
        .collect()
}

#[test]
fn the_audio_stages_on_the_card_follow_transformers() {
    if common::absent(Which::Omni) || common::golden_absent("audio_in.safetensors") {
        return;
    }
    let golden = common::golden_tensors("audio_in.safetensors");
    let mel = mel_rows(&golden["mel"]);
    let mut engine = common::open(Which::Omni, 1, BLOCK);
    for (stage, name) in ORDER.iter().zip(STAGES) {
        let got = engine.audio_rows(&mel, *stage).unwrap();
        let want = golden[name].f32();
        let gap = common::rel(&got, &want);
        println!(
            "card, {name}: relative error {gap:.3e}, largest difference {:.3e}",
            common::max_abs(&got, &want)
        );
        assert!(gap < DEFECT_BOUND, "{name}: relative error {gap}");
    }
}

/// The oracle clip has 19 positions, fewer than the window of 32 keys. These clips cover the window, one position,
/// an odd count of frames, and the longest clip.
#[test]
fn other_clips_on_the_card_follow_the_cpu_form() {
    if common::absent(Which::Omni) {
        return;
    }
    let dir = common::model_dir(Which::Omni);
    let mut cpu = CpuEncoder::load(&dir);
    let mut engine = common::open(Which::Omni, 1, BLOCK);
    for (frames, seed) in [(1usize, 1u64), (8, 2), (77, 3), (1000, 4), (MAX_CLIP_FRAMES, 5)] {
        let mel = synthetic_mel(frames, seed);
        let want = cpu.stages(&mel);
        for ((stage, name), want) in ORDER.iter().zip(STAGES).zip(&want) {
            let got = engine.audio_rows(&mel, *stage).unwrap();
            let gap = common::rel(&got, want);
            println!(
                "{frames} frames, {name}: relative error {gap:.3e} against the CPU form, largest difference {:.3e}",
                common::max_abs(&got, want)
            );
            assert!(gap < DEFECT_BOUND, "{frames} frames, {name}: relative error {gap}");
        }
    }
}

/// The engine declares `Determinism::Always`, so a clip gives the same bits each time, after another clip too.
#[test]
fn a_clip_gives_the_same_bits_each_time() {
    if common::absent(Which::Omni) {
        return;
    }
    let mut engine = common::open(Which::Omni, 1, BLOCK);
    let (a, b) = (synthetic_mel(300, 7), synthetic_mel(1000, 8));
    let first = engine.audio_rows(&a, Stage::Rows).unwrap();
    engine.audio_rows(&b, Stage::Rows).unwrap();
    let again = engine.audio_rows(&a, Stage::Rows).unwrap();
    let differ = first.iter().zip(&again).position(|(x, y)| x.to_bits() != y.to_bits());
    assert_eq!(differ, None, "the second run of a clip differs at value {differ:?}");
}

#[test]
fn the_text_model_and_a_clip_past_the_longest_are_refused() {
    if common::absent(Which::Omni) || common::absent(Which::Text) {
        return;
    }
    let mut omni = common::open(Which::Omni, 1, BLOCK);
    let long = synthetic_mel(MAX_CLIP_FRAMES + 1, 9);
    let error = omni.audio_rows(&long, Stage::Rows).unwrap_err();
    assert!(error.contains("takes 513 positions"), "{error}");
    drop(omni);
    let mut text = common::open(Which::Text, 1, BLOCK);
    let error = text.audio_rows(&synthetic_mel(8, 10), Stage::Rows).unwrap_err();
    assert!(error.contains("lfm3 has no audio encoder"), "{error}");
}
