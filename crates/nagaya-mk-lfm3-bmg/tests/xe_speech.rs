//! The speech path of Omni on a B70 card against the oracle of transformers in f32 (spec 11.11, "The LFM3 oracle";
//! engine-abi "The speech calls of LFM3"): the depth decoder with forced codes, the frames of the tts reply fed with
//! their codes, the codec on fixed codes, and the sampler of `frame`.
//!
//! Each gap prints. A stage asserts `common::STAGE_TOL` and a logit `common::LOGIT_TOL`, the tolerances that the card
//! run of 2026-11-06 measured (spec 11.11, "The tolerances of the LFM3 gates").
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_speech -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::speech::{CODEBOOKS, CODES, END_CODE, FRAME_SAMPLES};
use nagaya_abi::{Engine, FrameDraw};

/// The feed block of the gates, the default of the engine (engine-abi D11).
const BLOCK: usize = 512;

fn golden() -> Option<std::collections::HashMap<String, common::Tensor>> {
    if common::absent(Which::Omni) || common::golden_absent("speech.safetensors") {
        return None;
    }
    Some(common::golden_tensors("speech.safetensors"))
}

fn masks(t: &common::Tensor) -> Vec<bool> {
    t.values.iter().map(|v| *v != 0.0).collect()
}

#[test]
fn the_depth_decoder_with_forced_codes_follows_the_oracle() {
    let Some(golden) = golden() else { return };
    let mut engine = common::open(Which::Omni, 1, BLOCK);
    let hidden = golden["depth.hidden"].f32();
    let codes = golden["depth.codes"].ids();
    let frames = golden["depth.hidden"].shape[0];
    let d = 1024;
    for i in 0..4 {
        let masked = masks(&golden[&format!("depth.mask{i}")]);
        let want = golden[&format!("depth.logits{i}")].f32();
        let (mut worst, mut largest) = (0.0f64, 0.0f64);
        for f in 0..frames {
            let rows = f * CODEBOOKS..(f + 1) * CODEBOOKS;
            let nagaya_mk_lfm3_bmg::engine::DepthProbe { input, trunk, logits } = engine
                .depth_probe(&hidden[f * 2048..(f + 1) * 2048], &codes[rows.clone()], &masked[rows])
                .unwrap();
            if i == 0 {
                let span = f * CODEBOOKS * d..(f + 1) * CODEBOOKS * d;
                let gi = common::rel(&input, &golden["depth.in0"].f32()[span.clone()]);
                let gt = common::rel(&trunk, &golden["depth.trunk0"].f32()[span]);
                println!("depth frame {f}: input relative error {gi:.3e}, trunk {gt:.3e}");
                assert!(
                    gi < common::STAGE_TOL && gt < common::STAGE_TOL,
                    "frame {f}: {gi}, {gt}"
                );
            }
            let want = &want[f * CODEBOOKS * CODES..(f + 1) * CODEBOOKS * CODES];
            worst = worst.max(common::rel(&logits, want));
            largest = largest.max(common::max_abs(&logits, want));
        }
        println!("depth mask {i}: logits relative error {worst:.3e}, largest difference {largest:.3e}");
        assert!(worst < common::STAGE_TOL, "mask {i}: relative error {worst}");
    }
}

#[test]
fn the_frames_of_the_tts_reply_fed_with_their_codes_follow_the_oracle() {
    let Some(golden) = golden() else { return };
    let mut engine = common::open(Which::Omni, 1, BLOCK);
    let prompt = golden["tts.prompt_ids"].ids();
    let sequence = golden["tts.sequence"].ids();
    let codes = golden["tts.codes"].ids();
    let hidden = golden["tts.hidden"].f32();
    let frames = codes.len() / CODEBOOKS;
    assert!(codes[(frames - 1) * CODEBOOKS..].iter().all(|c| *c == END_CODE));
    common::feed_split(&mut engine, 0, 0, &prompt, &common::blocks(prompt.len(), BLOCK));
    let row = |k: usize| &hidden[k * 2048..(k + 1) * 2048];
    let got = engine.hidden_row(0).unwrap();
    let gap = common::rel(&got, row(0));
    println!("tts hidden after the prompt: relative error {gap:.3e}");
    assert!(gap < common::STAGE_TOL, "after the prompt: {gap}");
    let mut worst = 0.0f64;
    let mut last = None;
    for k in 0..frames {
        let end = k + 1 == frames;
        last = engine
            .feed_frame(0, &codes[k * CODEBOOKS..(k + 1) * CODEBOOKS], end)
            .unwrap_or_else(|e| panic!("frame {k}: {}", e.message));
        let gap = common::rel(&engine.hidden_row(0).unwrap(), row(k + 1));
        worst = worst.max(gap);
        if k < 3 || end {
            println!("tts hidden after frame {k}: relative error {gap:.3e}");
        }
        assert!(gap < common::STAGE_TOL, "after frame {k}: {gap}");
    }
    println!("tts hidden after each of {frames} frames: worst relative error {worst:.3e}");
    // After the end frame the reply is text again, and Omni's text is greedy.
    let logits = last.expect("the end frame asked for its logits");
    let top = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |m, (i, v)| if *v > m.1 { (i, *v) } else { m })
        .0;
    assert_eq!(
        top as u32,
        sequence[prompt.len() + frames],
        "the text token after the end frame"
    );
    // The logits of the last position of the reply, after that token.
    let at = (prompt.len() + frames) as u64;
    let out = common::feed_split(&mut engine, 0, at, &sequence[prompt.len() + frames..], &[1]);
    let want = golden["tts.end_logits"].f32();
    let gap = common::max_abs(&out, &want[..engine.vocab()]);
    println!("tts logits of the last position: largest difference {gap:.3e}");
    assert!(gap < common::LOGIT_TOL, "the last logits differ by {gap}");
}

#[test]
fn the_codec_on_the_card_follows_the_oracle_frame_by_frame() {
    let Some(golden) = golden() else { return };
    let mut engine = common::open(Which::Omni, 1, BLOCK);
    let speech = engine.frames().expect("Omni declares speech_frames");
    for (codes, want, label) in [
        (
            golden["codec.codes"].ids(),
            golden["codec.audio_frames"].f32(),
            "codec.codes",
        ),
        (
            {
                let all = golden["tts.codes"].ids();
                all[..all.len() - CODEBOOKS].to_vec()
            },
            golden["tts.audio"].f32(),
            "tts.codes",
        ),
    ] {
        let frames = codes.len() / CODEBOOKS;
        let mut audio = Vec::new();
        let mut sizes = Vec::new();
        for f in 0..frames {
            let piece = speech.codec(0, &codes[f * CODEBOOKS..(f + 1) * CODEBOOKS]).unwrap();
            sizes.push(piece.len());
            audio.extend(piece);
        }
        let tail = speech.codec_clear(0);
        sizes.push(tail.len());
        audio.extend(tail);
        let mut want_sizes = vec![FRAME_SAMPLES; frames + 1];
        want_sizes[0] = 1440;
        want_sizes[frames] = 480;
        assert_eq!(sizes, want_sizes, "{label}: the samples of each call");
        let (rel, abs) = (common::rel(&audio, &want), common::max_abs(&audio, &want));
        println!("{label}: {frames} frames, relative error {rel:.3e}, largest difference {abs:.3e}");
        assert!(rel < common::STAGE_TOL, "{label}: relative error {rel}");
    }
    let full = golden["codec.audio_full"].f32();
    let codes = golden["codec.codes"].ids();
    let mut audio = Vec::new();
    for f in 0..codes.len() / CODEBOOKS {
        audio.extend(speech.codec(0, &codes[f * CODEBOOKS..(f + 1) * CODEBOOKS]).unwrap());
    }
    audio.extend(speech.codec_clear(0));
    println!(
        "codec.codes against the whole clip: relative error {:.3e}",
        common::rel(&audio, &full)
    );
    assert!(
        speech.codec(0, &[END_CODE; CODEBOOKS]).is_err(),
        "the end frame makes no audio"
    );
    assert!(
        speech.codec_clear(0).is_empty(),
        "a slot with no stream gives no samples"
    );
}

#[test]
fn a_frame_samples_the_same_codes_for_the_same_seed_and_feeds_them() {
    let Some(golden) = golden() else { return };
    let mut engine = common::open(Which::Omni, 2, BLOCK);
    let prompt = golden["tts.prompt_ids"].ids();
    let history: Vec<u32> = golden["tts.codes"].ids()[..3 * CODEBOOKS].to_vec();
    let mut runs = Vec::new();
    for slot in 0..2u32 {
        common::feed_split(&mut engine, slot, 0, &prompt, &[prompt.len()]);
        let speech = engine.frames().unwrap();
        let draw = FrameDraw {
            seed: 1234,
            frame: 0,
            history: &history,
        };
        let out = speech.frame(slot, &draw, true).unwrap();
        assert_eq!(out.codes.len(), CODEBOOKS);
        assert!(out.codes.iter().all(|c| (*c as usize) < CODES));
        assert_eq!(out.logits.as_ref().map(|l| l.len()), Some(engine.vocab()));
        assert_eq!(
            engine.slot_high(slot),
            prompt.len() as u64 + 1,
            "the frame is the next position"
        );
        let next = engine
            .frames()
            .unwrap()
            .frame(slot, &FrameDraw { frame: 1, ..draw }, false)
            .unwrap();
        println!("slot {slot}: frame 0 {:?}, frame 1 {:?}", out.codes, next.codes);
        runs.push((out.codes, next.codes));
    }
    assert_eq!(
        runs[0], runs[1],
        "the same prompt, seed and frames give the same codes in each slot"
    );
    use nagaya_abi::Slots;
    engine.slot_clear(0);
    let draw = FrameDraw {
        seed: 1,
        frame: 0,
        history: &[],
    };
    let error = engine.frames().unwrap().frame(0, &draw, false).unwrap_err();
    assert!(error.message.contains("no hidden row"), "{}", error.message);
}

#[test]
fn the_text_model_has_no_speech_path() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 1, BLOCK);
    assert!(engine.frames().is_none());
    let limits = engine.limits();
    assert!(!limits.capabilities.contains(&nagaya_abi::Capability::SpeechFrames));
    assert!(limits.memory.mandatory.iter().all(|t| t.name != "speech"));
}
