//! The vision tower of `LFM3-Omni-2B` on a B70 card against transformers in f32 on the CPU (spec 8.3, 11.11 "The LFM3
//! oracle"): each stage of the tower for the two images of `vision.safetensors`, and the logits of a pass that reads
//! an image (`lfm3-omni.media.json`). The CPU forms of `vision_cpu.rs` gave gaps of 3.8e-7 to 5.7e-6 for the same
//! stages.
//!
//! Each gap prints. A stage asserts `common::STAGE_TOL` and a logit `common::LOGIT_TOL`, the tolerances that the card
//! run of 2026-11-06 measured (spec 11.11, "The tolerances of the LFM3 gates").
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_vision -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::time::Instant;

use common::Which;
use nagaya_abi::{Engine, OutputsWanted, PassErrorKind, Segment, SegmentInputs, State};
use nagaya_mk_lfm3_bmg::media::{self, PATCH_ROW, Span, SpanKind};
use nagaya_mk_lfm3_bmg::weights::f32_bytes;

/// The placeholder of an image (`image_token_id` of Omni).
const IMAGE: u32 = 125_030;

/// The images of the oracle.
const IMAGES: [&str; 2] = ["small", "mid"];

/// The grid of an image of the oracle, down by across.
fn grid(golden: &std::collections::HashMap<String, common::Tensor>, name: &str) -> (usize, usize) {
    let g = golden[&format!("{name}.grid_thw")].ids();
    assert_eq!(g[0], 1, "an image has one frame");
    (g[1] as usize, g[2] as usize)
}

/// The image span of `ids`: its placeholders, which are one run.
fn span_of(ids: &[u32], grid_h: usize, grid_w: usize) -> Span {
    let first = ids.iter().position(|t| *t == IMAGE).expect("the prompt holds an image");
    let count = ids.iter().filter(|t| **t == IMAGE).count();
    assert!(
        ids[first..first + count].iter().all(|t| *t == IMAGE),
        "the placeholders are one run"
    );
    let span = Span {
        kind: SpanKind::Image {
            grid: [1, grid_h, grid_w],
        },
        first: first as u64,
        count,
    };
    assert_eq!(media::decode(&media::encode(&[span])).unwrap(), vec![span]);
    span
}

/// The segment of the block of `ids` from `from`, with the rows of `span` whole when the block reads it, and the logits
/// of its last position when `last`.
fn block(slot: u32, from: u64, ids: &[u32], span: &Span, pixels: &[u8], last: bool) -> Segment {
    let end = from + ids.len() as u64;
    let reads = span.first < end && from < span.first + span.count as u64;
    let logit_rows = if last { vec![end - 1] } else { Vec::new() };
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: if reads {
            SegmentInputs::TokensAndRows {
                tokens: ids.to_vec(),
                rows: pixels.to_vec(),
            }
        } else {
            SegmentInputs::Tokens(ids.to_vec())
        },
        positions: (from..end).collect(),
        meta: if reads { media::encode(&[*span]) } else { Vec::new() },
        outputs: OutputsWanted { logit_rows },
    }
}

/// Feed `ids` with the image `span` into slot 0 in blocks of `block_len`, and give the logits of the last position
/// with the time of each pass.
fn feed(engine: &mut dyn Engine, ids: &[u32], span: &Span, pixels: &[u8], block_len: usize) -> (Vec<f32>, Vec<u64>) {
    let mut last = Vec::new();
    let mut ms = Vec::new();
    for (i, chunk) in ids.chunks(block_len).enumerate() {
        let from = (i * block_len) as u64;
        let is_last = from as usize + chunk.len() == ids.len();
        let out = engine
            .pass(&[block(0, from, chunk, span, pixels, is_last)], &[])
            .unwrap_or_else(|e| panic!("the pass at {from} of {} tokens: {}", chunk.len(), e.message));
        ms.push(out.ms);
        if let Some(row) = out.outputs.first() {
            last = row
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
        }
    }
    (last, ms)
}

#[test]
fn each_stage_of_the_tower_follows_transformers_on_the_card() {
    if common::absent(Which::Omni) || common::golden_absent("vision.safetensors") {
        return;
    }
    let golden = common::golden_tensors("vision.safetensors");
    let mut engine = common::open(Which::Omni, 1, 512);
    for name in IMAGES {
        let (gh, gw) = grid(&golden, name);
        let pixels = golden[&format!("{name}.pixel_values")].f32();
        let start = Instant::now();
        let [patches, tower, rows] = engine.vision_stages(&pixels, gh, gw).unwrap();
        let ms = start.elapsed().as_millis();
        for (stage, got, key) in [
            ("patch embedding", &patches, "patches"),
            ("tower", &tower, "tower"),
            ("projector", &rows, "rows"),
        ] {
            let want = golden[&format!("{name}.{key}")].f32();
            let gap = common::rel(got, &want);
            println!(
                "{name} ({gh} by {gw} patches) {stage}: relative error {gap:.3e}, largest difference {:.3e}",
                common::max_abs(got, &want)
            );
            assert!(gap < common::STAGE_TOL, "{name} {stage}: relative error {gap}");
        }
        println!("{name}: the three stages took {ms} ms");
        // The same pixels give the same bits (D3).
        let again = engine.vision_stages(&pixels, gh, gw).unwrap();
        assert_eq!(
            common::first_other_bits(&again[2], &rows),
            None,
            "{name}: a second run differs"
        );
    }
}

#[test]
fn a_pass_that_reads_an_image_gives_the_logits_of_transformers() {
    if common::absent(Which::Omni)
        || common::golden_absent("vision.safetensors")
        || common::golden_absent("lfm3-omni.media.json")
    {
        return;
    }
    let golden = common::golden_tensors("vision.safetensors");
    let oracle = common::golden_json("lfm3-omni.media.json");
    let prompts: Vec<&serde_json::Value> = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["name"].as_str().unwrap().starts_with("image_"))
        .collect();
    assert_eq!(prompts.len(), IMAGES.len());
    // The whole prompt in one block, then in blocks of 32, so the image spans 3 or 4 passes.
    let mut whole = Vec::new();
    for block_len in [512usize, 32] {
        let mut engine = common::open(Which::Omni, 1, block_len);
        for (i, prompt) in prompts.iter().enumerate() {
            let name = prompt["name"].as_str().unwrap();
            let image = name.trim_start_matches("image_");
            let ids: Vec<u32> = prompt["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(
                ids,
                golden[&format!("{image}.ids")].ids(),
                "{name}: the ids of the two oracle files"
            );
            let (gh, gw) = grid(&golden, image);
            let span = span_of(&ids, gh, gw);
            let pixels = f32_bytes(&golden[&format!("{image}.pixel_values")].f32());
            assert_eq!(pixels.len(), gh * gw * PATCH_ROW * 4);
            engine.slots().unwrap().slot_clear(0);
            let (logits, ms) = feed(&mut engine, &ids, &span, &pixels, block_len);
            assert_eq!(logits.len(), engine.vocab(), "a logit row has the width of the family");
            let top: Vec<usize> = prompt["top_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let want: Vec<f64> = prompt["top_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let gap = top
                .iter()
                .zip(&want)
                .map(|(id, w)| (logits[*id] as f64 - w).abs())
                .fold(0.0f64, f64::max);
            let best = (0..logits.len())
                .max_by(|a, b| logits[*a].total_cmp(&logits[*b]))
                .unwrap();
            println!(
                "{name} ({} ids, the image at {} for {}, blocks of {block_len}, passes of {ms:?} ms): top-64 gap \
                 {gap:.3e}, logsumexp {:.6} against {:.6} of 128,000 rows, top {best} against {}",
                ids.len(),
                span.first,
                span.count,
                common::logsumexp(&logits),
                prompt["logsumexp"].as_f64().unwrap(),
                top[0]
            );
            assert_eq!(best, top[0], "{name}: the top token differs");
            assert!(gap < common::LOGIT_TOL, "{name}: a top-64 logit differs by {gap}");
            if block_len == 512 {
                whole.push(logits);
            } else {
                println!(
                    "{name}: blocks of 32 against one block: largest difference {:.3e}",
                    common::max_abs(&logits, &whole[i])
                );
            }
        }
    }
}

#[test]
fn the_largest_image_fits_the_tower_and_each_pass_is_timed() {
    if common::absent(Which::Omni) {
        return;
    }
    // 64 by 128 patches: 8,192 patches and 2,048 positions, the most of an image (spec 11.11).
    let (gh, gw) = (64usize, 128usize);
    let pixels = common::values(gh * gw * PATCH_ROW, 77);
    let mut engine = common::open(Which::Omni, 1, 512);
    let start = Instant::now();
    let [_, _, rows] = engine.vision_stages(&pixels, gh, gw).unwrap();
    println!(
        "the three stages of an image of {gh} by {gw} patches took {} ms",
        start.elapsed().as_millis()
    );
    assert_eq!(rows.len(), gh * gw / 4 * 2048);
    assert!(
        rows.iter().all(|v| v.is_finite()),
        "a row of the projector is not finite"
    );
    // A prompt of the image between two text tokens: 2,050 positions, so 5 passes of 512 and each one runs the tower.
    let mut ids = vec![1u32];
    ids.extend(std::iter::repeat_n(IMAGE, gh * gw / 4));
    ids.push(2);
    let span = span_of(&ids, gh, gw);
    let (logits, ms) = feed(&mut engine, &ids, &span, &f32_bytes(&pixels), 512);
    println!("the passes of the largest image took {ms:?} ms; the engine declares max_quantum_ms 2000");
    assert!(logits.iter().all(|v| v.is_finite()), "a logit is not finite");
}

#[test]
fn the_text_model_refuses_an_image() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 1, 512);
    let ids = [1u32, IMAGE, 2];
    let span = Span {
        kind: SpanKind::Image { grid: [1, 2, 2] },
        first: 1,
        count: 1,
    };
    let pixels = vec![0u8; span.bytes()];
    let error = engine
        .pass(&[block(0, 0, &ids, &span, &pixels, true)], &[])
        .unwrap_err();
    assert!(matches!(error.kind, PassErrorKind::Refused), "{:?}", error.kind);
    assert!(error.message.contains("no vision tower"), "{}", error.message);
    assert_eq!(
        engine.slots().unwrap().slot_high(0),
        0,
        "a refused pass changes nothing"
    );
}

/// The placeholder of a clip (`audio_token_id` of Omni).
const AUDIO: u32 = 125_031;

/// A pass that reads a clip gives the logits of transformers: the audio encoder runs inside the pass, and its rows
/// replace the placeholders, as the rows of an image do. The clip is the chirp of `audio_in.safetensors`.
#[test]
fn a_pass_that_reads_a_clip_gives_the_logits_of_transformers() {
    if common::absent(Which::Omni)
        || common::golden_absent("audio_in.safetensors")
        || common::golden_absent("lfm3-omni.media.json")
    {
        return;
    }
    let golden = common::golden_tensors("audio_in.safetensors");
    let oracle = common::golden_json("lfm3-omni.media.json");
    let prompt = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "audio_chirp")
        .expect("the oracle holds the chirp prompt");
    let ids: Vec<u32> = prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids, golden["ids"].ids(), "the ids of the two oracle files");
    // The oracle holds the bins first, `[128][frames]`. A span holds one row of 128 bins for each frame.
    let bins_first = golden["mel"].f32();
    let frames = bins_first.len() / 128;
    let mel: Vec<f32> = (0..frames * 128)
        .map(|i| bins_first[(i % 128) * frames + i / 128])
        .collect();
    let first = ids.iter().position(|t| *t == AUDIO).expect("the prompt holds a clip");
    let span = Span {
        kind: SpanKind::Audio { frames },
        first: first as u64,
        count: ids.iter().filter(|t| **t == AUDIO).count(),
    };
    assert_eq!(span.count, frames.div_ceil(8));
    let rows = f32_bytes(&mel);
    let top: Vec<usize> = prompt["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want: Vec<f64> = prompt["top_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    // The whole prompt in one block, then in blocks of 8, so the clip spans 3 or 4 passes.
    let mut whole = Vec::new();
    for block_len in [512usize, 8] {
        let mut engine = common::open(Which::Omni, 1, block_len);
        let (logits, ms) = feed(&mut engine, &ids, &span, &rows, block_len);
        let gap = top
            .iter()
            .zip(&want)
            .map(|(id, w)| (logits[*id] as f64 - w).abs())
            .fold(0.0f64, f64::max);
        let best = (0..logits.len())
            .max_by(|a, b| logits[*a].total_cmp(&logits[*b]))
            .unwrap();
        println!(
            "audio_chirp ({} ids, the clip of {frames} frames at {} for {}, blocks of {block_len}, passes of {ms:?} \
             ms): top-64 gap {gap:.3e}, logsumexp {:.6} against {:.6}, top {best} against {}",
            ids.len(),
            span.first,
            span.count,
            common::logsumexp(&logits),
            prompt["logsumexp"].as_f64().unwrap(),
            top[0]
        );
        assert_eq!(best, top[0], "audio_chirp: the top token differs");
        assert!(gap < common::LOGIT_TOL, "audio_chirp: a top-64 logit differs by {gap}");
        if block_len == 512 {
            whole = logits;
        } else {
            println!(
                "audio_chirp: blocks of 8 against one block: largest difference {:.3e}",
                common::max_abs(&logits, &whole)
            );
        }
    }
}

/// The placeholder of a frame of speech.
const AUDIO_OUT: u32 = 125_032;

/// An assistant audio part feeds the codes of its frames (spec 11.11, "Read it back"): each position takes the sum of
/// its 12 codebook rows, the end frame last, and the logits after the conversation are those of transformers.
#[test]
fn a_pass_that_reads_the_speech_of_an_assistant_part_gives_the_logits_of_transformers() {
    if common::absent(Which::Omni)
        || common::golden_absent("audio_out.safetensors")
        || common::golden_absent("lfm3-omni.audio_out.json")
    {
        return;
    }
    let golden = common::golden_tensors("audio_out.safetensors");
    let oracle = common::golden_json("lfm3-omni.audio_out.json");
    let prompt = &oracle["prompts"][0];
    let ids: Vec<u32> = prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids, golden["ids"].ids(), "the ids of the two oracle files");
    let mut codes: Vec<u32> = golden["codes"].ids();
    // The end frame follows the speech, as the processor writes it.
    codes.extend(std::iter::repeat_n(2016u32, media::FRAME_CODES));
    let frames = codes.len() / media::FRAME_CODES;
    let first = ids
        .iter()
        .position(|t| *t == AUDIO_OUT)
        .expect("the prompt holds speech");
    let span = Span {
        kind: SpanKind::Codes,
        first: first as u64,
        count: ids.iter().filter(|t| **t == AUDIO_OUT).count(),
    };
    assert_eq!(span.count, frames, "a position for each frame and the end frame");
    let rows: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
    let top: Vec<usize> = prompt["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want: Vec<f64> = prompt["top_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for block_len in [512usize, 8] {
        let mut engine = common::open(Which::Omni, 1, block_len);
        let (logits, ms) = feed(&mut engine, &ids, &span, &rows, block_len);
        let gap = top
            .iter()
            .zip(&want)
            .map(|(id, w)| (logits[*id] as f64 - w).abs())
            .fold(0.0f64, f64::max);
        let best = (0..logits.len())
            .max_by(|a, b| logits[*a].total_cmp(&logits[*b]))
            .unwrap();
        println!(
            "audio_out ({} ids, {frames} frames at {}, blocks of {block_len}, passes of {ms:?} ms): top-64 gap \
             {gap:.3e}, top {best} against {}",
            ids.len(),
            span.first,
            top[0]
        );
        assert_eq!(best, top[0], "audio_out: the top token differs");
        assert!(gap < common::LOGIT_TOL, "audio_out: a top-64 logit differs by {gap}");
    }
}
