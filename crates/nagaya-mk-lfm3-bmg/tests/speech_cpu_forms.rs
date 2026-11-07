//! The CPU forms of the speech path against the oracle of transformers in f32 (spec 11.11, "The LFM3 oracle"): the
//! depth decoder with forced codes, the embedding of a fed frame, and the decoder of LiquidCodec on a whole clip, frame
//! by frame and in chunks of 4. They need no card. The card tests compare the engine with the same oracle.
//!
//! Each test prints its gaps. A form sums in f64, so its gap is the rounding of the f32 reference. The bounds are
//! about ten times the gaps that the first run measured on 2026-11-06. A wrong op gives gaps of 1e-2 or more.

mod common;

use common::Which;
use common::speech::*;

fn masks(t: &common::Tensor) -> Vec<bool> {
    t.values.iter().map(|v| *v != 0.0).collect()
}

#[test]
fn the_depth_decoder_with_forced_codes_follows_the_oracle() {
    if common::absent(Which::Omni) || common::golden_absent("speech.safetensors") {
        return;
    }
    let golden = common::golden_tensors("speech.safetensors");
    let depth = Depth::load(&common::model_dir(Which::Omni));
    let hidden = golden["depth.hidden"].f32();
    let codes = golden["depth.codes"].ids();
    let frames = golden["depth.hidden"].shape[0];
    let d = DEPTH.hidden;
    for i in 0..4 {
        let masked = masks(&golden[&format!("depth.mask{i}")]);
        let want = golden[&format!("depth.logits{i}")].f32();
        let mut worst = (0.0f64, 0.0f64);
        for f in 0..frames {
            let rows = f * CODEBOOKS..(f + 1) * CODEBOOKS;
            let (input, trunk, logits) = depth.run(
                &hidden[f * 2048..(f + 1) * 2048],
                &codes[rows.clone()],
                &masked[rows.clone()],
            );
            if i == 0 {
                let (gi, gt) = (
                    common::rel(
                        &input,
                        &golden["depth.in0"].f32()[f * CODEBOOKS * d..(f + 1) * CODEBOOKS * d],
                    ),
                    common::rel(
                        &trunk,
                        &golden["depth.trunk0"].f32()[f * CODEBOOKS * d..(f + 1) * CODEBOOKS * d],
                    ),
                );
                println!("depth frame {f}: input relative error {gi:.3e}, trunk {gt:.3e}");
                assert!(
                    gi < 2e-6 && gt < 5e-6,
                    "frame {f}: the input or the trunk is wrong: {gi}, {gt}"
                );
            }
            let want = &want[f * CODEBOOKS * CODES..(f + 1) * CODEBOOKS * CODES];
            worst.0 = worst.0.max(common::rel(&logits, want));
            worst.1 = worst.1.max(common::max_abs(&logits, want));
        }
        println!(
            "depth mask {i}: logits relative error {:.3e}, largest difference {:.3e}",
            worst.0, worst.1
        );
        assert!(
            worst.0 < 5e-6 && worst.1 < 2e-4,
            "mask {i}: the logits are wrong: {worst:?}"
        );
    }
}

#[test]
fn a_fed_frame_is_the_sum_of_twelve_rows_of_the_codebook_table() {
    if common::absent(Which::Omni) || common::golden_absent("speech.safetensors") {
        return;
    }
    let golden = common::golden_tensors("speech.safetensors");
    let depth = Depth::load(&common::model_dir(Which::Omni));
    let codes = golden["depth.codes"].ids();
    let want = golden["depth.audio_out_embeds"].f32();
    let mut got = Vec::new();
    for f in 0..codes.len() / CODEBOOKS {
        got.extend(depth.frame_embedding(&codes[f * CODEBOOKS..(f + 1) * CODEBOOKS]));
    }
    let gap = common::max_abs(&got, &want);
    println!("frame embedding: largest difference {gap:.3e}");
    assert!(gap < 1e-6, "the sum of the rows is wrong: {gap}");
}

#[test]
fn the_codec_decodes_a_clip_whole_frame_by_frame_and_in_chunks_as_the_oracle_does() {
    if common::absent(Which::Omni) || common::golden_absent("speech.safetensors") {
        return;
    }
    let golden = common::golden_tensors("speech.safetensors");
    let codec = Codec::load(&common::model_dir(Which::Omni));
    let codes = golden["codec.codes"].ids();
    let frames = codes.len() / CODEBOOKS;
    let full = codec.decode(&codes);
    assert_eq!(full.len(), frames * FRAME_SAMPLES, "a frame gives 1,920 samples");
    let want = golden["codec.audio_full"].f32();
    let (rel, abs) = (common::rel(&full, &want), common::max_abs(&full, &want));
    println!("codec whole: relative error {rel:.3e}, largest difference {abs:.3e}");
    assert!(rel < 1e-5 && abs < 1e-5, "the whole clip is wrong: {rel}, {abs}");
    for (chunk, label, sizes) in [
        (1usize, "frames", {
            let mut s = vec![1920usize; 20];
            s[0] = 1440;
            s[19] = 2400;
            s
        }),
        (4, "chunks4", vec![7200, 7680, 7680, 7680, 8160]),
    ] {
        let (audio, got_sizes) = codec.decode_stream(&codes, chunk);
        assert_eq!(got_sizes, sizes, "{label}: the pieces of the stream");
        let to_full = common::max_abs(&audio, &full);
        let want = golden[&format!("codec.audio_{label}")].f32();
        let (rel, abs) = (common::rel(&audio, &want), common::max_abs(&audio, &want));
        println!(
            "codec {label}: against the whole form {to_full:.3e}; against the oracle relative error {rel:.3e}, \
             largest difference {abs:.3e}"
        );
        assert!(
            to_full < 1e-9,
            "{label}: the stream of the form differs from its whole clip: {to_full}"
        );
        assert!(rel < 1e-4 && abs < 1e-4, "{label}: the stream is wrong: {rel}, {abs}");
    }
}

#[test]
fn the_codec_decodes_the_whole_tts_reply_as_the_oracle_does() {
    if common::absent(Which::Omni) || common::golden_absent("speech.safetensors") {
        return;
    }
    let golden = common::golden_tensors("speech.safetensors");
    let codec = Codec::load(&common::model_dir(Which::Omni));
    let all = golden["tts.codes"].ids();
    let frames = all.len() / CODEBOOKS;
    assert!(
        all[(frames - 1) * CODEBOOKS..].iter().all(|c| *c == END_CODE),
        "the last frame of the reply is the end frame"
    );
    let audio = codec.decode(&codes_of(&all, frames - 1));
    let want = golden["tts.audio"].f32();
    let (rel, abs) = (common::rel(&audio, &want), common::max_abs(&audio, &want));
    println!(
        "codec tts reply of {} frames: relative error {rel:.3e}, largest difference {abs:.3e}",
        frames - 1
    );
    assert!(rel < 1e-5 && abs < 1e-5, "the reply is wrong: {rel}, {abs}");
}
