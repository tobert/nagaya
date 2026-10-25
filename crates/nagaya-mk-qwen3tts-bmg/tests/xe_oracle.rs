//! The oracle of the speech engine (spec 8.3): a forced run of the reference's prompt and codes on a B70 card, against
//! the float32 reference of `tools/qwen3tts_reference.py`. The speech oracle forces the codes, so the sampler's
//! random stream does not enter the comparison.
#![cfg(feature = "xe")]

mod common;

use nagaya_abi::PromptPair;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, bf16_to_f32};
use nagaya_mk_qwen3tts_bmg::spec::{CODEBOOKS, CODEC_VOCAB, CP_VOCAB, HIDDEN, SAMPLES_PER_FRAME};
use nagaya_mk_qwen3tts_bmg::{Config, Qwen3TtsEngine};

fn config(frames_per_quantum: u32) -> Config {
    Config {
        model_dir: common::model_dir(),
        slots: 1,
        max_frames: 64,
        max_prompt: 64,
        frames_per_quantum,
    }
}

/// The reference's prompt, with -1 as an absent id.
fn golden_prompt(golden: &std::collections::HashMap<String, common::Tensor>) -> Vec<PromptPair> {
    let id = |v: f64| (v >= 0.0).then_some(v as u32);
    golden["prompt_text_ids"]
        .values
        .iter()
        .zip(&golden["prompt_codec_ids"].values)
        .map(|(text, codec)| PromptPair {
            text: id(*text),
            codec: id(*codec),
        })
        .collect()
}

#[test]
fn the_prompt_rows_match_the_reference_and_an_absent_text_adds_nothing() {
    if common::absent() || common::golden_absent("forced.safetensors") {
        return;
    }
    let golden = common::golden("forced.safetensors");
    let prompt = golden_prompt(&golden);
    let mut engine = Qwen3TtsEngine::open(config(4), &common::cards()[0]).expect("open the speech engine");
    let rows = engine.prompt_rows(&prompt).expect("the prompt rows");
    let err = common::rel(&rows, &golden["talker_input_embeds"].f32());
    eprintln!("prompt rows: relative error {err:.3e}");
    assert!(err < 1e-5, "the prompt rows differ by {err:.3e}");
    // Every position of the reference's prompt has a text id. A position with no text is the codec embedding alone.
    let codec = 2154u32;
    let rows = engine
        .prompt_rows(&[
            PromptPair {
                text: None,
                codec: Some(codec),
            },
            PromptPair {
                text: None,
                codec: None,
            },
        ])
        .expect("the prompt rows");
    let map = MappedSafeTensors::open(&common::model_dir().join("model.safetensors")).unwrap();
    let view = map.view().unwrap();
    let table = view
        .bf16_tensor("talker.model.codec_embedding.weight", &[CODEC_VOCAB, HIDDEN])
        .unwrap();
    let want: Vec<f32> = table[codec as usize * HIDDEN * 2..(codec as usize + 1) * HIDDEN * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
        .collect();
    assert_eq!(&rows[..HIDDEN], &want[..]);
    assert!(
        rows[HIDDEN..].iter().all(|v| *v == 0.0),
        "a position with no ids is not zero"
    );
}

#[test]
fn a_forced_run_matches_the_reference() {
    if common::absent() || common::golden_absent("forced.safetensors") {
        return;
    }
    let golden = common::golden("forced.safetensors");
    let prompt = golden_prompt(&golden);
    let frames = golden["codes"].shape[0];
    let codes: Vec<[u32; CODEBOOKS]> = (0..frames)
        .map(|f| std::array::from_fn(|g| golden["codes"].values[f * CODEBOOKS + g] as u32))
        .collect();
    let start = std::time::Instant::now();
    // Quanta of 4 frames decode 17 frames as 4 + 4 + 4 + 4 + 1, so the codec carries its state four times.
    let mut engine = Qwen3TtsEngine::open(config(4), &common::cards()[0]).expect("open the speech engine");
    eprintln!("open: {} ms", start.elapsed().as_millis());
    let start = std::time::Instant::now();
    let out = engine.forced(&prompt, &codes).expect("the forced run");
    eprintln!("forced run of {frames} frames: {} ms", start.elapsed().as_millis());

    let mut worst = [0.0f64; 3];
    let mut flips = 0;
    for step in 0..=frames {
        let want = golden["talker_logits"].row(step, CODEC_VOCAB);
        let got = &out.talker_logits[step];
        worst[0] = worst[0].max(common::rel(got, &want));
        worst[1] = worst[1].max(common::rel(
            &out.talker_hidden[step],
            &golden["talker_hidden"].row(step, HIDDEN),
        ));
        // The reference's codebook 0 at this step is the code of the next frame, or end-of-speech after the last.
        if common::argmax(&got[..2048]) != common::argmax(&want[..2048]) {
            flips += 1;
        }
    }
    for frame in 0..frames {
        let want = golden["cp_logits"].row(frame, (CODEBOOKS - 1) * CP_VOCAB);
        worst[2] = worst[2].max(common::rel(&out.cp_logits[frame], &want));
    }
    let audio = common::rel(&out.audio, &golden["audio_full"].f32());
    eprintln!(
        "relative error: talker logits {:.3e}, talker hidden {:.3e}, code predictor logits {:.3e}, audio {audio:.3e}; \
         {flips} argmax flips of codebook 0",
        worst[0], worst[1], worst[2]
    );
    assert_eq!(out.audio.len(), frames * SAMPLES_PER_FRAME);
    assert!(worst[0] < 1e-2, "the talker logits differ by {:.3e}", worst[0]);
    assert!(worst[1] < 1e-2, "the talker hidden state differs by {:.3e}", worst[1]);
    assert!(worst[2] < 1e-2, "the code predictor logits differ by {:.3e}", worst[2]);
    assert!(audio < 1e-2, "the audio differs by {audio:.3e}");
    // The PCM is the audio, clamped and rounded to 16 bits with ties to even, as the op test fixes it.
    let pcm: Vec<i16> = out
        .audio
        .iter()
        .map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16)
        .collect();
    assert_eq!(out.pcm, pcm);
}
