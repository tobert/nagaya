//! The held state of a speech context on a B70 card (engine-abi "Held speech state"): a feed after state, frame
//! positions, and the part `talker_kv` that a snapshot reads and a restore writes. Run with `--test-threads=1`: the
//! tests share one engine.
#![cfg(feature = "xe")]

mod common;

use std::sync::{Mutex, OnceLock};

use nagaya_abi::{Engine, PromptPair, SpeechPosition, SpeechSampler, State, WriteResult};
use nagaya_family_qwen3tts::Qwen3TtsFamily;
use nagaya_mk_qwen3tts_bmg::spec::{CODEBOOKS, CODEC_EOS, TTS_PAD};
use nagaya_mk_qwen3tts_bmg::{Config, KV_ROW_BYTES, Qwen3TtsEngine};

const MAX_FRAMES: u32 = 96;
const VOICE: &str = "The calm onboard AI of a starship. Female, clear and soft, slightly processed, unhurried.";

fn engine() -> std::sync::MutexGuard<'static, Qwen3TtsEngine> {
    static ENGINE: OnceLock<Mutex<Qwen3TtsEngine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let config = Config {
                model_dir: common::model_dir(),
                slots: 2,
                max_frames: MAX_FRAMES,
                // Smaller than a line with its frames, so a feed of frames runs in chunks.
                max_prompt: 128,
                frames_per_quantum: 6,
            };
            Mutex::new(Qwen3TtsEngine::open(config, &common::cards()[0]).expect("open the speech engine"))
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The positions of a line, with the voice when it is the first line of a context.
fn line(text: &str, voice: &str) -> Vec<SpeechPosition> {
    let family = Qwen3TtsFamily::from_dir(&common::model_dir()).expect("the family of the checkpoint");
    family
        .prompt(text, voice, "english")
        .expect("the prompt")
        .into_iter()
        .map(|p| {
            SpeechPosition::Pair(PromptPair {
                text: p.text,
                codec: p.codec,
            })
        })
        .collect()
}

/// The position that ends a line's audio: `tts_pad` with the codec end (spec 11.8a).
fn end_of_line() -> SpeechPosition {
    SpeechPosition::Pair(PromptPair {
        text: Some(TTS_PAD),
        codec: Some(CODEC_EOS),
    })
}

fn sampler(seed: u64) -> SpeechSampler {
    SpeechSampler {
        temperature: 0.9,
        top_k: 50,
        repetition_penalty: 1.05,
        seed,
    }
}

/// Feed `positions` into `slot` and make frames to the end of the line. Give the codes of each frame.
fn speak(engine: &mut Qwen3TtsEngine, slot: u32, positions: &[SpeechPosition], seed: u64) -> Vec<Vec<u32>> {
    let speech = engine.speech().expect("a speech engine");
    speech.speech_feed(slot, positions, Some(&sampler(seed))).expect("feed");
    let mut codes = Vec::new();
    loop {
        let out = speech.speech_frames(slot, 6).expect("frames");
        codes.extend(out.codes.chunks(CODEBOOKS).map(<[u32]>::to_vec));
        if out.end {
            return codes;
        }
        assert!(codes.len() <= MAX_FRAMES as usize, "the line makes no end");
    }
}

fn kv(engine: &mut Qwen3TtsEngine, slot: u32) -> Vec<u8> {
    let rows = engine.slots().expect("slots").slot_high(slot);
    let mut out = vec![0u8; rows as usize * KV_ROW_BYTES];
    engine
        .parts()
        .expect("parts")
        .cells_read(State::Slot(slot), 0, 0, rows, &mut out)
        .expect("read talker_kv");
    out
}

/// A slot restored from another slot's `talker_kv` speaks the next line with the same codes as the slot that held the
/// state. The restore is a copy of bytes, so the bits are the same.
#[test]
fn a_restored_slot_speaks_the_next_line_as_the_slot_that_held_the_state() {
    if common::absent() {
        return;
    }
    let mut engine = engine();
    for slot in [0, 1] {
        engine.speech().unwrap().speech_clear(slot);
    }
    let first = speak(&mut engine, 0, &line("Docking clamps released.", VOICE), 11);
    assert!(!first.is_empty());
    let held = kv(&mut engine, 0);
    let rows = (held.len() / KV_ROW_BYTES) as u64;
    let parts = engine.parts().unwrap();
    assert_eq!(parts.cells_write(State::Slot(1), 0, 0, rows, &held), WriteResult::Ok);
    assert_eq!(parts.set_n(State::Slot(1), rows), WriteResult::Ok);
    assert_eq!(kv(&mut engine, 1), held, "a write and a read give the same bytes");

    let mut next = vec![end_of_line()];
    next.extend(line("Course laid in for the outer ring.", ""));
    let kept = speak(&mut engine, 0, &next, 23);
    let restored = speak(&mut engine, 1, &next, 23);
    assert_eq!(kept, restored, "the same state, feed and seed give the same codes");
    for slot in [0, 1] {
        engine.speech().unwrap().speech_clear(slot);
    }
}

/// Frame positions feed a line's frames from the host. The talker KV that they give is the KV of the frames that the
/// engine made, within the rounding of passes of another size (the engine does not declare `segments_exact`).
#[test]
fn fed_frames_give_the_talker_kv_of_the_frames_that_the_engine_made() {
    if common::absent() {
        return;
    }
    let mut engine = engine();
    for slot in [0, 1] {
        engine.speech().unwrap().speech_clear(slot);
    }
    let prompt = line("Docking clamps released.", VOICE);
    let codes = speak(&mut engine, 0, &prompt, 5);
    let made = kv(&mut engine, 0);
    let mut fed = prompt.clone();
    fed.extend(codes.iter().map(|c| SpeechPosition::Frame(c.clone())));
    // The prompt and the frames are more than one chunk of 128 positions when the line has more than about 80 frames.
    engine
        .speech()
        .unwrap()
        .speech_feed(1, &fed, Some(&sampler(5)))
        .expect("feed of frames");
    let again = kv(&mut engine, 1);
    assert_eq!(
        again.len(),
        made.len(),
        "the same rows: {} and {}",
        again.len() / KV_ROW_BYTES,
        made.len() / KV_ROW_BYTES
    );
    let floats = |b: &[u8]| {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>()
    };
    let (a, b) = (floats(&made), floats(&again));
    let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!(
        "{} frames, {} rows: the largest difference is {worst} of a largest value {scale}",
        codes.len(),
        a.len() / (KV_ROW_BYTES / 4)
    );
    assert!(worst <= scale * 1e-3, "fed frames give another KV: {worst} of {scale}");
    for slot in [0, 1] {
        engine.speech().unwrap().speech_clear(slot);
    }
}

/// A feed into a slot whose line has not ended is refused, and so is a frame with the wrong count of codes.
#[test]
fn a_feed_during_a_line_and_a_short_frame_are_refused() {
    if common::absent() {
        return;
    }
    let mut engine = engine();
    let speech = engine.speech().unwrap();
    speech.speech_clear(0);
    speech
        .speech_feed(0, &line("One.", VOICE), Some(&sampler(1)))
        .expect("feed");
    let during = speech.speech_feed(0, &line("Two.", ""), Some(&sampler(1))).unwrap_err();
    assert!(during.message.contains("has not ended"), "{}", during.message);
    speech.speech_clear(0);
    let short = speech
        .speech_feed(0, &[SpeechPosition::Frame(vec![1, 2, 3])], Some(&sampler(1)))
        .unwrap_err();
    assert!(short.message.contains("16 codes"), "{}", short.message);
}

/// A feed with no sampler holds its rows and starts no line. A feed of the rest with a sampler then gives the KV of one
/// feed of all the positions, within the rounding of passes of another size.
#[test]
fn a_feed_with_no_sampler_holds_rows_and_starts_no_line() {
    if common::absent() {
        return;
    }
    let mut engine = engine();
    for slot in [0, 1] {
        engine.speech().unwrap().speech_clear(slot);
    }
    let positions = line("Docking clamps released.", VOICE);
    let cut = positions.len() / 2;
    let speech = engine.speech().unwrap();
    speech
        .speech_feed(0, &positions[..cut], None)
        .expect("feed with no sampler");
    let idle = speech.speech_frames(0, 6).unwrap_err();
    assert!(idle.message.contains("holds no request"), "{}", idle.message);
    speech
        .speech_feed(0, &positions[cut..], Some(&sampler(3)))
        .expect("feed of the rest");
    speech.speech_feed(1, &positions, Some(&sampler(3))).expect("one feed");
    assert_eq!(engine.slots().unwrap().slot_high(0), positions.len() as u64);
    let (split, whole) = (kv(&mut engine, 0), kv(&mut engine, 1));
    let floats = |b: &[u8]| {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>()
    };
    let (a, b) = (floats(&whole), floats(&split));
    let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("a split feed: the largest difference is {worst} of a largest value {scale}");
    assert!(
        worst <= scale * 1e-3,
        "a split feed gives another KV: {worst} of {scale}"
    );
    for slot in [0, 1] {
        engine.speech().unwrap().speech_clear(slot);
    }
}
