//! The speech calls of the engine on a B70 card (engine-abi "Speech"): the sampled run, its quanta, its end, and the
//! refusals. Run with `--test-threads=1`: the tests share one engine.
#![cfg(feature = "xe")]

mod common;

use std::sync::{Mutex, OnceLock};

use nagaya_abi::{Engine, PassErrorKind, PromptPair, Speech, SpeechOut, SpeechSampler};
use nagaya_mk_qwen3tts_bmg::spec::{CODEBOOKS, CODEC_EOS, SAMPLES_PER_FRAME};
use nagaya_mk_qwen3tts_bmg::{Config, Qwen3TtsEngine};
use nagaya_text_qwen3tts::Qwen3TtsFamily;

/// The old `speech_begin`: a feed of text and codec pairs into an empty slot.
fn begin(
    engine: &mut dyn nagaya_abi::Speech,
    slot: u32,
    prompt: &[PromptPair],
    sampler: &nagaya_abi::SpeechSampler,
) -> Result<(), nagaya_abi::PassError> {
    let positions: Vec<_> = prompt.iter().map(|p| nagaya_abi::SpeechPosition::Pair(*p)).collect();
    engine.speech_feed(slot, &positions, Some(sampler))
}

const MAX_FRAMES: u32 = 96;
const QUANTUM: u32 = 6;

fn engine() -> std::sync::MutexGuard<'static, Qwen3TtsEngine> {
    static ENGINE: OnceLock<Mutex<Qwen3TtsEngine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let config = Config {
                model_dir: common::model_dir(),
                slots: 2,
                max_frames: MAX_FRAMES,
                max_prompt: 128,
                frames_per_quantum: QUANTUM,
            };
            Mutex::new(Qwen3TtsEngine::open(config, &common::cards()[0]).expect("open the speech engine"))
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn prompt() -> Vec<PromptPair> {
    let family = Qwen3TtsFamily::from_dir(&common::model_dir()).expect("the family of the checkpoint");
    family
        .prompt(
            "The nagaya is awake.",
            "A calm female voice, clear and soft, unhurried.",
            "english",
        )
        .expect("the prompt")
        .into_iter()
        .map(|p| PromptPair {
            text: p.text,
            codec: p.codec,
        })
        .collect()
}

fn sampler(seed: u64) -> SpeechSampler {
    SpeechSampler {
        temperature: 0.9,
        top_k: 50,
        repetition_penalty: 1.05,
        seed,
    }
}

/// Every quantum of one request, in order, until its end.
fn speak(engine: &mut Qwen3TtsEngine, slot: u32, seed: u64, max: u32) -> Vec<SpeechOut> {
    begin(&mut *engine, slot, &prompt(), &sampler(seed)).expect("begin");
    let mut quanta = Vec::new();
    loop {
        let out = engine.speech_frames(slot, max).expect("frames");
        let end = out.end;
        quanta.push(out);
        if end {
            break;
        }
        assert!(quanta.len() <= MAX_FRAMES as usize, "the request makes no end");
    }
    engine.speech_clear(slot);
    quanta
}

fn joined(quanta: &[SpeechOut]) -> (Vec<u32>, Vec<i16>) {
    (
        quanta.iter().flat_map(|q| q.codes.clone()).collect(),
        quanta.iter().flat_map(|q| q.audio.clone()).collect(),
    )
}

#[test]
fn the_same_seed_gives_the_same_codes_and_audio() {
    if common::absent() {
        return;
    }
    let mut engine = engine();
    let start = std::time::Instant::now();
    let first = speak(&mut engine, 0, 7, QUANTUM);
    let ms = start.elapsed().as_millis();
    let (codes, audio) = joined(&first);
    let frames = codes.len() / CODEBOOKS;
    eprintln!(
        "seed 7: {frames} frames ({:.2} s of audio) in {ms} ms, {} quanta",
        frames as f64 * 0.08,
        first.len()
    );
    assert_eq!(audio.len(), frames * SAMPLES_PER_FRAME);
    assert!(frames > 4, "the line has {frames} frames");
    for q in &first {
        assert!(q.frames <= QUANTUM, "a quantum made {} frames", q.frames);
        assert_eq!(q.codes.len(), q.frames as usize * CODEBOOKS);
    }
    assert!(first.last().unwrap().end);
    assert!(
        codes.iter().step_by(CODEBOOKS).all(|c| *c < 2048),
        "codebook 0 holds an id outside the codebook"
    );
    assert!(codes.iter().all(|c| *c != CODEC_EOS));
    // Another slot, smaller quanta: the same codes and the same audio bits.
    let again = speak(&mut engine, 1, 7, 4);
    assert_eq!(joined(&again), (codes.clone(), audio));
    let other = speak(&mut engine, 0, 8, QUANTUM);
    assert_ne!(joined(&other).0, codes, "seed 8 gave the codes of seed 7");
}

#[test]
fn a_refusal_names_the_rule_and_changes_nothing() {
    if common::absent() {
        return;
    }
    let mut engine = engine();
    let refused = |result: Result<(), nagaya_abi::PassError>| {
        let error = result.expect_err("a refusal");
        assert_eq!(error.kind, PassErrorKind::Refused);
        error.message
    };
    let p = prompt();
    assert!(refused(begin(&mut *engine, 9, &p, &sampler(1))).contains("slot 9 does not exist"));
    assert!(refused(begin(&mut *engine, 0, &[], &sampler(1))).contains("0 positions"));
    let bad = [PromptPair {
        text: None,
        codec: Some(5000),
    }];
    assert!(refused(begin(&mut *engine, 0, &bad, &sampler(1))).contains("outside the vocabularies"));
    let cold = SpeechSampler {
        temperature: 0.0,
        ..sampler(1)
    };
    assert!(refused(begin(&mut *engine, 0, &p, &cold)).contains("temperature 0"));
    assert!(refused(engine.speech_frames(0, 4).map(|_| ())).contains("holds no request"));
    // A slot whose line has not ended refuses a feed, and its own line goes on as if nothing came.
    let alone = joined(&speak(&mut engine, 0, 11, QUANTUM));
    begin(&mut *engine, 0, &p, &sampler(11)).unwrap();
    let mut quanta = vec![engine.speech_frames(0, QUANTUM).unwrap()];
    assert!(refused(begin(&mut *engine, 0, &p, &sampler(12))).contains("has not ended"));
    while !quanta.last().unwrap().end {
        quanta.push(engine.speech_frames(0, QUANTUM).unwrap());
    }
    engine.speech_clear(0);
    assert_eq!(joined(&quanta), alone);
    // A speech engine takes no pass.
    let error = engine.pass(&[], &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(engine.last_error().unwrap().contains("takes no pass"));
}
