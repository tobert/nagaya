//! The conformance suite of spec 8.3 on a B70 card. A speech engine has no parts, no stages and no feed rule
//! (engine-abi "Speech"), so the gates that apply are the identity of two opens and the admission gate. The speech
//! oracle is `xe_oracle.rs`.
#![cfg(feature = "xe")]

mod common;

use nagaya_conformance::run_all;
use nagaya_mk_qwen3tts_bmg::{Config, Qwen3TtsEngine};

#[test]
fn the_conformance_suite_passes_on_the_card() {
    if common::absent() {
        return;
    }
    let card = common::cards()[0].clone();
    let suite = run_all(&mut || {
        let config = Config {
            model_dir: common::model_dir(),
            slots: 1,
            max_frames: 64,
            max_prompt: 64,
            frames_per_quantum: 4,
        };
        Qwen3TtsEngine::open(config, &card).expect("open the speech engine")
    });
    println!("{suite}");
    assert!(suite.passed(), "{suite}");
    let names: Vec<&str> = suite.not_applicable().map(|gate| gate.name).collect();
    assert_eq!(
        names,
        vec![
            "park_and_restore",
            "seal_only_at_boundaries",
            "quantum_isolation",
            "quantum_same_boundaries",
            "quantum_resume_fit",
            "segments_exact_bits",
            "stage_path_bits",
            "feed_rule_bits"
        ],
        "{suite}"
    );
}
