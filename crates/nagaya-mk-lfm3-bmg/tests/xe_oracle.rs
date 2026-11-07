//! The gates of the backbone on a B70 card against transformers in f32 on the CPU (spec 8.3, 11.11 "The LFM3
//! oracle"): the residual after each layer of one short prompt, and the top-64 logits and the logsumexp of five
//! prompts, for both models.
//!
//! Each gap prints. A stage asserts `common::STAGE_TOL` and a logit `common::LOGIT_TOL`, the tolerances that the card
//! run of 2026-11-06 measured (spec 11.11, "The tolerances of the LFM3 gates").
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_oracle -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use nagaya_abi::Engine;

/// The feed block of the gates, the default of the engine (engine-abi D11).
const BLOCK: usize = 512;

#[test]
fn the_residual_after_each_layer_follows_transformers() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        let file = format!("{}.layers.safetensors", which.label());
        if common::golden_absent(&file) {
            continue;
        }
        let golden = common::golden_tensors(&file);
        let ids = golden["ids"].ids();
        let mut engine = common::open(which, 1, BLOCK);
        let layers = 24;
        let mut worst = 0.0f64;
        let embed = engine.hidden(0, &ids, 0, false).unwrap();
        let gap = common::rel(&embed, &golden["embed"].f32());
        println!("{which:?} embed: relative error {gap:.3e}");
        assert!(
            gap == 0.0,
            "{which:?}: the embedding rows are BF16 rows in f32, so they match the bits: {gap}"
        );
        for l in 0..layers {
            let got = engine.hidden(0, &ids, l + 1, false).unwrap();
            let want = golden[&format!("layer{l}")].f32();
            let gap = common::rel(&got, &want);
            println!(
                "{which:?} layer {l:2}: relative error {gap:.3e}, largest difference {:.3e}",
                common::max_abs(&got, &want)
            );
            worst = worst.max(gap);
            assert!(gap < common::STAGE_TOL, "{which:?} layer {l}: relative error {gap}");
        }
        let norm = engine.hidden(0, &ids, layers, true).unwrap();
        let gap = common::rel(&norm, &golden["final_norm"].f32());
        println!("{which:?} final norm: relative error {gap:.3e}; worst layer {worst:.3e}");
        assert!(gap < common::STAGE_TOL, "{which:?} final norm: relative error {gap}");
        // The head of the plain prompt against the whole row of the oracle, on the first 125,033 ids.
        let logits = common::feed_split(&mut engine, 0, 0, &ids, &[ids.len()]);
        let want = golden["logits_last"].f32();
        let vocab = engine.vocab();
        let gap = common::max_abs(&logits, &want[..vocab]);
        println!("{which:?} last logits: largest difference {gap:.3e} over {vocab} ids");
        assert!(gap < common::LOGIT_TOL, "{which:?}: the logits differ by {gap}");
    }
}

#[test]
fn the_logits_of_five_prompts_follow_transformers() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        let file = format!("{}.oracle.json", which.label());
        if common::golden_absent(&file) {
            continue;
        }
        let oracle = common::golden_json(&file);
        let mut engine = common::open(which, 1, BLOCK);
        assert_eq!(
            engine.identity().tokenizer_hash,
            format!("sha256:{}", oracle["tokenizer_sha256"].as_str().unwrap())
        );
        for prompt in oracle["prompts"].as_array().unwrap() {
            let name = prompt["name"].as_str().unwrap();
            let ids: Vec<u32> = prompt["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            engine.slots().unwrap().slot_clear(0);
            let split = common::blocks(ids.len(), BLOCK);
            let logits = common::feed_split(&mut engine, 0, 0, &ids, &split);
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
            let lse = common::logsumexp(&logits);
            let lse_want = prompt["logsumexp"].as_f64().unwrap();
            println!(
                "{which:?} {name} ({} ids, split {split:?}): top-64 gap {gap:.3e}, logsumexp {lse:.6} against \
                 {lse_want:.6} of 128,000 rows, top {best} against {}",
                ids.len(),
                top[0]
            );
            assert_eq!(best, top[0], "{which:?} {name}: the top token differs");
            assert!(
                gap < common::LOGIT_TOL,
                "{which:?} {name}: a top-64 logit differs by {gap}"
            );
        }
    }
}
