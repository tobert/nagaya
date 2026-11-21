//! The gates of the Inkling 2 engine on a B70 card against the `f32` oracle of transformers (spec 8.3, milestone 35
//! step 5): the row after each of the 48 runs of the prompt "plain", the top-64 logits and the logsumexp of five
//! prompts, and a prefill with 7 decode steps on the held state, for both models.
//!
//! Each gap prints. A test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --features xe --test xe_oracle -- --test-threads=1
//! --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card::{self, Gaps};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use nagaya_mk_inkling2_bmg::host::weights::LOGITS;
use nagaya_mk_inkling2_bmg::spec::{DEFAULT_MAX_BLOCK, SLOT_TOKENS, runs};

/// The tolerance of a row after a run and of the final norm: a relative L2 error. The card run of 2026-11-21 measured
/// 9.1e-7 at most, so the tolerance is about 5 times that.
const ROW_TOL: f64 = 5e-6;
/// The tolerance of a logit and of a logsumexp: an absolute gap. The card run of 2026-11-21 measured 4.1e-5 at most,
/// over both models, the five prompts and the 8 decode rows.
const LOGIT_TOL: f64 = 2e-4;

/// The rows after each run of "plain", the final norm and the logits of its last position.
fn plain(engine: &mut nagaya_mk_inkling2_bmg::Inkling2Engine) {
    let file = "inkling2-mini.layers.safetensors";
    if common::golden_absent(file) {
        return;
    }
    let oracle = common::golden_tensors(file);
    let ids = oracle["ids"].ids();
    let layers: Vec<usize> = runs(3).iter().map(|r| r.layer).collect();
    let mut worst = 0f64;
    for (r, layer) in layers.iter().enumerate() {
        let got = engine.residual(0, &ids, r + 1, false).unwrap();
        let want = oracle[&format!("run{r:02}")].f32();
        let gap = common::worst_row(&got, &want, HIDDEN);
        eprintln!(
            "plain run {r:02} (layer {layer:2}): worst row {gap:.3e}, largest difference {:.3e}",
            common::max_abs(&got, &want)
        );
        worst = worst.max(gap);
        assert!(gap < ROW_TOL, "run {r}: the worst row differs by {gap:.3e}");
    }
    let norm = engine.residual(0, &ids, layers.len(), true).unwrap();
    let norm_gap = common::worst_row(&norm, &oracle["final_norm"].f32(), HIDDEN);
    let logits = card::feed_split(engine, 0, 0, &ids, &[ids.len()]);
    card::clear(engine, 0);
    let logit_gap = common::max_abs(&logits, &oracle["logits_last"].f32());
    eprintln!("plain: runs worst {worst:.3e}; final norm {norm_gap:.3e}; logits of the last position {logit_gap:.3e}");
    assert!(norm_gap < ROW_TOL, "the final norm differs by {norm_gap:.3e}");
    assert!(
        logit_gap < LOGIT_TOL,
        "a logit of the last position differs by {logit_gap:.3e}"
    );
}

/// The five prompts of the oracle file, each fed in blocks of 512 into slot 0.
fn prompts(which: Which, engine: &mut nagaya_mk_inkling2_bmg::Inkling2Engine) {
    let file = format!("{}.oracle.json", which.label());
    if common::golden_absent(&file) {
        return;
    }
    let oracle = common::golden_json(&file);
    let (mut worst_logit, mut worst_lse) = (0f64, 0f64);
    for prompt in oracle["prompts"].as_array().unwrap() {
        let ids: Vec<u32> = prompt["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let logits = card::feed_split(engine, 0, 0, &ids, &card::blocks(ids.len(), DEFAULT_MAX_BLOCK));
        card::clear(engine, 0);
        assert_eq!(logits.len(), LOGITS);
        let Gaps {
            logit,
            logsumexp,
            top1,
            outside,
        } = card::gaps(&logits, prompt);
        eprintln!(
            "{which:?} {} ({} tokens): top-64 logit {logit:.3e}, logsumexp {logsumexp:.3e}, argmax agrees {top1}, \
             {outside} of the top 64 outside",
            prompt["name"],
            ids.len()
        );
        worst_logit = worst_logit.max(logit);
        worst_lse = worst_lse.max(logsumexp);
        assert!(top1, "{which:?} {}: the argmax differs", prompt["name"]);
    }
    eprintln!("{which:?} prompts: worst top-64 logit {worst_logit:.3e}, worst logsumexp {worst_lse:.3e}");
    assert!(
        worst_logit < LOGIT_TOL,
        "{which:?}: a top-64 logit differs by {worst_logit:.3e}"
    );
    assert!(
        worst_lse < LOGIT_TOL,
        "{which:?}: a logsumexp differs by {worst_lse:.3e}"
    );
}

/// The prefill of "chat" and 7 decode steps of the greedy tokens on the held state.
fn decode(which: Which, engine: &mut nagaya_mk_inkling2_bmg::Inkling2Engine) {
    let file = format!("{}.decode.safetensors", which.label());
    if common::golden_absent(&file) {
        return;
    }
    let oracle = common::golden_tensors(&file);
    let ids = oracle["ids"].ids();
    let tokens = oracle["tokens"].ids();
    let want = oracle["logits"].f32();
    let row = |i: usize| &want[i * LOGITS..(i + 1) * LOGITS];
    let mut got = vec![card::feed_split(engine, 0, 0, &ids, &[ids.len()])];
    let started = std::time::Instant::now();
    for (k, token) in tokens.iter().take(tokens.len() - 1).enumerate() {
        let pos = (ids.len() + k) as u64;
        got.push(card::pass(engine, 0, pos, &[*token], &[pos]).remove(0));
    }
    let ms = started.elapsed().as_secs_f64() * 1e3 / (tokens.len() - 1) as f64;
    card::clear(engine, 0);
    let mut worst = 0f64;
    for (i, g) in got.iter().enumerate() {
        let gap = common::max_abs(g, row(i));
        let argmax = (0..LOGITS).max_by(|a, b| g[*a].total_cmp(&g[*b])).unwrap();
        eprintln!(
            "{which:?} decode step {i}: largest difference {gap:.3e}, argmax {argmax} (the oracle took {})",
            tokens[i]
        );
        assert_eq!(argmax as u32, tokens[i], "{which:?} step {i}: the greedy token differs");
        worst = worst.max(gap);
    }
    eprintln!("{which:?} decode: worst {worst:.3e}; {ms:.1} ms for each decode step at about position 20");
    assert!(
        worst < LOGIT_TOL,
        "{which:?}: a logit of a decode step differs by {worst:.3e}"
    );
}

fn gates(which: Which) {
    if common::model_absent(which) {
        return;
    }
    let mut engine = card::open(which, vec![SLOT_TOKENS; 2], DEFAULT_MAX_BLOCK);
    if which == Which::Mini {
        plain(&mut engine);
    }
    prompts(which, &mut engine);
    decode(which, &mut engine);
}

#[test]
fn mini_follows_the_oracle() {
    gates(Which::Mini);
}

#[test]
fn the_research_model_follows_the_oracle() {
    gates(Which::Varloop);
}
