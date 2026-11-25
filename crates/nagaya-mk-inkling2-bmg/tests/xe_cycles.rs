//! The gates of cycles in a call on a B70 card (engine-abi "Inkling 2"): each model at 1 to 5
//! cycles against the `f32` oracle of transformers, for the prompts "plain" and "chat". The reference reads the count
//! from its config; Mini is post-trained at 3 cycles, so the gate of Mini at another count is the reference, not
//! quality.
//!
//! The gates also check the bits: a segment that names the default count gives the bits of a segment that names
//! none, and the same split at 5 cycles gives the same bits twice. A pass of 5 cycles does not fit one ring, so it
//! runs as several launches.
//!
//! Each gap prints. A test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test --release -p nagaya-mk-inkling2-bmg --features xe --test xe_cycles --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card::{self, Gaps};
use nagaya_abi::{Engine, Segment};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::host::weights::LOGITS;
use nagaya_mk_inkling2_bmg::spec::{DEFAULT_MAX_BLOCK, MAX_EXTRA_RUNS};

/// The tolerance of a top-64 logit and of a logsumexp, as in `xe_oracle.rs`: an absolute gap. The card run of
/// 2026-11-21 at 3 cycles measured 2.0e-5 at most.
const LOGIT_TOL: f64 = 2e-4;

/// The logits of the last position of `ids`, fed from position 0 of slot 0 at `cycles`, in passes of `split`.
fn feed(engine: &mut Inkling2Engine, ids: &[u32], split: &[usize], cycles: Option<u8>) -> Vec<f32> {
    let mut at = 0usize;
    let mut last = Vec::new();
    for (i, n) in split.iter().enumerate() {
        let logits: Vec<u64> = if i + 1 == split.len() {
            vec![(at + n - 1) as u64]
        } else {
            Vec::new()
        };
        let segment = Segment {
            cycles,
            ..card::segment(0, at as u64, &ids[at..at + n], &logits)
        };
        let out = engine
            .pass(&[segment], &[])
            .unwrap_or_else(|e| panic!("a pass at {at} of {n} tokens at {cycles:?} cycles: {}", e.message));
        if let Some(row) = out.outputs.last() {
            last = row
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
        }
        at += n;
    }
    card::clear(engine, 0);
    last
}

fn ids_of(prompt: &serde_json::Value) -> Vec<u32> {
    prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn gates(which: Which) {
    if common::model_absent(which) {
        return;
    }
    let file = format!("{}.cycles.json", which.label());
    if common::golden_absent(&file) {
        return;
    }
    let oracle = common::golden_json(&file);
    // The reserve of 48 runs holds cycles 2 to 5.
    let mut engine = card::open_with(which, vec![2048], DEFAULT_MAX_BLOCK, MAX_EXTRA_RUNS);
    let limit = engine.limits().cycles.expect("the engine declares cycles");
    assert_eq!((limit.min, limit.max, limit.default), (1, 5, 3));
    let (mut worst_logit, mut worst_lse) = (0f64, 0f64);
    for prompt in oracle["prompts"].as_array().unwrap() {
        let ids = ids_of(prompt);
        let cycles = prompt["cycles"].as_u64().unwrap() as u8;
        let logits = feed(&mut engine, &ids, &[ids.len()], Some(cycles));
        assert_eq!(logits.len(), LOGITS);
        let launches = engine.launches();
        let Gaps {
            logit,
            logsumexp,
            top1,
            outside,
        } = card::gaps(&logits, prompt);
        eprintln!(
            "{which:?} {} ({} tokens, {cycles} cycles, {launches} launches): top-64 logit {logit:.3e}, logsumexp \
             {logsumexp:.3e}, argmax agrees {top1}, {outside} of the top 64 outside",
            prompt["name"],
            ids.len()
        );
        assert!(top1, "{which:?} {}: the argmax differs", prompt["name"]);
        assert!(
            logit < LOGIT_TOL && logsumexp < LOGIT_TOL,
            "{which:?} {}: a top-64 logit differs by {logit:.3e}, the logsumexp by {logsumexp:.3e}",
            prompt["name"]
        );
        if cycles == 5 {
            assert!(launches > 1, "{which:?}: a pass of 5 cycles fits one launch");
        }
        worst_logit = worst_logit.max(logit);
        worst_lse = worst_lse.max(logsumexp);
    }
    eprintln!("{which:?} cycles: worst top-64 logit {worst_logit:.3e}, worst logsumexp {worst_lse:.3e}");

    // The bits: the default count by name is no count, and a split at 5 cycles gives the same bits twice.
    let chat = ids_of(&oracle["prompts"][1]);
    let plain = feed(&mut engine, &chat, &[chat.len()], None);
    let named = feed(&mut engine, &chat, &[chat.len()], Some(3));
    let (count, gap) = card::other_bits(&plain, &named);
    eprintln!("{which:?}: 3 cycles by name against no count: {count} values differ, by {gap:.3e} at most");
    assert_eq!(count, 0, "{which:?}: the default count by name changes bits");
    let split = [5, chat.len() - 5];
    let first = feed(&mut engine, &chat, &split, Some(5));
    let again = feed(&mut engine, &chat, &split, Some(5));
    let (count, gap) = card::other_bits(&first, &again);
    eprintln!("{which:?}: the split {split:?} at 5 cycles twice: {count} values differ, by {gap:.3e} at most");
    assert_eq!(count, 0, "{which:?}: the same split at 5 cycles gives other bits");
    // A slot fed at 5 cycles refuses a pass at 3, and it changes nothing.
    let ids = &chat[..5];
    let _ = feed_open(&mut engine, ids, Some(5));
    let next = Segment {
        cycles: Some(3),
        ..card::segment(0, 5, &chat[5..6], &[5])
    };
    let error = engine.pass(&[next], &[]).unwrap_err();
    assert!(error.message.contains("5 cycles"), "{}", error.message);
    assert_eq!(engine.slots().unwrap().slot_high(0), 5);
    card::clear(&mut engine, 0);
}

/// Feed `ids` from position 0 of slot 0 at `cycles`, and keep the slot.
fn feed_open(engine: &mut Inkling2Engine, ids: &[u32], cycles: Option<u8>) -> usize {
    let segment = Segment {
        cycles,
        ..card::segment(0, 0, ids, &[])
    };
    engine.pass(&[segment], &[]).unwrap().outputs.len()
}

#[test]
fn the_research_model_follows_the_oracle_at_each_count() {
    gates(Which::Varloop);
}

#[test]
fn mini_follows_the_oracle_at_each_count() {
    gates(Which::Mini);
}
