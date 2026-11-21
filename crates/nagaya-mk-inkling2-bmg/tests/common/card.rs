//! The engine of a card test on the first card of `NAGAYA_TEST_CARDS`, and the passes that the gates feed.
#![allow(dead_code)]

use std::path::PathBuf;

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::spec::{Config, DEFAULT_EXTRA_RUNS};

use super::Which;

/// The model folder of `which`. A test that calls it has called `model_absent` before.
pub fn model_dir(which: Which) -> PathBuf {
    super::model_weights(which).parent().unwrap().to_path_buf()
}

/// The engine of `which` on the first card of `NAGAYA_TEST_CARDS`, with slots of the positions `slots`.
pub fn open(which: Which, slots: Vec<usize>, max_block: usize) -> Inkling2Engine {
    let config = Config {
        model_dir: model_dir(which),
        slots,
        max_block,
        extra_runs: DEFAULT_EXTRA_RUNS,
    };
    let started = std::time::Instant::now();
    let engine = Inkling2Engine::open(config, &nagaya_mk_bmg::test_cards::card())
        .unwrap_or_else(|e| panic!("open {which:?}: {e}"));
    eprintln!("{which:?}: the open took {:.1} s", started.elapsed().as_secs_f64());
    engine
}

/// A segment of `tokens` from position `from` of `slot`, with the logits of the positions `logits`.
pub fn segment(slot: u32, from: u64, tokens: &[u32], logits: &[u64]) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..from + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: logits.to_vec(),
        },
        layer_loop: None,
        cycles: None,
    }
}

/// The logit rows of a pass of `tokens` from `from`, for the positions `logits`.
pub fn pass(engine: &mut dyn Engine, slot: u32, from: u64, tokens: &[u32], logits: &[u64]) -> Vec<Vec<f32>> {
    let out = engine
        .pass(&[segment(slot, from, tokens, logits)], &[])
        .unwrap_or_else(|e| panic!("a pass at {from} of {} tokens: {}", tokens.len(), e.message));
    out.outputs
        .iter()
        .map(|o| {
            o.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        })
        .collect()
}

/// Feed `ids` into `slot` from position `from` in passes of the lengths of `split`, and give the logits of the last
/// position.
pub fn feed_split(engine: &mut dyn Engine, slot: u32, from: u64, ids: &[u32], split: &[usize]) -> Vec<f32> {
    assert_eq!(split.iter().sum::<usize>(), ids.len(), "the split covers the ids");
    let mut at = 0usize;
    let mut last = Vec::new();
    for (i, n) in split.iter().enumerate() {
        let pos = from + at as u64;
        let logits: Vec<u64> = if i + 1 == split.len() {
            vec![pos + *n as u64 - 1]
        } else {
            Vec::new()
        };
        if let Some(row) = pass(engine, slot, pos, &ids[at..at + n], &logits).pop() {
            last = row;
        }
        at += n;
    }
    last
}

/// The split of `n` tokens into feed blocks of `block`.
pub fn blocks(n: usize, block: usize) -> Vec<usize> {
    let mut out = vec![block; n / block];
    if !n.is_multiple_of(block) {
        out.push(n % block);
    }
    out
}

/// Empty `slot`.
pub fn clear(engine: &mut dyn Engine, slot: u32) {
    let result = engine.slots().unwrap().slot_clear(slot);
    assert_eq!(result, nagaya_abi::WriteResult::Ok, "a clear of slot {slot}");
}

/// The values whose bits differ between two rows, and the largest gap.
pub fn other_bits(a: &[f32], b: &[f32]) -> (usize, f64) {
    assert_eq!(a.len(), b.len(), "two rows of other lengths");
    let count = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    (count, super::max_abs(a, b))
}

/// The gaps of a logit row from an oracle prompt: the largest gap at the top-64 ids of the oracle, the gap of the
/// logsumexp, whether the argmax agrees, and the count of the oracle's top-64 ids outside the row's top 64.
pub struct Gaps {
    pub logit: f64,
    pub logsumexp: f64,
    pub top1: bool,
    pub outside: usize,
}

pub fn gaps(logits: &[f32], prompt: &serde_json::Value) -> Gaps {
    let ids: Vec<usize> = prompt["top_ids"]
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
    let logit = ids
        .iter()
        .zip(&want)
        .map(|(i, w)| (logits[*i] as f64 - w).abs())
        .fold(0.0, f64::max);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max + logits.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let top: std::collections::HashSet<usize> = order[..ids.len()].iter().copied().collect();
    Gaps {
        logit,
        logsumexp: (lse - prompt["logsumexp"].as_f64().unwrap()).abs(),
        top1: order[0] == ids[0],
        outside: ids.iter().filter(|i| !top.contains(i)).count(),
    }
}
