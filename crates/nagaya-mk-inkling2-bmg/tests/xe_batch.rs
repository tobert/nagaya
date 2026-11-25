//! A batch of decode rows of the Inkling 2 engine on a B70 card (engine-abi "A batch of
//! decode rows"). A row of a batch gets the bits of its pass alone, in a batch of 1 to 8 rows from contexts at other
//! positions. A block of 2 to 8 tokens gets the bits of the same tokens decoded one at a time.
//!
//! The test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --release --features xe --test xe_batch --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_abi::{Engine, Segment};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::spec::DEFAULT_MAX_BLOCK;

/// The prefix of each slot: other lengths, so the rows of a batch are at other positions. The last slot is longer
/// than its ring of 2,048 rows, so its sliding runs wrap.
const PREFIXES: [usize; 8] = [5, 37, 300, 512, 640, 1_000, 1_500, 2_200];
/// The slots: seven of 2,048 positions, and one of 4,096 for the prefix that wraps the rings.
const SLOTS: [usize; 8] = [2_048, 2_048, 2_048, 2_048, 2_048, 2_048, 2_048, 4_096];
/// The decode steps, and the rows of each pass of each step: 8, then 4 and 4, then four pairs, then 3, 1, 2 and 2.
const GROUPS: [&[usize]; 4] = [&[8], &[4, 4], &[2, 2, 2, 2], &[3, 1, 2, 2]];
/// The cycles of each pass. The batch runs at the default of 3, and again at 1.
const CYCLES: [u8; 2] = [3, 1];

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.oracle.json")
}

/// The prompt "longer" of the oracle, 637 tokens of prose, repeated to `n` ids from the offset `skip`.
fn prose(skip: usize, n: usize) -> Vec<u32> {
    let oracle = common::golden_json("inkling2-mini.oracle.json");
    let prompt = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "longer")
        .expect("the oracle holds the prompt longer")
        .clone();
    let ids: Vec<u32> = prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    ids.iter().copied().cycle().skip(skip).take(n).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap() as u32
}

/// A decode row of `slot` at `at` with the token `token`, at `cycles` cycles.
fn row(slot: usize, at: usize, token: u32, cycles: u8) -> Segment {
    let mut segment = card::segment(slot as u32, at as u64, &[token], &[at as u64]);
    segment.cycles = Some(cycles);
    segment
}

/// Empty each slot, and feed its prefix in blocks of 512 at `cycles` cycles. Give the logits of the last position of
/// each slot.
fn feed(engine: &mut Inkling2Engine, cycles: u8) -> Vec<Vec<f32>> {
    let mut last = Vec::new();
    for (slot, n) in PREFIXES.iter().enumerate() {
        card::clear(engine, slot as u32);
        let ids = prose(7 * slot, *n);
        let mut at = 0;
        let mut logits = Vec::new();
        for len in card::blocks(*n, DEFAULT_MAX_BLOCK) {
            let mut segment = card::segment(slot as u32, at as u64, &ids[at..at + len], &[(at + len - 1) as u64]);
            segment.cycles = Some(cycles);
            let out = engine
                .pass(&[segment], &[])
                .unwrap_or_else(|e| panic!("slot {slot}: a block at {at}: {}", e.message));
            logits = card::floats(&out.outputs[0].bytes);
            at += len;
        }
        last.push(logits);
    }
    last
}

/// The logits of each row of one pass of `segments`.
fn run(engine: &mut Inkling2Engine, segments: &[Segment]) -> Vec<Vec<f32>> {
    let out = engine
        .pass(segments, &[])
        .unwrap_or_else(|e| panic!("a pass of {} rows: {}", segments.len(), e.message));
    assert_eq!(out.outputs.len(), segments.len(), "one logit row for each segment");
    out.outputs.iter().map(|o| card::floats(&o.bytes)).collect()
}

/// Each row of a batch of 1 to 8 rows gets the bits of its pass alone, at each step, at 3 cycles and at 1. The
/// steps of the batches choose the tokens, and the passes alone feed the same tokens from the same prefixes.
#[test]
fn a_row_of_a_batch_gets_the_bits_of_its_pass_alone() {
    if absent() {
        return;
    }
    let mut engine = card::open(Which::Mini, SLOTS.to_vec(), DEFAULT_MAX_BLOCK);
    for cycles in CYCLES {
        // The batches. Each group of a step is one pass of the next rows in slot order.
        let prefix = feed(&mut engine, cycles);
        let mut logits = prefix.clone();
        let mut tokens: Vec<Vec<u32>> = Vec::new();
        let mut batched: Vec<Vec<Vec<f32>>> = Vec::new();
        for (step, groups) in GROUPS.iter().enumerate() {
            let next: Vec<u32> = logits.iter().map(|l| argmax(l)).collect();
            let mut out = Vec::new();
            let mut slot = 0;
            for n in *groups {
                let segments: Vec<Segment> = (slot..slot + n)
                    .map(|s| row(s, PREFIXES[s] + step, next[s], cycles))
                    .collect();
                out.extend(run(&mut engine, &segments));
                slot += n;
            }
            assert_eq!(slot, PREFIXES.len(), "the groups of step {step} cover each slot");
            tokens.push(next);
            batched.push(out.clone());
            logits = out;
        }
        // The passes alone, from the same prefixes, with the same tokens.
        let alone_prefix = feed(&mut engine, cycles);
        let mut moved = Vec::new();
        for (step, (next, rows)) in tokens.iter().zip(&batched).enumerate() {
            for slot in 0..PREFIXES.len() {
                let alone = run(&mut engine, &[row(slot, PREFIXES[slot] + step, next[slot], cycles)]).remove(0);
                let (count, gap) = card::other_bits(&rows[slot], &alone);
                if count > 0 {
                    moved.push(format!("step {step} slot {slot}: {count} values, gap {gap:.3e}"));
                }
            }
        }
        eprintln!(
            "{cycles} cycles: {} steps of 8 rows in groups {GROUPS:?}; {} rows of other bits than alone",
            GROUPS.len(),
            moved.len()
        );
        assert!(
            alone_prefix.iter().zip(&prefix).all(|(a, b)| bits(a) == bits(b)),
            "a second feed of the same prefixes gave other bits"
        );
        assert!(
            moved.is_empty(),
            "{cycles} cycles: rows of a batch gave other bits than their pass alone: {moved:?}"
        );
    }
}

/// A block of 2 to 8 tokens gets the bits of the same tokens decoded one at a time, at each position of the block
///. A block of 9 tokens or more takes other paths of the matmuls and the experts.
#[test]
fn a_block_of_up_to_8_tokens_gets_the_bits_of_its_decode_steps() {
    if absent() {
        return;
    }
    let mut engine = card::open(Which::Mini, vec![2_048], DEFAULT_MAX_BLOCK);
    let ids = prose(0, 700);
    let (prefix, tail) = ids.split_at(640);
    for n in [2usize, 5, 8] {
        let feed_prefix = |engine: &mut Inkling2Engine| {
            card::clear(engine, 0);
            card::feed_split(engine, 0, 0, prefix, &card::blocks(prefix.len(), DEFAULT_MAX_BLOCK));
        };
        feed_prefix(&mut engine);
        let rows: Vec<u64> = (640..640 + n as u64).collect();
        let block = card::pass(&mut engine, 0, 640, &tail[..n], &rows);
        feed_prefix(&mut engine);
        for (t, want) in block.iter().enumerate() {
            let step = card::pass(&mut engine, 0, 640 + t as u64, &tail[t..t + 1], &[640 + t as u64]).remove(0);
            let (count, gap) = card::other_bits(&step, want);
            eprintln!("a block of {n}, position {t}: {count} values of other bits, gap {gap:.3e}");
            assert_eq!(
                count, 0,
                "a block of {n}: the decode step of position {t} differs by {gap:.3e}"
            );
        }
    }
}
