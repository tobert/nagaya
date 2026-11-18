//! Layer loops of LFM3-1.7B on a B70 card against transformers in f32 on the CPU (spec 11.16, engine-abi "Layer loops
//! of LFM3"). The oracle is `lfm3.loop.safetensors` of `tools/lfm3_golden.sh --stages loop`. The reference runs the
//! whole sequence with no cache at each step, so each run of a layer attends only to its own run. The card keeps the
//! state of each run in the reserve of its slot, so a decode step reads that state.
//!
//! - A loop of count 1 gives the bits of no loop, in the prompt and in 8 decode steps.
//! - Each plan of the oracle gives its logits and its greedy tokens, from one pass of the prompt or from several.
//! - A batch of two rows under one loop gives the bits of each row alone.
//! - The engine refuses a loop past its reserve, and a batch of two loops.
//!
//! Stop `nagaya.service` first, or release the card: the nagaya must be the only client of the card (spec 3.1). Run
//! alone: `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_loop -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::collections::HashMap;

use common::{Tensor, Which};
use nagaya_abi::{Engine, LayerLoop, PassErrorKind, Segment};

const FILE: &str = "lfm3.loop.safetensors";
/// The plans of the oracle (`LOOP_PLANS` of `tools/lfm3_reference.py`).
const PLANS: [(u32, u32, u32); 2] = [(10, 14, 2), (6, 10, 3)];
/// The greedy tokens of each plan (`LOOP_THEN`).
const THEN: usize = 8;

fn golden() -> Option<HashMap<String, Tensor>> {
    if common::absent(Which::Text) || common::golden_absent(FILE) {
        return None;
    }
    Some(common::golden_tensors(FILE))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > row[best] { i } else { best })
}

fn looped(segment: Segment, layer_loop: Option<LayerLoop>) -> Segment {
    Segment { layer_loop, ..segment }
}

/// The logits of the last prompt position and of each greedy step, from an empty slot, with the prompt in passes of
/// the lengths of `split`. Each pass and each step runs under `layer_loop`.
fn run(
    engine: &mut dyn Engine,
    slot: u32,
    ids: &[u32],
    split: &[usize],
    layer_loop: Option<LayerLoop>,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    engine.slots().unwrap().slot_clear(slot);
    let mut at = 0usize;
    let mut rows = Vec::new();
    for (i, n) in split.iter().enumerate() {
        let logits: Vec<u64> = if i + 1 == split.len() {
            vec![(at + n - 1) as u64]
        } else {
            Vec::new()
        };
        let segment = looped(common::segment(slot, at as u64, &ids[at..at + n], &logits), layer_loop);
        let out = engine
            .pass(&[segment], &[])
            .unwrap_or_else(|e| panic!("a pass at {at}: {}", e.message));
        if let Some(row) = out.outputs.first() {
            rows.push(f32s(&row.bytes));
        }
        at += n;
    }
    let mut tokens = Vec::new();
    for _ in 0..THEN {
        let token = argmax(rows.last().unwrap()) as u32;
        tokens.push(token);
        let segment = looped(common::segment(slot, at as u64, &[token], &[at as u64]), layer_loop);
        let out = engine
            .pass(&[segment], &[])
            .unwrap_or_else(|e| panic!("a step at {at}: {}", e.message));
        rows.push(f32s(&out.outputs[0].bytes));
        at += 1;
    }
    (rows, tokens)
}

fn span((start, end, count): (u32, u32, u32)) -> Option<LayerLoop> {
    Some(LayerLoop { start, end, count })
}

#[test]
fn count_1_gives_the_bits_of_no_loop() {
    let Some(golden) = golden() else { return };
    let ids = golden["ids"].ids();
    let mut engine = common::open_looped(Which::Text, 2, 8);
    let (plain, plain_tokens) = run(&mut engine, 0, &ids, &[ids.len()], None);
    let (once, once_tokens) = run(&mut engine, 0, &ids, &[ids.len()], span((10, 14, 1)));
    assert_eq!(plain_tokens, once_tokens);
    for (i, (a, b)) in plain.iter().zip(&once).enumerate() {
        assert_eq!(common::first_other_bits(a, b), None, "row {i}");
    }
    // The reserve changes no bit of a pass with no loop: the plain rows match the oracle as the engine with none does.
    let want = golden["plain.logits"].f32();
    let vocab = engine.vocab();
    let gap = common::max_abs(&plain[0], &want[..vocab]);
    println!("plain: logit gap {gap:.3e}");
    assert!(gap < common::LOGIT_TOL);
}

#[test]
fn each_plan_gives_the_logits_and_the_tokens_of_the_reference() {
    let Some(golden) = golden() else { return };
    let ids = golden["ids"].ids();
    let mut engine = common::open_looped(Which::Text, 2, 8);
    let vocab = engine.vocab();
    let width = golden["plain.logits"].f32().len() / (THEN + 1);
    for plan in PLANS {
        let name = format!("loop{}_{}_{}", plan.0, plan.1, plan.2);
        let want = golden[&format!("{name}.logits")].f32();
        let want_tokens = golden[&format!("{name}.tokens")].ids();
        let half = ids.len() / 2;
        for split in [vec![ids.len()], vec![half, ids.len() - half]] {
            let (rows, tokens) = run(&mut engine, 0, &ids, &split, span(plan));
            let mut worst = 0.0f64;
            for (i, row) in rows.iter().enumerate() {
                worst = worst.max(common::max_abs(row, &want[i * width..i * width + vocab]));
            }
            println!("{name}, split {split:?}: tokens {tokens:?}, worst logit gap {worst:.3e}");
            assert_eq!(tokens, want_tokens, "{name}: the greedy tokens");
            assert!(worst < common::LOGIT_TOL, "{name}: logit gap {worst}");
        }
    }
}

#[test]
fn a_batch_under_one_loop_gives_the_bits_of_each_row_alone() {
    let Some(golden) = golden() else { return };
    let ids = golden["ids"].ids();
    let mut engine = common::open_looped(Which::Text, 2, 8);
    let lp = span((10, 14, 2));
    let n = ids.len() as u64;
    let prompt = |engine: &mut dyn Engine, slot: u32, take: usize| {
        engine.slots().unwrap().slot_clear(slot);
        let segment = looped(common::segment(slot, 0, &ids[..take], &[]), lp);
        engine.pass(&[segment], &[]).unwrap();
    };
    // Each slot alone: a prompt, then one decode step.
    let mut alone = Vec::new();
    for (slot, take) in [(0u32, ids.len()), (1u32, ids.len() - 3)] {
        prompt(&mut engine, slot, take);
        let at = take as u64;
        let out = engine
            .pass(&[looped(common::segment(slot, at, &[ids[0]], &[at]), lp)], &[])
            .unwrap();
        alone.push(f32s(&out.outputs[0].bytes));
    }
    // The same steps as one batch.
    prompt(&mut engine, 0, ids.len());
    prompt(&mut engine, 1, ids.len() - 3);
    let batch = [
        looped(common::segment(0, n, &[ids[0]], &[n]), lp),
        looped(common::segment(1, n - 3, &[ids[0]], &[n - 3]), lp),
    ];
    let out = engine.pass(&batch, &[]).unwrap();
    for i in 0..2 {
        assert_eq!(
            common::first_other_bits(&f32s(&out.outputs[i].bytes), &alone[i]),
            None,
            "row {i}"
        );
    }
    // A batch of two loops is refused before a slot changes. The batch above took one position of each slot.
    let mixed = [
        looped(common::segment(0, n + 1, &[ids[0]], &[n + 1]), lp),
        looped(common::segment(1, n - 2, &[ids[0]], &[n - 2]), span((6, 10, 2))),
    ];
    let error = engine.pass(&mixed, &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{}", error.message);
    assert!(error.message.contains("one loop"), "{}", error.message);
}

#[test]
fn a_loop_past_the_reserve_or_off_the_bounds_is_refused() {
    let Some(golden) = golden() else { return };
    let ids = golden["ids"].ids();
    let mut engine = common::open_looped(Which::Text, 1, 4);
    for (plan, words) in [
        ((6, 14, 2), "each slot holds 4"),
        ((11, 14, 2), "starts and ends"),
        ((10, 14, 5), "1 to 4"),
    ] {
        let segment = looped(common::segment(0, 0, &ids, &[]), span(plan));
        let error = engine.pass(&[segment], &[]).unwrap_err();
        assert_eq!(error.kind, PassErrorKind::Refused, "{plan:?}: {}", error.message);
        assert!(error.message.contains(words), "{plan:?}: {}", error.message);
    }
    let mut plain = common::open_looped(Which::Text, 1, 0);
    assert!(
        !plain
            .limits()
            .capabilities
            .contains(&nagaya_abi::Capability::LayerLoops)
    );
    let error = plain
        .pass(&[looped(common::segment(0, 0, &ids, &[]), span((10, 14, 2)))], &[])
        .unwrap_err();
    assert!(error.message.contains("no reserve"), "{}", error.message);
}
