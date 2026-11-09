//! The taps of a trace on a B70 card (`docs/lab.md`, "The rules of `trace`", spec 13.1): the tap `residual` and the
//! tap `lens` change no bit of a greedy reply, `along` of a vector after an ablate of 1 is 0, and the lens of the last
//! layer is the head. The oracle is `lfm3.control.safetensors` of `tools/lfm3_golden.sh`.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_trace -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::{Tensor, Which};
use nagaya_abi::{ControlItem, ControlOp, Engine, PassErrorKind, Tap};

const FILE: &str = "lfm3.control.safetensors";
const LAYERS: usize = 24;
const WIDTH: usize = 2048;
/// The greedy steps of the gate of the bits.
const STEPS: usize = 32;

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

fn taps(names: &[&str]) -> Vec<Tap> {
    names
        .iter()
        .map(|name| Tap {
            name: (*name).to_owned(),
        })
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|a, b| row[*a].total_cmp(&row[*b]).then(b.cmp(a)))
        .unwrap()
}

/// One step of a greedy reply: the logit row, and the outputs of the taps when the passes named them.
struct Step {
    logits: Vec<f32>,
    residual: Option<Vec<f32>>,
    lens: Option<Vec<f32>>,
}

/// A greedy reply of `steps` steps after `ids` in an empty slot 0. The prompt is one pass, and each step is a pass of
/// one token. Each pass gets `control`, and the pass of each logit row names `named`.
fn reply(engine: &mut dyn Engine, ids: &[u32], steps: usize, control: &[ControlItem], named: &[&str]) -> Vec<Step> {
    engine.slots().unwrap().slot_clear(0);
    let mut out = Vec::new();
    let mut tokens = ids.to_vec();
    let mut from = 0usize;
    for _ in 0..steps {
        let last = tokens.len() as u64 - 1;
        let mut segment = common::segment(0, from as u64, &tokens[from..], &[last]);
        segment.control = control.to_vec();
        let pass = engine
            .pass(&[segment], &taps(named))
            .unwrap_or_else(|e| panic!("a pass at {from}: {}", e.message));
        let get = |name: &str| pass.outputs.iter().find(|o| o.name == name).map(|o| f32s(&o.bytes));
        let step = Step {
            logits: get("logits").unwrap(),
            residual: get("residual"),
            lens: get("lens"),
        };
        from = tokens.len();
        tokens.push(argmax(&step.logits) as u32);
        out.push(step);
    }
    out
}

/// The unit vector of each layer of the oracle's vector, in `f64`, or `None` for a layer of zeros.
fn units(vector: &[f32]) -> Vec<Option<Vec<f64>>> {
    (0..LAYERS)
        .map(|l| {
            let v = &vector[l * WIDTH..(l + 1) * WIDTH];
            let norm = v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
            (norm > 0.0).then(|| v.iter().map(|x| *x as f64 / norm).collect())
        })
        .collect()
}

fn dot(h: &[f32], u: &[f64]) -> f64 {
    h.iter().zip(u).map(|(a, b)| *a as f64 * b).sum()
}

fn norm(h: &[f32]) -> f64 {
    h.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt()
}

#[test]
fn the_taps_change_no_bit_of_a_greedy_reply_and_the_lens_of_the_last_layer_is_the_head() {
    let Some(golden) = golden() else { return };
    let mut engine = common::open(Which::Text, 1, 512);
    let vector = golden["vector"].f32();
    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.control().unwrap().control_write(3, &bytes).unwrap();
    let units = units(&vector);
    let ids = golden["test.ids"].ids();
    let vocab = engine.vocab();
    let ablate = [ControlItem {
        index: 3,
        op: ControlOp::Ablate,
        strength: 1.0,
        layers: None,
    }];
    for (name, control) in [("plain", &[][..]), ("ablate 1", &ablate[..])] {
        let bare = reply(&mut engine, &ids, STEPS, control, &[]);
        let traced = reply(&mut engine, &ids, STEPS, control, &["residual", "lens"]);
        let lensed = reply(&mut engine, &ids, STEPS, control, &["lens"]);
        let mut worst_lens = 0.0f64;
        let mut worst_along = 0.0f64;
        for (i, ((b, t), l)) in bare.iter().zip(&traced).zip(&lensed).enumerate() {
            assert_eq!(
                common::first_other_bits(&t.logits, &b.logits),
                None,
                "{name} step {i}: the taps change a logit"
            );
            assert_eq!(common::first_other_bits(&l.logits, &b.logits), None, "{name} step {i}");
            assert!(l.residual.is_none(), "the lens alone gives no residual rows");
            let lens = t.lens.as_ref().unwrap();
            assert_eq!(lens.len(), LAYERS * vocab);
            assert_eq!(
                common::first_other_bits(l.lens.as_ref().unwrap(), lens),
                None,
                "{name} step {i}"
            );
            // The lens of the last layer is the head on the same row, by a matmul of 24 rows.
            let last = &lens[(LAYERS - 1) * vocab..];
            assert_eq!(
                argmax(last),
                argmax(&b.logits),
                "{name} step {i}: the lens of layer 23 is not the head"
            );
            worst_lens = worst_lens.max(common::max_abs(last, &b.logits));
            let rows = t.residual.as_ref().unwrap();
            assert_eq!(rows.len(), LAYERS * WIDTH);
            if control.is_empty() {
                continue;
            }
            // After an ablate of 1, each row has no part along the vector, to f32 rounding of the op.
            for (layer, unit) in units.iter().enumerate() {
                let Some(unit) = unit else { continue };
                let h = &rows[layer * WIDTH..(layer + 1) * WIDTH];
                worst_along = worst_along.max(dot(h, unit).abs() / norm(h));
            }
        }
        println!(
            "{name}: {STEPS} steps with the same logit bits; lens of layer 23 against the logits {worst_lens:.3e}; \
             worst |h.v|/|h| {worst_along:.3e}"
        );
        assert!(
            worst_lens < common::LOGIT_TOL,
            "{name}: the lens of layer 23 differs by {worst_lens}"
        );
        assert!(
            worst_along < 1e-5,
            "{name}: along after the ablate is {worst_along} of the row"
        );
    }
    // Step 0 of the plain reply reads the last prompt position, as the oracle's plain rows do.
    let plain = reply(&mut engine, &ids, 1, &[], &["residual"]);
    let rows = plain[0].residual.as_ref().unwrap();
    let want = golden["plain.rows"].f32();
    let mut worst = 0.0f64;
    for (layer, unit) in units.iter().enumerate() {
        let Some(unit) = unit else { continue };
        let w = &want[layer * WIDTH..(layer + 1) * WIDTH];
        let gap = (dot(&rows[layer * WIDTH..(layer + 1) * WIDTH], unit) - dot(w, unit)).abs() / norm(w);
        worst = worst.max(gap);
    }
    println!("plain along at step 0 against the oracle's rows: worst gap {worst:.3e} of the row");
    assert!(
        worst < common::STAGE_TOL,
        "along differs from the oracle by {worst} of the row"
    );
}

#[test]
fn the_lens_reads_one_logit_row() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        let mut engine = common::open(which, 1, 512);
        let error = engine
            .pass(&[common::segment(0, 0, &[1, 2, 3], &[1, 2])], &taps(&["lens"]))
            .unwrap_err();
        assert_eq!(error.kind, PassErrorKind::Refused, "{which:?}");
        assert!(error.message.contains("one logit row"), "{}", error.message);
        assert_eq!(
            engine.slots().unwrap().slot_high(0),
            0,
            "a refused pass changes nothing"
        );
        // Omni's text passes have the lens too, and its last layer is the head.
        let out = reply(&mut engine, &[1, 2, 3, 4, 5], 2, &[], &["lens"]);
        let vocab = engine.vocab();
        for step in &out {
            let last = &step.lens.as_ref().unwrap()[(LAYERS - 1) * vocab..];
            assert_eq!(argmax(last), argmax(&step.logits), "{which:?}");
            assert!(common::max_abs(last, &step.logits) < common::LOGIT_TOL, "{which:?}");
        }
    }
}

/// The time of 200 greedy steps with the tap `residual` alone, and with the lens too. The trace of `docs/lab.md` reads
/// both taps at each step.
#[test]
fn the_time_of_200_steps_with_and_without_the_lens() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 1, 512);
    let ids: Vec<u32> = (1..40).collect();
    for named in [&[][..], &["residual"][..], &["residual", "lens"][..]] {
        reply(&mut engine, &ids, 8, &[], named);
        let started = Instant::now();
        let steps = reply(&mut engine, &ids, 200, &[], named);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "taps {named:?}: 200 steps in {ms:.0} ms, {:.2} ms a step",
            ms / steps.len() as f64
        );
    }
}
