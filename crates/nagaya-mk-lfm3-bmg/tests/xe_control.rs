//! The residual tap and the control list of both LFM3 models on a B70 card against transformers in f32 on the CPU
//! (spec 11.12, 13.1, engine-abi "Control vectors"). The oracles are `lfm3.control.safetensors` and
//! `lfm3-omni.control.safetensors` of `tools/lfm3_golden.sh`. Omni's vector acts on its text passes (AGENTS.md
//! milestone 25 step 9).
//!
//! - The gate of the means: the tap gives the residual of each prompt of the two sets, and their means give the vector.
//! - With no item, and with an add or an ablate of strength 0, the logits and the tapped rows are the bits of plain.
//! - Each operation with the oracle's vector gives its rows and logits, and the logits of 8 decode steps.
//! - The table refuses an entry or a vector that does not fit, and a pass refuses an item outside the table.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_control -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::collections::HashMap;

use common::{Tensor, Which};
use nagaya_abi::{ControlItem, ControlOp, Engine, PassErrorKind, Tap};
use nagaya_council::control::vector_of_means;

/// The oracle file of a model.
fn file(which: Which) -> &'static str {
    match which {
        Which::Text => "lfm3.control.safetensors",
        Which::Omni => "lfm3-omni.control.safetensors",
    }
}
const LAYERS: usize = 24;
const WIDTH: usize = 2048;
/// The strengths of the oracle (`tools/lfm3_reference.py`, `CONTROL_STRENGTHS`).
const OPS: [(&str, ControlOp, f32); 3] = [
    ("add", ControlOp::Add, 4.0),
    ("ablate", ControlOp::Ablate, 1.0),
    ("clamp", ControlOp::Clamp, 8.0),
];

fn golden(which: Which) -> Option<HashMap<String, Tensor>> {
    if common::absent(which) || common::golden_absent(file(which)) {
        return None;
    }
    Some(common::golden_tensors(file(which)))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The logits and the tapped rows `[layers][width]` of the last position, after a feed of `ids` into an empty slot 0
/// in passes of the lengths of `split`. Each pass gets the items of `control`.
fn feed(engine: &mut dyn Engine, ids: &[u32], split: &[usize], control: &[ControlItem]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(split.iter().sum::<usize>(), ids.len());
    engine.slots().unwrap().slot_clear(0);
    let taps = [Tap {
        name: "residual".to_owned(),
    }];
    let mut at = 0usize;
    let mut last = (Vec::new(), Vec::new());
    for n in split {
        let end = at + n;
        let logits: Vec<u64> = if end == ids.len() {
            vec![end as u64 - 1]
        } else {
            Vec::new()
        };
        let mut segment = common::segment(0, at as u64, &ids[at..end], &logits);
        segment.control = control.to_vec();
        let out = engine
            .pass(&[segment], &taps)
            .unwrap_or_else(|e| panic!("a pass at {at} of {n} tokens: {}", e.message));
        if end == ids.len() {
            let names: Vec<&str> = out.outputs.iter().map(|o| o.name.as_str()).collect();
            assert_eq!(names, ["logits", "residual"]);
            last = (f32s(&out.outputs[0].bytes), f32s(&out.outputs[1].bytes));
        } else {
            let residual = out.outputs.iter().find(|o| o.name == "residual").unwrap();
            assert!(residual.bytes.is_empty(), "a pass with no logit row taps no row");
        }
        at = end;
    }
    assert_eq!(
        last.1.len(),
        LAYERS * WIDTH,
        "the tap gives each layer of the one logit row"
    );
    last
}

/// The largest relative error of a layer of `got` against `want`, both `[layers][width]`.
fn worst_layer(got: &[f32], want: &[f32]) -> (usize, f64) {
    (0..LAYERS)
        .map(|l| {
            (
                l,
                common::rel(&got[l * WIDTH..(l + 1) * WIDTH], &want[l * WIDTH..(l + 1) * WIDTH]),
            )
        })
        .fold((0, 0.0), |best, next| if next.1 > best.1 { next } else { best })
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).max_by(|a, b| row[*a].total_cmp(&row[*b])).unwrap()
}

fn vector_bytes(golden: &HashMap<String, Tensor>) -> Vec<u8> {
    golden["vector"].f32().iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn refs(set: &[Vec<f32>]) -> Vec<&[f32]> {
    set.iter().map(|r| r.as_slice()).collect()
}

fn item(op: ControlOp, strength: f32) -> ControlItem {
    ControlItem {
        index: 3,
        op,
        strength,
        layers: None,
    }
}

#[test]
fn the_tap_gives_the_rows_of_each_prompt_and_their_means_give_the_vector() {
    for which in Which::BOTH {
        let Some(golden) = golden(which) else { continue };
        println!("{which:?}");
        let mut engine = common::open(which, 1, 512);
        let mut sets: Vec<Vec<Vec<f32>>> = Vec::new();
        for set in ["positive", "negative"] {
            let mut rows = Vec::new();
            for i in 0..4 {
                let ids = golden[&format!("{set}{i}.ids")].ids();
                let (_, got) = feed(&mut engine, &ids, &[ids.len()], &[]);
                let want = golden[&format!("{set}{i}.rows")].f32();
                let (l, gap) = worst_layer(&got, &want);
                println!(
                    "{set}{i} ({} ids): worst layer {l}, relative error {gap:.3e}",
                    ids.len()
                );
                assert!(gap < common::STAGE_TOL, "{set}{i} layer {l}: relative error {gap}");
                rows.push(got);
            }
            sets.push(rows);
        }
        let vector = vector_of_means(&refs(&sets[0]), &refs(&sets[1])).unwrap();
        let want = golden["vector"].f32();
        // A layer of the vector is a difference of two means, and layer 0 cancels: its norm is 1.1e-7 in the oracle, with
        // rows of norm 0.84. So the error of each layer is relative to the norm of the positive mean of that layer.
        let mut worst = (0usize, 0.0f64);
        for l in 0..LAYERS {
            let at = l * WIDTH..(l + 1) * WIDTH;
            let diff: f64 = vector[at.clone()]
                .iter()
                .zip(&want[at.clone()])
                .map(|(g, w)| (*g as f64 - *w as f64).powi(2))
                .sum();
            let mean: f64 = (0..WIDTH)
                .map(|j| (sets[0].iter().map(|r| r[at.start + j] as f64).sum::<f64>() / 4.0).powi(2))
                .sum();
            let gap = (diff / mean).sqrt();
            if gap > worst.1 {
                worst = (l, gap);
            }
        }
        let (l, gap) = worst;
        println!(
            "vector: worst layer {l}, error {gap:.3e} of the norm of the positive mean, largest difference {:.3e}",
            common::max_abs(&vector, &want)
        );
        assert!(
            gap < common::STAGE_TOL,
            "the vector of layer {l}: error {gap} of the positive mean"
        );
        // The same rows give the same bits on each run (spec 11.12).
        let again = vector_of_means(&refs(&sets[0]), &refs(&sets[1])).unwrap();
        assert_eq!(common::first_other_bits(&vector, &again), None);
    }
}

#[test]
fn no_item_and_items_of_strength_0_give_the_bits_of_plain() {
    for which in Which::BOTH {
        let Some(golden) = golden(which) else { continue };
        println!("{which:?}");
        let mut engine = common::open(which, 1, 512);
        engine
            .control()
            .unwrap()
            .control_write(3, &vector_bytes(&golden))
            .unwrap();
        let ids = golden["test.ids"].ids();
        let (logits, rows) = feed(&mut engine, &ids, &[ids.len()], &[]);
        let vocab = engine.vocab();
        let gap = common::max_abs(&logits, &golden["plain.logits"].f32()[..vocab]);
        let (l, rows_gap) = worst_layer(&rows, &golden["plain.rows"].f32());
        println!("plain: logit gap {gap:.3e}; rows worst layer {l}, relative error {rows_gap:.3e}");
        assert!(gap < common::LOGIT_TOL && rows_gap < common::STAGE_TOL);
        for list in [
            vec![item(ControlOp::Add, 0.0)],
            vec![item(ControlOp::Ablate, 0.0)],
            vec![item(ControlOp::Add, -0.0), item(ControlOp::Ablate, 0.0)],
        ] {
            let (l2, r2) = feed(&mut engine, &ids, &[ids.len()], &list);
            assert_eq!(
                common::first_other_bits(&l2, &logits),
                None,
                "{list:?}: the logits differ"
            );
            assert_eq!(common::first_other_bits(&r2, &rows), None, "{list:?}: the rows differ");
        }
        // A clamp of strength 0 is not plain: it removes the concept.
        let (clamped, _) = feed(&mut engine, &ids, &[ids.len()], &[item(ControlOp::Clamp, 0.0)]);
        assert!(common::first_other_bits(&clamped, &logits).is_some());
    }
}

#[test]
fn each_operation_follows_transformers_in_its_rows_its_logits_and_8_decode_steps() {
    for which in Which::BOTH {
        let Some(golden) = golden(which) else { continue };
        println!("{which:?}");
        let mut engine = common::open(which, 1, 512);
        engine
            .control()
            .unwrap()
            .control_write(3, &vector_bytes(&golden))
            .unwrap();
        let ids = golden["test.ids"].ids();
        let vocab = engine.vocab();
        for (name, op, strength) in OPS {
            let list = [item(op, strength)];
            let (logits, rows) = feed(&mut engine, &ids, &[ids.len()], &list);
            let (l, rows_gap) = worst_layer(&rows, &golden[&format!("{name}.rows")].f32());
            let gap = common::max_abs(&logits, &golden[&format!("{name}.logits")].f32()[..vocab]);
            let off = common::max_abs(&logits, &golden["plain.logits"].f32()[..vocab]);
            println!(
                "{name} {strength}: rows worst layer {l}, relative error {rows_gap:.3e}; logit gap {gap:.3e}, and \
                 {off:.3e} from plain"
            );
            assert!(
                rows_gap < common::STAGE_TOL,
                "{name}: layer {l} relative error {rows_gap}"
            );
            assert!(gap < common::LOGIT_TOL, "{name}: a logit differs by {gap}");
            // The greedy decode of 8 tokens: the prompt pass gives the first row, and each decode step gets the same
            // items. The oracle's ids are the argmax of its logits (`greedy` in tools/lfm3_reference.py, with no
            // repetition penalty), so the argmax of each row of the card must be the oracle's id.
            let want: Vec<usize> = golden[&format!("{name}.decode_ids")]
                .ids()
                .iter()
                .map(|v| *v as usize)
                .collect();
            let top_ids = golden[&format!("{name}.decode_top_ids")].ids();
            let top = golden[&format!("{name}.decode_top")].f32();
            let mut row = logits;
            for step in 0..want.len() {
                let best = argmax(&row);
                let raw = top_ids[step * 64] as usize;
                let step_gap = (0..64)
                    .map(|k| (row[top_ids[step * 64 + k] as usize] as f64 - top[step * 64 + k] as f64).abs())
                    .fold(0.0f64, f64::max);
                let margin = top[step * 64] as f64 - top[step * 64 + 1] as f64;
                println!(
                    "{name} step {step}: argmax {best} against {raw}, oracle id {}, top-64 gap {step_gap:.3e}, oracle \
                     margin {margin:.3e}",
                    want[step]
                );
                assert_eq!(
                    want[step], raw,
                    "{name} step {step}: the oracle's id is not the argmax of its own logits"
                );
                assert_eq!(
                    best, want[step],
                    "{name} step {step}: the greedy id differs; the oracle's margin of its top two is {margin:.3e}"
                );
                assert!(
                    step_gap < common::LOGIT_TOL,
                    "{name} step {step}: a top-64 logit differs by {step_gap}"
                );
                if step + 1 == want.len() {
                    break;
                }
                let pos = (ids.len() + step) as u64;
                let mut segment = common::segment(0, pos, &[want[step] as u32], &[pos]);
                segment.control = list.to_vec();
                let out = engine.pass(&[segment], &[]).unwrap();
                row = f32s(&out.outputs[0].bytes);
            }
        }
    }
}

#[test]
fn a_feed_with_an_item_keeps_the_feed_rule() {
    for which in Which::BOTH {
        let Some(golden) = golden(which) else { continue };
        println!("{which:?}");
        let mut engine = common::open(which, 1, 512);
        engine
            .control()
            .unwrap()
            .control_write(3, &vector_bytes(&golden))
            .unwrap();
        let ids = golden["test.ids"].ids();
        let n = ids.len();
        for (name, op, strength) in OPS {
            let list = [item(op, strength)];
            let one = feed(&mut engine, &ids, &[n], &list);
            let again = feed(&mut engine, &ids, &[n], &list);
            assert_eq!(
                common::first_other_bits(&one.0, &again.0),
                None,
                "{name}: one feed twice"
            );
            assert_eq!(
                common::first_other_bits(&one.1, &again.1),
                None,
                "{name}: one feed twice"
            );
            for split in [vec![n / 2, n - n / 2], vec![n - 1, 1]] {
                let (logits, rows) = feed(&mut engine, &ids, &split, &list);
                let twice = feed(&mut engine, &ids, &split, &list);
                assert_eq!(
                    common::first_other_bits(&logits, &twice.0),
                    None,
                    "{name} {split:?}: the split twice"
                );
                let gap = common::max_abs(&logits, &one.0);
                let (l, rows_gap) = worst_layer(&rows, &one.1);
                println!(
                    "{name} split {split:?}: logit gap {gap:.3e} from one feed, rows layer {l} {rows_gap:.3e}, bits {}",
                    if common::first_other_bits(&logits, &one.0).is_none() {
                        "equal"
                    } else {
                        "differ"
                    }
                );
                assert!(gap < common::LOGIT_TOL, "{name} {split:?}: a logit differs by {gap}");
                assert!(
                    rows_gap < common::STAGE_TOL,
                    "{name} {split:?}: layer {l} relative error {rows_gap}"
                );
            }
        }
    }
}

#[test]
fn the_table_refuses_what_does_not_fit_and_a_pass_refuses_an_item_outside_it() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        println!("{which:?}");
        let mut engine = common::open(which, 1, 512);
        let limit = engine.limits().control.unwrap();
        assert_eq!((limit.tables, limit.layers, limit.width), (8, 24, 2048));
        let good = vec![0u8; limit.entry_bytes() as usize];
        let control = engine.control().unwrap();
        control.control_write(7, &good).unwrap();
        assert!(control.control_write(8, &good).unwrap_err().contains("8 entries"));
        assert!(
            control
                .control_write(0, &good[4..])
                .unwrap_err()
                .contains("196608 bytes")
        );
        let mut nan = good.clone();
        nan[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(control.control_write(0, &nan).unwrap_err().contains("value 10"));
        let mut segment = common::segment(0, 0, &[1, 2, 3], &[2]);
        segment.control = vec![ControlItem {
            index: 8,
            op: ControlOp::Add,
            strength: 1.0,
            layers: None,
        }];
        let error = engine.pass(&[segment], &[]).unwrap_err();
        assert_eq!(error.kind, PassErrorKind::Refused);
        assert!(error.message.contains("table entry 8"), "{}", error.message);
        let error = engine
            .pass(
                &[common::segment(0, 0, &[1, 2, 3], &[2])],
                &[Tap {
                    name: "routes".to_owned(),
                }],
            )
            .unwrap_err();
        assert!(error.message.contains("no tap `routes`"), "{}", error.message);
        assert_eq!(
            engine.slots().unwrap().slot_high(0),
            0,
            "a refused pass changes nothing"
        );
    }
}
