//! The gates of the lab of Inkling 2 on a B70 card (milestone 35 step 8, engine-abi "The lab of Inkling 2"): the taps
//! against the probe of the engine, the host form and the `f32` oracle of transformers, and control vectors against
//! the host form.
//!
//! - A pass with every tap gives the logit bits of the same pass with none.
//! - `residual`, `fold` and `blocks` give the bits of the probe of the engine, and the rows of the oracle.
//! - `messages` and `gates` give the rows of the oracle, and `routes` the picks of the host form.
//! - An add or an ablate of strength 0 gives the bits of no item. Each op follows the host form within a measured gap.
//! - The times of a decode step with a tap and with a control list, against a plain step.
//!
//! Each gap prints. A test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test --release -p nagaya-mk-inkling2-bmg --features xe --test xe_lab --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::card;
use common::{Tensor, Which};
use nagaya_abi::{ControlItem, ControlOp, Engine, Output, PassErrorKind, Segment, Tap};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::host::forward::{self, Context, Model, Trace};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use nagaya_mk_inkling2_bmg::host::weights::LAYERS;
use nagaya_mk_inkling2_bmg::spec::{DEFAULT_MAX_BLOCK, ROUTE_WEIGHTS, TAPS, runs};

/// The tolerance of a tapped row against the oracle: a relative L2 error, as `ROW_TOL` of `xe_oracle.rs`.
const ROW_TOL: f64 = 5e-6;
/// The tolerance of a steered logit against the host form: an absolute gap. It holds the gap of the plain engine
/// from the host form, about 4e-5 (`xe_oracle.rs`), and the gap of the op.
const STEERED_LOGIT_TOL: f64 = 2e-3;
/// The tolerance of a steered row against the host form: a relative L2 error.
const STEERED_ROW_TOL: f64 = 1e-4;
/// The tolerance of the weight of a pick against the host form: an absolute gap.
const WEIGHT_TOL: f64 = 1e-5;

const FILE: &str = "inkling2-mini.layers.safetensors";

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn golden() -> Option<HashMap<String, Tensor>> {
    if common::model_absent(Which::Mini) || common::golden_absent(FILE) {
        return None;
    }
    Some(common::golden_tensors(FILE))
}

/// The outputs of a pass of `ids` from position 0 of an empty slot 0, with the logits of `rows`, the control items and
/// the taps. It empties the slot after.
fn run(engine: &mut Inkling2Engine, ids: &[u32], rows: &[u64], control: &[ControlItem], taps: &[&str]) -> Vec<Output> {
    card::clear(engine, 0);
    let segment = Segment {
        control: control.to_vec(),
        ..card::segment(0, 0, ids, rows)
    };
    let taps: Vec<Tap> = taps
        .iter()
        .map(|name| Tap {
            name: (*name).to_owned(),
        })
        .collect();
    let out = engine
        .pass(&[segment], &taps)
        .unwrap_or_else(|e| panic!("a pass of {} tokens: {}", ids.len(), e.message));
    card::clear(engine, 0);
    out.outputs
}

fn output<'a>(outputs: &'a [Output], name: &str) -> &'a Output {
    let mut found = outputs.iter().filter(|o| o.name == name);
    let one = found
        .next()
        .unwrap_or_else(|| panic!("the pass gave no output `{name}`"));
    assert!(found.next().is_none(), "the pass gave two outputs `{name}`");
    one
}

/// The logit rows of the outputs, in order.
fn logits(outputs: &[Output]) -> Vec<Vec<f32>> {
    outputs
        .iter()
        .filter(|o| o.name == "logits")
        .map(|o| f32s(&o.bytes))
        .collect()
}

/// Row `p` of an oracle tensor of rows of `width` values.
fn oracle_row(t: &Tensor, p: usize, width: usize) -> Vec<f32> {
    t.f32()[p * width..(p + 1) * width].to_vec()
}

fn host_model() -> Model {
    let path = common::model_weights(Which::Mini);
    Model::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

#[test]
fn each_tap_gives_the_rows_of_the_probe_and_of_the_oracle_and_changes_no_logit() {
    let Some(oracle) = golden() else {
        return;
    };
    let ids = oracle["ids"].ids();
    let last = ids.len() - 1;
    let mut engine = card::open(Which::Mini, vec![4096], DEFAULT_MAX_BLOCK);
    // Two logit rows: the last position, and position 1.
    let rows = [last as u64, 1];
    let plain = run(&mut engine, &ids, &rows, &[], &[]);
    let tapped = run(&mut engine, &ids, &rows, &[], &TAPS);
    assert_eq!(
        logits(&plain),
        logits(&tapped),
        "a pass with every tap changes no logit bit"
    );
    let order: Vec<&str> = tapped
        .iter()
        .map(|o| o.name.as_str())
        .filter(|n| *n != "logits")
        .collect();
    assert_eq!(
        order,
        [
            "residual",
            "fold",
            "blocks",
            "messages",
            "gates",
            "routes",
            ROUTE_WEIGHTS
        ]
    );

    // residual: [2][48][2304], each row the bits of the probe of that many runs.
    let residual = f32s(&output(&tapped, "residual").bytes);
    assert_eq!(residual.len(), 2 * 48 * HIDDEN);
    let tap_row = |i: usize, r: usize| residual[(i * 48 + r) * HIDDEN..(i * 48 + r + 1) * HIDDEN].to_vec();
    let mut worst = 0f64;
    for r in 0..48 {
        let probe = engine.residual(0, &ids, r + 1, false).unwrap();
        for (i, p) in [(0usize, last), (1, 1)] {
            let want = &probe[p * HIDDEN..(p + 1) * HIDDEN];
            assert_eq!(
                card::other_bits(&tap_row(i, r), want).0,
                0,
                "run {r}, logit row {i}: the bits of the probe"
            );
            worst = worst.max(common::worst_row(
                &tap_row(i, r),
                &oracle_row(&oracle[&format!("run{r:02}")], p, HIDDEN),
                HIDDEN,
            ));
        }
    }
    eprintln!("residual: the bits of the probe at 48 runs and 2 rows; the oracle within {worst:.3e}");
    assert!(worst < ROW_TOL, "a tapped row differs from the oracle by {worst:.3e}");

    // fold: the final norm of the probe.
    let fold = f32s(&output(&tapped, "fold").bytes);
    let norm = engine.residual(0, &ids, 48, true).unwrap();
    for (i, p) in [(0usize, last), (1, 1)] {
        let got = &fold[i * HIDDEN..(i + 1) * HIDDEN];
        assert_eq!(
            card::other_bits(got, &norm[p * HIDDEN..(p + 1) * HIDDEN]).0,
            0,
            "fold of logit row {i}"
        );
    }
    let fold_gap = common::worst_row(
        &fold[..HIDDEN],
        &oracle_row(&oracle["final_norm"], last, HIDDEN),
        HIDDEN,
    );
    eprintln!("fold: the bits of the final norm of the probe; the oracle within {fold_gap:.3e}");
    assert!(fold_gap < ROW_TOL);

    // blocks: [2][6][2304]: the normed embedding, then the rows of runs 5, 11, 23, 35 and 41.
    let blocks = f32s(&output(&tapped, "blocks").bytes);
    assert_eq!(blocks.len(), 2 * 6 * HIDDEN);
    let embedded = engine.residual(0, &ids, 0, false).unwrap();
    for (i, p) in [(0usize, last), (1, 1)] {
        let block = |b: usize| blocks[(i * 6 + b) * HIDDEN..(i * 6 + b + 1) * HIDDEN].to_vec();
        assert_eq!(
            card::other_bits(&block(0), &embedded[p * HIDDEN..(p + 1) * HIDDEN]).0,
            0
        );
        for (b, r) in [(1, 5), (2, 11), (3, 23), (4, 35), (5, 41)] {
            assert_eq!(
                card::other_bits(&block(b), &tap_row(i, r)).0,
                0,
                "block {b} of logit row {i}"
            );
        }
    }

    // messages and gates: [2][24][4][2304] against the oracle at the last position.
    let messages = f32s(&output(&tapped, "messages").bytes);
    let gates = f32s(&output(&tapped, "gates").bytes);
    assert_eq!(
        (messages.len(), gates.len()),
        (2 * 24 * 4 * HIDDEN, 2 * 24 * 4 * HIDDEN)
    );
    let (mut worst_m, mut worst_g) = (0f64, 0f64);
    for g in 0..24 {
        let got = &messages[g * 4 * HIDDEN..(g + 1) * 4 * HIDDEN];
        let want = oracle_row(&oracle[&format!("message{g:02}")], last, 4 * HIDDEN);
        worst_m = worst_m.max(common::worst_row(got, &want, HIDDEN));
        for k in 0..4 {
            let name = oracle
                .keys()
                .find(|n| n.starts_with(&format!("gate{:02}.", 4 * g + k)))
                .unwrap();
            let got = &gates[(g * 4 + k) * HIDDEN..(g * 4 + k + 1) * HIDDEN];
            worst_g = worst_g.max(common::worst_row(got, &oracle_row(&oracle[name], last, HIDDEN), HIDDEN));
        }
    }
    eprintln!("messages: the oracle within {worst_m:.3e}; gates: {worst_g:.3e}");
    assert!(worst_m < ROW_TOL && worst_g < ROW_TOL);

    // routes: the picks of the host form at each run, and their weights.
    let picks = u32s(&output(&tapped, "routes").bytes);
    let weights = f32s(&output(&tapped, ROUTE_WEIGHTS).bytes);
    assert_eq!((picks.len(), weights.len()), (2 * 48 * 8, 2 * 48 * 8));
    let host = host_model();
    let mut ctx = Context::new(3);
    let mut trace = Trace::default();
    host.forward(&mut ctx, &ids, &[last], Some(&mut trace));
    let mut worst_w = 0f64;
    for (r, routes) in trace.routes.iter().enumerate() {
        for (i, p) in [(0usize, last), (1, 1)] {
            let want = routes[p];
            let at = (i * 48 + r) * 8;
            let got: Vec<usize> = picks[at..at + 8].iter().map(|e| *e as usize).collect();
            assert_eq!(
                got,
                want.map(|(e, _)| e).to_vec(),
                "the picks of run {r}, logit row {i}"
            );
            for (k, (_, w)) in want.iter().enumerate() {
                worst_w = worst_w.max((f64::from(weights[at + k]) - f64::from(*w)).abs());
            }
        }
    }
    eprintln!("routes: the picks of the host form at 48 runs; the weights within {worst_w:.3e}");
    assert!(worst_w < WEIGHT_TOL);

    // One cycle: 24 runs, 4 block outputs, and no messages.
    let segment = Segment {
        cycles: Some(1),
        ..card::segment(0, 0, &ids, &[last as u64])
    };
    let taps: Vec<Tap> = TAPS.iter().map(|n| Tap { name: (*n).to_owned() }).collect();
    let one = engine.pass(&[segment], &taps).unwrap().outputs;
    card::clear(&mut engine, 0);
    assert_eq!(output(&one, "residual").bytes.len(), 24 * HIDDEN * 4);
    assert_eq!(output(&one, "blocks").bytes.len(), 4 * HIDDEN * 4);
    assert!(output(&one, "messages").bytes.is_empty() && output(&one, "gates").bytes.is_empty());
    let unknown = engine.pass(
        &[card::segment(0, 0, &ids, &[])],
        &[Tap {
            name: "lens".to_owned(),
        }],
    );
    let e = unknown.unwrap_err();
    assert_eq!(e.kind, PassErrorKind::Refused);
    assert!(e.message.contains("no tap `lens`"), "{}", e.message);
}

/// The vector of the gate: at layers 8 and 20, the row after run 8 and after run 44 of "plain" at its last position.
fn vector(oracle: &HashMap<String, Tensor>, last: usize) -> Vec<f32> {
    let mut v = vec![0f32; LAYERS * HIDDEN];
    for (layer, run) in [(8usize, 8usize), (20, 44)] {
        v[layer * HIDDEN..(layer + 1) * HIDDEN].copy_from_slice(&oracle_row(
            &oracle[&format!("run{run:02}")],
            last,
            HIDDEN,
        ));
    }
    v
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
fn control_items_of_strength_0_keep_the_bits_and_each_op_follows_the_host_form() {
    let Some(oracle) = golden() else {
        return;
    };
    let ids = oracle["ids"].ids();
    let last = ids.len() - 1;
    let v = vector(&oracle, last);
    let mut engine = card::open(Which::Mini, vec![4096], DEFAULT_MAX_BLOCK);
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    engine.control().unwrap().control_write(3, &bytes).unwrap();
    let rows = [last as u64];
    let plain = run(&mut engine, &ids, &rows, &[], &["residual"]);
    for op in [ControlOp::Add, ControlOp::Ablate] {
        let zero = run(&mut engine, &ids, &rows, &[item(op, 0.0)], &["residual"]);
        assert_eq!(logits(&zero), logits(&plain), "{op:?} 0: the logit bits of no item");
        assert_eq!(
            output(&zero, "residual").bytes,
            output(&plain, "residual").bytes,
            "{op:?} 0: the rows"
        );
    }
    let host = host_model();
    let layers = forward::run_layers(3);
    for (op, s) in [
        (ControlOp::Add, 0.25f32),
        (ControlOp::Ablate, 1.0),
        (ControlOp::Clamp, 8.0),
    ] {
        let got = run(&mut engine, &ids, &rows, &[item(op, s)], &["residual"]);
        let mut ctx = Context::new(3);
        ctx.control = vec![forward::ControlItem {
            op,
            strengths: [Some(s); LAYERS],
            vector: v.clone(),
        }];
        let mut trace = Trace::default();
        let want = host.forward(&mut ctx, &ids, &[last], Some(&mut trace)).remove(0);
        let logit_gap = common::max_abs(&logits(&got)[0], &want);
        let residual = f32s(&output(&got, "residual").bytes);
        let mut row_gap = 0f64;
        for r in 0..48 {
            let want = &trace.runs[r][last * HIDDEN..(last + 1) * HIDDEN];
            row_gap = row_gap.max(common::worst_row(&residual[r * HIDDEN..(r + 1) * HIDDEN], want, HIDDEN));
        }
        let changed = common::max_abs(&logits(&got)[0], &logits(&plain)[0]);
        eprintln!(
            "{op:?} {s}: logits within {logit_gap:.3e} of the host form, rows within {row_gap:.3e}; the item moves a \
             logit by {changed:.3}"
        );
        assert!(changed > 1e-2, "{op:?} {s} changes the logits");
        assert!(logit_gap < STEERED_LOGIT_TOL && row_gap < STEERED_ROW_TOL);
        if op == ControlOp::Ablate {
            // No part along the vector is left after any of the 3 runs of layer 8.
            let u = &v[8 * HIDDEN..9 * HIDDEN];
            let norm = u.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            for (r, layer) in layers.iter().enumerate().filter(|(_, l)| **l == 8) {
                let h = &residual[r * HIDDEN..(r + 1) * HIDDEN];
                let along = h.iter().zip(u).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() / norm;
                assert!(
                    along.abs() < 1e-2,
                    "run {r} (layer {layer}) keeps {along} along the vector"
                );
            }
        }
    }
    // The table refuses what does not fit, and a pass refuses an item outside it.
    let table = engine.control().unwrap();
    assert!(table.control_write(8, &bytes).unwrap_err().contains("8 entries"));
    assert!(table.control_write(0, &bytes[4..]).unwrap_err().contains("bytes"));
    let outside = ControlItem {
        index: 8,
        ..item(ControlOp::Add, 1.0)
    };
    let segment = Segment {
        control: vec![outside],
        ..card::segment(0, 0, &ids, &rows)
    };
    let e = engine.pass(&[segment], &[]).unwrap_err();
    assert_eq!(e.kind, PassErrorKind::Refused);
    assert!(e.message.contains("table entry 8"), "{}", e.message);
}

/// The time of `steps` decode steps after a prompt of 300 tokens, each with the control items and the taps.
fn decode_ms(engine: &mut Inkling2Engine, control: &[ControlItem], taps: &[&str], steps: usize) -> f64 {
    card::clear(engine, 0);
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 37) % 5000).collect();
    let feed = Segment {
        control: control.to_vec(),
        ..card::segment(0, 0, &prompt, &[299])
    };
    engine.pass(&[feed], &[]).unwrap();
    let taps: Vec<Tap> = taps.iter().map(|n| Tap { name: (*n).to_owned() }).collect();
    let mut total = 0f64;
    for s in 0..steps {
        let pos = 300 + s as u64;
        let step = Segment {
            control: control.to_vec(),
            ..card::segment(0, pos, &[1234], &[pos])
        };
        let started = Instant::now();
        engine.pass(&[step], &taps).unwrap();
        total += started.elapsed().as_secs_f64() * 1e3;
    }
    card::clear(engine, 0);
    total / steps as f64
}

#[test]
fn a_decode_step_with_a_tap_or_a_control_list_costs_its_copies_and_its_ops() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let mut engine = card::open(Which::Mini, vec![4096], DEFAULT_MAX_BLOCK);
    let v: Vec<u8> = (0..LAYERS * HIDDEN)
        .flat_map(|i| (((i % 17) as f32 - 8.0) / 100.0).to_le_bytes())
        .collect();
    engine.control().unwrap().control_write(0, &v).unwrap();
    let steps = 16;
    // The first run warms the card.
    decode_ms(&mut engine, &[], &[], 4);
    let plain = decode_ms(&mut engine, &[], &[], steps);
    let residual = decode_ms(&mut engine, &[], &["residual"], steps);
    let all = decode_ms(&mut engine, &[], &TAPS, steps);
    let list = [ControlItem {
        index: 0,
        ..item(ControlOp::Ablate, 1.0)
    }];
    let steered = decode_ms(&mut engine, &list, &[], steps);
    let launches = engine.launches();
    eprintln!(
        "a decode step: plain {plain:.1} ms, with the tap residual {residual:.1} ms, with every tap {all:.1} ms, \
         with an ablate on 24 layers {steered:.1} ms ({launches} launches)"
    );
    assert_eq!(runs(3).len(), 48);
}
