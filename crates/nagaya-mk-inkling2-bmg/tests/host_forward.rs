//! The host form of the Inkling 2 text model against the `f32` oracle of transformers (`tools/inkling2_golden.sh`).
//!
//! Each gate prints its measured gaps. A test skips when the model or the oracle file is absent.

mod common;

use common::Which;
use nagaya_mk_inkling2_bmg::host::forward::{Context, Model, Trace, run_layers};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use std::sync::OnceLock;

/// The tolerance of a row of a run, a message, a gate or the final norm: a relative L2 error, as `STAGE_TOL` of the
/// LFM3 gates.
const ROW_TOL: f64 = 1e-4;
/// The tolerance of a logit of the prompt "plain": an absolute gap.
const LOGIT_TOL: f64 = 1e-3;

fn model(which: Which) -> &'static Model {
    static MINI: OnceLock<Model> = OnceLock::new();
    static VARLOOP: OnceLock<Model> = OnceLock::new();
    let cell = match which {
        Which::Mini => &MINI,
        Which::Varloop => &VARLOOP,
    };
    cell.get_or_init(|| {
        let path = common::model_weights(which);
        Model::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
    })
}

#[test]
fn each_run_message_and_gate_of_plain_matches_the_oracle() {
    if common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.layers.safetensors") {
        return;
    }
    let oracle = common::golden_tensors("inkling2-mini.layers.safetensors");
    let ids = oracle["ids"].ids();
    let m = model(Which::Mini);
    let mut ctx = Context::new(3);
    let mut trace = Trace::default();
    let started = std::time::Instant::now();
    let logits = m.forward(&mut ctx, &ids, &[ids.len() - 1], Some(&mut trace));
    eprintln!(
        "plain: {} positions in {:.1} s",
        ids.len(),
        started.elapsed().as_secs_f64()
    );
    let layers = run_layers(3);
    assert_eq!(trace.runs.len(), 48);
    let mut worst = 0f64;
    for (r, row) in trace.runs.iter().enumerate() {
        let gap = common::worst_row(row, &oracle[&format!("run{r:02}")].f32(), HIDDEN);
        eprintln!("run {r:02} (layer {:2}): {gap:.3e}", layers[r]);
        worst = worst.max(gap);
    }
    eprintln!("runs: worst {worst:.3e}");
    let mut worst_m = 0f64;
    assert_eq!(trace.messages.len(), 24);
    for (i, row) in trace.messages.iter().enumerate() {
        worst_m = worst_m.max(common::worst_row(row, &oracle[&format!("message{i:02}")].f32(), HIDDEN));
    }
    eprintln!("messages: worst {worst_m:.3e}");
    let mut worst_g = 0f64;
    assert_eq!(trace.gates.len(), 96);
    for (i, row) in trace.gates.iter().enumerate() {
        let name = oracle
            .keys()
            .find(|k| k.starts_with(&format!("gate{i:02}.")))
            .unwrap_or_else(|| panic!("the oracle holds no gate {i}"));
        let want = ["q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"][i % 4];
        assert_eq!(name, &format!("gate{i:02}.{want}"), "the order of the gates");
        worst_g = worst_g.max(common::worst_row(row, &oracle[name].f32(), HIDDEN));
    }
    eprintln!("gates: worst {worst_g:.3e}");
    let norm_gap = common::worst_row(&trace.final_norm, &oracle["final_norm"].f32(), HIDDEN);
    let logit_gap = common::max_abs(&logits[0], &oracle["logits_last"].f32());
    eprintln!("final norm: {norm_gap:.3e}; logits of the last position: max abs {logit_gap:.3e}");
    assert!(worst < ROW_TOL, "a run row differs by {worst:.3e}");
    assert!(worst_m < ROW_TOL, "a message differs by {worst_m:.3e}");
    assert!(worst_g < ROW_TOL, "a gate differs by {worst_g:.3e}");
    assert!(norm_gap < ROW_TOL, "the final norm differs by {norm_gap:.3e}");
    assert!(logit_gap < LOGIT_TOL, "a logit differs by {logit_gap:.3e}");
}

/// The gaps of one prompt of an oracle file: the largest gap of a logit at the top-64 ids of the oracle, and the gap
/// of the logsumexp.
struct Gaps {
    logit: f64,
    logsumexp: f64,
    top1: bool,
    /// The count of the top-64 ids of the oracle that are not in the top 64 of the host form.
    outside: usize,
}

fn gaps(logits: &[f32], prompt: &serde_json::Value) -> Gaps {
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

/// The tolerance of a logit and of the logsumexp of the last position of an oracle prompt, and of a decode step: an
/// absolute gap, as `LOGIT_TOL` of the LFM3 gates. Measured on 2026-11-21 over 30 prompts and 16 steps: the worst
/// logit was 3.3e-5 and the worst logsumexp 7.6e-6, with no growth from 5 to 637 positions. The `f32` factor of an
/// NVFP4 expert in place of the factor of the loader moves a logit by 0.37.
const PROMPT_TOL: f64 = 1e-3;

fn check_prompts(which: Which, file: &str, what: &str) {
    if common::model_absent(which) || common::golden_absent(file) {
        return;
    }
    let oracle = common::golden_json(file);
    let m = model(which);
    let mut worst = (0f64, 0f64);
    for prompt in oracle["prompts"].as_array().unwrap() {
        let ids: Vec<u32> = prompt["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let cycles = prompt.get("cycles").map_or(3, |c| c.as_u64().unwrap() as usize);
        let started = std::time::Instant::now();
        let mut ctx = Context::new(cycles);
        let logits = m.forward(&mut ctx, &ids, &[ids.len() - 1], None);
        let g = gaps(&logits[0], prompt);
        eprintln!(
            "{} {what} {}: {} positions, {cycles} cycles, {:.1} s: logit {:.3e}, logsumexp {:.3e}, top 1 {}, \
             {} of 64 outside",
            which.label(),
            prompt["name"].as_str().unwrap(),
            ids.len(),
            started.elapsed().as_secs_f64(),
            g.logit,
            g.logsumexp,
            if g.top1 { "equal" } else { "DIFFERS" },
            g.outside
        );
        worst = (worst.0.max(g.logit), worst.1.max(g.logsumexp));
        assert!(g.top1, "{} {}: the top id differs", which.label(), prompt["name"]);
        assert!(
            g.logit < PROMPT_TOL,
            "{} {}: a logit differs by {:.3e}",
            which.label(),
            prompt["name"],
            g.logit
        );
        assert!(
            g.logsumexp < PROMPT_TOL,
            "{} {}: the logsumexp differs by {:.3e}",
            which.label(),
            prompt["name"],
            g.logsumexp
        );
    }
    eprintln!(
        "{} {what}: worst logit {:.3e}, worst logsumexp {:.3e}",
        which.label(),
        worst.0,
        worst.1
    );
}

#[test]
fn the_five_oracle_prompts_of_mini_match() {
    check_prompts(Which::Mini, "inkling2-mini.oracle.json", "oracle");
}

#[test]
fn the_five_oracle_prompts_of_the_varloop_model_match() {
    check_prompts(Which::Varloop, "inkling2-varloop.oracle.json", "oracle");
}

#[test]
fn mini_at_1_to_5_cycles_matches() {
    check_prompts(Which::Mini, "inkling2-mini.cycles.json", "cycles");
}

#[test]
fn the_varloop_model_at_1_to_5_cycles_matches() {
    check_prompts(Which::Varloop, "inkling2-varloop.cycles.json", "cycles");
}

/// A prefill of the prompt, then one position at a time with the held state: the logits of each step against the
/// reference with no cache. The oracle holds the logits before each of its 8 tokens: the prefill and 7 steps.
fn check_decode(which: Which) {
    let file = format!("{}.decode.safetensors", which.label());
    if common::model_absent(which) || common::golden_absent(&file) {
        return;
    }
    let oracle = common::golden_tensors(&file);
    let ids = oracle["ids"].ids();
    let tokens = oracle["tokens"].ids();
    let want = oracle["logits"].f32();
    let width = want.len() / tokens.len();
    let m = model(which);
    let mut ctx = Context::new(3);
    let mut rows = m.forward(&mut ctx, &ids, &[ids.len() - 1], None);
    for token in &tokens[..tokens.len() - 1] {
        rows.extend(m.forward(&mut ctx, &[*token], &[0], None));
    }
    assert_eq!(ctx.position, ids.len() + tokens.len() - 1);
    let mut worst = 0f64;
    for (i, row) in rows.iter().enumerate() {
        let gap = common::max_abs(row, &want[i * width..(i + 1) * width]);
        let argmax = (0..row.len()).max_by(|a, b| row[*a].total_cmp(&row[*b])).unwrap();
        eprintln!(
            "{} decode step {i}: max abs {gap:.3e}, argmax {argmax}, token {}",
            which.label(),
            tokens[i]
        );
        assert_eq!(
            argmax as u32,
            tokens[i],
            "{} step {i}: the greedy token differs",
            which.label()
        );
        worst = worst.max(gap);
    }
    eprintln!("{} decode: worst {worst:.3e}", which.label());
    assert!(
        worst < PROMPT_TOL,
        "{} decode: a logit differs by {worst:.3e}",
        which.label()
    );
}

#[test]
fn a_prefill_then_7_steps_of_mini_match_the_reference() {
    check_decode(Which::Mini);
}

#[test]
fn a_prefill_then_7_steps_of_the_varloop_model_match_the_reference() {
    check_decode(Which::Varloop);
}

/// A span longer than the window of a sliding layer, fed whole, against the same positions fed as two spans and one
/// step. The held rings, their trim and the convolution states give the same logits. The reference has no prompt
/// this long, so this gate checks the state rules, not the window against transformers.
#[test]
fn a_span_past_the_window_gives_the_logits_of_two_spans_and_a_step() {
    if common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.oracle.json") {
        return;
    }
    let oracle = common::golden_json("inkling2-mini.oracle.json");
    let longer = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "longer")
        .expect("the oracle holds the prompt longer");
    let base: Vec<u32> = longer["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let ids: Vec<u32> = base.iter().cycle().take(1101).copied().collect();
    let m = model(Which::Mini);
    let mut whole = Context::new(3);
    let want = m.forward(&mut whole, &ids, &[ids.len() - 1], None);
    let mut split = Context::new(3);
    m.forward(&mut split, &ids[..600], &[], None);
    m.forward(&mut split, &ids[600..1100], &[], None);
    let got = m.forward(&mut split, &ids[1100..], &[0], None);
    let gap = common::max_abs(&got[0], &want[0]);
    eprintln!("1,101 positions whole against 600 + 500 + 1: max abs {gap:.3e}");
    for (a, b) in whole.runs.iter().zip(&split.runs) {
        assert_eq!(
            (a.keys.first, a.keys.len()),
            (b.keys.first, b.keys.len()),
            "layer {}",
            a.layer
        );
    }
    assert!(gap < PROMPT_TOL, "the split differs by {gap:.3e}");
}

/// A control item of the host form acts after each run of its layers (engine-abi "The lab of Inkling 2"). An add of
/// strength 0 gives the bits of no item. An ablate of 1 on layer 8 leaves no part along the vector in each of the 3
/// runs of layer 8, and the routes of each run name 8 experts in rising order.
#[test]
fn a_control_item_of_the_host_form_acts_after_each_run_of_its_layers() {
    use nagaya_mk_inkling2_bmg::host::forward::ControlItem;
    use nagaya_mk_inkling2_bmg::host::weights::LAYERS;
    if common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.layers.safetensors") {
        return;
    }
    let oracle = common::golden_tensors("inkling2-mini.layers.safetensors");
    let ids = oracle["ids"].ids();
    let m = model(Which::Mini);
    let run = |control: Vec<ControlItem>| {
        let mut ctx = Context::new(3);
        ctx.control = control;
        let mut trace = Trace::default();
        let logits = m.forward(&mut ctx, &ids, &[ids.len() - 1], Some(&mut trace));
        (logits, trace)
    };
    // The vector of layer 8 is the row after run 8 of "plain" at its last position, so it has a large part there.
    let v: Vec<f32> = oracle["run08"].f32()[(ids.len() - 1) * HIDDEN..].to_vec();
    let mut vector = vec![0f32; LAYERS * HIDDEN];
    vector[8 * HIDDEN..9 * HIDDEN].copy_from_slice(&v);
    let item = |op, s| ControlItem {
        op,
        strengths: std::array::from_fn(|l| (l == 8).then_some(s)),
        vector: vector.clone(),
    };
    let (plain, plain_trace) = run(Vec::new());
    let (zero, zero_trace) = run(vec![item(nagaya_abi::ControlOp::Add, 0.0)]);
    assert_eq!(plain, zero, "an add of strength 0 changes no logit bit");
    assert_eq!(plain_trace.runs, zero_trace.runs);
    let (_, ablated) = run(vec![item(nagaya_abi::ControlOp::Ablate, 1.0)]);
    let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let layers = run_layers(3);
    let mut acted = 0;
    for (r, row) in ablated.runs.iter().enumerate() {
        let worst = row
            .as_chunks::<HIDDEN>()
            .0
            .iter()
            .map(|h| {
                h.iter()
                    .zip(&v)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
                    / norm
            })
            .fold(0f64, |a, b| a.max(b.abs()));
        if layers[r] == 8 {
            acted += 1;
            assert!(worst < 1e-3, "run {r}: a part of {worst} along the vector is left");
        }
        if r < 8 {
            assert_eq!(row, &plain_trace.runs[r], "run {r} is before the first run of layer 8");
        }
    }
    assert_eq!(acted, 3);
    assert_ne!(ablated.runs[8], plain_trace.runs[8]);
    assert_eq!(ablated.routes.len(), 48);
    for (r, routes) in ablated.routes.iter().enumerate() {
        assert_eq!(routes.len(), ids.len(), "run {r}");
        assert!(
            routes.iter().all(|picks| picks.windows(2).all(|w| w[0].0 < w[1].0)),
            "run {r}"
        );
    }
}
