//! The host form of the image path of Inkling 2 against the `f32` oracle of transformers (`tools/inkling2_golden.sh
//! --stages vision`).
//!
//! Each gate prints its measured gaps. A test skips when the model or the oracle file is absent.

mod common;

use common::Which;
use nagaya_mk_inkling2_bmg::host::forward::{Context, Model};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use nagaya_mk_inkling2_bmg::host::vision::{self, Grid, Vision, VisionTrace, WIDTH};
use std::collections::HashMap;
use std::sync::OnceLock;

const ORACLE: &str = "inkling2-mini.vision.safetensors";
const PROMPT: &str = "inkling2-mini.vision.json";
/// The tolerance of a row of a stage of the tower or of the projector: a relative L2 error, as `ROW_TOL` of the text
/// gates.
const ROW_TOL: f64 = 1e-4;
/// The tolerance of a logit at the top-64 ids of the oracle and of the logsumexp: an absolute gap, as `PROMPT_TOL`
/// of the text gates.
const LOGIT_TOL: f64 = 1e-3;

fn oracle() -> &'static HashMap<String, common::Tensor> {
    static CELL: OnceLock<HashMap<String, common::Tensor>> = OnceLock::new();
    CELL.get_or_init(|| common::golden_tensors(ORACLE))
}

fn model() -> &'static (Model, Vision) {
    static CELL: OnceLock<(Model, Vision)> = OnceLock::new();
    CELL.get_or_init(|| {
        let path = common::model_weights(Which::Mini);
        let m = Model::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let v = Vision::open(&m.weights.checkpoint).unwrap_or_else(|e| panic!("the vision tower: {e}"));
        (m, v)
    })
}

/// The bytes, the width and the height of image `i` of the oracle.
fn image(i: usize) -> (Vec<u8>, usize, usize) {
    let t = &oracle()[&format!("image{i}.rgb")];
    let (h, w) = (t.shape[0], t.shape[1]);
    (t.values.iter().map(|v| *v as u8).collect(), w, h)
}

/// The patch rows and the grid of image `i`, from the rules of the processor.
fn prepared(i: usize) -> (Grid, Vec<f32>) {
    let (rgb, w, h) = image(i);
    let (g, px) = vision::pixel_values(&rgb, w, h).unwrap();
    (g, vision::patch_rows(&px, g))
}

/// The projector rows of image `i` from the host form, computed once.
fn projector_rows(i: usize) -> &'static Vec<f32> {
    static CELLS: [OnceLock<Vec<f32>>; 2] = [OnceLock::new(), OnceLock::new()];
    CELLS[i].get_or_init(|| {
        let (g, rows) = prepared(i);
        let started = std::time::Instant::now();
        let out = model().1.image_rows(&rows, g, None);
        eprintln!(
            "image {i}: {} patches in {:.1} s",
            g.patches(),
            started.elapsed().as_secs_f64()
        );
        out
    })
}

fn absent(needs_model: bool) -> bool {
    (needs_model && common::model_absent(Which::Mini)) || common::golden_absent(ORACLE)
}

#[test]
fn the_rules_of_the_processor_give_the_pixels_and_the_grid_of_the_reference() {
    if absent(false) {
        return;
    }
    for i in 0..2 {
        let (rgb, w, h) = image(i);
        let (g, px) = vision::pixel_values(&rgb, w, h).unwrap();
        let grid: Vec<usize> = oracle()[&format!("image{i}.grid")]
            .values
            .iter()
            .map(|v| *v as usize)
            .collect();
        assert_eq!(grid, [1, g.rows, g.columns], "the grid of image {i}");
        let want = oracle()[&format!("image{i}.pixel_values")].f32();
        assert_eq!(
            oracle()[&format!("image{i}.pixel_values")].shape,
            [g.tokens(), 40, 40, 3]
        );
        let differ = px.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!(
            "image {i}: {w} x {h} px, grid {grid:?}, {differ} of {} pixel values differ in bits",
            px.len()
        );
        assert_eq!(differ, 0, "the pixel values of image {i} differ from the processor");
    }
}

#[test]
fn each_stage_of_the_tower_of_image_0_matches_the_oracle() {
    if absent(true) {
        return;
    }
    let (g, rows) = prepared(0);
    let mut trace = VisionTrace::default();
    let (_m, v) = model();
    let out = v.image_rows(&rows, g, Some(&mut trace));
    let o = oracle();
    let mut gaps = vec![
        (
            "patch_embed",
            common::worst_row(&trace.patch_embed, &o["image0.patch_embed"].f32(), WIDTH),
        ),
        (
            "embeddings",
            common::worst_row(&trace.embeddings, &o["image0.embeddings"].f32(), WIDTH),
        ),
        (
            "pre_layernorm",
            common::worst_row(&trace.pre_layernorm, &o["image0.pre_layernorm"].f32(), WIDTH),
        ),
    ];
    assert_eq!(trace.layers.len(), vision::LAYERS);
    let mut worst_layer = 0f64;
    for (l, row) in trace.layers.iter().enumerate() {
        let gap = common::worst_row(row, &o[&format!("image0.layer{l:02}")].f32(), WIDTH);
        eprintln!("image 0, layer {l:02}: {gap:.3e}");
        worst_layer = worst_layer.max(gap);
    }
    gaps.push(("layers", worst_layer));
    gaps.push((
        "tower",
        common::worst_row(&trace.layers[vision::LAYERS - 1], &o["image0.tower"].f32(), WIDTH),
    ));
    gaps.push((
        "projector",
        common::worst_row(&out, &o["image0.projector"].f32(), HIDDEN),
    ));
    for (name, gap) in &gaps {
        eprintln!("image 0, {name}: worst row {gap:.3e}");
    }
    for (name, gap) in gaps {
        assert!(gap < ROW_TOL, "image 0: the stage {name} differs by {gap:.3e}");
    }
}

#[test]
fn the_projector_rows_of_image_1_match_the_oracle() {
    if absent(true) {
        return;
    }
    let gap = common::worst_row(projector_rows(1), &oracle()["image1.projector"].f32(), HIDDEN);
    eprintln!("image 1, projector: worst row {gap:.3e}");
    assert!(gap < ROW_TOL, "image 1: the projector differs by {gap:.3e}");
}

#[test]
fn the_logits_of_a_prompt_with_two_images_match_the_oracle() {
    if absent(true) || common::golden_absent(PROMPT) {
        return;
    }
    let ids = oracle()["prompt.ids"].ids();
    let placeholders = ids.iter().filter(|id| **id == vision::IMAGE_TOKEN).count();
    let counts = [prepared(0).0.tokens(), prepared(1).0.tokens()];
    assert_eq!(placeholders, counts[0] + counts[1], "the placeholders of the prompt");
    let (m, _v) = model();
    let mut rows = projector_rows(0).clone();
    rows.extend_from_slice(projector_rows(1));
    let mut x = m.embed(&ids);
    vision::place_rows(&ids, &mut x, &rows).unwrap();
    let started = std::time::Instant::now();
    let logits = m
        .forward_rows(&mut Context::new(3), x, &[ids.len() - 1], None)
        .remove(0);
    eprintln!(
        "prompt: {} positions in {:.1} s",
        ids.len(),
        started.elapsed().as_secs_f64()
    );
    let prompt = &common::golden_json(PROMPT)["prompts"][0];
    let top_ids: Vec<usize> = prompt["top_ids"]
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
    let gap = top_ids
        .iter()
        .zip(&want)
        .map(|(i, w)| (f64::from(logits[*i]) - w).abs())
        .fold(0.0, f64::max);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max + logits.iter().map(|v| (f64::from(*v) - max).exp()).sum::<f64>().ln();
    let lse_gap = (lse - prompt["logsumexp"].as_f64().unwrap()).abs();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let full = common::max_abs(&logits, &oracle()["prompt.logits_last"].f32());
    eprintln!(
        "prompt: top-64 max abs {gap:.3e}, logsumexp {lse_gap:.3e}, all logits max abs {full:.3e}, top 1 {} (oracle {})",
        order[0], top_ids[0]
    );
    assert_eq!(order[0], top_ids[0], "the top token differs");
    assert!(gap < LOGIT_TOL, "a top-64 logit differs by {gap:.3e}");
    assert!(lse_gap < LOGIT_TOL, "the logsumexp differs by {lse_gap:.3e}");
}
