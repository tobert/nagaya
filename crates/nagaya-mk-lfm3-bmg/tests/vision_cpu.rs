//! The vision path of `LFM3-Omni-2B` in plain Rust on the CPU, against transformers in f32 (spec 11.11, "The LFM3
//! oracle"). Each stage starts from the oracle's input of that stage, so a gap shows in the stage that has it:
//!
//! - the patch embedding and its norm, from `pixel_values` to `patches`;
//! - the 14 layers of the tower, from `patches` to `tower`;
//! - the merge of 2 by 2 patches and the projector, from `tower` to `rows`.
//!
//! The forms of the new ops are `nagaya-mk-bmg/tests/common/lfm3_vision.rs`, which the card tests of the ops compare
//! with the kernels. This test needs no card. It prints each gap, so the card run of the same stages can be read
//! beside it. It needs the Omni folder and the oracle file `vision.safetensors`, and it skips when either is absent.
//!
//! The checks of `AGENTS.md` run the stages of the small image. For both images and the whole path, run
//! `cargo test --release -p nagaya-mk-lfm3-bmg --test vision_cpu -- --nocapture`.

mod common;

#[path = "../../nagaya-mk-bmg/tests/common/lfm3_vision.rs"]
mod forms;

use common::Which;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView, bf16_to_f32};

/// The bound of a stage with no attention: the forms gave 3.8e-7 for the patch embedding and 5.2e-7 for the projector
/// on 2026-11-06. GELU with tanh in place of erf gave 2.8e-4.
const ROW_BOUND: f64 = 1e-5;

/// The bound of the tower and of the whole path: the forms gave 4.0e-6 and 5.7e-6. A swap of the two axes of the
/// rotation gave 0.70, and the conv in raster order gave 0.86.
const TOWER_BOUND: f64 = 1e-4;

const HIDDEN: usize = 1024;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const EPS: f32 = 1e-5;
const THETA: f32 = 1e6;
const PROJECTOR_EPS: f32 = 1e-6;

/// The BF16 tensor `name` of the view, in f32.
fn tensor(view: &SafeTensorsView<'_>, name: &str) -> Vec<f32> {
    let bytes = view
        .tensor_bytes(name)
        .unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"));
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
        .collect()
}

/// `y += x` for each value.
fn add(y: &mut [f32], x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += x;
    }
}

/// The patch embedding and its norm (modeling_lfm3_vl.py:479-480).
fn embed(view: &SafeTensorsView<'_>, pixels: &[f32]) -> Vec<f32> {
    let p = "model.vision_tower.";
    let x = forms::matmul(pixels, &tensor(view, &format!("{p}patchifier.weight")), 1536);
    forms::rmsnorm(&x, &tensor(view, &format!("{p}embedding_norm.weight")), EPS)
}

/// The 14 layers of the tower over the patches of one frame of `grid_h * grid_w` (modeling_lfm3_vl.py:341-374).
fn tower(view: &SafeTensorsView<'_>, patches: &[f32], grid_h: usize, grid_w: usize, kinds: &[&str]) -> Vec<f32> {
    let frame = grid_h * grid_w;
    let mut x = patches.to_vec();
    for (l, kind) in kinds.iter().enumerate() {
        let w = |name: &str| tensor(view, &format!("model.vision_tower.layers.{l}.{name}"));
        let h = forms::rmsnorm(&x, &w("operator_norm.weight"), EPS);
        let out = if *kind == "conv" {
            let bcx = forms::matmul(&h, &w("conv.in_proj.weight"), HIDDEN);
            let y = forms::vision_conv(&bcx, &w("conv.spatial_conv.weight"), HIDDEN, grid_h, grid_w);
            forms::matmul(&y, &w("conv.out_proj.weight"), HIDDEN)
        } else {
            let q = forms::matmul(&h, &w("self_attn.q_proj.weight"), HIDDEN);
            let k = forms::matmul(&h, &w("self_attn.k_proj.weight"), HIDDEN);
            let v = forms::matmul(&h, &w("self_attn.v_proj.weight"), HIDDEN);
            let q = forms::norm_axial(
                &q,
                &w("self_attn.q_layernorm.weight"),
                HEADS,
                HEAD_DIM,
                grid_h,
                grid_w,
                THETA,
                EPS,
            );
            let k = forms::norm_axial(
                &k,
                &w("self_attn.k_layernorm.weight"),
                KV_HEADS,
                HEAD_DIM,
                grid_h,
                grid_w,
                THETA,
                EPS,
            );
            let scale = (HEAD_DIM as f64).powf(-0.5);
            let y = forms::biattn(&q, &k, &v, HEADS, KV_HEADS, HEAD_DIM, frame, scale);
            forms::matmul(&y, &w("self_attn.out_proj.weight"), HIDDEN)
        };
        add(&mut x, &out);
        let h = forms::rmsnorm(&x, &w("ffn_norm.weight"), EPS);
        let m = forms::swiglu_mlp(
            &h,
            &w("feed_forward.w1.weight"),
            &w("feed_forward.w3.weight"),
            &w("feed_forward.w2.weight"),
            HIDDEN,
        );
        add(&mut x, &m);
    }
    x
}

/// The merge of each block of 2 by 2 patches into one row of 4,096 values and the projector to rows of 2,048
/// (`Lfm3OmniMultiModalProjector`, modeling_lfm3_omni.py:349-371). The rows of a block are next to each other, so the
/// merge is a view.
fn project(view: &SafeTensorsView<'_>, tower: &[f32]) -> Vec<f32> {
    let w = |name: &str| tensor(view, &format!("model.multi_modal_projector.{name}"));
    let ln = forms::layernorm(tower, &w("layer_norm.weight"), &w("layer_norm.bias"), PROJECTOR_EPS);
    let mut a = forms::matmul(&ln, &w("linear_1.weight"), 4 * HIDDEN);
    let b1 = w("linear_1.bias");
    for row in a.chunks_exact_mut(b1.len()) {
        for (v, b) in row.iter_mut().zip(&b1) {
            *v = forms::gelu(*v + b);
        }
    }
    let mut rows = forms::matmul(&a, &w("linear_2.weight"), 4 * HIDDEN);
    let b2 = w("linear_2.bias");
    for row in rows.chunks_exact_mut(b2.len()) {
        add(row, &b2);
    }
    rows
}

#[test]
fn the_merge_block_order_is_the_order_of_the_reference_positions() {
    // get_vision_position_ids for a grid of 4 by 6: the blocks go across, and each block is 2 by 2.
    let want = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (0, 4),
        (0, 5),
        (1, 4),
        (1, 5),
    ];
    for (r, at) in want.iter().enumerate() {
        assert_eq!(forms::patch_at(r, 6), *at, "row {r}");
        assert_eq!(forms::row_of(at.0, at.1, 6), r);
    }
    assert_eq!(forms::patch_at(12, 6), (2, 0));
    // The inverse frequencies of one axis: 1 / 1e6^(2j / 32) for j < 16 (compute_axial_rope_parameters).
    let inv = forms::axial_inv_freq(HEAD_DIM, THETA);
    assert_eq!(inv.len(), 16);
    assert_eq!(inv[0], 1.0);
    assert!((inv[8] as f64 - 1e-3).abs() < 1e-9, "{}", inv[8]);
}

#[test]
fn the_gelu_of_the_projector_uses_erf() {
    // Values of 0.5 x (1 + erf(x / sqrt 2)) from Python's math.erf.
    for (x, want) in [
        (-3.0f32, -0.004_049_698),
        (-1.0, -0.15865525),
        (0.5, 0.34573123),
        (2.0, 1.9544997),
        (4.5, 4.4999847),
    ] {
        assert!((forms::gelu(x) - want).abs() < 1e-6, "gelu({x}) = {}", forms::gelu(x));
    }
}

#[test]
fn each_stage_of_the_vision_path_follows_transformers() {
    if common::absent(Which::Omni) || common::golden_absent("vision.safetensors") {
        return;
    }
    let dir = common::model_dir(Which::Omni);
    let map = MappedSafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let view = map.view().unwrap();
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let kinds: Vec<&str> = config["vision_config"]["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(kinds.len(), 14);
    let golden = common::golden_tensors("vision.safetensors");
    // A debug build runs the math about 40 times slower: about 30 s on 32 threads for one tower of the small image.
    // So it checks the stages of the small image only, and a release build checks both images and the whole path.
    let images: &[&str] = if cfg!(debug_assertions) {
        &["small"]
    } else {
        &["small", "mid"]
    };
    for name in images {
        let get = |what: &str| golden[&format!("{name}.{what}")].f32();
        let grid = golden[&format!("{name}.grid_thw")].ids();
        let (t, gh, gw) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
        assert_eq!(t, 1, "an image has one frame");
        let start = std::time::Instant::now();

        let patches = embed(&view, &get("pixel_values"));
        let gap = common::rel(&patches, &get("patches"));
        println!("{name} ({gh} by {gw} patches) patch embedding: relative error {gap:.3e}");
        assert!(gap < ROW_BOUND, "{name}: the patch embedding differs by {gap}");

        let out = tower(&view, &get("patches"), gh, gw, &kinds);
        let gap = common::rel(&out, &get("tower"));
        println!(
            "{name} tower: relative error {gap:.3e}, largest difference {:.3e}",
            common::max_abs(&out, &get("tower"))
        );
        assert!(gap < TOWER_BOUND, "{name}: the tower differs by {gap}");

        let rows = project(&view, &get("tower"));
        let gap = common::rel(&rows, &get("rows"));
        println!(
            "{name} projector: relative error {gap:.3e}, largest difference {:.3e}",
            common::max_abs(&rows, &get("rows"))
        );
        assert!(gap < ROW_BOUND, "{name}: the projector differs by {gap}");

        if cfg!(debug_assertions) {
            println!("{name}: a release build also checks the whole path from the pixels");
            continue;
        }
        // The whole path from the pixels, as the engine runs it.
        let all = project(&view, &tower(&view, &patches, gh, gw, &kinds));
        let gap = common::rel(&all, &get("rows"));
        println!(
            "{name} pixels to rows: relative error {gap:.3e}, in {:.1} s",
            start.elapsed().as_secs_f64()
        );
        assert!(gap < TOWER_BOUND, "{name}: the whole path differs by {gap}");
    }
}
