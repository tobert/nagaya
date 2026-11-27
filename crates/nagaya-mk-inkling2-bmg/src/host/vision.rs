//! The `f32` host form of the image path of Inkling 2: the rules of the processor, the vision tower and the
//! projector (engine-abi "The images of Inkling 2").
//!
//! The reference is transformers `inkling2` at `046bdb8cc2`: `image_processing_inkling2.py`,
//! `processing_inkling2.py`, `Inkling2Vision*` in `modeling_inkling2.py`, and `vision_utils.py`. Each function cites
//! the line that it follows. The oracle is the stage `vision` of `tools/inkling2_reference.py`.
//!
//! # Where each part runs
//!
//! As for LFM3, the family library prepares the rows of an image (spec 11.11), and the engine runs the tower inside
//! the pass. The rules of the processor ([`grid`], [`pixel_values`] and [`patch_rows`]) are the family's
//! (`nagaya_family_inkling2::processor`), and this module uses them. Thus the family and the oracle read one copy. The tower and the
//! projector ([`Vision::tower`]) are the oracle of the device ops.
//!
//! # The names of the checkpoint
//!
//! The engine reads the names of the checkpoint (`model.visual.*`), not the names of transformers. The map, from
//! `conversion_mapping.py` (`inkling2`):
//!
//! | Checkpoint, under `model.visual.` | transformers, under `vision_tower.` | Use |
//! |---|---|---|
//! | `backbone.embeddings.patch_embedding.weight` [1024, 1200] | `embeddings.patch_embedding.weight` | the patch embedding, no bias |
//! | `backbone.embeddings.position_embedding` [1024, 1024] | `embeddings.position_embedding.weight` | the table of 32 x 32 positions |
//! | `backbone.embeddings.class_embedding`, `cls_position_embedding` | the same names | the class row and its position |
//! | `backbone.ln_pre.{weight,bias}` | `pre_layernorm` | the LayerNorm before the layers |
//! | `backbone.layers.L.ln_1`, `ln_2` | `layers.L.layernorm_before`, `layernorm_after` | the two LayerNorms of a layer |
//! | `backbone.layers.L.ls_1.gamma`, `ls_2.gamma` | `layers.L.lambda_1`, `lambda_2` | the layer scales |
//! | `backbone.layers.L.self_attn.{q,k,v,out}_proj` | the same names | the attention, with biases |
//! | `backbone.layers.L.mlp.c_fc`, `c_proj` | `layers.L.mlp.fc1`, `fc2` | the MLP, with biases |
//! | `proj_in.weight` [4096, 4096], `proj_out.weight` [2304, 4096] | `merger.linear_1`, `merger.linear_2` | the projector, no bias |

use super::ops::{self, HIDDEN};
use super::weights::Checkpoint;
// The rules of the processor are the family's (engine-abi "One copy of the rules"), so the oracle gates check them.
pub use nagaya_family_inkling2::processor::{
    Grid, IMAGE_TOKEN, MAX_PATCH_SIDE, MAX_SIDE, MAX_TOKENS, MERGE, MIN_SIDE, PAD_VALUE, PATCH_PX, PATCH_VALUES,
    TOKEN_PX, grid, patch_rows, pixel_values, rgba_to_rgb,
};
use rayon::prelude::*;

/// The width of the tower, its layers, its heads and the width of its MLP.
pub const WIDTH: usize = 1024;
pub const LAYERS: usize = 23;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = WIDTH / HEADS;
pub const MLP: usize = 4 * WIDTH;
/// The side of the table of learned positions (`posemb_grid_size`).
pub const GRID: usize = 32;
/// The inner width of the projector (`projection_hidden_dim`).
pub const PROJ_HIDDEN: usize = 4096;
/// The epsilon of every LayerNorm of the tower (`layer_norm_eps`).
pub const LN_EPS: f64 = 1e-5;
/// The base of the rotary embedding (`rope_theta` of the axial rope).
pub const ROPE_BASE: f32 = 10_000.0;
/// The positions of the rotary table: the class row at 0, then the lines `1..=MAX_PATCH_SIDE` of the patches.
pub const ROPE_POSITIONS: usize = MAX_PATCH_SIDE + 1;

/// The line and the column of patch `n` of an image, in patches. The patches go in the order of the merge blocks
/// (`get_vision_position_ids`, `vision_utils.py:119`).
pub fn patch_position(n: usize, g: Grid) -> (usize, usize) {
    let (token, inner) = (n / (MERGE * MERGE), n % (MERGE * MERGE));
    let (tr, tc) = (token / g.columns, token % g.columns);
    (MERGE * tr + inner / MERGE, MERGE * tc + inner % MERGE)
}

/// The two taps of one axis into the table of `GRID` positions and their bilinear weights, for `index` on an axis
/// of `size` patches. It follows `F.interpolate` with `align_corners=False` and a clamp at the border, in `f32`
/// (`_interpolation_axis_taps_weights`, `vision_utils.py:196`).
pub fn axis_taps(index: usize, size: usize) -> [(usize, f32); 2] {
    let src = (index as f32 + 0.5) * GRID as f32 / size as f32 - 0.5;
    let floor = src.floor();
    [0, 1].map(|offset| {
        let raw = floor as i64 + offset;
        let tap = raw.clamp(0, GRID as i64 - 1) as usize;
        let distance = (src - floor - offset as f32).abs();
        (tap, (1.0 - distance).max(0.0))
    })
}

/// The position embedding of patch `n`: the 4 taps of the table, in the order of the reference, weighted and summed
/// (`Inkling2VisionEmbeddings.forward`, `modeling_inkling2.py:1241`).
fn position_embedding(table: &[f32], n: usize, g: Grid, out: &mut [f32]) {
    let (line, column) = patch_position(n, g);
    let h = axis_taps(line, MERGE * g.rows);
    let w = axis_taps(column, MERGE * g.columns);
    out.fill(0.0);
    for (ht, hw) in h {
        for (wt, ww) in w {
            let weight = hw * ww;
            let row = &table[(ht * GRID + wt) * WIDTH..(ht * GRID + wt + 1) * WIDTH];
            for (o, v) in out.iter_mut().zip(row) {
                *o += v * weight;
            }
        }
    }
}

/// The GELU of `erf`, as the activation `gelu` of transformers. The error function is `f64`.
pub fn gelu(x: f32) -> f32 {
    (0.5 * f64::from(x) * (1.0 + erf(f64::from(x) / std::f64::consts::SQRT_2))) as f32
}

/// The error function in `f64`: its Taylor series below 3, and the continued fraction of `erfc` above. Both give
/// about 1e-13, far below the `f32` rounding of a GELU.
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let e = if z < 3.0 {
        let (mut term, mut sum, mut n) = (z, z, 0u32);
        while term.abs() > 1e-17 * sum.abs() {
            n += 1;
            term *= -z * z / f64::from(n);
            sum += term / f64::from(2 * n + 1);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        let mut f = z;
        for k in (1..=60).rev() {
            f = z + (f64::from(k) / 2.0) / f;
        }
        1.0 - (-z * z).exp() / std::f64::consts::PI.sqrt() / f
    };
    if x < 0.0 { -e } else { e }
}

/// The LayerNorm of each row of `x`, with its mean and variance in `f64`.
fn layer_norm(x: &[f32], w: &Norm) -> Vec<f32> {
    let k = w.weight.len();
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(k).zip(x.par_chunks(k)).for_each(|(o, row)| {
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / k as f64;
        let var = row.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / k as f64;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ((o, v), (g, b)) in o.iter_mut().zip(row).zip(w.weight.iter().zip(&w.bias)) {
            *o = ((f64::from(*v) - mean) * inv) as f32 * g + b;
        }
    });
    out
}

/// `xs W^T + b`: `n x rows` from `n x cols`, with `W` row-major `rows x cols` in `f32`.
fn linear(w: &Linear, xs: &[f32]) -> Vec<f32> {
    let cols = w.cols;
    let rows = w.weight.len() / cols;
    assert_eq!(xs.len() % cols, 0, "linear: {} inputs for rows of {cols}", xs.len());
    let mut out = vec![0f32; xs.len() / cols * rows];
    out.par_chunks_mut(rows).zip(xs.par_chunks(cols)).for_each(|(o, x)| {
        for (r, v) in o.iter_mut().enumerate() {
            *v = ops::dot(&w.weight[r * cols..(r + 1) * cols], x);
            if let Some(b) = &w.bias {
                *v += b[r];
            }
        }
    });
    out
}

/// The angle of channel pair `j` of a head at the position `(line, column)` (`Inkling2VisionRotaryEmbedding`,
/// `modeling_inkling2.py:1028` to 1038). Pairs 0 to 15 turn by the column, and pairs 16 to 31 by the line.
pub fn rope_angle(j: usize, line: usize, column: usize) -> f32 {
    let half = HEAD_DIM / 4;
    let k = j % half;
    let inv_freq = 1.0 / ROPE_BASE.powf((2 * k) as f32 / (HEAD_DIM / 2) as f32);
    let p = if j < half { column } else { line };
    p as f32 * inv_freq
}

/// Turn each channel pair of each head of a row of queries or keys by its angle (`apply_rotary_pos_emb_vision`,
/// `modeling_inkling2.py:1048`).
fn rope(row: &mut [f32], cos: &[f32], sin: &[f32]) {
    for head in row.as_chunks_mut::<HEAD_DIM>().0 {
        for (j, pair) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let [x1, x2] = *pair;
            *pair = [x1 * cos[j] + -x2 * sin[j], x2 * cos[j] + x1 * sin[j]];
        }
    }
}

/// A LayerNorm of the tower.
pub struct Norm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// A linear map of the tower in `f32`, row-major, with its bias if it has one.
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub cols: usize,
}

/// One layer of the tower.
pub struct Layer {
    pub ln_1: Norm,
    pub ln_2: Norm,
    pub ls_1: Vec<f32>,
    pub ls_2: Vec<f32>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub fc: Linear,
    pub proj: Linear,
}

/// The rows of each stage of one tower run, for a gate. Each holds `rows x width` values.
#[derive(Default)]
pub struct VisionTrace {
    /// The patch embedding alone, `4T x 1024`.
    pub patch_embed: Vec<f32>,
    /// The class row, then the patches with their position embedding: `(4T + 1) x 1024`.
    pub embeddings: Vec<f32>,
    /// After `ln_pre`.
    pub pre_layernorm: Vec<f32>,
    /// The rows after each layer.
    pub layers: Vec<Vec<f32>>,
}

/// The weights of the tower and the projector, widened to `f32` at load: about 1.26 GB.
pub struct Vision {
    pub patch_embedding: Linear,
    pub position_embedding: Vec<f32>,
    /// `class_embedding + cls_position_embedding`.
    pub class_row: Vec<f32>,
    pub ln_pre: Norm,
    pub layers: Vec<Layer>,
    pub proj_in: Linear,
    pub proj_out: Linear,
}

fn visual(name: &str) -> String {
    format!("model.visual.{name}")
}

fn norm(ck: &Checkpoint, name: &str) -> Result<Norm, String> {
    Ok(Norm {
        weight: ck.f32s(&visual(&format!("{name}.weight")), &[WIDTH])?,
        bias: ck.f32s(&visual(&format!("{name}.bias")), &[WIDTH])?,
    })
}

fn linear_of(ck: &Checkpoint, name: &str, rows: usize, cols: usize, bias: bool) -> Result<Linear, String> {
    Ok(Linear {
        weight: ck.f32s(&visual(&format!("{name}.weight")), &[rows, cols])?,
        bias: if bias {
            Some(ck.f32s(&visual(&format!("{name}.bias")), &[rows])?)
        } else {
            None
        },
        cols,
    })
}

impl Vision {
    /// Read the 422 tensors of `model.visual.` from the checkpoint. A tensor with another dtype or shape is an error.
    pub fn open(ck: &Checkpoint) -> Result<Self, String> {
        let emb = "backbone.embeddings";
        let class = ck.f32s(&visual(&format!("{emb}.class_embedding")), &[WIDTH])?;
        let class_pos = ck.f32s(&visual(&format!("{emb}.cls_position_embedding")), &[WIDTH])?;
        let layers = (0..LAYERS)
            .map(|l| {
                let p = format!("backbone.layers.{l}");
                Ok(Layer {
                    ln_1: norm(ck, &format!("{p}.ln_1"))?,
                    ln_2: norm(ck, &format!("{p}.ln_2"))?,
                    ls_1: ck.f32s(&visual(&format!("{p}.ls_1.gamma")), &[WIDTH])?,
                    ls_2: ck.f32s(&visual(&format!("{p}.ls_2.gamma")), &[WIDTH])?,
                    q: linear_of(ck, &format!("{p}.self_attn.q_proj"), WIDTH, WIDTH, true)?,
                    k: linear_of(ck, &format!("{p}.self_attn.k_proj"), WIDTH, WIDTH, true)?,
                    v: linear_of(ck, &format!("{p}.self_attn.v_proj"), WIDTH, WIDTH, true)?,
                    out: linear_of(ck, &format!("{p}.self_attn.out_proj"), WIDTH, WIDTH, true)?,
                    fc: linear_of(ck, &format!("{p}.mlp.c_fc"), MLP, WIDTH, true)?,
                    proj: linear_of(ck, &format!("{p}.mlp.c_proj"), WIDTH, MLP, true)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            patch_embedding: linear_of(ck, &format!("{emb}.patch_embedding"), WIDTH, PATCH_VALUES, false)?,
            position_embedding: ck.f32s(&visual(&format!("{emb}.position_embedding")), &[GRID * GRID, WIDTH])?,
            class_row: class.iter().zip(&class_pos).map(|(a, b)| a + b).collect(),
            ln_pre: norm(ck, "backbone.ln_pre")?,
            layers,
            proj_in: linear_of(ck, "proj_in", PROJ_HIDDEN, MERGE * MERGE * WIDTH, false)?,
            proj_out: linear_of(ck, "proj_out", HIDDEN, PROJ_HIDDEN, false)?,
        })
    }

    /// The tower over the patch rows of one image (`Inkling2VisionModel.forward`, `modeling_inkling2.py:1563`).
    /// It returns the rows of the last layer, `(4T + 1) x 1024` with the class row first.
    pub fn tower(&self, rows: &[f32], g: Grid, mut trace: Option<&mut VisionTrace>) -> Vec<f32> {
        let n = g.patches();
        assert_eq!(
            rows.len(),
            n * PATCH_VALUES,
            "tower: {} values for {n} patch rows",
            rows.len()
        );
        let patches = linear(&self.patch_embedding, rows);
        let mut x = Vec::with_capacity((n + 1) * WIDTH);
        x.extend_from_slice(&self.class_row);
        let mut pos = vec![0f32; WIDTH];
        for p in 0..n {
            position_embedding(&self.position_embedding, p, g, &mut pos);
            x.extend(patches[p * WIDTH..(p + 1) * WIDTH].iter().zip(&pos).map(|(a, b)| a + b));
        }
        if let Some(tr) = trace.as_deref_mut() {
            tr.patch_embed = patches;
            tr.embeddings = x.clone();
        }
        let mut x = layer_norm(&x, &self.ln_pre);
        if let Some(tr) = trace.as_deref_mut() {
            tr.pre_layernorm = x.clone();
        }
        // The class row sits at (0, 0), and each patch at its line and column plus one (line 1592).
        let pairs = HEAD_DIM / 2;
        let mut cos = vec![0f32; (n + 1) * pairs];
        let mut sin = vec![0f32; (n + 1) * pairs];
        for t in 0..=n {
            let (line, column) = if t == 0 {
                (0, 0)
            } else {
                let (l, c) = patch_position(t - 1, g);
                (l + 1, c + 1)
            };
            for j in 0..pairs {
                let a = rope_angle(j, line, column);
                cos[t * pairs + j] = a.cos();
                sin[t * pairs + j] = a.sin();
            }
        }
        for layer in &self.layers {
            x = self.layer(layer, &x, &cos, &sin);
            if let Some(tr) = trace.as_deref_mut() {
                tr.layers.push(x.clone());
            }
        }
        x
    }

    /// One layer of the tower (`Inkling2VisionEncoderLayer.forward`, `modeling_inkling2.py:1192`).
    fn layer(&self, w: &Layer, x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let t = x.len() / WIDTH;
        let pairs = HEAD_DIM / 2;
        let h = layer_norm(x, &w.ln_1);
        let mut q = linear(&w.q, &h);
        let mut k = linear(&w.k, &h);
        let v = linear(&w.v, &h);
        for (p, (qr, kr)) in q
            .as_chunks_mut::<WIDTH>()
            .0
            .iter_mut()
            .zip(k.as_chunks_mut::<WIDTH>().0)
            .enumerate()
        {
            let (c, s) = (&cos[p * pairs..(p + 1) * pairs], &sin[p * pairs..(p + 1) * pairs]);
            rope(qr, c, s);
            rope(kr, c, s);
        }
        let scale = (HEAD_DIM as f32).powf(-0.5);
        let mut attn = vec![0f32; t * WIDTH];
        attn.par_chunks_mut(WIDTH).enumerate().for_each(|(i, o)| {
            let mut scores = vec![0f32; t];
            for head in 0..HEADS {
                let cols = head * HEAD_DIM..(head + 1) * HEAD_DIM;
                let qi = &q[i * WIDTH..][cols.clone()];
                for (j, s) in scores.iter_mut().enumerate() {
                    *s = ops::dot(qi, &k[j * WIDTH..][cols.clone()]) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let out = &mut o[cols.clone()];
                for (j, s) in scores.iter().enumerate() {
                    let weight = s / sum;
                    for (o, v) in out.iter_mut().zip(&v[j * WIDTH..][cols.clone()]) {
                        *o += weight * v;
                    }
                }
            }
        });
        let attn = linear(&w.out, &attn);
        let r: Vec<f32> = x
            .iter()
            .zip(&attn)
            .enumerate()
            .map(|(i, (x, a))| x + w.ls_1[i % WIDTH] * a)
            .collect();
        let mut m = linear(&w.fc, &layer_norm(&r, &w.ln_2));
        for v in &mut m {
            *v = gelu(*v);
        }
        let m = linear(&w.proj, &m);
        r.iter()
            .zip(&m)
            .enumerate()
            .map(|(i, (r, m))| r + w.ls_2[i % WIDTH] * m)
            .collect()
    }

    /// The projector over the output of the tower: the class row goes, each 4 patch rows of a token join, and
    /// `proj_out(gelu(proj_in(x)))` gives `T x 2304` (`Inkling2VisionMerger`, `modeling_inkling2.py:1253`).
    pub fn project(&self, tower: &[f32]) -> Vec<f32> {
        assert_eq!(
            tower.len() % (MERGE * MERGE * WIDTH),
            WIDTH,
            "project: the rows are not a class row and tokens"
        );
        let mut h = linear(&self.proj_in, &tower[WIDTH..]);
        for v in &mut h {
            *v = gelu(*v);
        }
        linear(&self.proj_out, &h)
    }

    /// The rows that replace the placeholders of one image: the tower, then the projector.
    pub fn image_rows(&self, rows: &[f32], g: Grid, trace: Option<&mut VisionTrace>) -> Vec<f32> {
        self.project(&self.tower(rows, g, trace))
    }
}

/// Copy the projector rows over the input rows of the placeholders, in order (`masked_scatter`,
/// `modeling_inkling2_loop.py:1694`). `x` holds one row of `HIDDEN` values for each id. A count of placeholders that
/// is not the count of rows is an error, so a placeholder never enters the model as its embedding row.
pub fn place_rows(ids: &[u32], x: &mut [f32], rows: &[f32]) -> Result<(), String> {
    assert_eq!(
        x.len(),
        ids.len() * HIDDEN,
        "place_rows: {} values for {} ids",
        x.len(),
        ids.len()
    );
    let placeholders = ids.iter().filter(|id| **id == IMAGE_TOKEN).count();
    if rows.len() != placeholders * HIDDEN {
        return Err(format!(
            "the span holds {placeholders} image placeholders and {} image rows. The two counts must be equal.",
            rows.len() as f64 / HIDDEN as f64
        ));
    }
    let mut next = rows.as_chunks::<HIDDEN>().0.iter();
    for (t, id) in ids.iter().enumerate() {
        if *id == IMAGE_TOKEN {
            x[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(next.next().expect("counted above"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
