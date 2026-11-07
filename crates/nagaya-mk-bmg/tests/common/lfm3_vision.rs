//! The CPU forms of the ops of the LFM3 vision tower (milestone 24 step 7): the norm and the axial rotation of the
//! head rows, the gated depthwise convolution over the patch grid, and attention in both directions. The card test
//! `xe_lfm3_vision_ops.rs` checks the kernels against these forms. The engine test `vision_cpu.rs` of
//! `nagaya-mk-lfm3-bmg` runs the whole tower with them against transformers, so a wrong form fails there.
//!
//! The reference is `modeling_lfm3_vl.py` of `LFM3-Omni-2B` at `21d75123`. The rows of an image are in the order of
//! the merge blocks of 2 by 2 patches (`image_processing_lfm3_vl.py:186`).
#![allow(dead_code)]

/// The side of a merge block, in patches (`spatial_merge_size`).
pub const MERGE: usize = 2;

/// The patch of row `r` of a frame of `grid_w` patches across, as (down, across). Row `r` is patch `r % 4` of merge
/// block `r / 4`, and the blocks go across first (`get_vision_position_ids`, transformers `vision_utils.py:113-120`).
pub fn patch_at(r: usize, grid_w: usize) -> (usize, usize) {
    let (block, within) = (r / (MERGE * MERGE), r % (MERGE * MERGE));
    let blocks_across = grid_w / MERGE;
    (
        (block / blocks_across) * MERGE + within / MERGE,
        (block % blocks_across) * MERGE + within % MERGE,
    )
}

/// The row of the patch (down, across) of a frame of `grid_w` patches across: the inverse of [`patch_at`].
pub fn row_of(down: usize, across: usize, grid_w: usize) -> usize {
    let block = (down / MERGE) * (grid_w / MERGE) + across / MERGE;
    block * MERGE * MERGE + (down % MERGE) * MERGE + across % MERGE
}

/// The RMS norm of `x` with BF16-valued weights `w` (in f32), for rows of `w.len()` values.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let k = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(k) {
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / k as f32 + eps).sqrt();
        out.extend(row.iter().zip(w).map(|(v, w)| v * inv * w));
    }
    out
}

/// The inverse frequencies of one axis: `1 / theta^(2j / (d / 2))` for `j < d / 4` (`compute_axial_rope_parameters`).
pub fn axial_inv_freq(d: usize, theta: f32) -> Vec<f32> {
    let spatial = (d / 2) as f32;
    (0..d / 4).map(|j| 1.0 / theta.powf((2 * j) as f32 / spatial)).collect()
}

/// The norm and the axial rotation of head rows, as `OP_VISION_ROPE` computes them. `x` is `[rows][heads][d]` with
/// the rows of frames of `grid_h * grid_w` patches in merge-block order. Each head row is normalized with the weights
/// `w` of `d` values. The pair (i, i + d / 2) turns by the down coordinate times `inv[i]` for `i < d / 4`, and by the
/// across coordinate times `inv[i - d / 4]` for the rest (`Lfm3VlVisionRotaryEmbedding`, modeling_lfm3_vl.py:85-131).
#[allow(clippy::too_many_arguments)]
pub fn norm_axial(
    x: &[f32],
    w: &[f32],
    heads: usize,
    d: usize,
    grid_h: usize,
    grid_w: usize,
    theta: f32,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(w.len(), d);
    let (half, quarter) = (d / 2, d / 4);
    let inv = axial_inv_freq(d, theta);
    let normed = rmsnorm(x, w, eps);
    let mut out = normed.clone();
    for (r, row) in normed.chunks_exact(d).enumerate() {
        let patch = (r / heads) % (grid_h * grid_w);
        let (down, across) = patch_at(patch, grid_w);
        let y = &mut out[r * d..][..d];
        for i in 0..half {
            let ang = if i < quarter {
                down as f32 * inv[i]
            } else {
                across as f32 * inv[i - quarter]
            };
            let (s, c) = ang.sin_cos();
            y[i] = row[i] * c - row[i + half] * s;
            y[i + half] = row[i + half] * c + row[i] * s;
        }
    }
    out
}

/// The gated depthwise 3 by 3 convolution of a vision conv layer, as `OP_VISION_CONV` computes it, to the bit. `bcx`
/// holds the rows of `in_proj` `[rows][3d]` in the chunks B, C and x. `w` holds the BF16-valued taps `[d][9]`, down
/// then across. With u = B * x, y[r, c] = C[r, c] * the sum of the taps against u of the 3 by 3 patches around the
/// patch of row r, with zeros past the edge of its frame. The sum goes down then across, with a fused multiply-add
/// (`Lfm3VlVisionConv2d`, modeling_lfm3_vl.py:300-338).
pub fn vision_conv(bcx: &[f32], w: &[f32], d: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
    let frame = grid_h * grid_w;
    let rows = bcx.len() / (3 * d);
    assert_eq!(rows * 3 * d, bcx.len());
    assert_eq!(rows % frame, 0, "the rows are whole frames");
    assert_eq!(w.len(), d * 9);
    let mut y = vec![0.0f32; rows * d];
    for r in 0..rows {
        let base = r - r % frame;
        let (down, across) = patch_at(r % frame, grid_w);
        for c in 0..d {
            let mut acc = 0.0f32;
            for kh in 0..3 {
                for kw in 0..3 {
                    let (h, x) = (down as isize + kh as isize - 1, across as isize + kw as isize - 1);
                    if h < 0 || x < 0 || h >= grid_h as isize || x >= grid_w as isize {
                        continue;
                    }
                    let at = base + row_of(h as usize, x as usize, grid_w);
                    let row = &bcx[at * 3 * d..][..3 * d];
                    acc = w[c * 9 + kh * 3 + kw].mul_add(row[c] * row[2 * d + c], acc);
                }
            }
            y[r * d + c] = bcx[r * 3 * d + d + c] * acc;
        }
    }
    y
}

/// Attention in both directions of the query row of patch `t` and head `h` over the patches of its frame, in f64,
/// with `scale` on each score. `q` is `[rows][heads][d]`, and `k` and `v` are `[rows][kv][d]`. `frame` is the count
/// of patches of a frame.
#[allow(clippy::too_many_arguments)]
pub fn biattn_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    h: usize,
    heads: usize,
    kv: usize,
    d: usize,
    frame: usize,
    scale: f64,
) -> Vec<f32> {
    let g = h / (heads / kv);
    let (lo, hi) = (t - t % frame, t - t % frame + frame);
    let qr = &q[(t * heads + h) * d..][..d];
    let scores: Vec<f64> = (lo..hi)
        .map(|s| {
            let kr = &k[(s * kv + g) * d..][..d];
            qr.iter().zip(kr).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() * scale
        })
        .collect();
    let max = scores.iter().fold(f64::NEG_INFINITY, |m, s| m.max(*s));
    let p: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let z: f64 = p.iter().sum();
    (0..d)
        .map(|j| {
            ((lo..hi)
                .zip(&p)
                .map(|(s, p)| p * v[(s * kv + g) * d + j] as f64)
                .sum::<f64>()
                / z) as f32
        })
        .collect()
}

/// Attention in both directions of every row and head, frame by frame: `[rows][heads][d]`.
#[allow(clippy::too_many_arguments)]
pub fn biattn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    kv: usize,
    d: usize,
    frame: usize,
    scale: f64,
) -> Vec<f32> {
    let rows = q.len() / (heads * d);
    let mut out = vec![0.0f32; q.len()];
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = rows.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (i, part) in out.chunks_mut(chunk * heads * d).enumerate() {
            scope.spawn(move || {
                for (j, row) in part.chunks_mut(heads * d).enumerate() {
                    let t = i * chunk + j;
                    for h in 0..heads {
                        row[h * d..][..d].copy_from_slice(&biattn_row(q, k, v, t, h, heads, kv, d, frame, scale));
                    }
                }
            });
        }
    });
    out
}

/// `y[t, r] = x[t] . w[r]` for rows of `k` against `m` rows of `w`, with each dot in f64. The rows run on all the
/// threads of the machine, because a tower has about 160 GFLOP for the larger image of the oracle.
pub fn matmul(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
    let (n, m) = (x.len() / k, w.len() / k);
    assert_eq!((n * k, m * k), (x.len(), w.len()));
    let mut out = vec![0.0f32; n * m];
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = n.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (i, part) in out.chunks_mut(chunk * m).enumerate() {
            scope.spawn(move || {
                for (j, row) in part.chunks_mut(m).enumerate() {
                    let xr = &x[(i * chunk + j) * k..][..k];
                    for (r, y) in row.iter_mut().enumerate() {
                        let wr = &w[r * k..][..k];
                        let mut acc = [0.0f64; 4];
                        for (xs, ws) in xr.as_chunks::<4>().0.iter().zip(wr.as_chunks::<4>().0) {
                            for l in 0..4 {
                                acc[l] += xs[l] as f64 * ws[l] as f64;
                            }
                        }
                        *y = (acc[0] + acc[1] + acc[2] + acc[3]) as f32;
                    }
                }
            });
        }
    });
    out
}

/// LayerNorm of rows of `w.len()` values with the biased variance, then the weights and the bias.
pub fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let k = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(k) {
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / k as f64;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / k as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        out.extend(
            row.iter()
                .zip(w.iter().zip(b))
                .map(|(v, (w, b))| ((*v as f64 - mean) * inv) as f32 * w + b),
        );
    }
    out
}

/// GELU with erf, as `nn.GELU()` and the activation `gelu` of transformers.
pub fn gelu(x: f32) -> f32 {
    (0.5 * x as f64 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2))) as f32
}

/// The error function in f64: its Taylor series below 3, and the continued fraction of erfc above. Both give about
/// 1e-13 here, far below the f32 rounding of a GELU.
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let e = if z < 3.0 {
        // erf(z) = 2 / sqrt(pi) * the sum over n of (-1)^n z^(2n+1) / (n! (2n + 1)).
        let (mut term, mut sum, mut n) = (z, z, 0u32);
        while term.abs() > 1e-17 * sum.abs() {
            n += 1;
            term *= -z * z / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(z) = exp(-z^2) / sqrt(pi) / (z + (1/2) / (z + 1 / (z + (3/2) / (z + ...)))), from the far end.
        let mut f = z;
        for k in (1..=60).rev() {
            f = z + (k as f64 / 2.0) / f;
        }
        1.0 - (-z * z).exp() / std::f64::consts::PI.sqrt() / f
    };
    if x < 0.0 { -e } else { e }
}

/// The SwiGLU MLP of a vision layer: `w2(silu(w1 x) * w3 x)`.
pub fn swiglu_mlp(x: &[f32], w1: &[f32], w3: &[f32], w2: &[f32], k: usize) -> Vec<f32> {
    let (g, u) = (matmul(x, w1, k), matmul(x, w3, k));
    let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
    matmul(&a, w2, w1.len() / k)
}
