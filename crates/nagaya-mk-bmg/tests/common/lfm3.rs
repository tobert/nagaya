//! The inputs, the fixture and the CPU forms of the LFM3 ops (milestone 24 step 5). The host test
//! `lfm3_cpu_forms.rs` checks the CPU forms against the fixture, and the card test `xe_lfm3_ops.rs` checks the
//! kernels against both. `tools/lfm3_op_fixture.py` writes the fixture from plain torch, with the same generator.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` values in [-scale, scale): the top 24 bits of splitmix64 of the seed and the index (`values` of the tool).
pub fn values(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let key = mix(seed.wrapping_mul(GOLDEN));
    (0..n as u64)
        .map(|i| {
            let u = mix(key.wrapping_add((i + 1).wrapping_mul(GOLDEN)));
            ((u >> 40) as f32 / 8_388_608.0 - 1.0) * scale
        })
        .collect()
}

/// The BF16 bits of `x`, rounded to the nearest even.
pub fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

pub fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The arrays of `tests/fixtures/lfm3_ops.bin`, by name, and the metadata of `lfm3_ops.json`.
pub struct Fixture {
    arrays: HashMap<String, Vec<f32>>,
    pub meta: serde_json::Value,
}

impl Fixture {
    pub fn load() -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let index = std::fs::read_to_string(dir.join("lfm3_ops.json"))
            .unwrap_or_else(|e| panic!("cannot read lfm3_ops.json: {e}. Run tools/lfm3_op_fixture.py."));
        let index: serde_json::Value = serde_json::from_str(&index).expect("lfm3_ops.json is JSON");
        let blob = std::fs::read(dir.join("lfm3_ops.bin")).expect("cannot read lfm3_ops.bin");
        let mut arrays = HashMap::new();
        for (name, entry) in index["arrays"].as_object().expect("the index has arrays") {
            assert_eq!(entry["dtype"], "f32", "{name} is not f32");
            let offset = entry["offset"].as_u64().unwrap() as usize;
            let len = entry["len"].as_u64().unwrap() as usize;
            let (words, rest) = blob[offset..offset + len].as_chunks::<4>();
            assert!(rest.is_empty(), "{name} is not a whole number of floats");
            let floats = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
            arrays.insert(name.clone(), floats);
        }
        Self {
            arrays,
            meta: index["meta"].clone(),
        }
    }

    pub fn get(&self, name: &str) -> &[f32] {
        self.arrays
            .get(name)
            .unwrap_or_else(|| panic!("the fixture has no array {name}"))
    }

    /// The stored rows of an attention case.
    pub fn rows(&self, case: &str) -> Vec<usize> {
        self.meta[format!("attention.{case}.rows")]
            .as_array()
            .unwrap_or_else(|| panic!("the fixture has no rows for attention.{case}"))
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }
}

/// The largest absolute difference, relative to the largest absolute value of `want`.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}

/// The index of the first value whose bits differ, or None.
pub fn first_other_bits(got: &[f32], want: &[f32]) -> Option<usize> {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).position(|(g, w)| g.to_bits() != w.to_bits())
}

// The cases of the fixture; `tools/lfm3_op_fixture.py` has the same.
pub const CONV_D: usize = 48;
pub const CONV_TAPS: usize = 7;
pub const CONV_CASES: [(usize, u64); 3] = [(20, 100), (3, 110), (1, 120)];
pub const ROPE_D: usize = 128;
pub const ROPE_HEADS: usize = 2;
pub const ROPE_TOKENS: usize = 3;
pub const ROPE_P0: u32 = 4093;
pub const ROPE_EPS: f32 = 1e-5;
/// (name, theta, rotated pairs, seed): the full layers rotate half of each head, the sliding layers all of it.
pub const ROPE_CASES: [(&str, f32, usize, u64); 2] = [("full", 1e6, 32, 200), ("sliding", 1e4, 64, 210)];
pub const ATTN_T: usize = 300;
pub const ATTN_H: usize = 16;
pub const ATTN_KV: usize = 4;
pub const ATTN_D: usize = 128;
pub const ATTN_SEED: u64 = 300;
/// (name, window): the sliding layers attend to the last 256 positions, the full layers to all.
pub const ATTN_CASES: [(&str, usize); 2] = [("sliding", 256), ("full", 0)];
pub const GATE_T: usize = 3;
pub const GATE_H: usize = 4;
pub const GATE_D: usize = 32;
pub const GATE_STRIDE: usize = 20;
pub const GATE_OFFSET: usize = 3;
pub const GATE_SEED: u64 = 400;

/// The inputs of a short convolution case: the BF16 weights `[d][taps]`, the in_proj rows `[t][3d]` and the state
/// `[taps - 1][d]`.
pub fn conv_inputs(t: usize, seed: u64) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let w = values(seed, CONV_D * CONV_TAPS, 0.5).into_iter().map(bf16).collect();
    (
        w,
        values(seed + 1, t * 3 * CONV_D, 1.0),
        values(seed + 2, (CONV_TAPS - 1) * CONV_D, 1.0),
    )
}

/// The gated short convolution as `OP_SHORT_CONV` computes it, to the bit: u = B * x, the window of the state and
/// then u, a fused multiply-add over the taps in order from the oldest input, then y = C * the sum. Returns y `[t][d]`
/// and the new state, the last `taps - 1` values of u.
pub fn short_conv(bcx: &[f32], w: &[u16], state: &[f32], t: usize, d: usize, taps: usize) -> (Vec<f32>, Vec<f32>) {
    let hist = taps - 1;
    assert_eq!(bcx.len(), t * 3 * d);
    assert_eq!(state.len(), hist * d);
    let u = |s: isize, c: usize| -> f32 {
        if s < 0 {
            state[(hist as isize + s) as usize * d + c]
        } else {
            let row = &bcx[s as usize * 3 * d..][..3 * d];
            row[c] * row[2 * d + c]
        }
    };
    let mut y = vec![0.0f32; t * d];
    for p in 0..t {
        for c in 0..d {
            let mut acc = 0.0f32;
            for j in 0..taps {
                acc = from_bf16(w[c * taps + j]).mul_add(u(p as isize - hist as isize + j as isize, c), acc);
            }
            y[p * d + c] = bcx[p * 3 * d + d + c] * acc;
        }
    }
    let mut next = vec![0.0f32; hist * d];
    for j in 0..hist {
        for c in 0..d {
            next[j * d + c] = u(t as isize - hist as isize + j as isize, c);
        }
    }
    (y, next)
}

/// The norm and the partial rotation of head rows: `x` is `[tokens][heads][d]`, the first token at `p0`. Pairs
/// (i, i + d / 2) with i < `pairs` turn by pos * theta^(-2i / d); the rest stay as they are. Each value is then
/// multiplied by `scale`.
#[allow(clippy::too_many_arguments)]
pub fn norm_rope(
    x: &[f32],
    w: &[u16],
    heads: usize,
    d: usize,
    p0: u32,
    theta: f32,
    pairs: usize,
    eps: f32,
    scale: f32,
) -> Vec<f32> {
    let half = d / 2;
    let mut out = Vec::with_capacity(x.len());
    for (r, row) in x.chunks_exact(d).enumerate() {
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / d as f32 + eps).sqrt();
        let n: Vec<f32> = (0..d).map(|i| row[i] * inv * from_bf16(w[i])).collect();
        let pos = (p0 as usize + r / heads) as f32;
        let mut y = n.clone();
        for i in 0..pairs {
            let ang = pos * (1.0 / theta.powf((2 * i) as f32 / d as f32));
            let (s, c) = ang.sin_cos();
            y[i] = n[i] * c - n[i + half] * s;
            y[i + half] = n[i + half] * c + n[i] * s;
        }
        out.extend(y.into_iter().map(|v| v * scale));
    }
    out
}

/// The inputs of a rotation case: the rows `[tokens][heads][d]` and the BF16 norm weights near 1.
pub fn rope_inputs(seed: u64) -> (Vec<f32>, Vec<u16>) {
    (
        values(seed, ROPE_TOKENS * ROPE_HEADS * ROPE_D, 1.0),
        values(seed + 1, ROPE_D, 1.0)
            .into_iter()
            .map(|v| bf16(v + 1.0))
            .collect(),
    )
}

/// The inputs of the attention cases: q `[t][h][d]`, k and v `[t][kv][d]`.
pub fn attn_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        values(ATTN_SEED, ATTN_T * ATTN_H * ATTN_D, 1.0),
        values(ATTN_SEED + 1, ATTN_T * ATTN_KV * ATTN_D, 1.0),
        values(ATTN_SEED + 2, ATTN_T * ATTN_KV * ATTN_D, 1.0),
    )
}

/// Causal attention of the query row of token `t` and head `h` with scale 1, over the keys `t + 1 - window ..= t`
/// (`window` 0: every key from 0). `k` and `v` are `[keys][kv][d]`.
#[allow(clippy::too_many_arguments)]
pub fn attend_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    h: usize,
    heads: usize,
    kv: usize,
    d: usize,
    window: usize,
) -> Vec<f32> {
    let g = h / (heads / kv);
    let lo = if window > 0 { (t + 1).saturating_sub(window) } else { 0 };
    let qr = &q[(t * heads + h) * d..][..d];
    let scores: Vec<f64> = (lo..=t)
        .map(|s| {
            let kr = &k[(s * kv + g) * d..][..d];
            qr.iter().zip(kr).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
        })
        .collect();
    let max = scores.iter().fold(f64::NEG_INFINITY, |m, s| m.max(*s));
    let p: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let z: f64 = p.iter().sum();
    (0..d)
        .map(|j| {
            ((lo..=t)
                .zip(&p)
                .map(|(s, p)| p * v[(s * kv + g) * d + j] as f64)
                .sum::<f64>()
                / z) as f32
        })
        .collect()
}

/// The inputs of the gate case: the attention rows `[t][h][d]` and the rows of gates `[t][stride]`.
pub fn gate_inputs() -> (Vec<f32>, Vec<f32>) {
    (
        values(GATE_SEED, GATE_T * GATE_H * GATE_D, 1.0),
        values(GATE_SEED + 1, GATE_T * GATE_STRIDE, 4.0),
    )
}

/// The gate of the heads: y[t, h, j] = a[t, h, j] * sigmoid(g[t * stride + offset + h]).
pub fn head_gate(a: &[f32], g: &[f32], heads: usize, d: usize, stride: usize, offset: usize) -> Vec<f32> {
    a.iter()
        .enumerate()
        .map(|(i, v)| {
            let (t, h) = (i / (heads * d), (i / d) % heads);
            v * (1.0 / (1.0 + (-g[t * stride + offset + h]).exp()))
        })
        .collect()
}

// The control cases: rows of the residual width of LFM3, and the strength of each operation.
pub const CONTROL_K: usize = 2048;
pub const CONTROL_ROWS: usize = 5;
pub const CONTROL_SEED: u64 = 500;
pub const CONTROL_CASES: [(nagaya_mk_bmg::protocol::ControlOp, f32); 4] = [
    (nagaya_mk_bmg::protocol::ControlOp::Add, 4.0),
    (nagaya_mk_bmg::protocol::ControlOp::Ablate, 1.0),
    (nagaya_mk_bmg::protocol::ControlOp::Ablate, 0.5),
    (nagaya_mk_bmg::protocol::ControlOp::Clamp, 8.0),
];

/// The rows of the block case: more than twice `MEGAKERNEL_GROUPS` (32), so each work-group takes several rows.
pub const CONTROL_BLOCK: usize = 70;

/// The inputs of the control cases: the rows `[n][k]`, of the size of a residual, and the vector `[k]`.
pub fn control_inputs() -> (Vec<f32>, Vec<f32>) {
    control_rows(CONTROL_ROWS)
}

/// `n` rows of the residual width and the vector of the control cases.
pub fn control_rows(n: usize) -> (Vec<f32>, Vec<f32>) {
    (
        values(CONTROL_SEED, n * CONTROL_K, 30.0),
        values(CONTROL_SEED + 1, CONTROL_K, 0.1),
    )
}

/// One control item on rows of `k` values, in f64 (engine-abi "Control vectors"). A vector of zeros keeps each row.
pub fn control(h: &[f32], v: &[f32], k: usize, op: nagaya_mk_bmg::protocol::ControlOp, s: f32) -> Vec<f32> {
    use nagaya_mk_bmg::protocol::ControlOp;
    let vv: f64 = v.iter().map(|x| *x as f64 * *x as f64).sum();
    h.chunks(k)
        .flat_map(|row| {
            if vv == 0.0 {
                return row.to_vec();
            }
            let along = row.iter().zip(v).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / vv.sqrt();
            let coef = match op {
                ControlOp::Add => s as f64,
                ControlOp::Ablate => -(s as f64) * along / vv.sqrt(),
                ControlOp::Clamp => (s as f64 - along) / vv.sqrt(),
            };
            row.iter()
                .zip(v)
                .map(|(a, b)| (*a as f64 + coef * *b as f64) as f32)
                .collect()
        })
        .collect()
}

/// The part of each row along `v`: `h . v / |v|`, in f64.
pub fn along(h: &[f32], v: &[f32], k: usize) -> Vec<f64> {
    let norm = v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    h.chunks(k)
        .map(|row| row.iter().zip(v).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / norm)
        .collect()
}
