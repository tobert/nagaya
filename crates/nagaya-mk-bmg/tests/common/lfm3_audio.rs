//! The inputs and the CPU forms of the ops of the audio encoder of LFM3-Omni-2B (milestone 24 step 7). The host test
//! `lfm3_audio_cpu_forms.rs` checks the CPU forms against the plain formula in the layout of the reference, and the
//! card test `xe_lfm3_audio_ops.rs` checks the kernels against the CPU forms.
#![allow(dead_code)]

use super::lfm3::{bf16, from_bf16, values};

/// One case of `OP_SUBSAMPLE`: (name, time rows, mel rows, channels in, channels out, silu, seed).
pub type SubsampleCase = (&'static str, usize, usize, usize, usize, bool, u64);

/// The convolutions of the downsampler for a clip of 149 frames (the oracle clip), and small planes of odd sizes.
pub const SUBSAMPLE_CASES: [SubsampleCase; 5] = [
    ("conv_in", 149, 128, 1, 256, true, 600),
    ("conv_mid", 75, 64, 256, 256, false, 610),
    ("conv_end", 38, 32, 256, 256, true, 620),
    ("odd", 5, 3, 16, 16, false, 630),
    ("one", 1, 1, 1, 32, false, 640),
];

/// The inputs of a case: the plane `[n][k][cin]`, the BF16 taps `[m][9]` and the BF16 bias `[m]`.
pub fn subsample_inputs(n: usize, k: usize, cin: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<u16>, Vec<u16>) {
    let x = values(seed, n * k * cin, 1.0);
    let w = values(seed + 1, m * 9, 0.5).into_iter().map(bf16).collect();
    let b = values(seed + 2, m, 0.5).into_iter().map(bf16).collect();
    (x, w, b)
}

/// `OP_SUBSAMPLE` as the kernel computes it: a fused multiply-add over the taps in the order of the mel step `i`, then
/// the time step `j`, from 0, then the bias, then silu. With no silu the result is the bits of the kernel.
#[allow(clippy::too_many_arguments)]
pub fn subsample(x: &[f32], w: &[u16], b: &[u16], n: usize, k: usize, cin: usize, m: usize, silu: bool) -> Vec<f32> {
    assert_eq!(x.len(), n * k * cin);
    assert!(cin == 1 || cin == m);
    let (to_n, ho_n) = (n.div_ceil(2), k.div_ceil(2));
    let mut y = vec![0.0f32; to_n * ho_n * m];
    for to in 0..to_n {
        for ho in 0..ho_n {
            for ch in 0..m {
                let ci = if cin == 1 { 0 } else { ch };
                let mut acc = 0.0f32;
                for i in 0..3 {
                    let h = (2 * ho + i) as isize - 1;
                    for j in 0..3 {
                        let t = (2 * to + j) as isize - 1;
                        let inside = h >= 0 && (h as usize) < k && t >= 0 && (t as usize) < n;
                        let v = if inside {
                            x[(t as usize * k + h as usize) * cin + ci]
                        } else {
                            0.0
                        };
                        acc = from_bf16(w[ch * 9 + 3 * i + j]).mul_add(v, acc);
                    }
                }
                let v = acc + from_bf16(b[ch]);
                y[(to * ho_n + ho) * m + ch] = if silu { v / (1.0 + (-v).exp()) } else { v };
            }
        }
    }
    y
}

/// The same convolution as `nn.Conv2d(kernel_size=3, stride=2, padding=1)` in the layout of the reference, in f64:
/// the input is `[cin][k][n]` (channels, mel as the height, time as the width), and the output `[m][ho][to]`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_reference(
    x: &[f64],
    w: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    cin: usize,
    m: usize,
    silu: bool,
) -> Vec<f64> {
    let (to_n, ho_n) = (n.div_ceil(2), k.div_ceil(2));
    let mut y = vec![0.0f64; m * ho_n * to_n];
    for ch in 0..m {
        // groups = cin when cin > 1, so out channel ch reads in channel ch.
        let ci = if cin == 1 { 0 } else { ch };
        for ho in 0..ho_n {
            for to in 0..to_n {
                let mut s = b[ch];
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (hy, wx) = ((2 * ho + ky) as isize - 1, (2 * to + kx) as isize - 1);
                        if hy >= 0 && (hy as usize) < k && wx >= 0 && (wx as usize) < n {
                            s += w[ch * 9 + ky * 3 + kx] * x[(ci * k + hy as usize) * n + wx as usize];
                        }
                    }
                }
                y[(ch * ho_n + ho) * to_n + to] = if silu { s / (1.0 + (-s).exp()) } else { s };
            }
        }
    }
    y
}
