//! The CPU forms of the speech ops of the LFM3 binary: the attention of the depth decoder in
//! groups, the polar form, and the inverse real FFT with the window. `lfm3_speech_cpu_forms.rs` checks them on the
//! host against closed forms, and `xe_lfm3_speech_ops.rs` checks the kernels against them. Each form sums in f64.
#![allow(dead_code)]

/// Attention in both directions inside groups of `group` rows: `q [n][heads][d]`, `k` and `v` `[n][kv][d]`, each
/// score times `scale`.
#[allow(clippy::too_many_arguments)]
pub fn group_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    kv: usize,
    d: usize,
    group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * heads * d];
    for t in 0..n {
        let lo = t / group * group;
        let hi = (lo + group).min(n);
        for h in 0..heads {
            let g = h / (heads / kv);
            let qr = &q[(t * heads + h) * d..(t * heads + h + 1) * d];
            let scores: Vec<f64> = (lo..hi)
                .map(|s| {
                    let kr = &k[(s * kv + g) * d..(s * kv + g + 1) * d];
                    qr.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale as f64
                })
                .collect();
            let most = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|x| (x - most).exp()).collect();
            let sum: f64 = e.iter().sum();
            for j in 0..d {
                let acc: f64 = (lo..hi).zip(&e).map(|(s, p)| p * v[(s * kv + g) * d + j] as f64).sum();
                out[(t * heads + h) * d + j] = (acc / sum) as f32;
            }
        }
    }
    out
}

/// The bins `[n][2f]` (re, then im) of rows `[n][stride]` of `f` log magnitudes and then `f` angles.
pub fn polar(a: &[f32], n: usize, f: usize, stride: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * 2 * f];
    for t in 0..n {
        for i in 0..f {
            let mag = (a[t * stride + i] as f64).exp();
            let ang = a[t * stride + f + i] as f64;
            out[t * 2 * f + i] = (mag * ang.cos()) as f32;
            out[t * 2 * f + f + i] = (mag * ang.sin()) as f32;
        }
    }
    out
}

/// The inverse real FFT of `n` rows of `nf / 2 + 1` bins, times the window `w`, with no FFT.
pub fn irfft(bins: &[f32], n: usize, nf: usize, w: &[f32]) -> Vec<f32> {
    let f = nf / 2 + 1;
    let mut out = vec![0.0f32; n * nf];
    for t in 0..n {
        let re = &bins[t * 2 * f..t * 2 * f + f];
        let im = &bins[t * 2 * f + f..(t + 1) * 2 * f];
        for i in 0..nf {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let mut acc = re[0] as f64 + sign * re[f - 1] as f64;
            for k in 1..f - 1 {
                let a = 2.0 * std::f64::consts::PI * ((k * i) % nf) as f64 / nf as f64;
                acc += 2.0 * (re[k] as f64 * a.cos() - im[k] as f64 * a.sin());
            }
            out[t * nf + i] = (acc / nf as f64 * w[i] as f64) as f32;
        }
    }
    out
}

/// The periodic Hann window of `n` samples, as `torch.hann_window` gives it.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}
