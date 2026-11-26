//! Each kernel against a naive f64 form written here from the definitions of `plan::Task`. The naive forms share no
//! code with the kernels, so a wrong index, axis or order shows as a large gap, not a rounding gap.

use nagaya_family_beatthis::plan::Axis;
use nagaya_mk_beatthis_cpu::kernels;

/// Values in [-1, 1) from a linear congruential generator.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// The largest gap of `got` from `want`, relative to the largest value of `want`.
fn rel_gap(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0.0, f64::max)
        / peak
}

const GATE: f64 = 1e-5;

#[test]
fn dot_sums_in_eight_lanes_joined_in_pairs() {
    // 1e8 swallows a 1 when they meet first; the lane order says which ones meet.
    let mut a = vec![0.0f32; 17];
    a[0] = 1e8;
    a[8] = 1.0;
    a[16] = -1e8;
    a[1] = 3.0;
    let ones = vec![1.0f32; 17];
    // Lane 0 holds a[0] + a[8] + a[16] = (1e8 + 1) - 1e8 = 0 in f32; lane 1 holds 3.
    assert_eq!(kernels::dot(&a, &ones), 3.0);
}

#[test]
fn linear_matches_the_naive_form() {
    let (rows, din, dout) = (7, 37, 11);
    let x = values(rows * din, 1);
    let w = values(dout * din, 2);
    let b = values(dout, 3);
    for threads in [1, 3] {
        let mut y = vec![0.0f32; rows * dout];
        kernels::linear(&x, &mut y, din, dout, &w, Some(&b), threads);
        let mut want = vec![0.0f64; rows * dout];
        for r in 0..rows {
            for o in 0..dout {
                want[r * dout + o] = (0..din)
                    .map(|i| f64::from(w[o * din + i]) * f64::from(x[r * din + i]))
                    .sum::<f64>()
                    + f64::from(b[o]);
            }
        }
        assert!(rel_gap(&y, &want) < GATE, "threads {threads}");
    }
}

#[test]
fn conv_matches_the_naive_form() {
    let (frames, f_in, cin, cout, kf, kt, stride) = (5, 8, 3, 4, 2, 3, 2);
    let f_out = (f_in - kf) / stride + 1;
    let x = values(frames * f_in * cin, 4);
    let k = values(cout * kf * kt * cin, 5);
    let mut y = vec![0.0f32; frames * f_out * cout];
    kernels::conv(&x, &mut y, frames, f_in, cin, cout, kf, kt, stride, &k, 2);
    let mut want = vec![0.0f64; y.len()];
    for t in 0..frames as isize {
        for o in 0..f_out {
            for co in 0..cout {
                let mut s = 0.0f64;
                for i in 0..kf {
                    for j in 0..kt {
                        let tt = t + j as isize - (kt / 2) as isize;
                        if tt < 0 || tt >= frames as isize {
                            continue;
                        }
                        for ci in 0..cin {
                            let kv = k[((co * kf + i) * kt + j) * cin + ci];
                            let xv = x[(tt as usize * f_in + o * stride + i) * cin + ci];
                            s += f64::from(kv) * f64::from(xv);
                        }
                    }
                }
                want[(t as usize * f_out + o) * cout + co] = s;
            }
        }
    }
    assert!(rel_gap(&y, &want) < GATE);
}

#[test]
fn affine_gelu_rms_norm_and_add_match_their_definitions() {
    let (rows, dim) = (3, 10);
    let x = values(rows * dim, 6);
    let scale = values(dim, 7);
    let shift = values(dim, 8);
    let mut y = vec![0.0f32; rows * dim];
    kernels::affine(&x, &mut y, dim, &scale, &shift, 2);
    let want: Vec<f64> = (0..rows * dim)
        .map(|i| f64::from(x[i]) * f64::from(scale[i % dim]) + f64::from(shift[i % dim]))
        .collect();
    assert!(rel_gap(&y, &want) < GATE, "affine");

    let mut g = x.clone();
    kernels::gelu(&mut g, 2);
    let erf = |v: f64| libm::erf(v);
    let want: Vec<f64> = x
        .iter()
        .map(|&v| f64::from(v) * 0.5 * (1.0 + erf(f64::from(v) / 2f64.sqrt())))
        .collect();
    assert!(rel_gap(&g, &want) < GATE, "gelu");

    let gamma = values(dim, 9);
    let mut n = vec![0.0f32; rows * dim];
    kernels::rms_norm(&x, &mut n, dim, &gamma, 2);
    let mut want = vec![0.0f64; rows * dim];
    for r in 0..rows {
        let norm = (0..dim).map(|i| f64::from(x[r * dim + i]).powi(2)).sum::<f64>().sqrt();
        for i in 0..dim {
            want[r * dim + i] = f64::from(x[r * dim + i]) / norm * (dim as f64).sqrt() * f64::from(gamma[i]);
        }
    }
    assert!(rel_gap(&n, &want) < GATE, "rms_norm");

    let mut acc = x.clone();
    kernels::add_into(&mut acc, &scale.repeat(rows), 2);
    let want: Vec<f64> = (0..rows * dim)
        .map(|i| f64::from(x[i]) + f64::from(scale[i % dim]))
        .collect();
    assert!(rel_gap(&acc, &want) < GATE, "add_into");
}

#[test]
fn rms_norm_of_zeros_is_zero() {
    let mut y = vec![1.0f32; 8];
    kernels::rms_norm(&[0.0; 8], &mut y, 8, &[1.0; 8], 1);
    assert!(y.iter().all(|&v| v == 0.0));
}

/// The naive attention of `plan::Task::Attention` in f64.
#[allow(clippy::too_many_arguments)]
fn naive_attention(
    qkv: &[f32],
    gates: &[f32],
    a: usize,
    b: usize,
    along: Axis,
    heads: usize,
    hd: usize,
    freqs: &[f32],
) -> Vec<f64> {
    let width = heads * hd;
    let rows = a * b;
    // The tokens of each row's sequence, and its position in it.
    let seq = |r: usize| -> (Vec<usize>, usize) {
        let (i, j) = (r / b, r % b);
        match along {
            Axis::A => ((0..a).map(|ii| ii * b + j).collect(), i),
            Axis::B => ((0..b).map(|jj| i * b + jj).collect(), j),
        }
    };
    let rotate = |v: &[f32], p: usize| -> Vec<f64> {
        let mut out = vec![0.0f64; hd];
        for k in 0..hd / 2 {
            let ang = p as f64 * f64::from(freqs[k]);
            let (x0, x1) = (f64::from(v[2 * k]), f64::from(v[2 * k + 1]));
            out[2 * k] = x0 * ang.cos() - x1 * ang.sin();
            out[2 * k + 1] = x1 * ang.cos() + x0 * ang.sin();
        }
        out
    };
    let mut out = vec![0.0f64; rows * width];
    for r in 0..rows {
        let (tokens, p) = seq(r);
        for h in 0..heads {
            let q = rotate(&qkv[r * 3 * width + h * hd..][..hd], p);
            let scores: Vec<f64> = tokens
                .iter()
                .enumerate()
                .map(|(pos, &t)| {
                    let k = rotate(&qkv[t * 3 * width + width + h * hd..][..hd], pos);
                    q.iter().zip(&k).map(|(x, y)| x * y).sum::<f64>() / (hd as f64).sqrt()
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = e.iter().sum();
            let gate = 1.0 / (1.0 + (-f64::from(gates[r * heads + h])).exp());
            for d in 0..hd {
                let v: f64 = tokens
                    .iter()
                    .zip(&e)
                    .map(|(&t, w)| w / sum * f64::from(qkv[t * 3 * width + 2 * width + h * hd + d]))
                    .sum();
                out[r * width + h * hd + d] = v * gate;
            }
        }
    }
    out
}

#[test]
fn attention_matches_the_naive_form_on_each_axis() {
    let (a, b, heads, hd) = (6, 4, 2, 8);
    let width = heads * hd;
    let qkv = values(a * b * 3 * width, 10);
    let gates = values(a * b * heads, 11);
    let freqs: Vec<f32> = (0..hd / 2)
        .map(|k| 1.0 / 10_000f32.powf(2.0 * k as f32 / hd as f32))
        .collect();
    for along in [Axis::A, Axis::B] {
        for threads in [1, 4] {
            let mut out = vec![0.0f32; a * b * width];
            kernels::attention(&qkv, &gates, &mut out, a, b, along, heads, hd, &freqs, threads);
            let want = naive_attention(&qkv, &gates, a, b, along, heads, hd, &freqs);
            let gap = rel_gap(&out, &want);
            assert!(gap < GATE, "{along:?}, threads {threads}: gap {gap:e}");
        }
    }
}

#[test]
fn each_kernel_gives_the_same_bits_for_each_thread_count() {
    let (a, b, heads, hd) = (9, 5, 3, 8);
    let width = heads * hd;
    let qkv = values(a * b * 3 * width, 12);
    let gates = values(a * b * heads, 13);
    let freqs = values(hd / 2, 14);
    let run = |threads| {
        let mut out = vec![0.0f32; a * b * width];
        kernels::attention(&qkv, &gates, &mut out, a, b, Axis::A, heads, hd, &freqs, threads);
        let mut y = vec![0.0f32; a * b * 7];
        kernels::linear(&out, &mut y, width, 7, &values(7 * width, 15), None, threads);
        y
    };
    let one = run(1);
    for threads in [2, 3, 7, 16] {
        let got = run(threads);
        assert!(
            one.iter().zip(&got).all(|(x, y)| x.to_bits() == y.to_bits()),
            "threads {threads} moved bits"
        );
    }
}
