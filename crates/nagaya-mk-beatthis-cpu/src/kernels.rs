//! The seven task kinds of the plan (`nagaya_family_beatthis::plan::Task`) on the CPU, in f32.
//!
//! The bits do not depend on the thread count (spec 3.2c). Each kernel splits its output into whole rows, and one
//! worker computes each row whole. The sums inside a value have a fixed order: `dot` keeps 8 lanes by index modulo 8
//! and joins them in pairs. Rust does not join a multiply and an add into one rounding, and the transcendental
//! functions come from the `libm` crate, so the bits are the same on each machine for one build.

use std::f32::consts::FRAC_1_SQRT_2;

use nagaya_family_beatthis::plan::Axis;

/// `sum of a[i] * b[i]`: lane `i % 8` holds the terms of its indexes in order, and the lanes join as
/// `((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot of slices of two lengths");
    let mut l = [0.0f32; 8];
    let (ac, at) = a.as_chunks::<8>();
    let (bc, bt) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            l[k] += x[k] * y[k];
        }
    }
    for (k, (x, y)) in at.iter().zip(bt).enumerate() {
        l[k] += x * y;
    }
    join(l)
}

/// The sum of `v` in the lanes of `dot`.
fn sum(v: &[f32]) -> f32 {
    let mut l = [0.0f32; 8];
    let (c, t) = v.as_chunks::<8>();
    for x in c {
        for k in 0..8 {
            l[k] += x[k];
        }
    }
    for (k, x) in t.iter().enumerate() {
        l[k] += x;
    }
    join(l)
}

fn join(l: [f32; 8]) -> f32 {
    ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
}

/// Splits `out` into rows of `width` values, and gives each worker a run of whole rows. `f(first_row, rows)` writes
/// the rows from `first_row`. With one thread it runs on the caller's thread.
fn split<F>(out: &mut [f32], width: usize, threads: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    assert!(
        width > 0 && out.len().is_multiple_of(width),
        "the output is not whole rows of {width}"
    );
    let rows = out.len() / width;
    let threads = threads.clamp(1, rows.max(1));
    if threads == 1 {
        f(0, out);
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (n, chunk) in out.chunks_mut(per * width).enumerate() {
            let f = &f;
            s.spawn(move || f(n * per, chunk));
        }
    });
}

/// `y[r, c] = x[r, c] * scale[c] + shift[c]`.
pub fn affine(x: &[f32], y: &mut [f32], cols: usize, scale: &[f32], shift: &[f32], threads: usize) {
    assert_eq!(x.len(), y.len());
    split(y, cols, threads, |first, rows| {
        for (k, row) in rows.chunks_exact_mut(cols).enumerate() {
            let xr = &x[(first + k) * cols..][..cols];
            for c in 0..cols {
                row[c] = xr[c] * scale[c] + shift[c];
            }
        }
    });
}

/// The convolution of `Task::Conv`. The sum of a value runs over `(i, j)` in order, each term a `dot` over the input
/// channels. A tap past the first or the last frame reads zeros, so it adds nothing and is left out.
#[allow(clippy::too_many_arguments)]
pub fn conv(
    x: &[f32],
    y: &mut [f32],
    frames: usize,
    f_in: usize,
    cin: usize,
    cout: usize,
    kf: usize,
    kt: usize,
    stride: usize,
    kernel: &[f32],
    threads: usize,
) {
    let f_out = (f_in - kf) / stride + 1;
    assert_eq!(x.len(), frames * f_in * cin);
    assert_eq!(y.len(), frames * f_out * cout);
    assert_eq!(kernel.len(), cout * kf * kt * cin);
    let half = (kt / 2) as isize;
    // One row of the split is one frame and one output frequency: `cout` values.
    split(y, cout, threads, |first, rows| {
        for (k, row) in rows.chunks_exact_mut(cout).enumerate() {
            let (t, o) = ((first + k) / f_out, (first + k) % f_out);
            for (co, v) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for i in 0..kf {
                    for j in 0..kt {
                        let tt = t as isize + j as isize - half;
                        if tt < 0 || tt >= frames as isize {
                            continue;
                        }
                        let xs = &x[(tt as usize * f_in + o * stride + i) * cin..][..cin];
                        let ks = &kernel[((co * kf + i) * kt + j) * cin..][..cin];
                        acc += dot(ks, xs);
                    }
                }
                *v = acc;
            }
        }
    });
}

/// `x = x * 0.5 * (1 + erf(x / sqrt(2)))`, in place: torch's exact GELU, in its order of operations.
pub fn gelu(x: &mut [f32], threads: usize) {
    let width = 1024.min(x.len().max(1));
    let n = x.len();
    let mut tail = n % width;
    let (body, rest) = x.split_at_mut(n - tail);
    split(body, width, threads, |_, rows| {
        for v in rows.iter_mut() {
            *v = *v * 0.5 * (1.0 + libm::erff(*v * FRAC_1_SQRT_2));
        }
    });
    for v in rest.iter_mut() {
        *v = *v * 0.5 * (1.0 + libm::erff(*v * FRAC_1_SQRT_2));
        tail -= 1;
    }
    debug_assert_eq!(tail, 0);
}

/// `y[r] = x[r] / max(|x[r]|, 1e-12) * sqrt(dim) * gamma`, in that order, as `F.normalize` and the model's scale.
pub fn rms_norm(x: &[f32], y: &mut [f32], dim: usize, gamma: &[f32], threads: usize) {
    assert_eq!(x.len(), y.len());
    let scale = (dim as f32).sqrt();
    split(y, dim, threads, |first, rows| {
        for (k, row) in rows.chunks_exact_mut(dim).enumerate() {
            let xr = &x[(first + k) * dim..][..dim];
            let norm = dot(xr, xr).sqrt().max(1e-12);
            for i in 0..dim {
                row[i] = xr[i] / norm * scale * gamma[i];
            }
        }
    });
}

/// `y[r, o] = dot(weight[o], x[r]) + bias[o]`.
pub fn linear(x: &[f32], y: &mut [f32], din: usize, dout: usize, weight: &[f32], bias: Option<&[f32]>, threads: usize) {
    assert_eq!(weight.len(), dout * din);
    assert_eq!(x.len() / din, y.len() / dout, "x and y have different rows");
    split(y, dout, threads, |first, rows| {
        for (k, row) in rows.chunks_exact_mut(dout).enumerate() {
            let xr = &x[(first + k) * din..][..din];
            for (o, v) in row.iter_mut().enumerate() {
                let d = dot(&weight[o * din..][..din], xr);
                *v = match bias {
                    Some(b) => d + b[o],
                    None => d,
                };
            }
        }
    });
}

/// `acc[i] += x[i]`.
pub fn add_into(acc: &mut [f32], x: &[f32], threads: usize) {
    assert_eq!(acc.len(), x.len());
    let width = 1024.min(acc.len().max(1));
    let body = acc.len() - acc.len() % width;
    let (head, rest) = acc.split_at_mut(body);
    split(head, width, threads, |first, rows| {
        let xs = &x[first * width..][..rows.len()];
        for (a, b) in rows.iter_mut().zip(xs) {
            *a += b;
        }
    });
    for (a, b) in rest.iter_mut().zip(&x[body..]) {
        *a += b;
    }
}

/// The attention of `Task::Attention`. A first pass rotates q and k of each row by its position. A second pass
/// computes each output row whole: the scores of its sequence, `exp(s - max) / sum` in the lanes of `dot`, the sum of
/// `p * v` over the tokens in order, and the gate.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    qkv: &[f32],
    gates: &[f32],
    out: &mut [f32],
    a: usize,
    b: usize,
    along: Axis,
    heads: usize,
    hd: usize,
    freqs: &[f32],
    threads: usize,
) {
    let width = heads * hd;
    let rows = a * b;
    assert_eq!(qkv.len(), rows * 3 * width);
    assert_eq!(gates.len(), rows * heads);
    assert_eq!(out.len(), rows * width);
    assert_eq!(freqs.len(), hd / 2);
    let position = |r: usize| match along {
        Axis::A => r / b,
        Axis::B => r % b,
    };
    // q and k of each row, rotated: row r holds q (width values), then k (width values).
    let mut qk = vec![0.0f32; rows * 2 * width];
    split(&mut qk, 2 * width, threads, |first, chunk| {
        for (n, row) in chunk.chunks_exact_mut(2 * width).enumerate() {
            let r = first + n;
            let p = position(r) as f32;
            let src = &qkv[r * 3 * width..][..2 * width];
            for part in 0..2 {
                for h in 0..heads {
                    let base = part * width + h * hd;
                    for i in 0..hd / 2 {
                        let angle = p * freqs[i];
                        let (c, s) = (libm::cosf(angle), libm::sinf(angle));
                        let (x0, x1) = (src[base + 2 * i], src[base + 2 * i + 1]);
                        row[base + 2 * i] = x0 * c + -x1 * s;
                        row[base + 2 * i + 1] = x1 * c + x0 * s;
                    }
                }
            }
        }
    });
    let len = match along {
        Axis::A => a,
        Axis::B => b,
    };
    let token = |r: usize, pos: usize| match along {
        Axis::A => pos * b + r % b,
        Axis::B => (r / b) * b + pos,
    };
    let scale = 1.0 / (hd as f32).sqrt();
    split(out, width, threads, |first, chunk| {
        let mut scores = vec![0.0f32; len];
        for (n, row) in chunk.chunks_exact_mut(width).enumerate() {
            let r = first + n;
            for h in 0..heads {
                let q = &qk[r * 2 * width + h * hd..][..hd];
                for (pos, s) in scores.iter_mut().enumerate() {
                    let k = &qk[token(r, pos) * 2 * width + width + h * hd..][..hd];
                    *s = dot(q, k) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in scores.iter_mut() {
                    *s = libm::expf(*s - max);
                }
                let total = sum(&scores);
                for s in scores.iter_mut() {
                    *s /= total;
                }
                let o = &mut row[h * hd..][..hd];
                o.fill(0.0);
                for (pos, &p) in scores.iter().enumerate() {
                    let v = &qkv[token(r, pos) * 3 * width + 2 * width + h * hd..][..hd];
                    for d in 0..hd {
                        o[d] += p * v[d];
                    }
                }
                let g = gates[r * heads + h];
                let gate = 1.0 / (1.0 + libm::expf(-g));
                for v in o.iter_mut() {
                    *v *= gate;
                }
            }
        }
    });
}
