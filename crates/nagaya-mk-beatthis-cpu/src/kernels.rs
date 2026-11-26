//! The seven task kinds of the plan (`nagaya_family_beatthis::plan::Task`) on the CPU, in f32.
//!
//! The bits do not depend on the thread count (spec 3.2c). Each kernel splits its output into whole rows, and one
//! worker computes each row whole. The sums inside a value have a fixed order: `dot` keeps 8 lanes by index modulo 8
//! and joins them in pairs. Rust does not join a multiply and an add into one rounding, and the transcendental
//! functions come from the `libm` crate, so the bits are the same on each machine for one build. Some kernels have
//! code for each level of `isa` (SSE, AVX2, AVX-512); each level gives the same bits.

use std::f32::consts::FRAC_1_SQRT_2;

use nagaya_family_beatthis::plan::Axis;

use crate::isa::{self, leveled};
use crate::lanes::{F4, Isa, join};
use crate::pool;

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

/// Starts the `threads` workers of this thread (`pool`), so that the first task does not wait for them. With one
/// thread there are no workers.
pub fn start_workers(threads: usize) {
    if threads > 1 {
        pool::start(threads);
    }
}

/// The output of a job, shared by the workers. Each worker makes a slice of its own rows from it.
struct Out(*mut f32);

// SAFETY: the workers of `split` write rows that do not overlap.
unsafe impl Sync for Out {}

impl Out {
    /// A method, so that a closure takes all of `Out` (which is `Sync`), not its field.
    fn at(&self, i: usize) -> *mut f32 {
        self.0.wrapping_add(i)
    }
}

/// Splits `out` into rows of `width` values, and gives each worker of `pool` a run of whole rows. `f(first_row,
/// rows)` writes the rows from `first_row`. With one thread it runs on the caller's thread.
fn split<F>(out: &mut [f32], width: usize, threads: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    assert!(
        width > 0 && out.len().is_multiple_of(width),
        "the output is not whole rows of {width}"
    );
    let rows = out.len() / width;
    let threads = threads.max(1);
    if threads == 1 || rows <= 1 {
        f(0, out);
        return;
    }
    let per = rows.div_ceil(threads);
    let len = out.len();
    let base = Out(out.as_mut_ptr());
    pool::run(threads, &|i| {
        let start = (i * per * width).min(len);
        let end = ((i + 1) * per * width).min(len);
        if start == end {
            return;
        }
        // SAFETY: the runs [start, end) of the workers do not overlap and are in `out`, and `out` is borrowed
        // mutably until `pool::run` returns, after each worker ends.
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.at(start), end - start) };
        f(i * per, chunk);
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

/// `y[r, o] = dot(weight[o], x[r]) + bias[o]`. A worker takes a run of whole rows, and computes them in blocks of
/// rows by outputs (`Isa::dots`; the block size depends on the level). Each value is one `dot`, so the blocks and
/// the level do not change its bits.
pub fn linear(x: &[f32], y: &mut [f32], din: usize, dout: usize, weight: &[f32], bias: Option<&[f32]>, threads: usize) {
    assert_eq!(weight.len(), dout * din);
    assert_eq!(x.len() / din, y.len() / dout, "x and y have different rows");
    let level = isa::level();
    split(y, dout, threads, |first, rows| {
        let x = &x[first * din..][..rows.len() / dout * din];
        linear_rows(level, x, rows, din, dout, weight, bias)
    });
}

leveled! {
    /// The rows of `linear` for one worker: `y` from the rows of `x`, in the blocks of the level.
    fn linear_rows(x: &[f32], y: &mut [f32], din: usize, dout: usize, weight: &[f32], bias: Option<&[f32]>) =
        linear_rows_of;
}

#[inline(always)]
fn linear_rows_of<I: LinearBlock>(
    x: &[f32],
    y: &mut [f32],
    din: usize,
    dout: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) {
    let put = |v: f32, o: usize| match bias {
        Some(b) => v + b[o],
        None => v,
    };
    let n = y.len() / dout;
    let xr = |r: usize| &x[r * din..][..din];
    let wr = |o: usize| &weight[o * din..][..din];
    let full = n - n % I::ROWS;
    let outs = dout - dout % I::OUTS;
    for o0 in (0..outs).step_by(I::OUTS) {
        for r0 in (0..full).step_by(I::ROWS) {
            I::block(
                |r| xr(r0 + r),
                |o| wr(o0 + o),
                |r, o, v| y[(r0 + r) * dout + o0 + o] = put(v, o0 + o),
            );
        }
        for r in full..n {
            I::row(xr(r), |o| wr(o0 + o), |o, v| y[r * dout + o0 + o] = put(v, o0 + o));
        }
    }
    for o in outs..dout {
        for r in 0..n {
            let [[v]] = I::dots([xr(r)], [wr(o)]);
            y[r * dout + o] = put(v, o);
        }
    }
}

/// The block size of `linear` for each level: `ROWS` rows by `OUTS` outputs. `block` and `row` call `Isa::dots` with
/// those sizes, and give each value to `put`.
trait LinearBlock: Isa {
    const ROWS: usize;
    const OUTS: usize;
    fn block<'a>(x: impl Fn(usize) -> &'a [f32], w: impl Fn(usize) -> &'a [f32], put: impl FnMut(usize, usize, f32));
    fn row<'a>(x: &'a [f32], w: impl Fn(usize) -> &'a [f32], put: impl FnMut(usize, f32));
}

macro_rules! linear_block {
    ($t:ty, $rows:literal, $outs:literal) => {
        impl LinearBlock for $t {
            const ROWS: usize = $rows;
            const OUTS: usize = $outs;
            #[inline(always)]
            fn block<'a>(
                x: impl Fn(usize) -> &'a [f32],
                w: impl Fn(usize) -> &'a [f32],
                mut put: impl FnMut(usize, usize, f32),
            ) {
                let v = Self::dots::<$rows, $outs>(std::array::from_fn(x), std::array::from_fn(w));
                for (r, vr) in v.iter().enumerate() {
                    for (o, &vo) in vr.iter().enumerate() {
                        put(r, o, vo);
                    }
                }
            }
            #[inline(always)]
            fn row<'a>(x: &'a [f32], w: impl Fn(usize) -> &'a [f32], mut put: impl FnMut(usize, f32)) {
                let [v] = Self::dots::<1, $outs>([x], std::array::from_fn(w));
                for (o, &vo) in v.iter().enumerate() {
                    put(o, vo);
                }
            }
        }
    };
}

linear_block!(crate::lanes::Sse, 3, 2);
#[cfg(target_arch = "x86_64")]
linear_block!(crate::lanes::Avx2, 2, 4);
#[cfg(target_arch = "x86_64")]
linear_block!(crate::lanes::Avx512, 4, 8);

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

/// `x[i] = expf(x[i] - shift)`, with the bits of `libm::expf`: 4 values at once in `F4::exp`.
pub fn exps(x: &mut [f32], shift: f32) {
    let (body, tail) = x.as_chunks_mut::<4>();
    let s = F4::splat(shift);
    for c in body {
        F4::load(c).sub(s).exp().store(c);
    }
    for v in tail {
        *v = libm::expf(*v - shift);
    }
}

/// The tokens of one block of the scores: `scores` computes the scores of this many tokens at once.
const TOKENS: usize = 4;

/// `out[t] = dot(q, k_t) * scale` for each token `t`, from `kt`, the keys in blocks of `TOKENS` tokens: value `i` of
/// token `t` is at `kt[(t / TOKENS) * hd * TOKENS + i * TOKENS + t % TOKENS]`. It keeps the lanes of `dot` for each token (one `F4` of `TOKENS` tokens for each lane), so each
/// score has the bits of `dot`. `stride` is a multiple of `TOKENS`, and the tokens past the end of `out` are padding.
fn scores(q: &[f32], kt: &[f32], stride: usize, scale: f32, out: &mut [f32]) {
    let hd = q.len();
    debug_assert!(stride.is_multiple_of(TOKENS) && out.len() <= stride && kt.len() == hd * stride);
    let scale = F4::splat(scale);
    for t0 in (0..out.len()).step_by(TOKENS) {
        // l[k]: lane k of the dots of the tokens t0 to t0 + 3.
        let mut l = [F4::zero(); 8];
        let block = &kt[t0 * hd..][..TOKENS * hd];
        let term = |i: usize| F4::splat(q[i]).mul(F4::at(block, i * TOKENS));
        let chunks = hd / 8;
        for c in 0..chunks {
            for (k, lane) in l.iter_mut().enumerate() {
                *lane = lane.add(term(c * 8 + k));
            }
        }
        for (k, lane) in l.iter_mut().enumerate().take(hd % 8) {
            *lane = lane.add(term(chunks * 8 + k));
        }
        let j = ((l[0].add(l[1])).add(l[2].add(l[3]))).add((l[4].add(l[5])).add(l[6].add(l[7])));
        let v = j.mul(scale).array();
        let n = TOKENS.min(out.len() - t0);
        out[t0..t0 + n].copy_from_slice(&v[..n]);
    }
}

/// `o[d] = sum over t of p[t] * v[t * hd + d]`, the terms in the order of t, from `o[d] = 0`.
fn weighted_sum(p: &[f32], v: &[f32], o: &mut [f32]) {
    let hd = o.len();
    let mut d0 = 0;
    // Blocks of 32 values in 8 `F4`, then of 4 values, then single values. Each value is its own sum.
    while d0 + 32 <= hd {
        let mut acc = [F4::zero(); 8];
        for (t, &w) in p.iter().enumerate() {
            let w = F4::splat(w);
            for (j, a) in acc.iter_mut().enumerate() {
                *a = a.add(w.mul(F4::at(v, t * hd + d0 + 4 * j)));
            }
        }
        for (j, a) in acc.iter().enumerate() {
            o[d0 + 4 * j..][..4].copy_from_slice(&a.array());
        }
        d0 += 32;
    }
    while d0 + 4 <= hd {
        let mut acc = F4::zero();
        for (t, &w) in p.iter().enumerate() {
            acc = acc.add(F4::splat(w).mul(F4::at(v, t * hd + d0)));
        }
        o[d0..][..4].copy_from_slice(&acc.array());
        d0 += 4;
    }
    for d in d0..hd {
        let mut acc = 0.0f32;
        for (t, &w) in p.iter().enumerate() {
            acc += w * v[t * hd + d];
        }
        o[d] = acc;
    }
}

/// Rotates the pairs of `src` by the angles of `cs` (cos, sin), into `dst[j * step]` for value `j`.
fn rotate(src: &[f32], cs: &[(f32, f32)], dst: &mut [f32], step: usize) {
    for (i, &(c, s)) in cs.iter().enumerate() {
        let (x0, x1) = (src[2 * i], src[2 * i + 1]);
        dst[2 * i * step] = x0 * c + -x1 * s;
        dst[(2 * i + 1) * step] = x1 * c + x0 * s;
    }
}

/// The attention of `Task::Attention`. Each row of the output is computed whole, by one worker: the scores of its
/// sequence (each a `dot` of the rotated q and k), `exp(s - max) / sum` in the lanes of `dot`, the sum of `p * v` over
/// the tokens in order, and the gate.
///
/// A worker takes a run of the rows in the order (sequence, head, position). For each (sequence, head) that it meets,
/// it rotates the keys into blocks of `TOKENS` tokens (`scores`) and copies the values, so that the scores and the
/// sum of `p * v` of a query read memory in order. The rows go to a scratch buffer in that order, and a last pass puts them in the order of the output.
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
    let len = match along {
        Axis::A => a,
        Axis::B => b,
    };
    // The row of the token at `pos` in sequence `s`.
    let token = |s: usize, pos: usize| match along {
        Axis::A => pos * b + s,
        Axis::B => s * b + pos,
    };
    // cos and sin of `pos * freqs[i]` at `table[pos * hd / 2 + i]`, as the rotation of each row needs them.
    let half = hd / 2;
    let table: Vec<(f32, f32)> = (0..len * half)
        .map(|n| {
            let angle = (n / half) as f32 * freqs[n % half];
            (libm::cosf(angle), libm::sinf(angle))
        })
        .collect();
    let stride = len.next_multiple_of(TOKENS);
    let scale = 1.0 / (hd as f32).sqrt();
    // The rows in the order (sequence, head, position), `hd` values each.
    let mut ordered = vec![0.0f32; rows * width];
    split(&mut ordered, hd, threads, |first, chunk| {
        let mut kt = vec![0.0f32; hd * stride];
        let mut v = vec![0.0f32; len * hd];
        let mut q = vec![0.0f32; hd];
        let mut p = vec![0.0f32; len];
        let mut loaded = usize::MAX;
        for (n, o) in chunk.chunks_exact_mut(hd).enumerate() {
            let unit = (first + n) / len;
            let pos = (first + n) % len;
            let (s, h) = (unit / heads, unit % heads);
            if unit != loaded {
                for t in 0..len {
                    let row = &qkv[token(s, t) * 3 * width..][..3 * width];
                    rotate(
                        &row[width + h * hd..][..hd],
                        &table[t * half..][..half],
                        &mut kt[(t / TOKENS) * hd * TOKENS + t % TOKENS..],
                        TOKENS,
                    );
                    v[t * hd..][..hd].copy_from_slice(&row[2 * width + h * hd..][..hd]);
                }
                loaded = unit;
            }
            let r = token(s, pos);
            rotate(
                &qkv[r * 3 * width + h * hd..][..hd],
                &table[pos * half..][..half],
                &mut q,
                1,
            );
            scores(&q, &kt, stride, scale, &mut p);
            let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            exps(&mut p, max);
            let total = sum(&p);
            for x in p.iter_mut() {
                *x /= total;
            }
            weighted_sum(&p, &v, o);
            let g = gates[r * heads + h];
            let gate = 1.0 / (1.0 + libm::expf(-g));
            for x in o.iter_mut() {
                *x *= gate;
            }
        }
    });
    split(out, width, threads, |first, chunk| {
        for (n, row) in chunk.chunks_exact_mut(width).enumerate() {
            let r = first + n;
            let (s, pos) = match along {
                Axis::A => (r % b, r / b),
                Axis::B => (r / b, r % b),
            };
            for h in 0..heads {
                let src = &ordered[((s * heads + h) * len + pos) * hd..][..hd];
                row[h * hd..][..hd].copy_from_slice(src);
            }
        }
    });
}
