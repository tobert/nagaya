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
use crate::lanes::{self, Isa, Vf, join};
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
///
/// The workers compute a vector of output channels at once: a copy of the kernel holds the output channels last, and
/// each output channel keeps the 8 lanes of its `dot` in its lane of 8 vectors. So each value has the bits of the
/// scalar form at each level.
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
    // The kernel with the output channels last: value (co, i, j, ci) at ((i * kt + j) * cin + ci) * cout + co.
    let mut last = vec![0.0f32; kernel.len()];
    for co in 0..cout {
        for tap in 0..kf * kt {
            for ci in 0..cin {
                last[(tap * cin + ci) * cout + co] = kernel[(co * kf * kt + tap) * cin + ci];
            }
        }
    }
    let job = Conv {
        x,
        frames,
        f_in,
        f_out,
        cin,
        cout,
        kf,
        kt,
        stride,
        kernel,
        last: &last,
    };
    let level = isa::level();
    // One row of the split is one frame and one output frequency: `cout` values.
    split(y, cout, threads, |first, rows| conv_rows(level, &job, first, rows));
}

/// The inputs of the workers of `conv`.
struct Conv<'a> {
    x: &'a [f32],
    frames: usize,
    f_in: usize,
    f_out: usize,
    cin: usize,
    cout: usize,
    kf: usize,
    kt: usize,
    stride: usize,
    kernel: &'a [f32],
    last: &'a [f32],
}

leveled! {
    /// The rows of one worker of `conv`, from row `first`.
    fn conv_rows(job: &Conv, first: usize, rows: &mut [f32]) = conv_rows_of;
}

#[inline(always)]
fn conv_rows_of<I: Isa>(job: &Conv, first: usize, rows: &mut [f32]) {
    let (cin, cout, kt) = (job.cin, job.cout, job.kt);
    let n = I::V::N;
    let half = (kt / 2) as isize;
    let blocks = cout - cout % n;
    for (k, row) in rows.chunks_exact_mut(cout).enumerate() {
        let (t, o) = ((first + k) / job.f_out, (first + k) % job.f_out);
        // The taps (i, j) in order, with the input channels of each; a tap past the frames is left out.
        let taps = (0..job.kf * kt).filter_map(|tap| {
            let (i, j) = (tap / kt, tap % kt);
            let tt = t as isize + j as isize - half;
            (tt >= 0 && tt < job.frames as isize).then(|| {
                (
                    tap,
                    &job.x[(tt as usize * job.f_in + o * job.stride + i) * cin..][..cin],
                )
            })
        });
        for co0 in (0..blocks).step_by(n) {
            let mut acc = I::V::zero();
            for (tap, xs) in taps.clone() {
                let ks = &job.last[tap * cin * cout + co0..];
                let term = |ci: usize| I::V::splat(xs[ci]).mul(I::V::at(ks, ci * cout));
                let mut l = [I::V::zero(); 8];
                let chunks = cin / 8;
                for c in 0..chunks {
                    for (lane, lv) in l.iter_mut().enumerate() {
                        *lv = lv.add(term(c * 8 + lane));
                    }
                }
                for (lane, lv) in l.iter_mut().enumerate().take(cin % 8) {
                    *lv = lv.add(term(chunks * 8 + lane));
                }
                let d = ((l[0].add(l[1])).add(l[2].add(l[3]))).add((l[4].add(l[5])).add(l[6].add(l[7])));
                acc = acc.add(d);
            }
            acc.put(row, co0);
        }
        for co in blocks..cout {
            let mut acc = 0.0f32;
            for (tap, xs) in taps.clone() {
                acc += dot(&job.kernel[(co * job.kf * kt + tap) * cin..][..cin], xs);
            }
            row[co] = acc;
        }
    }
}

/// `x = x * 0.5 * (1 + erf(x / sqrt(2)))`, in place: torch's exact GELU, in its order of operations.
pub fn gelu(x: &mut [f32], threads: usize) {
    let width = 1024.min(x.len().max(1));
    let n = x.len();
    let (body, rest) = x.split_at_mut(n - n % width);
    let level = isa::level();
    split(body, width, threads, |_, rows| gelu_at(level, rows));
    gelu_at(level, rest);
}

leveled! {
    fn gelu_at(x: &mut [f32]) = gelu_of;
}

#[inline(always)]
fn gelu_of<I: Isa>(x: &mut [f32]) {
    let n = I::V::N;
    let body = x.len() - x.len() % n;
    let (half, r2) = (I::V::splat(0.5), I::V::splat(FRAC_1_SQRT_2));
    for i in (0..body).step_by(n) {
        let v = I::V::at(x, i);
        v.mul(half).mul(I::V::splat(1.0).add(lanes::erf(v.mul(r2)))).put(x, i);
    }
    for v in &mut x[body..] {
        *v = *v * 0.5 * (1.0 + libm::erff(*v * FRAC_1_SQRT_2));
    }
}

/// `x[i] = erff(x[i])`, with the bits of `libm::erff`: `lanes::erf` on the vectors of the level.
pub fn erfs(x: &mut [f32]) {
    erfs_at(isa::level(), x);
}

leveled! {
    fn erfs_at(x: &mut [f32]) = erfs_of;
}

#[inline(always)]
fn erfs_of<I: Isa>(x: &mut [f32]) {
    let n = I::V::N;
    let body = x.len() - x.len() % n;
    for i in (0..body).step_by(n) {
        lanes::erf(I::V::at(x, i)).put(x, i);
    }
    for v in &mut x[body..] {
        *v = libm::erff(*v);
    }
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

/// `x[i] = expf(x[i] - shift)`, with the bits of `libm::expf`: `lanes::exp` on the vectors of the level.
pub fn exps(x: &mut [f32], shift: f32) {
    exps_at(isa::level(), x, shift);
}

leveled! {
    fn exps_at(x: &mut [f32], shift: f32) = exps_of;
}

#[inline(always)]
fn exps_of<I: Isa>(x: &mut [f32], shift: f32) {
    let n = I::V::N;
    let body = x.len() - x.len() % n;
    let s = I::V::splat(shift);
    for i in (0..body).step_by(n) {
        lanes::exp(I::V::at(x, i).sub(s)).put(x, i);
    }
    for v in &mut x[body..] {
        *v = libm::expf(*v - shift);
    }
}

/// The tokens of one block of the keys (`scores`). A vector of each level holds at most this many tokens.
const TOKENS: usize = 16;

/// `out[t] = dot(q, k_t) * scale` for each token `t`, from `kt`, the keys in blocks of `TOKENS` tokens: value `i` of
/// token `t` is at `kt[(t / TOKENS) * hd * TOKENS + i * TOKENS + t % TOKENS]`. It keeps the 8 lanes of `dot` for
/// each token, one vector of tokens for each lane, so each score has the bits of `dot`. The tokens past the end of
/// `out` are padding.
#[inline(always)]
fn scores<V: Vf>(q: &[f32], kt: &[f32], scale: f32, out: &mut [f32]) {
    let hd = q.len();
    let scale = V::splat(scale);
    let mut tmp = [0.0f32; TOKENS];
    for t0 in (0..out.len()).step_by(V::N) {
        // l[k]: lane k of the dots of the tokens t0 to t0 + N - 1.
        let mut l = [V::zero(); 8];
        let block = &kt[(t0 / TOKENS) * hd * TOKENS + t0 % TOKENS..];
        let term = |i: usize| V::splat(q[i]).mul(V::at(block, i * TOKENS));
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
        j.mul(scale).put(&mut tmp, 0);
        let n = V::N.min(out.len() - t0);
        out[t0..t0 + n].copy_from_slice(&tmp[..n]);
    }
}

/// `o[g][d] = sum over t of p[g][t] * v[t * hd + d]` for `G` queries, the terms in the order of t, from 0. Each
/// value is its own sum. A pass over the tokens computes `B` vectors of values for each of the `G` queries, so it
/// keeps `G * B` sums apart and the adds do not wait on each other; then passes of 1 vector, then single values.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn weighted_sum<V: Vf, const G: usize, const B: usize>(p: [&[f32]; G], v: &[f32], o: &mut [&mut [f32]; G]) {
    let hd = o[0].len();
    let n = V::N;
    let mut d0 = 0;
    while d0 + B * n <= hd {
        let mut acc = [[V::zero(); B]; G];
        for t in 0..p[0].len() {
            let vs: [V; B] = std::array::from_fn(|j| V::at(v, t * hd + d0 + n * j));
            for (g, a) in acc.iter_mut().enumerate() {
                let w = V::splat(p[g][t]);
                for (aj, vj) in a.iter_mut().zip(&vs) {
                    *aj = aj.add(w.mul(*vj));
                }
            }
        }
        for (g, a) in acc.iter().enumerate() {
            for (j, aj) in a.iter().enumerate() {
                aj.put(o[g], d0 + n * j);
            }
        }
        d0 += B * n;
    }
    while d0 + n <= hd {
        let mut acc = [V::zero(); G];
        for t in 0..p[0].len() {
            let vt = V::at(v, t * hd + d0);
            for (g, a) in acc.iter_mut().enumerate() {
                *a = a.add(V::splat(p[g][t]).mul(vt));
            }
        }
        for (g, a) in acc.iter().enumerate() {
            a.put(o[g], d0);
        }
        d0 += n;
    }
    for d in d0..hd {
        for g in 0..G {
            let mut acc = 0.0f32;
            for (t, &w) in p[g].iter().enumerate() {
                acc += w * v[t * hd + d];
            }
            o[g][d] = acc;
        }
    }
}

/// The queries that `attention` takes at once in `weighted_sum`, and the vectors of values of each pass.
const QUERIES: usize = 4;
const PASS: usize = 2;

/// Rotates the pairs of `src` by the angles of `cs` (cos, sin), into `dst[j * step]` for value `j`.
fn rotate(src: &[f32], cs: &[(f32, f32)], dst: &mut [f32], step: usize) {
    for (i, &(c, s)) in cs.iter().enumerate() {
        let (x0, x1) = (src[2 * i], src[2 * i + 1]);
        dst[2 * i * step] = x0 * c + -x1 * s;
        dst[(2 * i + 1) * step] = x1 * c + x0 * s;
    }
}

/// The inputs of the workers of `attention`.
struct Attend<'a> {
    qkv: &'a [f32],
    gates: &'a [f32],
    /// cos and sin of `pos * freqs[i]` at `table[pos * hd / 2 + i]`.
    table: &'a [(f32, f32)],
    b: usize,
    along: Axis,
    len: usize,
    heads: usize,
    hd: usize,
    scale: f32,
}

impl Attend<'_> {
    /// The row of the token at `pos` in sequence `s`.
    fn token(&self, s: usize, pos: usize) -> usize {
        match self.along {
            Axis::A => pos * self.b + s,
            Axis::B => s * self.b + pos,
        }
    }
}

leveled! {
    /// The rows of one worker of `attention`, from row `first` in the order (sequence, head, position).
    fn attend(job: &Attend, first: usize, chunk: &mut [f32]) = attend_of;
}

#[inline(always)]
fn attend_of<I: Isa>(job: &Attend, first: usize, chunk: &mut [f32]) {
    let (len, heads, hd) = (job.len, job.heads, job.hd);
    let width = heads * hd;
    let half = hd / 2;
    let mut kt = vec![0.0f32; hd * len.next_multiple_of(TOKENS)];
    let mut v = vec![0.0f32; len * hd];
    let mut q = vec![0.0f32; hd];
    let mut p = vec![0.0f32; QUERIES * len];
    let mut loaded = usize::MAX;
    let rows = chunk.len() / hd;
    let mut n = 0;
    while n < rows {
        let unit = (first + n) / len;
        let pos = (first + n) % len;
        let (s, h) = (unit / heads, unit % heads);
        if unit != loaded {
            for t in 0..len {
                let row = &job.qkv[job.token(s, t) * 3 * width..][..3 * width];
                rotate(
                    &row[width + h * hd..][..hd],
                    &job.table[t * half..][..half],
                    &mut kt[(t / TOKENS) * hd * TOKENS + t % TOKENS..],
                    TOKENS,
                );
                v[t * hd..][..hd].copy_from_slice(&row[2 * width + h * hd..][..hd]);
            }
            loaded = unit;
        }
        // The queries of this group: in one (sequence, head), and in this chunk.
        let group = QUERIES.min(rows - n).min(len - pos);
        for (g, pg) in p.chunks_exact_mut(len).take(group).enumerate() {
            let r = job.token(s, pos + g);
            rotate(
                &job.qkv[r * 3 * width + h * hd..][..hd],
                &job.table[(pos + g) * half..][..half],
                &mut q,
                1,
            );
            scores::<I::V>(&q, &kt, job.scale, pg);
            let max = pg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            exps_of::<I>(pg, max);
            let total = sum(pg);
            for x in pg.iter_mut() {
                *x /= total;
            }
        }
        let os = &mut chunk[n * hd..(n + group) * hd];
        if group == QUERIES {
            let mut o: [&mut [f32]; QUERIES] = {
                let mut it = os.chunks_exact_mut(hd);
                std::array::from_fn(|_| it.next().unwrap())
            };
            let ps: [&[f32]; QUERIES] = std::array::from_fn(|g| &p[g * len..][..len]);
            weighted_sum::<I::V, QUERIES, PASS>(ps, &v, &mut o);
        } else {
            for (g, o) in os.chunks_exact_mut(hd).enumerate() {
                weighted_sum::<I::V, 1, 8>([&p[g * len..][..len]], &v, &mut [o]);
            }
        }
        for (g, o) in os.chunks_exact_mut(hd).enumerate() {
            let r = job.token(s, pos + g);
            let gate = 1.0 / (1.0 + libm::expf(-job.gates[r * heads + h]));
            for x in o.iter_mut() {
                *x *= gate;
            }
        }
        n += group;
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
    // cos and sin of `pos * freqs[i]` at `table[pos * hd / 2 + i]`, as the rotation of each row needs them.
    let half = hd / 2;
    let table: Vec<(f32, f32)> = (0..len * half)
        .map(|n| {
            let angle = (n / half) as f32 * freqs[n % half];
            (libm::cosf(angle), libm::sinf(angle))
        })
        .collect();
    let job = Attend {
        qkv,
        gates,
        table: &table,
        b,
        along,
        len,
        heads,
        hd,
        scale: 1.0 / (hd as f32).sqrt(),
    };
    let level = isa::level();
    // The rows in the order (sequence, head, position), `hd` values each.
    let mut ordered = vec![0.0f32; rows * width];
    split(&mut ordered, hd, threads, |first, chunk| {
        attend(level, &job, first, chunk)
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
