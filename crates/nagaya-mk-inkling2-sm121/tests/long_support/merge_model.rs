//! The host model of the merge order of `OP_ATTN`: the oracle of the order of its sums.
//!
//! The kernel cuts the keys of a row into chunks of 256. Each chunk gives a partial state (maximum, sum, output). The
//! partials merge in chunk order from the state (sink, 1, 0). This model does the same in f32, to tell which part of the
//! arithmetic makes the error grow with the depth. Measured on the seeded data of `cuda_long_attn.rs` (error against a
//! form with f64 sums, of the row maximum, 24 heads, one decode row):
//!
//! | positions | serial f32 merge | tree merge | serial double-float merge |
//! |---|---|---|---|
//! | 65,536 | 1.9e-6 | 3.2e-7 | 2.6e-7 |
//! | 131,072 | 3.7e-6 | 2.8e-7 | 2.3e-7 |
//! | 409,600 | 1.2e-5 | 2.8e-7 | 2.8e-7 |
//! | 524,288 | 2.1e-5 | 1.9e-7 | 1.8e-7 |
//!
//! The sums inside a chunk in f32 (steps of 16 keys, the sum of a step as a tree over 16 lanes) add less than 1e-7. The
//! serial f32 merge of the 2,048 partials at 524,288 positions makes the drift: each step rounds the running sum, and
//! the rounding errors of positions that repeat do not cancel. The kernel uses the serial double-float merge
//! (`serial_df`): its state is a sum of two f32 words, and the bits depend only on the chunk indices.

#![allow(dead_code)]
#![cfg(feature = "cuda")]

use nagaya_mk_cuda::attn::HEAD_DIM;

pub const CHUNK: usize = 256;
pub const STEP: usize = 16;

#[derive(Clone)]
pub struct St {
    pub m: f32,
    pub l: f32,
    pub o: Vec<f32>,
}

pub fn combine(x: &St, y: &St) -> St {
    if y.l <= 0.0 {
        return x.clone();
    }
    let nm = x.m.max(y.m);
    let a = (x.m - nm).exp();
    let b = (y.m - nm).exp();
    St {
        m: nm,
        l: x.l * a + y.l * b,
        o: x.o.iter().zip(&y.o).map(|(p, q)| p * a + q * b).collect(),
    }
}

pub fn combine64(x: &(f64, f64, Vec<f64>), y: &St) -> (f64, f64, Vec<f64>) {
    if y.l <= 0.0 {
        return x.clone();
    }
    let nm = x.0.max(y.m as f64);
    let a = (x.0 - nm).exp();
    let b = (y.m as f64 - nm).exp();
    (
        nm,
        x.1 * a + y.l as f64 * b,
        x.2.iter().zip(&y.o).map(|(p, q)| p * a + *q as f64 * b).collect(),
    )
}

/// The chunk partial in f32, in the order of the kernel (steps of 16 keys).
pub fn chunk_f32(s: &[f32], vals: &dyn Fn(usize) -> &'static [f32], lo: usize) -> St {
    let mut m_run = f32::NEG_INFINITY;
    let mut l_run = 0f32;
    let mut o = vec![0f32; HEAD_DIM];
    for (si, sc) in s.chunks(STEP).enumerate() {
        let tile_max = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let m_new = m_run.max(tile_max);
        let alpha = if m_new == f32::NEG_INFINITY {
            1.0
        } else {
            (m_run - m_new).exp()
        };
        let p: Vec<f32> = sc.iter().map(|x| (x - m_new).exp()).collect();
        // xor tree over 16 lanes: adjacent pairs first
        let mut t = p.clone();
        t.resize(16, 0.0);
        let mut w = 1;
        while w < 16 {
            let u: Vec<f32> = (0..16).map(|i| t[i] + t[i ^ w]).collect();
            t = u;
            w <<= 1;
        }
        l_run = l_run * alpha + t[0];
        m_run = m_new;
        for (d, od) in o.iter_mut().enumerate() {
            let mut acc = 0f64;
            for (k, pk) in p.iter().enumerate() {
                acc += *pk as f64 * vals(lo + si * STEP + k)[d] as f64;
            }
            *od = (*od * alpha) as f64 as f32 + acc as f32;
            let _ = d;
        }
    }
    St { m: m_run, l: l_run, o }
}

pub fn chunk_f64(s: &[f32], vals: &dyn Fn(usize) -> &'static [f32], lo: usize) -> St {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut l = 0f64;
    let mut o = vec![0f64; HEAD_DIM];
    for (k, x) in s.iter().enumerate() {
        let p = ((*x - m) as f64).exp();
        l += p;
        for (d, od) in o.iter_mut().enumerate() {
            *od += p * vals(lo + k)[d] as f64;
        }
    }
    St {
        m,
        l: l as f32,
        o: o.iter().map(|v| *v as f32).collect(),
    }
}

/// Double-float state: value = hi + lo.
pub fn df_scale_add(h: f32, l: f32, a: f32, t: f32, te: f32) -> (f32, f32) {
    let ph = h * a;
    let pe = h.mul_add(a, -ph);
    let pl = l * a + pe;
    // two-sum of ph and t
    let s = ph + t;
    let bb = s - ph;
    let err = (ph - (s - bb)) + (t - bb);
    let lo = pl + err + te;
    let hh = s + lo;
    (hh, lo - (hh - s))
}

pub fn serial_df(sink_st: &St, ps: &[St]) -> St {
    let (mut m, mut lh, mut ll) = (sink_st.m, 1f32, 0f32);
    let mut oh = vec![0f32; HEAD_DIM];
    let mut ol = vec![0f32; HEAD_DIM];
    for p in ps {
        if p.l <= 0.0 {
            continue;
        }
        let nm = m.max(p.m);
        let a = (m - nm).exp();
        let b = (p.m - nm).exp();
        let t = p.l * b;
        let te = p.l.mul_add(b, -t);
        (lh, ll) = df_scale_add(lh, ll, a, t, te);
        for d in 0..HEAD_DIM {
            let t = p.o[d] * b;
            let te = p.o[d].mul_add(b, -t);
            (oh[d], ol[d]) = df_scale_add(oh[d], ol[d], a, t, te);
        }
        m = nm;
    }
    St {
        m,
        l: lh + ll,
        o: oh.iter().zip(&ol).map(|(h, l)| h + l).collect(),
    }
}

pub fn tree(leaves: Vec<St>) -> St {
    let mut v = leaves;
    while v.len() > 1 {
        let mut n = Vec::new();
        for c in v.chunks(2) {
            n.push(if c.len() == 2 {
                combine(&c[0], &c[1])
            } else {
                c[0].clone()
            });
        }
        v = n;
    }
    v.pop().unwrap()
}
