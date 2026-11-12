//! The MoE block and its router. It ports `test_moe.py` and `test_moe_route.py` of mk.
//!
//! - Route: the router's top-10 and the grouping stage (`mk_moe_route`) give exact answers. A fixture of
//!   `tools/op_groups/moe.py` holds SHA-256 digests of the integer answers and the f32 weights.
//! - Block: `mk_moe` against a float64 reference in the same fixture, on two synthetic layers. mk compared against the HF
//!   module with the real weights of layers 0 and 3. The HF module needs many GB of dense expert weights, so the
//!   reference here is numpy with the maths of that module. The tolerances are the ones of mk.
//! - Block: the same bits for every grid size and over repeats, and with the expert lookahead off (`libmk_moe_nolook`).
//!
//! The synthetic pools of a layer have the byte sizes of the real tensors: 512 experts of IQ3_S gate and up (640 rows of
//! K = 2560), IQ4_NL down (2560 rows of K = 640), Q8_0 shared experts, and an F32 router. A pool takes about 1.2 GB.

use crate::fixture::Fixture;
use crate::inputs::{self, Qtype};
use crate::ops_lib::{self, assert_same, lib, nbytes, ptr, ptr_mut, same_bits};
use crate::sym;
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::sync::LazyLock;

const E: usize = 512;
const K: usize = 10;
const ROUTE_TS: [usize; 8] = [1, 7, 16, 17, 80, 81, 160, 256];

/// What `mk_moe_route` gives for one call.
struct Route {
    idx: Vec<i32>,
    w: Vec<f32>,
    cnt: Vec<u32>,
    off: Vec<u32>,
    asg: Vec<u32>,
    tok: Vec<u32>,
    igu: Vec<u32>,
    idn: Vec<u32>,
}

fn route(logits: &[f32], fast: u32, grid: u32) -> Route {
    let t = logits.len() / E;
    let mut r = Route {
        idx: vec![0; t * K],
        w: vec![0.0; t * K],
        cnt: vec![0; E],
        off: vec![0; E + 1],
        asg: vec![0; t * K],
        tok: vec![0; t * K],
        igu: vec![0; E + 1],
        idn: vec![0; E + 1],
    };
    let f = sym!(
        lib("moe"),
        "mk_moe_route",
        fn(
            *const c_void,
            u32,
            u32,
            u32,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> i32
    );
    let rc = unsafe {
        f(
            ptr(logits),
            t as u32,
            fast,
            grid,
            ptr_mut(&mut r.idx),
            ptr_mut(&mut r.w),
            ptr_mut(&mut r.cnt),
            ptr_mut(&mut r.off),
            ptr_mut(&mut r.asg),
            ptr_mut(&mut r.tok),
            ptr_mut(&mut r.igu),
            ptr_mut(&mut r.idn),
        )
    };
    assert_eq!(rc, 0, "mk_moe_route");
    r
}

fn digest<T: Copy>(values: &[T]) -> String {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) };
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Logits on a grid of 1/8 with many exact ties: about 49 levels over 512 experts.
fn grid_logits(t: usize, seed: u64) -> Vec<f32> {
    inputs::integers(seed, t * E, 49)
        .into_iter()
        .map(|v| (i64::from(v) - 24) as f32 / 8.0)
        .collect()
}

/// Popular experts, like real routing: a few experts get most tokens, in runs longer than a 16-token tile.
fn skewed_logits(t: usize, seed: u64) -> Vec<f32> {
    let mut lg = grid_logits(t, seed);
    for row in lg.chunks_mut(E) {
        for v in &mut row[..12] {
            *v += 4.0;
        }
    }
    lg
}

fn logits_of(maker: &str, t: usize) -> Vec<f32> {
    match maker {
        "grid" => grid_logits(t, t as u64),
        "skewed" => skewed_logits(t, 1000 + t as u64),
        _ => panic!("no logit maker {maker}"),
    }
}

#[test]
fn route_is_exact() {
    let fixture = Fixture::load("moe");
    let cases = fixture.meta("route_cases").as_array().unwrap().clone();
    assert_eq!(
        cases.len(),
        2 * ROUTE_TS.len(),
        "the fixture holds each maker and each token count"
    );
    for case in cases {
        let maker = case["maker"].as_str().unwrap();
        let t = case["t"].as_u64().unwrap() as usize;
        let want = &case["digest"];
        let want_w = fixture.f32_flat(&format!("route_w_{maker}_{t}"));
        let logits = logits_of(maker, t);
        for fast in [0u32, 1] {
            let got = route(&logits, fast, 0);
            let tag = format!("{maker} T={t} fast={fast}");
            assert_eq!(
                digest(&got.idx),
                want["idx"],
                "{tag}: top-10 experts (probability desc, index asc)"
            );
            let worst = got
                .w
                .iter()
                .zip(&want_w)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs() / f64::from(*b))
                .fold(0.0, f64::max);
            assert!(worst <= 1e-6, "{tag}: weights: max relative error {worst:.2e}");
            assert_eq!(digest(&got.cnt), want["cnt"], "{tag}: count");
            assert_eq!(digest(&got.off), want["off"], "{tag}: offset");
            assert_eq!(digest(&got.asg), want["asg"], "{tag}: assign");
            assert_eq!(digest(&got.tok), want["tok"], "{tag}: token");
            assert_eq!(digest(&got.igu), want[format!("igu{fast}")], "{tag}: item_gu");
            assert_eq!(digest(&got.idn), want[format!("idn{fast}")], "{tag}: item_dn");
        }
    }
}

#[test]
fn all_equal_logits_pick_the_lowest_ten() {
    let logits = vec![0f32; 5 * E];
    let r = route(&logits, 1, 0);
    assert!(
        r.idx.chunks(K).all(|row| row.iter().copied().eq(0..K as i32)),
        "{:?}",
        r.idx
    );
    assert!(r.w.iter().all(|w| *w == 0.1f32), "{:?}", r.w);
}

#[test]
fn route_is_the_same_for_every_grid() {
    let logits = skewed_logits(256, 7);
    for fast in [0u32, 1] {
        let reference = route(&logits, fast, 0);
        for grid in [1u32, 3, 80] {
            let got = route(&logits, fast, grid);
            let pairs: [(&str, bool); 8] = [
                ("idx", digest(&got.idx) == digest(&reference.idx)),
                ("w", same_bits(&got.w, &reference.w)),
                ("cnt", got.cnt == reference.cnt),
                ("off", got.off == reference.off),
                ("asg", got.asg == reference.asg),
                ("tok", got.tok == reference.tok),
                ("igu", got.igu == reference.igu),
                ("idn", got.idn == reference.idn),
            ];
            for (name, same) in pairs {
                assert!(same, "fast={fast} grid {grid}: {name} differs");
            }
        }
    }
}

// ---- the block ----

/// The pool of one synthetic layer. The seeds are `base` + 1 to 9, and `tools/op_groups/moe.py` has the same list.
struct Pool {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    sh_gate: Vec<u8>,
    sh_up: Vec<u8>,
    sh_down: Vec<u8>,
    router: Vec<f32>,
    shexp_gate: Vec<f32>,
    base: u64,
}

impl Pool {
    fn new(base: u64) -> Pool {
        Pool {
            gate: inputs::raw_rows(Qtype::Iq3S, base + 1, E * 640, 2560),
            up: inputs::raw_rows(Qtype::Iq3S, base + 2, E * 640, 2560),
            down: inputs::raw_rows(Qtype::Iq4Nl, base + 3, E * 2560, 640),
            sh_gate: inputs::raw_rows(Qtype::Q8_0, base + 4, 640, 2560),
            sh_up: inputs::raw_rows(Qtype::Q8_0, base + 5, 640, 2560),
            sh_down: inputs::raw_rows(Qtype::Q8_0, base + 6, 2560, 640),
            router: inputs::normals_scaled(base + 7, E * 2560, 0.02),
            shexp_gate: inputs::normals_scaled(base + 8, 2560, 0.02),
            base,
        }
    }

    fn x(&self, t: usize) -> Vec<f32> {
        inputs::normals(self.base + 9, t * 2560)
    }
}

struct BlockOut {
    out: Vec<f32>,
    idx: Vec<i32>,
}

/// `mk_moe` on `x` of `t` tokens, from the library `moe` (lookahead on) or `moe_nolook`.
fn block(pool: &Pool, x: &[f32], grid: u32, fast: u32, look: bool) -> BlockOut {
    let t = x.len() / 2560;
    let mut out = vec![0f32; t * 2560];
    let mut idx = vec![0i32; t * K];
    let mut w = vec![0f32; t * K];
    let f = sym!(
        lib(if look { "moe" } else { "moe_nolook" }),
        "mk_moe",
        fn(
            *const c_void,
            u64,
            u32,
            *const c_void,
            u64,
            *const c_void,
            u64,
            u32,
            *const c_void,
            u64,
            u32,
            *const c_void,
            u64,
            u32,
            *const c_void,
            u64,
            *const c_void,
            u64,
            *const c_void,
            u64,
            *const c_void,
            u64,
            *mut c_void,
            u64,
            *mut c_void,
            u64,
            *mut c_void,
            u64,
            u32,
            u32,
        ) -> i32
    );
    let rc = unsafe {
        f(
            ptr(x),
            nbytes(x),
            t as u32,
            ptr(&pool.router),
            nbytes(&pool.router),
            ptr(&pool.shexp_gate),
            nbytes(&pool.shexp_gate),
            Qtype::Iq3S as u32,
            ptr(&pool.gate),
            nbytes(&pool.gate),
            Qtype::Iq3S as u32,
            ptr(&pool.up),
            nbytes(&pool.up),
            Qtype::Iq4Nl as u32,
            ptr(&pool.down),
            nbytes(&pool.down),
            ptr(&pool.sh_gate),
            nbytes(&pool.sh_gate),
            ptr(&pool.sh_up),
            nbytes(&pool.sh_up),
            ptr(&pool.sh_down),
            nbytes(&pool.sh_down),
            ptr_mut(&mut out),
            nbytes(&out),
            ptr_mut(&mut idx),
            nbytes(&idx),
            ptr_mut(&mut w),
            nbytes(&w),
            grid,
            fast,
        )
    };
    assert_eq!(rc, 0, "mk_moe");
    BlockOut { out, idx }
}

const TOL: f64 = 1e-4;
const TOL_FAST: f64 = 2e-3;
const NEAR_TIE: f64 = 1e-5;

#[test]
fn moe_matches_reference() {
    let fixture = Fixture::load("moe");
    let t = fixture.meta("block_t").as_u64().unwrap() as usize;
    for name in ["a", "b"] {
        let base = fixture.meta("pool_bases")[name].as_u64().unwrap();
        let pool = Pool::new(base);
        let (ref_idx, _) = fixture.i32s(&format!("block_idx_{name}"));
        let p10 = fixture.f32_flat(&format!("block_p10_{name}"));
        let p11 = fixture.f32_flat(&format!("block_p11_{name}"));
        let want = fixture.f32_flat(&format!("block_out_{name}"));
        let x = pool.x(t);
        for fast in [0u32, 1] {
            let got = block(&pool, &x, 0, fast, true);
            let mut ok = Vec::new();
            for token in 0..t {
                let mut a: Vec<i32> = got.idx[token * K..(token + 1) * K].to_vec();
                let mut b: Vec<i32> = ref_idx[token * K..(token + 1) * K].to_vec();
                a.sort_unstable();
                b.sort_unstable();
                if a == b {
                    ok.push(token);
                    continue;
                }
                let gap = f64::from(p10[token] - p11[token]) / f64::from(p10[token]);
                assert!(
                    gap < NEAR_TIE,
                    "pool {name} token {token}: router picked {a:?}, reference {b:?} (not a near-tie: {gap})"
                );
                eprintln!("pool {name} token {token}: near-tie swap, excluded");
            }
            assert!(
                ok.len() >= t - 1,
                "pool {name}: only {} of {t} tokens have the reference's experts",
                ok.len()
            );
            let rows =
                |v: &[f32]| -> Vec<f32> { ok.iter().flat_map(|&r| v[r * 2560..(r + 1) * 2560].to_vec()).collect() };
            let e = ops_lib::relerr(&rows(&got.out), &rows(&want));
            let tol = if fast == 1 { TOL_FAST } else { TOL };
            eprintln!(
                "pool {name} fast={fast} MoE relerr {e:.2e} over {} tokens (tol {tol:.0e})",
                ok.len()
            );
            assert!(e < tol, "pool {name} fast={fast}: relerr {e:.3e} >= {tol:.0e}");
        }
    }
}

static POOL_A: LazyLock<Pool> = LazyLock::new(|| Pool::new(7000));

#[test]
fn moe_grid_independent() {
    let pool = &*POOL_A;
    let x = inputs::normals(7500, 19 * 2560);
    for fast in [0u32, 1] {
        let reference = block(pool, &x, 0, fast, true);
        for grid in [1u32, 13, 0] {
            let got = block(pool, &x, grid, fast, true);
            assert_same(&got.out, &reference.out, &format!("fast={fast} grid {grid}"));
            assert_eq!(got.idx, reference.idx, "fast={fast} grid {grid}: experts");
        }
        let mut last = None;
        for repeat in 0..5 {
            let got = block(pool, &x, 0, fast, true);
            assert_same(&got.out, &reference.out, &format!("fast={fast} repeat {repeat}"));
            last = Some(got);
        }
        if fast == 1 {
            // The f16 path is really taken.
            let exact = block(pool, &x, 0, 0, true);
            assert!(!same_bits(&last.unwrap().out, &exact.out), "fast=1 ran the exact path");
        }
    }
}

/// The one-step lookahead of the expert item only moves loads. The build without it (`libmk_moe_nolook`, the path of
/// `k_stage`) gives the same bits as the one with it, on the f16 path, for a decode step, a partial token block and a pass.
#[test]
fn moe_lookahead_off_same_bits() {
    let pool = &*POOL_A;
    for t in [1usize, 5, 19] {
        let x = inputs::normals(7600 + t as u64, t * 2560);
        let a = block(pool, &x, 0, 1, true);
        let b = block(pool, &x, 0, 1, false);
        assert_eq!(a.idx, b.idx, "T={t}: experts");
        assert_same(&a.out, &b.out, &format!("T={t}"));
    }
}
