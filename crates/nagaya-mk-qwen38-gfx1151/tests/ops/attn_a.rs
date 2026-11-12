//! The block selection of attention, its attend stage, and the attention cell mask. It ports `test_attn_select.py`,
//! `test_attn_split.py` and `test_attn_mask.py` of mk.
//!
//! - select: the parallel selection (`attn.h` `attn_select_emit`) equals the serial selection that it replaced
//!   (`attn_select_oracle.h`) bit for bit, and both equal the statement of the definition, which is ported to Rust below.
//! - split: stage 3 alone (`mk_attn_attend`), against its definition at q = 0 and against its numeric bound in f64.
//! - mask: `mk_attn_masked` against the f64 reference of mk, which `tools/op_groups/attn_a.py` runs for the fixture.
//!
//! The inputs come from the stateless generator in `inputs.rs`. mk used numpy's `default_rng`, so the values differ, and the
//! shapes and the meanings of the inputs are the same.

use crate::fixture::Fixture;
use crate::inputs;
use crate::ops_lib::{self, all_finite, nbytes, ptr, ptr_mut, relerr, same_bits};
use crate::sym;
use sha2::{Digest, Sha256};
use std::ffi::c_void;

const RATIO: usize = 4;

// ---- test_attn_select.py -----------------------------------------------------------------------------------------------

const BUDGET: usize = 2048;
const TOPK: usize = BUDGET / RATIO;
const MAX_SEL: usize = BUDGET + RATIO - 1;

/// Which selection to run.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Select {
    Serial,
    Parallel,
}

fn select_run(which: Select, t: usize, p0: usize, scores: &[f32], grid: u32) -> (Vec<i32>, Vec<u32>) {
    let max_nb = (p0 + t) / RATIO;
    let zeros = [0f32; 1];
    let scores: &[f32] = if max_nb == 0 { &zeros } else { scores };
    assert!(max_nb == 0 || scores.len() == t * max_nb);
    let mut sel = vec![0i32; t * MAX_SEL];
    let mut cnt = vec![0u32; t];
    let rc = match which {
        Select::Serial => {
            let f = sym!(
                ops_lib::lib("attn_select_oracle"),
                "mk_attn_select_serial",
                fn(u32, u32, u32, u32, *const c_void, *mut c_void, *mut c_void, u32) -> i32
            );
            unsafe {
                f(
                    t as u32,
                    p0 as u32,
                    BUDGET as u32,
                    RATIO as u32,
                    ptr(scores),
                    ptr_mut(&mut sel),
                    ptr_mut(&mut cnt),
                    grid,
                )
            }
        }
        Select::Parallel => {
            let f = sym!(
                ops_lib::lib("attn"),
                "mk_attn_select_emit",
                fn(u32, u32, u32, u32, *const c_void, *mut c_void, *mut c_void, u32) -> i32
            );
            unsafe {
                f(
                    t as u32,
                    p0 as u32,
                    BUDGET as u32,
                    RATIO as u32,
                    ptr(scores),
                    ptr_mut(&mut sel),
                    ptr_mut(&mut cnt),
                    grid,
                )
            }
        }
    };
    assert_eq!(rc, 0, "{which:?} selection");
    (sel, cnt)
}

/// The definition of the op, independent of both kernels: the top `select_count` blocks by (score descending, index
/// ascending), a strict total order, in ascending block order with `RATIO` cells each, then the tail cells, then -1.
fn select_definition(t: usize, p0: usize, scores: &[f32]) -> (Vec<i32>, Vec<u32>) {
    let max_nb = (p0 + t) / RATIO;
    let mut sel = vec![-1i32; t * MAX_SEL];
    let mut cnt = vec![0u32; t];
    for i in 0..t {
        let n_vis = p0 + i + 1;
        let nb = n_vis / RATIO;
        let row = &scores[i * max_nb..i * max_nb + nb];
        let mut order: Vec<usize> = (0..nb).collect();
        // A tie, including +0 and -0, goes to the lower index: the sort is stable and the comparison reports equality.
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).expect("a score is not NaN"));
        let mut blocks: Vec<usize> = order[..nb.min(TOPK)].to_vec();
        blocks.sort_unstable();
        let cells: Vec<i32> = blocks
            .iter()
            .flat_map(|&b| (0..RATIO).map(move |c| (b * RATIO + c) as i32))
            .chain((nb * RATIO..n_vis).map(|c| c as i32))
            .collect();
        sel[i * MAX_SEL..i * MAX_SEL + cells.len()].copy_from_slice(&cells);
        cnt[i] = cells.len() as u32;
    }
    (sel, cnt)
}

const KINDS: [&str; 9] = [
    "random",
    "levels3",
    "levels5_rare_top",
    "equal",
    "zeros",
    "signed_zeros",
    "one_huge",
    "inf",
    "denormal",
];

/// The block scores of one kind, aimed at what a selection gets wrong: ties, one huge score, +inf, and tiny keys.
fn make_scores(kind: &str, t: usize, p0: usize, seed: u64) -> Vec<f32> {
    let n = t * ((p0 + t) / RATIO);
    let u = inputs::uniforms(seed, n);
    match kind {
        "random" => u.iter().map(|v| v * 3.0).collect(),
        // Heavy ties: the threshold falls inside a tie group.
        "levels3" => inputs::integers(seed, n, 3).iter().map(|&v| v as f32 * 0.5).collect(),
        "levels5_rare_top" => u
            .iter()
            .map(|&v| match v {
                v if v < 0.3 => 0.0,
                v if v < 0.6 => 0.25,
                v if v < 0.9 => 1.0,
                v if v < 0.99 => 2.0,
                _ => 7.0,
            })
            .collect(),
        "equal" => vec![1.375; n],
        "zeros" => vec![0.0; n],
        // -0.0 equals +0.0, so it is a tie, and the index breaks it, as the serial selection compares floats.
        "signed_zeros" => {
            let rare = inputs::uniforms(seed + 1, n);
            (0..n)
                .map(|i| {
                    if rare[i] < 0.02 {
                        1.0
                    } else if u[i] < 0.5 {
                        -0.0
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        "one_huge" => {
            let mut s = u;
            if n > 0 {
                for at in inputs::integers(seed + 1, t.max(1), n as u64) {
                    s[at as usize] = 3e38;
                }
            }
            s
        }
        "inf" => {
            let rare = inputs::uniforms(seed + 1, n);
            inputs::integers(seed, n, 4)
                .iter()
                .enumerate()
                .map(|(i, &v)| if rare[i] < 0.1 { f32::INFINITY } else { v as f32 })
                .collect()
        }
        // Tiny keys: every decision is in the low digits of the radix sort.
        "denormal" => inputs::integers(seed, n, 40)
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect(),
        other => panic!("no score kind {other}"),
    }
}

/// (T, p0): nb = (p0 + i + 1) / 4 for token i.
const SHAPES: [(usize, usize); 11] = [
    (1, 0),
    (4, 0),
    (16, 3),    // nb 0..4, dense
    (16, 2040), // nb 510..514: across the dense/sparse boundary
    (1, 2043),
    (1, 2047),
    (1, 2051), // nb 511, 512, 513 exactly
    (3, 2300),
    (16, 4090),
    (2, 8191),  // sparse, a few chunks
    (1, 32767), // nb 8192 (the 32k context)
];
/// The shape (16, 32752) also has nb 8192. It is the last shape.
const SHAPE_LAST: (usize, usize) = (16, 32752);
const GRIDS: [u32; 4] = [0, 1, 3, 80];

fn shapes() -> Vec<(usize, usize)> {
    let mut v = SHAPES.to_vec();
    v.push(SHAPE_LAST);
    v
}

fn seed_of(t: usize, p0: usize, kind: usize) -> u64 {
    1_000_003 * t as u64 + 7 * p0 as u64 + 101 * kind as u64 + 5
}

fn matches_serial(kind_index: usize) {
    let kind = KINDS[kind_index];
    for (t, p0) in shapes() {
        let s = make_scores(kind, t, p0, seed_of(t, p0, kind_index));
        let (want_sel, want_cnt) = select_run(Select::Serial, t, p0, &s, 0);
        let grids: &[u32] = if t > 1 { &GRIDS } else { &[0, 1] };
        for &grid in grids {
            let (sel, cnt) = select_run(Select::Parallel, t, p0, &s, grid);
            assert_eq!(
                cnt, want_cnt,
                "{kind} T={t} p0={p0} grid={grid}: sel_count differs from the serial"
            );
            let bad: Vec<usize> = (0..sel.len()).filter(|&i| sel[i] != want_sel[i]).collect();
            assert!(
                bad.is_empty(),
                "{kind} T={t} p0={p0} grid={grid}: {} sel cells differ from the serial, first at [tok {}, slot {}]: {} != {}",
                bad.len(),
                bad[0] / MAX_SEL,
                bad[0] % MAX_SEL,
                sel[bad[0]],
                want_sel[bad[0]]
            );
        }
    }
}

macro_rules! serial_tests {
    ($($name:ident: $index:expr),*) => {
        $(#[test] fn $name() { matches_serial($index); })*
    };
}

serial_tests!(
    select_matches_serial_random: 0,
    select_matches_serial_levels3: 1,
    select_matches_serial_levels5_rare_top: 2,
    select_matches_serial_equal: 3,
    select_matches_serial_zeros: 4,
    select_matches_serial_signed_zeros: 5,
    select_matches_serial_one_huge: 6,
    select_matches_serial_inf: 7,
    select_matches_serial_denormal: 8
);

/// The serial oracle and the parallel op both equal the definition.
#[test]
fn select_matches_definition() {
    for (t, p0) in [(16, 2040), (2, 8191), (16, 32752)] {
        for kind in ["random", "levels3", "signed_zeros", "inf"] {
            let index = KINDS.iter().position(|k| *k == kind).unwrap();
            let s = make_scores(kind, t, p0, seed_of(t, p0, index) + 77);
            let (want_sel, want_cnt) = select_definition(t, p0, &s);
            for which in [Select::Serial, Select::Parallel] {
                let (sel, cnt) = select_run(which, t, p0, &s, 0);
                assert_eq!(cnt, want_cnt, "{which:?} {kind}: sel_count differs from the definition");
                assert!(
                    sel == want_sel,
                    "{which:?} {kind} T={t} p0={p0}: sel differs from the definition"
                );
            }
        }
    }
}

// ---- test_attn_split.py ------------------------------------------------------------------------------------------------

const U: f64 = 1.0 / 16_777_216.0;
const H: usize = 24;
const KVH: usize = 2;
const HD: usize = 256;
const SLICES: usize = 8;
const NS: [usize; 15] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 100, 229, 1024, 1500, 2051];
const NCELLS: usize = 2200;
const SPLIT_MAX_SEL: usize = 2051;

/// Stage 3 of attention alone: `out [t, 24, 256]`.
fn call_attend(q: &[f32], gate: &[f32], kv: &[f32], sel: &[i32], cnt: &[u32], grid: u32) -> Vec<f32> {
    let t = cnt.len();
    let mut out = vec![f32::NAN; t * H * HD];
    let f = sym!(
        ops_lib::lib("attn"),
        "mk_attn_attend",
        fn(
            u32,
            u32,
            u32,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *mut c_void,
            u32,
        ) -> i32
    );
    let rc = unsafe {
        f(
            t as u32,
            (kv.len() / (2 * KVH * HD)) as u32,
            (sel.len() / t) as u32,
            ptr(q),
            ptr(gate),
            ptr(kv),
            ptr(sel),
            ptr(cnt),
            ptr_mut(&mut out),
            grid,
        )
    };
    assert_eq!(rc, 0, "mk_attn_attend");
    out
}

/// A sorted selection of `n` of the cells for each token, padded with -1.
fn selections(seed: u64, ns: &[usize]) -> (Vec<i32>, Vec<u32>) {
    let mut sel = vec![-1i32; ns.len() * SPLIT_MAX_SEL];
    for (i, &n) in ns.iter().enumerate() {
        let mut cells: Vec<u32> = inputs::permutation(seed + i as u64, NCELLS)[..n].to_vec();
        cells.sort_unstable();
        for (slot, cell) in cells.iter().enumerate() {
            sel[i * SPLIT_MAX_SEL + slot] = *cell as i32;
        }
    }
    (sel, ns.iter().map(|&n| n as u32).collect())
}

/// A kv cache whose V values span many magnitudes: each V value is times 2^e with e in [-12, 24).
fn mixed_kv(seed: u64) -> Vec<f32> {
    let mut kv = inputs::normals(seed, NCELLS * 2 * KVH * HD);
    let exps = inputs::integers(seed + 1, NCELLS * KVH * HD, 36);
    for cell in 0..NCELLS {
        for j in 0..KVH * HD {
            let e = exps[cell * KVH * HD + j] as i32 - 12;
            let scale = f32::from_bits(((e + 127) as u32) << 23);
            kv[(cell * 2 + 1) * KVH * HD + j] *= scale;
        }
    }
    kv
}

fn slices(n: usize) -> Vec<(usize, usize)> {
    (0..SLICES).map(|s| (s * n / SLICES, (s + 1) * n / SLICES)).collect()
}

/// The definition at q = 0 and gate = 100, in exact f32: every add is a separate f32 add. `v` is the selected V rows.
fn emulate_q0(v: &[Vec<f32>]) -> Vec<f32> {
    let n = v.len();
    let mut a = vec![0f32; HD];
    for (lo, hi) in slices(n) {
        if lo == hi {
            continue;
        }
        let mut s = vec![0f32; HD];
        for row in &v[lo..hi] {
            for d in 0..HD {
                s[d] += row[d];
            }
        }
        for d in 0..HD {
            a[d] += s[d];
        }
    }
    a.iter().map(|x| x / n as f32).collect()
}

fn v_row(kv: &[f32], cell: usize, kh: usize) -> Vec<f32> {
    let at = ((cell * 2 + 1) * KVH + kh) * HD;
    kv[at..at + HD].to_vec()
}

/// With q = 0 every score is 0, every exp is 1, and a gate of 100 makes the sigmoid exactly 1. The op then sums each slice
/// of V rows in order, adds the slice sums in slice order, and divides by n. Mixed magnitudes make any other order show.
#[test]
fn split_exact_definition() {
    let kv = mixed_kv(1000);
    let (sel, cnt) = selections(1100, &NS);
    let t = NS.len();
    let q = vec![0f32; t * H * HD];
    let gate = vec![100f32; t * H * HD];
    let out = call_attend(&q, &gate, &kv, &sel, &cnt, 0);
    for (i, &n) in NS.iter().enumerate() {
        for kh in 0..KVH {
            let rows: Vec<Vec<f32>> = (0..n)
                .map(|j| v_row(&kv, sel[i * SPLIT_MAX_SEL + j] as usize, kh))
                .collect();
            let want = emulate_q0(&rows);
            for h in kh * (H / KVH)..(kh + 1) * (H / KVH) {
                let got = &out[(i * H + h) * HD..(i * H + h + 1) * HD];
                let bad: Vec<usize> = (0..HD).filter(|&d| got[d].to_bits() != want[d].to_bits()).collect();
                assert!(
                    bad.is_empty(),
                    "n={n} head {h}: {} elements differ from the definition, first d={} got {:?} want {:?}",
                    bad.len(),
                    bad[0],
                    got[bad[0]],
                    want[bad[0]]
                );
            }
        }
    }
}

struct BoundCase {
    q: Vec<f32>,
    gate: Vec<f32>,
    kv: Vec<f32>,
    sel: Vec<i32>,
    cnt: Vec<u32>,
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

fn kv_at(cell: usize, which: usize, kh: usize) -> usize {
    ((cell * 2 + which) * KVH + kh) * HD
}

fn bound_case(seed: u64, ns: &[usize], qscale: f32, shape: &str) -> BoundCase {
    let t = ns.len();
    let mut kv = inputs::normals(seed, NCELLS * 2 * KVH * HD);
    let q = inputs::normals_scaled(seed + 1, t * H * HD, qscale);
    let (sel, cnt) = selections(seed + 100, ns);
    let qrow = |i: usize, kh: usize| &q[(i * H + kh * 12) * HD..(i * H + kh * 12 + 1) * HD];
    if shape == "rising" {
        // The max changes at almost every step: the worst case of the running rescale. The keys of the selected cells move
        // along the direction of q, in a ramp from -8 to 8 (times 16 over the norm of q).
        for (i, &n) in ns.iter().enumerate() {
            for kh in 0..KVH {
                let norm = dot64(qrow(i, kh), qrow(i, kh)).sqrt();
                let qd: Vec<f64> = qrow(i, kh).iter().map(|&v| f64::from(v) / norm).collect();
                for j in 0..n {
                    let cell = sel[i * SPLIT_MAX_SEL + j] as usize;
                    let at = kv_at(cell, 0, kh);
                    let proj: f64 = (0..HD).map(|d| f64::from(kv[at + d]) * qd[d]).sum();
                    let ramp = if n == 1 {
                        -8.0
                    } else {
                        -8.0 + 16.0 * j as f64 / (n - 1) as f64
                    } * 16.0
                        / norm;
                    for d in 0..HD {
                        kv[at + d] = (f64::from(kv[at + d]) - proj * qd[d] + ramp * qd[d]) as f32;
                    }
                }
            }
        }
    }
    if shape == "spike" {
        // The winning cell is in the last slice: every earlier slice must rescale its partials.
        for (i, &n) in ns.iter().enumerate() {
            let cell = sel[i * SPLIT_MAX_SEL + n - 1] as usize;
            for kh in 0..KVH {
                let qh = qrow(i, kh);
                let scale = 6.0 * 16.0 / dot64(qh, qh);
                for d in 0..HD {
                    kv[kv_at(cell, 0, kh) + d] = (f64::from(qh[d]) * scale) as f32;
                }
            }
        }
    }
    let gate = inputs::normals(seed + 2, t * H * HD);
    BoundCase { q, gate, kv, sel, cnt }
}

/// The op against the exact f64 softmax of its own f32 inputs, within the bound `E (W_d + |o_d|) sig_d` of mk's design.
#[test]
fn split_bound() {
    for (shape, qscale, seed) in [
        ("random", 1.0f32, 2000u64),
        ("random", 4.0, 2010),
        ("rising", 1.0, 2020),
        ("spike", 1.0, 2030),
    ] {
        let c = bound_case(seed, &NS, qscale, shape);
        let out = call_attend(&c.q, &c.gate, &c.kv, &c.sel, &c.cnt, 0);
        assert!(all_finite(&out));
        let mut worst = 0f64;
        for (i, &n) in NS.iter().enumerate() {
            let ns = n.div_ceil(SLICES) as f64;
            let cells: Vec<usize> = (0..n).map(|j| c.sel[i * SPLIT_MAX_SEL + j] as usize).collect();
            for h in 0..H {
                let kh = h / (H / KVH);
                let qq = &c.q[(i * H + h) * HD..(i * H + h + 1) * HD];
                let k_of = |cell: usize| &c.kv[kv_at(cell, 0, kh)..kv_at(cell, 0, kh) + HD];
                let v_of = |cell: usize| &c.kv[kv_at(cell, 1, kh)..kv_at(cell, 1, kh) + HD];
                let x: Vec<f64> = cells.iter().map(|&cell| dot64(k_of(cell), qq) / 16.0).collect();
                let dx = 13.0
                    * U
                    * cells
                        .iter()
                        .map(|&cell| {
                            k_of(cell)
                                .iter()
                                .zip(qq)
                                .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
                                .sum::<f64>()
                                / 16.0
                        })
                        .fold(0.0, f64::max);
                let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
                let p: Vec<f64> = x.iter().map(|v| (v - xmax).exp()).collect();
                let psum: f64 = p.iter().sum();
                let mut o = vec![0f64; HD];
                let mut w = vec![0f64; HD];
                for (j, &cell) in cells.iter().enumerate() {
                    for (d, v) in v_of(cell).iter().enumerate() {
                        o[d] += p[j] * f64::from(*v);
                        w[d] += p[j] * f64::from(*v).abs();
                    }
                }
                let e = 2.0 * (U * (4.0 * ns + 24.0 + 3.0 * (xmax - xmin)) + 2.0 * dx);
                for d in 0..HD {
                    let (od, wd) = (o[d] / psum, w[d] / psum);
                    let sig = 1.0 / (1.0 + (-f64::from(c.gate[(i * H + h) * HD + d])).exp());
                    let bound = e * (wd + od.abs()) * sig;
                    let err = (f64::from(out[(i * H + h) * HD + d]) - od * sig).abs();
                    worst = worst.max(err / bound);
                    assert!(
                        err <= bound,
                        "{shape} n={n} head {h} d={d}: err {err:.3e} is outside the bound {bound:.3e}"
                    );
                }
            }
        }
        eprintln!("{shape} qscale={qscale}: worst error {worst:.3} of the design bound");
    }
}

/// The bits of a (token, head) are the same alone (T = 1), in a batch, on any grid, and on every run.
#[test]
fn split_invariance() {
    let c = bound_case(3000, &NS, 2.0, "random");
    let reference = call_attend(&c.q, &c.gate, &c.kv, &c.sel, &c.cnt, 0);
    for grid in [1, 3, 7, 80, 0] {
        let got = call_attend(&c.q, &c.gate, &c.kv, &c.sel, &c.cnt, grid);
        assert!(same_bits(&got, &reference), "grid={grid}: differs");
    }
    let row = H * HD;
    for i in 0..NS.len() {
        for grid in [0, 1] {
            let got = call_attend(
                &c.q[i * row..(i + 1) * row],
                &c.gate[i * row..(i + 1) * row],
                &c.kv,
                &c.sel[i * SPLIT_MAX_SEL..(i + 1) * SPLIT_MAX_SEL],
                &c.cnt[i..i + 1],
                grid,
            );
            assert!(
                same_bits(&got, &reference[i * row..(i + 1) * row]),
                "token {i} alone (grid={grid}): differs"
            );
        }
    }
    for repeat in 0..10 {
        let got = call_attend(&c.q, &c.gate, &c.kv, &c.sel, &c.cnt, 0);
        assert!(same_bits(&got, &reference), "repeat {repeat}: differs");
    }
}

// ---- test_attn_mask.py -------------------------------------------------------------------------------------------------

const IH: usize = 4;
const ID: usize = 128;
const MASK_MAX: usize = 4;
const TOL: f64 = 5e-4;
const SCORE_C: f64 = 8.0;
/// The heads of the stored reference output (`HEADS` in `tools/op_groups/attn_a.py`).
const HEADS: [usize; 2] = [0, 12];

fn max_sel_masked(budget: usize) -> usize {
    budget + RATIO - 1 + 2 * MASK_MAX * (RATIO - 1)
}

struct Case {
    q_raw: Vec<f32>,
    k_raw: Vec<f32>,
    v_raw: Vec<f32>,
    iq_raw: Vec<f32>,
    ik_raw: Vec<f32>,
    q_g: Vec<f32>,
    k_g: Vec<f32>,
    iq_g: Vec<f32>,
    ik_g: Vec<f32>,
    inv: Vec<f32>,
    kv: Vec<f32>,
    ik: Vec<f32>,
}

/// The inputs of case `idx`: the seeds and the shapes of `case_arrays` in `tools/op_groups/attn_a.py`.
fn make_case(idx: u64, p0: usize, t: usize, inv: &[f32]) -> Case {
    let base = 9000 + idx * 16;
    let n = |s: u64, len: usize| inputs::normals(base + s, len);
    let gamma = |s: u64, d: usize| -> Vec<f32> { n(s, d).iter().map(|v| 1.0f32 + 0.1f32 * v).collect() };
    let mut kv = vec![0f32; (p0 + t) * 2 * KVH * HD];
    kv[..p0 * 2 * KVH * HD].copy_from_slice(&n(9, p0 * 2 * KVH * HD));
    let mut ik = vec![0f32; (p0 + t) * ID];
    ik[..p0 * ID].copy_from_slice(&n(10, p0 * ID));
    Case {
        q_raw: n(0, t * H * 2 * HD),
        k_raw: n(1, t * KVH * HD),
        v_raw: n(2, t * KVH * HD),
        iq_raw: n(3, t * IH * ID),
        ik_raw: n(4, t * ID),
        q_g: gamma(5, HD),
        k_g: gamma(6, HD),
        iq_g: gamma(7, ID),
        ik_g: gamma(8, ID),
        inv: inv.to_vec(),
        kv,
        ik,
    }
}

struct Out {
    out: Vec<f32>,
    sel: Vec<i32>,
    cnt: Vec<u32>,
    scores: Vec<f32>,
    kv: Vec<f32>,
    ik: Vec<f32>,
}

/// Run `mk_attn` (`masked` false) or `mk_attn_masked` (true) on a copy of the case.
fn call(masked: bool, c: &Case, p0: usize, t: usize, budget: usize, ranges: &[(u32, u32)], grid: u32) -> Out {
    let mut kv = c.kv.clone();
    let mut ik = c.ik.clone();
    let max_nb = ((p0 + t) / RATIO).max(1);
    let stride = if masked {
        max_sel_masked(budget)
    } else {
        budget + RATIO - 1
    };
    let mut sel = vec![-7i32; t * stride];
    let mut cnt = vec![0u32; t];
    let mut scores = vec![0f32; t * max_nb];
    let mut out = vec![f32::NAN; t * H * HD];
    let lib = ops_lib::lib("attn");
    let rc = if masked {
        let lo: Vec<u32> = if ranges.is_empty() {
            vec![0]
        } else {
            ranges.iter().map(|r| r.0).collect()
        };
        let hi: Vec<u32> = if ranges.is_empty() {
            vec![0]
        } else {
            ranges.iter().map(|r| r.1).collect()
        };
        let f = sym!(
            lib,
            "mk_attn_masked",
            fn(
                u32,
                u32,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *mut c_void,
                u64,
                *mut c_void,
                u64,
                u32,
                u32,
                u32,
                *const c_void,
                *const c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                u32,
            ) -> i32
        );
        let (kv_bytes, ik_bytes) = (nbytes(&kv), nbytes(&ik));
        unsafe {
            f(
                t as u32,
                p0 as u32,
                ptr(&c.q_raw),
                ptr(&c.k_raw),
                ptr(&c.v_raw),
                ptr(&c.iq_raw),
                ptr(&c.ik_raw),
                ptr(&c.q_g),
                ptr(&c.k_g),
                ptr(&c.iq_g),
                ptr(&c.ik_g),
                ptr(&c.inv),
                ptr_mut(&mut kv),
                kv_bytes,
                ptr_mut(&mut ik),
                ik_bytes,
                budget as u32,
                RATIO as u32,
                ranges.len() as u32,
                ptr(&lo),
                ptr(&hi),
                ptr_mut(&mut sel),
                ptr_mut(&mut cnt),
                ptr_mut(&mut scores),
                ptr_mut(&mut out),
                grid,
            )
        }
    } else {
        let f = sym!(
            lib,
            "mk_attn",
            fn(
                u32,
                u32,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *const c_void,
                *mut c_void,
                *mut c_void,
                u32,
                u32,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                u32,
            ) -> i32
        );
        unsafe {
            f(
                t as u32,
                p0 as u32,
                ptr(&c.q_raw),
                ptr(&c.k_raw),
                ptr(&c.v_raw),
                ptr(&c.iq_raw),
                ptr(&c.ik_raw),
                ptr(&c.q_g),
                ptr(&c.k_g),
                ptr(&c.iq_g),
                ptr(&c.ik_g),
                ptr(&c.inv),
                ptr_mut(&mut kv),
                ptr_mut(&mut ik),
                budget as u32,
                RATIO as u32,
                ptr_mut(&mut sel),
                ptr_mut(&mut cnt),
                ptr_mut(&mut scores),
                ptr_mut(&mut out),
                grid,
            )
        }
    };
    assert_eq!(rc, 0, "mk_attn(_masked)");
    Out {
        out,
        sel,
        cnt,
        scores,
        kv,
        ik,
    }
}

fn masked_cells(ranges: &[(u32, u32)], n: usize) -> Vec<bool> {
    let mut m = vec![false; n];
    for &(lo, hi) in ranges {
        for cell in m.iter_mut().take(hi as usize).skip(lo as usize) {
            *cell = true;
        }
    }
    m
}

fn clear_blocks(masked: &[bool], nb: usize) -> Vec<bool> {
    (0..nb)
        .map(|b| !masked[b * RATIO..(b + 1) * RATIO].iter().any(|&m| m))
        .collect()
}

/// The masked selection of `attn_mask.h` from a row of block scores, as the ascending list of cells. Only clear blocks
/// compete. Dense when the clear blocks number at most `budget / 4`. Blocks that straddle a mask edge and the tail are
/// always taken, minus their masked cells.
fn masked_definition(n_vis: usize, scores_row: &[f32], ranges: &[(u32, u32)], budget: usize) -> Vec<i32> {
    let masked = masked_cells(ranges, n_vis);
    let nb = n_vis / RATIO;
    let clear = clear_blocks(&masked, nb);
    let topk = budget / RATIO;
    let mut chosen = clear.clone();
    if clear.iter().filter(|&&c| c).count() > topk {
        let mut idx: Vec<usize> = (0..nb).filter(|&b| clear[b]).collect();
        idx.sort_by(|&a, &b| {
            scores_row[b]
                .partial_cmp(&scores_row[a])
                .expect("a clear score is not NaN")
        });
        chosen = vec![false; nb];
        for &b in &idx[..topk] {
            chosen[b] = true;
        }
    }
    (0..n_vis)
        .filter(|&cell| !masked[cell] && (cell >= nb * RATIO || !clear[cell / RATIO] || chosen[cell / RATIO]))
        .map(|cell| cell as i32)
        .collect()
}

/// (budget, p0, t, ranges, label). Toy budgets make the sparse path reachable at small sizes.
type MaskCase = (usize, usize, usize, Vec<(u32, u32)>);

fn mask_cases(fixture: &Fixture) -> Vec<MaskCase> {
    fixture
        .meta("cases")
        .as_array()
        .expect("the cases")
        .iter()
        .map(|c| {
            let ranges = c["ranges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (r[0].as_u64().unwrap() as u32, r[1].as_u64().unwrap() as u32))
                .collect();
            (
                c["budget"].as_u64().unwrap() as usize,
                c["p0"].as_u64().unwrap() as usize,
                c["t"].as_u64().unwrap() as usize,
                ranges,
            )
        })
        .collect()
}

fn label(fixture: &Fixture, idx: usize) -> String {
    fixture.meta("cases")[idx]["label"].as_str().unwrap().to_owned()
}

/// The oracle of the masked op, against the f64 reference that the fixture holds. The selection equals the definition applied
/// to the scores of the op, with no tolerance. The reference selects the same cells (checked by a digest). The clear-block
/// scores are within the score bound of the reference. The output of two heads of each token is within `TOL` relative L2.
#[test]
fn mask_oracle() {
    let f = Fixture::load("attn_a");
    let inv = f.f32_flat("inv_freq");
    for (idx, (budget, p0, t, ranges)) in mask_cases(&f).into_iter().enumerate() {
        let name = label(&f, idx);
        let c = make_case(idx as u64, p0, t, &inv);
        let o = call(true, &c, p0, t, budget, &ranges, 0);
        let n_last = p0 + t;
        let clear_last = clear_blocks(&masked_cells(&ranges, n_last), n_last / RATIO)
            .iter()
            .filter(|&&c| c)
            .count();
        assert_eq!(
            clear_last > budget / RATIO,
            name.contains("sparse"),
            "{name}: the case is not on the path it names"
        );
        let ref_scores = f.f32_flat(&format!("score_{idx}"));
        let ref_abs = f.f32_flat(&format!("sabs_{idx}"));
        let ref_out = f.f32_flat(&format!("out_{idx}"));
        let digests = f.meta("cases")[idx]["sha256"].as_array().unwrap().clone();
        let stride = max_sel_masked(budget);
        let max_nb = ((p0 + t) / RATIO).max(1);
        let mut at = 0;
        for i in 0..t {
            let n_vis = p0 + i + 1;
            let nb = n_vis / RATIO;
            let masked = masked_cells(&ranges, n_vis);
            let clear = clear_blocks(&masked, nb);
            let row = &o.scores[i * max_nb..i * max_nb + nb];
            // Scores: the clear blocks are within the bound of the reference, and the others are -inf (never scored).
            for b in 0..nb {
                if clear[b] {
                    let bound = SCORE_C * ID as f64 * U * f64::from(ref_abs[at + b]) + 1e-30;
                    let err = (f64::from(row[b]) - f64::from(ref_scores[at + b])).abs();
                    assert!(
                        err <= bound,
                        "{name} token {i}: block {b} score error {:.2}x the bound",
                        err / bound
                    );
                } else {
                    assert!(
                        row[b] == f32::NEG_INFINITY,
                        "{name} token {i}: a block with a masked cell has a score"
                    );
                }
            }
            at += nb;
            // The selection is the definition on our own scores, exactly, then -1 padding.
            let want = masked_definition(n_vis, row, &ranges, budget);
            let n = o.cnt[i] as usize;
            let got = &o.sel[i * stride..i * stride + n];
            assert!(
                got == want.as_slice(),
                "{name} token {i}: selected {n} cells, the definition {}",
                want.len()
            );
            assert!(
                o.sel[i * stride + n..(i + 1) * stride].iter().all(|&s| s == -1),
                "{name} token {i}: padding is not -1"
            );
            assert!(
                got.iter().all(|&cell| !masked[cell as usize]),
                "{name} token {i}: a masked cell is selected"
            );
            // The reference selects the same cells, so the reference output is the output over our cells.
            let bytes: Vec<u8> = got.iter().flat_map(|c| c.to_le_bytes()).collect();
            let digest: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(
                digest,
                digests[i].as_str().unwrap(),
                "{name} token {i}: the selection differs from the reference selection (a near-tie, or a defect)"
            );
            let mut got_out = Vec::new();
            for h in HEADS {
                got_out.extend_from_slice(&o.out[(i * H + h) * HD..(i * H + h + 1) * HD]);
            }
            assert!(
                all_finite(&o.out[i * H * HD..(i + 1) * H * HD]),
                "{name} token {i}: attn_out is not finite"
            );
            let want_out = &ref_out[(HEADS.len() * i) * HD..(HEADS.len() * (i + 1)) * HD];
            let e = relerr(&got_out, want_out);
            assert!(e <= TOL, "{name} token {i}: attn_out relative L2 {e:.3e} > {TOL}");
        }
    }
}

fn same(a: &[f32], b: &[f32]) -> bool {
    same_bits(a, b)
}

/// The masked cells hold NaN or 1e30 in K, V and the raw index keys. Nothing may read them: every output is the bits of the
/// unpoisoned run.
fn poisoned_cells_are_never_read(fill: f32) {
    let f = Fixture::load("attn_a");
    let inv = f.f32_flat("inv_freq");
    for (idx, (budget, p0, t, ranges)) in mask_cases(&f).into_iter().enumerate() {
        let c = make_case(idx as u64, p0, t, &inv);
        let base = call(true, &c, p0, t, budget, &ranges, 0);
        let mut bad = make_case(idx as u64, p0, t, &inv);
        for &(lo, hi) in &ranges {
            bad.kv[lo as usize * 2 * KVH * HD..hi as usize * 2 * KVH * HD].fill(fill);
            bad.ik[lo as usize * ID..hi as usize * ID].fill(fill);
        }
        let o = call(true, &bad, p0, t, budget, &ranges, 0);
        let name = label(&f, idx);
        assert!(all_finite(&o.out), "{name}: a poisoned cell reached an output");
        assert!(
            same(&o.out, &base.out) && o.sel == base.sel && o.cnt == base.cnt,
            "{name}: poisoning masked cells changed an output"
        );
        let fin: Vec<usize> = (0..base.scores.len()).filter(|&i| base.scores[i].is_finite()).collect();
        assert!(
            fin.iter().all(|&i| o.scores[i].to_bits() == base.scores[i].to_bits())
                && o.scores.iter().filter(|v| v.is_finite()).count() == fin.len(),
            "{name}: poisoning masked cells changed a block score"
        );
    }
}

#[test]
fn mask_poisoned_cells_are_never_read_nan() {
    poisoned_cells_are_never_read(f32::NAN);
}

#[test]
fn mask_poisoned_cells_are_never_read_1e30() {
    poisoned_cells_are_never_read(1e30);
}

/// The control: the same NaN in V, with no mask, reaches the output of the unmasked op, so the test above can fail.
#[test]
fn mask_poison_reaches_unmasked_reads() {
    let f = Fixture::load("attn_a");
    let inv = f.f32_flat("inv_freq");
    let (p0, t) = (60, 8);
    let mut c = make_case(99, p0, t, &inv);
    for cell in 10..23 {
        let at = (cell * 2 + 1) * KVH * HD;
        c.kv[at..at + KVH * HD].fill(f32::NAN);
    }
    let o = call(false, &c, p0, t, 2048, &[], 0);
    assert!(
        o.out.iter().any(|v| v.is_nan()),
        "NaN V cells did not reach the unmasked output"
    );
}

#[test]
fn mask_no_ranges_is_the_unmasked_op() {
    let f = Fixture::load("attn_a");
    let inv = f.f32_flat("inv_freq");
    for (j, (budget, p0, t)) in [
        (2048usize, 60usize, 8usize),
        (16, 61, 7),
        (2048, 4400, 3),
        (2048, 2300, 1),
    ]
    .into_iter()
    .enumerate()
    {
        let c = make_case(100 + j as u64, p0, t, &inv);
        let a = call(false, &c, p0, t, budget, &[], 0);
        let b = call(true, &c, p0, t, budget, &[], 0);
        let w = budget + RATIO - 1;
        let ws = max_sel_masked(budget);
        let first_equal = (0..t).all(|i| a.sel[i * w..(i + 1) * w] == b.sel[i * ws..i * ws + w]);
        let pad_ok = (0..t).all(|i| b.sel[i * ws + w..(i + 1) * ws].iter().all(|&s| s == -1));
        assert!(
            a.cnt == b.cnt && first_equal && pad_ok,
            "budget {budget} p0 {p0}: selection differs"
        );
        assert!(
            same(&a.scores, &b.scores) && same(&a.out, &b.out) && same(&a.kv, &b.kv) && same(&a.ik, &b.ik),
            "budget {budget} p0 {p0}: an output differs from the unmasked op"
        );
    }
}

#[test]
fn mask_clear_block_scores_are_the_unmasked_ops() {
    let f = Fixture::load("attn_a");
    let inv = f.f32_flat("inv_freq");
    for (idx, (budget, p0, t, ranges)) in mask_cases(&f).into_iter().enumerate() {
        let c = make_case(idx as u64, p0, t, &inv);
        let m = call(true, &c, p0, t, budget, &ranges, 0);
        let p = call(false, &c, p0, t, budget, &[], 0);
        let fin: Vec<usize> = (0..m.scores.len()).filter(|&i| m.scores[i].is_finite()).collect();
        assert!(!fin.is_empty(), "{}: no clear block", label(&f, idx));
        assert!(
            fin.iter().all(|&i| m.scores[i].to_bits() == p.scores[i].to_bits()),
            "{}: a clear block score differs from the unmasked op",
            label(&f, idx)
        );
    }
}

#[test]
fn mask_grid_and_repeat_invariance() {
    let f = Fixture::load("attn_a");
    let inv = f.f32_flat("inv_freq");
    let (budget, p0, t) = (2048, 4400, 3);
    let ranges = [(101, 303), (2000, 2001), (2500, 2600)];
    let c = make_case(120, p0, t, &inv);
    let reference = call(true, &c, p0, t, budget, &ranges, 0);
    for grid in [1, 3, 7, 0, 0, 0] {
        let got = call(true, &c, p0, t, budget, &ranges, grid);
        assert!(
            same(&got.out, &reference.out) && got.sel == reference.sel && same(&got.scores, &reference.scores),
            "grid {grid}: differs"
        );
    }
}
