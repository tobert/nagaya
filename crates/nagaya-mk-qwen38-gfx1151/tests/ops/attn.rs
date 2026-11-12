//! QSA attention (`ops/attn.h`, `ops/attn.hip`) against the HF module `Qwen4ExpTextAttention`. It ports `test_attn.py` and
//! `test_attn_long.py` of the reference.
//!
//! The reference ran the HF module with the real weights of layer 3 on real embeddings. These tests give the op raw
//! projections that the generator makes. `tools/op_groups/attn.py` runs the HF module on the same projections, with
//! stand-ins for its projection layers, and writes the output before `o_proj`, the selected cells and, for the long
//! cases, the block scores. The expected outputs hold the heads 0, 11, 12 and 23 only, to keep the fixture small. The
//! relative L2 error is over those heads.
//!
//! The tolerances are the ones of the reference, and the reference's text gives their design: `TOL` 5e-4 on `attn_out`,
//! `NEAR_TIE` 1e-4 for the sparse selection, `KTOL` 1e-4 for the K cells, and 8 * 128 * 2^-24 * S_abs for a block
//! score. The fixture holds the bound of each block that it names, so the constants of the bound are in `attn.py`.

use crate::fixture::Fixture;
use crate::inputs::normal_at;
use crate::ops_lib::{all_finite, bits, lib, ptr, ptr_mut, relerr};
use crate::sym;
use std::ffi::c_void;

const TOL: f64 = 5e-4;
const NEAR_TIE: f64 = 1e-4;
const KTOL: f64 = 1e-4;
const H: usize = 24;
const HD: usize = 256;
const ID: usize = 128;
const RATIO: usize = 4;
/// The heads that the fixture holds.
const HEADS: [usize; 4] = [0, 11, 12, 23];
const DENSE_SEED: u64 = 71;
const SPARSE_SEED: u64 = 72;
const LONG_SEED: u64 = 73;

/// The width of each raw projection and the stream of the generator that makes it (see `attn.py`).
fn width(kind: &str) -> usize {
    match kind {
        "q" => 12288,
        "k" | "v" | "iq" => 512,
        "ik" => 128,
        _ => panic!("no projection {kind}"),
    }
}

fn stream(kind: &str) -> u64 {
    match kind {
        "q" => 1,
        "k" => 2,
        "v" => 3,
        "iq" => 4,
        "ik" => 5,
        _ => panic!("no projection {kind}"),
    }
}

/// The raw projections `kind` of the tokens `first..first + count` of the sequence of `seed`.
fn raw(seed: u64, kind: &str, first: usize, count: usize) -> Vec<f32> {
    let w = width(kind);
    let s = seed * 10 + stream(kind);
    ((first * w) as u64..((first + count) * w) as u64)
        .map(|i| normal_at(s, i))
        .collect()
}

struct Gammas {
    q: Vec<f32>,
    k: Vec<f32>,
    iq: Vec<f32>,
    ik: Vec<f32>,
}

fn gammas(fixture: &Fixture, family: &str) -> Gammas {
    let get = |name: &str| fixture.f32_flat(&format!("{family}_gamma_{name}"));
    Gammas {
        q: get("q"),
        k: get("k"),
        iq: get("iq"),
        ik: get("ik"),
    }
}

/// The projections of the tokens of one call.
struct Projections {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    iq: Vec<f32>,
    ik: Vec<f32>,
}

fn projections(seed: u64, first: usize, count: usize) -> Projections {
    Projections {
        q: raw(seed, "q", first, count),
        k: raw(seed, "k", first, count),
        v: raw(seed, "v", first, count),
        iq: raw(seed, "iq", first, count),
        ik: raw(seed, "ik", first, count),
    }
}

/// What `qwen38_attn` gives back besides the cache.
#[derive(Clone, PartialEq)]
struct Out {
    attn_out: Vec<f32>,
    sel: Vec<i32>,
    count: Vec<u32>,
    scores: Vec<f32>,
    max_sel: usize,
    max_nb: usize,
}

impl Out {
    fn cells(&self, token: usize) -> Vec<i32> {
        self.sel[token * self.max_sel..token * self.max_sel + self.count[token] as usize].to_vec()
    }

    fn scores_of(&self, token: usize, nb: usize) -> &[f32] {
        &self.scores[token * self.max_nb..token * self.max_nb + nb]
    }

    /// `attn_out` of the tokens, for the heads of the fixture only.
    fn heads(&self, token: usize) -> Vec<f32> {
        let row = &self.attn_out[token * H * HD..(token + 1) * H * HD];
        HEADS
            .iter()
            .flat_map(|&h| row[h * HD..(h + 1) * HD].iter().copied())
            .collect()
    }
}

/// `qwen38_attn`: the port of `call_attn` of `test_attn.py`. The cache holds `p0 + t` cells, and the call writes the
/// last `t`.
#[allow(clippy::too_many_arguments)]
fn call_attn(
    t: usize,
    p0: usize,
    p: &Projections,
    g: &Gammas,
    inv_freq: &[f32],
    kv: &mut [f32],
    ik: &mut [f32],
    budget: usize,
    grid: u32,
) -> Out {
    let max_nb = (p0 + t) / RATIO;
    let max_sel = budget + RATIO - 1;
    let mut sel = vec![-1i32; t * max_sel];
    let mut count = vec![0u32; t];
    let mut scores = vec![0f32; t * max_nb.max(1)];
    let mut attn_out = vec![f32::NAN; t * H * HD];
    let f = sym!(
        lib("attn"),
        "qwen38_attn",
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
    let rc = unsafe {
        f(
            t as u32,
            p0 as u32,
            ptr(&p.q),
            ptr(&p.k),
            ptr(&p.v),
            ptr(&p.iq),
            ptr(&p.ik),
            ptr(&g.q),
            ptr(&g.k),
            ptr(&g.iq),
            ptr(&g.ik),
            ptr(inv_freq),
            ptr_mut(kv),
            ptr_mut(ik),
            budget as u32,
            RATIO as u32,
            ptr_mut(&mut sel),
            ptr_mut(&mut count),
            ptr_mut(&mut scores),
            ptr_mut(&mut attn_out),
            grid,
        )
    };
    assert_eq!(rc, 0, "qwen38_attn");
    Out {
        attn_out,
        sel,
        count,
        scores,
        max_sel,
        max_nb: max_nb.max(1),
    }
}

/// The cells that the fixture says the reference selected for a token: its row without the padding.
fn reference_cells(sel: &[i32], width: usize, token: usize) -> Vec<i32> {
    sel[token * width..(token + 1) * width]
        .iter()
        .copied()
        .filter(|&c| c >= 0)
        .collect()
}

fn sorted(mut cells: Vec<i32>) -> Vec<i32> {
    cells.sort_unstable();
    cells
}

/// The fixture values of a family: the expected heads, the selected cells, the gammas and the rope table.
struct Family {
    out: Vec<f32>,
    sel: Vec<i32>,
    sel_width: usize,
    gammas: Gammas,
    inv_freq: Vec<f32>,
}

fn family(name: &str) -> Family {
    let fixture = Fixture::load("attn");
    let (out, shape) = fixture.f32s(&format!("{name}_out"));
    assert_eq!(shape[1..], [4, HD], "the shape of {name}_out");
    let (sel, sel_shape) = fixture.i32s(&format!("{name}_sel"));
    let gammas = gammas(&fixture, name);
    let inv_freq = fixture.f32_flat("inv_freq");
    Family {
        out,
        sel,
        sel_width: sel_shape[1],
        gammas,
        inv_freq,
    }
}

impl Family {
    fn tokens(&self) -> usize {
        self.out.len() / (HEADS.len() * HD)
    }

    fn expected(&self, token: usize) -> &[f32] {
        &self.out[token * HEADS.len() * HD..(token + 1) * HEADS.len() * HD]
    }

    fn visible(&self, token: usize) -> Vec<i32> {
        reference_cells(&self.sel, self.sel_width, token)
    }
}

fn empty_cache(cells: usize) -> (Vec<f32>, Vec<f32>) {
    (vec![0f32; cells * 2 * 2 * HD], vec![0f32; cells * ID])
}

#[test]
fn dense() {
    let fam = family("dense");
    for t in [1usize, 9, 24] {
        let p = projections(DENSE_SEED, 0, t);
        let (mut kv, mut ik) = empty_cache(t);
        let out = call_attn(t, 0, &p, &fam.gammas, &fam.inv_freq, &mut kv, &mut ik, 2048, 0);
        assert!(all_finite(&out.attn_out), "unwritten or non-finite attn_out");
        let ours: Vec<f32> = (0..t).flat_map(|i| out.heads(i)).collect();
        let want: Vec<f32> = (0..t).flat_map(|i| fam.expected(i).to_vec()).collect();
        let err = relerr(&ours, &want);
        eprintln!("dense T={t}: attn_out relerr {err:.2e}");
        assert!(err < TOL, "dense T={t}: attn_out relerr {err:.3e} >= {TOL}");
        for tok in 0..t {
            let got = sorted(out.cells(tok));
            let reference = sorted(fam.visible(tok));
            assert_eq!(
                got, reference,
                "T={t} tok={tok}: dense selection differs from the visible cells of the reference"
            );
            let causal: Vec<i32> = (0..=tok as i32).collect();
            assert_eq!(got, causal, "T={t} tok={tok}: dense selection is not the causal cells");
        }
    }
}

#[test]
fn continuation() {
    let fam = family("dense");
    let (p0, t2) = (16usize, 8usize);
    let ttot = p0 + t2;
    assert!(ttot <= fam.tokens());
    let all = projections(DENSE_SEED, 0, ttot);
    let slice = |lo: usize, hi: usize| Projections {
        q: all.q[lo * 12288..hi * 12288].to_vec(),
        k: all.k[lo * 512..hi * 512].to_vec(),
        v: all.v[lo * 512..hi * 512].to_vec(),
        iq: all.iq[lo * 512..hi * 512].to_vec(),
        ik: all.ik[lo * 128..hi * 128].to_vec(),
    };
    let (mut kv1, mut ik1) = empty_cache(p0);
    let out1 = call_attn(
        p0,
        0,
        &slice(0, p0),
        &fam.gammas,
        &fam.inv_freq,
        &mut kv1,
        &mut ik1,
        2048,
        0,
    );
    let ours: Vec<f32> = (0..p0).flat_map(|i| out1.heads(i)).collect();
    let want: Vec<f32> = (0..p0).flat_map(|i| fam.expected(i).to_vec()).collect();
    let err1 = relerr(&ours, &want);
    eprintln!("continuation head p0=0 T={p0}: attn_out relerr {err1:.2e}");
    assert!(err1 < TOL, "head: attn_out relerr {err1:.3e}");

    let (mut kv2, mut ik2) = empty_cache(ttot);
    kv2[..kv1.len()].copy_from_slice(&kv1);
    ik2[..ik1.len()].copy_from_slice(&ik1);
    let out2 = call_attn(
        t2,
        p0,
        &slice(p0, ttot),
        &fam.gammas,
        &fam.inv_freq,
        &mut kv2,
        &mut ik2,
        2048,
        0,
    );
    let ours: Vec<f32> = (0..t2).flat_map(|i| out2.heads(i)).collect();
    let want: Vec<f32> = (p0..ttot).flat_map(|i| fam.expected(i).to_vec()).collect();
    let err2 = relerr(&ours, &want);
    eprintln!("continuation tail p0={p0} T={t2}: attn_out relerr {err2:.2e}");
    assert!(err2 < TOL, "tail: attn_out relerr {err2:.3e}");
    for i in 0..t2 {
        assert_eq!(
            sorted(out2.cells(i)),
            sorted(fam.visible(p0 + i)),
            "continuation tok={}: dense selection differs from the visible cells of the reference",
            p0 + i
        );
    }
}

#[test]
fn sparse_selection() {
    let fam = family("sparse");
    let (budget, t) = (16usize, fam.tokens());
    let p = projections(SPARSE_SEED, 0, t);
    let (mut kv, mut ik) = empty_cache(t);
    let out = call_attn(t, 0, &p, &fam.gammas, &fam.inv_freq, &mut kv, &mut ik, budget, 0);
    let ours: Vec<f32> = (0..t).flat_map(|i| out.heads(i)).collect();
    let want: Vec<f32> = (0..t).flat_map(|i| fam.expected(i).to_vec()).collect();
    let err = relerr(&ours, &want);
    eprintln!("sparse budget={budget}: attn_out relerr {err:.2e}");
    assert!(err < TOL, "sparse: attn_out relerr {err:.3e}");

    let block_topk = budget / RATIO;
    let mut near_ties = Vec::new();
    for tok in 0..t {
        let nb = (tok + 1) / RATIO;
        let got = sorted(out.cells(tok));
        let reference = sorted(fam.visible(tok));
        if got == reference {
            continue;
        }
        assert!(
            nb > block_topk,
            "tok={tok}: mismatch but the selection should have been dense (nb={nb})"
        );
        // Each disagreement must come from a block whose own score is within NEAR_TIE of the border between the
        // block_topk-th and the next block. The score is ours: a real selection defect shows as a large gap.
        let mut s: Vec<f32> = out.scores_of(tok, nb).to_vec();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let gap = f64::from(s[block_topk - 1] - s[block_topk]);
        near_ties.push((tok, gap));
        assert!(
            gap < NEAR_TIE,
            "tok={tok}: selected-set mismatch with border gap {gap:.2e} >= {NEAR_TIE:.0e} (got {got:?} want {reference:?})"
        );
    }
    eprintln!(
        "sparse budget={budget}: {}/{t} tokens had a near-tie border {near_ties:?}",
        near_ties.len()
    );
}

#[test]
fn repeats_and_grid_independence() {
    let fam = family("dense");
    let t = 20;
    let p = projections(DENSE_SEED, 0, t);
    let (kv0, ik0) = empty_cache(t);
    let run = |grid: u32| {
        let (mut kv, mut ik) = (kv0.clone(), ik0.clone());
        call_attn(t, 0, &p, &fam.gammas, &fam.inv_freq, &mut kv, &mut ik, 2048, grid)
    };
    let reference = run(0);
    for grid in [1, 3, 0] {
        let out = run(grid);
        assert_eq!(
            bits(&out.attn_out),
            bits(&reference.attn_out),
            "grid={grid}: attn_out differs"
        );
        assert!(
            out.sel == reference.sel && out.count == reference.count,
            "grid={grid}: selection differs"
        );
    }
    for repeat in 0..20 {
        let out = run(0);
        assert_eq!(
            bits(&out.attn_out),
            bits(&reference.attn_out),
            "repeat {repeat}: attn_out not bit-identical"
        );
        assert!(
            out.sel == reference.sel && out.count == reference.count,
            "repeat {repeat}: selection not identical"
        );
    }
}

// ---- test_attn_long.py ----

/// The engine builds its rope table on the host as `1.0f / powf(1e7f, 2i/64.0f)` (`blocks.hip` `inv_freq_table`). At
/// position 32767 a difference of one ulp in `inv_freq` moves the angle by about 2e-3 rad, so the table must be the table
/// of the reference bit for bit. The test calls `powf` of libm, as the engine does.
#[test]
fn inv_freq_table() {
    let libm = unsafe { libloading::Library::new("libm.so.6") }.expect("load libm.so.6");
    let powf = sym!(&libm, "powf", fn(f32, f32) -> f32);
    let ours: Vec<f32> = (0..32)
        .map(|i| 1.0f32 / unsafe { powf(1e7f32, (2 * i) as f32 / 64.0f32) })
        .collect();
    let reference = Fixture::load("attn").f32_flat("inv_freq");
    assert_eq!(
        bits(&ours),
        bits(&reference),
        "the table of the engine is not the table of the reference"
    );
}

/// The K cells of the engine for a position: RMSNorm of 256 values with the gamma, then rope on the first 64 values. The
/// reference takes its cache from its own f32 code; this is the same computation in f64, so the cells differ by about 1e-7.
fn k_cell(k_raw: &[f32], gamma: &[f32], pos: usize, inv_freq: &[f32], out: &mut [f32]) {
    for head in 0..2 {
        let x: Vec<f64> = k_raw[head * HD..(head + 1) * HD]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let mean = x.iter().map(|v| v * v).sum::<f64>() / HD as f64;
        let scale = 1.0 / (mean + 1e-6).sqrt();
        let mut y: Vec<f64> = x.iter().zip(gamma).map(|(v, g)| v * scale * f64::from(*g)).collect();
        for i in 0..32 {
            let angle = f64::from(pos as f32 * inv_freq[i]);
            let (sn, cs) = angle.sin_cos();
            let (a, b) = (y[i], y[i + 32]);
            y[i] = a * cs - b * sn;
            y[i + 32] = b * cs + a * sn;
        }
        for (o, v) in out[head * HD..(head + 1) * HD].iter_mut().zip(&y) {
            *o = *v as f32;
        }
    }
}

/// The cache of the long cases: the cells `0..p0` hold K, V and the raw index keys, and the cells `p0..p0 + t` hold NaN, so
/// that a cell that the op does not write shows as NaN.
fn long_cache(g: &Gammas, inv_freq: &[f32], p0: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let cells = p0 + t;
    let k_all = raw(LONG_SEED, "k", 0, p0);
    let v_all = raw(LONG_SEED, "v", 0, p0);
    let ik_all = raw(LONG_SEED, "ik", 0, p0);
    let mut kv = vec![f32::NAN; cells * 2 * 2 * HD];
    for pos in 0..p0 {
        let cell = &mut kv[pos * 1024..(pos + 1) * 1024];
        k_cell(
            &k_all[pos * 512..(pos + 1) * 512],
            &g.k,
            pos,
            inv_freq,
            &mut cell[..512],
        );
        cell[512..].copy_from_slice(&v_all[pos * 512..(pos + 1) * 512]);
    }
    let mut ik = vec![f32::NAN; cells * ID];
    ik[..p0 * ID].copy_from_slice(&ik_all);
    (kv, ik)
}

const LONG_CASES: usize = 5;

fn long_case(n: usize) -> (usize, usize) {
    let fixture = Fixture::load("attn");
    let case = &fixture.meta("long_cases")[n];
    (
        case["p0"].as_u64().unwrap() as usize,
        case["t"].as_u64().unwrap() as usize,
    )
}

/// One long case: the top blocks of the reference for each token, with its score and the bound of the score.
struct LongReference {
    out: Vec<f32>,
    ids: Vec<i32>,
    scores: Vec<f32>,
    bounds: Vec<f32>,
    k_cells: Vec<f32>,
    nk: usize,
}

fn long_reference(fixture: &Fixture, n: usize) -> LongReference {
    let (ids, shape) = fixture.i32s(&format!("long{n}_ids"));
    LongReference {
        out: fixture.f32_flat(&format!("long{n}_out")),
        ids,
        scores: fixture.f32_flat(&format!("long{n}_scores")),
        bounds: fixture.f32_flat(&format!("long{n}_bounds")),
        k_cells: fixture.f32_flat(&format!("long{n}_k")),
        nk: shape[1],
    }
}

#[test]
fn long_position() {
    let fixture = Fixture::load("attn");
    let g = gammas(&fixture, "long");
    let inv_freq = fixture.f32_flat("inv_freq");
    assert_eq!(LONG_CASES, fixture.meta("long_cases").as_array().unwrap().len());
    for n in 0..LONG_CASES {
        let (p0, t) = long_case(n);
        let reference = long_reference(&fixture, n);
        let p = projections(LONG_SEED, p0, t);
        let (mut kv, mut ik) = long_cache(&g, &inv_freq, p0, t);
        let out = call_attn(t, p0, &p, &g, &inv_freq, &mut kv, &mut ik, 2048, 0);
        check_long(p0, t, &p, &kv, &ik, &out, &reference);
    }
}

/// The checks of `test_long_position` of the reference, for one case.
fn check_long(p0: usize, t: usize, p: &Projections, kv: &[f32], ik: &[f32], out: &Out, r: &LongReference) {
    let ttot = p0 + t;
    // prep: the new cells
    assert!(all_finite(kv) && all_finite(ik), "p0={p0}: prep left a poisoned cell");
    let k_new: Vec<f32> = (p0..ttot)
        .flat_map(|c| kv[c * 1024..c * 1024 + 512].iter().copied())
        .collect();
    let kerr = relerr(&k_new, &r.k_cells);
    assert!(kerr < KTOL, "K cells [{p0}, {ttot}): relerr {kerr:.2e} >= {KTOL}");
    let v_new: Vec<f32> = (p0..ttot)
        .flat_map(|c| kv[c * 1024 + 512..(c + 1) * 1024].iter().copied())
        .collect();
    assert_eq!(bits(&v_new), bits(&p.v), "V cells are not copies of v_raw");
    assert_eq!(
        bits(&ik[p0 * ID..]),
        bits(&p.ik),
        "index key cells are not copies of the raw keys"
    );

    let topk = 2048 / RATIO;
    let (mut worst_abs, mut ties, mut equal_tokens) = (0f64, Vec::new(), Vec::new());
    for i in 0..t {
        let n_vis = p0 + i + 1;
        let nb = n_vis / RATIO;
        let ours = out.scores_of(i, nb);
        let row = |a: &[f32]| a[i * r.nk..(i + 1) * r.nk].to_vec();
        let (ids, ref_scores, bounds) = (r.ids[i * r.nk..(i + 1) * r.nk].to_vec(), row(&r.scores), row(&r.bounds));
        // Each stored block: our score is within the bound of the score of the reference.
        let stored = ids.iter().take_while(|&&b| b >= 0).count();
        assert_eq!(
            stored,
            nb.min(r.nk),
            "tok {}: the fixture holds {stored} blocks for {nb}",
            p0 + i
        );
        for j in 0..stored {
            let err = f64::from((ours[ids[j] as usize] - ref_scores[j]).abs());
            worst_abs = worst_abs.max(err);
            assert!(
                err <= f64::from(bounds[j]),
                "tok {}: block {} ours {} ref {} bound {:.2e}",
                p0 + i,
                ids[j],
                ours[ids[j] as usize],
                ref_scores[j],
                bounds[j]
            );
        }
        // Our selection is exactly our definition (integer logic, no tolerance): the top blocks by (score desc, index asc),
        // the cells of each in ascending order, then the tail cells.
        let k = topk.min(nb);
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &b| ours[b].partial_cmp(&ours[a]).unwrap().then(a.cmp(&b)));
        let mut chosen: Vec<usize> = order[..k].to_vec();
        chosen.sort_unstable();
        let mut want: Vec<i32> = chosen
            .iter()
            .flat_map(|&b| (b * RATIO..b * RATIO + RATIO).map(|c| c as i32))
            .collect();
        want.extend((nb * RATIO..n_vis).map(|c| c as i32));
        let got = out.cells(i);
        assert_eq!(
            got,
            want,
            "tok {}: selection is not top-{k} by (score desc, index asc)",
            p0 + i
        );
        assert!(
            out.sel[i * out.max_sel + got.len()..(i + 1) * out.max_sel]
                .iter()
                .all(|&c| c == -1),
            "tok {}: unused selection slots are not -1",
            p0 + i
        );

        // Against the selection of the reference: a difference only at a named near tie.
        let mut reference_cells: Vec<i32> = ids[..k]
            .iter()
            .flat_map(|&b| (b as usize * RATIO..b as usize * RATIO + RATIO).map(|c| c as i32))
            .collect();
        reference_cells.extend((nb * RATIO..n_vis).map(|c| c as i32));
        let reference_set: std::collections::BTreeSet<i32> = reference_cells.into_iter().collect();
        let got_set: std::collections::BTreeSet<i32> = got.iter().copied().collect();
        let diff: Vec<i32> = got_set.symmetric_difference(&reference_set).copied().collect();
        if diff.is_empty() {
            equal_tokens.push(i);
            continue;
        }
        assert!(
            diff.iter().all(|&c| (c as usize) < nb * RATIO),
            "tok {}: tail cells differ {:?}",
            p0 + i,
            &diff[..diff.len().min(8)]
        );
        let (boundary_score, boundary_bound) = (ref_scores[k - 1], bounds[k - 1]);
        let blocks: std::collections::BTreeSet<usize> = diff.iter().map(|&c| c as usize / RATIO).collect();
        for b in blocks {
            let j = ids[..stored].iter().position(|&x| x as usize == b).unwrap_or_else(|| {
                panic!(
                    "tok {}: block {b} differs and is not among the top blocks of the reference",
                    p0 + i
                )
            });
            let gap = f64::from((ref_scores[j] - boundary_score).abs());
            let allow = f64::from(bounds[j] + boundary_bound);
            assert!(
                gap <= allow,
                "tok {}: block {b} differs from the reference selection {gap:.2e} from the border (allowed {allow:.2e}): not a near tie",
                p0 + i
            );
            ties.push((p0 + i, b, gap));
        }
    }

    assert!(all_finite(&out.attn_out), "unwritten or non-finite attn_out");
    let ours: Vec<f32> = (0..t).flat_map(|i| out.heads(i)).collect();
    let err = relerr(&ours, &r.out);
    let tok_err = equal_tokens
        .iter()
        .map(|&i| relerr(&out.heads(i), &r.out[i * 4 * HD..(i + 1) * 4 * HD]))
        .fold(f64::NAN, f64::max);
    eprintln!(
        "p0={p0} T={t}: attn_out relerr {err:.2e} (worst per token {tok_err:.2e}), K cells {kerr:.2e}, worst score error {worst_abs:.1e}, near-tie blocks {ties:?}"
    );
    assert!(err < TOL, "attn_out relerr {err:.2e} >= {TOL}");
    assert!(
        equal_tokens.is_empty() || tok_err < TOL,
        "per-token attn_out relerr {tok_err:.2e} >= {TOL}"
    );
}

#[test]
fn long_grid_and_repeats() {
    let fixture = Fixture::load("attn");
    let g = gammas(&fixture, "long");
    let inv_freq = fixture.f32_flat("inv_freq");
    let n = LONG_CASES - 2;
    let (p0, t) = long_case(n);
    let p = projections(LONG_SEED, p0, t);
    let (kv0, ik0) = long_cache(&g, &inv_freq, p0, t);
    let run = |grid: u32| {
        let (mut kv, mut ik) = (kv0.clone(), ik0.clone());
        call_attn(t, p0, &p, &g, &inv_freq, &mut kv, &mut ik, 2048, grid)
    };
    let reference = run(0);
    let same = |out: &Out, what: &str| {
        assert_eq!(
            bits(&out.attn_out),
            bits(&reference.attn_out),
            "{what}: attn_out differs"
        );
        assert!(out.sel == reference.sel, "{what}: sel differs");
        assert!(out.count == reference.count, "{what}: sel_count differs");
        assert_eq!(
            bits(&out.scores),
            bits(&reference.scores),
            "{what}: block_scores differ"
        );
    };
    for grid in [1, 3, 0] {
        same(&run(grid), &format!("grid={grid}"));
    }
    for repeat in 0..20 {
        same(&run(0), &format!("repeat {repeat}"));
    }
}
