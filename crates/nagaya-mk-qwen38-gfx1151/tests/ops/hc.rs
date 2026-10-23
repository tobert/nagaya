//! Split-K, fused staging, and the hyper-connection ops. It ports `test_qmmh_split.py`, `test_qmmh_xin.py`,
//! `test_qmmh_hcmix.py`, `test_hc.py` and `test_hc_fused.py` of mk.
//!
//! mk used real rows of the Qwen3.8 GGUF. These tests use rows that `inputs::raw_rows` makes, with the same types and
//! shapes. The tests of `test_hc.py` that compared with the HF module use synthetic weights: see `hc_pre`.

use crate::fixture::Fixture;
use crate::inputs::{self, Qtype};
use crate::mkops::{
    self, all_finite, assert_same, dequant, lib, matmul_f64, nbytes, opt_nbytes, opt_ptr, ptr, ptr_mut, qmm_fast,
    same_bits,
};
use crate::{child, sym};
use std::ffi::c_void;

/// The type numbers of `quant.h`: Q8_0, and Q8_0 in tiles.
const Q8_0: u32 = 8;
const Q8_0_TILES: u32 = 108;

fn x_of(seed: u64, ntok: usize, k: usize) -> Vec<f32> {
    inputs::normals(seed, ntok * k)
}

// ---- test_qmmh_split.py ----

/// One narrow matrix of the engine that the split-K path takes. The label is the tensor of mk's test.
#[derive(Clone, Copy)]
struct Narrow {
    label: &'static str,
    qtype: Qtype,
    rows: usize,
    k: usize,
    seed: u64,
}

/// `CASES` of `test_qmmh_split.py`: the narrow Q8_0 shapes that the engine splits (hc down, shared gate and up, attention
/// k and v), and Q6_K and IQ4_XS rows for the unit addressing of the other decoders (a K-quant superblock spans 8 units).
const SPLIT_CASES: [Narrow; 5] = [
    Narrow {
        label: "blk.3.hc_ffn_down.weight",
        qtype: Qtype::Q8_0,
        rows: 320,
        k: 10240,
        seed: 801,
    },
    Narrow {
        label: "blk.0.ffn_gate_shexp.weight",
        qtype: Qtype::Q8_0,
        rows: 640,
        k: 2560,
        seed: 802,
    },
    Narrow {
        label: "blk.3.attn_k.weight",
        qtype: Qtype::Q8_0,
        rows: 512,
        k: 2560,
        seed: 803,
    },
    Narrow {
        label: "output.weight",
        qtype: Qtype::Q6K,
        rows: 96,
        k: 2560,
        seed: 804,
    },
    Narrow {
        label: "IQ4_XS",
        qtype: Qtype::Iq4Xs,
        rows: 128,
        k: 2560,
        seed: 805,
    },
];

/// `y [t, n] = W x` in `ksplit` K slices and then their sum: `mk_qmmh_split`.
fn split(case: &Narrow, raw: &[u8], x: &[f32], ksplit: u32, grid: u32) -> Vec<f32> {
    let t = x.len() / case.k;
    let mut y = vec![0f32; t * case.rows];
    let f = sym!(
        lib("qmmh"),
        "mk_qmmh_split",
        fn(u32, *const c_void, u64, u32, u32, *const c_void, u64, u32, *mut c_void, u64, u32, u32) -> i32
    );
    let rc = unsafe {
        f(
            case.qtype as u32,
            ptr(raw),
            nbytes(raw),
            case.rows as u32,
            case.k as u32,
            ptr(x),
            nbytes(x),
            t as u32,
            ptr_mut(&mut y),
            nbytes(&y),
            ksplit,
            grid,
        )
    };
    assert_eq!(rc, 0, "mk_qmmh_split");
    y
}

/// The split of the engine for a matrix shape: `mk_qmmh_ksplit`.
fn ksplit_of(n: usize, k: usize) -> u32 {
    let f = sym!(lib("qmmh"), "mk_qmmh_ksplit", fn(u32, u32) -> u32);
    unsafe { f(n as u32, k as u32) }
}

#[test]
fn split_within_the_f16_bound_and_near_unsplit() {
    for (n, case) in SPLIT_CASES.iter().enumerate() {
        let raw = inputs::raw_rows(case.qtype, case.seed, case.rows, case.k);
        let w = dequant(case.qtype, &raw, case.rows, case.k);
        for (m, ntok) in [1usize, 17, 70].into_iter().enumerate() {
            let x = x_of(900 + 10 * n as u64 + m as u64, ntok, case.k);
            let base = split(case, &raw, &x, 1, 0);
            let (y64, mag) = matmul_f64(&x, &w, ntok, case.rows, case.k);
            for ks in [2u32, 3, 5, 10] {
                let y = split(case, &raw, &x, ks, 0);
                assert!(all_finite(&y), "{} ks={ks}: unwritten outputs", case.label);
                for i in 0..y.len() {
                    let s2 = (2f64.powi(-11) + case.k as f64 * 2f64.powi(-24)) * mag[i];
                    assert!(
                        (f64::from(y[i]) - y64[i]).abs() <= s2,
                        "{} ks={ks} T={ntok}: past the S2 bound at {i}",
                        case.label
                    );
                    assert!(
                        (f64::from(y[i]) - f64::from(base[i])).abs() <= 2.0 * case.k as f64 * 2f64.powi(-24) * mag[i],
                        "{} ks={ks}: split and unsplit differ past reassociation at {i}",
                        case.label
                    );
                }
            }
            assert!(
                !same_bits(&split(case, &raw, &x, 5, 0), &base),
                "{}: ksplit had no effect",
                case.label
            );
        }
    }
}

#[test]
fn ksplit_1_is_todays_kernel() {
    let case = SPLIT_CASES[0];
    let raw = inputs::raw_rows(case.qtype, case.seed, case.rows, case.k);
    let x = x_of(950, 40, case.k);
    assert_same(
        &split(&case, &raw, &x, 1, 0),
        &qmm_fast(case.qtype, &raw, case.rows, case.k, &x, 0),
        "ksplit 1 against the plain f16 kernel",
    );
}

#[test]
fn a_tokens_bits_dont_depend_on_the_pass() {
    for (n, case) in SPLIT_CASES[..3].iter().enumerate() {
        let raw = inputs::raw_rows(case.qtype, case.seed, case.rows, case.k);
        let ks = ksplit_of(case.rows, case.k);
        let x = x_of(960 + n as u64, 256, case.k);
        let full = split(case, &raw, &x, ks, 0);
        for (lo, hi) in [(0usize, 1usize), (37, 38), (100, 117), (0, 17), (200, 256)] {
            let part = split(case, &raw, &x[lo * case.k..hi * case.k], ks, 0);
            assert_same(
                &part,
                &full[lo * case.rows..hi * case.rows],
                &format!("{} tokens {lo}:{hi}", case.label),
            );
        }
    }
}

#[test]
fn split_grid_independent() {
    let case = SPLIT_CASES[0];
    let raw = inputs::raw_rows(case.qtype, case.seed, case.rows, case.k);
    let x = x_of(970, 70, case.k);
    let reference = split(&case, &raw, &x, 16, 0);
    for grid in [1, 7, 80] {
        assert_same(&split(&case, &raw, &x, 16, grid), &reference, &format!("grid {grid}"));
    }
}

/// `ksplit(n, k)` fills about 80 workgroups when a matrix has fewer than 20 row groups, with at least 8 units (256 values)
/// in each slice. Wide matrices stay whole. The function has no token count and no grid: that is the property.
#[test]
fn the_engines_split_policy() {
    assert_eq!(ksplit_of(320, 10240), 16, "hc down: 5 row groups");
    assert_eq!(ksplit_of(640, 2560), 8, "shared expert gate and up: 10");
    assert_eq!(
        ksplit_of(512, 2560),
        10,
        "k, v and indexer q: 8, capped at 80 units / 8"
    );
    assert_eq!(ksplit_of(128, 2560), 10, "indexer k: 2, capped");
    assert_eq!(ksplit_of(1280, 2560), 1, "20 row groups: whole");
    assert_eq!(ksplit_of(10240, 2560), 1);
    assert_eq!(ksplit_of(2560, 640), 1);
}

/// More slices than pairs of chunks would leave empty slices. The host refuses that. The refusal is an abort, so the call
/// runs in a child process.
#[test]
fn a_bad_ksplit_is_refused_before_running() {
    let output = child::run("hc::child_bad_ksplit", &[]);
    let stderr = child::stderr(&output);
    assert!(
        !output.status.success() && stderr.contains("mk_qmmh_split: ksplit 3 leaves an empty slice of K=64"),
        "no refusal naming ksplit: {:?} {stderr}",
        output.status
    );
}

#[test]
#[ignore = "child of a_bad_ksplit_is_refused_before_running"]
fn child_bad_ksplit() {
    let case = Narrow {
        label: "two rows",
        qtype: Qtype::Q8_0,
        rows: 2,
        k: 64,
        seed: 0,
    };
    let raw = vec![0u8; 68 * 2];
    let x = vec![0f32; 64];
    let _ = split(&case, &raw, &x, 3, 0);
    println!("NO REFUSAL");
}

// ---- Q8_0 tiles, for the next two groups ----

/// Q8_0 rows of the GGUF as the tiles of `mk_relayout_q8`. The output starts as 0xAB, so a byte that the kernel does not
/// write shows.
fn relayout_q8(raw: &[u8], rows: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0xABu8; raw.len()];
    let f = sym!(
        lib("tiles"),
        "mk_relayout_q8",
        fn(*const c_void, u64, u32, u32, *mut c_void, u64) -> i32
    );
    let rc = unsafe {
        f(
            ptr(raw),
            nbytes(raw),
            rows as u32,
            k as u32,
            ptr_mut(&mut out),
            nbytes(&out),
        )
    };
    assert_eq!(rc, 0, "mk_relayout_q8");
    out
}

/// The weights of a fused stage: the GGUF rows, or their tiles.
fn weights(raw: &[u8], rows: usize, k: usize, tiled: bool) -> (u32, Vec<u8>) {
    if tiled {
        (Q8_0_TILES, relayout_q8(raw, rows, k))
    } else {
        (Q8_0, raw.to_vec())
    }
}

// ---- test_qmmh_xin.py ----

/// The consumers: hc up (K = 320, from the 16 slices of hc down) and the shared expert's down (K = 640, from the 8 slices
/// of gate and of up). `(label, rows, kind, slices, k, seed)`.
const XIN_CASES: [(&str, usize, u32, usize, usize, u64); 2] = [
    ("blk.3.hc_ffn_up.weight", 64, 1, 16, 320, 811),
    ("blk.0.ffn_down_shexp.weight", 48, 2, 8, 640, 812),
];

struct Xin<'a> {
    w: &'a [u8],
    qtype: u32,
    n: usize,
    k: usize,
    parts: &'a [f32],
    parts2: Option<&'a [f32]>,
    xks: u32,
    kind: u32,
    grid: u32,
}

/// The fused stage and the stages it replaces, on the same partials: `mk_qmmh_xin`. It returns the two outputs. The staged
/// x of both must be equal in f32, before the f16 staging could hide a rounding that differs.
fn xin(a: &Xin) -> (Vec<f32>, Vec<f32>) {
    let t = a.parts.len() / (a.xks as usize * a.k);
    let (mut yf, mut yr) = (vec![f32::NAN; t * a.n], vec![f32::NAN; t * a.n]);
    let (mut xf, mut xr) = (vec![f32::NAN; t * a.k], vec![f32::NAN; t * a.k]);
    let f = sym!(
        lib("qmmh"),
        "mk_qmmh_xin",
        fn(
            u32,
            *const c_void,
            u64,
            u32,
            u32,
            u32,
            *const c_void,
            u64,
            *const c_void,
            u64,
            u32,
            u32,
            f32,
            *mut c_void,
            u64,
            *mut c_void,
            u64,
            *mut c_void,
            u64,
            *mut c_void,
            u64,
            u32,
        ) -> i32
    );
    let rc = unsafe {
        f(
            a.qtype,
            ptr(a.w),
            nbytes(a.w),
            a.n as u32,
            a.k as u32,
            t as u32,
            ptr(a.parts),
            nbytes(a.parts),
            opt_ptr(a.parts2),
            opt_nbytes(a.parts2),
            a.xks,
            a.kind,
            0.25,
            ptr_mut(&mut yf),
            nbytes(&yf),
            ptr_mut(&mut yr),
            nbytes(&yr),
            ptr_mut(&mut xf),
            nbytes(&xf),
            ptr_mut(&mut xr),
            nbytes(&xr),
            a.grid,
        )
    };
    assert_eq!(rc, 0, "mk_qmmh_xin");
    assert!(all_finite(&xr), "the staged x of the stages is not finite");
    assert_same(&xf, &xr, "the staged x: fused values against the stages they replace");
    (yf, yr)
}

#[test]
fn fused_staging_bit_identical_to_the_stages_it_replaces() {
    for (c, &(label, rows, kind, ks, k, seed)) in XIN_CASES.iter().enumerate() {
        let raw = inputs::raw_rows(Qtype::Q8_0, seed, rows, k);
        for tiled in [true, false] {
            let (qtype, w) = weights(&raw, rows, k, tiled);
            for (m, ntok) in [1usize, 2, 5, 16, 17, 40].into_iter().enumerate() {
                // Partials of the size that real ones have: a K slice of a dot product of 2560 or 10240 terms.
                let s = 1000 + 100 * c as u64 + 10 * m as u64;
                let parts = inputs::normals_scaled(s, ks * ntok * k, 2.0);
                let parts2 = (kind == 2).then(|| inputs::normals_scaled(s + 1, ks * ntok * k, 2.0));
                let (yf, yr) = xin(&Xin {
                    w: &w,
                    qtype,
                    n: rows,
                    k,
                    parts: &parts,
                    parts2: parts2.as_deref(),
                    xks: ks as u32,
                    kind,
                    grid: 0,
                });
                assert!(all_finite(&yr) && all_finite(&yf), "T={ntok}: unwritten outputs");
                assert_same(&yf, &yr, &format!("{label} kind {kind} tiled={tiled} T={ntok}"));
            }
        }
    }
}

#[test]
fn fused_staging_grid_independent() {
    let (rows, k) = (48usize, 640usize);
    let raw = inputs::raw_rows(Qtype::Q8_0, 812, rows, k);
    let (qtype, w) = weights(&raw, rows, k, true);
    let parts = inputs::normals_scaled(1500, 8 * 3 * k, 2.0);
    let parts2 = inputs::normals_scaled(1501, 8 * 3 * k, 2.0);
    let run = |grid| {
        xin(&Xin {
            w: &w,
            qtype,
            n: rows,
            k,
            parts: &parts,
            parts2: Some(&parts2),
            xks: 8,
            kind: 2,
            grid,
        })
        .0
    };
    let reference = run(0);
    for grid in [1, 7, 80] {
        assert_same(&run(grid), &reference, &format!("grid {grid}"));
    }
}

// ---- test_qmmh_hcmix.py ----

/// The rank of hc up, which is the K of the fused stage.
const HC_K: usize = 320;

struct HcMix<'a> {
    qtype: u32,
    w: &'a [u8],
    e: usize,
    t: usize,
    x: Option<&'a [f32]>,
    xn: &'a [f32],
    z: Option<&'a [f32]>,
    parts: Option<(&'a [f32], u32)>,
    grid: u32,
}

/// The fused stage and the matmul then `op_hc_mix`: `mk_qmmh_hcmix`. The outputs start as NaN, so an element that the
/// fused stage does not write fails the comparison. It returns `(mixed_f, inject_f, mixed_r, inject_r)`.
fn hcmix(a: &HcMix) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (e, t) = (a.e, a.t);
    let (mut mf, mut mr) = (vec![f32::NAN; t * e], vec![f32::NAN; t * e]);
    let (mut jf, mut jr) = (vec![f32::NAN; t * 4], vec![f32::NAN; t * 4]);
    let parts = a.parts.map(|p| p.0);
    let f = sym!(
        lib("qmmh"),
        "mk_qmmh_hcmix",
        fn(
            u32,
            *const c_void,
            u64,
            u32,
            u32,
            u32,
            *const c_void,
            u64,
            *const c_void,
            u64,
            u32,
            f32,
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
            *mut c_void,
            u64,
            u32,
        ) -> i32
    );
    let rc = unsafe {
        f(
            a.qtype,
            ptr(a.w),
            nbytes(a.w),
            HC_K as u32,
            e as u32,
            t as u32,
            opt_ptr(a.x),
            opt_nbytes(a.x),
            opt_ptr(parts),
            opt_nbytes(parts),
            a.parts.map_or(0, |p| p.1),
            0.25,
            ptr(a.xn),
            nbytes(a.xn),
            opt_ptr(a.z),
            opt_nbytes(a.z),
            ptr_mut(&mut mf),
            nbytes(&mf),
            ptr_mut(&mut jf),
            nbytes(&jf),
            ptr_mut(&mut mr),
            nbytes(&mr),
            ptr_mut(&mut jr),
            nbytes(&jr),
            a.grid,
        )
    };
    assert_eq!(rc, 0, "mk_qmmh_hcmix");
    (mf, jf, mr, jr)
}

/// The fused stage of hc up and hc_mix equals, bit for bit, the plain f16 matmul that writes `up` and then `op_hc_mix`.
/// Row n of the matmul is the same computation wherever it sits in the item, and the epilogue combines the four streams
/// in stream order, so the float operations of each element are the same.
#[test]
fn fused_hc_up_mix_bit_identical_to_matmul_then_hc_mix() {
    // `e` = 16 is one row group, so only the item of group 0 writes the inject.
    for e in [16usize, 64, 2560] {
        let raw = inputs::raw_rows(Qtype::Q8_0, 1000 + e as u64, 4 * e, HC_K);
        for tiled in [true, false] {
            let (qtype, w) = weights(&raw, 4 * e, HC_K, tiled);
            for with_z in [true, false] {
                for (m, t) in [1usize, 3, 16, 17, 50, 64, 65].into_iter().enumerate() {
                    let s = 2000 + 100 * e as u64 + 10 * m as u64;
                    let x = x_of(s, t, HC_K);
                    let xn = inputs::normals(s + 1, t * 4 * e);
                    let z = with_z.then(|| inputs::normals_scaled(s + 2, t * 4, 3.0));
                    for grid in [1u32, 7, 80] {
                        if e == 2560 && grid == 1 && t > 17 {
                            continue; // 160 items on one workgroup at 65 tokens: slow, adds nothing
                        }
                        let (mf, jf, mr, jr) = hcmix(&HcMix {
                            qtype,
                            w: &w,
                            e,
                            t,
                            x: Some(&x),
                            xn: &xn,
                            z: z.as_deref(),
                            parts: None,
                            grid,
                        });
                        let tag = format!("E={e} tiled={tiled} z={with_z} T={t} grid={grid}");
                        assert!(all_finite(&mr), "{tag}: the reference left a NaN");
                        assert_same(&mf, &mr, &format!("{tag}: mixed"));
                        if with_z {
                            assert!(all_finite(&jr), "{tag}: the reference left a NaN in inject");
                            assert_same(&jf, &jr, &format!("{tag}: inject"));
                        } else {
                            assert!(
                                jf.iter().all(|v| v.is_nan()) && jr.iter().all(|v| v.is_nan()),
                                "{tag}: inject written without z"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The decode form: x staged from the 16 split-K partials of hc down (`QmmArgs::xin`), with one token.
#[test]
fn fused_hc_up_mix_with_xin_silu_sum_from_split_k_partials() {
    let e = 2560usize;
    let raw = inputs::raw_rows(Qtype::Q8_0, 77, 4 * e, HC_K);
    for tiled in [true, false] {
        let (qtype, w) = weights(&raw, 4 * e, HC_K, tiled);
        let parts = inputs::normals_scaled(3000, 16 * HC_K, 2.0);
        let xn = inputs::normals(3001, 4 * e);
        let z = inputs::normals(3002, 4);
        let (mf, jf, mr, jr) = hcmix(&HcMix {
            qtype,
            w: &w,
            e,
            t: 1,
            x: None,
            xn: &xn,
            z: Some(&z),
            parts: Some((&parts, 16)),
            grid: 80,
        });
        assert!(
            all_finite(&mr) && all_finite(&jr),
            "tiled={tiled}: the reference left a NaN"
        );
        assert_same(&mf, &mr, &format!("tiled={tiled}: mixed"));
        assert_same(&jf, &jr, &format!("tiled={tiled}: inject"));
    }
}

// ---- test_hc_fused.py ----

/// `mk_hc_combine_rms`: `h += out x inject`, then the RMS norm of `h` by groups of `e`. `fused` runs the one fused stage.
fn combine_rms(
    h: &[f32],
    out: &[f32],
    inj: &[f32],
    gamma: &[f32],
    e: usize,
    grid: u32,
    fused: u32,
) -> (Vec<f32>, Vec<f32>) {
    let t = h.len() / (4 * e);
    let (mut h_out, mut xn) = (vec![0f32; h.len()], vec![0f32; h.len()]);
    let f = sym!(
        lib("small"),
        "mk_hc_combine_rms",
        fn(
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            u32,
            u32,
            u32,
            u32,
            *mut c_void,
            *mut c_void,
        ) -> i32
    );
    let rc = unsafe {
        f(
            ptr(h),
            ptr(out),
            ptr(inj),
            ptr(gamma),
            t as u32,
            e as u32,
            grid,
            fused,
            ptr_mut(&mut h_out),
            ptr_mut(&mut xn),
        )
    };
    assert_eq!(rc, 0, "mk_hc_combine_rms");
    (h_out, xn)
}

/// The fused item is one token and one group of streams. It reduces exactly the values it combined, in the order of the
/// RMS op, so `h` and `xn` equal the two ops in sequence bit for bit. There is no tolerance. Random data makes every element
/// distinct, so a wrong offset of the group, a wrong stream or a changed order of the sum changes bits.
#[test]
fn fused_equals_combine_then_rms() {
    const E: usize = 2560;
    for t in [1usize, 2, 3, 5, 8, 16, 17, 40] {
        for grid in [1u32, 3, 7, 80] {
            let s = 4000 + 100 * t as u64 + grid as u64;
            let h = inputs::normals_scaled(s, t * 4 * E, 3.0);
            let out = inputs::normals(s + 1000, t * E);
            let inj: Vec<f32> = inputs::uniforms(s + 2000, t * 4).iter().map(|v| v * 2.0).collect();
            let gamma: Vec<f32> = inputs::normals(s + 3000, 4 * E).iter().map(|v| 1.0 + 0.1 * v).collect();
            let (h0, x0) = combine_rms(&h, &out, &inj, &gamma, E, 64, 0);
            let (h1, x1) = combine_rms(&h, &out, &inj, &gamma, E, grid, 1);
            assert!(
                all_finite(&x0) && !same_bits(&h0, &h),
                "T={t}: the unfused reference did nothing"
            );
            assert_same(&h1, &h0, &format!("T={t} grid={grid}: h"));
            assert_same(&x1, &x0, &format!("T={t} grid={grid}: xn"));
        }
    }
}

// ---- test_hc.py ----

/// The d of the Q8_0 weights of the hc pre test: `tools/op_groups/hc.py` has the same two values. A small d keeps the
/// activations of the module out of the flat part of the sigmoid.
const D_DOWN: u16 = 0x0C00;
const D_UP: u16 = 0x1C00;

fn fixed_d(mut raw: Vec<u8>, bits: u16) -> Vec<u8> {
    for block in raw.as_chunks_mut::<34>().0.iter_mut() {
        block[..2].copy_from_slice(&bits.to_le_bytes());
    }
    raw
}

/// The hyper-connection pre stage against the HF module `Qwen4ExpTextGatedResidual` (`mixed` and `inject`).
///
/// mk used the real weights of layer 0 and the embeddings of random tokens. This test uses synthetic weights at the real
/// shapes, which the generator makes, and the fixture holds the output of the HF module for them. The tolerance is mk's:
/// f32 accumulation has a typical error of `sqrt(K) 2^-24` (about 6e-6 at K = 10240), four stages with Lipschitz 1 or less
/// stay under 3e-5, and 1e-4 allows for that. A wrong stream, gamma or weight is an error of order 1.
#[test]
fn hc_pre_matches_reference() {
    const E: usize = 2560;
    const R: usize = 320;
    const D4: usize = 4 * E;
    const TOL: f64 = 1e-4;
    let fixture = Fixture::load("hc");
    let t = fixture.meta("pre")["t"].as_u64().unwrap() as usize;
    assert_eq!(
        fixture.meta("pre")["d_down"],
        u64::from(D_DOWN),
        "the fixture and the test use the same d"
    );
    assert_eq!(
        fixture.meta("pre")["d_up"],
        u64::from(D_UP),
        "the fixture and the test use the same d"
    );
    let down = fixed_d(inputs::raw_rows(Qtype::Q8_0, 701, R, D4), D_DOWN);
    let up = fixed_d(inputs::raw_rows(Qtype::Q8_0, 702, D4, R), D_UP);
    let inject = inputs::normals_scaled(703, 4 * D4, 0.01);
    let gamma: Vec<f32> = inputs::normals(704, D4).iter().map(|v| 1.0 + v * 0.1).collect();
    let base = inputs::normals(705, t * E);
    let noise = inputs::normals(706, t * D4);
    let h: Vec<f32> = (0..t * D4)
        .map(|i| base[i / D4 * E + i % D4 % E] + 0.01 * noise[i])
        .collect();
    let (mut mixed, mut inj, mut xn) = (vec![0f32; t * E], vec![0f32; t * 4], vec![0f32; t * D4]);
    let f = sym!(
        lib("small"),
        "mk_hc_pre",
        fn(
            *const c_void,
            u32,
            u32,
            u32,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> i32
    );
    let rc = unsafe {
        f(
            ptr(&h),
            t as u32,
            E as u32,
            R as u32,
            ptr(&gamma),
            ptr(&down),
            ptr(&up),
            ptr(&inject),
            ptr_mut(&mut mixed),
            ptr_mut(&mut inj),
            ptr_mut(&mut xn),
        )
    };
    assert_eq!(rc, 0, "mk_hc_pre");
    let (e_mix, e_inj) = (
        mkops::relerr(&mixed, &fixture.f32_flat("pre_mixed")),
        mkops::relerr(&inj, &fixture.f32_flat("pre_inject")),
    );
    eprintln!("mixed relerr {e_mix:.2e}, inject relerr {e_inj:.2e}");
    assert!(
        e_mix < TOL && e_inj < TOL,
        "mixed {e_mix:.2e}, inject {e_inj:.2e}, tolerance {TOL:.0e}"
    );
}

/// The combine is one multiply and one add for each element in both torch and the op, so the results are equal in bits.
#[test]
fn hc_combine_exact() {
    const E: usize = 2560;
    let fixture = Fixture::load("hc");
    let t = 2usize;
    let mut h = inputs::normals(710, t * 4 * E);
    let out = inputs::normals(711, t * E);
    let inj = inputs::normals(712, t * 4);
    let f = sym!(
        lib("small"),
        "mk_hc_combine",
        fn(*mut c_void, *const c_void, *const c_void, u32, u32) -> i32
    );
    assert_eq!(
        unsafe { f(ptr_mut(&mut h), ptr(&out), ptr(&inj), t as u32, E as u32) },
        0
    );
    assert_same(&h, &fixture.f32_flat("combine"), "hc_combine against torch");
}
