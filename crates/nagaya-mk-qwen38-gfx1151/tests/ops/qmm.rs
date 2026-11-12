//! GPU dequant and quantized matmul (`ops/quant.h`, `ops/qmm.h`). It ports `test_qmm.py` and `test_qmmh.py` of the
//! reference.
//!
//! - dequant: bit-identical to gguf-py, with the expected values in a fixture of `tools/op_fixtures.py`.
//! - qmm: `y = x W^T` against an f64 reference. The weights of the reference come from `qwen38_dequant`, which the
//!   dequant test holds to gguf-py. The bound is the one of the reference: `K * 2^-24 * sum_k |w_k x_k|` (Higham,
//!   recursive summation).
//! - the exact kernel equals the oracle kernel bit for bit, for every grid size and every row map.
//!
//! The reference used real rows of the Qwen3.8 GGUF. These tests use rows that `inputs::raw_rows` makes, with the same
//! types and shapes.

use crate::fixture::Fixture;
use crate::inputs::{self, Qtype};
use crate::ops_lib::{
    self, Kernel, QmmOptions, all_finite, assert_same, bit_diff, dequant, matmul_f64, qmm, qmm_fast, qmm_oracle,
    qmm_with, same_bits,
};
use crate::{child, sym};

/// One weight matrix: the type, the rows, the K, and the seed of its bytes. The label is the tensor of the reference's
/// test.
#[derive(Clone, Copy)]
pub struct Case {
    pub label: &'static str,
    pub qtype: Qtype,
    pub rows: usize,
    pub k: usize,
    pub seed: u64,
}

impl Case {
    pub fn raw(&self) -> Vec<u8> {
        inputs::raw_rows(self.qtype, self.seed, self.rows, self.k)
    }
}

/// `CASES` of `test_qmm.py`: one case for each type, and the IQ4_XS case that the reference finds in the file.
pub const CASES: [Case; 7] = [
    Case {
        label: "blk.0.attn_qkv.weight",
        qtype: Qtype::Q8_0,
        rows: 40,
        k: 2560,
        seed: 101,
    },
    Case {
        label: "output.weight",
        qtype: Qtype::Q6K,
        rows: 24,
        k: 2560,
        seed: 102,
    },
    Case {
        label: "blk.0.ffn_down_exps.weight",
        qtype: Qtype::Iq4Nl,
        rows: 24,
        k: 640,
        seed: 103,
    },
    Case {
        label: "blk.0.ffn_gate_exps.weight",
        qtype: Qtype::Iq3S,
        rows: 24,
        k: 2560,
        seed: 104,
    },
    Case {
        label: "blk.0.ffn_gate_inp.weight",
        qtype: Qtype::F32,
        rows: 16,
        k: 2560,
        seed: 105,
    },
    Case {
        label: "blk.3.indexer.q_proj.weight",
        qtype: Qtype::Bf16,
        rows: 16,
        k: 2560,
        seed: 106,
    },
    Case {
        label: "blk.2.ffn_gate_exps.weight (IQ4_XS)",
        qtype: Qtype::Iq4Xs,
        rows: 24,
        k: 2560,
        seed: 107,
    },
];

/// The cases of the f16 path: the quantized types only (`QUANT_CASES` of `test_qmmh.py`).
pub fn quant_cases() -> Vec<Case> {
    CASES
        .iter()
        .copied()
        .filter(|c| !matches!(c.qtype, Qtype::F32 | Qtype::Bf16))
        .collect()
}

/// The narrow Q8_0 shapes of the hyper-connection (`FAST_EDGES` of `test_qmmh.py`): rows that are not a multiple of 8,
/// below one item, and a K that ends on a whole 32-unit chunk.
pub const FAST_EDGES: [Case; 3] = [
    Case {
        label: "blk.3.hc_ffn_down.weight",
        qtype: Qtype::Q8_0,
        rows: 4,
        k: 10240,
        seed: 111,
    },
    Case {
        label: "blk.3.hc_ffn_down.weight",
        qtype: Qtype::Q8_0,
        rows: 13,
        k: 10240,
        seed: 112,
    },
    Case {
        label: "blk.3.hc_ffn_up.weight",
        qtype: Qtype::Q8_0,
        rows: 37,
        k: 320,
        seed: 113,
    },
];

/// Check that a case has the fixture values: dequantized values from gguf-py for the raw bytes of the generator.
#[test]
fn dequant_bit_identical() {
    let fixture = Fixture::load("qmm");
    let types = fixture.meta("dequant_cases").as_array().expect("the cases").clone();
    let mut seen = std::collections::BTreeSet::new();
    for entry in &types {
        let name = entry["type"].as_str().unwrap();
        let qtype = [
            Qtype::F32,
            Qtype::Q8_0,
            Qtype::Q6K,
            Qtype::Iq4Nl,
            Qtype::Iq3S,
            Qtype::Iq4Xs,
            Qtype::Bf16,
        ]
        .into_iter()
        .find(|q| q.name() == name)
        .expect("a known type");
        let (rows, k, seed) = (
            entry["rows"].as_u64().unwrap() as usize,
            entry["k"].as_u64().unwrap() as usize,
            entry["seed"].as_u64().unwrap(),
        );
        let raw = inputs::raw_rows(qtype, seed, rows, k);
        let want = fixture.f32_flat(&format!("dequant_{name}"));
        let got = dequant(qtype, &raw, rows, k);
        assert_same(&got, &want, &format!("{name} dequant against gguf-py"));
        seen.insert(name.to_owned());
    }
    let expect: std::collections::BTreeSet<String> = ["Q8_0", "Q6_K", "IQ4_NL", "IQ3_S", "IQ4_XS", "F32", "BF16"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    assert_eq!(seen, expect, "the fixture holds each type");
}

/// The f64 reference of a case: the weights from the kernel dequant, and `x W^T`.
struct Reference {
    w: Vec<f32>,
}

impl Reference {
    fn new(case: &Case, raw: &[u8]) -> Reference {
        Reference {
            w: dequant(case.qtype, raw, case.rows, case.k),
        }
    }
}

fn x_of(seed: u64, ntok: usize, k: usize) -> Vec<f32> {
    inputs::normals(seed, ntok * k)
}

#[test]
fn qmm_against_f64() {
    for (n, case) in CASES.iter().enumerate() {
        let raw = case.raw();
        let reference = Reference::new(case, &raw);
        for (m, ntok) in [1usize, 5, 19].into_iter().enumerate() {
            let x = x_of(2000 + 10 * n as u64 + m as u64, ntok, case.k);
            let y = qmm(case.qtype, &raw, case.rows, case.k, &x, 0);
            let (y64, mag) = matmul_f64(&x, &reference.w, ntok, case.rows, case.k);
            assert!(all_finite(&y), "{}: unwritten or non-finite outputs", case.label);
            let mut worst = 0f64;
            for i in 0..y.len() {
                let bound = case.k as f64 * 2f64.powi(-24) * mag[i];
                let err = (f64::from(y[i]) - y64[i]).abs();
                worst = worst.max(err / bound);
            }
            assert!(
                worst <= 1.0,
                "{} ({}) T={ntok}: worst err/bound {worst:.3}",
                case.label,
                case.qtype.name()
            );
        }
    }
}

#[test]
fn qmm_grid_independent_and_repeatable() {
    let case = Case {
        label: "blk.0.attn_qkv.weight",
        qtype: Qtype::Q8_0,
        rows: 64,
        k: 2560,
        seed: 121,
    };
    let raw = case.raw();
    let x = x_of(2100, 17, case.k);
    let reference = qmm(case.qtype, &raw, case.rows, case.k, &x, 0);
    for grid in [1, 7, 0] {
        assert_same(
            &qmm(case.qtype, &raw, case.rows, case.k, &x, grid),
            &reference,
            &format!("grid {grid}"),
        );
    }
    for repeat in 0..20 {
        assert_same(
            &qmm(case.qtype, &raw, case.rows, case.k, &x, 0),
            &reference,
            &format!("repeat {repeat}"),
        );
    }
}

/// The token counts cross the tile edges of 8 and 32 and reach the pass maximum (256).
const BIT_TOKS: [usize; 9] = [1, 5, 8, 9, 31, 32, 33, 70, 256];

#[test]
fn qmm_bit_identical_to_oracle() {
    for (n, case) in CASES.iter().enumerate() {
        let raw = case.raw();
        for (m, ntok) in BIT_TOKS.into_iter().enumerate() {
            let x = x_of(2200 + 20 * n as u64 + m as u64, ntok, case.k);
            let want = qmm_oracle(case.qtype, &raw, case.rows, case.k, &x, 0);
            let got = qmm(case.qtype, &raw, case.rows, case.k, &x, 0);
            assert!(all_finite(&want), "{}: oracle left unwritten outputs", case.label);
            assert_same(&got, &want, &format!("{} ({}) T={ntok}", case.label, case.qtype.name()));
        }
    }
}

/// A sorted list of `n` row numbers below `bound`, with repeats and gaps, like a token list for one expert.
fn sorted_rows(seed: u64, n: usize, bound: u64) -> Vec<u32> {
    let mut rows = inputs::integers(seed, n, bound);
    rows.sort_unstable();
    rows
}

fn gather(x: &[f32], rows: &[u32], k: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| x[r as usize * k..(r as usize + 1) * k].iter().copied())
        .collect()
}

#[test]
fn qmm_gathered_rows_bit_identical_to_oracle() {
    // The MoE path: x rows gathered through x_row.
    let cases = [
        Case {
            label: "blk.0.ffn_gate_exps.weight",
            qtype: Qtype::Iq3S,
            rows: 24,
            k: 2560,
            seed: 131,
        },
        Case {
            label: "blk.0.ffn_down_exps.weight",
            qtype: Qtype::Iq4Nl,
            rows: 24,
            k: 640,
            seed: 132,
        },
    ];
    for (n, case) in cases.iter().enumerate() {
        let raw = case.raw();
        let x = x_of(2300 + n as u64, 300, case.k);
        for (m, ntok) in [1usize, 6, 40].into_iter().enumerate() {
            let x_row = sorted_rows(2310 + 10 * n as u64 + m as u64, ntok, 300);
            let options = QmmOptions {
                x_row: Some(&x_row),
                ..Default::default()
            };
            let want = qmm_with(Kernel::Oracle, case.qtype, &raw, case.rows, case.k, &x, &options);
            let got = qmm_with(Kernel::Exact, case.qtype, &raw, case.rows, case.k, &x, &options);
            assert!(all_finite(&want));
            assert_same(&got, &want, &format!("{} T={ntok}: gathered rows", case.label));
            let plain = qmm_oracle(case.qtype, &raw, case.rows, case.k, &gather(&x, &x_row, case.k), 0);
            assert!(same_bits(&want, &plain), "x_row gather is not a plain gather");
        }
    }
}

#[test]
fn qmm_bit_identical_across_grids() {
    let case = Case {
        label: "blk.0.ffn_gate_exps.weight",
        qtype: Qtype::Iq3S,
        rows: 24,
        k: 2560,
        seed: 141,
    };
    let raw = case.raw();
    let x = x_of(2400, 70, case.k);
    let reference = qmm_oracle(case.qtype, &raw, case.rows, case.k, &x, 0);
    for grid in [1, 3, 40, 0] {
        assert_same(
            &qmm(case.qtype, &raw, case.rows, case.k, &x, grid),
            &reference,
            &format!("grid {grid}"),
        );
    }
}

#[test]
fn qmm_edges_bit_identical_to_oracle() {
    // Shapes of the engine that CASES does not hold (kaibo review, 2026-09-27): row counts that are not a multiple
    // of 8 and below one item (hc_inject is N=4: 7 of 8 waves have no row), K that ends on a whole 32-unit chunk
    // (10240 = 320 units), and K = 320 (10 units). Each runs with grid 1 too, so one workgroup takes every item.
    for (n, case) in FAST_EDGES.iter().enumerate() {
        let raw = case.raw();
        for (m, ntok) in [1usize, 9, 33].into_iter().enumerate() {
            let x = x_of(2500 + 10 * n as u64 + m as u64, ntok, case.k);
            let want = qmm_oracle(case.qtype, &raw, case.rows, case.k, &x, 0);
            assert!(all_finite(&want));
            for grid in [0, 1] {
                assert_same(
                    &qmm(case.qtype, &raw, case.rows, case.k, &x, grid),
                    &want,
                    &format!("{} N={} T={ntok} grid={grid}", case.label, case.rows),
                );
            }
        }
    }
}

#[test]
fn qmm_scatter_and_ldy_bit_identical_to_oracle() {
    // The MoE paths: op_moe_down scatters y rows through y_row, and op_moe_gu writes with ldy = 2 N. Outputs land where
    // the oracle puts them and nowhere else: the rest stays NaN in both.
    let cases = [
        Case {
            label: "blk.0.ffn_down_exps.weight",
            qtype: Qtype::Iq4Nl,
            rows: 24,
            k: 640,
            seed: 151,
        },
        Case {
            label: "blk.0.ffn_gate_exps.weight",
            qtype: Qtype::Iq3S,
            rows: 24,
            k: 2560,
            seed: 152,
        },
    ];
    for (n, case) in cases.iter().enumerate() {
        let raw = case.raw();
        for (m, ntok) in [3usize, 11, 40].into_iter().enumerate() {
            let x = x_of(2600 + 10 * n as u64 + m as u64, ntok, case.k);
            let y_row: Vec<u32> = inputs::permutation(2650 + 10 * n as u64 + m as u64, 100)[..ntok].to_vec();
            let options = QmmOptions {
                y_row: Some(&y_row),
                y_rows: Some(100),
                ldy: Some(2 * case.rows),
                ..Default::default()
            };
            let want = qmm_with(Kernel::Oracle, case.qtype, &raw, case.rows, case.k, &x, &options);
            let got = qmm_with(Kernel::Exact, case.qtype, &raw, case.rows, case.k, &x, &options);
            let written = want.iter().filter(|v| v.is_finite()).count();
            let ldy = 2 * case.rows;
            assert!(
                y_row
                    .iter()
                    .all(|&r| want[r as usize * ldy..r as usize * ldy + case.rows]
                        .iter()
                        .all(|v| v.is_finite()))
                    && written == ntok * case.rows,
                "oracle wrote outside its rows"
            );
            assert_same(
                &got,
                &want,
                &format!("{} T={ntok}: scatter and ldy outputs", case.label),
            );
            for grid in [1, 3] {
                let options = QmmOptions {
                    grid,
                    ..options.clone()
                };
                let got = qmm_with(Kernel::Exact, case.qtype, &raw, case.rows, case.k, &x, &options);
                assert_same(&got, &want, &format!("{} T={ntok} grid {grid}", case.label));
            }
        }
    }
}

/// The shapes of the stages that are exact in F32 and BF16 (the router logits, the hyper-connection inject, GDN alpha and
/// beta, the indexer, the shared-expert gate). The kernel must equal the oracle on the shapes the engine runs, at decode
/// token counts and a few larger. Router near-ties move the scoreboard, so there is no tolerance. The rows are the whole N.
pub const EXACT_SHAPES: [Case; 7] = [
    Case {
        label: "blk.0.ffn_gate_inp.weight",
        qtype: Qtype::F32,
        rows: 512,
        k: 2560,
        seed: 161,
    },
    Case {
        label: "blk.0.hc_ffn_inject.weight",
        qtype: Qtype::F32,
        rows: 4,
        k: 10240,
        seed: 162,
    },
    Case {
        label: "blk.0.ssm_alpha.weight",
        qtype: Qtype::F32,
        rows: 48,
        k: 2560,
        seed: 163,
    },
    Case {
        label: "blk.0.ssm_beta.weight",
        qtype: Qtype::F32,
        rows: 48,
        k: 2560,
        seed: 164,
    },
    Case {
        label: "blk.0.ffn_gate_inp_shexp.weight",
        qtype: Qtype::F32,
        rows: 1,
        k: 2560,
        seed: 165,
    },
    Case {
        label: "blk.3.indexer.k_proj.weight",
        qtype: Qtype::Bf16,
        rows: 128,
        k: 2560,
        seed: 166,
    },
    Case {
        label: "blk.3.indexer.q_proj.weight",
        qtype: Qtype::Bf16,
        rows: 512,
        k: 2560,
        seed: 167,
    },
];

#[test]
fn qmm_exact_engine_shapes_bit_identical_to_oracle() {
    for ntok in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 33, 64] {
        for case in &EXACT_SHAPES {
            let raw = case.raw();
            let x = x_of(100 + ntok as u64, ntok, case.k);
            let want = qmm_oracle(case.qtype, &raw, case.rows, case.k, &x, 0);
            assert!(all_finite(&want), "{}: oracle left unwritten outputs", case.label);
            for grid in [0, 1, 7, 80] {
                let got = qmm(case.qtype, &raw, case.rows, case.k, &x, grid);
                let (count, first) = bit_diff(&got, &want);
                assert!(
                    first.is_none(),
                    "{} ({}) T={ntok} grid={grid}: {count} outputs differ, first {:?}",
                    case.label,
                    case.qtype.name(),
                    first
                );
            }
        }
    }
}

#[test]
fn qmm_exact_row_end_and_x_row_gather() {
    // Clamped or prefetched loads must never feed the arithmetic. A short K slice of a wider row is in play: the first 320
    // columns (10 units, so lanes 10..31 own no unit) of the router rows. The test repacks the rows, because the test ABI
    // has no row stride.
    let case = EXACT_SHAPES[0];
    let whole = case.raw();
    for (n, k) in [320usize, 1024, 1568, 2080, 2112, 2560].into_iter().enumerate() {
        let x = x_of(2700 + n as u64, 40, k);
        let mut sub = Vec::with_capacity(case.rows * k * 4);
        for row in 0..case.rows {
            let start = row * case.k * 4;
            sub.extend_from_slice(&whole[start..start + k * 4]);
        }
        let want = qmm_oracle(Qtype::F32, &sub, case.rows, k, &x, 0);
        for grid in [0, 1, 7, 80] {
            assert_same(
                &qmm(Qtype::F32, &sub, case.rows, k, &x, grid),
                &want,
                &format!("k={k} grid={grid}"),
            );
        }
    }
}

// ---- test_qmmh.py: the f16 WMMA path ----

/// The bound of the f16 path from its numeric design: the weight codes are exact in f16 and the scales apply in f32.
/// The activations round once to f16 (relative error up to 2^-11). The products are exact in f32 and the sums are in f32.
fn f16_bound(k: usize, mag: f64) -> f64 {
    (2f64.powi(-11) + k as f64 * 2f64.powi(-24)) * mag
}

#[test]
fn qmm_fast_against_f64() {
    let mut cases = quant_cases();
    cases.push(CASES[6]);
    cases.extend(FAST_EDGES);
    let mut seen = std::collections::BTreeSet::new();
    for (n, case) in cases.iter().enumerate() {
        let raw = case.raw();
        let reference = Reference::new(case, &raw);
        for (m, ntok) in [1usize, 5, 16, 17, 70, 256].into_iter().enumerate() {
            let x = x_of(2800 + 10 * n as u64 + m as u64, ntok, case.k);
            let y = qmm_fast(case.qtype, &raw, case.rows, case.k, &x, 0);
            let (y64, mag) = matmul_f64(&x, &reference.w, ntok, case.rows, case.k);
            assert!(all_finite(&y), "{}: unwritten or non-finite outputs", case.label);
            let mut worst = 0f64;
            for i in 0..y.len() {
                worst = worst.max((f64::from(y[i]) - y64[i]).abs() / f16_bound(case.k, mag[i]));
            }
            assert!(
                worst <= 1.0,
                "{} ({}) N={} T={ntok}: worst err/bound {worst:.3}",
                case.label,
                case.qtype.name(),
                case.rows
            );
            // The f16 path is really taken: rounding the activations changes some bits against the exact kernel.
            let exact = qmm(case.qtype, &raw, case.rows, case.k, &x, 0);
            assert!(
                !same_bits(&y, &exact),
                "{} T={ntok}: fast=1 ran the exact kernel",
                case.label
            );
        }
        seen.insert(case.qtype.name());
    }
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        ["IQ3_S", "IQ4_NL", "IQ4_XS", "Q6_K", "Q8_0"]
    );
}

#[test]
fn qmm_fast_gathered_scattered() {
    let cases = [
        Case {
            label: "blk.0.ffn_gate_exps.weight",
            qtype: Qtype::Iq3S,
            rows: 24,
            k: 2560,
            seed: 171,
        },
        Case {
            label: "blk.0.ffn_down_exps.weight",
            qtype: Qtype::Iq4Nl,
            rows: 24,
            k: 640,
            seed: 172,
        },
    ];
    for (n, case) in cases.iter().enumerate() {
        let raw = case.raw();
        let x = x_of(2900 + n as u64, 300, case.k);
        for (m, ntok) in [1usize, 6, 40].into_iter().enumerate() {
            let x_row = sorted_rows(2910 + 10 * n as u64 + m as u64, ntok, 300);
            let y_row: Vec<u32> = inputs::permutation(2950 + 10 * n as u64 + m as u64, 100)[..ntok].to_vec();
            let ldy = 2 * case.rows;
            let options = QmmOptions {
                x_row: Some(&x_row),
                y_row: Some(&y_row),
                y_rows: Some(100),
                ldy: Some(ldy),
                ..Default::default()
            };
            let got = qmm_with(Kernel::Fast, case.qtype, &raw, case.rows, case.k, &x, &options);
            let plain = qmm_fast(case.qtype, &raw, case.rows, case.k, &gather(&x, &x_row, case.k), 0);
            // Nothing outside the scattered rows of N values is written.
            let mut outside = got.clone();
            for (t, &r) in y_row.iter().enumerate() {
                for j in 0..case.rows {
                    assert!(
                        got[r as usize * ldy + j].to_bits() == plain[t * case.rows + j].to_bits(),
                        "{} T={ntok}: gather and scatter changed a value",
                        case.label
                    );
                    outside[r as usize * ldy + j] = f32::NAN;
                }
            }
            assert!(
                outside.iter().all(|v| v.is_nan()),
                "{} T={ntok}: wrote outside",
                case.label
            );
        }
    }
}

#[test]
fn qmm_fast_grid_independent_and_repeatable() {
    let x = x_of(3000, 70, 2560);
    let cases = [
        Case {
            label: "blk.0.attn_qkv.weight",
            qtype: Qtype::Q8_0,
            rows: 40,
            k: 2560,
            seed: 181,
        },
        Case {
            label: "blk.0.ffn_gate_exps.weight",
            qtype: Qtype::Iq3S,
            rows: 24,
            k: 2560,
            seed: 182,
        },
    ];
    for case in cases {
        let raw = case.raw();
        let reference = qmm_fast(case.qtype, &raw, case.rows, 2560, &x, 0);
        for grid in [1, 3, 40] {
            assert_same(
                &qmm_fast(case.qtype, &raw, case.rows, 2560, &x, grid),
                &reference,
                &format!("{} grid {grid}", case.label),
            );
        }
        for repeat in 0..20 {
            assert_same(
                &qmm_fast(case.qtype, &raw, case.rows, 2560, &x, 0),
                &reference,
                &format!("{} repeat {repeat}", case.label),
            );
        }
    }
}

/// The budget of the megakernel at grid 80: no scratch, at most 192 VGPRs (8 waves a SIMD). The f16 tile lives in the
/// shared LDS arena, so the static LDS of `k_qmmh` is the arena alone.
#[test]
fn qmmh_resources() {
    let f = sym!(
        ops_lib::lib("qmmh"),
        "qwen38_qmmh_resources",
        fn(*mut i32, *mut i32, *mut i32) -> i32
    );
    let (mut regs, mut scratch, mut lds) = (0i32, 0i32, 0i32);
    assert_eq!(unsafe { f(&mut regs, &mut scratch, &mut lds) }, 0);
    eprintln!("k_qmmh: {regs} VGPRs, {scratch} B/lane scratch, {lds} B LDS");
    assert_eq!(scratch, 0, "k_qmmh spills: {scratch} B/lane");
    assert!(regs <= 192, "k_qmmh uses {regs} VGPRs");
    assert!(lds <= 9216, "k_qmmh LDS {lds} B: more than the arena");
}

/// An activation beyond the range of f16 must stop the program, and never become infinite. The trap aborts the process
/// that launched the kernel, so the launch runs in a child process.
#[test]
fn qmm_fast_overflow_traps() {
    let output = child::run("qmm::child_fast_overflow", &[]);
    assert!(
        !output.status.success() && !child::stdout(&output).contains("NO TRAP"),
        "no trap: {:?} {}",
        output.status,
        child::stdout(&output)
    );
}

#[test]
#[ignore = "child of qmm_fast_overflow_traps"]
fn child_fast_overflow() {
    let case = Case {
        label: "blk.0.attn_qkv.weight",
        qtype: Qtype::Q8_0,
        rows: 16,
        k: 2560,
        seed: 191,
    };
    let raw = case.raw();
    let mut x = vec![0f32; 2 * 2560];
    x[2560 + 7] = 70000.0;
    let _ = qmm_fast(case.qtype, &raw, 16, 2560, &x, 0);
    println!("NO TRAP");
}
