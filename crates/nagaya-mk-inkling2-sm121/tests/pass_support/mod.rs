//! Helpers of the gates of the runs of the pass: the reader of the tape of the quantized reference, the pair lists of the
//! experts, and the counts of the flips of quantized bytes. The tape is a self-contained realization of the CPU reference:
//! every gate reads the inputs of a run from it, and never from the layers file of the golden.

#![allow(dead_code)]
#![cfg(feature = "cuda")]

use std::fs::File;
use std::path::PathBuf;

use nagaya_mk_cuda::experts::{ExpertFormat, QuantRowsLayout};
use nagaya_mk_cuda::protocol::{EXPERT_MAX_ROWS, ExpertPair, TOP_K};
use nagaya_mk_cuda::safetensors::SafeTensorsHeader;

pub const TAPE_FILE: &str = "inkling2-mini.quant.tape.safetensors";

pub struct Tape {
    header: SafeTensorsHeader,
    file: File,
    pub formats: Vec<ExpertFormat>,
}

/// The tape, or `None` when it is absent and the test skips. With `NAGAYA_REQUIRE_MODELS` an absence fails.
pub fn tape(who: &str) -> Option<Tape> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = std::env::var_os("INKLING2_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/inkling2-golden")));
    let path = dir.join(TAPE_FILE);
    if !path.exists() {
        assert!(
            !require,
            "{} is absent and NAGAYA_REQUIRE_MODELS is set",
            path.display()
        );
        eprintln!("{who}: skipped, {} is absent", path.display());
        return None;
    }
    let header = SafeTensorsHeader::read(&path).unwrap_or_else(|e| panic!("{e}"));
    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    // The formats of the runs are in the metadata: "nvfp4" or "mxfp8".
    let raw = std::fs::read(&path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let meta: serde_json::Value = serde_json::from_str(json["__metadata__"]["meta"].as_str().unwrap()).unwrap();
    let formats = meta["run_formats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| match f.as_str().unwrap() {
            "nvfp4" => ExpertFormat::Nvfp4,
            "mxfp8" => ExpertFormat::Mxfp8,
            other => panic!("the tape has the format {other}"),
        })
        .collect();
    Some(Tape { header, file, formats })
}

impl Tape {
    pub fn bytes(&mut self, key: &str) -> Vec<u8> {
        self.header
            .read_tensor(&mut self.file, key)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn f32s(&mut self, key: &str) -> Vec<f32> {
        self.bytes(key)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    pub fn i32s(&mut self, key: &str) -> Vec<i32> {
        self.bytes(key)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect()
    }

    pub fn ids(&mut self) -> Vec<u32> {
        self.bytes("ids")
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as u32)
            .collect()
    }

    pub fn has(&self, key: &str) -> bool {
        self.header.tensors.contains_key(key)
    }
}

/// The list of pairs that the rules give for the picks `ids` (8 for each row, ascending) and their weights: one pair for
/// each expert and each chunk of 16 rows, in ascending expert id; `y_row` is `8 t + j` for the pick `j` of the row `t`.
pub fn expected_pairs(ids: &[u32], weights: &[f32]) -> Vec<ExpertPair> {
    let rows = ids.len() / TOP_K;
    let mut out = Vec::new();
    for e in 0..128u32 {
        let hits: Vec<(usize, usize)> = (0..rows)
            .filter_map(|r| {
                ids[r * TOP_K..(r + 1) * TOP_K]
                    .iter()
                    .position(|x| *x == e)
                    .map(|j| (r, j))
            })
            .collect();
        for chunk in hits.chunks(EXPERT_MAX_ROWS) {
            let mut pair = ExpertPair {
                expert: e,
                count: chunk.len() as u32,
                ..Default::default()
            };
            for (i, (r, j)) in chunk.iter().enumerate() {
                pair.x_row[i] = *r as u16;
                pair.y_row[i] = (r * TOP_K + j) as u16;
                pair.weight[i] = weights[r * TOP_K + j];
            }
            out.push(pair);
        }
    }
    out
}

/// Whether two lists have the same experts, counts and rows, and the worst relative difference of the weights.
pub fn pairs_close(a: &[ExpertPair], b: &[ExpertPair]) -> (bool, f64) {
    let mut worst = 0.0f64;
    let same = a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| {
            let ok = p.expert == q.expert
                && p.count == q.count
                && (0..p.count as usize).all(|i| p.x_row[i] == q.x_row[i] && p.y_row[i] == q.y_row[i]);
            for i in 0..p.count as usize {
                let (x, y) = (f64::from(p.weight[i]), f64::from(q.weight[i]));
                worst = worst.max((x - y).abs() / y.abs().max(1e-30));
            }
            ok
        });
    (same, worst)
}

/// The live pairs that the device wrote in a pair-list buffer.
pub fn parse_pairs(raw: &[u8]) -> Vec<ExpertPair> {
    let word = |at: usize| u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
    let live = word(0) as usize;
    let size = std::mem::size_of::<ExpertPair>();
    (0..live)
        .map(|i| {
            let base = 16 + i * size;
            let mut p = ExpertPair {
                expert: word(base),
                count: word(base + 4),
                ..Default::default()
            };
            for j in 0..EXPERT_MAX_ROWS {
                p.x_row[j] = u16::from_le_bytes(raw[base + 8 + 2 * j..][..2].try_into().unwrap());
                p.y_row[j] = u16::from_le_bytes(raw[base + 40 + 2 * j..][..2].try_into().unwrap());
                p.weight[j] = f32::from_bits(word(base + 72 + 4 * j));
            }
            p
        })
        .collect()
}

pub fn same_pair(a: &ExpertPair, b: &ExpertPair) -> bool {
    a.expert == b.expert
        && a.count == b.count
        && (0..a.count as usize).all(|i| {
            a.x_row[i] == b.x_row[i] && a.y_row[i] == b.y_row[i] && a.weight[i].to_bits() == b.weight[i].to_bits()
        })
}

/// The rows `0..take` of a buffer of quantized rows in the layout `from`, as a buffer in the layout `to`.
pub fn convert_quant(bytes: &[u8], from: &QuantRowsLayout, to: &QuantRowsLayout, take: usize) -> Vec<u8> {
    assert_eq!((from.format, from.k), (to.format, to.k));
    let (code, scale) = (from.format.code_bytes(from.k), from.format.scale_bytes(from.k));
    let mut out = vec![0u8; to.bytes()];
    for r in 0..take {
        out[to.codes_offset(r)..][..code].copy_from_slice(&bytes[from.codes_offset(r)..][..code]);
        out[to.scales_offset(r)..][..scale].copy_from_slice(&bytes[from.scales_offset(r)..][..scale]);
        if from.format == ExpertFormat::Nvfp4 {
            out[to.amax_offset(r)..][..4].copy_from_slice(&bytes[from.amax_offset(r)..][..4]);
        }
    }
    out
}

/// The flips between two buffers of quantized rows of the same layout: the count of codes that differ (a nibble for
/// NVFP4, a byte for MXFP8), the count of scale bytes that differ, and the count of amax words that differ.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Flips {
    pub codes: usize,
    pub scales: usize,
    pub amax: usize,
    pub total_codes: usize,
}

pub fn count_flips(a: &[u8], b: &[u8], layout: &QuantRowsLayout, rows: usize) -> Flips {
    let (code, scale) = (layout.format.code_bytes(layout.k), layout.format.scale_bytes(layout.k));
    let mut f = Flips::default();
    for r in 0..rows {
        let (ca, cb) = (
            &a[layout.codes_offset(r)..][..code],
            &b[layout.codes_offset(r)..][..code],
        );
        match layout.format {
            ExpertFormat::Nvfp4 => {
                f.codes += ca
                    .iter()
                    .zip(cb)
                    .map(|(x, y)| usize::from(x & 0xF != y & 0xF) + usize::from(x >> 4 != y >> 4))
                    .sum::<usize>();
                f.total_codes += 2 * code;
            }
            ExpertFormat::Mxfp8 => {
                f.codes += ca.iter().zip(cb).filter(|(x, y)| x != y).count();
                f.total_codes += code;
            }
        }
        f.scales += a[layout.scales_offset(r)..][..scale]
            .iter()
            .zip(&b[layout.scales_offset(r)..][..scale])
            .filter(|(x, y)| x != y)
            .count();
        if layout.format == ExpertFormat::Nvfp4 {
            // The amax is a float of the row: a small change of the row moves it by a few ulps, and that is no flip.
            let (x, y) = (
                f32::from_le_bytes(a[layout.amax_offset(r)..][..4].try_into().unwrap()),
                f32::from_le_bytes(b[layout.amax_offset(r)..][..4].try_into().unwrap()),
            );
            f.amax += usize::from((x - y).abs() > 1e-4 * y.abs().max(1e-30));
        }
    }
    f
}

/// The differences of two f32 slices, as the worst absolute error, and the rms of the reference. The bound of a gate is
/// the first over the second.
#[derive(Debug, Clone, Copy)]
pub struct Gap {
    pub max_abs: f64,
    pub rms_ref: f64,
    pub at: usize,
}

impl Gap {
    pub fn relative(&self) -> f64 {
        self.max_abs / self.rms_ref.max(1e-30)
    }
}

pub fn gap(got: &[f32], want: &[f32]) -> Gap {
    assert_eq!(got.len(), want.len());
    let mut worst = Gap {
        max_abs: 0.0,
        rms_ref: 0.0,
        at: 0,
    };
    let mut sum = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (f64::from(*g) - f64::from(*w)).abs();
        assert!(g.is_finite(), "a non-finite value at {i}");
        if d > worst.max_abs {
            worst.max_abs = d;
            worst.at = i;
        }
        sum += f64::from(*w) * f64::from(*w);
    }
    worst.rms_ref = (sum / got.len().max(1) as f64).sqrt();
    worst
}

pub fn row_factor(x: &[f32]) -> f32 {
    let mean = nagaya_mk_inkling2_bmg::host::ops::dot(x, x) / x.len() as f32;
    1.0 / (mean + nagaya_mk_inkling2_bmg::host::ops::EPS).sqrt()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect()
}
