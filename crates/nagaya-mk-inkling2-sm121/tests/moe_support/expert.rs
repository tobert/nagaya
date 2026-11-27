//! The expert matrices, the quantized rows and the host dot products of the gates of the up and down ops.

#![allow(dead_code)]
#![cfg(feature = "cuda")]

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{EXPERT_MAX_ROWS, ExpertPair};

use super::Checkpoint;

use nagaya_mk_cuda::experts::{ExpertFormat, QuantRowsLayout, repack_expert, repack_expert_up};
use nagaya_mk_inkling2_sm121::quant::{
    E2M1, Mxfp8Row, Nvfp4Row, decode_e4m3, decode_ue8m0, dot_mxfp8_block_scaled, dot_nvfp4_block_scaled,
    quantize_mxfp8_ue8m0, quantize_nvfp4_fouroversix_mae,
};

/// The experts of the gate. They are at the start, the middle and the end of the 128.
pub const EXPERTS: [usize; 8] = [0, 17, 31, 42, 64, 85, 99, 127];

/// The checkpoint rows of one expert matrix: `w13` has 2 * 1152 rows, and `w2` has 2304.
pub const N: usize = 2304;

/// The weights of a matrix: what the gate needs of one matrix of the layer.
pub struct Matrix {
    pub format: ExpertFormat,
    pub k: usize,
    /// The codes and scales of each expert of `EXPERTS`, as the checkpoint has them.
    pub codes: Vec<Vec<u8>>,
    pub scales: Vec<Vec<u8>>,
    /// NVFP4 only: the amax of each expert, and the denominator of the matrix.
    pub amax: Vec<f32>,
    pub denominator: f32,
    pub name: String,
}

impl Checkpoint {
    /// One matrix of the experts of `layer`: `w13` or `w2`.
    pub fn matrix(&mut self, layer: usize, which: &str, format: ExpertFormat, k: usize) -> Matrix {
        let prefix = format!("model.llm.layers.{layer}.mlp.experts.{which}");
        let (code_row, scale_row) = (format.code_bytes(k), format.scale_bytes(k));
        let mut m = Matrix {
            format,
            k,
            codes: Vec::new(),
            scales: Vec::new(),
            amax: Vec::new(),
            denominator: 1.0,
            name: format!("layer {layer} {which}"),
        };
        for &e in &EXPERTS {
            m.codes
                .push(self.range(&format!("{prefix}_weight"), e * N * code_row, N * code_row));
            m.scales
                .push(self.range(&format!("{prefix}_weight_scale"), e * N * scale_row, N * scale_row));
            if format == ExpertFormat::Nvfp4 {
                m.amax.push(self.f32s(&format!("{prefix}_weight_amax"), e, 1)[0]);
            }
        }
        if format == ExpertFormat::Nvfp4 {
            m.denominator = self.f32s(&format!("{prefix}_weight_outer_scale_denominator"), 0, 1)[0];
        }
        m
    }
}

pub enum QuantRow {
    Nvfp4(Nvfp4Row),
    Mxfp8(Mxfp8Row),
}

pub fn quantize_row(format: ExpertFormat, row: &[f32]) -> QuantRow {
    match format {
        ExpertFormat::Nvfp4 => QuantRow::Nvfp4(quantize_nvfp4_fouroversix_mae(row).unwrap()),
        ExpertFormat::Mxfp8 => QuantRow::Mxfp8(quantize_mxfp8_ue8m0(row).unwrap()),
    }
}

/// The bytes of a buffer of quantized rows, in the layout of the quantizer op.
pub fn quant_rows_bytes(layout: &QuantRowsLayout, rows: &[QuantRow]) -> Vec<u8> {
    let mut bytes = vec![0u8; layout.bytes()];
    for (r, row) in rows.iter().enumerate() {
        let (codes, scales) = match row {
            QuantRow::Nvfp4(q) => (&q.codes, &q.scales),
            QuantRow::Mxfp8(q) => (&q.codes, &q.scales),
        };
        bytes[layout.codes_offset(r)..][..codes.len()].copy_from_slice(codes);
        bytes[layout.scales_offset(r)..][..scales.len()].copy_from_slice(scales);
        if let QuantRow::Nvfp4(q) = row {
            bytes[layout.amax_offset(r)..][..4].copy_from_slice(&q.amax.to_le_bytes());
        }
    }
    bytes
}

/// The nibble `i` of a packed row.
fn nibble(codes: &[u8], i: usize) -> usize {
    usize::from(if i.is_multiple_of(2) {
        codes[i / 2] & 0xF
    } else {
        codes[i / 2] >> 4
    })
}

/// The host dot product of `quant.rs` for the checkpoint row `col` of the expert `e`, and the sum of the absolute terms in
/// the units of the result.
pub fn host_dot(m: &Matrix, e: usize, col: usize, row: &QuantRow) -> (f32, f64) {
    let k = m.k;
    let (code_row, scale_row) = (m.format.code_bytes(k), m.format.scale_bytes(k));
    let w_codes = &m.codes[e][col * code_row..(col + 1) * code_row];
    let w_scales = &m.scales[e][col * scale_row..(col + 1) * scale_row];
    match row {
        QuantRow::Nvfp4(q) => {
            let value = dot_nvfp4_block_scaled(q, w_codes, w_scales, m.amax[e], m.denominator, 1.0).unwrap();
            let mut total = 0.0f64;
            for b in 0..scale_row {
                let scale = f64::from(decode_e4m3(q.scales[b]).unwrap()) * f64::from(decode_e4m3(w_scales[b]).unwrap());
                let block: f64 = (b * 16..(b + 1) * 16)
                    .map(|i| f64::from(E2M1[nibble(&q.codes, i)] * E2M1[nibble(w_codes, i)]).abs())
                    .sum();
                total += block * scale.abs();
            }
            let row_scale = f64::from(q.amax) * f64::from(m.amax[e]) / (1536.0 * f64::from(m.denominator));
            (value, total * row_scale.abs())
        }
        QuantRow::Mxfp8(q) => {
            let value = dot_mxfp8_block_scaled(q, w_codes, w_scales, 1.0).unwrap();
            let mut total = 0.0f64;
            for b in 0..scale_row {
                let scale =
                    f64::from(decode_ue8m0(q.scales[b]).unwrap()) * f64::from(decode_ue8m0(w_scales[b]).unwrap());
                let block: f64 = (b * 32..(b + 1) * 32)
                    .map(|i| {
                        (f64::from(decode_e4m3(q.codes[i]).unwrap()) * f64::from(decode_e4m3(w_codes[i]).unwrap()))
                            .abs()
                    })
                    .sum();
                total += block * scale;
            }
            (value, total)
        }
    }
}

/// The bank on the device, and what the op needs of the matrix.
pub struct Bank {
    pub ptr: CUdeviceptr,
    pub amax: CUdeviceptr,
    pub bytes: usize,
}

/// Upload the bank of the matrix. `up` repacks `w13` for `FLAG_UP`.
pub fn upload_bank(dev: &CudaDevice, m: &Matrix, up: bool) -> Bank {
    let mut bytes = Vec::new();
    for (codes, scales) in m.codes.iter().zip(&m.scales) {
        bytes.extend(if up {
            repack_expert_up(m.format, N, m.k, codes, scales).unwrap()
        } else {
            repack_expert(m.format, N, m.k, codes, scales).unwrap()
        });
    }
    let amax = if m.format == ExpertFormat::Nvfp4 {
        crate::common::gpu::upload(dev, &m.amax)
    } else {
        0
    };
    Bank {
        ptr: crate::common::gpu::upload(dev, &bytes),
        amax,
        bytes: bytes.len(),
    }
}

/// A pair for the op: the bank index of the expert, the rows with the slots that they write, and the route weights.
pub fn pair(expert: usize, x_rows: &[usize], y_rows: &[usize], weights: &[f32]) -> ExpertPair {
    assert!(!x_rows.is_empty() && x_rows.len() <= EXPERT_MAX_ROWS);
    assert!(x_rows.len() == y_rows.len() && x_rows.len() == weights.len());
    let mut p = ExpertPair {
        expert: expert as u32,
        count: x_rows.len() as u32,
        ..Default::default()
    };
    for i in 0..x_rows.len() {
        p.x_row[i] = x_rows[i] as u16;
        p.y_row[i] = y_rows[i] as u16;
        p.weight[i] = weights[i];
    }
    p
}

/// The pairs of `tokens` tokens that each use the 8 experts of the bank in the order of the bank (so the slot `j` of a
/// token is the bank expert `j`). Each expert takes the tokens in chunks of 16. The route weight of a token and an expert
/// comes from `weight`.
pub fn bank_pairs(tokens: usize, weight: impl Fn(usize, usize) -> f32) -> Vec<ExpertPair> {
    let mut pairs = Vec::new();
    for e in 0..8 {
        for chunk in (0..tokens).collect::<Vec<_>>().chunks(EXPERT_MAX_ROWS) {
            let y: Vec<usize> = chunk.iter().map(|t| t * 8 + e).collect();
            let w: Vec<f32> = chunk.iter().map(|t| weight(*t, e)).collect();
            pairs.push(pair(e, chunk, &y, &w));
        }
    }
    pairs
}

/// The words of a buffer, as the bits.
pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The expert index of a pair list (the words that ROUTE writes for the block form): for each of the 128 experts,
/// `first_pair | pairs << 16`. The pairs of an expert must be contiguous, as ROUTE writes them.
pub fn expert_index(pairs: &[ExpertPair]) -> Vec<u32> {
    let mut index = vec![0u32; 128];
    let mut e = 0;
    while e < pairs.len() {
        let expert = pairs[e].expert as usize;
        let mut n = 0;
        while e + n < pairs.len() && pairs[e + n].expert as usize == expert {
            n += 1;
        }
        assert_eq!(index[expert], 0, "the pairs of expert {expert} are not contiguous");
        index[expert] = e as u32 | (n as u32) << 16;
        e += n;
    }
    index
}

/// Pairs of an uneven route: expert `e` takes the token `t` when `keep(t, e)`, in chunks of 16 like ROUTE. The slot of
/// the row of token `t` is `t * 8 + e`.
pub fn uneven_pairs(
    tokens: usize,
    keep: impl Fn(usize, usize) -> bool,
    weight: impl Fn(usize, usize) -> f32,
) -> Vec<ExpertPair> {
    let mut pairs = Vec::new();
    for e in 0..8 {
        let rows: Vec<usize> = (0..tokens).filter(|t| keep(*t, e)).collect();
        for chunk in rows.chunks(EXPERT_MAX_ROWS) {
            let y: Vec<usize> = chunk.iter().map(|t| t * 8 + e).collect();
            let w: Vec<f32> = chunk.iter().map(|t| weight(*t, e)).collect();
            pairs.push(pair(e, chunk, &y, &w));
        }
    }
    pairs
}
