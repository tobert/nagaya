//! The dequantizer: GGML blocks to f32, bit for bit as the reference's C++ copy and ggml give them.
//!
//! Each body is a port of ggml's `dequantize_row_*` (`ggml/src/ggml-quants.c`, llama.cpp `5cf3a3528`), through the
//! reference's `src/gguf/dequant.cpp`. The ports keep ggml's float operations and their
//! order, so the bits are equal. The tests compare the bits with digests that the reference's C++ copy wrote. The tool
//! that wrote them, `tools/mk_dequant_digests.cpp`, and the C++ copy are in commit `7d474c2`; a later commit removed
//! both.
//!
//! The tables and the f16 conversion are verbatim copies of ggml (`ggml-common.h`, `ggml-impl.h`), MIT licensed:
//!
//! Copyright (c) 2023-2026 The ggml authors
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//! documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
//! rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//! permit persons to whom the Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all copies or substantial portions of
//! the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
//! WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
//! OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/// A GGML tensor type that the reader knows. The ids are the ids of a GGUF tensor record.
///
/// The Qwen3.8 UD-IQ4_XS shards hold these seven types only. The reader refuses any other id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GgmlType {
    F32,
    Bf16,
    Q8_0,
    Q6K,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
}

impl GgmlType {
    /// Every known type, in the order of the reference's C++ table.
    pub(crate) const ALL: [GgmlType; 7] = [
        GgmlType::F32,
        GgmlType::Bf16,
        GgmlType::Q8_0,
        GgmlType::Q6K,
        GgmlType::Iq3S,
        GgmlType::Iq4Nl,
        GgmlType::Iq4Xs,
    ];

    /// The type of a GGUF type id, or `None` for an id that the reader does not know.
    pub(crate) fn from_id(id: u32) -> Option<GgmlType> {
        GgmlType::ALL.into_iter().find(|kind| kind.id() == id)
    }

    /// The GGUF type id.
    pub(crate) fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::Q8_0 => 8,
            GgmlType::Q6K => 14,
            GgmlType::Iq4Nl => 20,
            GgmlType::Iq3S => 21,
            GgmlType::Iq4Xs => 23,
            GgmlType::Bf16 => 30,
        }
    }

    /// The name that ggml and the reference give the type.
    pub(crate) fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::Bf16 => "BF16",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q6K => "Q6_K",
            GgmlType::Iq3S => "IQ3_S",
            GgmlType::Iq4Nl => "IQ4_NL",
            GgmlType::Iq4Xs => "IQ4_XS",
        }
    }

    /// The values in one block. F32 and BF16 have blocks of one value.
    pub(crate) fn block_values(self) -> u64 {
        match self {
            GgmlType::F32 | GgmlType::Bf16 => 1,
            GgmlType::Q8_0 | GgmlType::Iq4Nl => 32,
            GgmlType::Q6K | GgmlType::Iq3S | GgmlType::Iq4Xs => QK_K as u64,
        }
    }

    /// The bytes of one block, as ggml's block structs lay them out with no padding.
    pub(crate) fn block_bytes(self) -> u64 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::Bf16 => 2,
            GgmlType::Q8_0 => 34,
            GgmlType::Q6K => 210,
            GgmlType::Iq3S => 110,
            GgmlType::Iq4Nl => 18,
            GgmlType::Iq4Xs => 136,
        }
    }
}

/// The values in one super-block of the K and IQ types (ggml `QK_K`).
const QK_K: usize = 256;

/// The sign masks of the eight values that one signs byte covers (ggml `kmask_iq2xs`).
const MASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// The non-linear 4-bit code book of IQ4_NL and IQ4_XS (ggml `kvalues_iq4nl`).
const VALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// The grid of IQ3_S, indexed by a 9-bit code (ggml `iq3s_grid`). Each entry holds four magnitudes, one in each byte.
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305, 0x01010309,
    0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905, 0x0101090b, 0x0101090f,
    0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09, 0x01010f0f, 0x01030101, 0x01030103,
    0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b, 0x01030501, 0x01030507, 0x0103050f, 0x01030703,
    0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f,
    0x01050301, 0x01050307, 0x0105030d, 0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b,
    0x0105090f, 0x01050b03, 0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501,
    0x01070505, 0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901, 0x01090907,
    0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 0x010b0707, 0x010b0903,
    0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303, 0x010d0307, 0x010d0703, 0x010d0b05,
    0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501, 0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01,
    0x010f0b09, 0x03010101, 0x03010103, 0x03010105, 0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b,
    0x0301030f, 0x03010501, 0x03010505, 0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03,
    0x03010f05, 0x03030101, 0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701,
    0x03030707, 0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01, 0x03050f05,
    0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f, 0x03070701, 0x03070709,
    0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305, 0x03090309, 0x03090703, 0x03090707,
    0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103, 0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701,
    0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509, 0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107,
    0x030f0301, 0x030f0305, 0x030f0503, 0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103,
    0x05010107, 0x0501010b, 0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507,
    0x0501050f, 0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f, 0x05030505,
    0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109, 0x0505010f, 0x05050503,
    0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f, 0x05050f03, 0x05050f09, 0x05070101,
    0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509, 0x05070703, 0x05070707, 0x05070905, 0x05070b01,
    0x05070d0d, 0x05090103, 0x0509010f, 0x05090501, 0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05,
    0x05090f0b, 0x050b0109, 0x050b0303, 0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101,
    0x050d0105, 0x050d010f, 0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701,
    0x050f0907, 0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03, 0x07030103,
    0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01, 0x07030f05, 0x07030f0d,
    0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01, 0x07070103, 0x07070301, 0x07070309,
    0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903, 0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07,
    0x07090107, 0x07090303, 0x0709030d, 0x07090505, 0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103,
    0x070b0301, 0x070b0305, 0x070b050b, 0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903,
    0x070f0103, 0x070f0107, 0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501,
    0x09010509, 0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103, 0x09050107,
    0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 0x09050f01, 0x09070109,
    0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b, 0x09090101, 0x09090105, 0x09090509,
    0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f, 0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709,
    0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701, 0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309,
    0x0b010505, 0x0b010901, 0x0b010909, 0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107,
    0x0b03010b, 0x0b030305, 0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701,
    0x0b05070d, 0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d, 0x0b0b0b03,
    0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307, 0x0d01030b, 0x0d010703,
    0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09, 0x0d050305, 0x0d050709, 0x0d050905,
    0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309, 0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907,
    0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307,
    0x0f010101, 0x0f010109, 0x0f01010f, 0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05,
    0x0f030105, 0x0f030303, 0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d,
    0x0f050503, 0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// ggml's `ggml_compute_fp16_to_fp32`, verbatim: an f16 bit pattern to f32, with no use of a hardware conversion.
pub(crate) fn fp16_to_fp32(h: u16) -> f32 {
    let w = u32::from(h) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);

    let exp_offset: u32 = 0xE0 << 23;
    // 0x1.0p-112f.
    let exp_scale = f32::from_bits(15 << 23);
    let normalized_value = f32::from_bits((two_w >> 4) + exp_offset) * exp_scale;

    let magic_mask: u32 = 126 << 23;
    let magic_bias = 0.5f32;
    let denormalized_value = f32::from_bits((two_w >> 17) | magic_mask) - magic_bias;

    let denormalized_cutoff: u32 = 1 << 27;
    let result_bits = if two_w < denormalized_cutoff {
        denormalized_value.to_bits()
    } else {
        normalized_value.to_bits()
    };
    f32::from_bits(sign | result_bits)
}

/// ggml's `ggml_compute_bf16_to_fp32`: a shift.
pub(crate) fn bf16_to_fp32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// The little-endian u16 at `at`.
fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// Dequantize `src`, whole blocks of `kind`, into `dst`, one f32 for each value.
///
/// `dst` must hold the values of the blocks in `src` exactly. Any other length is a defect of the caller, so the call
/// panics. It gives no partial result.
pub(crate) fn dequant_f32(kind: GgmlType, src: &[u8], dst: &mut [f32]) {
    let values = kind.block_values() as usize;
    let bytes = kind.block_bytes() as usize;
    assert!(
        dst.len().is_multiple_of(values),
        "dequant {}: {} values are not whole blocks of {values}",
        kind.name(),
        dst.len()
    );
    assert_eq!(
        src.len(),
        dst.len() / values * bytes,
        "dequant {}: {} bytes do not hold {} values",
        kind.name(),
        src.len(),
        dst.len()
    );
    let blocks = src.chunks_exact(bytes).zip(dst.chunks_exact_mut(values));
    match kind {
        GgmlType::F32 => blocks.for_each(|(x, y)| y[0] = f32::from_le_bytes([x[0], x[1], x[2], x[3]])),
        GgmlType::Bf16 => blocks.for_each(|(x, y)| y[0] = bf16_to_fp32(u16_at(x, 0))),
        GgmlType::Q8_0 => blocks.for_each(|(x, y)| q8_0(x, y)),
        GgmlType::Q6K => blocks.for_each(|(x, y)| q6_k(x, y)),
        GgmlType::Iq3S => blocks.for_each(|(x, y)| iq3_s(x, y)),
        GgmlType::Iq4Nl => blocks.for_each(|(x, y)| iq4_nl(x, y)),
        GgmlType::Iq4Xs => blocks.for_each(|(x, y)| iq4_xs(x, y)),
    }
}

/// One block of ggml's `dequantize_row_q8_0`. The block is `d: f16, qs: [i8; 32]`.
fn q8_0(x: &[u8], y: &mut [f32]) {
    let d = fp16_to_fp32(u16_at(x, 0));
    for j in 0..32 {
        y[j] = f32::from(x[2 + j] as i8) * d;
    }
}

/// One block of ggml's `dequantize_row_q6_K`. The block is `ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16`.
fn q6_k(x: &[u8], y: &mut [f32]) {
    let d = fp16_to_fp32(u16_at(x, 208));
    for half in 0..2 {
        let ql = &x[64 * half..];
        let qh = &x[128 + 32 * half..];
        let sc = &x[192 + 8 * half..];
        let y = &mut y[128 * half..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = (i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32) as f32;
            let q2 = (i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32) as f32;
            let q3 = (i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32) as f32;
            let q4 = (i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32) as f32;
            y[l] = d * f32::from(sc[is] as i8) * q1;
            y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2;
            y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3;
            y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4;
        }
    }
}

/// The four magnitudes of IQ3_S grid entry `index`, in native byte order as ggml reads them on x86_64.
fn iq3s_grid(index: u32) -> [u8; 4] {
    IQ3S_GRID[index as usize].to_le_bytes()
}

/// One block of ggml's `dequantize_row_iq3_s`. The block is `d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32],
/// scales: [u8; 4]`.
fn iq3_s(x: &[u8], y: &mut [f32]) {
    let d = fp16_to_fp32(u16_at(x, 0));
    let scales = &x[106..110];
    let mut qs = 2;
    let mut qh = 66;
    let mut signs = 74;
    let mut out = 0;
    for ib32 in (0..QK_K / 32).step_by(2) {
        let db1 = d * (1 + 2 * i32::from(scales[ib32 / 2] & 0xf)) as f32;
        let db2 = d * (1 + 2 * i32::from(scales[ib32 / 2] >> 4)) as f32;
        for (db, high) in [(db1, x[qh]), (db2, x[qh + 1])] {
            for l in 0..4 {
                let high = u32::from(high);
                let grid1 = iq3s_grid(u32::from(x[qs + 2 * l]) | ((high << (8 - 2 * l)) & 256));
                let grid2 = iq3s_grid(u32::from(x[qs + 2 * l + 1]) | ((high << (7 - 2 * l)) & 256));
                let sign = x[signs + l];
                for j in 0..4 {
                    let s1 = if sign & MASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0 };
                    let s2 = if sign & MASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0 };
                    y[out + j] = db * f32::from(grid1[j]) * s1;
                    y[out + j + 4] = db * f32::from(grid2[j]) * s2;
                }
                out += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

/// One block of ggml's `dequantize_row_iq4_nl`. The block is `d: f16, qs: [u8; 16]`.
fn iq4_nl(x: &[u8], y: &mut [f32]) {
    let d = fp16_to_fp32(u16_at(x, 0));
    for j in 0..16 {
        let q = x[2 + j];
        y[j] = d * f32::from(VALUES_IQ4NL[usize::from(q & 0xf)]);
        y[j + 16] = d * f32::from(VALUES_IQ4NL[usize::from(q >> 4)]);
    }
}

/// One block of ggml's `dequantize_row_iq4_xs`. The block is `d: f16, scales_h: u16, scales_l: [u8; 4],
/// qs: [u8; 128]`.
fn iq4_xs(x: &[u8], y: &mut [f32]) {
    let d = fp16_to_fp32(u16_at(x, 0));
    let scales_h = u32::from(u16_at(x, 2));
    let scales_l = &x[4..8];
    for ib in 0..QK_K / 32 {
        let ls = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4) as i32;
        let dl = d * (ls - 32) as f32;
        let qs = &x[8 + 16 * ib..8 + 16 * ib + 16];
        let y = &mut y[32 * ib..32 * ib + 32];
        for j in 0..16 {
            y[j] = dl * f32::from(VALUES_IQ4NL[usize::from(qs[j] & 0xf)]);
            y[j + 16] = dl * f32::from(VALUES_IQ4NL[usize::from(qs[j] >> 4)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{GgmlType, bf16_to_fp32, dequant_f32, fp16_to_fp32};

    /// The digests that the reference's C++ copy wrote (`tools/mk_dequant_digests.cpp blocks`, commit `7d474c2`).
    const BLOCKS: &str = include_str!("../tests/fixtures/dequant/blocks.txt");

    /// The blocks of each type in the fixture.
    const FIXTURE_BLOCKS: usize = 64;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The tool's input: `count` bytes of splitmix64 output from `seed`, eight little-endian bytes for each step.
    fn random_bytes(seed: u64, count: usize) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(count + 8);
        while out.len() < count {
            out.extend_from_slice(&splitmix64(&mut state).to_le_bytes());
        }
        out.truncate(count);
        out
    }

    fn digest(values: impl IntoIterator<Item = f32>) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// The fixture's digest for `name`.
    fn expected(name: &str) -> String {
        let line = BLOCKS
            .lines()
            .find(|line| line.split(' ').next() == Some(name))
            .unwrap_or_else(|| panic!("the fixture has no line for {name}"));
        line.split(' ').nth(2).expect("a digest").to_owned()
    }

    fn type_digest(kind: GgmlType) -> String {
        let src = random_bytes(u64::from(kind.id()), FIXTURE_BLOCKS * kind.block_bytes() as usize);
        let mut dst = vec![0f32; FIXTURE_BLOCKS * kind.block_values() as usize];
        dequant_f32(kind, &src, &mut dst);
        digest(dst)
    }

    #[test]
    fn each_type_gives_the_bits_of_the_cpp_copy() {
        let wrong: Vec<&str> = GgmlType::ALL
            .into_iter()
            .filter(|kind| type_digest(*kind) != expected(kind.name()))
            .map(GgmlType::name)
            .collect();
        assert!(
            wrong.is_empty(),
            "these types differ from the reference's C++ dequant: {wrong:?}"
        );
    }

    #[test]
    fn every_f16_and_bf16_value_converts_as_the_cpp_copy() {
        assert_eq!(digest((0..=u16::MAX).map(fp16_to_fp32)), expected("f16-all"));
        assert_eq!(digest((0..=u16::MAX).map(bf16_to_fp32)), expected("bf16-all"));
    }

    #[test]
    fn the_fixture_covers_every_type() {
        for kind in GgmlType::ALL {
            assert_eq!(expected(kind.name()).len(), 64, "{}", kind.name());
            assert_eq!(GgmlType::from_id(kind.id()), Some(kind));
        }
        assert_eq!(GgmlType::from_id(1), None, "F16 is not a type of these shards");
    }

    #[test]
    #[should_panic(expected = "do not hold")]
    fn a_short_source_is_refused() {
        let mut dst = [0f32; 32];
        dequant_f32(GgmlType::Q8_0, &[0u8; 33], &mut dst);
    }

    #[test]
    #[should_panic(expected = "not whole blocks")]
    fn a_part_of_a_block_is_refused() {
        let mut dst = [0f32; 31];
        dequant_f32(GgmlType::Q8_0, &[0u8; 34], &mut dst);
    }
}
