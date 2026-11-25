//! Host quantization oracle for the NVFP4 (`fouroversix_mae`) and MXFP8 (`ue8m0`) activations of Inkling 2.
//!
//! The rules are those of the Triton kernels that vLLM ships for Inkling 2 (`nvfp4.py` and
//! `mxfp8.py`). Transformers quantizes no activation. Each quantizer gives the bytes of those kernels bit for bit:
//! every `f32` operation here is the same operation in the same order. The test `tests/quant_fixture.rs` compares
//! the bytes. A non-finite input or a NaN code is an error (spec 8.5).

use std::fmt;

/// Exact E2M1 values for codes 0 to 15. Codes 8 to 15 are negative copies of 0 to 7.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Block size of NVFP4 (`e2m1` values sharing one `e4m3fn` scale).
pub const NVFP4_BLOCK: usize = 16;

/// Block size of MXFP8 (`e4m3fn` values sharing one `ue8m0` scale).
pub const MXFP8_BLOCK: usize = 32;

/// Maximum finite magnitude representable in FP8 E4M3FN (`0x7E`).
pub const E4M3_MAX: f32 = 448.0;

/// The divisor of the row amax in NVFP4. The outer scale of a row is `amax / 1536`, and the encoder scale is its
/// inverse. The value is a rule of vLLM's kernel, not `6 * 448`.
pub const NVFP4_AMAX_DIVISOR: f32 = 1536.0;

/// The largest finite `f32`.
const FP32_MAX: f32 = f32::MAX;

/// Quantization and format errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatError {
    /// An E4M3FN byte is NaN (`0x7F` or `0xFF`).
    E4m3Nan(u8),
    /// A UE8M0 byte is 255 (NaN in OCP MX).
    Ue8m0Nan,
    /// An `f32` input is NaN or infinite.
    NonFinite,
    /// A scale factor of a weight is zero, negative, NaN or infinite, so the row scale is not finite.
    Scale(&'static str),
    /// A slice length does not match the block size or companion slice.
    Length {
        what: &'static str,
        got: usize,
        want: usize,
    },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::E4m3Nan(c) => write!(
                f,
                "E4M3FN code 0x{c:02x} is NaN; the checkpoint or activation buffer is corrupt"
            ),
            Self::Ue8m0Nan => write!(
                f,
                "UE8M0 scale 255 is NaN; the checkpoint or activation buffer is corrupt"
            ),
            Self::NonFinite => write!(f, "non-finite f32 value encountered during quantization"),
            Self::Scale(what) => write!(
                f,
                "{what} must be a finite number above zero; the weight scale of the checkpoint is corrupt"
            ),
            Self::Length { what, got, want } => {
                write!(f, "length of {what} is {got}, expected {want}")
            }
        }
    }
}

impl std::error::Error for FormatError {}

fn pow2(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// Decode one FP8 E4M3FN byte (4 exponent bits, bias 7, 3 mantissa bits, no infinities).
pub fn decode_e4m3(code: u8) -> Result<f32, FormatError> {
    if code & 0x7F == 0x7F {
        return Err(FormatError::E4m3Nan(code));
    }
    let exp = i32::from((code >> 3) & 0xF);
    let man = f32::from(code & 7);
    let mag = if exp == 0 {
        man * pow2(-9)
    } else {
        (8.0 + man) * pow2(exp - 10)
    };
    Ok(if code & 0x80 == 0 { mag } else { -mag })
}

/// Encode a finite `f32` to FP8 E4M3FN using round-to-nearest-even, clamping to `[-448, 448]`.
pub fn encode_e4m3(value: f32) -> Result<u8, FormatError> {
    if !value.is_finite() {
        return Err(FormatError::NonFinite);
    }
    let sign: u8 = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let mag = value.abs();
    if mag == 0.0 {
        return Ok(sign);
    }
    if mag >= E4M3_MAX {
        return Ok(sign | 0x7E);
    }
    // Search the 126 positive finite E4M3FN codes (0x00..=0x7E) using exact distance and ties-to-even.
    // This is exact for all normals and subnormals and serves as the reference oracle.
    let mut best_code: u8 = 0;
    let mut best_err = mag;
    for code in 1u8..=0x7E {
        let repr = decode_e4m3(code).expect("0x01..=0x7E are finite");
        let err = (mag - repr).abs();
        if err < best_err || (err == best_err && (code & 1) == 0 && (best_code & 1) == 1) {
            best_err = err;
            best_code = code;
        }
    }
    Ok(sign | best_code)
}

/// Decode one UE8M0 scale byte: `2^(code - 127)`. Code 255 is NaN.
pub fn decode_ue8m0(code: u8) -> Result<f32, FormatError> {
    match code {
        255 => Err(FormatError::Ue8m0Nan),
        0 => Ok(f32::from_bits(1 << 22)),
        c => Ok(f32::from_bits(u32::from(c) << 23)),
    }
}

/// The UE8M0 scale byte of a block with max magnitude `amax`: the exponent that rounds `amax * (1 / 448)` up to a
/// power of two. A raw scale of `2^-127` or less gives 0, and the byte stops at 254. The bit trick is the rule of
/// vLLM's `_mxfp8_block`: it adds the mantissa mask, so any nonzero mantissa carries into the exponent.
pub fn encode_ue8m0_ceil(amax: f32) -> Result<u8, FormatError> {
    if !amax.is_finite() || amax < 0.0 {
        return Err(FormatError::NonFinite);
    }
    let raw = amax * reciprocal_constant(448.0);
    if raw <= f32::from_bits(1 << 22) {
        return Ok(0);
    }
    let exponent = ((raw.to_bits() + 0x007F_FFFF) & 0x7F80_0000) >> 23;
    Ok(exponent.min(254) as u8)
}

/// A reciprocal constant as the kernel source has it: the `f64` quotient of Python, rounded once to `f32`.
fn reciprocal_constant(divisor: f64) -> f32 {
    (1.0f64 / divisor) as f32
}

/// Encode a `f32` to FP4 E2M1 (4-bit code `0..=15`) with round-to-nearest-even and saturation.
/// This is `cvt.rn.satfinite.e2m1x2.f32` of `sm_121`. A value past 6.0, and an infinity, give the code of 6.0.
/// Ties break to the even code: 0.25->0, 0.75->2, 1.25->2, 1.75->4, 2.5->4, 3.5->6, 5.0->6. NaN is an error.
pub fn encode_e2m1(value: f32) -> Result<u8, FormatError> {
    if value.is_nan() {
        return Err(FormatError::NonFinite);
    }
    let sign: u8 = if value.is_sign_negative() { 8 } else { 0 };
    let mag = value.abs();
    let code: u8 = if mag <= 0.25 {
        0
    } else if mag < 0.75 {
        1
    } else if mag <= 1.25 {
        2
    } else if mag < 1.75 {
        3
    } else if mag <= 2.5 {
        4
    } else if mag < 3.5 {
        5
    } else if mag <= 5.0 {
        6
    } else {
        7
    };
    Ok(sign | code)
}

/// One activation row quantized to NVFP4 (`fouroversix_mae`).
#[derive(Clone, Debug, PartialEq)]
pub struct Nvfp4Row {
    /// Packed E2M1 nibbles (`len / 2` bytes; even element in low nibble, odd in high nibble).
    pub codes: Vec<u8>,
    /// One E4M3FN scale per 16 elements (`len / 16` bytes), row-major.
    pub scales: Vec<u8>,
    /// The max magnitude of the row. The outer scale of the row is `amax * (1 / 1536)`. It is 0 for a zero row.
    pub amax: f32,
}

impl Nvfp4Row {
    /// The outer scale of the row: `amax * (1 / 1536)`, the factor that the block scales leave out.
    pub fn outer_scale(&self) -> f32 {
        self.amax * reciprocal_constant(f64::from(NVFP4_AMAX_DIVISOR))
    }
}

/// The sum of 16 errors as the pairwise tree of `_block_error`: adjacent pairs, then pairs of pairs, and so on.
fn tree_sum16(mut e: [f32; NVFP4_BLOCK]) -> f32 {
    let mut width = NVFP4_BLOCK;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            e[i] = e[2 * i] + e[2 * i + 1];
        }
    }
    e[0]
}

/// One candidate of a block: its e2m1 codes and its reconstruction error against the input.
/// The reconstruction is `((value * scale) * outer)`, as in `_pack_roundtrip` and `_quantize_block`.
fn nvfp4_candidate(
    block: &[f32],
    scale: f32,
    decoder: f32,
    outer: f32,
) -> Result<([u8; NVFP4_BLOCK], f32), FormatError> {
    let inverse = (1.0f32 / (scale * decoder)).min(FP32_MAX);
    let mut codes = [0u8; NVFP4_BLOCK];
    let mut errors = [0.0f32; NVFP4_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let code = encode_e2m1(x * inverse)?;
        codes[i] = code;
        errors[i] = ((E2M1[usize::from(code)] * scale) * outer - x).abs();
    }
    Ok((codes, tree_sum16(errors)))
}

/// Quantize one activation row to NVFP4 with the per-token `fouroversix_mae` rule of vLLM.
///
/// The row amax gives the encoder scale `min(1536 / amax, f32::MAX)`, its inverse and the outer scale. Each block
/// tries a 6-based and a 4-based e4m3 scale, and it keeps the candidate with the smaller error. A tie keeps the
/// 6-based candidate (`err4 < err6`).
pub fn quantize_nvfp4_fouroversix_mae(input: &[f32]) -> Result<Nvfp4Row, FormatError> {
    if !input.len().is_multiple_of(NVFP4_BLOCK) {
        return Err(FormatError::Length {
            what: "NVFP4 activation row",
            got: input.len(),
            want: input.len().next_multiple_of(NVFP4_BLOCK),
        });
    }
    let mut amax = 0.0f32;
    for &x in input {
        if !x.is_finite() {
            return Err(FormatError::NonFinite);
        }
        amax = amax.max(x.abs());
    }
    let encoder = (NVFP4_AMAX_DIVISOR / amax).min(FP32_MAX);
    let decoder = 1.0f32 / encoder;
    let outer = amax * reciprocal_constant(f64::from(NVFP4_AMAX_DIVISOR));
    let sixth = reciprocal_constant(6.0);
    let mut codes = vec![0u8; input.len() / 2];
    let mut scales = vec![0u8; input.len() / NVFP4_BLOCK];
    for (b, block) in input.as_chunks::<NVFP4_BLOCK>().0.iter().enumerate() {
        let block_max = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let byte6 = encode_e4m3((block_max * (encoder * sixth)).min(E4M3_MAX))?;
        let byte4 = encode_e4m3((block_max * (encoder * 0.25)).min(E4M3_MAX))?;
        let (codes6, error6) = nvfp4_candidate(block, decode_e4m3(byte6)?, decoder, outer)?;
        let (codes4, error4) = nvfp4_candidate(block, decode_e4m3(byte4)?, decoder, outer)?;
        let (byte, chosen) = if error4 < error6 {
            (byte4, codes4)
        } else {
            (byte6, codes6)
        };
        scales[b] = byte;
        for i in 0..NVFP4_BLOCK / 2 {
            codes[b * (NVFP4_BLOCK / 2) + i] = chosen[2 * i] | (chosen[2 * i + 1] << 4);
        }
    }
    Ok(Nvfp4Row { codes, scales, amax })
}

/// One activation row quantized to MXFP8 (`ue8m0` scales, block size 32).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mxfp8Row {
    /// E4M3FN element codes (`len` bytes).
    pub codes: Vec<u8>,
    /// One UE8M0 scale byte per 32 elements (`len / 32` bytes), row-major.
    pub scales: Vec<u8>,
}

/// Quantize one activation row to MXFP8 with the round-up UE8M0 block scales of vLLM.
///
/// A value is `x * inverse`, with `inverse` the exact reciprocal of the scale (2^127 for the byte 0), clamped to
/// `[-448, 448]` and rounded to e4m3. A zero block takes the byte 0 and gives zero codes.
pub fn quantize_mxfp8_ue8m0(input: &[f32]) -> Result<Mxfp8Row, FormatError> {
    if !input.len().is_multiple_of(MXFP8_BLOCK) {
        return Err(FormatError::Length {
            what: "MXFP8 activation row",
            got: input.len(),
            want: input.len().next_multiple_of(MXFP8_BLOCK),
        });
    }
    if input.iter().any(|x| !x.is_finite()) {
        return Err(FormatError::NonFinite);
    }
    let mut codes = vec![0u8; input.len()];
    let mut scales = vec![0u8; input.len() / MXFP8_BLOCK];
    for (b, block) in input.as_chunks::<MXFP8_BLOCK>().0.iter().enumerate() {
        let block_max = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let byte = encode_ue8m0_ceil(block_max)?;
        let inverse = if byte == 0 {
            pow2(127)
        } else {
            1.0f32 / decode_ue8m0(byte)?
        };
        scales[b] = byte;
        for (i, &x) in block.iter().enumerate() {
            codes[b * MXFP8_BLOCK + i] = encode_e4m3((x * inverse).clamp(-E4M3_MAX, E4M3_MAX))?;
        }
    }
    Ok(Mxfp8Row { codes, scales })
}

fn check_len(what: &'static str, got: usize, want: usize) -> Result<(), FormatError> {
    if got == want {
        Ok(())
    } else {
        Err(FormatError::Length { what, got, want })
    }
}

fn nibble(codes: &[u8], index: usize) -> u8 {
    let byte = codes[index / 2];
    if index.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

/// The NVFP4 dot product of an activation row and one weight row, as the epilogue of vLLM's expert GEMM gives it.
///
/// The sum is block by block in ascending k, in `f32`: the products of the 16 codes of a block, in ascending order,
/// then `* (act_scale * weight_scale)`, then added to the total. Then the epilogue follows: `* expert_scale`
/// (1.0 on sm_120 and sm_121), then `* row_scale` with `row_scale = (act.amax * w_amax) * (1 / (1536 *
/// w_denominator))`. The checkpoint gives `w_amax` and `w_denominator` (`weight_scale_format: amax_denominator`).
/// The order of the sums inside one hardware MMA is not defined by vLLM. A card gate compares with a tolerance.
pub fn dot_nvfp4_block_scaled(
    act: &Nvfp4Row,
    w_codes: &[u8],
    w_scales: &[u8],
    w_amax: f32,
    w_denominator: f32,
    expert_scale: f32,
) -> Result<f32, FormatError> {
    check_len("NVFP4 weight codes", w_codes.len(), act.codes.len())?;
    check_len("NVFP4 weight scales", w_scales.len(), act.scales.len())?;
    if !w_amax.is_finite() || w_amax < 0.0 {
        return Err(FormatError::Scale("the weight amax"));
    }
    if !w_denominator.is_finite() || w_denominator <= 0.0 || !(NVFP4_AMAX_DIVISOR * w_denominator).is_finite() {
        return Err(FormatError::Scale("the weight denominator"));
    }
    if !expert_scale.is_finite() {
        return Err(FormatError::Scale("the expert scale"));
    }
    let mut total = 0.0f32;
    for b in 0..act.scales.len() {
        let scale = decode_e4m3(act.scales[b])? * decode_e4m3(w_scales[b])?;
        let mut block_sum = 0.0f32;
        for i in b * NVFP4_BLOCK..(b + 1) * NVFP4_BLOCK {
            block_sum += E2M1[usize::from(nibble(&act.codes, i))] * E2M1[usize::from(nibble(w_codes, i))];
        }
        total += block_sum * scale;
    }
    let row_scale = (act.amax * w_amax) * (1.0f32 / (NVFP4_AMAX_DIVISOR * w_denominator));
    Ok((total * expert_scale) * row_scale)
}

/// The MXFP8 dot product of an activation row and one weight row, as the sm_120 GEMM of vLLM gives it.
///
/// vLLM dequantizes each code as `value * scale` and sums the products in `f32`. The host does the same, in
/// ascending k, then `* expert_scale` (1.0). It uses no row scale. vLLM rounds the result to `bf16`, and the host
/// does not. A card gate compares with a tolerance.
pub fn dot_mxfp8_block_scaled(
    act: &Mxfp8Row,
    w_codes: &[u8],
    w_scales: &[u8],
    expert_scale: f32,
) -> Result<f32, FormatError> {
    check_len("MXFP8 weight codes", w_codes.len(), act.codes.len())?;
    check_len("MXFP8 weight scales", w_scales.len(), act.scales.len())?;
    if !expert_scale.is_finite() {
        return Err(FormatError::Scale("the expert scale"));
    }
    let mut total = 0.0f32;
    for b in 0..act.scales.len() {
        let sa = decode_ue8m0(act.scales[b])?;
        let sw = decode_ue8m0(w_scales[b])?;
        for i in b * MXFP8_BLOCK..(b + 1) * MXFP8_BLOCK {
            total += (decode_e4m3(act.codes[i])? * sa) * (decode_e4m3(w_codes[i])? * sw);
        }
    }
    Ok(total * expert_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_ties_round_to_even() {
        assert_eq!(encode_e2m1(0.25).unwrap(), 0); // 0.0
        assert_eq!(encode_e2m1(0.75).unwrap(), 2); // 1.0
        assert_eq!(encode_e2m1(1.25).unwrap(), 2); // 1.0
        assert_eq!(encode_e2m1(1.75).unwrap(), 4); // 2.0
        assert_eq!(encode_e2m1(2.5).unwrap(), 4); // 2.0
        assert_eq!(encode_e2m1(3.5).unwrap(), 6); // 4.0
        assert_eq!(encode_e2m1(5.0).unwrap(), 6); // 4.0
        assert_eq!(encode_e2m1(-5.0).unwrap(), 8 | 6); // -4.0
        assert_eq!(encode_e2m1(100.0).unwrap(), 7); // saturates to 6.0
        assert_eq!(encode_e2m1(f32::INFINITY).unwrap(), 7);
        assert_eq!(encode_e2m1(-0.1).unwrap(), 8); // negative zero keeps its sign
        assert!(encode_e2m1(f32::NAN).is_err());
    }

    #[test]
    fn e4m3_round_trip_subnormals_max_and_nan() {
        assert_eq!(decode_e4m3(0x00).unwrap(), 0.0);
        assert_eq!(decode_e4m3(0x7E).unwrap(), 448.0);
        assert_eq!(decode_e4m3(0xFE).unwrap(), -448.0);
        assert_eq!(decode_e4m3(0x7F), Err(FormatError::E4m3Nan(0x7F)));
        assert_eq!(decode_e4m3(0xFF), Err(FormatError::E4m3Nan(0xFF)));
        assert_eq!(encode_e4m3(448.0).unwrap(), 0x7E);
        assert_eq!(encode_e4m3(1000.0).unwrap(), 0x7E);
        assert_eq!(encode_e4m3(-1000.0).unwrap(), 0xFE);
        assert!(encode_e4m3(f32::INFINITY).is_err());
        for code in 0u8..=0x7E {
            let v = decode_e4m3(code).unwrap();
            assert_eq!(encode_e4m3(v).unwrap(), code, "round trip failed for 0x{code:02x}");
        }
    }

    #[test]
    fn ue8m0_rounds_up_and_rejects_nan() {
        assert_eq!(decode_ue8m0(127).unwrap(), 1.0);
        assert_eq!(decode_ue8m0(255), Err(FormatError::Ue8m0Nan));
        assert_eq!(encode_ue8m0_ceil(0.0).unwrap(), 0);
        // 448 / 448 = 1 = 2^0 is exact, and the next f32 above it rounds up to 2^1.
        assert_eq!(encode_ue8m0_ceil(448.0).unwrap(), 127);
        assert_eq!(encode_ue8m0_ceil(f32::from_bits(448.0f32.to_bits() + 1)).unwrap(), 128);
        assert_eq!(encode_ue8m0_ceil(f32::MAX).unwrap(), 247); // 2^120 is the largest raw scale
        for amax in [1.0e-3f32, 0.5, 449.0, 1000.0, 1.0e5] {
            let s = encode_ue8m0_ceil(amax).unwrap();
            assert!(decode_ue8m0(s).unwrap() * E4M3_MAX >= amax);
        }
        assert!(encode_ue8m0_ceil(f32::NAN).is_err());
    }

    #[test]
    fn nvfp4_row_keeps_amax_and_the_outer_scale_is_amax_over_1536() {
        let mut row = [0.5f32; 32];
        row[20] = -3.0;
        let q = quantize_nvfp4_fouroversix_mae(&row).unwrap();
        assert_eq!(q.amax, 3.0);
        assert_eq!(q.outer_scale(), 3.0 * (1.0f64 / 1536.0) as f32);
        assert_eq!(q.codes.len(), 16);
        assert_eq!(q.scales.len(), 2);
    }

    #[test]
    fn nvfp4_zero_row_gives_zero_bytes_and_zero_amax() {
        let q = quantize_nvfp4_fouroversix_mae(&[0.0f32; 32]).unwrap();
        assert_eq!(q.amax, 0.0);
        assert!(q.codes.iter().all(|&b| b == 0));
        assert!(q.scales.iter().all(|&b| b == 0));
    }

    #[test]
    fn quantizers_reject_non_finite_input_and_bad_lengths() {
        let mut row = [1.0f32; 32];
        row[3] = f32::NAN;
        assert_eq!(quantize_nvfp4_fouroversix_mae(&row), Err(FormatError::NonFinite));
        row[3] = f32::INFINITY;
        assert_eq!(quantize_mxfp8_ue8m0(&row), Err(FormatError::NonFinite));
        assert!(matches!(
            quantize_nvfp4_fouroversix_mae(&row[..17]),
            Err(FormatError::Length { .. })
        ));
        assert!(matches!(
            quantize_mxfp8_ue8m0(&row[..16]),
            Err(FormatError::Length { .. })
        ));
    }

    #[test]
    fn error_sum_is_the_pairwise_tree_not_left_to_right() {
        // The tree adds (e2 + e3) before it adds 1.0. Each small term is under half an ulp of 1.0, but their sum
        // is over it. A left-to-right sum loses both terms.
        let mut e = [0.0f32; 16];
        e[0] = 1.0;
        e[2] = 2.0f32.powi(-24) * 0.75;
        e[3] = 2.0f32.powi(-24) * 0.75;
        let left_to_right = e.iter().fold(0.0f32, |a, &b| a + b);
        assert_ne!(tree_sum16(e), left_to_right);
    }

    #[test]
    fn nvfp4_dot_follows_the_epilogue_of_vllm() {
        let act_f: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.25).collect();
        let w_f: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let qa = quantize_nvfp4_fouroversix_mae(&act_f).unwrap();
        let qw = quantize_nvfp4_fouroversix_mae(&w_f).unwrap();
        // Use the row of the weight as an expert: its amax, and a denominator of 1536, so its outer scale is exact.
        let dot = dot_nvfp4_block_scaled(&qa, &qw.codes, &qw.scales, qw.amax, 1536.0, 1.0).unwrap();
        // The reference dequantizes both sides and sums in f64.
        let deq = |q: &Nvfp4Row, outer: f32| -> Vec<f64> {
            (0..32)
                .map(|i| {
                    f64::from(E2M1[usize::from(nibble(&q.codes, i))])
                        * f64::from(decode_e4m3(q.scales[i / 16]).unwrap())
                        * f64::from(outer)
                })
                .collect()
        };
        let (a, w) = (deq(&qa, qa.outer_scale()), deq(&qw, qw.amax / 1536.0));
        let reference: f64 = a.iter().zip(&w).map(|(x, y)| x * y).sum();
        assert!(
            (f64::from(dot) - reference).abs() <= 1e-4 * reference.abs().max(1.0),
            "{dot} {reference}"
        );
        let exact: f32 = act_f.iter().zip(&w_f).map(|(a, b)| a * b).sum();
        assert!((dot - exact).abs() < 1.5, "dot={dot}, exact={exact}");
        // The expert scale multiplies the sum, and a zero row scale gives zero.
        let doubled = dot_nvfp4_block_scaled(&qa, &qw.codes, &qw.scales, qw.amax, 1536.0, 2.0).unwrap();
        assert_eq!(doubled, dot * 2.0);
        let zero = quantize_nvfp4_fouroversix_mae(&[0.0f32; 32]).unwrap();
        assert_eq!(
            dot_nvfp4_block_scaled(&zero, &qw.codes, &qw.scales, qw.amax, 1536.0, 1.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn dot_products_reject_bad_scales_and_lengths() {
        let qa = quantize_nvfp4_fouroversix_mae(&[1.0f32; 16]).unwrap();
        let ok = |amax: f32, den: f32, e: f32| dot_nvfp4_block_scaled(&qa, &qa.codes, &qa.scales, amax, den, e);
        assert!(ok(1.0, 1.0, 1.0).is_ok());
        assert!(matches!(ok(1.0, 0.0, 1.0), Err(FormatError::Scale(_))));
        assert!(matches!(ok(1.0, f32::NAN, 1.0), Err(FormatError::Scale(_))));
        assert!(matches!(ok(f32::INFINITY, 1.0, 1.0), Err(FormatError::Scale(_))));
        assert!(matches!(ok(1.0, 1.0, f32::NAN), Err(FormatError::Scale(_))));
        assert!(matches!(
            dot_nvfp4_block_scaled(&qa, &qa.codes[1..], &qa.scales, 1.0, 1.0, 1.0),
            Err(FormatError::Length { .. })
        ));
    }

    #[test]
    fn mxfp8_dot_matches_dequantized_math() {
        let act_f: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.25).collect();
        let w_f: Vec<f32> = (0..64).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let ma = quantize_mxfp8_ue8m0(&act_f).unwrap();
        let mw = quantize_mxfp8_ue8m0(&w_f).unwrap();
        let dot = dot_mxfp8_block_scaled(&ma, &mw.codes, &mw.scales, 1.0).unwrap();
        let exact: f32 = act_f.iter().zip(&w_f).map(|(a, b)| a * b).sum();
        assert!((dot - exact).abs() < 0.5, "dot={dot}, exact={exact}");
        assert_eq!(
            dot_mxfp8_block_scaled(&ma, &mw.codes, &mw.scales, 0.5).unwrap(),
            dot * 0.5
        );
    }
}
