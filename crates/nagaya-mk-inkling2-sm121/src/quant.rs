//! Host quantization oracle for native NVFP4 (`fouroversix_mae`) and MXFP8 (`ue8m0`) activations and weights.
//!
//! Each op matches the GB10 (`sm_121a`) block-scaled MMA specification in `docs/engine-abi.md`
//! ("The GB10 Inkling 2 binding"). Non-finite inputs and NaN codes are errors (spec 8.5).

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

/// Maximum finite magnitude representable in FP4 E2M1 (`0x7`).
pub const E2M1_MAX: f32 = 6.0;

/// Alternative block scale target used by the Four Over Six (`fouroversix_mae`) selection rule.
pub const FOUR_OVER_SIX_ALT: f32 = 4.0;

/// Quantization and format errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatError {
    /// An E4M3FN byte is NaN (`0x7F` or `0xFF`).
    E4m3Nan(u8),
    /// A UE8M0 byte is 255 (NaN in OCP MX).
    Ue8m0Nan,
    /// An `f32` input is NaN or infinite.
    NonFinite,
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

/// Compute the smallest UE8M0 power-of-two scale `2^e >= amax / 448.0`, clamped to `0..=254`.
/// Returns `0` for `amax == 0.0`.
pub fn encode_ue8m0_ceil(amax: f32) -> Result<u8, FormatError> {
    if !amax.is_finite() || amax < 0.0 {
        return Err(FormatError::NonFinite);
    }
    if amax == 0.0 {
        return Ok(0);
    }
    let needed = amax / E4M3_MAX;
    let exp = needed.log2().ceil() as i32;
    let biased = (exp + 127).clamp(0, 254) as u8;
    // Verify `decode_ue8m0(biased) * E4M3_MAX >= amax` (handle floating log2 rounding edge).
    if biased < 254 && decode_ue8m0(biased)? * E4M3_MAX < amax {
        Ok(biased + 1)
    } else {
        Ok(biased)
    }
}

/// Encode a finite `f32` to FP4 E2M1 (4-bit code `0..=15`) using round-to-nearest-even.
/// Ties break to even code index: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5.0->4.
pub fn encode_e2m1(value: f32) -> Result<u8, FormatError> {
    if !value.is_finite() {
        return Err(FormatError::NonFinite);
    }
    let sign: u8 = if value.is_sign_negative() { 8 } else { 0 };
    let mag = value.abs();
    let code: u8 = if mag <= 0.25 {
        0 // 0.0 (even; tie at 0.25 goes to 0)
    } else if mag < 0.75 {
        1 // 0.5
    } else if mag <= 1.25 {
        2 // 1.0 (even; ties at 0.75 and 1.25 go to 2)
    } else if mag < 1.75 {
        3 // 1.5
    } else if mag <= 2.5 {
        4 // 2.0 (even; ties at 1.75 and 2.5 go to 4)
    } else if mag < 3.5 {
        5 // 3.0
    } else if mag <= 5.0 {
        6 // 4.0 (even; ties at 3.5 and 5.0 go to 6)
    } else {
        7 // 6.0 (clamped)
    };
    Ok(sign | code)
}

/// One activation row quantized to NVFP4 (`fouroversix_mae`).
#[derive(Clone, Debug, PartialEq)]
pub struct Nvfp4Row {
    /// Packed E2M1 nibbles (`len / 2` bytes; even element in low nibble, odd in high nibble).
    pub codes: Vec<u8>,
    /// One E4M3FN scale per 16 elements (`len / 16` bytes).
    pub scales: Vec<u8>,
    /// Per-token global scale (`row_amax / (6.0 * 448.0)`), or `0.0` for an all-zero row.
    pub global_scale: f32,
}

/// Quantize one activation row to NVFP4 using the per-token `fouroversix_mae` rule.
pub fn quantize_nvfp4_fouroversix_mae(input: &[f32]) -> Result<Nvfp4Row, FormatError> {
    if !input.len().is_multiple_of(NVFP4_BLOCK) {
        return Err(FormatError::Length {
            what: "NVFP4 activation row",
            got: input.len(),
            want: input.len().next_multiple_of(NVFP4_BLOCK),
        });
    }
    let mut row_amax = 0.0f32;
    for &x in input {
        if !x.is_finite() {
            return Err(FormatError::NonFinite);
        }
        row_amax = row_amax.max(x.abs());
    }
    let n_blocks = input.len() / NVFP4_BLOCK;
    if row_amax == 0.0 {
        return Ok(Nvfp4Row {
            codes: vec![0u8; input.len() / 2],
            scales: vec![0u8; n_blocks],
            global_scale: 0.0,
        });
    }
    let global_scale = row_amax / (E2M1_MAX * E4M3_MAX);
    let mut codes = vec![0u8; input.len() / 2];
    let mut scales = vec![0u8; n_blocks];

    for (b, block) in input.as_chunks::<NVFP4_BLOCK>().0.iter().enumerate() {
        let block_amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if block_amax == 0.0 {
            scales[b] = 0;
            continue;
        }
        let eval = |target_max: f32| -> Result<(u8, [u8; NVFP4_BLOCK], f32), FormatError> {
            let raw_scale = block_amax / (target_max * global_scale);
            let scale_byte = encode_e4m3(raw_scale)?;
            let eff_scale = decode_e4m3(scale_byte)? * global_scale;
            let mut q = [0u8; NVFP4_BLOCK];
            let mut mae = 0.0f32;
            for (i, &x) in block.iter().enumerate() {
                let code = if eff_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(x / eff_scale)?
                };
                q[i] = code;
                let deq = E2M1[usize::from(code)] * eff_scale;
                mae += (x - deq).abs();
            }
            Ok((scale_byte, q, mae))
        };

        let (s6, q6, mae6) = eval(E2M1_MAX)?;
        let (s4, q4, mae4) = eval(FOUR_OVER_SIX_ALT)?;
        let (scale_byte, q) = if mae4 < mae6 { (s4, q4) } else { (s6, q6) };
        scales[b] = scale_byte;
        for i in 0..(NVFP4_BLOCK / 2) {
            codes[b * (NVFP4_BLOCK / 2) + i] = (q[2 * i] & 0x0F) | ((q[2 * i + 1] & 0x0F) << 4);
        }
    }
    Ok(Nvfp4Row {
        codes,
        scales,
        global_scale,
    })
}

/// One activation row quantized to MXFP8 (`ue8m0` scales, block size 32).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mxfp8Row {
    /// E4M3FN element codes (`len` bytes).
    pub codes: Vec<u8>,
    /// One UE8M0 scale byte per 32 elements (`len / 32` bytes).
    pub scales: Vec<u8>,
}

/// Quantize one activation row to MXFP8 with UE8M0 block scales.
pub fn quantize_mxfp8_ue8m0(input: &[f32]) -> Result<Mxfp8Row, FormatError> {
    if !input.len().is_multiple_of(MXFP8_BLOCK) {
        return Err(FormatError::Length {
            what: "MXFP8 activation row",
            got: input.len(),
            want: input.len().next_multiple_of(MXFP8_BLOCK),
        });
    }
    for &x in input {
        if !x.is_finite() {
            return Err(FormatError::NonFinite);
        }
    }
    let n_blocks = input.len() / MXFP8_BLOCK;
    let mut codes = vec![0u8; input.len()];
    let mut scales = vec![0u8; n_blocks];
    for (b, block) in input.as_chunks::<MXFP8_BLOCK>().0.iter().enumerate() {
        let block_amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if block_amax == 0.0 {
            scales[b] = 0;
            continue;
        }
        let scale_byte = encode_ue8m0_ceil(block_amax)?;
        let scale = decode_ue8m0(scale_byte)?;
        scales[b] = scale_byte;
        for (i, &x) in block.iter().enumerate() {
            codes[b * MXFP8_BLOCK + i] = encode_e4m3(x / scale)?;
        }
    }
    Ok(Mxfp8Row { codes, scales })
}

/// Exact block-scaled NVFP4 dot product mirroring `sm_121a` `m16n8k64` `f32` accumulation.
pub fn dot_nvfp4_block_scaled(
    act: &Nvfp4Row,
    w_codes: &[u8],
    w_scales: &[u8],
    w_factor: f32,
) -> Result<f32, FormatError> {
    if !w_factor.is_finite() {
        return Err(FormatError::NonFinite);
    }
    if act.codes.len() != w_codes.len() {
        return Err(FormatError::Length {
            what: "NVFP4 weight codes",
            got: w_codes.len(),
            want: act.codes.len(),
        });
    }
    if act.scales.len() != w_scales.len() {
        return Err(FormatError::Length {
            what: "NVFP4 weight scales",
            got: w_scales.len(),
            want: act.scales.len(),
        });
    }
    if act.global_scale == 0.0 || w_factor == 0.0 {
        return Ok(0.0);
    }
    let mut total = 0.0f32;
    for b in 0..act.scales.len() {
        let sa = decode_e4m3(act.scales[b])?;
        let sw = decode_e4m3(w_scales[b])?;
        let mut block_sum = 0.0f32;
        for i in 0..NVFP4_BLOCK {
            let byte_idx = b * (NVFP4_BLOCK / 2) + i / 2;
            let a_nib = if i % 2 == 0 {
                act.codes[byte_idx] & 0x0F
            } else {
                act.codes[byte_idx] >> 4
            };
            let w_nib = if i % 2 == 0 {
                w_codes[byte_idx] & 0x0F
            } else {
                w_codes[byte_idx] >> 4
            };
            block_sum += E2M1[usize::from(a_nib)] * E2M1[usize::from(w_nib)];
        }
        total += block_sum * (sa * sw);
    }
    Ok(total * (act.global_scale * w_factor))
}

/// Exact block-scaled MXFP8 dot product mirroring `sm_121a` `m16n8k32` `f32` accumulation.
pub fn dot_mxfp8_block_scaled(act: &Mxfp8Row, w_codes: &[u8], w_scales: &[u8]) -> Result<f32, FormatError> {
    if act.codes.len() != w_codes.len() {
        return Err(FormatError::Length {
            what: "MXFP8 weight codes",
            got: w_codes.len(),
            want: act.codes.len(),
        });
    }
    if act.scales.len() != w_scales.len() {
        return Err(FormatError::Length {
            what: "MXFP8 weight scales",
            got: w_scales.len(),
            want: act.scales.len(),
        });
    }
    let mut total = 0.0f32;
    for b in 0..act.scales.len() {
        let sa = decode_ue8m0(act.scales[b])?;
        let sw = decode_ue8m0(w_scales[b])?;
        let mut block_sum = 0.0f32;
        for i in 0..MXFP8_BLOCK {
            let idx = b * MXFP8_BLOCK + i;
            block_sum += decode_e4m3(act.codes[idx])? * decode_e4m3(w_codes[idx])?;
        }
        total += block_sum * (sa * sw);
    }
    Ok(total)
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
        assert_eq!(encode_e2m1(100.0).unwrap(), 7); // clamped to 6.0
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
    fn ue8m0_covers_amax_and_rejects_nan() {
        assert_eq!(decode_ue8m0(127).unwrap(), 1.0);
        assert_eq!(decode_ue8m0(255), Err(FormatError::Ue8m0Nan));
        assert_eq!(encode_ue8m0_ceil(0.0).unwrap(), 0);
        for amax in [1.0e-3f32, 0.5, 448.0, 449.0, 1000.0, 1.0e5] {
            let s = encode_ue8m0_ceil(amax).unwrap();
            assert!(decode_ue8m0(s).unwrap() * E4M3_MAX >= amax);
        }
    }

    #[test]
    fn fouroversix_mae_selects_four_for_dense_block_and_six_for_outlier_block() {
        // Block 0 has values near 3.0 and 4.0 (target 4 hits 3.0 and 4.0 exactly, whereas target 6 maps 4.0 to 4.0
        // and distorts the intermediate values). Block 1 has a 6.0 outlier alongside smaller values.
        // Make row_amax = 6.0 * 448.0 so global_scale = 1.0.
        // Block 0: values in [0, 4.0] scaled by 1.0 -> with target 4, scale = 1.0, {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0}
        // are exact! With target 6, scale = 4/6 = 0.6667 (E4M3 0.6875), every value incurs quantization error.
        let dense = [
            4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 3.0, 2.0, 1.5, 1.0,
        ];
        // Block 1: values scaled by 16.0 with max 6.0 * 16.0 = 96.0 (exact for target 6, scale = 16.0).
        let outlier_base = [
            96.0, 64.0, 48.0, 32.0, 16.0, 96.0, 64.0, 48.0, 32.0, 16.0, 96.0, 64.0, 48.0, 32.0, 16.0, 8.0,
        ];
        // Add a global scale reference by making row_amax = 6.0 * 448.0 in a 3rd block.
        let mut full = [0.0f32; 48];
        full[..16].copy_from_slice(&dense);
        full[16..32].copy_from_slice(&outlier_base);
        full[32] = 6.0 * 448.0;

        let q = quantize_nvfp4_fouroversix_mae(&full).unwrap();
        assert_eq!(q.global_scale, 1.0);
        // Block 0 picks target 4 -> scale = 4.0 / 4.0 = 1.0 (0x38).
        assert_eq!(decode_e4m3(q.scales[0]).unwrap(), 1.0);
        // Block 1 picks target 6 -> scale = 96.0 / 6.0 = 16.0.
        assert_eq!(decode_e4m3(q.scales[1]).unwrap(), 16.0);
    }

    #[test]
    fn zero_rows_and_block_scaled_dot_products_match_dequantized_math() {
        let zeros = vec![0.0f32; 32];
        let qz = quantize_nvfp4_fouroversix_mae(&zeros).unwrap();
        assert_eq!(qz.global_scale, 0.0);
        assert!(qz.codes.iter().all(|&b| b == 0));

        let act_f: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.25).collect();
        let w_f: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let qa = quantize_nvfp4_fouroversix_mae(&act_f).unwrap();
        let qw = quantize_nvfp4_fouroversix_mae(&w_f).unwrap();
        let nv_dot = dot_nvfp4_block_scaled(&qa, &qw.codes, &qw.scales, qw.global_scale).unwrap();
        let ref_dot: f32 = act_f.iter().zip(&w_f).map(|(a, b)| a * b).sum();
        assert!((nv_dot - ref_dot).abs() < 1.5, "nv_dot={nv_dot}, ref_dot={ref_dot}");

        let ma = quantize_mxfp8_ue8m0(&act_f).unwrap();
        let mw = quantize_mxfp8_ue8m0(&w_f).unwrap();
        let mx_dot = dot_mxfp8_block_scaled(&ma, &mw.codes, &mw.scales).unwrap();
        assert!((mx_dot - ref_dot).abs() < 0.25, "mx_dot={mx_dot}, ref_dot={ref_dot}");
    }
}
