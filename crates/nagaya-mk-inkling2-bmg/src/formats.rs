//! The expansion of the NVFP4 and MXFP8 expert weights of Inkling 2 to `bf16` bits.
//!
//! Each op follows the order of the `f32` operations of the reference, because the order changes bits.
//! A code that has no value is an error. The nagaya does not turn it into a number (spec 8.5).

use std::fmt;

/// The values of the E2M1 codes 0 to 15. Codes 8 to 15 are the negative copies of codes 0 to 7.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The count of values that share one E4M3 scale in NVFP4.
pub const NVFP4_BLOCK: usize = 16;

/// The count of values that share one UE8M0 scale in MXFP8.
pub const MXFP8_BLOCK: usize = 32;

/// What a weight op refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatError {
    /// The E4M3 code is NaN (`0x7F` or `0xFF`). The checkpoint must not hold it.
    E4m3Nan(u8),
    /// The UE8M0 scale is 255, which is NaN in the OCP MX format. The checkpoint must not hold it.
    Ue8m0Nan,
    /// The result of an expansion is NaN. The expert factor is NaN or infinite, or the scale overflowed.
    ResultNan,
    /// A slice has a length that does not fit the others.
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
                "the E4M3 code 0x{c:02x} is NaN. The checkpoint is damaged, or it is not Inkling 2. Download it again."
            ),
            Self::Ue8m0Nan => write!(
                f,
                "the UE8M0 scale 255 is NaN. The checkpoint is damaged, or it is not Inkling 2. Download it again."
            ),
            Self::ResultNan => write!(
                f,
                "an expanded weight is NaN. The factor of the expert is not finite. Check amax and its denominator."
            ),
            Self::Length { what, got, want } => {
                write!(
                    f,
                    "the length of {what} is {got}, and the format needs {want}. Pass whole blocks of a row."
                )
            }
        }
    }
}

impl std::error::Error for FormatError {}

/// Decode one E4M3 code (4 exponent bits, bias 7, 3 mantissa bits, no infinity). The result is exact in `f32`.
/// The largest value is 448 (`0x7E`). The codes `0x7F` and `0xFF` are an error.
pub fn decode_e4m3(code: u8) -> Result<f32, FormatError> {
    if code & 0x7F == 0x7F {
        return Err(FormatError::E4m3Nan(code));
    }
    let exponent = i32::from((code >> 3) & 0xF);
    let mantissa = f32::from(code & 7);
    // A subnormal code has no implicit bit, and its exponent is 1 - 7.
    let magnitude = if exponent == 0 {
        mantissa * pow2(-9)
    } else {
        (8.0 + mantissa) * pow2(exponent - 10)
    };
    Ok(if code & 0x80 == 0 { magnitude } else { -magnitude })
}

/// Decode one UE8M0 scale: the value is 2 to the power of `code - 127`. The result is exact in `f32`.
/// The code 0 gives 2^-127, which is subnormal in `f32`. The code 255 is an error.
pub fn decode_ue8m0(code: u8) -> Result<f32, FormatError> {
    match code {
        255 => Err(FormatError::Ue8m0Nan),
        0 => Ok(f32::from_bits(1 << 22)),
        c => Ok(f32::from_bits(u32::from(c) << 23)),
    }
}

/// The power of two for an exponent from -126 to 127, built from its bits so the value is exact.
fn pow2(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// Round an `f32` to `bf16` bits, to the nearest, ties to even. This is the rounding of `Tensor.to(torch.bfloat16)`.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// The factor of one expert in NVFP4: `amax / denominator`, divided once in `f32`.
pub fn expert_factor(amax: f32, denominator: f32) -> f32 {
    amax / denominator
}

/// Expand one row of an NVFP4 expert to `bf16` bits.
///
/// `codes` holds two E2M1 codes in each byte, and the even element is the low nibble. `scales` holds one E4M3 byte for
/// each 16 values. The value is `(e2m1 * scale) * factor` in `f32`, and then it is rounded to `bf16`.
pub fn expand_nvfp4_row(codes: &[u8], scales: &[u8], factor: f32, out: &mut [u16]) -> Result<(), FormatError> {
    check_length("the output row", out.len(), codes.len() * 2)?;
    check_length("the scales of the row", scales.len(), out.len() / NVFP4_BLOCK)?;
    check_length("the part of the row after whole blocks", out.len() % NVFP4_BLOCK, 0)?;
    for (block, scale) in scales.iter().enumerate() {
        let scale = decode_e4m3(*scale)?;
        for i in block * NVFP4_BLOCK..(block + 1) * NVFP4_BLOCK {
            let byte = codes[i / 2];
            let code = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            out[i] = round_checked((E2M1[usize::from(code)] * scale) * factor)?;
        }
    }
    Ok(())
}

/// Expand one row of an MXFP8 expert to `bf16` bits.
///
/// `values` holds one E4M3 byte for each value. `scales` holds one UE8M0 byte for each 32 values. The value is
/// `e4m3 * 2^(scale - 127)` in `f32`, and then it is rounded to `bf16`.
pub fn expand_mxfp8_row(values: &[u8], scales: &[u8], out: &mut [u16]) -> Result<(), FormatError> {
    check_length("the output row", out.len(), values.len())?;
    check_length("the scales of the row", scales.len(), values.len() / MXFP8_BLOCK)?;
    check_length("the part of the row after whole blocks", out.len() % MXFP8_BLOCK, 0)?;
    for (block, scale) in scales.iter().enumerate() {
        let scale = decode_ue8m0(*scale)?;
        for i in block * MXFP8_BLOCK..(block + 1) * MXFP8_BLOCK {
            out[i] = round_checked(decode_e4m3(values[i])? * scale)?;
        }
    }
    Ok(())
}

/// The checkpoint row that holds row `row` of the loaded `gate_up` matrix of `rows` rows.
///
/// The checkpoint interleaves: row 2i is gate row i, and row 2i+1 is up row i. The loaded matrix holds all the gate
/// rows, then all the up rows (`Interleave(dim=1)` of transformers). `rows` is even.
pub fn gate_up_source_row(row: usize, rows: usize) -> usize {
    assert!(
        rows.is_multiple_of(2) && row < rows,
        "gate_up_source_row: row {row} of {rows} rows (the count must be even)"
    );
    let half = rows / 2;
    if row < half { 2 * row } else { 2 * (row - half) + 1 }
}

fn round_checked(value: f32) -> Result<u16, FormatError> {
    if value.is_nan() {
        return Err(FormatError::ResultNan);
    }
    Ok(f32_to_bf16(value))
}

fn check_length(what: &'static str, got: usize, want: usize) -> Result<(), FormatError> {
    if got == want {
        Ok(())
    } else {
        Err(FormatError::Length { what, got, want })
    }
}
