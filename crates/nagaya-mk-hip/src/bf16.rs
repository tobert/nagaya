//! The bf16 numbers of the ops: the upper 16 bits of an f32.

/// The bf16 bits of `x`, rounded to the nearest, ties to even. A NaN stays a quiet NaN.
pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// The f32 value of the bf16 bits `h`. It is exact.
pub fn to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// `x` rounded to bf16, as an f32.
pub fn round(x: f32) -> f32 {
    to_f32(from_f32(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_to_the_nearest_with_ties_to_even() {
        assert_eq!(from_f32(1.0), 0x3f80);
        // 1 + 2^-8 is half way between 1 and 1 + 2^-7: the tie goes to the even 1.
        assert_eq!(from_f32(1.0 + 1.0 / 256.0), 0x3f80);
        // 1 + 3 * 2^-8 is half way between 1 + 2^-7 and 1 + 2^-6: the tie goes to the even 1 + 2^-6.
        assert_eq!(from_f32(1.0 + 3.0 / 256.0), 0x3f82);
        assert_eq!(from_f32(1.0 + 1.0 / 256.0 + 1.0 / 65536.0), 0x3f81);
        assert_eq!(from_f32(-2.5), 0xc020);
        assert!(to_f32(from_f32(f32::NAN)).is_nan());
        assert_eq!(from_f32(f32::INFINITY), 0x7f80);
        assert_eq!(to_f32(0x4049), 3.140625);
    }
}
