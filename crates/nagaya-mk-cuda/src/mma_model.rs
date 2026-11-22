//! The host model of the bits of `OP_MATMUL_BF16`.
//!
//! The tensor core does not add in `f32`. The gate measured its rule on the GB10 (`bf16` `m16n8k16`, 25,600 results
//! of random tiles, and then the whole matmul on real weights; both match bit for bit): an MMA adds the 16 exact
//! products and the accumulator in one step.
//! - The reference exponent of a product is the sum of the exponent fields of its two factors, which is the exponent
//!   before the product is normalized. The reference exponent of the accumulator is its own. Let `top` be the largest.
//! - Each operand is cut, toward zero, to a multiple of `2^(top - 25)`. The cut operands add exactly.
//! - The sum is cut, toward zero, to the 24 bits of an `f32`.
//!
//! With this rule a host function gives the bits of each result, and a change of the order of the terms, of the tree
//! or of the merge changes the bits. An accuracy test cannot see such a change, because it moves a result by less
//! than its bound. The tests of the GPU ops use this module as the oracle of the bits.

use crate::repack::K_WARPS;

/// The fraction bits that an MMA keeps below the largest reference exponent.
const MMA_KEPT_BITS: i32 = 25;

/// `bf16(x)`: round to nearest even, as `cvt.rn.bf16.f32` does, and return it as an `f32`.
pub fn bf16_rn(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
}

/// The three terms of `x` in the order of the accumulation: lo, mid, hi.
pub fn split3(x: f32) -> [f32; 3] {
    let hi = bf16_rn(x);
    let r1 = x - hi;
    let mid = bf16_rn(r1);
    let lo = bf16_rn(r1 - mid);
    [lo, mid, hi]
}

/// `(negative, mantissa, exponent)`: the value of a finite `f32` is `mantissa * 2^exponent`.
type Operand = (bool, u64, i32);

pub fn operand(v: f32) -> Operand {
    let b = v.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    let f = u64::from(b & 0x7F_FFFF);
    if e == 0 {
        (b >> 31 != 0, f, -149)
    } else {
        (b >> 31 != 0, f | 0x80_0000, e - 150)
    }
}

/// The exponent field of a normal `f32`, without the bias.
pub fn exponent_field(v: f32) -> i32 {
    ((v.to_bits() >> 23) & 0xFF) as i32 - 127
}

/// One `mma.sync.m16n8k16` for one result: `acc + sum a[i] * b[i]`, where each `a[i]` and `b[i]` is a `bf16` value.
pub fn mma(acc: f32, a: &[f32], b: &[f32]) -> f32 {
    let mut operands: Vec<Operand> = Vec::with_capacity(a.len() + 1);
    let mut top: Option<i32> = None;
    let mut see = |reference: i32| top = Some(top.map_or(reference, |t| t.max(reference)));
    for (x, y) in a.iter().zip(b) {
        if *x != 0.0 && *y != 0.0 {
            see(exponent_field(*x) + exponent_field(*y));
            let ((sa, ma, ea), (sb, mb, eb)) = (operand(*x), operand(*y));
            operands.push((sa ^ sb, ma * mb, ea + eb));
        }
    }
    if acc != 0.0 {
        see(exponent_field(acc));
        operands.push(operand(acc));
    }
    let Some(top) = top else { return 0.0 };
    let unit = top - MMA_KEPT_BITS;
    let mut sum: i128 = 0;
    for (negative, mantissa, exponent) in operands {
        let shift = exponent - unit;
        let magnitude = match shift {
            0.. => i128::from(mantissa) << shift,
            -99..=-1 => i128::from(mantissa) >> -shift,
            _ => 0,
        };
        sum += if negative { -magnitude } else { magnitude };
    }
    if sum == 0 {
        return 0.0;
    }
    let magnitude = sum.unsigned_abs();
    let bits = 128 - magnitude.leading_zeros() as i32;
    let drop = (bits - 24).max(0);
    let value = ((magnitude >> drop) as f64 * 2f64.powi(unit + drop)) as f32;
    if sum < 0 { -value } else { value }
}

/// The bits that the op gives for one result: the dot of the `f32` row `x` and the `bf16` values `w` of one column
/// (widened to `f32`), with the K slice `kt`. It is the order of the sums in `kernels/matmul_bf16.cuh`.
pub fn dot_bits(w: &[f32], x: &[f32], kt: usize) -> f32 {
    let k = x.len();
    assert_eq!(w.len(), k, "dot_bits: the column and the row differ in length");
    let split: Vec<[f32; 3]> = x.iter().map(|v| split3(*v)).collect();
    let steps_per_warp = kt / 16 / K_WARPS;
    let mut slices = Vec::new();
    for slice in 0..k / kt {
        // Each warp runs its steps in order, and in each step the terms lo, mid and hi.
        let mut warps = [0f32; K_WARPS];
        for (warp, acc) in warps.iter_mut().enumerate() {
            for step in warp * steps_per_warp..(warp + 1) * steps_per_warp {
                let k0 = slice * kt + step * 16;
                for term in 0..3 {
                    let b: Vec<f32> = split[k0..k0 + 16].iter().map(|t| t[term]).collect();
                    *acc = mma(*acc, &w[k0..k0 + 16], &b);
                }
            }
        }
        // The tree of the warps in shared memory.
        let pair = |i: usize| warps[i] + warps[i + 1];
        slices.push((pair(0) + pair(2)) + (pair(4) + pair(6)));
    }
    // The merge group adds the slices in index order.
    slices[1..].iter().fold(slices[0], |sum, p| sum + p)
}
