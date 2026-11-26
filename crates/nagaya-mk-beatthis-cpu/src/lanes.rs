//! `F4`: four f32 values in one register, for the kernels. Each operation is one IEEE operation on each value, with
//! the rounding of the same scalar operation, so a kernel that uses `F4` gives the bits of the scalar form. There is
//! no fused multiply-add.
//!
//! On x86_64 it is an SSE register (SSE2 is in each x86_64 CPU). On other machines it is an array.

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct F4(__m128);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> Self {
            // SAFETY: SSE is in each x86_64 CPU.
            F4(unsafe { _mm_setzero_ps() })
        }
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            // SAFETY: SSE is in each x86_64 CPU.
            F4(unsafe { _mm_set1_ps(v) })
        }
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> Self {
            // SAFETY: `a` holds 4 values, and an unaligned load reads exactly them.
            F4(unsafe { _mm_loadu_ps(a.as_ptr()) })
        }
        #[inline(always)]
        pub fn store(self, a: &mut [f32; 4]) {
            // SAFETY: `a` holds 4 values, and an unaligned store writes exactly them.
            unsafe { _mm_storeu_ps(a.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            // SAFETY: SSE is in each x86_64 CPU.
            F4(unsafe { _mm_add_ps(self.0, o.0) })
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            // SAFETY: SSE is in each x86_64 CPU.
            F4(unsafe { _mm_sub_ps(self.0, o.0) })
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            // SAFETY: SSE is in each x86_64 CPU.
            F4(unsafe { _mm_mul_ps(self.0, o.0) })
        }

        /// `libm::expf` of each value, with its bits. The code is the code of `libm::expf` (FreeBSD's `expf`) for
        /// the values in its main range, `2^-14 < |x| < 87.34`, done on 4 values at once with selects for its
        /// branches. There `k` is in -126..=126, so `scalbnf(y, k)` is one multiply by `2^k`, as in `libm`. A value
        /// of at most `2^-14` in magnitude gives `1 + x`, and a value of at most -103.97 gives 0, as in `libm`. When
        /// a value is in neither range (NaN, a large positive value, or a value from -103.97 to -87.34) in a lane, all 4
        /// values go to `libm::expf` one by one.
        #[inline(always)]
        pub fn exp(self) -> Self {
            const HALF: f32 = 0.5;
            // The constants of libm's expf, by their bits.
            const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 6.9314575195e-01
            const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 1.4286067653e-06
            const INV_LN2: f32 = f32::from_bits(0x3fb8_aa3b); // 1.4426950216e+00
            const P1: f32 = f32::from_bits(0x3e2a_aa8f); // 1.6666625440e-1
            const P2: f32 = f32::from_bits(0xbb35_5215); // -2.7667332906e-3
            // SAFETY: SSE and SSE2 are in each x86_64 CPU.
            unsafe {
                let x = self.0;
                let bits = _mm_castps_si128(x);
                let hx = _mm_and_si128(bits, _mm_set1_epi32(0x7fff_ffff));
                // All ones in each negative lane.
                let neg = _mm_srai_epi32::<31>(bits);
                let gt = |c: u32| _mm_cmpgt_epi32(hx, _mm_set1_epi32(c as i32));
                let tiny = _mm_xor_si128(gt(0x3900_0000), _mm_set1_epi32(-1));
                let big = gt(0x42ae_ac4f);
                // A NaN is not a zero lane, whatever its sign: libm gives it back.
                let zero = _mm_andnot_si128(gt(0x7f80_0000), _mm_and_si128(neg, gt(0x42cf_f1b4)));
                // The lanes that are big but not zero go to libm.
                if _mm_movemask_ps(_mm_castsi128_ps(_mm_andnot_si128(zero, big))) != 0 {
                    let mut a = [0.0f32; 4];
                    _mm_storeu_ps(a.as_mut_ptr(), x);
                    for v in a.iter_mut() {
                        *v = libm::expf(*v);
                    }
                    return F4(_mm_loadu_ps(a.as_ptr()));
                }
                // k: (INV_LN2 * x + HALF[sign]) as i32 above 1.5 ln2, 1 - 2 * sign above 0.5 ln2, else 0.
                let half = _mm_or_ps(_mm_set1_ps(HALF), _mm_and_ps(_mm_castsi128_ps(neg), _mm_set1_ps(-0.0)));
                let k_far = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(INV_LN2), x), half));
                let k_near = _mm_sub_epi32(_mm_set1_epi32(1), _mm_and_si128(neg, _mm_set1_epi32(2)));
                let far = gt(0x3f85_1592);
                let near = _mm_andnot_si128(far, gt(0x3eb1_7218));
                let k = _mm_or_si128(_mm_and_si128(far, k_far), _mm_and_si128(near, k_near));
                let kf = _mm_cvtepi32_ps(k);
                let hi = _mm_sub_ps(x, _mm_mul_ps(kf, _mm_set1_ps(LN2_HI)));
                let lo = _mm_mul_ps(kf, _mm_set1_ps(LN2_LO));
                let r = _mm_sub_ps(hi, lo);
                let rr = _mm_mul_ps(r, r);
                let c = _mm_sub_ps(
                    r,
                    _mm_mul_ps(rr, _mm_add_ps(_mm_set1_ps(P1), _mm_mul_ps(rr, _mm_set1_ps(P2)))),
                );
                let q = _mm_div_ps(_mm_mul_ps(r, c), _mm_sub_ps(_mm_set1_ps(2.0), c));
                let y = _mm_add_ps(_mm_set1_ps(1.0), _mm_add_ps(_mm_sub_ps(q, lo), hi));
                let scale = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(k, _mm_set1_epi32(127))));
                let main = _mm_mul_ps(y, scale);
                let one_plus = _mm_add_ps(_mm_set1_ps(1.0), x);
                let tiny = _mm_castsi128_ps(tiny);
                let zero = _mm_castsi128_ps(zero);
                let v = _mm_or_ps(_mm_and_ps(tiny, one_plus), _mm_andnot_ps(tiny, main));
                F4(_mm_andnot_ps(zero, v))
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> Self {
            F4([0.0; 4])
        }
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            F4([v; 4])
        }
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> Self {
            F4(*a)
        }
        #[inline(always)]
        pub fn store(self, a: &mut [f32; 4]) {
            *a = self.0;
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            F4(std::array::from_fn(|k| self.0[k] + o.0[k]))
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            F4(std::array::from_fn(|k| self.0[k] - o.0[k]))
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            F4(std::array::from_fn(|k| self.0[k] * o.0[k]))
        }

        /// `libm::expf` of each value.
        #[inline(always)]
        pub fn exp(self) -> Self {
            F4(self.0.map(libm::expf))
        }
    }
}

pub use imp::F4;

impl F4 {
    /// The 4 values from `s[at..at + 4]`.
    #[inline(always)]
    pub fn at(s: &[f32], at: usize) -> Self {
        F4::load(s[at..at + 4].try_into().unwrap())
    }

    /// The 4 values as an array.
    #[inline(always)]
    pub fn array(self) -> [f32; 4] {
        let mut a = [0.0; 4];
        self.store(&mut a);
        a
    }
}
