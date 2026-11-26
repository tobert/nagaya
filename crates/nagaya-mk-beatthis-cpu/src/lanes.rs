//! The vectors of the kernels: `F4` (SSE), `F8` (AVX2) and `F16` (AVX-512F), behind the trait `Vf`, and the code of
//! each level (`Isa`). Each operation is one IEEE operation on each value, with the rounding of the same scalar
//! operation, so a kernel gives the bits of its scalar form at each level. There is no fused multiply-add.
//!
//! On x86_64 `F4` is an SSE register (SSE2 is in each x86_64 CPU). On other machines it is an array, and there are
//! no `F8` and `F16`. The methods of `F8` and `F16` run only in the functions that `isa::leveled!` makes for their
//! level, which run only when the CPU has the instructions.

/// A vector of `N` f32 values, and a mask of `N` lanes (`M`).
pub trait Vf: Copy {
    const N: usize;
    type M: Copy;
    fn zero() -> Self;
    fn splat(v: f32) -> Self;
    /// The `N` values from `s[i..i + N]`.
    fn at(s: &[f32], i: usize) -> Self;
    /// Writes the `N` values to `s[i..i + N]`.
    fn put(self, s: &mut [f32], i: usize);
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    /// The lanes where `bits & 0x7fff_ffff > c`.
    fn abs_bits_gt(self, c: u32) -> Self::M;
    /// The lanes with the sign bit.
    fn negative(self) -> Self::M;
    fn and(a: Self::M, b: Self::M) -> Self::M;
    /// `!a & b`.
    fn and_not(a: Self::M, b: Self::M) -> Self::M;
    fn any(m: Self::M) -> bool;
    /// `m ? a : b` in each lane.
    fn select(m: Self::M, a: Self, b: Self) -> Self;
    /// `(v as i32) as f32` in each lane: the truncation toward 0, for values in the range of i32.
    fn trunc(self) -> Self;
    /// `2^k` in each lane, for `k` a whole number in -126..=127.
    fn pow2(self) -> Self;

    /// `f` of each value, one by one.
    #[inline(always)]
    fn map(self, f: impl Fn(f32) -> f32) -> Self {
        let mut a = [0.0f32; 16];
        self.put(&mut a, 0);
        for v in &mut a[..Self::N] {
            *v = f(*v);
        }
        Self::at(&a, 0)
    }
}

/// `libm::expf` of each value, with its bits. The code is the code of `libm::expf` (FreeBSD's `expf`) for the values
/// in its main range, `2^-13 < |x| < 87.34`, with selects for its branches. There `k` is in -126..=126, so
/// `scalbnf(y, k)` is one multiply by `2^k`, as in `libm`. A value of at most `2^-13` in magnitude gives `1 + x`,
/// and a value of at most -103.97 gives 0, as in `libm`. When a lane holds a value in neither range (NaN, a large
/// positive value, or a value from -103.97 to -87.34), all the values go to `libm::expf` one by one.
#[inline(always)]
pub fn exp<V: Vf>(x: V) -> V {
    // The constants of libm's expf, by their bits.
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 6.9314575195e-01
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 1.4286067653e-06
    const INV_LN2: f32 = f32::from_bits(0x3fb8_aa3b); // 1.4426950216e+00
    const P1: f32 = f32::from_bits(0x3e2a_aa8f); // 1.6666625440e-1
    const P2: f32 = f32::from_bits(0xbb35_5215); // -2.7667332906e-3
    let neg = x.negative();
    let big = x.abs_bits_gt(0x42ae_ac4f);
    // A NaN is not a zero lane, whatever its sign: libm gives it back.
    let zero = V::and_not(x.abs_bits_gt(0x7f80_0000), V::and(neg, x.abs_bits_gt(0x42cf_f1b4)));
    // The lanes that are big but not zero go to libm.
    if V::any(V::and_not(zero, big)) {
        return x.map(libm::expf);
    }
    // k: (INV_LN2 * x + HALF[sign]) as i32 above 1.5 ln2, 1 - 2 * sign above 0.5 ln2, else 0.
    let half = V::select(neg, V::splat(-0.5), V::splat(0.5));
    let k_far = V::splat(INV_LN2).mul(x).add(half).trunc();
    let k_near = V::select(neg, V::splat(-1.0), V::splat(1.0));
    let kf = V::select(
        x.abs_bits_gt(0x3f85_1592),
        k_far,
        V::select(x.abs_bits_gt(0x3eb1_7218), k_near, V::zero()),
    );
    let hi = x.sub(kf.mul(V::splat(LN2_HI)));
    let lo = kf.mul(V::splat(LN2_LO));
    let r = hi.sub(lo);
    let rr = r.mul(r);
    let c = r.sub(rr.mul(V::splat(P1).add(rr.mul(V::splat(P2)))));
    let q = r.mul(c).div(V::splat(2.0).sub(c));
    let y = V::splat(1.0).add(q.sub(lo).add(hi));
    let main = y.mul(kf.pow2());
    let v = V::select(x.abs_bits_gt(0x3900_0000), main, V::splat(1.0).add(x));
    V::select(zero, V::zero(), v)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Vf;

    /// 4 values in an SSE register.
    #[derive(Clone, Copy)]
    pub struct F4(__m128);

    // SAFETY (for each `unsafe` block of F4): SSE and SSE2 are in each x86_64 CPU, and each load and store reads
    // or writes exactly the values of a checked slice.
    impl Vf for F4 {
        const N: usize = 4;
        type M = __m128;
        #[inline(always)]
        fn zero() -> Self {
            F4(unsafe { _mm_setzero_ps() })
        }
        #[inline(always)]
        fn splat(v: f32) -> Self {
            F4(unsafe { _mm_set1_ps(v) })
        }
        #[inline(always)]
        fn at(s: &[f32], i: usize) -> Self {
            let c = &s[i..i + 4];
            F4(unsafe { _mm_loadu_ps(c.as_ptr()) })
        }
        #[inline(always)]
        fn put(self, s: &mut [f32], i: usize) {
            let c = &mut s[i..i + 4];
            unsafe { _mm_storeu_ps(c.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            F4(unsafe { _mm_add_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            F4(unsafe { _mm_sub_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            F4(unsafe { _mm_mul_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            F4(unsafe { _mm_div_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn abs_bits_gt(self, c: u32) -> __m128 {
            unsafe {
                let hx = _mm_and_si128(_mm_castps_si128(self.0), _mm_set1_epi32(0x7fff_ffff));
                _mm_castsi128_ps(_mm_cmpgt_epi32(hx, _mm_set1_epi32(c as i32)))
            }
        }
        #[inline(always)]
        fn negative(self) -> __m128 {
            unsafe { _mm_castsi128_ps(_mm_srai_epi32::<31>(_mm_castps_si128(self.0))) }
        }
        #[inline(always)]
        fn and(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_and_ps(a, b) }
        }
        #[inline(always)]
        fn and_not(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_andnot_ps(a, b) }
        }
        #[inline(always)]
        fn any(m: __m128) -> bool {
            unsafe { _mm_movemask_ps(m) != 0 }
        }
        #[inline(always)]
        fn select(m: __m128, a: Self, b: Self) -> Self {
            F4(unsafe { _mm_or_ps(_mm_and_ps(m, a.0), _mm_andnot_ps(m, b.0)) })
        }
        #[inline(always)]
        fn trunc(self) -> Self {
            F4(unsafe { _mm_cvtepi32_ps(_mm_cvttps_epi32(self.0)) })
        }
        #[inline(always)]
        fn pow2(self) -> Self {
            unsafe {
                let k = _mm_cvtps_epi32(self.0);
                F4(_mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(
                    k,
                    _mm_set1_epi32(127),
                ))))
            }
        }
    }

    /// 8 values in an AVX register.
    #[derive(Clone, Copy)]
    pub struct F8(pub(super) __m256);

    // SAFETY (for each `unsafe` block of F8): the methods run only in functions compiled for AVX2 (isa::leveled!),
    // which run only when the CPU has AVX2. Each load and store reads or writes exactly the values of a checked
    // slice.
    impl Vf for F8 {
        const N: usize = 8;
        type M = __m256;
        #[inline(always)]
        fn zero() -> Self {
            F8(unsafe { _mm256_setzero_ps() })
        }
        #[inline(always)]
        fn splat(v: f32) -> Self {
            F8(unsafe { _mm256_set1_ps(v) })
        }
        #[inline(always)]
        fn at(s: &[f32], i: usize) -> Self {
            let c = &s[i..i + 8];
            F8(unsafe { _mm256_loadu_ps(c.as_ptr()) })
        }
        #[inline(always)]
        fn put(self, s: &mut [f32], i: usize) {
            let c = &mut s[i..i + 8];
            unsafe { _mm256_storeu_ps(c.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            F8(unsafe { _mm256_add_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            F8(unsafe { _mm256_sub_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            F8(unsafe { _mm256_mul_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            F8(unsafe { _mm256_div_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn abs_bits_gt(self, c: u32) -> __m256 {
            unsafe {
                let hx = _mm256_and_si256(_mm256_castps_si256(self.0), _mm256_set1_epi32(0x7fff_ffff));
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(hx, _mm256_set1_epi32(c as i32)))
            }
        }
        #[inline(always)]
        fn negative(self) -> __m256 {
            unsafe { _mm256_castsi256_ps(_mm256_srai_epi32::<31>(_mm256_castps_si256(self.0))) }
        }
        #[inline(always)]
        fn and(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_and_ps(a, b) }
        }
        #[inline(always)]
        fn and_not(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_andnot_ps(a, b) }
        }
        #[inline(always)]
        fn any(m: __m256) -> bool {
            unsafe { _mm256_movemask_ps(m) != 0 }
        }
        #[inline(always)]
        fn select(m: __m256, a: Self, b: Self) -> Self {
            F8(unsafe { _mm256_blendv_ps(b.0, a.0, m) })
        }
        #[inline(always)]
        fn trunc(self) -> Self {
            F8(unsafe { _mm256_cvtepi32_ps(_mm256_cvttps_epi32(self.0)) })
        }
        #[inline(always)]
        fn pow2(self) -> Self {
            unsafe {
                let k = _mm256_cvtps_epi32(self.0);
                F8(_mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
                    k,
                    _mm256_set1_epi32(127),
                ))))
            }
        }
    }

    /// 16 values in an AVX-512 register.
    #[derive(Clone, Copy)]
    pub struct F16(pub(super) __m512);

    // SAFETY (for each `unsafe` block of F16): as F8, with AVX-512F.
    impl Vf for F16 {
        const N: usize = 16;
        type M = __mmask16;
        #[inline(always)]
        fn zero() -> Self {
            F16(unsafe { _mm512_setzero_ps() })
        }
        #[inline(always)]
        fn splat(v: f32) -> Self {
            F16(unsafe { _mm512_set1_ps(v) })
        }
        #[inline(always)]
        fn at(s: &[f32], i: usize) -> Self {
            let c = &s[i..i + 16];
            F16(unsafe { _mm512_loadu_ps(c.as_ptr()) })
        }
        #[inline(always)]
        fn put(self, s: &mut [f32], i: usize) {
            let c = &mut s[i..i + 16];
            unsafe { _mm512_storeu_ps(c.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            F16(unsafe { _mm512_add_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            F16(unsafe { _mm512_sub_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            F16(unsafe { _mm512_mul_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            F16(unsafe { _mm512_div_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn abs_bits_gt(self, c: u32) -> __mmask16 {
            unsafe {
                let hx = _mm512_and_si512(_mm512_castps_si512(self.0), _mm512_set1_epi32(0x7fff_ffff));
                _mm512_cmpgt_epi32_mask(hx, _mm512_set1_epi32(c as i32))
            }
        }
        #[inline(always)]
        fn negative(self) -> __mmask16 {
            unsafe { _mm512_cmplt_epi32_mask(_mm512_castps_si512(self.0), _mm512_setzero_si512()) }
        }
        #[inline(always)]
        fn and(a: __mmask16, b: __mmask16) -> __mmask16 {
            a & b
        }
        #[inline(always)]
        fn and_not(a: __mmask16, b: __mmask16) -> __mmask16 {
            !a & b
        }
        #[inline(always)]
        fn any(m: __mmask16) -> bool {
            m != 0
        }
        #[inline(always)]
        fn select(m: __mmask16, a: Self, b: Self) -> Self {
            F16(unsafe { _mm512_mask_blend_ps(m, b.0, a.0) })
        }
        #[inline(always)]
        fn trunc(self) -> Self {
            F16(unsafe { _mm512_cvtepi32_ps(_mm512_cvttps_epi32(self.0)) })
        }
        #[inline(always)]
        fn pow2(self) -> Self {
            unsafe {
                let k = _mm512_cvtps_epi32(self.0);
                F16(_mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
                    k,
                    _mm512_set1_epi32(127),
                ))))
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod plain {
    use super::Vf;

    /// 4 values in an array.
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    impl F4 {
        #[inline(always)]
        fn zip(self, o: Self, f: impl Fn(f32, f32) -> f32) -> Self {
            F4(std::array::from_fn(|k| f(self.0[k], o.0[k])))
        }
    }

    impl Vf for F4 {
        const N: usize = 4;
        type M = [bool; 4];
        fn zero() -> Self {
            F4([0.0; 4])
        }
        fn splat(v: f32) -> Self {
            F4([v; 4])
        }
        fn at(s: &[f32], i: usize) -> Self {
            F4(s[i..i + 4].try_into().unwrap())
        }
        fn put(self, s: &mut [f32], i: usize) {
            s[i..i + 4].copy_from_slice(&self.0);
        }
        fn add(self, o: Self) -> Self {
            self.zip(o, |a, b| a + b)
        }
        fn sub(self, o: Self) -> Self {
            self.zip(o, |a, b| a - b)
        }
        fn mul(self, o: Self) -> Self {
            self.zip(o, |a, b| a * b)
        }
        fn div(self, o: Self) -> Self {
            self.zip(o, |a, b| a / b)
        }
        fn abs_bits_gt(self, c: u32) -> [bool; 4] {
            self.0.map(|v| (v.to_bits() & 0x7fff_ffff) > c)
        }
        fn negative(self) -> [bool; 4] {
            self.0.map(|v| v.to_bits() >> 31 == 1)
        }
        fn and(a: [bool; 4], b: [bool; 4]) -> [bool; 4] {
            std::array::from_fn(|k| a[k] && b[k])
        }
        fn and_not(a: [bool; 4], b: [bool; 4]) -> [bool; 4] {
            std::array::from_fn(|k| !a[k] && b[k])
        }
        fn any(m: [bool; 4]) -> bool {
            m.iter().any(|&b| b)
        }
        fn select(m: [bool; 4], a: Self, b: Self) -> Self {
            F4(std::array::from_fn(|k| if m[k] { a.0[k] } else { b.0[k] }))
        }
        fn trunc(self) -> Self {
            F4(self.0.map(|v| v as i32 as f32))
        }
        fn pow2(self) -> Self {
            F4(self
                .0
                .map(|k| f32::from_bits(((k as i32).wrapping_add(127) as u32) << 23)))
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::{F4, F8, F16};

#[cfg(not(target_arch = "x86_64"))]
pub use plain::F4;

/// `((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))`: the join of the 8 lanes of a dot.
#[inline(always)]
pub fn join(l: [f32; 8]) -> f32 {
    ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
}

/// Adds the tail terms `a[i] * b[i]` (the values past the last chunk of 8) to their lanes, and joins the lanes.
#[inline(always)]
fn finish(mut lanes: [f32; 8], a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    for (k, lane) in lanes.iter_mut().enumerate().take(a.len() % 8) {
        let i = chunks * 8 + k;
        *lane += a[i] * b[i];
    }
    join(lanes)
}

/// The kernel code of one level (`isa::Level`). Each method gives the bits of the scalar form.
pub trait Isa {
    /// The vector of the level.
    type V: Vf;

    /// The dots of `R` rows of x against `O` rows of a weight, each with the 8 lanes and the join of
    /// `kernels::dot`. A block loads each chunk of 8 values once for `R * O` dots, and keeps the `R * O` sums
    /// apart, so the adds do not wait on each other.
    fn dots<const R: usize, const O: usize>(xs: [&[f32]; R], ws: [&[f32]; O]) -> [[f32; O]; R];
}

/// SSE: the 8 lanes of a dot in two `F4`.
pub struct Sse;

impl Isa for Sse {
    type V = F4;
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn dots<const R: usize, const O: usize>(xs: [&[f32]; R], ws: [&[f32]; O]) -> [[f32; O]; R] {
        let din = xs[0].len();
        // l[r][o][h]: lanes 4h to 4h + 3 of the dot of row r and output o.
        let mut l = [[[F4::zero(); 2]; O]; R];
        for c in 0..din / 8 {
            for h in 0..2 {
                let xc: [F4; R] = std::array::from_fn(|r| F4::at(xs[r], c * 8 + 4 * h));
                let wc: [F4; O] = std::array::from_fn(|o| F4::at(ws[o], c * 8 + 4 * h));
                for r in 0..R {
                    for o in 0..O {
                        l[r][o][h] = l[r][o][h].add(xc[r].mul(wc[o]));
                    }
                }
            }
        }
        std::array::from_fn(|r| {
            std::array::from_fn(|o| {
                let mut lanes = [0.0f32; 8];
                l[r][o][0].put(&mut lanes, 0);
                l[r][o][1].put(&mut lanes, 4);
                finish(lanes, xs[r], ws[o])
            })
        })
    }
}

#[cfg(target_arch = "x86_64")]
pub use wide::{Avx2, Avx512};

/// The AVX2 and AVX-512 levels. Their methods run only in the functions of `isa::avx2` and `isa::avx512`, which the
/// dispatch calls when the CPU has the instructions.
#[cfg(target_arch = "x86_64")]
mod wide {
    use std::arch::x86_64::*;

    use super::{F8, F16, Isa, finish};

    /// AVX2: the 8 lanes of a dot in one register.
    pub struct Avx2;

    /// AVX-512F: the 8 lanes of two dots in one register.
    pub struct Avx512;

    #[inline(always)]
    fn lanes8(v: __m256) -> [f32; 8] {
        let mut a = [0.0f32; 8];
        // SAFETY: `a` holds 8 values; the caller runs with AVX (isa::avx2 or isa::avx512).
        unsafe { _mm256_storeu_ps(a.as_mut_ptr(), v) };
        a
    }

    /// 8 values from `s[at..at + 8]`.
    #[inline(always)]
    fn load8(s: &[f32], at: usize) -> __m256 {
        let c = &s[at..at + 8];
        // SAFETY: `c` holds 8 values; the caller runs with AVX.
        unsafe { _mm256_loadu_ps(c.as_ptr()) }
    }

    impl Isa for Avx2 {
        type V = F8;
        #[inline(always)]
        #[allow(clippy::needless_range_loop)]
        fn dots<const R: usize, const O: usize>(xs: [&[f32]; R], ws: [&[f32]; O]) -> [[f32; O]; R] {
            let din = xs[0].len();
            // SAFETY: the caller runs with AVX2.
            unsafe {
                let mut l = [[_mm256_setzero_ps(); O]; R];
                for c in 0..din / 8 {
                    let xc: [__m256; R] = std::array::from_fn(|r| load8(xs[r], c * 8));
                    for o in 0..O {
                        let w = load8(ws[o], c * 8);
                        for r in 0..R {
                            l[r][o] = _mm256_add_ps(l[r][o], _mm256_mul_ps(xc[r], w));
                        }
                    }
                }
                std::array::from_fn(|r| std::array::from_fn(|o| finish(lanes8(l[r][o]), xs[r], ws[o])))
            }
        }
    }

    impl Isa for Avx512 {
        type V = F16;
        /// The outputs go in pairs: one register holds the lanes of output `2p` in its low half and of `2p + 1` in
        /// its high half, and the chunk of x is in both halves. An odd last output uses an AVX register.
        #[inline(always)]
        #[allow(clippy::needless_range_loop)]
        fn dots<const R: usize, const O: usize>(xs: [&[f32]; R], ws: [&[f32]; O]) -> [[f32; O]; R] {
            let din = xs[0].len();
            let pairs = O / 2;
            let mut out = [[0.0f32; O]; R];
            // SAFETY: the caller runs with AVX-512F.
            unsafe {
                let mut l = [[_mm512_setzero_ps(); O]; R];
                let mut odd = [_mm256_setzero_ps(); R];
                for c in 0..din / 8 {
                    let xc: [__m512; R] = std::array::from_fn(|r| {
                        _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castps_pd(load8(xs[r], c * 8))))
                    });
                    for p in 0..pairs {
                        let lo = _mm512_castps256_ps512(load8(ws[2 * p], c * 8));
                        let w = _mm512_castpd_ps(_mm512_insertf64x4::<1>(
                            _mm512_castps_pd(lo),
                            _mm256_castps_pd(load8(ws[2 * p + 1], c * 8)),
                        ));
                        for r in 0..R {
                            l[r][p] = _mm512_add_ps(l[r][p], _mm512_mul_ps(xc[r], w));
                        }
                    }
                    if O % 2 == 1 {
                        let w = load8(ws[O - 1], c * 8);
                        for r in 0..R {
                            odd[r] = _mm256_add_ps(odd[r], _mm256_mul_ps(_mm512_castps512_ps256(xc[r]), w));
                        }
                    }
                }
                for r in 0..R {
                    for p in 0..pairs {
                        let mut a = [0.0f32; 16];
                        _mm512_storeu_ps(a.as_mut_ptr(), l[r][p]);
                        let (lo, hi) = a.split_at(8);
                        out[r][2 * p] = finish(lo.try_into().unwrap(), xs[r], ws[2 * p]);
                        out[r][2 * p + 1] = finish(hi.try_into().unwrap(), xs[r], ws[2 * p + 1]);
                    }
                    if O % 2 == 1 {
                        out[r][O - 1] = finish(lanes8(odd[r]), xs[r], ws[O - 1]);
                    }
                }
            }
            out
        }
    }
}
