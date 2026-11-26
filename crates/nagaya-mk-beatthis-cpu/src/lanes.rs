//! `F4`: four f32 values in one register, for the kernels. Each operation is one IEEE operation on each value, with
//! the rounding of the same scalar operation, so a kernel that uses `F4` gives the bits of the scalar form. There is
//! no fused multiply-add.
//!
//! On x86_64 it is an SSE register (SSE2 is in each x86_64 CPU). On other machines it is an array.

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::{__m128, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps};

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
        pub fn mul(self, o: Self) -> Self {
            // SAFETY: SSE is in each x86_64 CPU.
            F4(unsafe { _mm_mul_ps(self.0, o.0) })
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
        pub fn mul(self, o: Self) -> Self {
            F4(std::array::from_fn(|k| self.0[k] * o.0[k]))
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
