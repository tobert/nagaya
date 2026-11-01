//! The shared HIP ops of gfx1151 (`AGENTS.md` milestone 22): the hipBLASLt matrix multiplies with their plan table,
//! and the HIP kernels of the Qwen-Image engine. `nagaya-mk-bmg` holds the ops of the B70 cards in the same way.
//!
//! The plan table and the bf16 rounding are plain Rust and run with no GPU. The binding of the HIP runtime, the
//! hipBLASLt shim and the kernels are behind the feature `hip`.

pub mod bf16;
pub mod plan;
pub mod rope;

#[cfg(feature = "hip")]
pub mod gemm;
#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "hip")]
pub mod ops;

/// The build facts of the static library (spec 12.1): its SHA-256, the compiler, the HIP version and the hipBLASLt
/// version. `None` in a build without `hip`.
pub fn build_facts() -> Option<BuildFacts> {
    #[cfg(feature = "hip")]
    {
        Some(BuildFacts {
            sha256: env!("NAGAYA_MK_HIP_BUILD_SHA256"),
            compiler: env!("NAGAYA_MK_HIP_BUILD_COMPILER"),
            hip: env!("NAGAYA_MK_HIP_BUILD_HIP"),
            hipblaslt: env!("NAGAYA_MK_HIP_BUILD_HIPBLASLT"),
        })
    }
    #[cfg(not(feature = "hip"))]
    {
        None
    }
}

/// The facts of one build of the static library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildFacts {
    pub sha256: &'static str,
    pub compiler: &'static str,
    pub hip: &'static str,
    pub hipblaslt: &'static str,
}
