//! Beat This! on gfx1151: the plan of `nagaya-family-beatthis` run as one HIP launch for each task (spec 11.19).
//!
//! The kernels are the tile functions of `kernels/beatthis_ops.h`, a source in a subset of C that the CUDA and OpenCL
//! ports share. This crate holds the HIP launchers and the host loop. Its bits are its own: the same identity and
//! input give the same bits on gfx1151, and the tests check them against the oracle with a tolerance.

#[cfg(feature = "hip")]
pub mod gpu;
