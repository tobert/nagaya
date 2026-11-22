//! The Inkling 2 engine on one NVIDIA GB10 (`sm_121`) GPU.
//!
//! `quant` holds the host quantization oracle for native NVFP4 (`fouroversix_mae`) and MXFP8 (`ue8m0`) activations
//! and block-scaled dot products. `spec` holds the config, model validation, long-context slot sizing, and `Limits`.
//! None of these needs a GPU.

pub mod cache;
pub mod quant;
pub mod spec;
