//! The Inkling 2 engine on one NVIDIA GB10 (`sm_121`) GPU.
//!
//! `quant` holds the host quantization oracle for native NVFP4 (`fouroversix_mae`) and MXFP8 (`ue8m0`) activations
//! and block-scaled dot products. `spec` holds the config, model validation, long-context slot sizing, and `Limits`.
//! `held` holds the rules of the held state of a slot: the checks of a pass and of the part calls. None of these needs
//! a GPU.

pub mod cache;
pub mod held;
pub mod identity;
pub mod pass;
pub mod quant;
pub mod spec;
