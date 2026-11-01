//! The Qwen-Image 2.1 engine on gfx1151 (`AGENTS.md` milestone 22, spec 11.9c): the text tower of Qwen3-VL, the
//! transformer and the VAE decoder, on the shared ops of `nagaya-mk-hip`.
//!
//! The file reader and the tensor names are plain Rust and run with no GPU. The parts that run on the device are
//! behind the feature `hip`.

pub mod names;
pub mod safetensors;
pub mod sampler;

#[cfg(feature = "hip")]
pub mod encoder;
#[cfg(feature = "hip")]
pub mod transformer;
#[cfg(feature = "hip")]
pub mod vae;
