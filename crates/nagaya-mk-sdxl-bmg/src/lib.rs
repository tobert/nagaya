//! The SDXL engine on one B70 card (spec 11.9, engine-abi "Image").
//!
//! The text encoders, the UNet and the VAE run on the card, through the shared interpreter (`nagaya-mk-bmg`). The
//! small work of each step runs on the host: the guidance, the sampler, its noise, the token weights and the previews.
//! The latents of 4 images of 1.5 MP are a few MB, so a step reads and writes them cheaply.

#[cfg(feature = "xe")]
pub mod device;
#[cfg(feature = "xe")]
pub mod engine;
pub mod host;
pub mod job;
pub mod lora;
pub mod model;
pub mod noise;
pub mod plan;
pub mod sampler;
pub mod single;
pub mod spec;
pub mod weights;

#[cfg(feature = "xe")]
pub use engine::SdxlEngine;
pub use plan::plan;
pub use spec::Config;
