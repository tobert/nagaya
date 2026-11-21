//! The Inkling 2 engine on one B70 card.
//!
//! The weight formats are in `formats`, the `f32` host form of the text model and the oracle of each device op in
//! `host`, and the layout of the expert planes on the card in `repack`. `spec` holds the config, the limits and the
//! rules of the slots. `weights` puts the weights into an upload sink, and `model` builds the ring of a pass. None of
//! these needs a card. The feature `xe` adds `device` and `engine`: the `Engine` trait on one card.
//!
//! The experts of Inkling 2 Mini come in two formats. Layers 0 to 19 hold NVFP4: two E2M1 codes in a byte, one E4M3 scale for each 16 values, and one factor for each
//! expert. Layers 20 to 23 hold MXFP8: E4M3 values with one UE8M0 scale for each 32 values. The reference expands
//! both formats to `bf16` at load (`Inkling2LoopDequantizeExperts` in transformers, commit `046bdb8cc2`). The ops
//! in `formats` give the same bits. A kernel of the card checks itself against them.

#[cfg(feature = "xe")]
pub mod device;
#[cfg(feature = "xe")]
pub mod engine;
pub mod formats;
pub mod host;
pub mod model;
pub mod repack;
pub mod spec;
pub mod weights;

#[cfg(feature = "xe")]
pub use engine::Inkling2Engine;
