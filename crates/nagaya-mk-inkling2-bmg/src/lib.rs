//! The Inkling 2 engine on one B70 card.
//!
//! This crate holds the host ops so far: the weight formats in `formats`, the `f32` host form of the text model in
//! `host`, the oracle of each device op, and the layout of the expert planes on the card in `repack`.
//!
//! The experts of Inkling 2 Mini come in two formats. Layers 0 to 19 hold NVFP4: two E2M1 codes in a byte, one E4M3 scale for each 16 values, and one factor for each
//! expert. Layers 20 to 23 hold MXFP8: E4M3 values with one UE8M0 scale for each 32 values. The reference expands
//! both formats to `bf16` at load (`Inkling2LoopDequantizeExperts` in transformers, commit `046bdb8cc2`). The ops
//! in `formats` give the same bits. A kernel of the card checks itself against them.

pub mod formats;
pub mod host;
pub mod repack;
