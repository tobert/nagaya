//! The LFM3 engine on one B70 card (milestone 24, spec 11.11, engine-abi "LFM3"). One engine kind serves `LFM3-1.7B`
//! and the language model of `LFM3-Omni-2B`, which have the same backbone.
//!
//! The backbone runs as one ring of tasks for each pass on the shared interpreter (`nagaya-mk-bmg`), with the LFM3
//! binary. `spec` reads `config.json` and holds the limits and the rules of the slots, which need no card. `weights`
//! checks the tensors of the backbone and uploads them, and `model` builds the ring of a pass.
//!
//! With no feature, the crate plans with no card. The feature `xe` loads the engine on a card.

pub mod audio;
pub mod codec;
pub mod depth;
#[cfg(feature = "xe")]
pub mod device;
#[cfg(feature = "xe")]
pub mod engine;
pub mod frames;
pub mod model;
pub mod spec;
pub mod weights;

#[cfg(feature = "xe")]
pub use engine::Lfm3Engine;
pub use spec::{Backbone, Config, plan};
