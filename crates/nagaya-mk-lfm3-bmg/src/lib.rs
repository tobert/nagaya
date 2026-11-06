//! The LFM3 engine on one B70 card (milestone 24, spec 11.11, engine-abi "LFM3"). One engine kind serves `LFM3-1.7B`
//! and the language model of `LFM3-Omni-2B`, which have the same backbone.
//!
//! The backbone runs as one ring of tasks for each pass on the shared interpreter (`nagaya-mk-bmg`), with the LFM3
//! binary. `spec` reads `config.json` and holds the limits and the rules of the slots, which need no card. `weights`
//! checks the tensors of the backbone and uploads them, and `model` builds the ring of a pass.
//!
//! With no feature, the crate plans with no card.

pub mod model;
pub mod spec;
pub mod weights;

pub use spec::{Backbone, Config, plan};
