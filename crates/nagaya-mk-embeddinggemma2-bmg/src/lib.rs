//! The EmbeddingGemma 2 engine on one B70 card (spec 11.10, engine-abi "Embedding").
//!
//! The text tower runs on the card as one ring of tasks on the shared interpreter (`nagaya-mk-bmg`). The host takes the
//! mean of the last output over the positions. The audio and vision towers of the checkpoint are not loaded.
//!
//! With no feature, the crate plans with no card. The feature `xe` loads the engine on a card.

#[cfg(feature = "xe")]
pub mod device;
#[cfg(feature = "xe")]
pub mod engine;
pub mod model;
pub mod plan;
pub mod spec;
pub mod weights;

#[cfg(feature = "xe")]
pub use engine::EmbeddingGemma2Engine;
pub use plan::plan;
pub use spec::Config;
