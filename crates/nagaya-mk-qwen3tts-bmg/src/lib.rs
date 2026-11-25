//! The Qwen3-TTS 12 Hz 1.7B VoiceDesign engine on one B70 card. It runs the talker, the code predictor
//! and the codec decoder as rings of tasks on the shared interpreter of `nagaya-mk-bmg` (engine-abi "Speech").
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
pub use engine::{Forced, KV_ROW_BYTES, Qwen3TtsEngine};
pub use plan::plan;
pub use spec::Config;
