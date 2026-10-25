//! The family library of Qwen3-TTS 12Hz VoiceDesign (spec 11.3, 11.8 "The speech family").
//!
//! The library turns a speech request into the prompt of the speech engine. A prompt is a list of positions, each a
//! pair of a text id and a codec id (engine-abi "Speech"). The rules come from the reference
//! `Qwen3TTSModel.generate_voice_design` with `non_streaming_mode=True`. The family has no detokenizer and no reply
//! parser, because the output is audio.

mod error;
mod family;
mod ids;
mod prompt;
mod tokenizer;

pub use error::Error;
pub use family::Qwen3TtsFamily;
pub use ids::{AUTO, CodecIds, SpeechIds, TextIds};
pub use prompt::PromptPosition;
