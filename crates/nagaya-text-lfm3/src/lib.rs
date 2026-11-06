//! The text rules of the LFM3 family (spec 11.3, 11.11): the tokenizer, the chat template, the escapes, council
//! rendering, the media positions, the media preprocessing and the reply parser of `LFM3-Omni-2B` and `LFM3-1.7B`.
//!
//! One library serves both models. It reads `tokenizer.json`, `chat_template.jinja`, `generation_config.json` and
//! `config.json` of the model folder. The tests compare each result with fixed outputs of Hugging Face's Python
//! (`tools/lfm3_family_fixture.py`).

pub mod call;
pub mod chat;
pub mod council;
mod error;
pub mod media;
pub mod tokenizer;

pub use council::{Files, Kind, Lfm3Family, MediaKind, MediaPart, MediaRendering, MediaSpan, Mode};
pub use error::Error;
pub use tokenizer::Tokenizer;
