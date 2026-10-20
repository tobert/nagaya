//! The text rules of the Qwen3.8 family (spec 11.3): the tokenizer, the chat template and the detokenizer.
//!
//! Every rule comes from the GGUF metadata of the model. The tests compare each result with the output of mk's
//! Python front-end, byte for byte and id for id.

pub mod chars;
pub mod detok;
mod error;
pub mod family;
pub mod template;
pub mod tokenizer;
#[rustfmt::skip]
mod unicode_tables;
pub mod gguf;
pub mod json;

pub use error::Error;
pub use family::{Qwen38, Rendered};
pub use template::{ChatTemplate, RenderOptions};
pub use tokenizer::Tokenizer;
