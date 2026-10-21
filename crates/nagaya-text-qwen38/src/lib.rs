//! The text rules of the Qwen3.8 family (spec 11.3): tokenizer, chat template, detokenizer and reply parser.
//!
//! Every rule comes from the GGUF metadata of the model. The tests compare each result with the output of mk's
//! Python front-end, byte for byte and id for id.

pub mod chars;
pub mod council;
pub mod detok;
mod error;
pub mod family;
pub mod hash;
mod methods;
pub mod template;
pub mod tokenizer;
#[rustfmt::skip]
mod unicode_tables;
pub mod gguf;
pub mod json;
pub mod pystr;
pub mod reply;

pub use council::Qwen38Family;
pub use error::Error;
pub use family::{Qwen38, Rendered};
pub use hash::tokenizer_hash;
pub use template::{ChatTemplate, RenderOptions};
pub use tokenizer::Tokenizer;
