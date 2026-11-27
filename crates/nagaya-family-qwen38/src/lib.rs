//! The text rules of the Qwen3.8 family (spec 11.3): tokenizer, chat template, detokenizer and reply parser.
//!
//! Every rule comes from the GGUF metadata of the model. The tests compare each result with the output of the
//! reference's Python front-end, byte for byte and id for id.

pub mod council;
pub mod detok;
mod error;
pub mod family;
pub mod gguf;
pub mod hash;
pub mod reply;
pub mod template;
pub mod tokenizer;

pub use nagaya_jinja::{chars, json, pystr};

pub use council::Qwen38Family;
pub use error::Error;
pub use family::{Qwen38, Rendered};
pub use hash::tokenizer_hash;
pub use template::{ChatTemplate, RenderOptions};
pub use tokenizer::Tokenizer;
