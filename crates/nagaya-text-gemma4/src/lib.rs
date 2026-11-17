//! The text rules of the Gemma 4 family (spec 11.3): tokenizer, chat template, escapes, council rendering and reply
//! parser.
//!
//! This is the port of the text code of the first Gemma 4 engine: the tokenizer, the chat, the template and
//! the reply splitter. The tests compare each result with fixed outputs of Hugging Face's Python.

/// Google's chat template for Gemma 4 (`google/gemma-4-26B-A4B-it`), as the first Gemma 4 engine held it. The
/// identity's `template` is this text (spec 11.3).
pub const CHAT_TEMPLATE: &str = include_str!("../assets/gemma4_chat_template.jinja");

pub mod call;
pub mod chat;
pub mod council;
mod error;
pub mod tokenizer;

pub use council::Gemma4Family;
pub use error::Error;
pub use tokenizer::Tokenizer;
