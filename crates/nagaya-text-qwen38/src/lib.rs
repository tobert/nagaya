//! The text rules of the Qwen3.8 family (spec 11.3): the tokenizer, the chat template and the detokenizer.
//!
//! Every rule comes from the GGUF metadata of the model. The tests compare each result with the output of mk's
//! Python front-end, byte for byte and id for id.

mod error;
pub mod gguf;
pub mod json;

pub use error::Error;
