//! The text rules of the Inkling 2 family (spec 11.3, 11.3a, milestone 35 step 3): the tokenizer, the chat
//! template, the escapes, council rendering, and the reply parser of `Inkling-2-Mini` and `TML-d24-varloop-research`.
//!
//! One library serves both models: they share identical tokenizer and template files. It reads `tokenizer.json`,
//! `chat_template.jinja`, `generation_config.json` and `config.json` of the model folder. The tests compare each
//! result with fixed outputs of Hugging Face's Python (`tools/inkling2_family_fixture.py`).
//!
//! **Notes on open choices:**
//! - `vocab_size()` returns `unpadded_vocab_size` (200,058) rather than `vocab_size` (201,024) because `lm_head` and the
//!   logit rows have width `unpadded_vocab_size` and the tokenizer holds 200,058 entries.
//! - Milestone 35 question 4 (`reasoning_effort`): a context with `thinking: false` renders the effort "none", and one
//!   with `thinking: true` the default of the template, 0.9. `chat::Options` keeps a typed `Effort` for a later number.

pub mod chat;
pub mod council;
mod error;
pub mod media;
pub mod tokenizer;

pub use chat::{Chat, Effort, Options, Rendered};
pub use council::{Files, ImagePart, Inkling2Family, Objects};
pub use error::Error;
pub use tokenizer::Tokenizer;
