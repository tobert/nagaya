//! The SDXL family library (spec 11.3, 11.9a): the CLIP tokenizer, the A1111 prompt, and its chunks.
//!
//! The family turns the text of a prompt into the ids and the token weights of each encoder. The engine never takes
//! text (engine-abi D2). It scales the encoder output by the weights.

mod chunks;
mod error;
pub mod inputs;
mod prompt;
mod tokenizer;

use std::path::Path;

pub use chunks::{BOS, CHUNK_IDS, CHUNK_TOKENS, Chunk, EOS, MAX_CHUNKS, PAD2, chunks};
pub use error::Error;
pub use prompt::{Piece, parse};

/// The family of an SDXL checkpoint.
pub struct SdxlFamily {
    tokenizer: tokenizer::Tokenizer,
}

/// The prompt of one encoder pass: its chunks, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub chunks: Vec<Chunk>,
}

impl SdxlFamily {
    /// Build the family from the checkpoint directory `dir`, the folder layout of diffusers. It reads
    /// `tokenizer/vocab.json` and `tokenizer/merges.txt`.
    pub fn from_dir(dir: &Path) -> Result<SdxlFamily, Error> {
        Ok(SdxlFamily {
            tokenizer: tokenizer::Tokenizer::from_dir(&dir.join("tokenizer"))?,
        })
    }

    /// The ids of a text, with no start token and no end token.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.tokenizer.encode(text)
    }

    /// The chunks of an A1111 prompt (spec 11.9a). A prompt of more than 3 chunks is an error.
    pub fn prompt(&self, text: &str) -> Result<Prompt, Error> {
        Ok(Prompt {
            chunks: chunks(&parse(text), |run| self.tokenize(run))?,
        })
    }
}
