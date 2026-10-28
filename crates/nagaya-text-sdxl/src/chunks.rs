//! The chunks of a prompt (spec 11.9a): 75 tokens between a start token and an end token, at most 3 chunks.
//!
//! Both encoders take the same ids. The first pads with the end id, and the second pads with 0, as A1111 does.

use crate::{Error, Piece};

/// The tokens of a prompt in one chunk.
pub const CHUNK_TOKENS: usize = 75;
/// The ids of one chunk: the start token, the tokens and the end token, padded.
pub const CHUNK_IDS: usize = CHUNK_TOKENS + 2;
/// The most chunks of one prompt.
pub const MAX_CHUNKS: usize = 3;
/// The start token `<|startoftext|>`.
pub const BOS: u32 = 49406;
/// The end token `<|endoftext|>`. The first encoder pads with it.
pub const EOS: u32 = 49407;
/// The pad id of the second encoder.
pub const PAD2: u32 = 0;

/// One chunk: the ids of each encoder and the weight of each id.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub ids1: [u32; CHUNK_IDS],
    pub ids2: [u32; CHUNK_IDS],
    /// The weight of each id. The start, the end and the pad ids have weight 1.
    pub weights: [f32; CHUNK_IDS],
}

/// The chunks of parsed pieces. `tokenize` gives the ids of a text with no start or end token. Each piece of text is
/// tokenized alone, so a weight changes at a token boundary.
pub fn chunks(pieces: &[Piece], tokenize: impl Fn(&str) -> Result<Vec<u32>, Error>) -> Result<Vec<Chunk>, Error> {
    let mut done: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut current: Vec<(u32, f32)> = Vec::new();
    let mut tokens = 0;
    for piece in pieces {
        match piece {
            Piece::Text { text, weight } => {
                for id in tokenize(text)? {
                    if current.len() == CHUNK_TOKENS {
                        done.push(std::mem::take(&mut current));
                    }
                    current.push((id, *weight));
                    tokens += 1;
                }
            }
            Piece::Break => {
                if !current.is_empty() {
                    done.push(std::mem::take(&mut current));
                }
            }
        }
    }
    if !current.is_empty() || done.is_empty() {
        done.push(current);
    }
    if done.len() > MAX_CHUNKS {
        return Err(Error::TooLong {
            tokens,
            chunks: done.len(),
        });
    }
    Ok(done.iter().map(|tokens| chunk(tokens)).collect())
}

fn chunk(tokens: &[(u32, f32)]) -> Chunk {
    let mut out = Chunk {
        ids1: [EOS; CHUNK_IDS],
        ids2: [PAD2; CHUNK_IDS],
        weights: [1.0; CHUNK_IDS],
    };
    out.ids1[0] = BOS;
    out.ids2[0] = BOS;
    for (i, (id, weight)) in tokens.iter().enumerate() {
        out.ids1[i + 1] = *id;
        out.ids2[i + 1] = *id;
        out.weights[i + 1] = *weight;
    }
    out.ids2[tokens.len() + 1] = EOS;
    out
}
