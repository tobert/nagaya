//! The Gemma 4 tokenizer: the checkpoint's `tokenizer.json`, read by Hugging Face's `tokenizers` crate (mkviolet
//! `src/tokenizer.rs`).
//!
//! mkviolet reads EmbeddingGemma 2's file. The port reads the file of the Gemma 4 checkpoint (spec 12.3). The two files
//! differ only in the post-processor: EmbeddingGemma 2 adds `<bos>` and `<eos>`. The library never runs the
//! post-processor, so both files give the same ids.

use crate::Error;
use sha2::{Digest, Sha256};
use std::path::Path;

/// The id of `<bos>`. The template writes it first.
pub const BOS: &str = "<bos>";

/// A Unicode noncharacter: U+FDD0 to U+FDEF, and the last two code points of each plane.
pub fn is_noncharacter(c: char) -> bool {
    let cp = c as u32;
    (0xFDD0..=0xFDEF).contains(&cp) || cp & 0xFFFE == 0xFFFE
}

/// Refuse text that holds a noncharacter. The error names the character and its offset in characters.
pub fn check_text(text: &str) -> Result<(), Error> {
    match text.chars().enumerate().find(|(_, c)| is_noncharacter(*c)) {
        Some((offset, ch)) => Err(Error::Reserved { ch, offset }),
        None => Ok(()),
    }
}

/// The tokenizer of one Gemma 4 checkpoint.
pub struct Tokenizer {
    /// The tokenizer with special-token matching on: template text that spells `<|turn>` gives the token.
    inner: tokenizers::Tokenizer,
    /// The same tokenizer with special-token matching off: client text that spells `<|turn>` stays text.
    content: tokenizers::Tokenizer,
    /// The text of each special token, longest first, so that `<|tool_call>` wins over a shorter prefix.
    specials: Vec<String>,
    /// `sha256:` and the 64 hex digits of the SHA-256 of the file (spec 12.3).
    hash: String,
    vocab: usize,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("hash", &self.hash).finish()
    }
}

/// `sha256:` and the hex digits of the SHA-256 of `bytes`.
pub fn sha256_of(bytes: &[u8]) -> String {
    let hex: String = Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

impl Tokenizer {
    /// Read the tokenizer file at `path`.
    pub fn from_file(path: &Path) -> Result<Tokenizer, Error> {
        let bytes = std::fs::read(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        Tokenizer::from_bytes(&bytes).map_err(|e| match e {
            Error::Tokenizer(m) => Error::Tokenizer(format!(
                "{}: {m}. Use the tokenizer.json of a Gemma 4 checkpoint.",
                path.display()
            )),
            other => other,
        })
    }

    /// Read a tokenizer from the bytes of a `tokenizer.json`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Tokenizer, Error> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut content = inner.clone();
        content.set_encode_special_tokens(true);
        let mut specials: Vec<String> = inner
            .get_added_tokens_decoder()
            .values()
            .filter(|t| t.special)
            .map(|t| t.content.clone())
            .collect();
        specials.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        if specials.iter().any(String::is_empty) {
            return Err(Error::Tokenizer("a special token has no text".to_string()));
        }
        let vocab = inner.get_vocab_size(true);
        Ok(Tokenizer {
            inner,
            content,
            specials,
            hash: sha256_of(bytes),
            vocab,
        })
    }

    /// The tokenizer hash of the identity (spec 12.1, 12.3).
    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// The number of tokens in the vocabulary, the special tokens too.
    pub fn len(&self) -> usize {
        self.vocab
    }

    /// True when the vocabulary is empty. A real tokenizer never is.
    pub fn is_empty(&self) -> bool {
        self.vocab == 0
    }

    /// The text of each special token, longest first.
    pub fn specials(&self) -> &[String] {
        &self.specials
    }

    /// The ids of template text: text that spells a special token gives that token. No `<bos>` or `<eos>` is added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// The ids of client text: text that spells a special token stays text.
    pub fn encode_content(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.content
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// The text of `ids`, special tokens included. An id outside the vocabulary is an error.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(Error::TokenId { id, count: self.vocab });
        }
        self.inner
            .decode(ids, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// The id of one exact vocabulary entry, such as `<|turn>` or `▁A`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// The special tokens that `text` spells, in order of their place in the text.
    pub fn control_hits(&self, text: &str) -> Vec<String> {
        let mut hits: Vec<(usize, String)> = Vec::new();
        for special in &self.specials {
            for (at, _) in text.match_indices(special.as_str()) {
                hits.push((at, special.clone()));
            }
        }
        hits.sort();
        hits.into_iter().map(|(_, t)| t).collect()
    }
}
