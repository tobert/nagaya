//! The Qwen-Image 2.1 family library (spec 11.9c): the tokenizer, the template and the cut of the system ids.
//!
//! The family turns one prompt into the template ids that the engine encodes. The engine never takes text
//! (engine-abi D2). The ids are those of the reference processor in diffusers, which the fixture recorded (spec 8.3).

mod error;

use std::path::Path;

use sha2::{Digest, Sha256};

pub use error::Error;

/// The system message of the template. The engine cuts its ids from the encoder output (spec 11.9c).
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

/// The template text up to the prompt.
const BEFORE: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n";

/// The template text after the prompt.
const AFTER: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// The text-to-image template of the reference (`pipeline_qwenimage21.py:217-221`). `{}` is the prompt.
pub const TEMPLATE: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n\
{}<|im_end|>\n<|im_start|>assistant\n";

/// The count of system ids that the engine cuts from the encoder output (spec 11.9c).
pub const CUT: usize = 14;

/// The most ids of a template (spec 11.9c).
pub const MAX_TEMPLATE_IDS: usize = 1024;

/// The family of a Qwen-Image 2.1 checkpoint.
pub struct QwenImage21Family {
    /// The tokenizer. It reads the text of a special token as its special id, as the reference does.
    tokenizer: tokenizers::Tokenizer,
    /// The text of each special token. A prompt that holds one is refused (spec 11.9c).
    specials: Vec<String>,
    tokenizer_hash: String,
}

impl std::fmt::Debug for QwenImage21Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImage21Family")
            .field("tokenizer_hash", &self.tokenizer_hash)
            .field("specials", &self.specials.len())
            .finish()
    }
}

fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|source| Error::Io {
        path: path.display().to_string(),
        source,
    })
}

impl QwenImage21Family {
    /// Build the family from the base directory `dir`. It reads `processor/tokenizer.json`.
    pub fn from_dir(dir: &Path) -> Result<QwenImage21Family, Error> {
        QwenImage21Family::from_bytes(&read(&dir.join("processor/tokenizer.json"))?)
    }

    /// Build the family from the bytes of `tokenizer.json`. The cut of this tokenizer must be `CUT`.
    pub fn from_bytes(tokenizer_json: &[u8]) -> Result<QwenImage21Family, Error> {
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut specials: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|token| token.special)
            .map(|token| token.content.clone())
            .collect();
        specials.sort();
        if specials.is_empty() {
            return Err(Error::Tokenizer("tokenizer.json has no special token".to_owned()));
        }
        let family = QwenImage21Family {
            tokenizer,
            specials,
            tokenizer_hash: format!(
                "sha256:{}",
                Sha256::digest(tokenizer_json)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            ),
        };
        let cut = family.system_ids()?.len();
        if cut != CUT {
            return Err(Error::Tokenizer(format!(
                "its system message gives {cut} ids, and the cut of spec 11.9c is {CUT}"
            )));
        }
        Ok(family)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The ids of the template with `prompt` in it. An empty prompt is one space, as in the reference
    /// (`pipeline_qwenimage21.py:249`). A prompt with the text of a special token is refused, and so is a template of
    /// more than `MAX_TEMPLATE_IDS` ids.
    pub fn ids(&self, prompt: &str) -> Result<Vec<u32>, Error> {
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let first = self
            .specials
            .iter()
            .filter_map(|token| prompt.find(token.as_str()).map(|at| (at, token)))
            .min_by_key(|&(at, token)| (at, std::cmp::Reverse(token.len())));
        if let Some((_, token)) = first {
            return Err(Error::SpecialToken { token: token.clone() });
        }
        let ids = self.encode(&format!("{BEFORE}{prompt}{AFTER}"))?;
        if ids.len() > MAX_TEMPLATE_IDS {
            return Err(Error::TooLong {
                ids: ids.len(),
                max: MAX_TEMPLATE_IDS,
            });
        }
        Ok(ids)
    }

    /// The ids of the system message, which the cut removes from the encoder output.
    pub fn system_ids(&self) -> Result<Vec<u32>, Error> {
        self.encode(SYSTEM)
    }

    /// The tokenizer hash of the identity: `sha256:` and the hex digits of the SHA-256 of `tokenizer.json`.
    pub fn tokenizer_hash(&self) -> &str {
        &self.tokenizer_hash
    }

    /// The template of the identity (spec 11.9c): the template text.
    pub fn template(&self) -> &'static str {
        TEMPLATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parts_of_the_template_make_the_template() {
        assert_eq!(format!("{BEFORE}{{}}{AFTER}"), TEMPLATE);
        assert!(BEFORE.starts_with(SYSTEM));
    }
}
