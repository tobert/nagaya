//! The CLIP tokenizer, built from `vocab.json` and `merges.txt`.
//!
//! It is the `CLIPTokenizer` of transformers 5.21, which diffusers loads for SDXL. The pipeline is NFC, a run of white
//! space to one space, lowercase, the CLIP split, the byte-level map, and byte-level BPE with the suffix `</w>` at the
//! end of each word. The start and the end token match as whole tokens before all of it.

use std::path::Path;

use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::replace::Replace;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::normalizers::utils::{Lowercase, Sequence as NormalizerSequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior};

use crate::Error;

/// The split pattern of the reference tokenizer.
const SPLIT_PATTERN: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

const START: &str = "<|startoftext|>";
const END: &str = "<|endoftext|>";

pub(crate) struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

fn failed(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(format!("{what}: {e}"))
}

impl Tokenizer {
    /// Build the tokenizer from the folder `dir` that holds `vocab.json` and `merges.txt`.
    pub(crate) fn from_dir(dir: &Path) -> Result<Tokenizer, Error> {
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        for path in [&vocab, &merges] {
            std::fs::metadata(path).map_err(|source| Error::Io {
                path: path.display().to_string(),
                source,
            })?;
        }
        let model = BPE::from_file(&vocab.to_string_lossy(), &merges.to_string_lossy())
            .end_of_word_suffix("</w>".to_owned())
            .unk_token(END.to_owned())
            .build()
            .map_err(|e| failed("the BPE model of vocab.json and merges.txt", e))?;
        let mut inner = tokenizers::Tokenizer::new(model);
        let spaces = Replace::new(
            tokenizers::normalizers::replace::ReplacePattern::Regex(r"\s+".to_owned()),
            " ",
        )
        .map_err(|e| failed("the white space pattern", e))?;
        inner
            .with_normalizer(Some(NormalizerSequence::new(vec![
                NFC.into(),
                spaces.into(),
                Lowercase.into(),
            ])))
            .map_err(|e| failed("the normalizer", e))?;
        let split = Split::new(
            SplitPattern::Regex(SPLIT_PATTERN.to_owned()),
            SplitDelimiterBehavior::Removed,
            true,
        )
        .map_err(|e| failed("the split pattern", e))?;
        inner.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, true, true).into(),
        ])));
        inner
            .add_special_tokens(vec![
                AddedToken::from(START, true).normalized(true),
                AddedToken::from(END, true).normalized(true),
            ])
            .map_err(|e| failed("the start and the end token", e))?;
        Ok(Tokenizer { inner })
    }

    /// The ids of a text, with no start token and no end token.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self.inner.encode(text, false).map_err(|e| failed("encode", e))?;
        Ok(encoding.get_ids().to_vec())
    }
}
