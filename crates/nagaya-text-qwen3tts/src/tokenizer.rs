//! The tokenizer of the text vocabulary, built from `vocab.json`, `merges.txt` and `tokenizer_config.json`.
//!
//! It is the tokenizer that the reference processor runs. That processor is `Qwen2TokenizerFast`, loaded with
//! `fix_mistral_regex=True` by transformers 4.57.3. Thus the split pattern is Mistral's, not Qwen2's: the checkpoint
//! names transformers 4.57.3, and that version applies the fix to it. The pipeline is NFC, the split, the byte-level
//! map, and byte-level BPE. The added tokens of `tokenizer_config.json` match as whole tokens before all of it.

use crate::Error;
use serde_json::Value;
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior};

/// The split pattern of the reference processor (transformers 4.57.3, `fix_mistral_regex`).
const SPLIT_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub(crate) struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

fn tokenizer_error(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(format!("{what}: {e}"))
}

fn flag(entry: &Value, id: &str, key: &str) -> Result<bool, Error> {
    entry[key].as_bool().ok_or_else(|| {
        Error::Checkpoint(format!(
            "tokenizer_config.json: added_tokens_decoder.{id}.{key} is not true or false"
        ))
    })
}

/// The added tokens of `tokenizer_config.json`, in order of id.
fn added_tokens(text: &str) -> Result<Vec<(u32, AddedToken)>, Error> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| Error::Checkpoint(format!("tokenizer_config.json is not JSON ({e})")))?;
    let decoder = v["added_tokens_decoder"]
        .as_object()
        .ok_or_else(|| Error::Checkpoint("tokenizer_config.json has no object added_tokens_decoder".into()))?;
    let mut out = Vec::with_capacity(decoder.len());
    for (key, entry) in decoder {
        let id: u32 = key.parse().map_err(|_| {
            Error::Checkpoint(format!(
                "tokenizer_config.json: the added token key {key:?} is not an id"
            ))
        })?;
        let content = entry["content"].as_str().ok_or_else(|| {
            Error::Checkpoint(format!(
                "tokenizer_config.json: added_tokens_decoder.{key}.content is not text"
            ))
        })?;
        let token = AddedToken::from(content, flag(entry, key, "special")?)
            .single_word(flag(entry, key, "single_word")?)
            .lstrip(flag(entry, key, "lstrip")?)
            .rstrip(flag(entry, key, "rstrip")?)
            .normalized(flag(entry, key, "normalized")?);
        out.push((id, token));
    }
    out.sort_by_key(|(id, _)| *id);
    Ok(out)
}

fn read(dir: &Path, name: &str) -> Result<String, Error> {
    let path = dir.join(name);
    std::fs::read_to_string(&path).map_err(|source| Error::Io {
        path: path.display().to_string(),
        source,
    })
}

impl Tokenizer {
    /// Build the tokenizer from the files of the checkpoint directory `dir`.
    pub(crate) fn from_dir(dir: &Path) -> Result<Tokenizer, Error> {
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        for path in [&vocab, &merges] {
            std::fs::metadata(path).map_err(|source| Error::Io {
                path: path.display().to_string(),
                source,
            })?;
        }
        let path_text = |p: &Path| {
            p.to_str()
                .map(str::to_string)
                .ok_or_else(|| Error::Checkpoint(format!("the path {} is not UTF-8", p.display())))
        };
        let bpe = BPE::from_file(&path_text(&vocab)?, &path_text(&merges)?)
            .build()
            .map_err(|e| tokenizer_error("vocab.json and merges.txt make no BPE model", e))?;
        let mut inner = tokenizers::Tokenizer::new(bpe);
        inner
            .with_normalizer(Some(NFC))
            .map_err(|e| tokenizer_error("the NFC normalizer", e))?;
        let split = Split::new(
            SplitPattern::Regex(SPLIT_PATTERN.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| tokenizer_error("the split pattern", e))?;
        inner.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, true, false).into(),
        ])));
        for (id, token) in added_tokens(&read(dir, "tokenizer_config.json")?)? {
            let content = token.content.clone();
            let added = if token.special {
                inner.add_special_tokens([token])
            } else {
                inner.add_tokens([token])
            };
            added.map_err(|e| tokenizer_error("an added token", e))?;
            // The library gives an added token the next free id. The id of the file must be that id.
            let given = inner.token_to_id(&content);
            if given != Some(id) {
                return Err(Error::Checkpoint(format!(
                    "tokenizer_config.json: the added token {content:?} has the id {id}, but the tokenizer gives it \
                     {given:?}"
                )));
            }
        }
        Ok(Tokenizer { inner })
    }

    /// The id of the token with this exact text, from the vocabulary or the added tokens.
    pub(crate) fn token_id(&self, text: &str) -> Option<u32> {
        self.inner.token_to_id(text)
    }

    /// The token ids of `text`. The text of an added token becomes that token.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| tokenizer_error("the encode", e))?;
        Ok(encoding.get_ids().to_vec())
    }
}
