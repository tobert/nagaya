//! The Inkling 2 tokenizer: the `tokenizer.json` of the model folder, read by Hugging Face's `tokenizers` crate.
//!
//! The file is a byte-level BPE with 200,058 entries (`unpadded_vocab_size`), including 60 added special tokens
//! (ids 199,998 through 200,057). Client text that spells a marker is content: the escapes swap the marker text for a
//! sentinel, and the sentinel tokenizes with no added token.

use crate::Error;
use sha2::{Digest, Sha256};
use std::path::Path;

/// The first and the last character of a sentinel. Between them is the index of the marker in `markers`.
const SENTINEL_OPEN: char = '\u{FDD0}';
const SENTINEL_CLOSE: char = '\u{FDD1}';

/// A Unicode noncharacter: U+FDD0 to U+FDEF, and the last two code points of each plane. There are 66.
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

/// `sha256:` and the hex digits of the SHA-256 of `bytes` (spec 12.3).
pub fn sha256_of(bytes: &[u8]) -> String {
    let hex: String = Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

/// The GPT-2 map from a byte to a printable character. The vocabulary of a byte-level BPE uses it.
fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let printable = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        *slot = if printable {
            char::from_u32(b as u32).expect("a byte is a scalar value")
        } else {
            let c = char::from_u32(next).expect("256 and up are scalar values");
            next += 1;
            c
        };
    }
    table
}

/// The tokenizer of one Inkling 2 model folder.
pub struct Tokenizer {
    /// The tokenizer of the file: text that spells an added token gives that token.
    inner: tokenizers::Tokenizer,
    /// The same tokenizer with no added token: text that spells a marker stays text.
    plain: tokenizers::Tokenizer,
    /// The text of each added token, longest first, so that a longer marker wins over its prefix.
    markers: Vec<String>,
    /// The first character of each marker.
    firsts: Vec<char>,
    /// The byte of each character of the byte-level map.
    char_byte: std::collections::HashMap<char, u8>,
    /// `sha256:` and the 64 hex digits of the SHA-256 of the file (spec 12.3).
    hash: String,
    count: usize,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("hash", &self.hash).finish()
    }
}

fn tokenizer_error(e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(e.to_string())
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
                "{}: {m}. Use the tokenizer.json of an Inkling 2 model.",
                path.display()
            )),
            other => other,
        })
    }

    /// Read a tokenizer from the bytes of a `tokenizer.json`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Tokenizer, Error> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes).map_err(tokenizer_error)?;
        let mut body: serde_json::Value = serde_json::from_slice(bytes).map_err(tokenizer_error)?;
        let Some(added) = body.get_mut("added_tokens") else {
            return Err(Error::Tokenizer("the file has no added_tokens".to_string()));
        };
        *added = serde_json::Value::Array(Vec::new());
        let plain = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&body).map_err(tokenizer_error)?)
            .map_err(tokenizer_error)?;
        let mut markers: Vec<String> = inner
            .get_added_tokens_decoder()
            .values()
            .map(|t| t.content.clone())
            .collect();
        if markers.iter().any(String::is_empty) {
            return Err(Error::Tokenizer("an added token has no text".to_string()));
        }
        if let Some(bad) = markers.iter().find(|m| m.chars().any(is_noncharacter)) {
            return Err(Error::Tokenizer(format!(
                "the added token {bad:?} holds a Unicode noncharacter, which the escapes keep"
            )));
        }
        markers.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let char_byte = byte_chars().iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let count = inner.get_vocab_size(true);
        let mut firsts: Vec<char> = markers.iter().filter_map(|m| m.chars().next()).collect();
        firsts.sort_unstable();
        firsts.dedup();
        Ok(Tokenizer {
            inner,
            plain,
            markers,
            firsts,
            char_byte,
            hash: sha256_of(bytes),
            count,
        })
    }

    /// The tokenizer hash of the identity (spec 12.1, 12.3).
    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// The number of tokens, the added tokens too.
    pub fn len(&self) -> usize {
        self.count
    }

    /// True when the vocabulary is empty. A real tokenizer never is.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The text of each added token, longest first.
    pub fn markers(&self) -> &[String] {
        &self.markers
    }

    /// The id of one exact vocabulary entry or added token, such as `<|message_model|>`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// The ids of template text: text that spells an added token gives that token. No token is added at the ends.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(tokenizer_error)
    }

    /// The ids of client text: text that spells an added token stays text.
    pub fn encode_content(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.plain
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(tokenizer_error)
    }

    /// Client text as escaped content. Each marker text becomes a sentinel, so no template rule and no tokenizer sees
    /// a marker in it. A noncharacter is an error.
    pub fn escape(&self, text: &str) -> Result<String, Error> {
        check_text(text)?;
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        'outer: while let Some(c) = rest.chars().next() {
            if self.firsts.contains(&c) {
                for (i, marker) in self.markers.iter().enumerate() {
                    if rest.starts_with(marker.as_str()) {
                        out.push(SENTINEL_OPEN);
                        out.push_str(&i.to_string());
                        out.push(SENTINEL_CLOSE);
                        rest = &rest[marker.len()..];
                        continue 'outer;
                    }
                }
            }
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
        Ok(out)
    }

    /// The parts of escaped text: plain text, or the index of an escaped marker.
    fn parts<'a>(&self, escaped: &'a str) -> Result<Vec<Part<'a>>, Error> {
        let bad = || Error::Template("the template changed a sentinel of the escapes".to_string());
        let mut parts = Vec::new();
        let mut rest = escaped;
        while !rest.is_empty() {
            let Some(at) = rest.find(SENTINEL_OPEN) else {
                parts.push(Part::Text(rest));
                break;
            };
            if at > 0 {
                parts.push(Part::Text(&rest[..at]));
            }
            let after = &rest[at + SENTINEL_OPEN.len_utf8()..];
            let close = after.find(SENTINEL_CLOSE).ok_or_else(bad)?;
            let index: usize = after[..close].parse().map_err(|_| bad())?;
            if index >= self.markers.len() {
                return Err(bad());
            }
            parts.push(Part::Marker(index));
            rest = &after[close + SENTINEL_CLOSE.len_utf8()..];
        }
        Ok(parts)
    }

    /// The ids of escaped text: its own marker text gives the marker tokens, and each sentinel gives the tokens of its
    /// marker as plain text.
    pub fn encode_escaped(&self, escaped: &str) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        for part in self.parts(escaped)? {
            match part {
                Part::Text(t) => ids.extend(self.encode(t)?),
                Part::Marker(i) => ids.extend(self.encode_content(&self.markers[i])?),
            }
        }
        Ok(ids)
    }

    /// Escaped text with each sentinel back as the text that it stands for.
    pub fn unescape(&self, escaped: &str) -> Result<String, Error> {
        let mut out = String::with_capacity(escaped.len());
        for part in self.parts(escaped)? {
            match part {
                Part::Text(t) => out.push_str(t),
                Part::Marker(i) => out.push_str(&self.markers[i]),
            }
        }
        Ok(out)
    }

    /// The bytes of one token alone. An added token is its text. Any other token is its text through the byte-level
    /// map. An id outside the vocabulary is an error.
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>, Error> {
        if id as usize >= self.count {
            return Err(Error::TokenId { id, count: self.count });
        }
        if let Some(added) = self.inner.get_added_tokens_decoder().get(&id) {
            return Ok(added.content.clone().into_bytes());
        }
        let Some(token) = self.inner.id_to_token(id) else {
            return Err(Error::Tokenizer(format!("the vocabulary has no text for token {id}")));
        };
        token
            .chars()
            .map(|c| {
                self.char_byte.get(&c).copied().ok_or_else(|| {
                    Error::Tokenizer(format!(
                        "token {id} holds {c:?}, which is not a character of the byte-level map"
                    ))
                })
            })
            .collect()
    }

    /// The text of `ids` when their bytes are whole UTF-8 text. Tokens that end or start inside a character are
    /// [`Error::Bytes`].
    pub fn decode_whole(&self, ids: &[u32]) -> Result<String, Error> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_bytes(id)?);
        }
        String::from_utf8(bytes).map_err(|e| {
            Error::Bytes(format!(
                "the tokens are not whole text: byte {} is inside a character",
                e.utf8_error().valid_up_to()
            ))
        })
    }

    /// The text of `ids`, with U+FFFD for bytes that are not whole characters.
    pub fn decode_lossy(&self, ids: &[u32]) -> Result<String, Error> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The markers that `text` spells, in order of their place in the text. A marker inside a longer marker does not
    /// count.
    pub fn control_hits(&self, text: &str) -> Vec<String> {
        let mut hits = Vec::new();
        let mut rest = text;
        'outer: while let Some(c) = rest.chars().next() {
            if self.firsts.contains(&c) {
                for marker in &self.markers {
                    if rest.starts_with(marker.as_str()) {
                        hits.push(marker.clone());
                        rest = &rest[marker.len()..];
                        continue 'outer;
                    }
                }
            }
            rest = &rest[c.len_utf8()..];
        }
        hits
    }
}

/// A part of escaped text.
enum Part<'a> {
    Text(&'a str),
    Marker(usize),
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;

    /// A byte-level BPE with an added token that holds a space. The byte-level map has no space character.
    const MINI: &str = r#"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{"id": 2, "content": "<a b>", "single_word": false, "lstrip": false, "rstrip": false,
                          "normalized": false, "special": false}],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": {"a": 0, "Ġ": 1}, "merges": []}
    }"#;

    #[test]
    fn an_added_token_is_its_own_text_and_not_byte_mapped() {
        let tok = Tokenizer::from_bytes(MINI.as_bytes()).unwrap();
        assert_eq!(tok.token_bytes(2).unwrap(), b"<a b>");
        assert_eq!(tok.token_bytes(1).unwrap(), b" ");
        assert_eq!(tok.decode_whole(&[0, 1, 2]).unwrap(), "a <a b>");
        assert!(tok.token_bytes(3).is_err());
    }
}
