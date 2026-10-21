//! Helpers that the integration tests share.
#![allow(dead_code)]

use nagaya_text_qwen38::json::Json;
use std::path::PathBuf;

pub mod model;
#[allow(unused_imports)]
pub use model::{absent, gguf_path};

/// A fixture file, read with the library's own JSON reader.
pub fn fixture(name: &str) -> Json {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    Json::parse(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// Short accessors that panic with the value when its shape is wrong.
pub trait JsonExt {
    fn get(&self, key: &str) -> &Json;
    fn has(&self, key: &str) -> bool;
    fn str(&self) -> &str;
    fn items(&self) -> &[Json];
    fn int(&self) -> i128;
    fn ids(&self) -> Vec<u32>;
}

impl JsonExt for Json {
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Object(items) => items
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("no key {key:?}")),
            other => panic!("not an object: {other:?}"),
        }
    }
    fn has(&self, key: &str) -> bool {
        matches!(self, Json::Object(items) if items.iter().any(|(k, _)| k == key))
    }
    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            other => panic!("not a string: {other:?}"),
        }
    }
    fn items(&self) -> &[Json] {
        match self {
            Json::Array(a) => a,
            other => panic!("not an array: {other:?}"),
        }
    }
    fn int(&self) -> i128 {
        match self {
            Json::Int(i) => *i,
            other => panic!("not an integer: {other:?}"),
        }
    }
    fn ids(&self) -> Vec<u32> {
        self.items()
            .iter()
            .map(|i| u32::try_from(i.int()).expect("id"))
            .collect()
    }
}

/// The tokenizer of the model file, built once for each test binary.
pub fn tokenizer() -> &'static nagaya_text_qwen38::Tokenizer {
    static TOKENIZER: std::sync::OnceLock<nagaya_text_qwen38::Tokenizer> = std::sync::OnceLock::new();
    TOKENIZER.get_or_init(|| {
        nagaya_text_qwen38::Tokenizer::from_gguf(&gguf_path())
            .unwrap_or_else(|e| panic!("cannot build the tokenizer: {e}"))
    })
}

fn gguf_string(s: &str) -> Vec<u8> {
    let mut out = (s.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(s.as_bytes());
    out
}

fn gguf_entry(key: &str, value_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = gguf_string(key);
    out.extend_from_slice(&value_type.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// GGUF metadata with the tokenizer keys, built in memory. `tokens` are (text, token type) pairs.
pub fn small_metadata(
    pre: &str,
    model: &str,
    tokens: &[(&str, i32)],
    types: Option<&[i32]>,
    merges: &[&str],
) -> nagaya_text_qwen38::gguf::Metadata {
    let types: Vec<i32> = types
        .map(|t| t.to_vec())
        .unwrap_or_else(|| tokens.iter().map(|t| t.1).collect());
    let values: Vec<Vec<u8>> = types.iter().map(|t| t.to_le_bytes().to_vec()).collect();
    let texts: Vec<&str> = tokens.iter().map(|t| t.0).collect();
    small_metadata_with_types(pre, model, &texts, 5, &values, merges)
}

/// Like `small_metadata`, but the array `tokenizer.ggml.token_type` has the GGUF element type `kind_type`, and
/// `kind_values` holds the bytes of each element.
pub fn small_metadata_with_types(
    pre: &str,
    model: &str,
    tokens: &[&str],
    kind_type: u32,
    kind_values: &[Vec<u8>],
    merges: &[&str],
) -> nagaya_text_qwen38::gguf::Metadata {
    let mut texts = 8u32.to_le_bytes().to_vec();
    texts.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        texts.extend(gguf_string(t));
    }
    let mut kinds = kind_type.to_le_bytes().to_vec();
    kinds.extend_from_slice(&(kind_values.len() as u64).to_le_bytes());
    for v in kind_values {
        kinds.extend_from_slice(v);
    }
    let mut merge_list = 8u32.to_le_bytes().to_vec();
    merge_list.extend_from_slice(&(merges.len() as u64).to_le_bytes());
    for m in merges {
        merge_list.extend(gguf_string(m));
    }
    let mut file = b"GGUF".to_vec();
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&5u64.to_le_bytes());
    file.extend(gguf_entry("tokenizer.ggml.pre", 8, &gguf_string(pre)));
    file.extend(gguf_entry("tokenizer.ggml.model", 8, &gguf_string(model)));
    file.extend(gguf_entry("tokenizer.ggml.tokens", 9, &texts));
    file.extend(gguf_entry("tokenizer.ggml.token_type", 9, &kinds));
    file.extend(gguf_entry("tokenizer.ggml.merges", 9, &merge_list));
    nagaya_text_qwen38::gguf::Metadata::read(file.as_slice(), file.len() as u64).expect("a built file")
}
