//! Helpers that the integration tests share.
#![allow(dead_code)]

use nagaya_text_qwen38::json::Json;
use std::path::PathBuf;

pub const GGUF_VARIABLE: &str = "NAGAYA_QWEN38_GGUF";
pub const GGUF_DEFAULT: &str =
    "/tank/ml/models/llama.cpp/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";

/// The path of the model file. A missing file fails the test: the tests never skip.
pub fn gguf_path() -> PathBuf {
    let path = PathBuf::from(std::env::var(GGUF_VARIABLE).unwrap_or_else(|_| GGUF_DEFAULT.to_string()));
    assert!(
        path.is_file(),
        "the Qwen3.8 GGUF file {} does not exist. Put the file there, or set {GGUF_VARIABLE} to its path.",
        path.display()
    );
    path
}

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
