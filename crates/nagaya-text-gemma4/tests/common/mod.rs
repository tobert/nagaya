//! Helpers that the integration tests share.
#![allow(dead_code)]

use nagaya_jinja::Json;
use std::path::PathBuf;

pub mod model;

/// A fixture file, read with the interpreter's own JSON reader.
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
    fn ids(&self) -> Vec<u32> {
        self.items()
            .iter()
            .map(|i| match i {
                Json::Int(n) => u32::try_from(*n).expect("a token id fits 32 bits"),
                other => panic!("not a token id: {other:?}"),
            })
            .collect()
    }
}
