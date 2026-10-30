//! Helpers that the integration tests share.
//!
//! Amy, 2026-10-21: "adjust the tests so they don't break our tests when we don't have a model downloaded locally.
//! we won't replicate all models to all machines so this state is normal." An absent checkpoint skips the test. The
//! variable `EMBEDDINGGEMMA2_MODEL_DIR` names the directory, and the code holds no default path (AGENTS.md). The
//! variable `NAGAYA_REQUIRE_MODELS=1` makes an absent checkpoint or oracle fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use nagaya_text_embeddinggemma2::EmbeddingGemma2Family;

pub const DIR_VARIABLE: &str = "EMBEDDINGGEMMA2_MODEL_DIR";
pub const GOLDEN_VARIABLE: &str = "EMBEDDINGGEMMA2_GOLDEN_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

fn require() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// A fixture file of `tests/fixtures`, as JSON.
pub fn fixture(name: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// A JSON array of ids.
pub fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|i| u32::try_from(i.as_u64().expect("an id")).expect("an id fits 32 bits"))
        .collect()
}

/// True when the checkpoint is absent and the test must return. It prints one line to stderr.
/// Call it first in each test that needs the checkpoint.
pub fn absent() -> bool {
    let test = std::thread::current().name().unwrap_or("a test").to_owned();
    let skip = |what: String| {
        if require() {
            panic!("{what}. {REQUIRE_VARIABLE} is set. Set {DIR_VARIABLE} to the EmbeddingGemma 2 snapshot directory.");
        }
        eprintln!("skipped: {test}: {what}; set {DIR_VARIABLE}");
        true
    };
    let Some(dir) = std::env::var_os(DIR_VARIABLE).map(PathBuf::from) else {
        return skip(format!("{DIR_VARIABLE} is not set"));
    };
    let file = dir.join("tokenizer.json");
    match std::fs::metadata(&file) {
        Ok(m) if m.is_file() => false,
        Ok(_) => panic!(
            "{} is not a file. Set {DIR_VARIABLE} to a snapshot directory.",
            file.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            skip(format!("the tokenizer file {} does not exist", file.display()))
        }
        Err(e) => panic!("cannot read the tokenizer file {}: {e}", file.display()),
    }
}

/// The checkpoint directory. Call `absent` first.
pub fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var_os(DIR_VARIABLE).expect("the test must call `absent` first"))
}

/// The family of the checkpoint, built once for each test binary. Call `absent` first.
pub fn family() -> &'static EmbeddingGemma2Family {
    static FAMILY: std::sync::OnceLock<EmbeddingGemma2Family> = std::sync::OnceLock::new();
    FAMILY.get_or_init(|| {
        EmbeddingGemma2Family::from_dir(&model_dir()).unwrap_or_else(|e| panic!("the checkpoint builds no family: {e}"))
    })
}

/// The directory of the oracle: `EMBEDDINGGEMMA2_GOLDEN_DIR`, or `target/embeddinggemma2-golden` of the workspace.
pub fn golden_dir() -> PathBuf {
    std::env::var_os(GOLDEN_VARIABLE)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/embeddinggemma2-golden"))
}

/// The path of an oracle file, or `None` when it is absent and absence may skip.
pub fn golden(file: &str) -> Option<PathBuf> {
    let path = golden_dir().join(file);
    if !path.is_file() {
        assert!(
            !require(),
            "{REQUIRE_VARIABLE} is set, and {} is absent; run tools/embeddinggemma2_golden.sh",
            path.display()
        );
        eprintln!(
            "skipped: {} is absent; run tools/embeddinggemma2_golden.sh",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// The i64 tensor `name` of a safetensors file, as ids.
pub fn safetensors_ids(path: &Path, name: &str) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let entry = &header[name];
    assert_eq!(entry["dtype"], "I64", "{name} of {} is not I64", path.display());
    let start = 8 + len + entry["data_offsets"][0].as_u64().unwrap() as usize;
    let end = 8 + len + entry["data_offsets"][1].as_u64().unwrap() as usize;
    let (values, rest) = bytes[start..end].as_chunks::<8>();
    assert!(rest.is_empty(), "{name} of {} is not whole i64 values", path.display());
    values
        .iter()
        .map(|b| u32::try_from(i64::from_le_bytes(*b)).expect("an id fits 32 bits"))
        .collect()
}
