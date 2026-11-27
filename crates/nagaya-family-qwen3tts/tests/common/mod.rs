//! Helpers that the integration tests share.
#![allow(dead_code)]

use std::path::PathBuf;

pub mod model;
#[allow(unused_imports)]
pub use model::{absent, family, model_dir};

/// A fixture file of `tests/fixtures`, as JSON.
pub fn fixture(name: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// A JSON array of ids, or a JSON id that can be null.
pub fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|i| id(i).unwrap_or_else(|| panic!("a null in a list of ids: {v}")))
        .collect()
}

pub fn id(v: &serde_json::Value) -> Option<u32> {
    if v.is_null() {
        return None;
    }
    let n = v.as_u64().unwrap_or_else(|| panic!("not an id: {v}"));
    Some(u32::try_from(n).expect("an id fits 32 bits"))
}

/// The `config.json` of the checkpoint, cut to the keys that this family reads. The tests change it to check refusals.
pub const SMALL_CONFIG: &str = r#"{
  "assistant_token_id": 77091,
  "im_end_token_id": 151645,
  "im_start_token_id": 151644,
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "model_type": "qwen3_tts",
  "tts_model_type": "voice_design",
  "talker_config": {
    "codec_bos_id": 2149,
    "codec_eos_token_id": 2150,
    "codec_think_id": 2154,
    "codec_language_id": {"chinese": 2055, "english": 2050, "sichuan_dialect": 2062},
    "codec_nothink_id": 2155,
    "codec_pad_id": 2148,
    "codec_think_bos_id": 2156,
    "codec_think_eos_id": 2157,
    "vocab_size": 3072
  }
}"#;
