//! The prompt of a reply with audio of `lfm3-omni` is the prompt of the oracle (spec 11.11, "The marker of a mode";
//! `tools/lfm3_reference.py`, `run_speech`). The family renders the context with its generation prompt, and the
//! speech path of the opener adds the marker of the mode.
//!
//! It needs `LFM3_OMNI_MODEL_DIR` and the oracle file `speech.safetensors` in `target/lfm3-golden` or
//! `LFM3_GOLDEN_DIR`. An absent folder or file skips the test, and fails it with `NAGAYA_REQUIRE_MODELS=1`.

use std::path::PathBuf;

use gogi_proto::context::Context;
use nagaya_abi::Engine;
use serde_json::{Value, json};

fn required() -> bool {
    std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1")
}

fn skip(what: &str) -> Option<PathBuf> {
    if required() {
        panic!("{what} is absent and NAGAYA_REQUIRE_MODELS is set");
    }
    eprintln!("skipped: {what} is absent");
    None
}

fn omni() -> Option<PathBuf> {
    match std::env::var_os("LFM3_OMNI_MODEL_DIR") {
        Some(dir) => Some(PathBuf::from(dir)),
        None => skip("LFM3_OMNI_MODEL_DIR"),
    }
}

fn golden() -> Option<PathBuf> {
    let dir = std::env::var_os("LFM3_GOLDEN_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"),
        PathBuf::from,
    );
    if dir.join("speech.safetensors").exists() && dir.join("facts.json").exists() {
        Some(dir)
    } else {
        skip(&format!("{}/speech.safetensors", dir.display()))
    }
}

/// The I64 tensor `name` of a safetensors file.
fn ids(path: &std::path::Path, name: &str) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap();
    let size = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + size]).unwrap();
    let entry = &header[name];
    assert_eq!(entry["dtype"], "I64", "{name}");
    let offsets = entry["data_offsets"].as_array().unwrap();
    let (start, end) = (
        offsets[0].as_u64().unwrap() as usize,
        offsets[1].as_u64().unwrap() as usize,
    );
    bytes[8 + size + start..8 + size + end]
        .chunks(8)
        .map(|chunk| u32::try_from(i64::from_le_bytes(chunk.try_into().unwrap())).unwrap())
        .collect()
}

#[test]
fn the_prompts_of_the_tts_and_mixed_replies_are_the_rendering_and_the_marker() {
    let (Some(dir), Some(golden)) = (omni(), golden()) else {
        return;
    };
    let mut identity = nagaya_abi::fake::FakeAlways::new().with_family("lfm3").identity();
    identity.model = "lfm3-omni".to_owned();
    let table: toml::Table = format!("model_dir = \"{}\"", dir.display()).parse().unwrap();
    let opened = nagaya::lfm3::open_family(&identity, &table).unwrap();
    let speech = opened.speech.expect("Omni speaks");
    let facts: Value = serde_json::from_str(&std::fs::read_to_string(golden.join("facts.json")).unwrap()).unwrap();
    for (run, marker) in [("tts", speech.audio_start), ("mixed", speech.mixed_start)] {
        let record = &facts["speech"][run];
        let context: Context = serde_json::from_value(json!({"messages": [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["text"]}
        ]}))
        .unwrap();
        let mut prompt = opened.family.render(&context, true).unwrap().tokens;
        prompt.push(marker);
        assert_eq!(
            prompt,
            ids(&golden.join("speech.safetensors"), &format!("{run}.prompt_ids")),
            "{run}"
        );
    }
}
