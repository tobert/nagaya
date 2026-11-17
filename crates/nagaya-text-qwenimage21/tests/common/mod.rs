//! Helpers that the integration tests share.
//!
//! A machine does not hold every model, so this state is normal. An absent checkpoint skips the test. The
//! variable `QWENIMAGE21_MODEL_DIR` names the base directory, and the code holds no default path (AGENTS.md). The
//! variable `NAGAYA_REQUIRE_MODELS=1` makes an absent checkpoint fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::PathBuf;

use nagaya_text_qwenimage21::QwenImage21Family;

pub const DIR_VARIABLE: &str = "QWENIMAGE21_MODEL_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

fn require() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// The fixture `tests/fixtures/ids.json`, which `tools/qwenimage21_family_fixture.py` writes.
pub fn fixture() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ids.json");
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

/// The case `name` of the fixture.
pub fn case(name: &str) -> serde_json::Value {
    fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["case"] == name)
        .unwrap_or_else(|| panic!("the fixture has no case {name}"))
        .clone()
}

/// The path of `processor/tokenizer.json` under the base directory. Call `absent` first.
pub fn tokenizer_path() -> PathBuf {
    PathBuf::from(std::env::var_os(DIR_VARIABLE).expect("the test must call `absent` first"))
        .join("processor/tokenizer.json")
}

/// True when the checkpoint is absent and the test must return. It prints one line to stderr.
/// Call it first in each test that needs the checkpoint.
pub fn absent() -> bool {
    let test = std::thread::current().name().unwrap_or("a test").to_owned();
    let skip = |what: String| {
        if require() {
            panic!("{what}. {REQUIRE_VARIABLE} is set. Set {DIR_VARIABLE} to the Qwen-Image-2.1 directory.");
        }
        eprintln!("skipped: {test}: {what}; set {DIR_VARIABLE}");
        true
    };
    let Some(dir) = std::env::var_os(DIR_VARIABLE).map(PathBuf::from) else {
        return skip(format!("{DIR_VARIABLE} is not set"));
    };
    let file = dir.join("processor/tokenizer.json");
    match std::fs::metadata(&file) {
        Ok(m) if m.is_file() => false,
        Ok(_) => panic!(
            "{} is not a file. Set {DIR_VARIABLE} to the Qwen-Image-2.1 directory.",
            file.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            skip(format!("the tokenizer file {} does not exist", file.display()))
        }
        Err(e) => panic!("cannot read the tokenizer file {}: {e}", file.display()),
    }
}

/// The family of the checkpoint, built once for each test binary. Call `absent` first.
pub fn family() -> &'static QwenImage21Family {
    static FAMILY: std::sync::OnceLock<QwenImage21Family> = std::sync::OnceLock::new();
    FAMILY.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os(DIR_VARIABLE).expect("the test must call `absent` first"));
        QwenImage21Family::from_dir(&dir).unwrap_or_else(|e| panic!("the checkpoint builds no family: {e}"))
    })
}
