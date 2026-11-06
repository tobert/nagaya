//! The model folders of the tests, and the rule for a machine that has no copy of them.
//!
//! Amy, 2026-10-21: "adjust the tests so they don't break our tests when we don't have a model downloaded locally.
//! we won't replicate all models to all machines so this state is normal." An absent folder skips the test. The
//! variables `LFM3_OMNI_MODEL_DIR` and `LFM3_MODEL_DIR` name the folders, and the code holds no default path
//! (AGENTS.md). The variable `NAGAYA_REQUIRE_MODELS=1` makes an absent folder fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use nagaya_text_lfm3::Lfm3Family;

pub const OMNI_VARIABLE: &str = "LFM3_OMNI_MODEL_DIR";
pub const TEXT_VARIABLE: &str = "LFM3_MODEL_DIR";
pub const GOLDEN_VARIABLE: &str = "LFM3_GOLDEN_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

/// What the check found for a file.
#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Present,
    Skip(String),
}

/// Check the file `file` in the folder `dir` that the variable `variable` names, or the unset variable when `dir` is
/// `None`. An absent file skips, or panics when `require` is set. Any other error panics.
pub fn check(variable: &str, dir: Option<&Path>, file: &str, require: bool, test: &str) -> Found {
    let absent = |what: String| {
        if require {
            panic!("{what}. {REQUIRE_VARIABLE} is set. Set {variable} to the folder.");
        }
        Found::Skip(format!("skipped: {test}: {what}; set {variable}"))
    };
    let Some(dir) = dir else {
        return absent(format!("{variable} is not set"));
    };
    let path = dir.join(file);
    match std::fs::metadata(&path) {
        Ok(m) if m.is_file() => Found::Present,
        Ok(_) => panic!("{} is not a file. Set {variable} to a model folder.", path.display()),
        Err(e) if e.kind() == ErrorKind::NotFound => absent(format!("{} does not exist", path.display())),
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    }
}

fn configured(variable: &str) -> Option<PathBuf> {
    std::env::var_os(variable).map(PathBuf::from)
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

fn absent_in(variable: &str, dir: Option<PathBuf>, file: &str) -> bool {
    let thread = std::thread::current();
    match check(
        variable,
        dir.as_deref(),
        file,
        required(),
        thread.name().unwrap_or("a test"),
    ) {
        Found::Present => false,
        Found::Skip(line) => {
            eprintln!("{line}");
            true
        }
    }
}

/// True when the Omni folder is absent and the test must return. It prints one line to stderr. Call it first.
pub fn absent_omni() -> bool {
    absent_in(OMNI_VARIABLE, configured(OMNI_VARIABLE), "tokenizer.json")
}

/// True when the folder of `LFM3-1.7B` is absent and the test must return. It prints one line to stderr.
pub fn absent_text() -> bool {
    absent_in(TEXT_VARIABLE, configured(TEXT_VARIABLE), "tokenizer.json")
}

/// True when either folder is absent.
pub fn absent_any() -> bool {
    let omni = absent_omni();
    let text = absent_text();
    omni || text
}

/// The folder of the variable. A test that calls it has called `absent_*` before.
pub fn dir(variable: &str) -> PathBuf {
    let dir = configured(variable).expect("the test must call absent_* first");
    assert!(
        dir.join("tokenizer.json").is_file(),
        "the test must call absent_* first"
    );
    dir
}

/// The family of Omni, built once for each test binary.
pub fn omni() -> &'static Lfm3Family {
    static FAMILY: OnceLock<Lfm3Family> = OnceLock::new();
    FAMILY.get_or_init(|| Lfm3Family::from_dir(&dir(OMNI_VARIABLE)).expect("the Omni folder builds a family"))
}

/// The family of `LFM3-1.7B`, built once for each test binary.
pub fn text() -> &'static Lfm3Family {
    static FAMILY: OnceLock<Lfm3Family> = OnceLock::new();
    FAMILY.get_or_init(|| Lfm3Family::from_dir(&dir(TEXT_VARIABLE)).expect("the text folder builds a family"))
}

/// The family of a fixture's model name, `omni` or `text`.
pub fn family(model: &str) -> &'static Lfm3Family {
    match model {
        "omni" => omni(),
        "text" => text(),
        other => panic!("the fixture names the model {other:?}, which is not omni or text"),
    }
}

/// The folder of the oracle files: `LFM3_GOLDEN_DIR`, or `target/lfm3-golden` of the workspace.
pub fn golden_dir() -> PathBuf {
    configured(GOLDEN_VARIABLE)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"))
}

/// The oracle file `file`, or `None` when it is absent and the test must return. It prints one line to stderr.
pub fn golden(file: &str) -> Option<serde_json::Value> {
    if absent_in(GOLDEN_VARIABLE, Some(golden_dir()), file) {
        return None;
    }
    let path = golden_dir().join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", path.display())))
}

/// A fixture of this crate.
pub fn fixture(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", path.display()))
}
