//! The checkpoint of the tests, and the rule for a machine that has no copy of it.
//!
//! A machine does not hold every model, so this state is normal. An absent checkpoint skips the test. The
//! variable `GEMMA4_MODEL_DIR` names the directory, and the code holds no default path (AGENTS.md). The variable
//! `NAGAYA_REQUIRE_MODELS=1` makes an absent checkpoint fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use nagaya_text_gemma4::Gemma4Family;

pub const DIR_VARIABLE: &str = "GEMMA4_MODEL_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

/// What the check found for the checkpoint.
#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Present,
    Skip(String),
}

/// Check the tokenizer file in the directory `dir`, or the unset variable when `dir` is `None`. An absent file
/// skips, or panics when `require` is set. Any other error panics.
pub fn check(dir: Option<&Path>, require: bool, test: &str) -> Found {
    let absent = |what: String| {
        if require {
            panic!("{what}. {REQUIRE_VARIABLE} is set. Set {DIR_VARIABLE} to the Gemma 4 checkpoint directory.");
        }
        Found::Skip(format!("skipped: {test}: {what}; set {DIR_VARIABLE}"))
    };
    let Some(dir) = dir else {
        return absent(format!("{DIR_VARIABLE} is not set"));
    };
    let file = dir.join("tokenizer.json");
    match std::fs::metadata(&file) {
        Ok(m) if m.is_file() => Found::Present,
        Ok(_) => panic!(
            "{} is not a file. Set {DIR_VARIABLE} to a checkpoint directory.",
            file.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            absent(format!("the Gemma 4 tokenizer file {} does not exist", file.display()))
        }
        Err(e) => panic!("cannot read the Gemma 4 tokenizer file {}: {e}", file.display()),
    }
}

fn configured() -> Option<PathBuf> {
    std::env::var_os(DIR_VARIABLE).map(PathBuf::from)
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when the checkpoint is absent and the test must return. It prints one line to stderr.
/// Call it first in each test that needs the checkpoint.
pub fn absent() -> bool {
    let thread = std::thread::current();
    match check(configured().as_deref(), required(), thread.name().unwrap_or("a test")) {
        Found::Present => false,
        Found::Skip(line) => {
            eprintln!("{line}");
            true
        }
    }
}

/// The checkpoint directory. A test that calls it has called `absent` before, so the directory is there.
pub fn model_dir() -> PathBuf {
    let dir = configured().expect("the test must call `absent` first");
    assert!(
        dir.join("tokenizer.json").is_file(),
        "the test must call `absent` first"
    );
    dir
}

/// The family of the checkpoint, built once for each test binary.
pub fn family() -> &'static Gemma4Family {
    static FAMILY: std::sync::OnceLock<Gemma4Family> = std::sync::OnceLock::new();
    FAMILY.get_or_init(|| Gemma4Family::from_dir(&model_dir()).expect("the checkpoint builds a family"))
}
