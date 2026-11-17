//! The model file of the tests, and the rule for a machine that has no copy of it.
//!
//! A machine does not hold every model, so this state is normal. An absent file skips the test.
//! The variable `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// The variable that names the first shard of the Qwen3.8 GGUF. It has no default, as each model folder has none.
pub const GGUF_VARIABLE: &str = "NAGAYA_QWEN38_GGUF";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

/// What the check found for the model file.
#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Present,
    Skip(String),
}

/// Check the file at `path`. An absent file skips, or panics when `require` is set. Any other error panics.
pub fn check(path: &Path, require: bool, test: &str) -> Found {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() => Found::Present,
        Ok(_) => panic!(
            "the Qwen3.8 GGUF path {} is not a file. Set {GGUF_VARIABLE} to a file.",
            path.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let what = format!("the Qwen3.8 GGUF file {} does not exist", path.display());
            if require {
                panic!("{what}. {REQUIRE_VARIABLE} is set. Put the file there, or set {GGUF_VARIABLE} to its path.");
            }
            Found::Skip(format!("skipped: {test}: {what}; set {GGUF_VARIABLE}"))
        }
        Err(e) => panic!("cannot read the Qwen3.8 GGUF file {}: {e}", path.display()),
    }
}

fn configured() -> Option<PathBuf> {
    std::env::var_os(GGUF_VARIABLE).map(PathBuf::from)
}

/// Check the file that the variable names. An unset variable skips, as an absent file does, and it panics when
/// `require` is set.
pub fn check_variable(path: Option<PathBuf>, require: bool, test: &str) -> Found {
    match path {
        Some(path) => check(&path, require, test),
        None if require => panic!(
            "{GGUF_VARIABLE} is not set, and {REQUIRE_VARIABLE} is set. Set {GGUF_VARIABLE} to the first shard of the \
             Qwen3.8 GGUF."
        ),
        None => Found::Skip(format!("skipped: {test}: {GGUF_VARIABLE} is not set")),
    }
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when the model file is absent and the test must return. It prints one line to stderr.
/// Call it first in each test that needs the model.
pub fn absent() -> bool {
    let thread = std::thread::current();
    match check_variable(configured(), required(), thread.name().unwrap_or("a test")) {
        Found::Present => false,
        Found::Skip(line) => {
            eprintln!("{line}");
            true
        }
    }
}

/// The path of the model file. A test that calls it has called `absent` before, so the file is there.
pub fn gguf_path() -> PathBuf {
    let path = configured().expect("the test must call `absent` first: NAGAYA_QWEN38_GGUF is not set");
    assert!(
        path.is_file(),
        "the Qwen3.8 GGUF file {} does not exist; the test must call `absent` first",
        path.display()
    );
    path
}
