//! The model folder of the tests, and the rule for a machine that has no copy of it.
//!
//! A machine does not hold every model, so this state is normal. An absent file skips the test.
//! The variable `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// The variable that names the folder of Inkling 2 Mini. It has no default.
pub const MODEL_VARIABLE: &str = "INKLING2_MODEL_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";
/// The one file of the weights in the folder.
pub const WEIGHTS_FILE: &str = "model-00001-of-00001.safetensors";

/// What the check found for the weights file.
#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Present,
    Skip(String),
}

/// Check the weights file at `path`. An absent file skips, or panics when `require` is set. Any other error panics.
pub fn check(path: &Path, require: bool, test: &str) -> Found {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() => Found::Present,
        Ok(_) => panic!(
            "the weights path {} is not a file. Set {MODEL_VARIABLE} to the model folder.",
            path.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let what = format!("the weights file {} does not exist", path.display());
            if require {
                panic!("{what}. {REQUIRE_VARIABLE} is set. Put the file there, or set {MODEL_VARIABLE}.");
            }
            Found::Skip(format!("skipped: {test}: {what}; set {MODEL_VARIABLE}"))
        }
        Err(e) => panic!("cannot read the weights file {}: {e}", path.display()),
    }
}

/// Check the file in the folder that the variable names. An unset variable skips, and it panics when `require` is set.
pub fn check_variable(dir: Option<PathBuf>, require: bool, test: &str) -> Found {
    match dir {
        Some(dir) => check(&dir.join(WEIGHTS_FILE), require, test),
        None if require => {
            panic!(
                "{MODEL_VARIABLE} is not set, and {REQUIRE_VARIABLE} is set. Set it to the folder of Inkling 2 Mini."
            )
        }
        None => Found::Skip(format!("skipped: {test}: {MODEL_VARIABLE} is not set")),
    }
}

fn configured() -> Option<PathBuf> {
    std::env::var_os(MODEL_VARIABLE).map(PathBuf::from)
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when the weights file is absent and the test must return. It prints one line to stderr.
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

/// The path of the weights file. A test that calls it has called `absent` before.
pub fn weights_path() -> PathBuf {
    configured()
        .expect("the test must call `absent` first")
        .join(WEIGHTS_FILE)
}
