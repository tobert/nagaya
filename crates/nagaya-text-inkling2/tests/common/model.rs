//! The model folders of the tests, and the rule for a machine that has no copy of them.
//!
//! An absent folder skips the test. The variable `NAGAYA_REQUIRE_MODELS=1` makes an absent folder fail the test. Only
//! absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const MINI_VARIABLE: &str = "INKLING2_MODEL_DIR";
pub const VARLOOP_VARIABLE: &str = "INKLING2_VARLOOP_MODEL_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Present,
    Skip(String),
}

pub fn check(path: &Path, variable: &str, require: bool, test: &str) -> Found {
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => Found::Present,
        Ok(_) => panic!(
            "the Inkling 2 model path {} is not a directory. Set {variable} to a directory.",
            path.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let what = format!("the Inkling 2 model folder {} does not exist", path.display());
            if require {
                panic!("{what}. {REQUIRE_VARIABLE} is set. Put the folder there, or set {variable} to its path.");
            }
            Found::Skip(format!("skipped: {test}: {what}; set {variable}"))
        }
        Err(e) => panic!("cannot read the Inkling 2 model folder {}: {e}", path.display()),
    }
}

pub fn check_variable(path: Option<PathBuf>, variable: &str, require: bool, test: &str) -> Found {
    match path {
        Some(path) => check(&path, variable, require, test),
        None if require => panic!("{variable} is not set, and {REQUIRE_VARIABLE} is set. Set {variable}."),
        None => Found::Skip(format!("skipped: {test}: {variable} is not set")),
    }
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when the Mini model folder is absent and the test must return. It prints one line to stderr.
/// Call it first in each test that needs the Mini model.
pub fn absent() -> bool {
    absent_var(MINI_VARIABLE)
}

/// True when the Varloop model folder is absent and the test must return.
pub fn absent_varloop() -> bool {
    absent_var(VARLOOP_VARIABLE)
}

fn absent_var(variable: &str) -> bool {
    let thread = std::thread::current();
    let path = std::env::var_os(variable).map(PathBuf::from);
    match check_variable(path, variable, required(), thread.name().unwrap_or("a test")) {
        Found::Present => false,
        Found::Skip(line) => {
            eprintln!("{line}");
            true
        }
    }
}

pub fn mini_dir() -> PathBuf {
    let path = std::env::var_os(MINI_VARIABLE)
        .map(PathBuf::from)
        .expect("the test must call `absent` first: INKLING2_MODEL_DIR is not set");
    assert!(path.is_dir(), "the Inkling 2 folder {} does not exist", path.display());
    path
}

pub fn varloop_dir() -> PathBuf {
    let path = std::env::var_os(VARLOOP_VARIABLE)
        .map(PathBuf::from)
        .expect("the test must call `absent_varloop` first: INKLING2_VARLOOP_MODEL_DIR is not set");
    assert!(
        path.is_dir(),
        "the Inkling 2 varloop folder {} does not exist",
        path.display()
    );
    path
}
