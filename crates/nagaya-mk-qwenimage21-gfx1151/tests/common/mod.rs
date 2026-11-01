//! Helpers that the integration tests share.
//!
//! Amy, 2026-10-21: "adjust the tests so they don't break our tests when we don't have a model downloaded locally."
//! An absent file skips the test with one line on stderr. `NAGAYA_REQUIRE_MODELS=1` makes it fail. A path that
//! exists but cannot be read fails the test. Only absence skips.
#![allow(dead_code)]

use std::path::PathBuf;

fn require() -> bool {
    std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1")
}

fn skip(what: String) -> Option<PathBuf> {
    if require() {
        panic!("{what}, and NAGAYA_REQUIRE_MODELS is set");
    }
    let test = std::thread::current().name().unwrap_or("a test").to_owned();
    eprintln!("skipped: {test}: {what}");
    None
}

/// The directory in `variable`, when it is set and exists.
fn directory(variable: &str) -> Option<PathBuf> {
    let Some(dir) = std::env::var_os(variable).map(PathBuf::from) else {
        return skip(format!("{variable} is not set"));
    };
    match std::fs::metadata(&dir) {
        Ok(m) if m.is_dir() => Some(dir),
        Ok(_) => panic!("{} is not a directory. Set {variable} to the directory.", dir.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => skip(format!("{} does not exist", dir.display())),
        Err(e) => panic!("cannot read {}: {e}", dir.display()),
    }
}

/// The base model's directory, from `QWENIMAGE21_MODEL_DIR`.
pub fn model_dir() -> Option<PathBuf> {
    directory("QWENIMAGE21_MODEL_DIR")
}

/// FUSAL's directory, from `QWENIMAGE21_FUSAL_DIR`.
pub fn fusal_dir() -> Option<PathBuf> {
    directory("QWENIMAGE21_FUSAL_DIR")
}

/// The oracle directory: `QWENIMAGE21_GOLDEN_DIR`, or `target/qwenimage21-golden` of the workspace. It must hold
/// `encode.safetensors`.
pub fn oracle() -> Option<PathBuf> {
    let dir = std::env::var_os("QWENIMAGE21_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/qwenimage21-golden"));
    match std::fs::metadata(dir.join("encode.safetensors")) {
        Ok(_) => Some(dir),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => skip(format!(
            "the oracle {} does not exist; write it with tools/qwenimage21_golden.sh",
            dir.display()
        )),
        Err(e) => panic!("cannot read the oracle in {}: {e}", dir.display()),
    }
}
