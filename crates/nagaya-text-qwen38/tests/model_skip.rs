//! Tests of the rule for an absent model file (`tests/common/model.rs`).

mod common;

use common::model::{Found, check};
use std::path::Path;

fn present_file() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

#[test]
fn an_absent_file_skips_when_the_variable_is_not_set() {
    let found = check(Path::new("/nonexistent/nagaya/model.gguf"), false, "the_test");
    let Found::Skip(line) = found else {
        panic!("want a skip, got {found:?}")
    };
    assert!(line.starts_with("skipped: the_test: "), "{line}");
    assert!(line.contains("/nonexistent/nagaya/model.gguf"), "{line}");
    assert!(line.contains("NAGAYA_QWEN38_GGUF"), "{line}");
    assert!(!line.contains('\n'), "{line}");
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_absent_file_fails_when_the_variable_is_set() {
    check(Path::new("/nonexistent/nagaya/model.gguf"), true, "the_test");
}

#[test]
fn a_present_file_runs_in_both_modes() {
    assert_eq!(check(&present_file(), false, "the_test"), Found::Present);
    assert_eq!(check(&present_file(), true, "the_test"), Found::Present);
}

#[test]
#[should_panic(expected = "is not a file")]
fn a_directory_fails_and_never_skips() {
    check(Path::new(env!("CARGO_MANIFEST_DIR")), false, "the_test");
}

#[cfg(unix)]
#[test]
#[should_panic(expected = "cannot read")]
fn a_file_that_cannot_be_reached_fails_and_never_skips() {
    // A path below a plain file gives "not a directory", which is not an absence.
    let path = present_file().join("model.gguf");
    check(&path, false, "the_test");
}
