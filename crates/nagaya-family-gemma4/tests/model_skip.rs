//! The rule for a test that needs the checkpoint (AGENTS.md, "Tests that need a model file").

mod common;

use common::model::{Found, check};
use std::path::Path;

#[test]
fn an_unset_variable_skips() {
    assert!(matches!(check(None, false, "t"), Found::Skip(line) if line.contains("GEMMA4_MODEL_DIR")));
}

#[test]
fn an_absent_file_skips() {
    let dir = std::env::temp_dir().join("nagaya-gemma4-no-such-dir");
    assert!(matches!(check(Some(&dir), false, "t"), Found::Skip(line) if line.starts_with("skipped: t: ")));
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_absent_file_fails_when_the_variable_is_set() {
    check(Some(Path::new("/nonexistent-nagaya")), true, "t");
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_unset_variable_fails_when_the_variable_is_set() {
    check(None, true, "t");
}

#[test]
#[should_panic(expected = "is not a file")]
fn a_directory_fails_and_never_skips() {
    let dir = std::env::temp_dir().join(format!("nagaya-gemma4-dir-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("tokenizer.json")).unwrap();
    check(Some(&dir), false, "t");
}

#[test]
fn the_fixtures_stay_small() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let total: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(total < 1_000_000, "the fixtures take {total} bytes, which is over 1 MB");
}
