//! The rule for a machine with no checkpoint: absence skips, and anything else fails.

mod common;

use common::model::{Found, check};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nagaya-qwen3tts-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    dir
}

#[test]
fn an_unset_variable_skips() {
    match check(None, false, "t") {
        Found::Skip(line) => assert!(line.contains("QWEN3TTS_MODEL_DIR"), "{line}"),
        Found::Present => panic!("an unset variable is not a checkpoint"),
    }
}

#[test]
fn an_absent_vocabulary_skips() {
    let dir = temp_dir("absent");
    assert!(matches!(check(Some(&dir), false, "t"), Found::Skip(_)));
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_absent_vocabulary_fails_when_models_are_required() {
    let dir = temp_dir("required");
    check(Some(&dir), true, "t");
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_unset_variable_fails_when_models_are_required() {
    check(None, true, "t");
}

#[test]
#[should_panic(expected = "is not a file")]
fn a_vocabulary_path_that_is_a_directory_fails() {
    let dir = temp_dir("dir");
    std::fs::create_dir_all(dir.join("vocab.json")).expect("a directory");
    check(Some(&dir), false, "t");
}

#[test]
fn a_present_vocabulary_is_present() {
    let dir = temp_dir("present");
    std::fs::write(dir.join("vocab.json"), "{}").expect("a file");
    assert_eq!(check(Some(&dir), false, "t"), Found::Present);
}
