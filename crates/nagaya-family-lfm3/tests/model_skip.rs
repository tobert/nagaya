//! The rule for a test that needs a model folder (AGENTS.md, "Tests that need a model file"), and the size of the
//! fixtures.

mod common;

use common::model::{Found, check};
use std::path::Path;

const V: &str = "LFM3_OMNI_MODEL_DIR";

#[test]
fn an_unset_variable_skips() {
    assert!(matches!(check(V, None, "tokenizer.json", false, "t"), Found::Skip(line) if line.contains(V)));
}

#[test]
fn an_absent_file_skips() {
    let dir = std::env::temp_dir().join("nagaya-lfm3-no-such-dir");
    let found = check(V, Some(&dir), "tokenizer.json", false, "t");
    assert!(matches!(found, Found::Skip(line) if line.starts_with("skipped: t: ")));
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_absent_file_fails_when_the_variable_is_set() {
    check(V, Some(Path::new("/nonexistent-nagaya")), "tokenizer.json", true, "t");
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_unset_variable_fails_when_the_variable_is_set() {
    check(V, None, "tokenizer.json", true, "t");
}

#[test]
#[should_panic(expected = "is not a file")]
fn a_directory_fails_and_never_skips() {
    let dir = std::env::temp_dir().join(format!("nagaya-lfm3-dir-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("tokenizer.json")).unwrap();
    check(V, Some(&dir), "tokenizer.json", false, "t");
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

#[test]
fn the_fixtures_record_their_source() {
    for name in ["ids.json", "chat.json", "reply.json"] {
        let header = &common::model::fixture(name)["header"];
        assert_eq!(header["tool"], "tools/lfm3_family_fixture.py", "{name}");
        for (model, revision) in [
            ("omni", "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"),
            ("text", "0dc4268b561499c648c7ba7c9466bf3d8d01517a"),
        ] {
            let source = &header["models"][model];
            if source.is_null() {
                continue;
            }
            assert_eq!(source["revision"], revision, "{name} {model}");
            assert!(source["files"]["tokenizer.json"].is_string(), "{name} {model}");
        }
    }
}
