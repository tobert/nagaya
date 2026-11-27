//! The fixture files themselves: one header, one digest of the reference, a small size.

mod common;

use common::{JsonExt, fixture};

const FILES: [&str; 5] = ["tokenizer.json", "chat.json", "utf8.json", "classes.json", "jinja.json"];

#[test]
fn all_fixtures_come_from_one_run_of_one_reference() {
    let first = fixture(FILES[0]);
    let header = first.get("header");
    let sha = header.get("reference_sha").str();
    assert!(
        sha.len() == 16 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "reference_sha {sha:?} is not the digest of tools/qwen38_reference"
    );
    for name in FILES {
        assert_eq!(fixture(name).get("header"), header, "the header of {name}");
    }
}

/// The convention of the crates is 1 MB. These fixtures hold the full repository paths of the reference in each header,
/// which took them past it, so the limit of this crate is 2 MB, as for the op fixtures of the Qwen3.8 engine.
#[test]
fn the_fixtures_stay_small() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let total: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(total < 2_000_000, "the fixtures take {total} bytes, which is over 2 MB");
}

#[test]
fn the_fixtures_name_the_files_of_the_reference_that_made_them() {
    let header = fixture("chat.json");
    let files: Vec<&str> = header
        .get("header")
        .get("source_files")
        .items()
        .iter()
        .map(|f| f.str())
        .collect();
    assert_eq!(
        files,
        [
            "tools/qwen38_reference/tokenizer.py",
            "tools/qwen38_reference/template.py",
            "tools/qwen38_reference/common.py",
            "tools/qwen38_reference/chatfmt.py"
        ]
    );
}
