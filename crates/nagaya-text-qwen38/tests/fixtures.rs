//! The fixture files themselves: one header, one revision of the reference, a small size.

mod common;

use common::{JsonExt, fixture};

const FILES: [&str; 5] = ["tokenizer.json", "chat.json", "utf8.json", "classes.json", "jinja.json"];

#[test]
fn all_fixtures_come_from_one_run_of_one_revision_of_the_reference() {
    let first = fixture(FILES[0]);
    let header = first.get("header");
    let rev = header.get("reference_rev").str();
    assert!(
        rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
        "reference_rev {rev:?} is not a git revision"
    );
    for name in FILES {
        assert_eq!(fixture(name).get("header"), header, "the header of {name}");
    }
}

#[test]
fn the_fixtures_stay_small() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let total: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(total < 1_000_000, "the fixtures take {total} bytes, which is over 1 MB");
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
            "harness/tokenizer.py",
            "service/template.py",
            "service/common.py",
            "service/chatfmt.py"
        ]
    );
}
