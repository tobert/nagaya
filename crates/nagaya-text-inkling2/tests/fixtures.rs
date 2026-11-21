mod common;

use common::model::{Found, MINI_VARIABLE, check, check_variable};
use std::path::Path;

const LIMIT_BYTES: u64 = 1_000_000;

#[test]
fn fixtures_stay_under_one_megabyte() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut total = 0u64;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let meta = entry.unwrap().metadata().unwrap();
        if meta.is_file() {
            total += meta.len();
        }
    }
    assert!(total > 0 && total < LIMIT_BYTES, "fixtures total {total} bytes");
}

#[test]
fn model_skip_rules_follow_agents_md() {
    assert!(matches!(
        check_variable(None, MINI_VARIABLE, false, "t"),
        Found::Skip(_)
    ));
    let missing = Path::new("/nonexistent/inkling2/model/dir");
    assert!(matches!(check(missing, MINI_VARIABLE, false, "t"), Found::Skip(_)));
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn model_require_panics_when_missing() {
    check_variable(None, MINI_VARIABLE, true, "t");
}
