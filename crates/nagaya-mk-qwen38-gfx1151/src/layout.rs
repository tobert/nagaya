//! The layouts that cross between Rust and mk's C++ (milestone 15 step 4).
//!
//! The C++ side states each layout in `static_assert` lines, and the compiler checks them in a `hip` build. These
//! tests read the same lines and compare each number with the Rust mirror. Thus a change on one side fails a test,
//! with or without the feature `hip`.

use std::mem::{offset_of, size_of};
use std::path::Path;

use crate::binding::MkSegment;
use crate::context::MkContext;
use crate::schedule::call::MkAccess;
use crate::spec;
use crate::table::{MkWeight, MkWeightTable};

/// The `static_assert(<expression> == <number>, "<note>");` lines of `text`, as (expression, number).
///
/// Each assert must be on one line. A line that starts an assert in another form stops the test, so that no assert is
/// skipped.
fn asserts(text: &str) -> Vec<(String, u64)> {
    let mut found = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("static_assert(") else {
            continue;
        };
        let line_number = index + 1;
        let body = rest
            .strip_suffix(");")
            .unwrap_or_else(|| panic!("line {line_number}: put the static_assert on one line: {line}"));
        let (condition, _note) = body
            .split_once(", \"")
            .unwrap_or_else(|| panic!("line {line_number}: the static_assert has no note: {line}"));
        let (expression, number) = condition
            .split_once(" == ")
            .unwrap_or_else(|| panic!("line {line_number}: the static_assert is not <expression> == <number>: {line}"));
        let number = number
            .parse()
            .unwrap_or_else(|_| panic!("line {line_number}: {number} is not a number: {line}"));
        found.push((expression.to_owned(), number));
    }
    found
}

/// The asserts of one file of `hip/src/engine`.
fn engine_asserts(file: &str) -> Vec<(String, u64)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("hip/src/engine").join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    asserts(&text)
}

/// Compare the asserts of a C++ file with the Rust numbers, as sorted lists.
fn assert_same(file: &str, rust: &[(&str, usize)]) {
    let mut cpp = engine_asserts(file);
    cpp.sort();
    let mut rust: Vec<(String, u64)> = rust
        .iter()
        .map(|(expression, number)| ((*expression).to_owned(), *number as u64))
        .collect();
    rust.sort();
    assert!(!cpp.is_empty(), "{file} states no layout");
    assert_eq!(cpp, rust, "the static_asserts of {file} and the Rust mirror differ");
}

#[test]
fn the_weight_table_has_the_layout_of_weight_table_h() {
    assert_same(
        "weight_table.h",
        &[
            ("sizeof(mk_weight)", size_of::<MkWeight>()),
            ("offsetof(mk_weight, name)", offset_of!(MkWeight, name)),
            ("offsetof(mk_weight, type)", offset_of!(MkWeight, kind)),
            ("offsetof(mk_weight, ne)", offset_of!(MkWeight, ne)),
            ("offsetof(mk_weight, row_bytes)", offset_of!(MkWeight, row_bytes)),
            ("offsetof(mk_weight, d)", offset_of!(MkWeight, d)),
            ("sizeof(mk_weight_table)", size_of::<MkWeightTable>()),
            (
                "offsetof(mk_weight_table, n_weights)",
                offset_of!(MkWeightTable, n_weights),
            ),
            ("offsetof(mk_weight_table, weights)", offset_of!(MkWeightTable, weights)),
            (
                "offsetof(mk_weight_table, schedule)",
                offset_of!(MkWeightTable, schedule),
            ),
            (
                "offsetof(mk_weight_table, footprint)",
                offset_of!(MkWeightTable, footprint),
            ),
            ("sizeof(mk_access)", size_of::<MkAccess>()),
            ("offsetof(mk_access, lo)", offset_of!(MkAccess, lo)),
            ("offsetof(mk_access, hi)", offset_of!(MkAccess, hi)),
            ("offsetof(mk_access, write)", offset_of!(MkAccess, write)),
        ],
    );
}

#[test]
fn a_segment_and_the_buffer_numbers_are_those_of_extern_hip() {
    assert_same(
        "extern.hip",
        &[
            ("sizeof(Segment)", size_of::<MkSegment>()),
            ("offsetof(Segment, c)", offset_of!(MkSegment, context)),
            ("offsetof(Segment, tokens)", offset_of!(MkSegment, tokens)),
            ("offsetof(Segment, t)", offset_of!(MkSegment, t)),
            ("sizeof(bool)", size_of::<bool>()),
            ("sizeof(Context)", size_of::<MkContext>()),
            ("offsetof(Context, n)", offset_of!(MkContext, n)),
            ("offsetof(Context, max)", offset_of!(MkContext, max)),
            ("offsetof(Context, gdn_state)", offset_of!(MkContext, gdn_state)),
            ("offsetof(Context, gdn_conv)", offset_of!(MkContext, gdn_conv)),
            ("offsetof(Context, ple_conv)", offset_of!(MkContext, ple_conv)),
            ("offsetof(Context, kv)", offset_of!(MkContext, kv)),
            ("offsetof(Context, idx_keys)", offset_of!(MkContext, idx_keys)),
            ("offsetof(Context, idx_pooled)", offset_of!(MkContext, idx_pooled)),
            ("offsetof(Context, pooled_n)", offset_of!(MkContext, pooled_n)),
            ("offsetof(Context, ple_hist)", offset_of!(MkContext, ple_hist)),
            ("offsetof(Context, poisoned)", offset_of!(MkContext, poisoned)),
            ("kMaxSegments", spec::MAX_SEGMENTS as usize),
            ("model::kVocab", spec::VOCAB as usize),
            ("model::kMaxPassTokens", spec::MAX_PASS_TOKENS as usize),
            ("model::kMaxContext", spec::MAX_CONTEXT as usize),
        ],
    );
}

#[test]
fn the_reader_of_the_asserts_reads_one_line_asserts() {
    let text = "x\n  static_assert(sizeof(a) == 8, \"a\");\nstatic_assert(offsetof(a, b) == 4, \"a.b\");\n";
    assert_eq!(
        asserts(text),
        [("sizeof(a)".to_owned(), 8), ("offsetof(a, b)".to_owned(), 4)]
    );
}

#[test]
#[should_panic(expected = "put the static_assert on one line")]
fn an_assert_on_two_lines_stops_the_reader() {
    asserts("static_assert(sizeof(a) == 8,\n  \"a\");\n");
}

#[test]
#[should_panic(expected = "is not <expression> == <number>")]
fn an_assert_in_another_form_stops_the_reader() {
    asserts("static_assert(sizeof(a) <= 8, \"a\");\n");
}
