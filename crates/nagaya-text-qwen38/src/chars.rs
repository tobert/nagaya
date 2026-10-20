//! The character classes of the pre-tokenizer pattern.
//!
//! mk's pattern uses `\p{L}`, `\p{M}`, `\p{N}` and `\s` of Python's `regex` module. Rust's own tables are of an older
//! Unicode version, so this module holds the tables of that module (`unicode_tables.rs`). A test compares each class
//! with the fixture for every code point.

use crate::unicode_tables::{LETTER, MARK, NUMBER, SPACE};

fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    let at = table.partition_point(|&(_, last)| last < cp);
    table.get(at).is_some_and(|&(first, _)| first <= cp)
}

/// A letter: Unicode category L.
pub fn is_letter(c: char) -> bool {
    in_table(&LETTER, c)
}

/// A mark: Unicode category M.
pub fn is_mark(c: char) -> bool {
    in_table(&MARK, c)
}

/// A number: Unicode category N.
pub fn is_number(c: char) -> bool {
    in_table(&NUMBER, c)
}

/// White space: the `\s` of the pattern.
pub fn is_space(c: char) -> bool {
    in_table(&SPACE, c)
}
