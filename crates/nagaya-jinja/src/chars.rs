//! The character classes of the pre-tokenizer pattern.
//!
//! The pattern of the reference uses `\p{L}`, `\p{M}`, `\p{N}` and `\s` of Python's `regex` module. Rust's own tables
//! are of an older Unicode version, so this module holds the tables of that module (`unicode_tables.rs`). A test
//! compares each class with the fixture for every code point.

use crate::unicode_tables::{
    LETTER, MARK, NUMBER, PYTHON_ALPHA, PYTHON_CASE_IGNORABLE, PYTHON_CASED, PYTHON_DIGIT, PYTHON_PRINTABLE,
    PYTHON_SPACE, SPACE,
};

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

/// What Python's `str.isspace` accepts, and so what `str.strip` strips. The table comes from Python, like the others. It holds U+001C to U+001F, which `is_space` does not.
pub fn is_python_space(c: char) -> bool {
    in_table(&PYTHON_SPACE, c)
}

/// What Python's `str.isalpha` accepts for one character: the letters, category L.
pub fn is_python_alpha(c: char) -> bool {
    in_table(&PYTHON_ALPHA, c)
}

/// What Python's `str.isdigit` accepts for one character: the characters with a digit value, such as U+00B2.
pub fn is_python_digit(c: char) -> bool {
    in_table(&PYTHON_DIGIT, c)
}

/// What Python's `str.isprintable` accepts for one character. `repr` shows the other characters as escapes.
pub fn is_python_printable(c: char) -> bool {
    in_table(&PYTHON_PRINTABLE, c)
}

/// The Unicode property Cased. The case methods of Python read it.
pub fn is_cased(c: char) -> bool {
    in_table(&PYTHON_CASED, c)
}

/// The Unicode property Case_Ignorable. The rule of the final sigma in `str.lower` reads it.
pub fn is_case_ignorable(c: char) -> bool {
    in_table(&PYTHON_CASE_IGNORABLE, c)
}
