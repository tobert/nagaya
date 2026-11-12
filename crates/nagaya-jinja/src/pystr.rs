//! The `str` methods of Python.
//!
//! The reference runs the chat template with Python's jinja2, and a template may call any method of a Python string. This module
//! gives those methods with the results of Python 3.14, character for character. The Unicode tables come from Python
//! (`unicode_tables.rs`), because the tables of Rust are of another Unicode version.

use crate::chars;
use crate::unicode_tables::{LOWER_MANY, LOWER_ONE, TITLE_MANY, TITLE_ONE, UPPER_MANY, UPPER_ONE};

/// The code point `c` mapped by a table of single results and a table of results of other lengths, or `None` when the
/// character maps to itself.
fn map(one: &[(u32, u32)], many: &[(u32, &'static str)], c: char, out: &mut String) -> bool {
    let cp = c as u32;
    if let Ok(at) = one.binary_search_by_key(&cp, |&(from, _)| from) {
        out.push(char::from_u32(one[at].1).expect("a table holds scalar values"));
        return true;
    }
    if let Ok(at) = many.binary_search_by_key(&cp, |&(from, _)| from) {
        out.push_str(many[at].1);
        return true;
    }
    false
}

fn push_upper(c: char, out: &mut String) {
    if !map(&UPPER_ONE, &UPPER_MANY, c, out) {
        out.push(c);
    }
}

/// The title case of a character. A character that has no title case of its own takes its upper case.
fn push_title(c: char, out: &mut String) {
    if !map(&TITLE_ONE, &TITLE_MANY, c, out) {
        push_upper(c, out);
    }
}

/// The lower case of the character at `at`. The capital sigma depends on its place in the text.
fn push_lower(chars: &[char], at: usize, out: &mut String) {
    let c = chars[at];
    if c == '\u{3a3}' {
        out.push(if final_sigma(chars, at) { '\u{3c2}' } else { '\u{3c3}' });
    } else if !map(&LOWER_ONE, &LOWER_MANY, c, out) {
        out.push(c);
    }
}

/// A capital sigma is in the final position when a cased letter comes before it and no cased letter comes after it.
/// Characters that are case-ignorable do not count. This is the rule `Final_Sigma` of the Unicode standard.
fn final_sigma(chars: &[char], at: usize) -> bool {
    let before = chars[..at].iter().rev().find(|&&c| !chars::is_case_ignorable(c));
    if !before.is_some_and(|&c| chars::is_cased(c)) {
        return false;
    }
    let after = chars[at + 1..].iter().find(|&&c| !chars::is_case_ignorable(c));
    !after.is_some_and(|&c| chars::is_cased(c))
}

/// `str.lower`.
pub fn lower(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for at in 0..chars.len() {
        push_lower(&chars, at, &mut out);
    }
    out
}

/// `str.upper`.
pub fn upper(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        push_upper(c, &mut out);
    }
    out
}

/// `str.title`: the first letter of each run of cased characters in title case, and the other letters in lower case.
pub fn title(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut after_cased = false;
    for (at, &c) in chars.iter().enumerate() {
        if after_cased {
            push_lower(&chars, at, &mut out);
        } else {
            push_title(c, &mut out);
        }
        after_cased = chars::is_cased(c);
    }
    out
}

/// `str.capitalize`: the first character in title case, and the others in lower case.
pub fn capitalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (at, &c) in chars.iter().enumerate() {
        if at == 0 {
            push_title(c, &mut out);
        } else {
            push_lower(&chars, at, &mut out);
        }
    }
    out
}

/// `str.isalpha`: not empty, and every character is a letter.
pub fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(chars::is_python_alpha)
}

/// `str.isdigit`: not empty, and every character has a digit value.
pub fn is_digit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(chars::is_python_digit)
}

/// `str.isspace`: not empty, and every character is white space.
pub fn is_space(s: &str) -> bool {
    !s.is_empty() && s.chars().all(chars::is_python_space)
}

/// `str.isprintable`: every character is printable. The empty string is printable.
pub fn is_printable(s: &str) -> bool {
    s.chars().all(chars::is_python_printable)
}

/// `repr(str)`: the text in quotes, with escapes for what is not printable.
pub fn repr(s: &str) -> String {
    // Python takes single quotes, unless the text holds a single quote and no double quote.
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            c if c == quote || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            ' '..='~' => out.push(c),
            c if (c as u32) >= 0x80 && chars::is_python_printable(c) => out.push(c),
            c if (c as u32) <= 0xff => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(quote);
    out
}

// ---------------------------------------------------------------- the methods that take arguments

/// Python's rule for the arguments `start` and `end` of a method: a negative index counts from the end, and an index
/// outside the text moves to the nearest end. A missing `end` is the length.
fn adjust(len: usize, start: Option<i64>, end: Option<i64>) -> (i64, i64) {
    let len = len as i64;
    let mut end = end.unwrap_or(len);
    if end > len {
        end = len;
    } else if end < 0 {
        end = (end + len).max(0);
    }
    let mut start = start.unwrap_or(0);
    if start < 0 {
        start = (start + len).max(0);
    }
    (start, end)
}

/// The first place of `sub` in `text[from..to]`, as an index into `text`.
fn search(text: &[char], sub: &[char], from: usize, to: usize) -> Option<usize> {
    if to < from || to - from < sub.len() {
        return None;
    }
    (from..=to - sub.len()).find(|&at| text[at..at + sub.len()] == *sub)
}

/// `str.startswith` for one prefix.
pub fn starts_with(s: &str, prefix: &str, start: Option<i64>, end: Option<i64>) -> bool {
    tail_match(s, prefix, start, end, false)
}

/// `str.endswith` for one suffix.
pub fn ends_with(s: &str, suffix: &str, start: Option<i64>, end: Option<i64>) -> bool {
    tail_match(s, suffix, start, end, true)
}

fn tail_match(s: &str, sub: &str, start: Option<i64>, end: Option<i64>, at_end: bool) -> bool {
    let text: Vec<char> = s.chars().collect();
    let sub: Vec<char> = sub.chars().collect();
    let (start, mut end) = adjust(text.len(), start, end);
    end -= sub.len() as i64;
    if end < start {
        return false;
    }
    let at = if at_end { end } else { start } as usize;
    text[at..at + sub.len()] == *sub
}

/// `str.find`: the index in characters of the first `sub` in the part of the text, or -1.
pub fn find(s: &str, sub: &str, start: Option<i64>, end: Option<i64>) -> i64 {
    let text: Vec<char> = s.chars().collect();
    let sub: Vec<char> = sub.chars().collect();
    let (start, end) = adjust(text.len(), start, end);
    if end < start {
        return -1;
    }
    search(&text, &sub, start as usize, end as usize).map_or(-1, |at| at as i64)
}

/// `str.count`: how many times `sub` occurs in the part of the text, without overlap.
pub fn count(s: &str, sub: &str, start: Option<i64>, end: Option<i64>) -> i64 {
    let text: Vec<char> = s.chars().collect();
    let sub: Vec<char> = sub.chars().collect();
    let (start, end) = adjust(text.len(), start, end);
    if end < start {
        return 0;
    }
    if sub.is_empty() {
        return end - start + 1;
    }
    let (mut from, mut found) = (start as usize, 0);
    while let Some(at) = search(&text, &sub, from, end as usize) {
        found += 1;
        from = at + sub.len();
    }
    found
}

/// `str.replace`. A negative `count` replaces every occurrence. An empty `old` matches between all characters.
pub fn replace(s: &str, old: &str, new: &str, count: i64) -> String {
    if count == 0 {
        return s.to_string();
    }
    let limit = if count < 0 { usize::MAX } else { count as usize };
    let mut out = String::with_capacity(s.len());
    let mut done = 0;
    if old.is_empty() {
        for c in s.chars() {
            if done < limit {
                out.push_str(new);
                done += 1;
            }
            out.push(c);
        }
        if done < limit {
            out.push_str(new);
        }
        return out;
    }
    let mut rest = s;
    while done < limit {
        let Some(at) = rest.find(old) else { break };
        out.push_str(&rest[..at]);
        out.push_str(new);
        rest = &rest[at + old.len()..];
        done += 1;
    }
    out.push_str(rest);
    out
}

/// `str.strip`, `str.lstrip` and `str.rstrip`. Without `chars`, the white space of Python goes.
pub fn strip(s: &str, chars: Option<&str>, left: bool, right: bool) -> String {
    let strips = |c: char| match chars {
        Some(set) => set.contains(c),
        None => chars::is_python_space(c),
    };
    let mut text = s;
    if left {
        text = text.trim_start_matches(strips);
    }
    if right {
        text = text.trim_end_matches(strips);
    }
    text.to_string()
}

/// `str.split`. Without `sep`, runs of white space separate the words. A negative `maxsplit` means no limit.
/// An empty `sep` is an error, as in Python.
pub fn split(s: &str, sep: Option<&str>, maxsplit: i64) -> Result<Vec<String>, String> {
    let limit = if maxsplit < 0 { usize::MAX } else { maxsplit as usize };
    let Some(sep) = sep else {
        return Ok(split_white_space(s, limit));
    };
    if sep.is_empty() {
        return Err("empty separator".to_string());
    }
    let mut parts = Vec::new();
    let mut rest = s;
    while parts.len() < limit {
        let Some(at) = rest.find(sep) else { break };
        parts.push(rest[..at].to_string());
        rest = &rest[at + sep.len()..];
    }
    parts.push(rest.to_string());
    Ok(parts)
}

fn split_white_space(s: &str, limit: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let (mut parts, mut at) = (Vec::new(), 0);
    while parts.len() < limit {
        while at < chars.len() && chars::is_python_space(chars[at]) {
            at += 1;
        }
        if at == chars.len() {
            return parts;
        }
        let from = at;
        while at < chars.len() && !chars::is_python_space(chars[at]) {
            at += 1;
        }
        parts.push(chars[from..at].iter().collect());
    }
    // The limit is reached. The rest keeps its end, and loses the white space before it.
    while at < chars.len() && chars::is_python_space(chars[at]) {
        at += 1;
    }
    if at < chars.len() {
        parts.push(chars[at..].iter().collect());
    }
    parts
}
