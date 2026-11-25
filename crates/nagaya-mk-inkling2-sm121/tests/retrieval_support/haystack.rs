//! The haystack: books of PG-19 in the order of their numbers, with the needles put in at fixed places in the text.

use super::needles::Needle;

/// The text of the context and where each needle is in it.
pub struct Haystack {
    pub text: String,
    /// The byte offset of the sentence of each needle in `text`, in the order of the needles.
    pub at: Vec<usize>,
}

/// The first byte at or after `from` that starts a line, or the end of the text.
fn line_start(text: &str, from: usize) -> usize {
    let mut from = from.min(text.len());
    while !text.is_char_boundary(from) {
        from += 1;
    }
    match text[from..].find('\n') {
        Some(i) => from + i + 1,
        None => text.len(),
    }
}

/// Join the books until `count` says that the text holds `want` tokens, cut the last book where the share of tokens
/// ends, and put each needle in at its percent of the text, at the start of a line. `needles` is empty for the
/// control run that holds no needle.
///
/// `book(n)` gives the text of book `n` from 0. `count` gives the tokens of a text.
pub fn assemble(
    mut book: impl FnMut(usize) -> String,
    count: impl Fn(&str) -> usize,
    want: usize,
    needles: &[Needle],
) -> Haystack {
    let mut text = String::new();
    let mut tokens = 0usize;
    let mut n = 0;
    while tokens < want {
        let next = book(n);
        let counted = count(&next);
        assert!(counted > 0, "book {n} has no token");
        if tokens + counted > want {
            // Cut the book where the missing share of its tokens ends.
            let share = (want - tokens) as f64 / counted as f64;
            let mut cut = (next.len() as f64 * share) as usize;
            while !next.is_char_boundary(cut) {
                cut -= 1;
            }
            let cut = next[..cut].rfind('\n').unwrap_or(cut);
            text.push_str(&next[..cut]);
            tokens = want;
        } else {
            text.push_str(&next);
            text.push('\n');
            tokens += counted;
        }
        n += 1;
    }
    let base = text.len();
    let mut places: Vec<(usize, usize)> = needles
        .iter()
        .enumerate()
        .map(|(i, needle)| (line_start(&text, base / 100 * needle.percent as usize), i))
        .collect();
    places.sort();
    let mut at = vec![0; needles.len()];
    let mut out = String::with_capacity(base + 256 * needles.len());
    let mut last = 0;
    for (place, i) in places {
        assert!(place >= last, "two needles share a place in the text");
        out.push_str(&text[last..place]);
        out.push('\n');
        at[i] = out.len();
        out.push_str(&needles[i].sentence());
        out.push_str("\n\n");
        last = place;
    }
    out.push_str(&text[last..]);
    Haystack { text: out, at }
}

/// Check that `context` is the start of `full`. A template that changes an earlier token would break the snapshot, so
/// the error names the first token that differs.
pub fn check_prefix(context: &[u32], full: &[u32], what: &str) -> Result<(), String> {
    if full.len() >= context.len() && full[..context.len()] == *context {
        return Ok(());
    }
    let first = context.iter().zip(full).position(|(a, b)| a != b);
    Err(match first {
        Some(i) => format!("the tokens of the context are not the start of {what}: token {i} differs"),
        None => format!(
            "the tokens of the context ({}) are not the start of {what}, which has {} tokens",
            context.len(),
            full.len()
        ),
    })
}
