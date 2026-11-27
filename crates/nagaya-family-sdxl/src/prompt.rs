//! The A1111 prompt syntax (spec 11.9a): `(text)` multiplies the weight of its text by 1.1, `[text]` divides it by
//! 1.1, and `(text:w)` multiplies it by `w`. `\(`, `\)`, `\[`, `\]` and `\\` are plain characters. `BREAK`, as a word
//! of its own, ends a chunk.
//!
//! This parser follows the syntax as A1111 documents it. It is not a port of A1111's code. A group that does not end
//! weights the rest of the prompt, and a closing bracket with no open group is text, as in A1111.

/// A piece of a parsed prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    /// Text and the weight of each of its tokens.
    Text { text: String, weight: f32 },
    /// The end of a chunk.
    Break,
}

const ROUND: f32 = 1.1;

/// Parse an A1111 prompt into its pieces. Text with equal weights next to each other joins, and empty text goes.
pub fn parse(prompt: &str) -> Vec<Piece> {
    let chars: Vec<char> = prompt.chars().collect();
    let mut pieces: Vec<Piece> = Vec::new();
    // The place in `pieces` where each open group starts.
    let mut round: Vec<usize> = Vec::new();
    let mut square: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' if chars.get(i + 1).is_some_and(|next| "()[]\\".contains(*next)) => {
                push_text(&mut pieces, chars[i + 1]);
                i += 2;
                continue;
            }
            '(' => {
                round.push(start_group(&mut pieces));
            }
            '[' => {
                square.push(start_group(&mut pieces));
            }
            ')' => match round.pop() {
                Some(start) => scale(&mut pieces, start, ROUND),
                None => push_text(&mut pieces, c),
            },
            ']' => match square.pop() {
                Some(start) => scale(&mut pieces, start, 1.0 / ROUND),
                None => push_text(&mut pieces, c),
            },
            ':' if !round.is_empty() => match weight_close(&chars[i + 1..]) {
                Some((weight, used)) => {
                    let start = round.pop().expect("the group is open");
                    scale(&mut pieces, start, weight);
                    i += 1 + used;
                    continue;
                }
                None => push_text(&mut pieces, c),
            },
            'B' if is_break(&chars, i) => {
                trim_end(&mut pieces);
                pieces.push(Piece::Break);
                i += "BREAK".len();
                while chars.get(i).is_some_and(|c| c.is_whitespace()) {
                    i += 1;
                }
                continue;
            }
            _ => push_text(&mut pieces, c),
        }
        i += 1;
    }
    // A group that does not end weights the rest of the prompt.
    for start in round {
        scale(&mut pieces, start, ROUND);
    }
    for start in square {
        scale(&mut pieces, start, 1.0 / ROUND);
    }
    join(pieces)
}

/// Start a new piece of text, so that a group weights only its own text. Gives the place of the group's start.
fn start_group(pieces: &mut Vec<Piece>) -> usize {
    pieces.push(Piece::Text {
        text: String::new(),
        weight: 1.0,
    });
    pieces.len() - 1
}

/// Multiply the weight of each piece of text from `start` by `factor`, and start a new piece after them.
fn scale(pieces: &mut Vec<Piece>, start: usize, factor: f32) {
    for piece in &mut pieces[start..] {
        if let Piece::Text { weight, .. } = piece {
            *weight *= factor;
        }
    }
    start_group(pieces);
}

/// Add a character to the last piece of text, or start one.
fn push_text(pieces: &mut Vec<Piece>, c: char) {
    match pieces.last_mut() {
        Some(Piece::Text { text, .. }) => text.push(c),
        _ => pieces.push(Piece::Text {
            text: c.to_string(),
            weight: 1.0,
        }),
    }
}

/// Remove the white space at the end of the text before a `BREAK`.
fn trim_end(pieces: &mut [Piece]) {
    if let Some(Piece::Text { text, .. }) = pieces.last_mut() {
        text.truncate(text.trim_end().len());
    }
}

/// A weight and the closing bracket after a colon: optional spaces, a number of digits, `.`, `+` or `-`, optional
/// spaces, and `)`. Gives the weight and the count of characters after the colon.
fn weight_close(rest: &[char]) -> Option<(f32, usize)> {
    let mut i = 0;
    while rest.get(i).is_some_and(|c| c.is_whitespace()) {
        i += 1;
    }
    let start = i;
    while rest
        .get(i)
        .is_some_and(|c| c.is_ascii_digit() || *c == '.' || ((*c == '+' || *c == '-') && i == start))
    {
        i += 1;
    }
    let number: String = rest[start..i].iter().collect();
    while rest.get(i).is_some_and(|c| c.is_whitespace()) {
        i += 1;
    }
    if rest.get(i) != Some(&')') {
        return None;
    }
    Some((number.parse().ok()?, i + 1))
}

/// True at a `BREAK` that is a word of its own.
fn is_break(chars: &[char], i: usize) -> bool {
    let word = |c: Option<&char>| c.is_some_and(|c| c.is_alphanumeric() || *c == '_');
    chars[i..].starts_with(&['B', 'R', 'E', 'A', 'K']) && (i == 0 || !word(chars.get(i - 1))) && !word(chars.get(i + 5))
}

/// Join text with equal weights next to each other, and drop empty text.
fn join(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    for piece in pieces {
        match (&piece, out.last_mut()) {
            (Piece::Text { text, .. }, _) if text.is_empty() => {}
            (
                Piece::Text { text, weight },
                Some(Piece::Text {
                    text: last,
                    weight: last_weight,
                }),
            ) if weight == last_weight => last.push_str(text),
            _ => out.push(piece),
        }
    }
    out
}
