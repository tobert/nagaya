//! The byte-level BPE tokenizer of Qwen3.8, from the vocabulary in the GGUF metadata.
//!
//! This is a port of mk's `harness/tokenizer.py`. Special tokens match as whole tokens before BPE, longest text first.
//! Content that a client sends is escaped first: each special token's text becomes one Unicode noncharacter, so that
//! the client cannot forge a role of the chat template.

use crate::Error;
use crate::chars::{is_letter, is_mark, is_number, is_space};
use crate::gguf::{Metadata, Value};
use std::collections::HashMap;
use std::path::Path;

/// The values of `tokenizer.ggml.token_type` that this library tells apart.
const TYPE_NORMAL: i64 = 1;
const TYPE_UNKNOWN: i64 = 2;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

/// The number of noncharacters in Unicode: U+FDD0 to U+FDEF, and U+FFFE and U+FFFF of each of 17 planes.
const RESERVED_COUNT: usize = 32 + 34;

/// The noncharacter with this index. The index order is the order of mk's `RESERVED` list.
fn reserved_char(index: usize) -> char {
    let cp = if index < 32 {
        0xFDD0 + index as u32
    } else {
        let k = (index - 32) as u32;
        ((k / 2) << 16) | (0xFFFE + (k % 2))
    };
    char::from_u32(cp).expect("a noncharacter is a scalar value")
}

/// The index of a noncharacter, or `None` for any other character.
fn reserved_index(c: char) -> Option<usize> {
    let cp = c as u32;
    if (0xFDD0..=0xFDEF).contains(&cp) {
        Some((cp - 0xFDD0) as usize)
    } else if cp & 0xFFFE == 0xFFFE {
        Some(32 + (cp >> 16) as usize * 2 + (cp & 1) as usize)
    } else {
        None
    }
}

/// The GPT-2 map from a byte to a printable character. The vocabulary was built with this map.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let printable = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        *slot = if printable {
            char::from_u32(b as u32).expect("a byte is a scalar value")
        } else {
            let c = char::from_u32(next).expect("256 and up are scalar values");
            next += 1;
            c
        };
    }
    table
}

/// The tokenizer. It holds the vocabulary, the merge ranks and the tables of special tokens.
pub struct Tokenizer {
    tokens: Vec<String>,
    types: Vec<i64>,
    token_to_id: HashMap<String, u32>,
    /// The rank of each merge, with the key "left right". The pieces never hold a space, because the byte map sends
    /// the space byte to another character.
    merge_rank: HashMap<String, u32>,
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
    /// The ids of special tokens (types control and user-defined), in order of id.
    special_ids: Vec<u32>,
    /// The special tokens, longest text first.
    specials_longest_first: Vec<(String, u32)>,
    /// For each first byte of a text, the indexes into `specials_longest_first` in that order.
    special_start: Vec<Vec<usize>>,
    /// The text of the special token whose noncharacter has this index.
    escape_text: Vec<String>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tokenizer({} tokens, {} merges)",
            self.tokens.len(),
            self.merge_rank.len()
        )
    }
}

const FAMILY_HINT: &str = "Use a GGUF made for this family.";

fn string_array<'a>(m: &'a Metadata, key: &str) -> Result<Vec<&'a str>, Error> {
    match m.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                Value::Str(s) => Ok(s.as_str()),
                other => Err(Error::Metadata(format!(
                    "{key} holds {other:?}, not a string. {FAMILY_HINT}"
                ))),
            })
            .collect(),
        Some(_) => Err(Error::Metadata(format!("{key} is not an array. {FAMILY_HINT}"))),
        None => Err(Error::Metadata(format!("the key {key} is missing. {FAMILY_HINT}"))),
    }
}

fn int_array(m: &Metadata, key: &str) -> Result<Vec<i64>, Error> {
    match m.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                Value::U8(x) => Ok(*x as i64),
                Value::I8(x) => Ok(*x as i64),
                Value::U16(x) => Ok(*x as i64),
                Value::I16(x) => Ok(*x as i64),
                Value::U32(x) => Ok(*x as i64),
                Value::I32(x) => Ok(*x as i64),
                Value::I64(x) => Ok(*x),
                Value::U64(x) => i64::try_from(*x).map_err(|_| {
                    Error::Metadata(format!(
                        "{key} holds {x}, which does not fit a signed 64-bit integer. {FAMILY_HINT}"
                    ))
                }),
                other @ (Value::F32(_) | Value::F64(_)) => Err(Error::Metadata(format!(
                    "{key} holds {other:?}, a float, not an integer. {FAMILY_HINT}"
                ))),
                other => Err(Error::Metadata(format!(
                    "{key} holds {other:?}, not an integer. {FAMILY_HINT}"
                ))),
            })
            .collect(),
        Some(_) => Err(Error::Metadata(format!("{key} is not an array. {FAMILY_HINT}"))),
        None => Err(Error::Metadata(format!("the key {key} is missing. {FAMILY_HINT}"))),
    }
}

fn require_string(m: &Metadata, key: &str, wanted: &str) -> Result<(), Error> {
    match m.get(key) {
        Some(Value::Str(s)) if s == wanted => Ok(()),
        Some(Value::Str(s)) => Err(Error::Metadata(format!(
            "{key} is {s:?}. This tokenizer reads {wanted:?} only. {FAMILY_HINT}"
        ))),
        Some(_) => Err(Error::Metadata(format!("{key} is not a string. {FAMILY_HINT}"))),
        None => Err(Error::Metadata(format!("the key {key} is missing. {FAMILY_HINT}"))),
    }
}

impl Tokenizer {
    /// Build the tokenizer from the keys `tokenizer.ggml.pre`, `.model`, `.tokens`, `.token_type` and `.merges`.
    pub fn from_metadata(m: &Metadata) -> Result<Tokenizer, Error> {
        require_string(m, "tokenizer.ggml.pre", "qwen35")?;
        require_string(m, "tokenizer.ggml.model", "gpt2")?;
        let texts = string_array(m, "tokenizer.ggml.tokens")?;
        let types = int_array(m, "tokenizer.ggml.token_type")?;
        let merges = string_array(m, "tokenizer.ggml.merges")?;
        if texts.len() != types.len() {
            return Err(Error::Metadata(format!(
                "tokenizer.ggml.tokens ({}) and tokenizer.ggml.token_type ({}) differ in length. {FAMILY_HINT}",
                texts.len(),
                types.len()
            )));
        }
        if u32::try_from(texts.len()).is_err() {
            return Err(Error::Metadata(format!(
                "{} tokens do not fit in 32-bit ids",
                texts.len()
            )));
        }
        let mut token_to_id = HashMap::with_capacity(texts.len());
        for (id, text) in texts.iter().enumerate() {
            if token_to_id.insert(text.to_string(), id as u32).is_some() {
                return Err(Error::Vocab(format!(
                    "duplicate token text {text:?}: the BPE lookup would be ambiguous. {FAMILY_HINT}"
                )));
            }
        }
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            if merge.split(' ').count() != 2 {
                return Err(Error::Metadata(format!(
                    "merge {rank} is not two pieces with one space between them: {merge:?}"
                )));
            }
            // A repeated merge keeps its last rank, as the dict of mk does.
            merge_rank.insert(merge.to_string(), rank as u32);
        }
        let special_ids: Vec<u32> = types
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == TYPE_CONTROL || t == TYPE_USER_DEFINED)
            .map(|(i, _)| i as u32)
            .collect();
        if special_ids.len() > RESERVED_COUNT {
            return Err(Error::Vocab(format!(
                "{} special tokens, but only {RESERVED_COUNT} noncharacters to escape them with",
                special_ids.len()
            )));
        }
        let mut specials_longest_first: Vec<(String, u32)> = special_ids
            .iter()
            .map(|&id| (texts[id as usize].to_string(), id))
            .collect();
        // The sort is stable, so equal lengths keep the order of ids. A length is a count of characters, as in Python.
        specials_longest_first.sort_by_key(|(text, _)| std::cmp::Reverse(text.chars().count()));
        let mut special_start = vec![Vec::new(); 256];
        for (i, (text, id)) in specials_longest_first.iter().enumerate() {
            match text.as_bytes().first() {
                Some(&b) => special_start[b as usize].push(i),
                // mk builds an empty alternative here and splits the text at every position. We refuse the file.
                None => {
                    return Err(Error::Vocab(format!(
                        "special token {id} has an empty text. {FAMILY_HINT}"
                    )));
                }
            }
        }
        let escape_text = special_ids.iter().map(|&id| texts[id as usize].to_string()).collect();
        let byte_char = byte_to_char_table();
        let char_byte = byte_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Ok(Tokenizer {
            tokens: texts.iter().map(|t| t.to_string()).collect(),
            types,
            token_to_id,
            merge_rank,
            byte_char,
            char_byte,
            special_ids,
            specials_longest_first,
            special_start,
            escape_text,
        })
    }

    /// Build the tokenizer from the metadata of the GGUF file at `path`.
    pub fn from_gguf(path: &Path) -> Result<Tokenizer, Error> {
        Tokenizer::from_metadata(&Metadata::open(path)?)
    }

    /// The number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The ids of the special tokens (control and user-defined), in order of id.
    pub fn special_ids(&self) -> &[u32] {
        &self.special_ids
    }

    fn check_id(&self, id: u32) -> Result<usize, Error> {
        if (id as usize) < self.tokens.len() {
            Ok(id as usize)
        } else {
            Err(Error::TokenId {
                id,
                count: self.tokens.len(),
            })
        }
    }

    /// The text of a token as the vocabulary holds it (byte-mapped, not decoded).
    pub fn token_text(&self, id: u32) -> Result<&str, Error> {
        Ok(&self.tokens[self.check_id(id)?])
    }

    /// The value of `tokenizer.ggml.token_type` for a token.
    pub fn token_type(&self, id: u32) -> Result<i64, Error> {
        Ok(self.types[self.check_id(id)?])
    }

    /// The id of the token with this exact vocabulary text.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    /// Token ids for raw text. The text of a special token becomes that token. A Unicode noncharacter is an error.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        refuse_reserved(text, 0)?;
        self.encode_checked(text)
    }

    /// Token ids for text that may hold escapes. Each escape character becomes the text of its special token, which
    /// goes through BPE as plain text. A noncharacter that escapes nothing is an error.
    pub fn encode_escaped(&self, text: &str) -> Result<Vec<u32>, Error> {
        refuse_reserved(text, self.escape_text.len())?;
        self.encode_checked(text)
    }

    /// Content that is safe to place in a rendered template: the text of each special token becomes its escape
    /// character. A noncharacter in the text is an error.
    pub fn escape(&self, text: &str) -> Result<String, Error> {
        refuse_reserved(text, 0)?;
        let mut out = String::with_capacity(text.len());
        let mut plain_from = 0;
        let mut at = 0;
        while at < text.len() {
            if let Some((len, k)) = self.special_at(text, at) {
                out.push_str(&text[plain_from..at]);
                out.push(reserved_char(k));
                at += len;
                plain_from = at;
            } else {
                at += text[at..].chars().next().expect("at is inside the text").len_utf8();
            }
        }
        out.push_str(&text[plain_from..]);
        Ok(out)
    }

    /// The text that `escape` stands for: each escape character becomes the text of its special token.
    pub fn unescape(&self, text: &str) -> Result<String, Error> {
        refuse_reserved(text, self.escape_text.len())?;
        Ok(self.replace_escapes(text))
    }

    fn replace_escapes(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            match reserved_index(c) {
                Some(k) if k < self.escape_text.len() => out.push_str(&self.escape_text[k]),
                _ => out.push(c),
            }
        }
        out
    }

    /// The special token texts that `text` spells, in order of position. At one position the longest text wins.
    ///
    /// A family library reports these as control text. It tokenizes the text as content, so no hit is a token.
    pub fn control_hits(&self, text: &str) -> Vec<String> {
        let mut hits = Vec::new();
        let mut at = 0;
        while at < text.len() {
            if let Some((len, _)) = self.special_at(text, at) {
                hits.push(text[at..at + len].to_string());
                at += len;
            } else {
                at += text[at..].chars().next().expect("at is inside the text").len_utf8();
            }
        }
        hits
    }

    /// The special token whose text starts at byte `at`: the length of its text and the index of its escape.
    fn special_at(&self, text: &str, at: usize) -> Option<(usize, usize)> {
        let rest = &text.as_bytes()[at..];
        for &i in &self.special_start[rest[0] as usize] {
            let (special, id) = &self.specials_longest_first[i];
            if rest.starts_with(special.as_bytes()) {
                let k = self.special_ids.binary_search(id).expect("a special id is in the list");
                return Some((special.len(), k));
            }
        }
        None
    }

    fn encode_checked(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        let mut plain_from = 0;
        let mut at = 0;
        while at < text.len() {
            if let Some((len, k)) = self.special_at(text, at) {
                self.encode_plain(&text[plain_from..at], &mut ids)?;
                ids.push(self.special_ids[k]);
                at += len;
                plain_from = at;
            } else {
                at += text[at..].chars().next().expect("at is inside the text").len_utf8();
            }
        }
        self.encode_plain(&text[plain_from..], &mut ids)?;
        Ok(ids)
    }

    /// BPE of a run of text that holds no special token text. An escape character stands for its token's text.
    fn encode_plain(&self, chunk: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        if chunk.is_empty() {
            return Ok(());
        }
        let replaced;
        let chunk = if chunk.chars().any(|c| reserved_index(c).is_some()) {
            replaced = self.replace_escapes(chunk);
            replaced.as_str()
        } else {
            chunk
        };
        let mut word = String::new();
        let mut pair = String::new();
        for piece in pretokenize(chunk) {
            word.clear();
            word.extend(piece.bytes().map(|b| self.byte_char[b as usize]));
            self.bpe(&word, &mut pair, piece, ids)?;
        }
        Ok(())
    }

    /// Merge one byte-mapped word: apply the merge of lowest rank, leftmost on a tie, until none is left.
    fn bpe(&self, word: &str, pair: &mut String, original: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        let mut symbols: Vec<(usize, usize)> = word.char_indices().map(|(i, c)| (i, i + c.len_utf8())).collect();
        while symbols.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len() - 1 {
                pair.clear();
                pair.push_str(&word[symbols[i].0..symbols[i].1]);
                pair.push(' ');
                pair.push_str(&word[symbols[i + 1].0..symbols[i + 1].1]);
                if let Some(&rank) = self.merge_rank.get(pair.as_str())
                    && best.is_none_or(|(r, _)| rank < r)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, i)) = best else { break };
            symbols[i].1 = symbols[i + 1].1;
            symbols.remove(i + 1);
        }
        for (a, b) in symbols {
            match self.token_to_id.get(&word[a..b]) {
                Some(&id) => ids.push(id),
                None => {
                    return Err(Error::Vocab(format!(
                        "BPE produced the piece {:?}, which is not in the vocabulary (from the word {original:?}). The \
                         merges and the vocabulary do not fit.",
                        &word[a..b]
                    )));
                }
            }
        }
        Ok(())
    }

    /// The text for a list of ids. Special tokens give their own text, and the other tokens give their bytes. The
    /// bytes must make valid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let mut bytes = Vec::new();
        for &id in ids {
            let i = self.check_id(id)?;
            let text = &self.tokens[i];
            if self.types[i] == TYPE_CONTROL || self.types[i] == TYPE_USER_DEFINED {
                bytes.extend_from_slice(text.as_bytes());
            } else {
                self.push_bytes(id, text, &mut bytes)?;
            }
        }
        String::from_utf8(bytes).map_err(|e| Error::Bytes(format!("the tokens do not make UTF-8 text ({e})")))
    }

    fn push_bytes(&self, id: u32, text: &str, out: &mut Vec<u8>) -> Result<(), Error> {
        for c in text.chars() {
            match self.char_byte.get(&c) {
                Some(&b) => out.push(b),
                None => {
                    return Err(Error::Bytes(format!(
                        "token {id} ({text:?}) has the character {c:?}, which is not in the GPT-2 byte table"
                    )));
                }
            }
        }
        Ok(())
    }

    /// One token's bytes, as llama.cpp's `token_to_piece` gives them. Control and unknown tokens are empty unless
    /// `special` is set. User-defined tokens are their text. Normal tokens are their decoded bytes. Other types are
    /// empty.
    pub fn token_piece(&self, id: u32, special: bool) -> Result<Vec<u8>, Error> {
        let i = self.check_id(id)?;
        let text = &self.tokens[i];
        match self.types[i] {
            TYPE_CONTROL | TYPE_UNKNOWN => Ok(if special { text.as_bytes().to_vec() } else { Vec::new() }),
            TYPE_USER_DEFINED => Ok(text.as_bytes().to_vec()),
            TYPE_NORMAL => {
                let mut out = Vec::with_capacity(text.len());
                self.push_bytes(id, text, &mut out)?;
                Ok(out)
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Refuse a noncharacter. The characters with an index below `allowed` are escapes and pass.
fn refuse_reserved(text: &str, allowed: usize) -> Result<(), Error> {
    for (offset, ch) in text.chars().enumerate() {
        if let Some(k) = reserved_index(ch)
            && k >= allowed
        {
            return Err(Error::Reserved { ch, offset });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- the pre-tokenizer pattern

/// Split `text` into the words that mk's `qwen35` pattern gives. The pattern is, in order:
///
/// 1. a contraction: an apostrophe and `s`, `t`, `re`, `ve`, `m`, `ll` or `d`, in any case;
/// 2. one character that is not CR, LF, a letter or a number, then one or more letters or marks;
/// 3. one number;
/// 4. a space or none, one or more characters that are not space, letter, mark or number, then CR and LF;
/// 5. white space that ends at its last CR or LF;
/// 6. white space, but not the last character of it when a non-space follows;
/// 7. white space.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<char> = text.chars().collect();
    let mut offsets = Vec::with_capacity(chars.len() + 1);
    let mut at = 0;
    for c in &chars {
        offsets.push(at);
        at += c.len_utf8();
    }
    offsets.push(at);
    let mut words = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = word_end(&chars, i);
        assert!(
            end > i,
            "the pattern matched nothing at character {i}: a defect in the pre-tokenizer"
        );
        words.push(&text[offsets[i]..offsets[end]]);
        i = end;
    }
    words
}

fn is_crlf(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn is_letter_or_mark(c: char) -> bool {
    is_letter(c) || is_mark(c)
}

/// A character of the class `[^\s\p{L}\p{M}\p{N}]`.
fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_mark(c) && !is_number(c)
}

fn run(chars: &[char], from: usize, class: impl Fn(char) -> bool) -> usize {
    from + chars[from..].iter().take_while(|&&c| class(c)).count()
}

/// The end of the word that starts at character `i`.
fn word_end(chars: &[char], i: usize) -> usize {
    let c = chars[i];
    // 1. A contraction.
    if c == '\'' {
        let rest = |k: usize| chars.get(i + k).copied();
        match rest(1) {
            Some('s' | 'S' | 't' | 'T' | 'm' | 'M' | 'd' | 'D') => return i + 2,
            Some('r' | 'R') if matches!(rest(2), Some('e' | 'E')) => return i + 3,
            Some('v' | 'V') if matches!(rest(2), Some('e' | 'E')) => return i + 3,
            Some('l' | 'L') if matches!(rest(2), Some('l' | 'L')) => return i + 3,
            _ => {}
        }
    }
    // 2. An optional first character, then letters and marks.
    if !is_crlf(c) && !is_letter(c) && !is_number(c) && chars.get(i + 1).is_some_and(|&n| is_letter_or_mark(n)) {
        return run(chars, i + 1, is_letter_or_mark);
    }
    if is_letter_or_mark(c) {
        return run(chars, i, is_letter_or_mark);
    }
    // 3. One number.
    if is_number(c) {
        return i + 1;
    }
    // 4. Other characters, with an optional space before them, and line ends after them.
    let start = if c == ' ' && chars.get(i + 1).is_some_and(|&n| is_other(n)) {
        i + 1
    } else {
        i
    };
    if is_other(chars[start]) {
        return run(chars, run(chars, start, is_other), is_crlf);
    }
    // 5 to 7. White space. Here `c` is white space, because every other character matched above.
    let end = run(chars, i, is_space);
    if let Some(last) = chars[i..end].iter().rposition(|&c| is_crlf(c)) {
        return i + last + 1;
    }
    if end == chars.len() {
        return end;
    }
    if end - 1 > i {
        return end - 1;
    }
    end
}
