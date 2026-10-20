//! JSON that keeps the order of keys.
//!
//! The chat template shows the order of keys in a tool schema, so the order is part of the result. This reader keeps
//! it. It refuses what Python's `json` module accepts and the template cannot show: `NaN`, `Infinity` and an integer of
//! more than 128 bits.

use crate::Error;

/// A JSON value. An object is a list of pairs in the order of the text.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

/// How deep arrays and objects may nest. Python stops at about 1000, and a deep text must not overflow the stack.
const MAX_DEPTH: usize = 512;

impl Json {
    /// Read one JSON value. Only whitespace may follow it.
    pub fn parse(text: &str) -> Result<Json, Error> {
        let mut p = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        p.skip_space();
        let value = p.value(0)?;
        p.skip_space();
        if p.at != p.bytes.len() {
            return Err(p.fail("text follows the value"));
        }
        Ok(value)
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn fail(&self, what: &str) -> Error {
        Error::Json(format!("{what}, at byte {}", self.at))
    }

    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn word(&mut self, word: &str, value: Json) -> Result<Json, Error> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.fail("this is not a JSON value"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, Error> {
        match self.peek() {
            None => Err(self.fail("the text ends where a value should be")),
            Some(b'n') => self.word("null", Json::Null),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.fail("this is not a JSON value")),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, Error> {
        if depth >= MAX_DEPTH {
            return Err(self.fail("the nesting is too deep"));
        }
        self.at += 1;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_space();
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.fail("an array needs a comma or a closing bracket")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, Error> {
        if depth >= MAX_DEPTH {
            return Err(self.fail("the nesting is too deep"));
        }
        self.at += 1;
        let mut pairs: Vec<(String, Json)> = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(pairs));
        }
        loop {
            self.skip_space();
            if self.peek() != Some(b'"') {
                return Err(self.fail("an object key must be a string"));
            }
            let key = self.string()?;
            self.skip_space();
            if self.peek() != Some(b':') {
                return Err(self.fail("an object key needs a colon"));
            }
            self.at += 1;
            self.skip_space();
            let value = self.value(depth + 1)?;
            // A repeated key keeps its first place and its last value, as in Python.
            match pairs.iter_mut().find(|(k, _)| *k == key) {
                Some(pair) => pair.1 = value,
                None => pairs.push((key, value)),
            }
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(pairs));
                }
                _ => return Err(self.fail("an object needs a comma or a closing brace")),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.at;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        self.at - start
    }

    fn number(&mut self) -> Result<Json, Error> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.at += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.fail("a number must not start with a zero digit"));
                }
            }
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.fail("a number needs a digit")),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.at += 1;
            if self.digits() == 0 {
                return Err(self.fail("a fraction needs a digit"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if self.digits() == 0 {
                return Err(self.fail("an exponent needs a digit"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).expect("digits are ASCII");
        if is_float {
            let f: f64 = text.parse().expect("the grammar above gives a valid float");
            if !f.is_finite() {
                return Err(Error::Json(format!(
                    "the number {text} is not a finite float, at byte {start}"
                )));
            }
            Ok(Json::Float(f))
        } else {
            text.parse::<i128>()
                .map(Json::Int)
                .map_err(|_| Error::Json(format!("the integer {text} has more than 128 bits, at byte {start}")))
        }
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let end = self.at + 4;
        let digits = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| self.fail("a \\u escape needs four digits"))?;
        let text = std::str::from_utf8(digits).map_err(|_| self.fail("a \\u escape needs four hex digits"))?;
        if !text.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(self.fail("a \\u escape needs four hex digits"));
        }
        self.at = end;
        Ok(u32::from_str_radix(text, 16).expect("four hex digits"))
    }

    fn string(&mut self) -> Result<String, Error> {
        self.at += 1;
        let mut out = String::new();
        loop {
            // Copy the run of plain bytes up to the next quote, backslash or control byte.
            let start = self.at;
            while let Some(&b) = self.bytes.get(self.at) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.at += 1;
            }
            out.push_str(std::str::from_utf8(&self.bytes[start..self.at]).expect("the input is a str"));
            match self.peek() {
                None => return Err(self.fail("the string has no closing quote")),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = self.peek().ok_or_else(|| self.fail("the string ends in a backslash"))?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                if !self.bytes[self.at..].starts_with(b"\\u") {
                                    return Err(self.fail("a high surrogate needs a low surrogate after it"));
                                }
                                self.at += 2;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err(self.fail("a high surrogate needs a low surrogate after it"));
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&first) {
                                return Err(self.fail("a low surrogate has no high surrogate before it"));
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).expect("a surrogate pair gives a scalar value"));
                        }
                        _ => return Err(self.fail("this escape is not JSON")),
                    }
                }
                Some(_) => return Err(self.fail("a control character must be an escape in a string")),
            }
        }
    }
}
