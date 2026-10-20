//! The detokenizer: token bytes to text, as the stream of tokens arrives.
//!
//! These are ports of `Utf8Stream` in mk's `service/chatfmt.py` and `piece_text` in `service/common.py`.

/// Turns the bytes of tokens into text. A character split across tokens comes out whole with the token that completes
/// it. A byte that cannot start or continue a character becomes U+FFFD. Nothing fails in the middle of a stream.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    /// The start of a character that the last push did not finish. It is at most three bytes.
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream::default()
    }

    /// Add bytes. The result is the text that is complete now.
    pub fn push(&mut self, raw: &[u8]) -> String {
        let mut bytes = std::mem::take(&mut self.pending);
        bytes.extend_from_slice(raw);
        let mut out = String::new();
        let mut rest = bytes.as_slice();
        loop {
            match std::str::from_utf8(rest) {
                Ok(text) => {
                    out.push_str(text);
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(valid).expect("the bytes up to the error are UTF-8"));
                    match e.error_len() {
                        // The bytes end inside a character. Keep them until the next push.
                        None => {
                            self.pending = after.to_vec();
                            break;
                        }
                        // Python keeps 0xED and a second byte of 0xA0 to 0xBF too: its decoder tests the range of a
                        // surrogate only when the third byte comes. The same timing keeps the pieces of a stream equal.
                        Some(1) if after.len() == 2 && after[0] == 0xED && (0xA0..=0xBF).contains(&after[1]) => {
                            self.pending = after.to_vec();
                            break;
                        }
                        Some(n) => {
                            out.push('\u{fffd}');
                            rest = &after[n..];
                        }
                    }
                }
            }
        }
        out
    }

    /// End the stream. Bytes of an unfinished character become U+FFFD, and the result is empty otherwise.
    pub fn flush(&mut self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned()
    }
}

/// A token's bytes as llama-server shows them in JSON.
///
/// The server drops a multi-byte character that is cut off at the end of the bytes. It replaces any other byte that is
/// not UTF-8 with U+FFFD. (llama.cpp `validate_utf8` in `tools/server/server-common.cpp`.)
pub fn piece_text(raw: &[u8]) -> String {
    let mut n = raw.len();
    for i in 1..=n.min(4) {
        let c = raw[n - i];
        if (c & 0xE0) == 0xC0 && i < 2 || (c & 0xF0) == 0xE0 && i < 3 || (c & 0xF8) == 0xF0 && i < 4 {
            n -= i;
            break;
        }
    }
    String::from_utf8_lossy(&raw[..n]).into_owned()
}
