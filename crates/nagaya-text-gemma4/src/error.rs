use std::fmt;

/// What can fail in this library. Each message says what failed and what the reader can do.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read.
    Io { path: String, source: std::io::Error },
    /// The `tokenizers` crate refused the tokenizer file, or failed to encode or decode.
    Tokenizer(String),
    /// Text holds a Unicode noncharacter. The library keeps those for its escapes of special tokens.
    Reserved { ch: char, offset: usize },
    /// A token id is not in the vocabulary.
    TokenId { id: u32, count: usize },
    /// The chat template refused the messages, or failed to run.
    Template(String),
    /// JSON that the template cannot take.
    Json(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Error::Tokenizer(m) => write!(f, "the tokenizer failed: {m}"),
            Error::Reserved { ch, offset } => write!(
                f,
                "U+{:04X} at offset {offset} is a Unicode noncharacter. The library keeps it for escapes. Remove it from the text.",
                *ch as u32
            ),
            Error::TokenId { id, count } => write!(
                f,
                "token id {id} is not in the vocabulary of {count} tokens. Use an id below {count}."
            ),
            Error::Template(m) => write!(f, "the chat template refused these messages: {m}"),
            Error::Json(m) => write!(f, "not usable JSON: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<nagaya_jinja::Error> for Error {
    fn from(e: nagaya_jinja::Error) -> Self {
        match e {
            nagaya_jinja::Error::Json(m) => Error::Json(m),
            nagaya_jinja::Error::Template(m) => Error::Template(m),
        }
    }
}
