use std::fmt;

/// What can fail in this library. Each message says what failed and what the reader can do.
#[derive(Debug)]
pub enum Error {
    /// A file of the checkpoint could not be read.
    Io { path: String, source: std::io::Error },
    /// The `tokenizers` crate refused `tokenizer.json`, or failed to encode.
    Tokenizer(String),
    /// The prompt holds the text of a special token (spec 11.9c).
    SpecialToken { token: String },
    /// The template with the prompt has more ids than the most of spec 11.9c.
    TooLong { ids: usize, max: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Error::Tokenizer(m) => write!(
                f,
                "the tokenizer failed: {m}. Use processor/tokenizer.json of the Qwen-Image-2.1 checkpoint."
            ),
            Error::SpecialToken { token } => write!(
                f,
                "the prompt holds the text of the special token {token}. Remove it from the prompt."
            ),
            Error::TooLong { ids, max } => write!(
                f,
                "the prompt gives {ids} ids with its template, and the most is {max}. Make the prompt shorter."
            ),
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
