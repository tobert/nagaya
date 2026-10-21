use std::fmt;

/// What can fail in the interpreter. Each message says what failed and what the reader can do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// JSON text that the interpreter does not read.
    Json(String),
    /// The chat template refused its variables, or failed to run.
    Template(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Json(m) => write!(f, "not readable JSON: {m}"),
            Error::Template(m) => write!(f, "the chat template refused these messages: {m}"),
        }
    }
}

impl std::error::Error for Error {}
