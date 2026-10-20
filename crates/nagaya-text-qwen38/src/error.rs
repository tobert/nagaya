use std::fmt;

/// What can fail in this library. Each message says what failed and what the reader can do.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read.
    Io { path: String, source: std::io::Error },
    /// The bytes are not GGUF metadata that this library reads.
    Gguf(String),
    /// A metadata key is missing, has the wrong type, or holds a value that this family does not support.
    Metadata(String),
    /// JSON text that this library does not read.
    Json(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Error::Gguf(m) => write!(f, "not readable GGUF metadata: {m}"),
            Error::Metadata(m) => write!(f, "unusable GGUF metadata: {m}"),
            Error::Json(m) => write!(f, "not readable JSON: {m}"),
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
