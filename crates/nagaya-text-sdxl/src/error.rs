use std::fmt;

/// What can fail in this library. Each message says what failed and what the reader can do.
#[derive(Debug)]
pub enum Error {
    /// A file of the checkpoint could not be read.
    Io { path: String, source: std::io::Error },
    /// The `tokenizers` crate refused the vocabulary or the merges, or failed to encode.
    Tokenizer(String),
    /// The prompt has more tokens than the chunks hold, or more chunks than `MAX_CHUNKS`.
    TooLong { tokens: usize, chunks: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Error::Tokenizer(m) => write!(f, "the tokenizer failed: {m}"),
            Error::TooLong { tokens, chunks } => write!(
                f,
                "the prompt has {tokens} tokens in {chunks} chunks, and a prompt holds {} chunks of {} tokens; \
                 shorten it, or remove a BREAK",
                crate::MAX_CHUNKS,
                crate::CHUNK_TOKENS
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
