use std::fmt;

/// What can fail in this library. Each message says what failed and what the reader can do.
#[derive(Debug)]
pub enum Error {
    /// A file of the checkpoint could not be read.
    Io { path: String, source: std::io::Error },
    /// A file of the checkpoint does not have the keys or the values that this family needs.
    Checkpoint(String),
    /// The `tokenizers` crate refused the vocabulary or the merges, or failed to encode.
    Tokenizer(String),
    /// The language of a request is not `auto` and not a language of the checkpoint.
    Language { given: String, known: Vec<String> },
    /// The token ids of the assistant text do not have the form of the template. This is a defect of the library.
    Prompt(String),
    /// Item `item` of a speech context breaks a rule of its order (spec 11.8a).
    Context { item: usize, message: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Error::Checkpoint(m) => write!(f, "{m}. Use a Qwen3-TTS 12Hz VoiceDesign checkpoint."),
            Error::Tokenizer(m) => write!(f, "the tokenizer failed: {m}"),
            Error::Language { given, known } => write!(
                f,
                "the language {given:?} is not known. Use one of these: {}.",
                known.join(", ")
            ),
            Error::Prompt(m) => write!(f, "{m}. Report this defect of nagaya-family-qwen3tts."),
            Error::Context { item, message } => write!(f, "item {item} of the speech context: {message}"),
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
