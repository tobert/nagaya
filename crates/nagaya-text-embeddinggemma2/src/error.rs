use std::fmt;

/// What can fail in this library. Each message says what failed and what the reader can do.
#[derive(Debug)]
pub enum Error {
    /// A file of the checkpoint could not be read.
    Io { path: String, source: std::io::Error },
    /// The `tokenizers` crate refused `tokenizer.json`, or failed to encode.
    Tokenizer(String),
    /// `config_sentence_transformers.json` does not decode, or has no `prompts` object.
    Config(String),
    /// The prompts of the model lack the prompt of a task of spec 11.10.
    Prompt { task: &'static str, name: &'static str },
    /// The request names a task that the family does not have.
    Task { task: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Error::Tokenizer(m) => write!(f, "the tokenizer failed: {m}"),
            Error::Config(m) => write!(
                f,
                "config_sentence_transformers.json is not usable: {m}. Use the file of an EmbeddingGemma 2 checkpoint."
            ),
            Error::Prompt { task, name } => write!(
                f,
                "config_sentence_transformers.json has no prompt {name:?} for the task {task}. Use the file of an \
                 EmbeddingGemma 2 checkpoint."
            ),
            Error::Task { task } => {
                let tasks: Vec<&str> = crate::TASKS.iter().map(|(task, _)| *task).collect();
                write!(f, "the model has no task {task:?}; its tasks are {}", tasks.join(", "))
            }
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
