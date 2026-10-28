//! Why a council operation did not give a result. The routes map each kind to a status (spec 11.5).

use gogi_proto::Invalid;
use nagaya_council::family::FamilyError;
use nagaya_ledger::LedgerError;
use nagaya_telemetry::EngineState;

/// Why a service call failed.
#[derive(Clone, Debug, PartialEq)]
pub enum ServiceError {
    /// The request breaks the contract. The route answers 400.
    Invalid(Invalid),
    /// An unknown spec, snapshot or pin owner. The route answers 404.
    NotFound(String),
    /// A request names an object that the server does not hold. The route answers 404, and `param` names the member.
    NotHeld {
        /// The member that names the object, such as `messages[2].content[0]`.
        param: String,
        /// What is missing, and what the client can do.
        message: String,
    },
    /// A request is over a limit. `param` names the limit. The route answers 413 `too_large`.
    TooLarge {
        /// The limit.
        param: String,
        /// What is over it, and what the client can do.
        message: String,
    },
    /// The prompt leaves no room for a reply. The route answers 413 `context_full`.
    ContextFull {
        /// The tokens of the prompt.
        prompt_tokens: u64,
        /// The tokens that the served model holds.
        context_tokens: u64,
    },
    /// All slots are in use. The route answers 429 `busy`.
    Busy(String),
    /// The ledger refused the charge of a context (spec 4.3). Nothing changed.
    ContextRefused(LedgerError),
    /// The process is stopping. The route answers 503 `unavailable`.
    Stopping,
    /// The engine of the served model is not `ready`, so nothing ran (spec 3.3). The route answers 503
    /// `model_unavailable`.
    Unavailable {
        /// The name of the engine in the profile.
        engine: String,
        /// The state of the engine.
        state: EngineState,
    },
    /// The job ended before it gave a result: a stop cancelled it, or its client went away.
    Cancelled,
    /// The deadline of the request passed before the job ended (spec 5.8). The route answers 504 `timeout`.
    Timeout,
    /// A defect or a failed engine call. The route answers 500, and the process stops (spec 8.5, 11.5).
    Internal(String),
}

impl ServiceError {
    /// An internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        ServiceError::Internal(message.into())
    }

    /// The 413 `too_large` of a build or a raw read that needs more than `context_tokens` tokens. `context_full` is
    /// only for the prompt of a generation (gogi "Errors").
    pub fn past_context(tokens: usize, context_tokens: usize) -> Self {
        ServiceError::TooLarge {
            param: "context_tokens".to_owned(),
            message: format!(
                "nothing ran: the request needs {tokens} tokens, and the served model holds {context_tokens}; \
                 shorten the context"
            ),
        }
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::Invalid(invalid) => write!(formatter, "{}", invalid.message),
            ServiceError::NotFound(message) | ServiceError::NotHeld { message, .. } => write!(formatter, "{message}"),
            ServiceError::TooLarge { message, .. } => write!(formatter, "{message}"),
            ServiceError::ContextFull {
                prompt_tokens,
                context_tokens,
            } => write!(
                formatter,
                "the prompt has {prompt_tokens} tokens and leaves no room in {context_tokens} tokens; shorten the context"
            ),
            ServiceError::Busy(message) => write!(formatter, "{message}"),
            ServiceError::ContextRefused(error) => write!(formatter, "{error}"),
            ServiceError::Stopping => write!(formatter, "the process is stopping"),
            ServiceError::Unavailable { engine, state } => write!(
                formatter,
                "nothing ran: engine `{engine}` is {}, not ready; load it, then retry",
                state.as_str()
            ),
            ServiceError::Cancelled => write!(formatter, "the job was cancelled before it gave a result"),
            ServiceError::Timeout => write!(
                formatter,
                "nothing was returned: the request passed its timeout_ms; send a longer timeout_ms, or fewer contexts"
            ),
            ServiceError::Internal(message) => write!(formatter, "internal error: {message}"),
        }
    }
}

impl std::error::Error for ServiceError {}

impl From<FamilyError> for ServiceError {
    fn from(error: FamilyError) -> Self {
        match error {
            FamilyError::Client(invalid) => ServiceError::Invalid(invalid),
            FamilyError::Internal(message) => ServiceError::Internal(message),
        }
    }
}

impl From<Invalid> for ServiceError {
    fn from(invalid: Invalid) -> Self {
        ServiceError::Invalid(invalid)
    }
}
