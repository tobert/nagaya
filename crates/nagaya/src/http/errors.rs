//! gogi's error body for each failure (spec 11.5, gogi "Errors").

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use gogi_proto::error::{ErrorBody, ErrorType};
use gogi_proto::identity::ServedModel;

use super::RequestId;
use crate::council::error::ServiceError;

/// A request that failed: the status and the error body, without the request id.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The HTTP status.
    pub status: StatusCode,
    /// The error body.
    pub body: Box<ErrorBody>,
}

impl Failure {
    /// A failure of one status and type.
    pub fn new(status: StatusCode, r#type: ErrorType, message: impl Into<String>) -> Self {
        Self {
            status,
            body: Box::new(ErrorBody::new(r#type, message)),
        }
    }

    /// The 503 of a served model that is not `ready`.
    pub fn unavailable(model: &ServedModel) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: Box::new(ErrorBody::unavailable(model)),
        }
    }

    /// The same failure with another message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.body.error.message = message.into();
        self
    }

    /// The response, with the request id in the body.
    pub fn respond(self, id: &RequestId) -> Response {
        let body = (*self.body).with_request_id(id.0.clone());
        (self.status, axum::Json(body)).into_response()
    }
}

impl From<ServiceError> for Failure {
    /// Each service error maps to one row of gogi's error table.
    ///
    /// A refused ledger charge is a 429 `busy`, because the server has more requests than it runs. A job that a
    /// stop cancelled is a 503 `unavailable`, because the server is stopping. An internal error is a 500.
    fn from(error: ServiceError) -> Self {
        let message = error.to_string();
        match error {
            ServiceError::Invalid(invalid) => Self {
                status: StatusCode::BAD_REQUEST,
                body: Box::new(ErrorBody::from(invalid)),
            },
            ServiceError::NotFound(_) => Self::new(StatusCode::NOT_FOUND, ErrorType::NotFound, message),
            ServiceError::NotHeld { param, .. } => {
                let mut failure = Self::new(StatusCode::NOT_FOUND, ErrorType::NotFound, message);
                failure.body.error.param = Some(param);
                failure
            }
            ServiceError::TooLarge { param, .. } => {
                let mut failure = Self::new(StatusCode::PAYLOAD_TOO_LARGE, ErrorType::TooLarge, message);
                failure.body.error.param = Some(param);
                failure
            }
            ServiceError::ContextFull {
                prompt_tokens,
                context_tokens,
            } => {
                let mut failure = Self::new(StatusCode::PAYLOAD_TOO_LARGE, ErrorType::ContextFull, message);
                failure.body.error.prompt_tokens = Some(prompt_tokens);
                failure.body.error.context_tokens = Some(context_tokens);
                failure
            }
            ServiceError::Busy(_) | ServiceError::ContextRefused(_) => {
                Self::new(StatusCode::TOO_MANY_REQUESTS, ErrorType::Busy, message)
            }
            ServiceError::Stopping | ServiceError::Cancelled => {
                Self::new(StatusCode::SERVICE_UNAVAILABLE, ErrorType::Unavailable, message)
            }
            ServiceError::Timeout => Self::new(StatusCode::GATEWAY_TIMEOUT, ErrorType::Timeout, message),
            ServiceError::Internal(_) => Self::new(StatusCode::INTERNAL_SERVER_ERROR, ErrorType::Internal, message),
        }
    }
}

/// Decode a request body. A missing or unknown member, or a `continue` that is not a boolean, is a 400 that names it
/// in `param` (gogi "Errors", "Continuations").
pub fn parse<T: serde::de::DeserializeOwned>(body: &[u8], what: &str) -> Result<T, Failure> {
    serde_json::from_slice(body).map_err(|error| {
        let text = error.to_string();
        let field = gogi_proto::error::decode_param(&text);
        let message = format!("nothing ran: the {what} body does not decode: {text}");
        let invalid = match field {
            Some(field) => gogi_proto::Invalid::param(field, message),
            None => gogi_proto::Invalid::new(message),
        };
        Failure::from(ServiceError::Invalid(invalid))
    })
}
