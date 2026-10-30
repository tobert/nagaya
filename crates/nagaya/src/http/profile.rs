//! The profile routes (spec 3.3, "The profile routes"): `GET` and `PUT /nagaya/v1/profile`.
//!
//! The main loop owns the active profile, so each route sends it a call and waits for the answer on a call thread.

use std::sync::Arc;
use std::sync::mpsc::{self, Sender};

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use gogi_proto::error::ErrorType;
use serde_json::{Value, json};

use super::errors::Failure;
use super::routes::{bytes, json};
use super::{Council, RequestId};
use crate::council::error::ServiceError;
use crate::profile::PlacementProfile;
use crate::reprofile::{ProfileCall, PutAnswer};
use crate::swap::Shortfall;

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The profile routes. The caller adds the fallbacks.
pub(super) fn routes() -> Router<Arc<Council>> {
    Router::new().route("/nagaya/v1/profile", get(read).put(change))
}

/// The main loop, or the 404 of a council that has none.
fn main_loop(council: &Council) -> Result<Sender<ProfileCall>, Failure> {
    council.profile_calls().ok_or_else(|| {
        Failure::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            "this server has no profile; the profile routes need a process",
        )
    })
}

/// The failure of a main loop that stopped before it answered.
fn stopped() -> ServiceError {
    ServiceError::Stopping
}

async fn read(State(council): Shared, Extension(id): Id) -> Response {
    let calls = match main_loop(&council) {
        Ok(calls) => calls,
        Err(failure) => return failure.respond(&id),
    };
    let result = council
        .call(move || {
            let (reply, answer) = mpsc::channel();
            calls.send(ProfileCall::Get(reply)).map_err(|_| stopped())?;
            answer.recv().map_err(|_| stopped())
        })
        .await;
    match result {
        Ok(profile) => json(StatusCode::OK, &profile),
        Err(failure) => failure.respond(&id),
    }
}

/// A 400 that names `profile`.
fn invalid(message: String) -> Failure {
    let mut failure = Failure::new(
        StatusCode::BAD_REQUEST,
        ErrorType::InvalidRequest,
        format!("nothing changed: {message}"),
    );
    failure.body.error.param = Some("profile".to_owned());
    failure
}

async fn change(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let calls = match main_loop(&council) {
        Ok(calls) => calls,
        Err(failure) => return failure.respond(&id),
    };
    if council.draining() {
        return Failure::from(ServiceError::Stopping)
            .with_message("nothing changed: the process is stopping; send the profile after its start")
            .respond(&id);
    }
    let body = match bytes(body).await {
        Ok(body) => body,
        Err(failure) => return failure.respond(&id),
    };
    let profile: PlacementProfile = match serde_json::from_slice(&body) {
        Ok(profile) => profile,
        Err(error) => return invalid(format!("the profile does not decode: {error}")).respond(&id),
    };
    let result = council
        .call(move || {
            let (reply, answer) = mpsc::channel();
            calls.send(ProfileCall::Put(profile, reply)).map_err(|_| stopped())?;
            answer.recv().map_err(|_| stopped())
        })
        .await;
    match result {
        Err(failure) => failure.respond(&id),
        Ok(PutAnswer::Unchanged(change)) => json(StatusCode::OK, &change),
        Ok(PutAnswer::Accepted(change)) => json(StatusCode::ACCEPTED, &change),
        Ok(PutAnswer::Invalid(message)) => invalid(message).respond(&id),
        Ok(PutAnswer::Refused { reason, shortfall }) => refused(&id, &reason, &shortfall),
        Ok(PutAnswer::Busy) => {
            let mut failure = Failure::new(
                StatusCode::CONFLICT,
                ErrorType::InvalidRequest,
                "nothing changed: a swap or a profile change runs now; send the profile again when it ends",
            );
            failure.body.error.param = Some("profile".to_owned());
            failure.respond(&id)
        }
    }
}

/// The 507 of a profile that does not fit: the error body, with the shortfall of each short budget (spec 2.4).
fn refused(id: &RequestId, reason: &str, shortfall: &[Shortfall]) -> Response {
    let mut failure = Failure::new(
        StatusCode::INSUFFICIENT_STORAGE,
        ErrorType::TooLarge,
        format!("nothing changed: {reason}; the shortfall gives the bytes that each budget lacks"),
    );
    failure.body.error.param = Some("profile".to_owned());
    let mut body = serde_json::to_value((*failure.body).clone().with_request_id(id.0.clone()))
        .expect("an error body is JSON data");
    let entries: Vec<Value> = shortfall
        .iter()
        .map(|entry| {
            let tiers: Vec<String> = entry.tiers.iter().map(ToString::to_string).collect();
            json!({"tiers": tiers, "bytes": entry.bytes})
        })
        .collect();
    body["shortfall"] = Value::Array(entries);
    (failure.status, axum::Json(body)).into_response()
}
