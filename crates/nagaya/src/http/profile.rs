//! The profile routes (spec 3.3, "The profile routes"): `GET` and `PUT /nagaya/v1/profile`.
//!
//! The main loop owns the active profile, so each route sends it a call and waits for the answer on a call thread.

use std::sync::Arc;
use std::sync::mpsc::{self, Sender};

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use gogi_proto::error::ErrorType;
use serde_json::{Value, json};

use super::errors::Failure;
use super::routes::{bytes, json};
use super::{Council, RequestId};
use crate::council::error::ServiceError;
use crate::gossip::Key;
use crate::profile::PlacementProfile;
use crate::reprofile::{Desire, DesireAnswer, ProfileCall, PutAnswer};
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

/// A failure that names a query parameter.
fn named(status: StatusCode, kind: ErrorType, param: &str, message: String) -> Failure {
    let mut failure = Failure::new(status, kind, message);
    failure.body.error.param = Some(param.to_owned());
    failure
}

/// The 404 of a host that no live node has (spec 2.4).
fn unknown_host(host: &str) -> Failure {
    named(
        StatusCode::NOT_FOUND,
        ErrorType::NotFound,
        "host",
        format!("nothing changed: no live node has the name `{host}`; GET /nagaya/v1/cluster lists the nodes"),
    )
}

/// The other host that the query `host` names, or `None` for this node (spec 2.4, "The routes of a profile for another
/// host").
fn other_host(council: &Council, uri: &Uri) -> Option<String> {
    let host = super::peers::query_value(uri, "host")?;
    let own = council.cluster().map(|cluster| cluster.own().to_owned());
    (own.as_deref() != Some(host)).then(|| host.to_owned())
}

async fn read(State(council): Shared, Extension(id): Id, uri: Uri) -> Response {
    if let Some(host) = other_host(&council, &uri) {
        // The `profile:active` of a live node, from gossip (spec 2.4).
        let state = council.cluster().map(|cluster| cluster.state());
        let active = state.as_ref().and_then(|state| {
            state
                .view()
                .live
                .iter()
                .find(|node| node.name == host)
                .map(|node| node.value(&Key::ProfileActive))
        });
        return match active {
            Some(Some(profile)) => json(StatusCode::OK, &profile),
            Some(None) => named(
                StatusCode::NOT_FOUND,
                ErrorType::NotFound,
                "host",
                format!("node `{host}` has published no active profile yet; try again after its start"),
            )
            .respond(&id),
            None => unknown_host(&host).respond(&id),
        };
    }
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

async fn change(State(council): Shared, Extension(id): Id, uri: Uri, body: Body) -> Response {
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
    let version = super::peers::query_value(&uri, "version");
    if let Some(host) = other_host(&council, &uri) {
        return desire(&council, calls, host, version, &body, &id).await;
    }
    if version.is_some() {
        return named(
            StatusCode::BAD_REQUEST,
            ErrorType::InvalidRequest,
            "version",
            "nothing changed: a PUT for this node takes no version; send the version with the host of another node"
                .to_owned(),
        )
        .respond(&id);
    }
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
        Ok(PutAnswer::Stopping) => Failure::from(ServiceError::Stopping)
            .with_message("nothing changed: the process is stopping; send the profile after its start")
            .respond(&id),
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

/// Publish a desired profile for another host in `profile:desired:<host>` (spec 2.4, "The routes of a profile for
/// another host"). The main loop owns the gossip, so it checks the host and the version and writes the key. The target
/// checks the profile, so this node checks only that the body is a JSON object.
async fn desire(
    council: &Council,
    calls: Sender<ProfileCall>,
    host: String,
    version: Option<&str>,
    body: &[u8],
    id: &RequestId,
) -> Response {
    let version = match version.map(str::parse::<u64>) {
        Some(Ok(version)) if version > 0 => version,
        _ => {
            return named(
                StatusCode::BAD_REQUEST,
                ErrorType::InvalidRequest,
                "version",
                format!("nothing changed: a PUT for node `{host}` needs `version`, a whole number above 0"),
            )
            .respond(id);
        }
    };
    let profile: Value = match serde_json::from_slice(body) {
        Ok(profile @ Value::Object(_)) => profile,
        _ => return invalid("the body is not a JSON object; send the profile as JSON".to_owned()).respond(id),
    };
    let target = host.clone();
    let result = council
        .call(move || {
            let (reply, answer) = mpsc::channel();
            let desire = Desire {
                host: target,
                version,
                profile,
                reply,
            };
            calls.send(ProfileCall::Desire(desire)).map_err(|_| stopped())?;
            answer.recv().map_err(|_| stopped())
        })
        .await;
    match result {
        Err(failure) => failure.respond(id),
        Ok(DesireAnswer::Published) => json(StatusCode::ACCEPTED, &json!({"host": host, "version": version})),
        Ok(DesireAnswer::UnknownHost) => unknown_host(&host).respond(id),
        Ok(DesireAnswer::Conflict { highest }) => conflict(
            id,
            format!(
                "nothing changed: node `{host}` has version {highest} of its desired profile; send a higher version"
            ),
        ),
        Ok(DesireAnswer::TooLarge(message)) => named(
            StatusCode::PAYLOAD_TOO_LARGE,
            ErrorType::TooLarge,
            "profile",
            format!("nothing changed: {message}"),
        )
        .respond(id),
    }
}

/// The 409 `conflict` of a profile `PUT` with an old version (spec 11.5). gogi has no such type, so the body is written
/// here, in the shape of gogi's error body.
fn conflict(id: &RequestId, message: String) -> Response {
    let body = json!({"error": {"type": "conflict", "message": message, "param": "version"}, "request_id": id.0});
    (StatusCode::CONFLICT, axum::Json(body)).into_response()
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
