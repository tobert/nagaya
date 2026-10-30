//! The load routes (spec 3.3b): `POST /nagaya/v1/engines/{name}/load` and `POST /nagaya/v1/engines/{name}/unload`.
//!
//! A route asks the group thread of the engine to plan a swap. The answer comes when the group plans it, not when the
//! swap ends. The status feed and identity show the states that follow (spec 9.4, 11.1).

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::post;
use gogi_proto::error::ErrorType;
use serde::{Deserialize, Serialize};

use super::errors::Failure;
use super::routes::json;
use super::{Council, RequestId};
use crate::process::{LoadError, LoadReply};

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The load routes. The caller adds the fallbacks.
pub(super) fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/nagaya/v1/engines/{name}/load", post(load))
        .route("/nagaya/v1/engines/{name}/unload", post(unload))
}

/// The body of a 202: the swap that the group planned.
#[derive(Serialize)]
struct Started {
    engine: String,
    unload: Vec<String>,
    load: Vec<String>,
}

/// The body of a load: the engines to unload first (spec 3.3a, "Explicit layout"). An empty body names none.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct LoadBody {
    #[serde(default)]
    unload: Vec<String>,
}

/// The body of a 200: the state of the engine, which needs no swap.
#[derive(Serialize)]
struct Already {
    engine: String,
    status: &'static str,
}

async fn load(State(council): Shared, Extension(id): Id, Path(name): Path<String>, body: Body) -> Response {
    let bytes = match super::routes::bytes(body).await {
        Ok(bytes) => bytes,
        Err(failure) => return failure.respond(&id),
    };
    let body: LoadBody = if bytes.iter().all(u8::is_ascii_whitespace) {
        LoadBody::default()
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(body) => body,
            Err(error) => {
                let mut failure = Failure::new(
                    StatusCode::BAD_REQUEST,
                    ErrorType::InvalidRequest,
                    format!(
                        "nothing changed: the body of a load does not decode: {error}; send {{\"unload\": [names]}} or no body"
                    ),
                );
                failure.body.error.param = Some("unload".to_owned());
                return failure.respond(&id);
            }
        }
    };
    ask(council, id, name, Some(body.unload)).await
}

async fn unload(State(council): Shared, Extension(id): Id, Path(name): Path<String>) -> Response {
    ask(council, id, name, None).await
}

/// Ask the group of `name` for a load after the engines of `load` unload, or for an unload when `load` is `None`.
async fn ask(council: Arc<Council>, id: RequestId, name: String, load: Option<Vec<String>>) -> Response {
    let Some(ports) = council.loads() else {
        return Failure::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            "this server has no engines to load; the load routes need a process",
        )
        .respond(&id);
    };
    let engine = name.clone();
    // The group answers after the swap that it runs now, so the ask waits on a call thread (spec 3.2).
    let result = council
        .call(move || {
            Ok(match load {
                Some(unload) => ports.load_with(&engine, &unload),
                None => ports.unload(&engine),
            })
        })
        .await;
    match result {
        Err(failure) => failure.respond(&id),
        Ok(Ok(LoadReply::Started { unload, load })) => json(
            StatusCode::ACCEPTED,
            &Started {
                engine: name,
                unload,
                load,
            },
        ),
        Ok(Ok(LoadReply::Already(state))) => json(
            StatusCode::OK,
            &Already {
                engine: name,
                status: state.as_str(),
            },
        ),
        Ok(Err(error)) => refusal(&error).respond(&id),
    }
}

/// The failure of a load or an unload that did not start (spec 3.3b).
fn refusal(error: &LoadError) -> Failure {
    let message = format!("nothing changed: {error}");
    let named = |status, kind, param: &str| {
        let mut failure = Failure::new(status, kind, message.clone());
        failure.body.error.param = Some(param.to_owned());
        failure
    };
    match error {
        LoadError::Unknown(_) => named(StatusCode::NOT_FOUND, ErrorType::NotFound, "name"),
        // gogi has no type for a conflict, so the 409 is an invalid request (spec 3.3b).
        LoadError::Pinned(_) | LoadError::NoClose(_) | LoadError::InTheWay { .. } => {
            named(StatusCode::CONFLICT, ErrorType::InvalidRequest, "name")
        }
        LoadError::BadUnload(_) => named(StatusCode::BAD_REQUEST, ErrorType::InvalidRequest, "unload"),
        LoadError::Stopping | LoadError::Stopped => {
            Failure::new(StatusCode::SERVICE_UNAVAILABLE, ErrorType::Unavailable, message)
        }
    }
}
