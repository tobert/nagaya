//! The load routes (spec 3.3b): `POST /nagaya/v1/engines/{name}/load` and `POST /nagaya/v1/engines/{name}/unload`.
//!
//! A route asks the group thread of the engine to plan a swap. The answer comes when the group plans it, not when the
//! swap ends. The status feed and identity show the states that follow (spec 9.4, 11.1).

use std::sync::Arc;

use axum::Router;
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::post;
use gogi_proto::error::ErrorType;
use serde::Serialize;

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

/// The body of a 200: the state of the engine, which needs no swap.
#[derive(Serialize)]
struct Already {
    engine: String,
    status: &'static str,
}

async fn load(State(council): Shared, Extension(id): Id, Path(name): Path<String>) -> Response {
    ask(council, id, name, true).await
}

async fn unload(State(council): Shared, Extension(id): Id, Path(name): Path<String>) -> Response {
    ask(council, id, name, false).await
}

async fn ask(council: Arc<Council>, id: RequestId, name: String, load: bool) -> Response {
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
            Ok(if load {
                ports.load(&engine)
            } else {
                ports.unload(&engine)
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
    let named = |status, kind| {
        let mut failure = Failure::new(status, kind, message.clone());
        failure.body.error.param = Some("name".to_owned());
        failure
    };
    match error {
        LoadError::Unknown(_) => named(StatusCode::NOT_FOUND, ErrorType::NotFound),
        // gogi has no type for a conflict, so the 409 is an invalid request (spec 3.3b).
        LoadError::Resident(_) => named(StatusCode::CONFLICT, ErrorType::InvalidRequest),
        LoadError::Stopping | LoadError::Stopped => {
            Failure::new(StatusCode::SERVICE_UNAVAILABLE, ErrorType::Unavailable, message)
        }
    }
}
