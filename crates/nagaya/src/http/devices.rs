//! The routes of a device (spec 3.3b, "The release of a device"): `POST /nagaya/v1/devices/{id}/release` and
//! `POST /nagaya/v1/devices/{id}/reclaim`.
//!
//! A release unloads each loaded engine of the device, and the device takes no load until a reclaim. A reclaim loads
//! the resident engines of the device again. Thus a card test runs on the device while the other devices serve.

use std::sync::Arc;
use std::sync::mpsc;

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
use crate::council::error::ServiceError;
use crate::process::DeviceReply;
use crate::reprofile::ProfileCall;

/// The device routes. The caller adds the fallbacks.
pub(super) fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/nagaya/v1/devices/{id}/release", post(release))
        .route("/nagaya/v1/devices/{id}/reclaim", post(reclaim))
}

/// The body of a 202 of a release: the engines that unload, in the order of the profile.
#[derive(Serialize)]
struct Releasing {
    device: u32,
    unload: Vec<String>,
}

/// The body of a 202 of a reclaim: the engines that load, in the order of the profile.
#[derive(Serialize)]
struct Reclaiming {
    device: u32,
    load: Vec<String>,
}

/// The body of a 200: the device needs no swap.
#[derive(Serialize)]
struct Already {
    device: u32,
    status: &'static str,
}

async fn release(
    State(council): State<Arc<Council>>,
    Extension(id): Extension<RequestId>,
    Path(device): Path<String>,
) -> Response {
    ask(council, id, device, true).await
}

async fn reclaim(
    State(council): State<Arc<Council>>,
    Extension(id): Extension<RequestId>,
    Path(device): Path<String>,
) -> Response {
    ask(council, id, device, false).await
}

/// A 404 that names `id`.
fn not_found(message: String) -> Failure {
    let mut failure = Failure::new(StatusCode::NOT_FOUND, ErrorType::NotFound, message);
    failure.body.error.param = Some("id".to_owned());
    failure
}

/// Release or reclaim the device of the path, by the active profile.
async fn ask(council: Arc<Council>, id: RequestId, device: String, release: bool) -> Response {
    let Ok(device) = device.parse::<u32>() else {
        return not_found(format!(
            "nothing changed: `{device}` is not a device id; send the number of a device, as the profile names it"
        ))
        .respond(&id);
    };
    let (Some(ports), Some(calls)) = (council.loads(), council.profile_calls()) else {
        return Failure::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            "this server has no devices to release; the device routes need a process",
        )
        .respond(&id);
    };
    // The groups answer after the swaps that they run now, so the ask waits on a call thread (spec 3.2).
    let result = council
        .call(move || {
            let (reply, answer) = mpsc::channel();
            calls
                .send(ProfileCall::Get(reply))
                .map_err(|_| ServiceError::Stopping)?;
            let profile = answer.recv().map_err(|_| ServiceError::Stopping)?;
            Ok(if release {
                ports.release_device(device, &profile)
            } else {
                ports.reclaim_device(device, &profile)
            })
        })
        .await;
    match result {
        Err(failure) => failure.respond(&id),
        Ok(Err(error)) => super::loads::refusal(&error).respond(&id),
        Ok(Ok(DeviceReply::Started(names))) if release => {
            json(StatusCode::ACCEPTED, &Releasing { device, unload: names })
        }
        Ok(Ok(DeviceReply::Started(names))) => json(StatusCode::ACCEPTED, &Reclaiming { device, load: names }),
        Ok(Ok(DeviceReply::Already)) => json(
            StatusCode::OK,
            &Already {
                device,
                status: if release { "released" } else { "in use" },
            },
        ),
    }
}
