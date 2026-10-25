//! The council v2 routes (gogi "Endpoints").

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use gogi_proto::SnapshotId;
use gogi_proto::context::{ContextBuild, ContextBuilt};
use gogi_proto::decision::{DecisionRequest, Signals};
use gogi_proto::error::ErrorType;
use gogi_proto::generation::GenerationRequest;
use gogi_proto::pins::PinsPut;
use gogi_proto::reads::ReadsRequest;
use serde::Serialize;
use serde_json::json;

use super::errors::{Failure, parse};
use super::{Council, MAX_BODY_BYTES, RequestId};
use crate::council::Schedule;
use crate::council::error::ServiceError;

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The routes of gogi's endpoint table. A route that does not exist is a 404 `not_found`, also for a known path with
/// another method.
pub fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/council/v2/identity", get(identity))
        .route("/council/v2/contexts", post(contexts))
        .route("/council/v2/snapshots/{id}", get(snapshot))
        .route(
            "/council/v2/pins/{owner}",
            get(pins_get).put(pins_put).delete(pins_delete),
        )
        .route("/council/v2/specs", get(specs_list).post(specs_post))
        .route("/council/v2/specs/{spec_id}", get(spec_get).delete(spec_delete))
        .route("/council/v2/decisions", post(decisions))
        .route("/council/v2/reads", post(reads))
        .route("/council/v2/generations", post(generations))
        .route("/v1/audio/speech", post(super::speech::speech))
        .merge(super::extensions::routes())
        .fallback(unknown)
        .method_not_allowed_fallback(unknown)
}

/// A JSON response.
pub(super) fn json<T: Serialize>(status: StatusCode, value: &T) -> Response {
    (status, axum::Json(value)).into_response()
}

pub(super) fn answer<T: Serialize>(id: &RequestId, result: Result<T, Failure>) -> Response {
    match result {
        Ok(value) => json(StatusCode::OK, &value),
        Err(failure) => failure.respond(id),
    }
}

fn empty(id: &RequestId, result: Result<(), Failure>) -> Response {
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(failure) => failure.respond(id),
    }
}

/// The bytes of a request body. A body past [`MAX_BODY_BYTES`] is a 413 `too_large`. A body that does not arrive whole
/// is a 400 `invalid_request`, because the body is outside the schema (gogi "Errors").
pub(super) async fn bytes(body: Body) -> Result<Bytes, Failure> {
    axum::body::to_bytes(body, MAX_BODY_BYTES).await.map_err(|error| {
        let limit =
            std::error::Error::source(&error).is_some_and(|source| source.is::<http_body_util::LengthLimitError>());
        if limit {
            let mut failure = Failure::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                ErrorType::TooLarge,
                format!("nothing ran: the body is past {MAX_BODY_BYTES} bytes; send less"),
            );
            failure.body.error.param = Some("body".to_owned());
            failure
        } else {
            Failure::from(ServiceError::Invalid(gogi_proto::Invalid::param(
                "body",
                format!("nothing ran: the body did not arrive whole ({error}); send the request again"),
            )))
        }
    })
}

async fn unknown(Extension(id): Id) -> Response {
    Failure::new(
        StatusCode::NOT_FOUND,
        ErrorType::NotFound,
        "no route has this method and path; gogi's docs/council-api.md lists the routes",
    )
    .respond(&id)
}

async fn identity(State(council): Shared) -> Response {
    json(StatusCode::OK, &council.identity())
}

async fn contexts(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: ContextBuild = parse(&bytes(body).await?, "context")?;
        let served = council.route(&request.model)?;
        let dry_run = request.dry_run.unwrap_or(false);
        let schedule = Schedule::of(request.class, request.priority);
        let model = served.service.model().to_owned();
        let built = council
            .call(move || served.service.build(&request.context, dry_run, schedule))
            .await?;
        Ok(ContextBuilt {
            model,
            head: built.head.unwrap_or_else(|| recipe_head(&built.head_recipe)),
            usage: built.usage,
            from: built.from,
            dry_run,
            snapshots: built.snapshots,
        })
    }
    .await;
    answer(&id, result)
}

/// The head of a dry run that finds no held head: `recipe:` and the hex of the head's recipe (spec 11.3c). No store
/// entry has this id, so a snapshot read or a pin of it finds nothing.
fn recipe_head(recipe: &nagaya_store::RecipeId) -> SnapshotId {
    SnapshotId::parse(format!("recipe:{}", recipe.hex())).expect("a recipe id makes a valid snapshot id")
}

async fn snapshot(State(council): Shared, Extension(id): Id, Path(snapshot): Path<String>) -> Response {
    let result = async {
        let home = council.home()?;
        let snapshot = SnapshotId::parse(snapshot.clone())
            .map_err(|_| Failure::from(ServiceError::NotFound(format!("no snapshot {snapshot} is held"))))?;
        council.call(move || home.service.snapshot(&snapshot)).await
    }
    .await;
    answer(&id, result)
}

async fn pins_put(State(council): Shared, Extension(id): Id, Path(owner): Path<String>, body: Body) -> Response {
    let result = async {
        let home = council.home()?;
        let put: PinsPut = parse(&bytes(body).await?, "pins")?;
        council.call(move || home.service.put_pins(&owner, &put)).await
    }
    .await;
    answer(&id, result)
}

async fn pins_get(State(council): Shared, Extension(id): Id, Path(owner): Path<String>) -> Response {
    let result = async {
        let home = council.home()?;
        council.call(move || home.service.get_pins(&owner)).await
    }
    .await;
    answer(&id, result)
}

async fn pins_delete(State(council): Shared, Extension(id): Id, Path(owner): Path<String>) -> Response {
    let result = async {
        let home = council.home()?;
        council.call(move || home.service.delete_pins(&owner)).await
    }
    .await;
    empty(&id, result)
}

async fn specs_post(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let home = council.home()?;
        let spec: serde_json::Value = parse(&bytes(body).await?, "spec")?;
        council.call(move || home.service.hold_spec(spec)).await
    }
    .await;
    answer(&id, result)
}

async fn specs_list(State(council): Shared, Extension(id): Id) -> Response {
    let result = async {
        let home = council.home()?;
        let specs = council.call(move || Ok(home.service.specs())).await?;
        Ok(json!({"specs": specs}))
    }
    .await;
    answer(&id, result)
}

async fn spec_get(State(council): Shared, Extension(id): Id, Path(spec_id): Path<String>) -> Response {
    let result = async {
        let home = council.home()?;
        council.call(move || home.service.spec(&spec_id)).await
    }
    .await;
    answer(&id, result)
}

async fn spec_delete(State(council): Shared, Extension(id): Id, Path(spec_id): Path<String>) -> Response {
    let result = async {
        let home = council.home()?;
        council.call(move || home.service.drop_spec(&spec_id)).await
    }
    .await;
    empty(&id, result)
}

async fn decisions(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: DecisionRequest = parse(&bytes(body).await?, "decision")?;
        let served = council.route(&request.model)?;
        let mut response = council.call(move || served.service.decide(&request)).await?;
        if let Some(signals) = response.signals.as_mut() {
            keep_gogi_places(signals);
        }
        Ok(response)
    }
    .await;
    answer(&id, result)
}

async fn reads(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: ReadsRequest = parse(&bytes(body).await?, "reads")?;
        let served = council.route(&request.model)?;
        let mut response = council.call(move || served.service.read(&request)).await?;
        if let Some(signals) = response.signals.as_mut() {
            keep_gogi_places(signals);
        }
        Ok(response)
    }
    .await;
    answer(&id, result)
}

async fn generations(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let request = async {
        let request: GenerationRequest = parse(&bytes(body).await?, "generation")?;
        let served = council.route(&request.model)?;
        Ok::<_, Failure>((request, served))
    }
    .await;
    let (request, served) = match request {
        Ok(found) => found,
        Err(failure) => return failure.respond(&id),
    };
    if request.stream.unwrap_or(false) {
        return super::stream::generation(council, served, request, id).await;
    }
    let result = council.call(move || served.service.generate(&request, None)).await;
    answer(&id, result)
}

/// Keep the control text hits whose place gogi names: `case`, `spec` and `suffix:<i>` (gogi "Rules every server
/// keeps", rule 2). A family also finds hits in the messages of a context, as `message:<i>`. gogi names no place for
/// those, so a response does not report them.
pub fn keep_gogi_places(signals: &mut Signals) {
    signals.control_text.retain(|hit| {
        hit.r#where == "case"
            || hit.r#where == "spec"
            || hit
                .r#where
                .strip_prefix("suffix:")
                .is_some_and(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
    });
}
