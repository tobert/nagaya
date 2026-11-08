//! The council v2 routes (gogi "Endpoints").

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Extension, Path, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use gogi_proto::SnapshotId;
use gogi_proto::context::{ContextBuild, ContextBuilt};
use gogi_proto::decision::{DecisionRequest, Signals};
use gogi_proto::embeddings::EmbeddingsRequest;
use gogi_proto::error::ErrorType;
use gogi_proto::generation::GenerationRequest;
use gogi_proto::identity::{Capability, ServedModel};
use gogi_proto::pins::PinsPut;
use gogi_proto::reads::ReadsRequest;
use serde::Serialize;
use serde_json::json;

use super::errors::{Failure, parse};
use super::peers;
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
        .route("/council/v2/embeddings", post(embeddings))
        .route("/council/v2/control-vectors", post(control_vectors))
        .merge(super::extensions::routes())
        .merge(super::objects::routes())
        .merge(super::loads::routes())
        .merge(super::profile::routes())
        .merge(super::cluster::routes())
        .merge(super::lab::routes())
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
        let served = council.route_contexts(&request.model, &[&request.context], false)?;
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

/// The head of a build that names no held head, such as a dry run: `blake3:` and the hex of the head's recipe (spec
/// 11.3c, Proposed). No store entry has this id, so a snapshot read or a pin of it finds nothing.
fn recipe_head(recipe: &nagaya_store::RecipeId) -> SnapshotId {
    SnapshotId::parse(format!("blake3:{}", recipe.hex())).expect("a recipe id is a whole BLAKE3 hash")
}

/// A read by id: from this node, or on a miss from the first peer that holds the thing (spec 11.1a, "The routes with
/// no model"). A node that serves no text model holds no snapshot or spec, so it asks the peers too. Only the peers
/// that declare `pins` serve a text model, so only they get the call.
async fn read_by_id(council: &Council, local: Response, uri: &Uri, headers: &HeaderMap, id: &RequestId) -> Response {
    let missed = peers::is_miss(local.status()) || council.home().is_err();
    let call = peers::call(Method::GET, &peers::raw_path(uri), headers, &[], Bytes::new(), id);
    peers::read_on_miss(council, local, missed, false, Some(Capability::Pins), call, headers).await
}

/// The answer of this node to a write of pins or specs, when it serves a text model and so takes part.
fn home_answer(council: &Council) -> Option<Arc<super::Served>> {
    council.home().ok()
}

/// The response of a write that no node took: the failure of a node that serves no text model.
fn no_home(council: &Council, id: &RequestId) -> Response {
    match council.home() {
        Ok(_) => unreachable!("a node with a home answers a write itself"),
        Err(failure) => failure.respond(id),
    }
}

async fn snapshot(
    State(council): Shared,
    Extension(id): Id,
    Path(snapshot): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let local = snapshot_here(&council, snapshot, &id).await;
    read_by_id(&council, local, &uri, &headers, &id).await
}

async fn snapshot_here(council: &Council, snapshot: String, id: &RequestId) -> Response {
    let result = async {
        let home = council.home()?;
        let snapshot = path_id(&snapshot)
            .and_then(|_| SnapshotId::parse(snapshot.clone()).map_err(|e| Failure::from(ServiceError::internal(e))))?;
        council.call(move || home.service.snapshot(&snapshot)).await
    }
    .await;
    answer(id, result)
}

/// A `PUT` of pins goes to each live node that declares `pins`, and each node pins the snapshots that it holds (spec
/// 11.1a, "The routes with no model").
async fn pins_put(
    State(council): Shared,
    Extension(id): Id,
    Path(owner): Path<String>,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let body = match bytes(body).await {
        Ok(body) => body,
        Err(failure) => return failure.respond(&id),
    };
    let local = match home_answer(&council) {
        Some(home) => {
            let result = async {
                let put: PinsPut = parse(&body, "pins")?;
                council.call(move || home.service.put_pins(&owner, &put)).await
            }
            .await;
            Some(answer(&id, result))
        }
        None => None,
    };
    let call = peers::call(
        Method::PUT,
        &peers::raw_path(&uri),
        &headers,
        &[CONTENT_TYPE],
        body,
        &id,
    );
    let answers = peers::write_all(&council, local, call, Capability::Pins, &headers).await;
    peers::join_sets(answers, "snapshots", true, || no_home(&council, &id))
}

async fn pins_get(
    State(council): Shared,
    Extension(id): Id,
    Path(owner): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let local = match home_answer(&council) {
        Some(home) => Some(answer(&id, council.call(move || home.service.get_pins(&owner)).await)),
        None => None,
    };
    let call = peers::call(Method::GET, &peers::raw_path(&uri), &headers, &[], Bytes::new(), &id);
    let answers = peers::write_all(&council, local, call, Capability::Pins, &headers).await;
    peers::join_sets(answers, "snapshots", false, || no_home(&council, &id))
}

async fn pins_delete(
    State(council): Shared,
    Extension(id): Id,
    Path(owner): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let local = match home_answer(&council) {
        Some(home) => Some(empty(&id, council.call(move || home.service.delete_pins(&owner)).await)),
        None => None,
    };
    let call = peers::call(Method::DELETE, &peers::raw_path(&uri), &headers, &[], Bytes::new(), &id);
    let answers = peers::write_all(&council, local, call, Capability::Pins, &headers).await;
    peers::join_delete(answers, || no_home(&council, &id))
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

/// Checks a content id in a path: an id that is not whole is a `400` naming `id`, and the server never looks up a
/// prefix (gogi "Content ids").
pub(super) fn path_id(id: &str) -> Result<(), Failure> {
    match gogi_proto::content_id_problem(id) {
        None => Ok(()),
        Some(problem) => Err(Failure::from(ServiceError::Invalid(gogi_proto::Invalid::param(
            "id",
            format!("nothing was looked up: {problem}"),
        )))),
    }
}

async fn spec_get(
    State(council): Shared,
    Extension(id): Id,
    Path(spec_id): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    if let Err(failure) = path_id(&spec_id) {
        return failure.respond(&id);
    }
    let result = async {
        let home = council.home()?;
        council.call(move || home.service.spec(&spec_id)).await
    }
    .await;
    read_by_id(&council, answer(&id, result), &uri, &headers, &id).await
}

/// A `DELETE` of a spec goes to each live node that declares `pins`: a 204 when any node held the spec (spec 11.1a).
async fn spec_delete(
    State(council): Shared,
    Extension(id): Id,
    Path(spec_id): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    if let Err(failure) = path_id(&spec_id) {
        return failure.respond(&id);
    }
    let local = match home_answer(&council) {
        Some(home) => Some(empty(&id, council.call(move || home.service.drop_spec(&spec_id)).await)),
        None => None,
    };
    let call = peers::call(Method::DELETE, &peers::raw_path(&uri), &headers, &[], Bytes::new(), &id);
    let answers = peers::write_all(&council, local, call, Capability::Pins, &headers).await;
    peers::join_delete(answers, || no_home(&council, &id))
}

async fn decisions(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: DecisionRequest = parse(&bytes(body).await?, "decision")?;
        let contexts: Vec<_> = request.contexts.iter().collect();
        let served = council.route_contexts(&request.model, &contexts, true)?;
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
        let contexts: Vec<_> = request.contexts.iter().collect();
        let served = council.route_contexts(&request.model, &contexts, true)?;
        let mut response = council.call(move || served.service.read(&request)).await?;
        if let Some(signals) = response.signals.as_mut() {
            keep_gogi_places(signals);
        }
        Ok(response)
    }
    .await;
    answer(&id, result)
}

/// Embeddings (gogi "Embeddings"). A served embedding model gets its job (spec 11.10). Each other served model is a
/// 400 that names `model`, as gogi's contract asks. gogi's checks run before the 503 of a model that is not ready.
async fn embeddings(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let request = async {
        let request: EmbeddingsRequest = parse(&bytes(body).await?, "embeddings")?;
        if let Some((service, model)) = super::embeddings::route(&council, &request.model) {
            request.check(&model).map_err(ServiceError::Invalid)?;
            if model.status != gogi_proto::identity::ModelStatus::Ready {
                return Err(Failure::unavailable(&model));
            }
            return Ok((service, model, request));
        }
        // Any other served model has no embedding service, so gogi's check refuses it with a 400 that names `model`.
        let identity = council.identity();
        let served = identity.route(&request.model).map_err(ServiceError::Invalid)?;
        request.check(served).map_err(ServiceError::Invalid)?;
        if served.status != gogi_proto::identity::ModelStatus::Ready {
            return Err(Failure::unavailable(served));
        }
        Err(Failure::from(ServiceError::Invalid(gogi_proto::Invalid::param(
            "model",
            format!(
                "nothing ran: model {} declares embed, but it has no embedding service here",
                served.model
            ),
        ))))
    }
    .await;
    match request {
        Ok((service, model, request)) => super::embeddings::serve(council, service, model, request, id).await,
        Err(failure) => failure.respond(&id),
    }
}

/// A control vector (gogi "Control vectors", spec 11.12). gogi's `validate` and `check` run first, then the 503 of a
/// model that is not ready. The job of the text service makes the vector, and the object store holds its bytes.
async fn control_vectors(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: gogi_proto::control::ControlVectorRequest = parse(&bytes(body).await?, "control-vectors")?;
        request.validate().map_err(ServiceError::Invalid)?;
        let identity = council.identity();
        let model = identity.route(&request.model).map_err(ServiceError::Invalid)?;
        request.check(model).map_err(ServiceError::Invalid)?;
        if model.status != gogi_proto::identity::ModelStatus::Ready {
            return Err(Failure::unavailable(model));
        }
        let served = council.route(&request.model)?;
        let inner = Arc::clone(&council);
        council
            .call(move || {
                let put = |bytes: &[u8]| {
                    let id = super::voice::seal(&inner, bytes, gogi_proto::control::MEDIA_TYPE)?;
                    Ok(gogi_proto::ids::ContentId::parse(format!("blake3:{}", id.hex()))
                        .expect("a content id makes a valid gogi id"))
                };
                served.service.control_vector(&request, &put)
            })
            .await
    }
    .await;
    answer(&id, result)
}

/// A generation of a served model with no text replies.
enum Media {
    Speech(Arc<crate::speech::SpeechService>, ServedModel, GenerationRequest),
    Image(Arc<crate::image::ImageService>, ServedModel, GenerationRequest),
}

async fn generations(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let request = async {
        let request: GenerationRequest = parse(&bytes(body).await?, "generation")?;
        if let Some((service, model)) = super::voice::route(&council, &request.model) {
            if model.status != gogi_proto::identity::ModelStatus::Ready {
                return Err(Failure::unavailable(&model));
            }
            Council::check_thinking(&model, &[&request.context], false)?;
            return Ok(Err(Media::Speech(service, model, request)));
        }
        if let Some((service, model)) = super::image::route(&council, &request.model) {
            if model.status != gogi_proto::identity::ModelStatus::Ready {
                return Err(Failure::unavailable(&model));
            }
            Council::check_thinking(&model, &[&request.context], false)?;
            return Ok(Err(Media::Image(service, model, request)));
        }
        let served = council.route_contexts(&request.model, &[&request.context], false)?;
        Ok::<_, Failure>(Ok((request, served)))
    }
    .await;
    let (request, served) = match request {
        Ok(Ok(found)) => found,
        Ok(Err(Media::Speech(service, model, request))) => {
            return super::voice::generation(council, service, model, request, id).await;
        }
        Ok(Err(Media::Image(service, model, request))) => {
            return super::image::generation(council, service, model, request, id).await;
        }
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
