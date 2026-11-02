//! The routes of objects and keeps (gogi "Objects and keeps", spec 6.15).
//!
//! A put streams the body to a staged file and hashes it as it arrives, so a large object never sits in memory. Each
//! call on the store runs on a call thread through [`Council::call`], so it never blocks the control runtime, a panic
//! in it is an internal error, and an internal error stops the process (spec 8.5).

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, Path, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::put;
use futures_util::StreamExt;
use gogi_proto::error::ErrorType;
use gogi_proto::objects::{ByteRange, KeepsPut, byte_range, mime_problem};
use nagaya_store::ContentId;
use nagaya_store::objects::{ObjectError, ObjectStore};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use super::errors::{Failure, parse};
use super::peers;
use super::pending::{Lookup, PendingOutput};
use super::routes::{answer, bytes, path_id};
use super::{Council, RequestId};
use crate::council::error::ServiceError;

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The most puts that stream at once. Each stages up to `object_bytes`, so this bounds the staged bytes on disk.
const PUTS_AT_ONCE: usize = 4;

/// The most bytes past `object_bytes` that a refused put reads and drops before its 413, so that the client is not
/// still sending when the answer comes. A body past even this gets its 413 before it is read.
const DRAIN_BYTES: u64 = 64 << 20;

static PUTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(PUTS_AT_ONCE);

pub(super) fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/council/v2/objects/{id}", put(put_object).get(get_object))
        .route(
            "/council/v2/keeps/{owner}",
            put(put_keep).get(get_keep).delete(delete_keep),
        )
}

/// Read and drop the rest of a body, up to [`DRAIN_BYTES`]. Give true when the body ended.
async fn drain(stream: &mut axum::body::BodyDataStream) -> bool {
    let mut dropped: u64 = 0;
    while dropped < DRAIN_BYTES {
        match stream.next().await {
            Some(Ok(chunk)) => dropped += chunk.len() as u64,
            Some(Err(_)) => return false,
            None => return true,
        }
    }
    false
}

/// The wall clock in milliseconds, as the store takes it.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn invalid(param: &str, message: String) -> Failure {
    Failure::from(ServiceError::Invalid(gogi_proto::Invalid::param(param, message)))
}

fn too_large(param: &str, message: String) -> Failure {
    let mut failure = Failure::new(StatusCode::PAYLOAD_TOO_LARGE, ErrorType::TooLarge, message);
    failure.body.error.param = Some(param.to_owned());
    failure
}

fn not_found(message: String) -> Failure {
    Failure::new(StatusCode::NOT_FOUND, ErrorType::NotFound, message)
}

/// The service error of a store error. A file system failure is internal, so it stops the process.
fn from_store(error: ObjectError) -> ServiceError {
    match error {
        ObjectError::TooLarge { limit, .. } => ServiceError::TooLarge {
            param: limit.to_owned(),
            message: error.to_string(),
        },
        ObjectError::Io(message) => ServiceError::internal(message),
    }
}

/// The store of the process, or the 404 of a profile with no objects budget.
fn store(council: &Council) -> Result<Arc<Mutex<ObjectStore>>, Failure> {
    council.objects().ok_or_else(|| {
        not_found(
            "the server holds no objects: its profile names no `objects` budget in `[store]` (spec 6.15)".to_owned(),
        )
    })
}

/// A whole content id from a path, as the store names it.
fn content(id: &str) -> Result<ContentId, Failure> {
    path_id(id)?;
    ContentId::from_hex(id.trim_start_matches("blake3:"))
        .map_err(|_| invalid("id", format!("nothing was looked up: {id:?} is not a content id")))
}

async fn put_object(
    State(council): Shared,
    Extension(id): Id,
    Path(text): Path<String>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let mut stream = body.into_data_stream();
    let result = async {
        let store = store(&council)?;
        if council.draining() {
            return Err(Failure::from(ServiceError::Stopping).with_message("nothing is held: the server is stopping"));
        }
        let content = content(&text)?;
        let mime = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).ok_or_else(|| {
            invalid(
                "content-type",
                "nothing is held: the request has no content-type header; send the object's media type, such as \
                 `audio/wav`"
                    .to_owned(),
            )
        })?;
        if let Some(problem) = mime_problem(mime) {
            return Err(invalid("content-type", format!("nothing is held: content-type {problem}")));
        }
        let _put = PUTS.try_acquire().map_err(|_| {
            Failure::from(ServiceError::Busy(format!(
                "nothing is held: {PUTS_AT_ONCE} puts stream now, the most that this server takes at once; try again later"
            )))
        })?;
        let (limit, staged) = {
            let store = Arc::clone(&store);
            council
                .call(move || {
                    let mut store = store.lock().expect("the object store lock is poisoned");
                    Ok((store.object_bytes(), store.staging_path()))
                })
                .await?
        };
        if let Some(n) = declared.filter(|n| *n > limit) {
            if n <= limit + DRAIN_BYTES {
                let _ = drain(&mut stream).await;
            }
            return Err(too_large(
                "object_bytes",
                format!("nothing is held: the object is {n} bytes; object_bytes is {limit}"),
            ));
        }
        // Stream the body to the staged file, and hash it as it arrives.
        let write_error = |e: std::io::Error| {
            council.failed(ServiceError::internal(format!("cannot write `{}`: {e}", staged.display())))
        };
        let mut file = tokio::fs::File::create(&staged).await.map_err(write_error)?;
        let mut hasher = blake3::Hasher::new();
        let mut size: u64 = 0;
        let outcome = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| {
                    invalid(
                        "body",
                        format!("nothing is held: the body did not arrive whole ({e}); send the request again"),
                    )
                })?;
                size += chunk.len() as u64;
                if size > limit {
                    let _ = drain(&mut stream).await;
                    return Err(too_large(
                        "object_bytes",
                        format!("nothing is held: the object is more than {limit} bytes, which is object_bytes"),
                    ));
                }
                hasher.update(&chunk);
                file.write_all(&chunk).await.map_err(write_error)?;
            }
            file.sync_all().await.map_err(write_error)?;
            let actual = format!("blake3:{}", hasher.finalize().to_hex());
            if actual != text {
                return Err(invalid(
                    "id",
                    format!(
                        "nothing is held: the {size} bytes sent hash to {actual}, not {text}; PUT them at {actual}, or \
                         send the bytes {text} names"
                    ),
                ));
            }
            Ok(())
        }
        .await;
        if let Err(failure) = outcome {
            let _ = tokio::fs::remove_file(&staged).await;
            return Err(failure);
        }
        let mime = mime.to_owned();
        let committed = council
            .call(move || {
                let mut store = store.lock().expect("the object store lock is poisoned");
                store.commit(&content, &mime, &staged, size, now_ms()).map_err(from_store)
            })
            .await?;
        let status = if committed.created {
            StatusCode::CREATED
        } else {
            StatusCode::OK
        };
        Ok((status, json!({"id": text, "mime": committed.mime, "bytes": size})))
    }
    .await;
    match result {
        Ok((status, value)) => (status, axum::Json(value)).into_response(),
        Err(failure) => {
            // A refusal can come before the body arrives. Read the rest of the body first, so that the client can
            // send its next request on this connection. A body that is too large to read gets `connection: close`.
            let ended = declared.is_none_or(|n| n <= DRAIN_BYTES) && drain(&mut stream).await;
            let mut response = failure.respond(&id);
            if !ended {
                response
                    .headers_mut()
                    .insert(header::CONNECTION, header::HeaderValue::from_static("close"));
            }
            response
        }
    }
}

/// The bytes of an output as the server makes them. The body ends when the output is sealed or cancelled.
fn pending_stream(output: Arc<PendingOutput>) -> Response {
    let mime = output.mime.clone();
    let changes = output.changed();
    let stream = futures_util::stream::unfold((output, 0usize, changes), |(output, at, mut changes)| async move {
        loop {
            let (bytes, end) = output.read(at);
            if !bytes.is_empty() {
                let next = at + bytes.len();
                return Some((Ok::<_, std::convert::Infallible>(bytes), (output, next, changes)));
            }
            if end.is_some() {
                return None;
            }
            // A change after the read above wakes this wait, so no byte is missed.
            if changes.changed().await.is_err() {
                return None;
            }
        }
    });
    let mut response = (StatusCode::OK, Body::from_stream(stream)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        mime.parse().expect("a pending media type is a header value"),
    );
    response
}

async fn get_object(
    State(council): Shared,
    Extension(id): Id,
    Path(text): Path<String>,
    method: Method,
    headers: HeaderMap,
) -> Response {
    read(&council, method, &text, &headers, &id).await
}

/// A `GET` or a `HEAD` of an object or a pending id: from this node, or on a miss from the peer that holds it (spec
/// 6.15, "Objects in the cluster"). The answer of the peer comes back as it is, and this node keeps no copy.
pub(super) async fn read(
    council: &Council,
    method: Method,
    text: &str,
    headers: &HeaderMap,
    id: &RequestId,
) -> Response {
    let local = read_here(council, text, headers, id).await;
    let missed = peers::is_miss(local.status());
    let call = peers::call(
        method,
        &format!("/council/v2/objects/{text}"),
        headers,
        &[header::RANGE],
        axum::body::Bytes::new(),
        id,
    );
    peers::read_on_miss(council, local, missed, true, None, call, headers).await
}

/// A read of an object or a pending id from this node alone.
async fn read_here(council: &Council, text: &str, headers: &HeaderMap, id: &RequestId) -> Response {
    let text = text.to_owned();
    let id = id.clone();
    let result = async {
        let store = store(council)?;
        if text.starts_with("pending:") {
            if let Some(problem) = gogi_proto::ids::pending_id_problem(&text) {
                return Err(invalid("id", format!("nothing was looked up: {problem}")));
            }
            return match council.pending().lookup(&text) {
                Lookup::Making(output) => Ok(pending_stream(output)),
                Lookup::Sealed(sealed) => {
                    let mut response = StatusCode::PERMANENT_REDIRECT.into_response();
                    let location = format!("/council/v2/objects/{sealed}");
                    response.headers_mut().insert(
                        header::LOCATION,
                        location.parse().expect("an object path is a header value"),
                    );
                    Ok(response)
                }
                Lookup::Unknown => Err(not_found(format!(
                    "no output is pending as {text}: the server never gave it, its output was cancelled, or it was \
                     sealed long ago; read the content id that `done` named"
                ))),
            };
        }
        let content = content(&text)?;
        let found = council
            .call(move || {
                let mut store = store.lock().expect("the object store lock is poisoned");
                store.get(&content, now_ms()).map_err(from_store)
            })
            .await?
            .ok_or_else(|| not_found(format!("no object {text} is held; PUT it again")))?;
        let range = headers.get(header::RANGE).and_then(|v| v.to_str().ok());
        let (status, start, length, content_range) = match byte_range(range, found.bytes) {
            ByteRange::Whole => (StatusCode::OK, 0, found.bytes, None),
            ByteRange::Part { start, end } => (
                StatusCode::PARTIAL_CONTENT,
                start,
                end - start + 1,
                Some(format!("bytes {start}-{end}/{}", found.bytes)),
            ),
            ByteRange::Unsatisfiable => {
                let mut failure = Failure::new(
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    ErrorType::InvalidRequest,
                    format!(
                        "nothing was read: the range `{}` starts past the end of the {} bytes of {text}",
                        range.unwrap_or(""),
                        found.bytes
                    ),
                );
                failure.body.error.param = Some("range".to_owned());
                let mut response = failure.respond(&id);
                let unsatisfied = format!("bytes */{}", found.bytes);
                response.headers_mut().insert(
                    header::CONTENT_RANGE,
                    unsatisfied.parse().expect("a content range is a header value"),
                );
                return Ok(response);
            }
        };
        // A put can evict the object after the lookup. Its file is then gone, which is the 404 of an object that the
        // store does not hold, not an internal error.
        let mut file = tokio::fs::File::open(&found.path).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => not_found(format!("no object {text} is held; PUT it again")),
            _ => council.failed(ServiceError::internal(format!(
                "cannot read `{}`: {e}",
                found.path.display()
            ))),
        })?;
        file.seek(std::io::SeekFrom::Start(start)).await.map_err(|e| {
            council.failed(ServiceError::internal(format!(
                "cannot read `{}`: {e}",
                found.path.display()
            )))
        })?;
        let body = Body::from_stream(tokio_util::io::ReaderStream::new(file.take(length)));
        let mut response = (status, body).into_response();
        let h = response.headers_mut();
        h.insert(
            header::CONTENT_TYPE,
            found.mime.parse().expect("a held media type is a header value"),
        );
        h.insert(header::CONTENT_LENGTH, length.into());
        h.insert(header::ACCEPT_RANGES, "bytes".parse().expect("a header value"));
        if let Some(r) = content_range {
            h.insert(
                header::CONTENT_RANGE,
                r.parse().expect("a content range is a header value"),
            );
        }
        Ok(response)
    }
    .await;
    match result {
        Ok(response) => response,
        Err(failure) => failure.respond(&id),
    }
}

/// An owner from a path, or the 400 that names it.
fn owner_of(owner: &str) -> Result<(), Failure> {
    if gogi_proto::pins::is_owner(owner) {
        Ok(())
    } else {
        Err(invalid(
            "owner",
            format!(
                "nothing changed: the owner {owner:?} is not printable text of 1 to 128 characters without a slash"
            ),
        ))
    }
}

/// The response of a keep route that no node took: the 404 of a node with no object store.
fn no_store(council: &Council, id: &RequestId) -> Response {
    match store(council) {
        Ok(_) => unreachable!("a node with an object store answers a keep route itself"),
        Err(failure) => failure.respond(id),
    }
}

/// Each keep route goes to each live node that declares `objects`, and the answers join as for pins (spec 11.1a, "The
/// routes with no model"). This node takes part when it has an object store.
async fn keep_route(
    council: &Council,
    local: Option<Response>,
    method: Method,
    uri: &Uri,
    headers: &HeaderMap,
    body: axum::body::Bytes,
    id: &RequestId,
) -> Response {
    let delete = method == Method::DELETE;
    let put = method == Method::PUT;
    let call = peers::call(
        method,
        &peers::raw_path(uri),
        headers,
        &[header::CONTENT_TYPE],
        body,
        id,
    );
    let answers = peers::write_all(council, local, call, gogi_proto::identity::Capability::Objects, headers).await;
    if delete {
        peers::join_delete(answers, || no_store(council, id))
    } else {
        peers::join_sets(answers, "objects", put, || no_store(council, id))
    }
}

async fn put_keep(
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
    let local = match council.objects() {
        Some(_) => Some(put_keep_here(&council, &id, owner, &body).await),
        None => None,
    };
    keep_route(&council, local, Method::PUT, &uri, &headers, body, &id).await
}

async fn get_keep(
    State(council): Shared,
    Extension(id): Id,
    Path(owner): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let local = match council.objects() {
        Some(_) => Some(get_keep_here(&council, &id, owner).await),
        None => None,
    };
    keep_route(
        &council,
        local,
        Method::GET,
        &uri,
        &headers,
        axum::body::Bytes::new(),
        &id,
    )
    .await
}

async fn delete_keep(
    State(council): Shared,
    Extension(id): Id,
    Path(owner): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let local = match council.objects() {
        Some(_) => Some(delete_keep_here(&council, &id, owner).await),
        None => None,
    };
    keep_route(
        &council,
        local,
        Method::DELETE,
        &uri,
        &headers,
        axum::body::Bytes::new(),
        &id,
    )
    .await
}

async fn put_keep_here(council: &Council, id: &RequestId, owner: String, body: &[u8]) -> Response {
    let result = async {
        let store = store(council)?;
        owner_of(&owner)?;
        let put: KeepsPut = parse(body, "keeps")?;
        put.validate().map_err(|e| Failure::from(ServiceError::Invalid(e)))?;
        let ids: Vec<ContentId> = put.objects.iter().map(|text| content(text)).collect::<Result<_, _>>()?;
        let ttl_ms = put.ttl_s.saturating_mul(1000);
        let keeper = owner.clone();
        let held = council
            .call(move || {
                let mut store = store.lock().expect("the object store lock is poisoned");
                store.put_keep(&keeper, &ids, ttl_ms, now_ms()).map_err(from_store)
            })
            .await?;
        let objects: Vec<_> = put
            .objects
            .iter()
            .zip(&held)
            .map(|(id, held)| json!({"id": id, "held": held}))
            .collect();
        let expires = held.iter().any(|h| *h).then_some(put.ttl_s);
        Ok(json!({"owner": owner, "objects": objects, "expires_in_s": expires}))
    }
    .await;
    answer(id, result)
}

async fn get_keep_here(council: &Council, id: &RequestId, owner: String) -> Response {
    let result = async {
        let store = store(council)?;
        owner_of(&owner)?;
        let keeper = owner.clone();
        let (ids, left_ms) = council
            .call(move || {
                Ok(store
                    .lock()
                    .expect("the object store lock is poisoned")
                    .keep(&keeper, now_ms()))
            })
            .await?
            .ok_or_else(|| not_found(format!("{owner} keeps nothing")))?;
        let objects: Vec<String> = ids.iter().map(|id| format!("blake3:{}", id.hex())).collect();
        Ok(json!({"owner": owner, "objects": objects, "expires_in_s": left_ms.div_ceil(1000)}))
    }
    .await;
    answer(id, result)
}

async fn delete_keep_here(council: &Council, id: &RequestId, owner: String) -> Response {
    let result = async {
        let store = store(council)?;
        owner_of(&owner)?;
        let keeper = owner.clone();
        let dropped = council
            .call(move || {
                let mut store = store.lock().expect("the object store lock is poisoned");
                store.drop_keep(&keeper, now_ms()).map_err(from_store)
            })
            .await?;
        if dropped {
            Ok(())
        } else {
            Err(not_found(format!("{owner} keeps nothing")))
        }
    }
    .await;
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(failure) => failure.respond(id),
    }
}
