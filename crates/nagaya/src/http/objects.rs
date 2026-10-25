//! The routes of objects and keeps (gogi "Objects and keeps", spec 6.15).
//!
//! A put streams the body to a staged file and hashes it as it arrives, so a large object never sits in memory. The
//! store's index lives behind a mutex, and each call holds it only to look up or commit, never during a transfer.

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, Path, State};
use axum::http::{HeaderMap, StatusCode, header};
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
use super::routes::{answer, bytes, path_id};
use super::{Council, RequestId};
use crate::council::error::ServiceError;

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

pub(super) fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/council/v2/objects/{id}", put(put_object).get(get_object))
        .route(
            "/council/v2/keeps/{owner}",
            put(put_keep).get(get_keep).delete(delete_keep),
        )
}

/// The most bytes past `object_bytes` that a refused put reads and drops before its 413, so that the client is not
/// still sending when the answer comes. A body past even this gets its 413 before it is read.
const DRAIN_BYTES: u64 = 64 << 20;

/// Read and drop the rest of a body, up to [`DRAIN_BYTES`].
async fn drain(stream: &mut axum::body::BodyDataStream) {
    let mut dropped: u64 = 0;
    while dropped < DRAIN_BYTES {
        match stream.next().await {
            Some(Ok(chunk)) => dropped += chunk.len() as u64,
            _ => break,
        }
    }
}

fn now_ms() -> u64 {
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

fn from_store(error: ObjectError) -> Failure {
    match error {
        ObjectError::TooLarge { limit, .. } => too_large(limit, error.to_string()),
        ObjectError::Io(message) => Failure::from(ServiceError::internal(message)),
    }
}

/// The store of the process, or the 404 of a profile with no objects budget.
fn store(council: &Council) -> Result<Arc<Mutex<ObjectStore>>, Failure> {
    council.objects().ok_or_else(|| {
        Failure::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            "the server holds no objects: its profile names no `objects` budget in `[store]` (spec 6.15)",
        )
    })
}

/// A whole content id from a path, as the store names it.
fn content(id: &str) -> Result<ContentId, Failure> {
    path_id(id)?;
    ContentId::from_hex(id.trim_start_matches("blake3:"))
        .map_err(|e| Failure::from(ServiceError::internal(format!("{e:?}"))))
}

async fn put_object(
    State(council): Shared,
    Extension(id): Id,
    Path(text): Path<String>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let result = async {
        let store = store(&council)?;
        let content = content(&text)?;
        let mime = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).ok_or_else(|| {
            invalid("content-type", "nothing is held: the request has no content-type header; send the object's media type, such as `audio/wav`".to_owned())
        })?;
        if let Some(problem) = mime_problem(mime) {
            return Err(invalid("content-type", format!("nothing is held: content-type {problem}")));
        }
        let (limit, staged) = {
            let mut store = store.lock().expect("the object store lock is poisoned");
            (store.object_bytes(), store.staging_path())
        };
        let declared = headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok());
        let mut stream = body.into_data_stream();
        if let Some(n) = declared.filter(|n| *n > limit) {
            if n <= limit + DRAIN_BYTES {
                drain(&mut stream).await;
            }
            return Err(too_large("object_bytes", format!("nothing is held: the object is {n} bytes; object_bytes is {limit}")));
        }
        // Stream the body to the staged file, and hash it as it arrives.
        let mut file = tokio::fs::File::create(&staged).await.map_err(|e| Failure::from(ServiceError::internal(format!("cannot make `{}`: {e}", staged.display()))))?;
        let mut hasher = blake3::Hasher::new();
        let mut size: u64 = 0;
        let outcome = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| invalid("body", format!("nothing is held: the body did not arrive whole ({e}); send the request again")))?;
                size += chunk.len() as u64;
                if size > limit {
                    drain(&mut stream).await;
                    return Err(too_large("object_bytes", format!("nothing is held: the object is more than {limit} bytes, which is object_bytes")));
                }
                hasher.update(&chunk);
                file.write_all(&chunk).await.map_err(|e| Failure::from(ServiceError::internal(format!("cannot write `{}`: {e}", staged.display()))))?;
            }
            file.sync_all().await.map_err(|e| Failure::from(ServiceError::internal(format!("cannot write `{}`: {e}", staged.display()))))?;
            let actual = format!("blake3:{}", hasher.finalize().to_hex());
            if actual != text {
                return Err(invalid("id", format!("nothing is held: the {size} bytes sent hash to {actual}, not {text}; PUT them at {actual}, or send the bytes {text} names")));
            }
            Ok(())
        }
        .await;
        if let Err(failure) = outcome {
            let _ = tokio::fs::remove_file(&staged).await;
            return Err(failure);
        }
        let mime = mime.to_owned();
        let committed = tokio::task::spawn_blocking(move || store.lock().expect("the object store lock is poisoned").commit(&content, &mime, &staged, size, now_ms()))
            .await
            .map_err(|e| Failure::from(ServiceError::internal(format!("the commit of an object stopped: {e}"))))?
            .map_err(from_store)?;
        let status = if committed.created { StatusCode::CREATED } else { StatusCode::OK };
        Ok((status, json!({"id": text, "mime": committed.mime, "bytes": size})))
    }
    .await;
    match result {
        Ok((status, value)) => (status, axum::Json(value)).into_response(),
        Err(failure) => failure.respond(&id),
    }
}

async fn get_object(
    State(council): Shared,
    Extension(id): Id,
    Path(text): Path<String>,
    headers: HeaderMap,
) -> Response {
    let result = async {
        let store = store(&council)?;
        if text.starts_with("pending:") {
            // The nagaya writes no outputs yet, so it gives no pending ids (spec 11.1, "Text only").
            return Err(match gogi_proto::ids::pending_id_problem(&text) {
                Some(problem) => invalid("id", format!("nothing was looked up: {problem}")),
                None => Failure::new(
                    StatusCode::NOT_FOUND,
                    ErrorType::NotFound,
                    format!("no output is pending as {text}"),
                ),
            });
        }
        let content = content(&text)?;
        let found = store
            .lock()
            .expect("the object store lock is poisoned")
            .get(&content, now_ms())
            .ok_or_else(|| {
                Failure::new(
                    StatusCode::NOT_FOUND,
                    ErrorType::NotFound,
                    format!("no object {text} is held; PUT it again"),
                )
            })?;
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
                return Err(failure);
            }
        };
        let mut file = tokio::fs::File::open(&found.path).await.map_err(|e| {
            Failure::from(ServiceError::internal(format!(
                "cannot read `{}`: {e}",
                found.path.display()
            )))
        })?;
        file.seek(std::io::SeekFrom::Start(start)).await.map_err(|e| {
            Failure::from(ServiceError::internal(format!(
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

async fn put_keep(State(council): Shared, Extension(id): Id, Path(owner): Path<String>, body: Body) -> Response {
    let result = async {
        let store = store(&council)?;
        owner_of(&owner)?;
        let put: KeepsPut = parse(&bytes(body).await?, "keeps")?;
        put.validate().map_err(|e| Failure::from(ServiceError::Invalid(e)))?;
        let ids: Vec<ContentId> = put.objects.iter().map(|text| content(text)).collect::<Result<_, _>>()?;
        let held = store
            .lock()
            .expect("the object store lock is poisoned")
            .put_keep(&owner, &ids, put.ttl_s * 1000, now_ms())
            .map_err(from_store)?;
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
    answer(&id, result)
}

async fn get_keep(State(council): Shared, Extension(id): Id, Path(owner): Path<String>) -> Response {
    let result = async {
        let store = store(&council)?;
        owner_of(&owner)?;
        let (ids, left_ms) = store
            .lock()
            .expect("the object store lock is poisoned")
            .keep(&owner, now_ms())
            .ok_or_else(|| {
                Failure::new(
                    StatusCode::NOT_FOUND,
                    ErrorType::NotFound,
                    format!("{owner} keeps nothing"),
                )
            })?;
        let objects: Vec<String> = ids.iter().map(|id| format!("blake3:{}", id.hex())).collect();
        Ok(json!({"owner": owner, "objects": objects, "expires_in_s": left_ms.div_ceil(1000)}))
    }
    .await;
    answer(&id, result)
}

async fn delete_keep(State(council): Shared, Extension(id): Id, Path(owner): Path<String>) -> Response {
    let result = async {
        let store = store(&council)?;
        owner_of(&owner)?;
        if store
            .lock()
            .expect("the object store lock is poisoned")
            .drop_keep(&owner, now_ms())
            .map_err(from_store)?
        {
            Ok(())
        } else {
            Err(Failure::new(
                StatusCode::NOT_FOUND,
                ErrorType::NotFound,
                format!("{owner} keeps nothing"),
            ))
        }
    }
    .await;
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(failure) => failure.respond(&id),
    }
}
