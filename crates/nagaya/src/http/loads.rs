//! The load routes (spec 3.3b): `POST /nagaya/v1/engines/{name}/load` and `POST /nagaya/v1/engines/{name}/unload`.
//!
//! A route asks the group thread of the engine to plan a swap. The answer comes when the group plans it, not when the
//! swap ends. The status feed and identity show the states that follow (spec 9.4, 11.1).

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Extension, Path, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::Response;
use axum::routing::post;
use gogi_proto::error::ErrorType;
use serde::{Deserialize, Serialize};

use super::errors::Failure;
use super::peers;
use super::routes::json;
use super::{Council, RequestId};
use crate::cluster::call::Peer;
use crate::gossip::Key;
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

async fn load(
    State(council): Shared,
    Extension(id): Id,
    Path(name): Path<String>,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let bytes = match super::routes::bytes(body).await {
        Ok(bytes) => bytes,
        Err(failure) => return failure.respond(&id),
    };
    match node_of(&council, &name, &uri, &headers) {
        Ok(None) => {}
        Ok(Some(peer)) => return forward(&council, peer, &name, &uri, &headers, bytes, &id).await,
        Err(failure) => return failure.respond(&id),
    }
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

async fn unload(
    State(council): Shared,
    Extension(id): Id,
    Path(name): Path<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    match node_of(&council, &name, &uri, &headers) {
        Ok(None) => ask(council, id, name, None).await,
        Ok(Some(peer)) => forward(&council, peer, &name, &uri, &headers, Bytes::new(), &id).await,
        Err(failure) => failure.respond(&id),
    }
}

/// The peer that holds the engine `name` of a load route, or `None` for this node (spec 3.3b, "The node of a load
/// route"). The query `host` names the node. Without it, an engine of this node comes first, then the one live peer
/// that publishes `engine:<name>`. A request that a peer sent runs here.
fn node_of(council: &Council, name: &str, uri: &Uri, headers: &HeaderMap) -> Result<Option<Peer>, Failure> {
    if peers::forwarded(headers) {
        return Ok(None);
    }
    let Some(cluster) = council.cluster() else {
        return Ok(None);
    };
    let host_failure = |status, kind, message: String| {
        let mut failure = Failure::new(status, kind, message);
        failure.body.error.param = Some("host".to_owned());
        failure
    };
    let state = cluster.state();
    let live: Vec<_> = state.view().live.iter().filter(|node| !node.is_self).collect();
    let node = match peers::query_value(uri, "host") {
        Some(host) if host == cluster.own() => return Ok(None),
        Some(host) => {
            if !live.iter().any(|node| node.name == host) {
                return Err(host_failure(
                    StatusCode::NOT_FOUND,
                    ErrorType::NotFound,
                    format!(
                        "nothing changed: no live node has the name `{host}`; GET /nagaya/v1/cluster lists the nodes"
                    ),
                ));
            }
            host.to_owned()
        }
        None => {
            if council.loads().is_some_and(|ports| ports.engine_state(name).is_some()) {
                return Ok(None);
            }
            let key = Key::Engine(name.to_owned()).name();
            let mut holders: Vec<&str> = live
                .iter()
                .filter(|node| node.keys.contains_key(&key))
                .map(|node| node.name.as_str())
                .collect();
            holders.sort_unstable();
            holders.dedup();
            match holders.as_slice() {
                [] => return Ok(None),
                [holder] => (*holder).to_owned(),
                _ => {
                    return Err(host_failure(
                        StatusCode::BAD_REQUEST,
                        ErrorType::InvalidRequest,
                        format!(
                            "nothing changed: the nodes {} each have an engine `{name}`; name one with the query `host`",
                            holders.join(", ")
                        ),
                    ));
                }
            }
        }
    };
    let unavailable = |why: &str| {
        Failure::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::Unavailable,
            format!("nothing changed: node {node} holds engine `{name}`, and {why}; retry later"),
        )
    };
    match state.servers().iter().find(|server| server.node == node) {
        Some(server) if server.duplicate => Err(unavailable("another live node has its name (spec 2.3)")),
        Some(server) => Ok(Some(Peer {
            node: server.node.clone(),
            http: server.http,
        })),
        None => Err(unavailable("it publishes no HTTP address in gossip yet")),
    }
}

/// Send a load route to the peer that holds the engine, with its body. Its answer comes back as it is (spec 3.3b). A
/// peer that does not connect is a 503 `unavailable` that names it (spec 11.1a, Proposed).
async fn forward(
    council: &Council,
    peer: Peer,
    name: &str,
    uri: &Uri,
    headers: &HeaderMap,
    body: Bytes,
    id: &RequestId,
) -> Response {
    let cluster = council.cluster().expect("a load route forwards only with a cluster");
    let call = peers::call(
        Method::POST,
        &peers::raw_path(uri),
        headers,
        &[header::CONTENT_TYPE],
        body,
        id,
    );
    match cluster.caller().routed(&peer, &call).await {
        Ok(response) => peers::pass_on(response),
        Err(error) => Failure::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::Unavailable,
            format!(
                "nothing changed: node {} holds engine `{name}`, and it did not answer: {error}; retry later",
                peer.node
            ),
        )
        .respond(id),
    }
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
        LoadError::Pinned(_) | LoadError::NoClose(_) | LoadError::InTheWay { .. } | LoadError::Busy(_) => {
            named(StatusCode::CONFLICT, ErrorType::InvalidRequest, "name")
        }
        LoadError::BadUnload(_) => named(StatusCode::BAD_REQUEST, ErrorType::InvalidRequest, "unload"),
        LoadError::Stopping | LoadError::Stopped => {
            Failure::new(StatusCode::SERVICE_UNAVAILABLE, ErrorType::Unavailable, message)
        }
    }
}
