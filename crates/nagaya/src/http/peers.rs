//! The calls of the routes with no model to the peers of the cluster (spec 11.1a, "The routes with no model", and 6.15,
//! "Objects in the cluster").
//!
//! A route answers from this node first. On a miss, a read asks each live peer through the cluster call. A write goes
//! to each live peer that declares the capability of the route, and the route joins the answers. A request that a peer
//! sent never calls another node.

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, header};
use axum::response::Response;
use gogi_proto::identity::Capability;
use hyper::body::Incoming;

use super::{Council, RequestId};
use crate::cluster::call::{Call, FORWARDED_HEADER, Peer};

/// True when a peer sent the request. Such a request is answered from this node alone (spec 11.1a).
pub(super) fn forwarded(headers: &HeaderMap) -> bool {
    headers.contains_key(FORWARDED_HEADER)
}

/// The live peers that a call of a route goes to: each peer with a `server` value, and with `capability` when the route
/// names one (spec 11.1a). A peer whose name another live node has is left out, because a call to it can reach either
/// node (spec 2.3).
pub(super) fn peers(council: &Council, capability: Option<Capability>) -> Vec<Peer> {
    let Some(cluster) = council.cluster() else {
        return Vec::new();
    };
    let mut peers: Vec<Peer> = cluster
        .state()
        .servers()
        .iter()
        .filter(|server| !server.duplicate)
        .filter(|server| capability.is_none_or(|capability| server.capabilities.contains(&capability)))
        .map(|server| Peer {
            node: server.node.clone(),
            http: server.http,
        })
        .collect();
    peers.sort_by(|left, right| left.node.cmp(&right.node));
    peers
}

/// A call of `method` on `path`, with the headers of `names` that `headers` has.
pub(super) fn call(
    method: Method,
    path: &str,
    headers: &HeaderMap,
    names: &[HeaderName],
    body: Bytes,
    id: &RequestId,
) -> Call {
    let mut sent = HeaderMap::new();
    for name in names {
        if let Some(value) = headers.get(name) {
            sent.insert(name.clone(), value.clone());
        }
    }
    Call {
        method,
        path: path.to_owned(),
        headers: sent,
        body,
        request_id: id.0.clone(),
    }
}

/// The answer of a peer as it came: its status, its headers and its body as a stream (spec 6.15, 11.1a). The headers
/// of one connection stay out, because this node frames its own response.
pub(super) fn pass_on(response: Response<Incoming>) -> Response {
    let (mut head, incoming) = response.into_parts();
    for name in [header::TRANSFER_ENCODING, header::CONNECTION] {
        head.headers.remove(name);
    }
    head.headers.remove("keep-alive");
    Response::from_parts(head, Body::new(incoming))
}

/// The answer of a read by id: `local` when this node holds the thing, or else the answer of the first peer that holds
/// it (spec 11.1a, "The cluster call"). `missed` tells whether `local` is a miss. With `probe`, each peer gets a `HEAD`
/// first, and the peer that holds the thing gets the request itself (spec 6.15, "The read"). With no hit, `local`
/// stays the answer.
pub(super) async fn read_on_miss(
    council: &Council,
    local: Response,
    missed: bool,
    probe: bool,
    capability: Option<Capability>,
    call: Call,
    headers: &HeaderMap,
) -> Response {
    if !missed || forwarded(headers) {
        return local;
    }
    let Some(cluster) = council.cluster() else {
        return local;
    };
    let peers = peers(council, capability);
    if peers.is_empty() {
        return local;
    }
    let caller = cluster.caller();
    if !probe {
        return match caller.broadcast_read(&peers, &call).await {
            Some((_, response)) => pass_on(response),
            None => local,
        };
    }
    let head = Call {
        method: Method::HEAD,
        headers: HeaderMap::new(),
        body: Bytes::new(),
        ..call.clone()
    };
    let Some((holder, _)) = caller.broadcast_read(&peers, &head).await else {
        return local;
    };
    match caller.routed(&holder, &call).await {
        Ok(response) => pass_on(response),
        Err(error) => {
            // The holder went away between the HEAD and the read. No other node is known to hold the thing.
            tracing::warn!("a read of {} on node {} got no answer: {error}", call.path, holder.node);
            local
        }
    }
}

/// True for a status that a read by id counts as a miss.
pub(super) fn is_miss(status: StatusCode) -> bool {
    status == StatusCode::NOT_FOUND
}

/// The answer of one node to a write that goes to each node.
pub(super) struct Answer {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
}

impl Answer {
    /// The answer as a response, as the node gave it.
    fn respond(self) -> Response {
        let mut response = Response::new(Body::from(self.body));
        *response.status_mut() = self.status;
        *response.headers_mut() = self.headers;
        response
    }

    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }
}

/// The answers of a write: `local`, the answer of this node when it takes part, and the answer of each live peer that
/// declares `capability` (spec 11.1a, "The routes with no model"). A request that a peer sent goes to no peer.
pub(super) async fn write_all(
    council: &Council,
    local: Option<Response>,
    call: Call,
    capability: Capability,
    headers: &HeaderMap,
) -> Vec<Answer> {
    let mut answers = Vec::new();
    if let Some(local) = local {
        let (head, body) = local.into_parts();
        let body = axum::body::to_bytes(body, super::MAX_BODY_BYTES)
            .await
            .expect("the answer of a local route is a whole body");
        answers.push(Answer {
            status: head.status,
            headers: head.headers,
            body,
        });
    }
    if forwarded(headers) {
        return answers;
    }
    let Some(cluster) = council.cluster() else {
        return answers;
    };
    let peers = peers(council, Some(capability));
    for (_, response) in cluster.caller().broadcast_write(&peers, &call).await {
        let (head, body) = response.into_parts();
        let mut headers = head.headers;
        for name in [header::TRANSFER_ENCODING, header::CONNECTION] {
            headers.remove(name);
        }
        answers.push(Answer {
            status: head.status,
            headers,
            body,
        });
    }
    answers
}

/// The first answer that is not a 2xx or a 404: the refusal of a node is the response (spec 11.1a, Proposed).
fn refusal(answers: &mut Vec<Answer>) -> Option<Answer> {
    let at = answers
        .iter()
        .position(|answer| !answer.status.is_success() && answer.status != StatusCode::NOT_FOUND)?;
    Some(answers.swap_remove(at))
}

/// The join of a `DELETE` on each node: a 204 when any node held the thing, and else the first 404 (spec 11.1a).
/// `none` is the response when no node answered.
pub(super) fn join_delete(mut answers: Vec<Answer>, none: impl FnOnce() -> Response) -> Response {
    if let Some(refused) = refusal(&mut answers) {
        return refused.respond();
    }
    if let Some(at) = answers.iter().position(|answer| answer.status.is_success()) {
        return answers.swap_remove(at).respond();
    }
    answers.into_iter().next().map_or_else(none, Answer::respond)
}

/// The join of the sets of pins or keeps of an owner, under the member `key` (spec 11.1a, "The routes with no model").
///
/// A `PUT` gives each id of the request once, held when any node held it. A `GET` gives the union of the held ids.
/// `expires_in_s` is the smallest that a node gave, and a `PUT` with no held id gives `null`. An owner that no node
/// knows is the first 404. `none` is the response when no node answered.
pub(super) fn join_sets(mut answers: Vec<Answer>, key: &str, put: bool, none: impl FnOnce() -> Response) -> Response {
    if let Some(refused) = refusal(&mut answers) {
        return refused.respond();
    }
    let sets: Vec<serde_json::Value> = answers
        .iter()
        .filter(|answer| answer.status.is_success())
        .map(Answer::json)
        .collect();
    let Some(first) = sets.first() else {
        return answers.into_iter().next().map_or_else(none, Answer::respond);
    };
    let mut joined = first.clone();
    let expires = sets.iter().filter_map(|set| set["expires_in_s"].as_u64()).min();
    joined["expires_in_s"] = expires.map_or(serde_json::Value::Null, serde_json::Value::from);
    if put {
        let held = |at: usize| sets.iter().any(|set| set[key][at]["held"] == true);
        if let Some(entries) = joined[key].as_array_mut() {
            for (at, entry) in entries.iter_mut().enumerate() {
                entry["held"] = serde_json::Value::Bool(held(at));
            }
        }
    } else {
        let mut ids: Vec<serde_json::Value> = Vec::new();
        for id in sets
            .iter()
            .flat_map(|set| set[key].as_array().cloned().unwrap_or_default())
        {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        joined[key] = serde_json::Value::Array(ids);
    }
    super::routes::json(StatusCode::OK, &joined)
}
