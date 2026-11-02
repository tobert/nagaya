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
    call: Call,
    headers: &HeaderMap,
) -> Response {
    if !missed || forwarded(headers) {
        return local;
    }
    let Some(cluster) = council.cluster() else {
        return local;
    };
    let peers = peers(council, None);
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
