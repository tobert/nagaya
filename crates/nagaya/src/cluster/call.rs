//! The cluster call: each call from this node to a peer goes through this layer (spec 11.1a, "The cluster call").
//!
//! A routed call goes to the one peer that gossip names. A broadcast goes to each live peer at once. Each call carries
//! `nagaya-forwarded`, so the peer answers from its own store and calls no other node. A peer that does not connect in
//! [`CONNECT_LIMIT`] gets no call. A call has no other timeout, because the peer keeps its own deadlines.

use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;

use axum::body::Bytes;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode};
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;

/// The header that marks a call from a peer. Its value is the name of the node that sent it (spec 11.1a).
pub const FORWARDED_HEADER: &str = "nagaya-forwarded";

/// The header that carries the request id of the node that sent a call (spec 11.1a, "One request id").
pub const REQUEST_ID_HEADER: &str = "nagaya-request-id";

/// The longest wait for a connection to a peer (spec 11.1a).
pub const CONNECT_LIMIT: Duration = Duration::from_secs(2);

/// The largest answer body that a broadcast write reads, in bytes: the body limit of the routes (spec 11.7).
const ANSWER_BYTES: usize = crate::http::MAX_BODY_BYTES;

/// A peer that a call goes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peer {
    /// The node name of the peer.
    pub node: String,
    /// The HTTP address of the peer, from gossip (spec 2.3).
    pub http: SocketAddr,
}

/// One call to a peer: the request as it goes on the wire, without the headers of the cluster.
#[derive(Clone, Debug)]
pub struct Call {
    /// The method.
    pub method: Method,
    /// The path and the query.
    pub path: String,
    /// The headers to send. The layer adds `host`, `nagaya-forwarded` and `nagaya-request-id`.
    pub headers: HeaderMap,
    /// The body bytes.
    pub body: Bytes,
    /// The request id of this node for the request that makes the call.
    pub request_id: String,
}

/// Why a call to a peer got no answer.
#[derive(Debug)]
pub enum CallError {
    /// The connection did not open in [`CONNECT_LIMIT`], or the peer refused it.
    Refused {
        /// The node name of the peer.
        node: String,
        /// What failed.
        reason: String,
    },
    /// The connection opened and broke before the status of the answer.
    Lost {
        /// The node name of the peer.
        node: String,
        /// What failed.
        reason: String,
    },
}

impl fmt::Display for CallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallError::Refused { node, reason } => write!(formatter, "node {node} did not connect: {reason}"),
            CallError::Lost { node, reason } => {
                write!(
                    formatter,
                    "node {node} closed the connection before its answer: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for CallError {}

/// The calls of this node to its peers.
#[derive(Clone, Debug)]
pub struct Caller {
    /// The node name of this node: the value of `nagaya-forwarded`.
    own: String,
}

impl Caller {
    /// The caller of the node `own`.
    pub fn new(own: &str) -> Self {
        Self { own: own.to_owned() }
    }

    /// The node name of this node.
    pub fn own(&self) -> &str {
        &self.own
    }

    /// Send `call` to `peer` and give its answer as it comes. A 4xx or a 5xx is an answer. The body streams.
    pub async fn routed(&self, peer: &Peer, call: &Call) -> Result<Response<Incoming>, CallError> {
        let refused = |reason: String| CallError::Refused {
            node: peer.node.clone(),
            reason,
        };
        let stream = match tokio::time::timeout(CONNECT_LIMIT, tokio::net::TcpStream::connect(peer.http)).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(error)) => return Err(refused(format!("{}: {error}", peer.http))),
            Err(_) => return Err(refused(format!("{}: no connection in {CONNECT_LIMIT:?}", peer.http))),
        };
        // A small write must not wait for more bytes, because a forward carries a stream of events.
        let _ = stream.set_nodelay(true);
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|error| refused(format!("{}: {error}", peer.http)))?;
        // The connection task moves the bytes. It ends when the answer ends or when the answer is dropped.
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::debug!("a connection to a peer ended with an error: {error}");
            }
        });
        let request = self.request(peer, call);
        sender.send_request(request).await.map_err(|error| CallError::Lost {
            node: peer.node.clone(),
            reason: error.to_string(),
        })
    }

    /// Send `call` to each peer at once. Give the first answer that is not a 404, and drop the rest. Give `None` when
    /// each peer that answered gave a 404, or when no peer answered. A peer with no answer is left out and logged.
    pub async fn broadcast_read(&self, peers: &[Peer], call: &Call) -> Option<(Peer, Response<Incoming>)> {
        let mut calls: FuturesUnordered<_> = peers
            .iter()
            .map(|peer| async move { (peer, self.routed(peer, call).await) })
            .collect();
        while let Some((peer, answer)) = calls.next().await {
            match answer {
                Ok(response) if response.status() == StatusCode::NOT_FOUND => {}
                Ok(response) => return Some((peer.clone(), response)),
                Err(error) => left_out(&call.path, &error),
            }
        }
        None
    }

    /// Send `call` to each peer at once, and give each answer with its whole body, in the order of `peers`. The route
    /// joins them. A peer with no answer is left out and logged. A body past the body limit is left out too.
    pub async fn broadcast_write(&self, peers: &[Peer], call: &Call) -> Vec<(Peer, Response<Bytes>)> {
        let calls = peers.iter().map(|peer| async move {
            let response = match self.routed(peer, call).await {
                Ok(response) => response,
                Err(error) => {
                    left_out(&call.path, &error);
                    return None;
                }
            };
            let (parts, body) = response.into_parts();
            match Limited::new(body, ANSWER_BYTES).collect().await {
                Ok(body) => Some((peer.clone(), Response::from_parts(parts, body.to_bytes()))),
                Err(error) => {
                    let error = CallError::Lost {
                        node: peer.node.clone(),
                        reason: format!("the answer body did not arrive whole: {error}"),
                    };
                    left_out(&call.path, &error);
                    None
                }
            }
        });
        futures_util::future::join_all(calls)
            .await
            .into_iter()
            .flatten()
            .collect()
    }

    /// The request of `call` to `peer`, with the headers of the cluster.
    fn request(&self, peer: &Peer, call: &Call) -> Request<Full<Bytes>> {
        let mut request = Request::new(Full::new(call.body.clone()));
        *request.method_mut() = call.method.clone();
        *request.uri_mut() = call
            .path
            .parse()
            .unwrap_or_else(|error| panic!("the path `{}` of a call is not a URI: {error}", call.path));
        let headers = request.headers_mut();
        headers.clone_from(&call.headers);
        headers.insert(
            axum::http::header::HOST,
            HeaderValue::from_str(&peer.http.to_string()).expect("a socket address is a header value"),
        );
        headers.insert(HeaderName::from_static(FORWARDED_HEADER), header_value(&self.own));
        headers.insert(
            HeaderName::from_static(REQUEST_ID_HEADER),
            header_value(&call.request_id),
        );
        request
    }
}

/// A header value of a node name or a request id. Both are printable ASCII (spec 2.1, gogi "Request ids and
/// tracing"), so another value is a defect.
fn header_value(text: &str) -> HeaderValue {
    HeaderValue::from_str(text).unwrap_or_else(|error| panic!("`{text}` is not a header value: {error}"))
}

/// Log a peer that a broadcast left out (spec 11.1a).
fn left_out(path: &str, error: &CallError) {
    tracing::warn!("a call to the cluster for {path} left a peer out: {error}");
}
