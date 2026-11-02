//! The forward of a request to the node that holds its served model (spec 11.1a, "The forward").
//!
//! A layer in front of the job routes and the text routes reads `model` from the body. A model of this node runs here.
//! A model of a peer goes to that peer through the cluster call, and the answer comes back as it is: the status, the
//! headers and the body, event by event for a stream. The forward runs on the control runtime and takes no place among
//! the calls of 11.7.

use std::pin::Pin;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;
use futures_util::StreamExt;
use gogi_proto::error::{ErrorBody, ErrorType};
use gogi_proto::identity::{ModelStatus, ServedModel};
use gogi_proto::ids::TRACEPARENT_HEADER;
use http_body_util::BodyExt;
use tracing::Instrument;

use super::{Council, Failure, RequestId, TRACESTATE_HEADER};
use crate::cluster::call::{CONNECT_LIMIT, Call, CallError, FORWARDED_HEADER, Peer};
use crate::cluster::{Cluster, PeerModel};
use crate::council::error::ServiceError;

/// The job routes. They carry `model`, and they forward (spec 11.1a).
const JOB_ROUTES: [&str; 5] = [
    "/council/v2/contexts",
    "/council/v2/decisions",
    "/council/v2/reads",
    "/council/v2/generations",
    "/council/v2/embeddings",
];

/// The text routes. They run here when a local family has the model, and else they forward (spec 11.1a).
const TEXT_ROUTES: [&str; 3] = ["/nagaya/v1/render", "/nagaya/v1/tokenize", "/nagaya/v1/detokenize"];

/// Where a request for a served model runs.
#[derive(Debug)]
pub(crate) enum Place {
    /// On this node.
    Local,
    /// On the peer that holds the model.
    Peer(Box<PeerModel>),
    /// No node serves the name. The local route gives its 400.
    Unknown,
}

/// The layer of the forward. Each other request goes on to its route.
pub(super) async fn forward(State(council): State<Arc<Council>>, request: Request, next: Next) -> Response {
    if request.method() != Method::POST {
        return next.run(request).await;
    }
    let path = request.uri().path();
    let text = TEXT_ROUTES.contains(&path);
    if !text && !JOB_ROUTES.contains(&path) {
        return next.run(request).await;
    }
    let Some(cluster) = council.cluster().cloned() else {
        return next.run(request).await;
    };
    let id = request
        .extensions()
        .get::<RequestId>()
        .cloned()
        .expect("the observe layer sets the request id");
    let (parts, body) = request.into_parts();
    let bytes = match super::routes::bytes(body).await {
        Ok(bytes) => bytes,
        Err(failure) => return failure.respond(&id),
    };
    let local = |parts, bytes: Bytes| next.run(Request::from_parts(parts, Body::from(bytes)));
    // A body that does not decode, or that has no `model`, gets the 400 of its route.
    let Some(model) = model_of(&bytes) else {
        return local(parts, bytes).await;
    };
    let forwarded = parts.headers.contains_key(FORWARDED_HEADER);
    match council.place(&model, text) {
        Place::Local => local(parts, bytes).await,
        Place::Unknown if !forwarded => local(parts, bytes).await,
        Place::Unknown | Place::Peer(_) if forwarded => {
            // Gossip can lag, so two nodes with old views must not send a request back and forth (spec 11.1a).
            let from = parts.headers[FORWARDED_HEADER].to_str().unwrap_or("?").to_owned();
            not_served(
                &model,
                &format!(
                    "nothing ran: model {model} is not served on this node, and a request that node {from} forwarded \
                 does not forward again; retry when identity reports it ready"
                ),
            )
            .respond(&id)
        }
        Place::Unknown => unreachable!("a request that was not forwarded matched the arm above"),
        Place::Peer(peer) => {
            if council.draining() {
                return Failure::from(ServiceError::Stopping)
                    .with_message("nothing ran: the server is stopping; send the request to another node")
                    .respond(&id);
            }
            if peer.entry.status != ModelStatus::Ready {
                return Failure::unavailable(&peer.entry).respond(&id);
            }
            if peer.duplicate {
                return unreachable_peer(
                    &peer,
                    &format!(
                        "nothing ran: model {model} is on node {}, and another live node has that name (spec 2.3); \
                     rename one node",
                        peer.node
                    ),
                )
                .respond(&id);
            }
            forward_to(cluster, *peer, parts, bytes, id).await
        }
    }
}

/// The `model` of a JSON body, when it has one.
fn model_of(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value.get("model")?.as_str().map(str::to_owned)
}

/// The 503 `model_unavailable` with the status `unloaded` of a name that this node does not serve.
fn not_served(model: &str, message: &str) -> Failure {
    let mut body = ErrorBody::new(ErrorType::ModelUnavailable, message);
    body.error.model = Some(model.to_owned());
    body.error.status = Some(ModelStatus::Unloaded);
    Failure {
        status: StatusCode::SERVICE_UNAVAILABLE,
        body: Box::new(body),
    }
}

/// The 503 `model_unavailable` with the status `unloaded` of a peer model that this node cannot reach.
fn unreachable_peer(peer: &PeerModel, message: &str) -> Failure {
    let mut entry: ServedModel = peer.entry.clone();
    entry.status = ModelStatus::Unloaded;
    Failure::unavailable(&entry).with_message(message)
}

/// How a forward ended (spec 9.2, "A forwarded request").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Done,
    Refused,
    Lost,
    Cancelled,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Done => "done",
            Outcome::Refused => "refused",
            Outcome::Lost => "lost",
            Outcome::Cancelled => "cancelled",
        }
    }
}

/// The span of one forward. It records `cancelled` when it drops before another outcome.
struct ForwardSpan {
    span: tracing::Span,
    ended: bool,
    node: String,
    route: String,
    request_id: String,
}

impl ForwardSpan {
    fn end(&mut self, outcome: Outcome) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.span.record("outcome", outcome.as_str());
        if matches!(outcome, Outcome::Refused | Outcome::Lost) {
            self.span.record("otel.status_code", "ERROR");
            tracing::warn!(
                parent: &self.span,
                "the forward to node {} for {} ended {} (request {})",
                self.node,
                self.route,
                outcome.as_str(),
                self.request_id
            );
        }
    }
}

impl Drop for ForwardSpan {
    fn drop(&mut self) {
        self.end(Outcome::Cancelled);
    }
}

/// The future that ends when the peer leaves the live set (spec 11.1a).
type Lost = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

fn lost(cluster: &Cluster, node_id: String) -> Lost {
    let mut live = cluster.watch_live();
    Box::pin(async move {
        // A closed channel means the cluster is gone, so the peer is gone too.
        let _ = live.wait_for(|ids| !ids.contains(&node_id)).await;
    })
}

/// Send the request to the peer, and give its answer as it comes.
async fn forward_to(
    cluster: Arc<Cluster>,
    peer: PeerModel,
    parts: axum::http::request::Parts,
    body: Bytes,
    id: RequestId,
) -> Response {
    let span = tracing::info_span!(
        "nagaya.forward",
        nagaya.peer = %peer.node,
        http.response.status_code = tracing::field::Empty,
        outcome = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    let mut record = ForwardSpan {
        span: span.clone(),
        ended: false,
        node: peer.node.clone(),
        route: parts.uri.path().to_owned(),
        request_id: id.0.clone(),
    };
    let mut headers = HeaderMap::new();
    for name in [
        header::CONTENT_TYPE,
        header::ACCEPT,
        HeaderName::from_static(TRACESTATE_HEADER),
    ] {
        if let Some(value) = parts.headers.get(&name) {
            headers.insert(name, value.clone());
        }
    }
    // The forward names its own span as the parent of the target's span (spec 9.2). With no trace layer, the span has
    // no trace context, and the caller's header goes on as it came.
    match nagaya_telemetry::traceparent_of(&span) {
        Some(traceparent) => {
            let value = HeaderValue::from_str(&traceparent).expect("a traceparent is a header value");
            headers.insert(TRACEPARENT_HEADER, value);
        }
        None => {
            if let Some(value) = parts.headers.get(TRACEPARENT_HEADER) {
                headers.insert(TRACEPARENT_HEADER, value.clone());
            }
        }
    }
    let call = Call {
        method: parts.method.clone(),
        path: parts
            .uri
            .path_and_query()
            .map_or_else(|| parts.uri.path().to_owned(), |path| path.as_str().to_owned()),
        headers,
        body,
        request_id: id.0.clone(),
    };
    let target = Peer {
        node: peer.node.clone(),
        http: peer.http,
    };
    let mut gone = lost(&cluster, peer.node_id.clone());
    let answer = tokio::select! {
        answer = cluster.caller().routed(&target, &call).instrument(span.clone()) => answer,
        () = &mut gone => Err(CallError::Lost {
            node: peer.node.clone(),
            reason: "the node left the live set".to_owned(),
        }),
    };
    let response = match answer {
        Ok(response) => response,
        Err(error) => {
            let (outcome, message) = match &error {
                CallError::Refused { .. } => (
                    Outcome::Refused,
                    format!(
                        "nothing ran: node {} holds model {}, and it did not connect in {CONNECT_LIMIT:?}; retry when \
                         identity reports it ready",
                        peer.node, peer.entry.model
                    ),
                ),
                CallError::Lost { .. } => (
                    Outcome::Lost,
                    format!(
                        "model {} on node {} is not reachable: {error}; retry when identity reports it ready",
                        peer.entry.model, peer.node
                    ),
                ),
            };
            let failure = unreachable_peer(&peer, &message);
            span.record("http.response.status_code", failure.status.as_u16());
            record.end(outcome);
            return failure.respond(&id);
        }
    };
    span.record("http.response.status_code", response.status().as_u16());
    let stream = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    let failure = unreachable_peer(
        &peer,
        &format!(
            "model {} on node {} is not reachable: the node left the live set during the reply",
            peer.entry.model, peer.node
        ),
    );
    let (mut head, incoming) = response.into_parts();
    for name in [header::TRANSFER_ENCODING, header::CONNECTION] {
        head.headers.remove(name);
    }
    head.headers.remove("keep-alive");
    let relay = Relay {
        frames: incoming.into_data_stream().boxed(),
        gone,
        record,
        error_event: stream.then(|| error_event(&failure, &id)),
    };
    Response::from_parts(head, Body::from_stream(relay.into_stream()))
}

/// The `error` event of a stream whose peer left: the 503 body with the request id (gogi "Streams").
fn error_event(failure: &Failure, id: &RequestId) -> Bytes {
    let body = (*failure.body).clone().with_request_id(id.0.clone());
    let data = serde_json::to_string(&body).expect("an error body is JSON");
    Bytes::from(format!("event: error\ndata: {data}\n\n"))
}

/// The next step of a relay.
enum Step {
    /// Bytes of the peer.
    Frame(Bytes),
    /// The peer ended its reply.
    End,
    /// The peer left the live set, or its connection broke.
    Broken,
}

/// The body of a forward: the frames of the peer, until the peer leaves the live set.
struct Relay {
    frames: futures_util::stream::BoxStream<'static, Result<Bytes, hyper::Error>>,
    gone: Lost,
    record: ForwardSpan,
    /// The last event of a stream whose peer leaves. A body that is not a stream closes instead.
    error_event: Option<Bytes>,
}

impl Relay {
    fn into_stream(self) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send {
        futures_util::stream::unfold(Some(self), |relay| async move {
            let mut relay = relay?;
            let step = tokio::select! {
                frame = relay.frames.next() => match frame {
                    Some(Ok(bytes)) => Step::Frame(bytes),
                    None => Step::End,
                    Some(Err(_)) => Step::Broken,
                },
                () = &mut relay.gone => Step::Broken,
            };
            match step {
                Step::Frame(bytes) => Some((Ok(bytes), Some(relay))),
                Step::End => {
                    relay.record.end(Outcome::Done);
                    None
                }
                Step::Broken => {
                    relay.record.end(Outcome::Lost);
                    match relay.error_event.take() {
                        Some(event) => Some((Ok(event), None)),
                        None => Some((
                            Err(std::io::Error::other(
                                "the peer of a forward left before the reply ended",
                            )),
                            None,
                        )),
                    }
                }
            }
        })
    }
}
