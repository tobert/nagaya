//! Helpers for the in-process tests of the forward (spec 11.1a): a council with a cluster that a test sets by hand, and
//! stub peers on loopback that keep what they saw.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use http_body_util::BodyExt;
use nagaya::cluster::{Cluster, ClusterState};
use nagaya::gossip::{ClusterView, NodeView};
use nagaya::http::{Council, Served, router};
use serde_json::{Value, json};
use tower::ServiceExt;

#[path = "../council_support/mod.rs"]
mod council_support;
pub use council_support::Fixture;

/// The name of this node in the tests.
pub const OWN: &str = "host-a";

/// The name of the peer in the tests.
pub const PEER: &str = "host-b";

/// What a stub peer saw of one request.
#[derive(Clone, Debug)]
pub struct Seen {
    pub method: String,
    pub uri: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

/// How a stub peer answers.
#[derive(Clone)]
pub enum Answer {
    /// A whole body with a status and a content type.
    Whole(u16, &'static str, String),
    /// A stream that sends these chunks and then waits for ever.
    Hang(Vec<String>),
}

type Shared = (Answer, Arc<Mutex<Vec<Seen>>>);

async fn answer(State((answer, seen)): State<Shared>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = body.collect().await.unwrap().to_bytes();
    seen.lock().unwrap().push(Seen {
        method: parts.method.to_string(),
        uri: parts.uri.to_string(),
        headers: parts.headers,
        body,
    });
    match answer {
        Answer::Whole(status, content_type, text) => Response::builder()
            .status(StatusCode::from_u16(status).unwrap())
            .header("content-type", content_type)
            .header("x-peer", "stub")
            .body(Body::from(text))
            .unwrap(),
        Answer::Hang(chunks) => {
            let sent = futures_util::stream::iter(
                chunks
                    .into_iter()
                    .map(|chunk| Ok::<_, std::io::Error>(Bytes::from(chunk))),
            );
            let stream = futures_util::StreamExt::chain(sent, futures_util::stream::pending());
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }
    }
}

/// A stub peer on its own runtime.
pub struct Stub {
    pub http: SocketAddr,
    pub seen: Arc<Mutex<Vec<Seen>>>,
    _runtime: tokio::runtime::Runtime,
}

impl Stub {
    pub fn start(answer_with: Answer) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .fallback(answer)
            .with_state((answer_with, Arc::clone(&seen)));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let http = listener.local_addr().unwrap();
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            http,
            seen,
            _runtime: runtime,
        }
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

/// An address with no listener, so each connection is refused.
pub fn refused_addr() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

/// A council of the fixture's toy model `toy` on the node `host-a`, with a cluster that the test sets.
pub fn council(fixture: &Fixture) -> (Arc<Council>, Arc<Cluster>) {
    let served = Served {
        service: fixture.attach_again(),
        aliases: Vec::new(),
        device: None,
    };
    let cluster = Arc::new(Cluster::new(OWN));
    let council = Council::new(vec![served], None)
        .unwrap()
        .with_cluster(Arc::clone(&cluster));
    (Arc::new(council), cluster)
}

/// The gossip entry of the local toy model under another name: the same family, so the same tokenizer and template.
pub fn entry_like_local(council: &Council, model: &str) -> Value {
    let local = council.local_identity().models[0].clone();
    let mut value = serde_json::to_value(local).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("status");
    object.insert("model".to_owned(), json!(model));
    value
}

/// The peer node `name` with one engine `toy` that serves `entries`, in `state`, at `http`.
pub fn peer_node(name: &str, id: &str, http: SocketAddr, state: &str, entries: &[Value]) -> NodeView {
    let mut keys = BTreeMap::new();
    keys.insert(
        "engine:toy".to_owned(),
        json!({"identity": null, "models": [], "served": entries, "state": state, "http": http.to_string()})
            .to_string(),
    );
    keys.insert(
        "server".to_owned(),
        json!({"http": http.to_string(), "limits": {"case_bytes": 1, "choice_options": 2}, "capabilities": ["classes"]})
            .to_string(),
    );
    NodeView {
        name: name.to_owned(),
        id: id.to_owned(),
        generation: 1,
        addr: "127.0.0.1:9".parse().unwrap(),
        is_self: false,
        keys,
    }
}

/// This node in a view.
pub fn own_node() -> NodeView {
    NodeView {
        name: OWN.to_owned(),
        id: format!("{OWN}:1:127.0.0.1:1"),
        generation: 1,
        addr: "127.0.0.1:1".parse().unwrap(),
        is_self: true,
        keys: BTreeMap::new(),
    }
}

/// Set the cluster of `council` to this node and `peers`.
pub fn set_peers(council: &Council, cluster: &Cluster, peers: Vec<NodeView>) {
    let mut live = vec![own_node()];
    live.extend(peers);
    let view = ClusterView { live, dead: Vec::new() };
    let local: Vec<_> = council.served_by_engine().into_values().flatten().collect();
    cluster.set(ClusterState::observe(&cluster.state(), &view, OWN, &local));
}

/// One response through the router.
pub struct Reply {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|_| panic!("not JSON: {:?}", self.body))
    }
}

/// Send one request through the router on this thread, so that a capture of this thread sees its spans.
pub fn send(council: &Arc<Council>, method: &str, path: &str, body: &str, headers: &[(&str, &str)]) -> Reply {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = axum::http::Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let response = router(Arc::clone(council))
            .oneshot(request.body(Body::from(body.to_owned())).unwrap())
            .await
            .unwrap();
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        Reply { status, headers, body }
    })
}
