//! The cluster call of spec 11.1a: the routed form and the two broadcast forms, against stub peers on loopback.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt;
use nagaya::cluster::call::{CONNECT_LIMIT, Call, CallError, Caller, FORWARDED_HEADER, Peer, REQUEST_ID_HEADER};

/// What a stub peer saw of one request.
#[derive(Clone, Debug)]
struct Seen {
    method: String,
    uri: String,
    headers: HeaderMap,
    body: Bytes,
}

/// What a stub peer answers, and where it keeps what it saw.
type Behavior = (u16, String, Arc<Mutex<Vec<Seen>>>);

/// A stub peer: it answers each request with one status and body, and keeps what it saw.
struct Stub {
    peer: Peer,
    seen: Arc<Mutex<Vec<Seen>>>,
}

async fn answer(State((status, text, seen)): State<Behavior>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = body.collect().await.unwrap().to_bytes();
    seen.lock().unwrap().push(Seen {
        method: parts.method.to_string(),
        uri: parts.uri.to_string(),
        headers: parts.headers,
        body,
    });
    (StatusCode::from_u16(status).unwrap(), text).into_response()
}

fn stub(runtime: &tokio::runtime::Runtime, node: &str, status: u16, text: &str) -> Stub {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .fallback(answer)
        .with_state((status, text.to_owned(), Arc::clone(&seen)));
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let http = listener.local_addr().unwrap();
    runtime.spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(listener, app).await.unwrap();
    });
    Stub {
        peer: Peer {
            node: node.to_owned(),
            http,
        },
        seen,
    }
}

/// A peer whose port has no listener, so each connection is refused.
fn refused(node: &str) -> Peer {
    let http: SocketAddr = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    Peer {
        node: node.to_owned(),
        http,
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

fn call(method: Method, path: &str, body: &str) -> Call {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().unwrap());
    headers.insert("accept", "text/event-stream".parse().unwrap());
    Call {
        method,
        path: path.to_owned(),
        headers,
        body: Bytes::from(body.to_owned()),
        request_id: "req_test_1".to_owned(),
    }
}

async fn text(response: Response<hyper::body::Incoming>) -> (u16, String) {
    let status = response.status().as_u16();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

#[test]
fn a_routed_call_sends_the_request_as_it_came_with_the_headers_of_the_cluster() {
    let runtime = runtime();
    let peer = stub(&runtime, "host-b", 400, "{\"error\":1}");
    let caller = Caller::new("host-a");
    let sent = call(Method::POST, "/council/v2/generations?x=1", "{\"model\":\"m\"}");
    let (status, body) = runtime.block_on(async { text(caller.routed(&peer.peer, &sent).await.unwrap()).await });
    // A 4xx is the answer of the peer. The layer does not hide it.
    assert_eq!((status, body.as_str()), (400, "{\"error\":1}"));
    let seen = peer.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "one call and no retry");
    let seen = &seen[0];
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.uri, "/council/v2/generations?x=1");
    assert_eq!(seen.body, Bytes::from_static(b"{\"model\":\"m\"}"));
    assert_eq!(seen.headers["content-type"], "application/json");
    assert_eq!(seen.headers["accept"], "text/event-stream");
    assert_eq!(seen.headers[FORWARDED_HEADER], "host-a");
    assert_eq!(seen.headers[REQUEST_ID_HEADER], "req_test_1");
}

#[test]
fn a_routed_call_to_a_peer_that_does_not_connect_is_refused_within_the_limit() {
    let runtime = runtime();
    let caller = Caller::new("host-a");
    let started = Instant::now();
    let error = runtime
        .block_on(caller.routed(&refused("host-gone"), &call(Method::GET, "/x", "")))
        .unwrap_err();
    assert!(started.elapsed() < CONNECT_LIMIT + Duration::from_secs(1));
    match error {
        CallError::Refused { node, .. } => assert_eq!(node, "host-gone"),
        other => panic!("want a refused call, got {other:?}"),
    }
    assert_eq!(CONNECT_LIMIT, Duration::from_secs(2), "spec 11.1a");
}

#[test]
fn a_broadcast_read_takes_the_first_answer_that_is_not_a_404() {
    let runtime = runtime();
    let miss = stub(&runtime, "host-b", 404, "miss");
    let hit = stub(&runtime, "host-c", 200, "hit");
    let caller = Caller::new("host-a");
    let peers = [refused("host-gone"), miss.peer.clone(), hit.peer.clone()];
    let (peer, response) = runtime
        .block_on(caller.broadcast_read(&peers, &call(Method::GET, "/council/v2/objects/blake3:00", "")))
        .expect("one peer holds the id");
    assert_eq!(peer, hit.peer);
    assert_eq!(runtime.block_on(text(response)), (200, "hit".to_owned()));
    // Each live peer got the call at once. The read returns at the hit, so the miss can still be answering: the test
    // waits for it, at most 5 s. Under load, a check at once failed (2026-11-07 and 2026-11-08).
    let waited = std::time::Instant::now();
    while miss.seen.lock().unwrap().is_empty() && waited.elapsed() < std::time::Duration::from_secs(5) {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(miss.seen.lock().unwrap().len(), 1);
    assert_eq!(hit.seen.lock().unwrap()[0].headers[FORWARDED_HEADER], "host-a");
}

#[test]
fn a_broadcast_read_takes_a_5xx_as_an_answer() {
    let runtime = runtime();
    let miss = stub(&runtime, "host-b", 404, "miss");
    let failed = stub(&runtime, "host-c", 500, "failed");
    let caller = Caller::new("host-a");
    let (peer, response) = runtime
        .block_on(caller.broadcast_read(&[miss.peer.clone(), failed.peer.clone()], &call(Method::GET, "/x", "")))
        .expect("a 500 is an answer");
    assert_eq!(peer, failed.peer);
    assert_eq!(runtime.block_on(text(response)).0, 500);
}

#[test]
fn a_broadcast_read_is_a_miss_only_when_each_peer_gave_a_404() {
    let runtime = runtime();
    let one = stub(&runtime, "host-b", 404, "miss");
    let two = stub(&runtime, "host-c", 404, "miss");
    let caller = Caller::new("host-a");
    let peers = [one.peer.clone(), refused("host-gone"), two.peer.clone()];
    let answer = runtime.block_on(caller.broadcast_read(&peers, &call(Method::GET, "/x", "")));
    assert!(answer.is_none());
    assert_eq!(one.seen.lock().unwrap().len(), 1);
    assert_eq!(two.seen.lock().unwrap().len(), 1);
    assert!(
        runtime
            .block_on(caller.broadcast_read(&[], &call(Method::GET, "/x", "")))
            .is_none()
    );
}

#[test]
fn a_broadcast_write_gives_each_answer_and_leaves_out_a_peer_that_does_not_connect() {
    let runtime = runtime();
    let done = stub(&runtime, "host-b", 204, "");
    let failed = stub(&runtime, "host-c", 503, "busy");
    let caller = Caller::new("host-a");
    let peers = [failed.peer.clone(), refused("host-gone"), done.peer.clone()];
    let answers = runtime.block_on(caller.broadcast_write(&peers, &call(Method::DELETE, "/council/v2/specs/x", "")));
    let seen: Vec<(String, u16, Bytes)> = answers
        .into_iter()
        .map(|(peer, response)| (peer.node, response.status().as_u16(), response.into_body()))
        .collect();
    assert_eq!(
        seen,
        [
            ("host-c".to_owned(), 503, Bytes::from_static(b"busy")),
            ("host-b".to_owned(), 204, Bytes::new()),
        ]
    );
    assert_eq!(failed.seen.lock().unwrap().len(), 1, "a 5xx is not tried again");
    assert_eq!(done.seen.lock().unwrap()[0].method, "DELETE");
}
