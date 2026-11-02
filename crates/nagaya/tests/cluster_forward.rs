//! The forward of spec 11.1a, in one process: a council with a cluster set by hand, and stub peers on loopback.

mod forward_support;

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use forward_support::{
    Answer, Fixture, OWN, PEER, Stub, council, entry_like_local, peer_node, refused_addr, send, set_peers,
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

const PEER_ID: &str = "host-b:1:127.0.0.1:9";

fn generation(model: &str) -> String {
    json!({"model": model, "context": {"messages": [{"role": "user", "content": "hi"}]}, "max_tokens": 4}).to_string()
}

#[test]
fn a_job_route_for_a_peer_model_sends_the_request_as_it_came_and_gives_the_answer_as_it_came() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(
        200,
        "application/json",
        "{\"from\": \"the peer\"}".to_owned(),
    ));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    // The body bytes go as they came, also their spaces and the order of their members.
    let body = format!("  {}  ", generation("toy-b"));
    let reply = send(
        &council,
        "POST",
        "/council/v2/generations?trace=1",
        &body,
        &[
            ("accept", "application/json"),
            ("tracestate", "k=v"),
            ("x-other", "dropped"),
        ],
    );
    assert_eq!(reply.status, 200);
    assert_eq!(reply.body.as_ref(), b"{\"from\": \"the peer\"}");
    assert_eq!(reply.headers["x-peer"], "stub", "the headers of the peer come back");
    let seen = stub.seen();
    assert_eq!(seen.len(), 1);
    let seen = &seen[0];
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.uri, "/council/v2/generations?trace=1");
    assert_eq!(seen.body.as_ref(), body.as_bytes());
    assert_eq!(seen.headers["content-type"], "application/json");
    assert_eq!(seen.headers["accept"], "application/json");
    assert_eq!(seen.headers["tracestate"], "k=v");
    assert!(
        seen.headers.get("x-other").is_none(),
        "only the headers of spec 11.1a go"
    );
    assert_eq!(seen.headers["nagaya-forwarded"], OWN);
    // One request id: the coordinator's id goes to the peer, and the client sees it.
    assert_eq!(seen.headers["nagaya-request-id"], reply.headers["request-id"]);
}

#[test]
fn each_job_route_forwards() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let routes = ["contexts", "decisions", "reads", "generations", "embeddings"];
    for route in routes {
        let reply = send(
            &council,
            "POST",
            &format!("/council/v2/{route}"),
            "{\"model\": \"toy-b\"}",
            &[],
        );
        assert_eq!(reply.status, 200, "{route}");
    }
    let uris: Vec<String> = stub.seen().into_iter().map(|seen| seen.uri).collect();
    let want: Vec<String> = routes.iter().map(|route| format!("/council/v2/{route}")).collect();
    assert_eq!(uris, want);
}

#[test]
fn a_4xx_of_the_peer_is_the_answer() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(413, "application/json", "{\"error\": {}}".to_owned()));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy-b"), &[]);
    assert_eq!(
        (reply.status, reply.body.as_ref()),
        (413, b"{\"error\": {}}".as_slice())
    );
    assert_eq!(stub.seen().len(), 1, "no retry");
}

#[test]
fn a_model_of_this_node_never_forwards() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy"), &[]);
    assert_eq!(reply.status, 200, "{:?}", reply.body);
    assert!(stub.seen().is_empty());
}

#[test]
fn a_peer_model_that_is_not_ready_in_gossip_gets_its_503_at_once() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    for (state, status) in [("loading", "loading"), ("draining", "draining"), ("fatal", "unloaded")] {
        set_peers(
            &council,
            &cluster,
            vec![peer_node(
                PEER,
                PEER_ID,
                stub.http,
                state,
                &[entry_like_local(&council, "toy-b")],
            )],
        );
        let reply = send(&council, "POST", "/council/v2/generations", &generation("toy-b"), &[]);
        assert_eq!(reply.status, 503);
        let body = reply.json();
        assert_eq!(body["error"]["type"], "model_unavailable");
        assert_eq!(body["error"]["status"], status);
        assert_eq!(body["error"]["model"], "toy-b");
    }
    assert!(stub.seen().is_empty());
}

#[test]
fn a_peer_that_does_not_connect_gives_a_503_unloaded_that_names_the_node() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            refused_addr(),
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy-b"), &[]);
    assert_eq!(reply.status, 503);
    let body = reply.json();
    assert_eq!(body["error"]["type"], "model_unavailable");
    assert_eq!(body["error"]["status"], "unloaded");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("node host-b"),
        "{body}"
    );
    assert_eq!(body["request_id"], reply.headers["request-id"].to_str().unwrap());
}

#[test]
fn a_forwarded_request_is_never_forwarded_again() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let forwarded = [("nagaya-forwarded", "host-c"), ("nagaya-request-id", "req_from_c")];
    for model in ["toy-b", "nowhere"] {
        let reply = send(
            &council,
            "POST",
            "/council/v2/generations",
            &generation(model),
            &forwarded,
        );
        assert_eq!(reply.status, 503, "{model}");
        let body = reply.json();
        assert_eq!(body["error"]["type"], "model_unavailable");
        assert_eq!(body["error"]["status"], "unloaded");
        assert_eq!(body["error"]["model"], model);
        // The target takes the coordinator's request id.
        assert_eq!(reply.headers["request-id"], "req_from_c");
        assert_eq!(body["request_id"], "req_from_c");
    }
    assert!(stub.seen().is_empty());
    // A request that no node forwarded and that names no served model is the 400 of the route.
    let reply = send(&council, "POST", "/council/v2/generations", &generation("nowhere"), &[]);
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["param"], "model");
    // A forwarded request for a model of this node runs here, with the coordinator's id.
    let reply = send(
        &council,
        "POST",
        "/council/v2/generations",
        &generation("toy"),
        &forwarded,
    );
    assert_eq!(reply.status, 200);
    assert_eq!(reply.headers["request-id"], "req_from_c");
}

#[test]
fn a_request_id_without_the_forward_header_is_not_taken() {
    let fixture = Fixture::new();
    let (council, _cluster) = council(&fixture);
    let reply = send(
        &council,
        "GET",
        "/council/v2/identity",
        "",
        &[("nagaya-request-id", "req_mine")],
    );
    assert_ne!(reply.headers["request-id"], "req_mine");
}

#[test]
fn a_peer_with_a_duplicate_node_name_gets_no_forward() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    let entry = entry_like_local(&council, "toy-b");
    set_peers(
        &council,
        &cluster,
        vec![
            peer_node(PEER, PEER_ID, stub.http, "ready", std::slice::from_ref(&entry)),
            peer_node(PEER, "host-b:2:127.0.0.1:10", stub.http, "ready", &[]),
        ],
    );
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy-b"), &[]);
    assert_eq!(reply.status, 503);
    let body = reply.json();
    assert_eq!(body["error"]["status"], "unloaded");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("another live node has that name")
    );
    assert!(stub.seen().is_empty());
}

#[test]
fn a_stopping_coordinator_answers_unavailable_for_a_peer_model() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    council.drain();
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy-b"), &[]);
    assert_eq!(reply.status, 503);
    assert_eq!(reply.json()["error"]["type"], "unavailable");
    assert!(stub.seen().is_empty());
}

#[test]
fn a_text_route_runs_here_when_a_local_family_has_the_model() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{}".to_owned()));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let local = send(
        &council,
        "POST",
        "/nagaya/v1/tokenize",
        "{\"model\": \"toy\", \"text\": \"hello\"}",
        &[],
    );
    let peer = send(
        &council,
        "POST",
        "/nagaya/v1/tokenize",
        "{\"model\": \"toy-b\", \"text\": \"hello\"}",
        &[],
    );
    assert_eq!(peer.status, 200, "{:?}", peer.body);
    let (local, peer) = (local.json(), peer.json());
    assert_eq!(peer["model"], "toy-b", "the response names the model of the request");
    assert_eq!(peer["tokens"], local["tokens"]);
    let detokenized = send(
        &council,
        "POST",
        "/nagaya/v1/detokenize",
        &json!({"model": "toy-b", "tokens": local["tokens"]}).to_string(),
        &[],
    );
    assert_eq!(detokenized.json(), json!({"model": "toy-b", "text": "hello"}));
    let rendered = send(
        &council,
        "POST",
        "/nagaya/v1/render",
        &json!({"model": "toy-b", "context": {"messages": [{"role": "user", "content": "hi"}]}}).to_string(),
        &[],
    );
    assert_eq!(rendered.status, 200);
    assert_eq!(rendered.json()["model"], "toy-b");
    assert!(stub.seen().is_empty(), "no text route forwarded");
    // A peer model that is not ready is a 503 here too (spec 11.5).
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "loading",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let reply = send(
        &council,
        "POST",
        "/nagaya/v1/tokenize",
        "{\"model\": \"toy-b\", \"text\": \"x\"}",
        &[],
    );
    assert_eq!(
        (reply.status, reply.json()["error"]["status"].clone()),
        (503, json!("loading"))
    );
}

#[test]
fn a_text_route_forwards_when_no_local_family_has_the_model() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(
        200,
        "application/json",
        "{\"model\": \"other\"}".to_owned(),
    ));
    let mut entry = entry_like_local(&council, "other");
    entry["template"] = Value::from("another-template");
    set_peers(
        &council,
        &cluster,
        vec![peer_node(PEER, PEER_ID, stub.http, "ready", &[entry])],
    );
    for route in ["tokenize", "detokenize", "render"] {
        let reply = send(
            &council,
            "POST",
            &format!("/nagaya/v1/{route}"),
            "{\"model\": \"other\"}",
            &[],
        );
        assert_eq!(reply.body.as_ref(), b"{\"model\": \"other\"}", "{route}");
    }
    assert_eq!(stub.seen().len(), 3);
}

#[test]
fn a_local_model_whose_name_a_peer_keeps_forwards_to_the_keeper() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let stub = Stub::start(Answer::Whole(200, "application/json", "{\"kept\": true}".to_owned()));
    // The node `host-0` has a lower name than `host-a`, so it keeps the name `toy`.
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            "host-0",
            "host-0:1:127.0.0.1:9",
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy")],
        )],
    );
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy"), &[]);
    assert_eq!(reply.body.as_ref(), b"{\"kept\": true}");
    assert_eq!(stub.seen().len(), 1);
    let identity = send(&council, "GET", "/council/v2/identity", "", &[]).json();
    assert_eq!(identity["models"].as_array().unwrap().len(), 1, "{identity}");
}

/// Read the first frame of a forwarded stream, then take the peer out of the live set, then read the rest.
#[test]
fn a_stream_whose_peer_leaves_ends_with_an_error_event() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    let first = "event: token\ndata: {\"n\":0}\n\n".to_owned();
    let stub = Stub::start(Answer::Hang(vec![first.clone()]));
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            stub.http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (head, rest) = runtime.block_on(async {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/council/v2/generations")
            .header("content-type", "application/json")
            .body(Body::from(generation("toy-b")))
            .unwrap();
        let response = nagaya::http::router(Arc::clone(&council))
            .oneshot(request)
            .await
            .unwrap();
        let id = response.headers()["request-id"].to_str().unwrap().to_owned();
        let mut body = response.into_body();
        let head = body.frame().await.unwrap().unwrap().into_data().unwrap();
        set_peers(&council, &cluster, Vec::new());
        let rest = tokio::time::timeout(Duration::from_secs(10), body.collect())
            .await
            .expect("the stream ends when the peer leaves")
            .unwrap()
            .to_bytes();
        (head, (id, rest))
    });
    assert_eq!(head.as_ref(), first.as_bytes());
    let (id, rest) = rest;
    let text = String::from_utf8(rest.to_vec()).unwrap();
    let data = text
        .strip_prefix("event: error\ndata: ")
        .and_then(|rest| rest.strip_suffix("\n\n"))
        .unwrap_or_else(|| panic!("not one error event: {text:?}"));
    let body: Value = serde_json::from_str(data).unwrap();
    assert_eq!(body["error"]["type"], "model_unavailable");
    assert_eq!(body["error"]["status"], "unloaded");
    assert_eq!(body["request_id"], id);
}

#[test]
fn a_forward_whose_peer_leaves_before_the_status_gets_the_503() {
    let fixture = Fixture::new();
    let (council, cluster) = council(&fixture);
    // A listener that accepts and never answers.
    let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let http = silent.local_addr().unwrap();
    set_peers(
        &council,
        &cluster,
        vec![peer_node(
            PEER,
            PEER_ID,
            http,
            "ready",
            &[entry_like_local(&council, "toy-b")],
        )],
    );
    let leaver = {
        let council = Arc::clone(&council);
        let cluster = Arc::clone(&cluster);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            set_peers(&council, &cluster, Vec::new());
        })
    };
    let reply = send(&council, "POST", "/council/v2/generations", &generation("toy-b"), &[]);
    leaver.join().unwrap();
    assert_eq!(reply.status, 503);
    assert_eq!(reply.json()["error"]["status"], "unloaded");
    drop(silent);
}
