//! The span `nagaya.forward` of spec 9.2, "A forwarded request". The test has its own binary, because a tracing call
//! site caches its interest across the threads of one process.

mod forward_support;

use forward_support::{
    Answer, Fixture, PEER, Stub, council, entry_like_local, peer_node, refused_addr, send, set_peers,
};
use nagaya_telemetry::testing::{Capture, SpanData};

const TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
const PARENT_ID: &str = "00f067aa0ba902b7";
const PEER_ID: &str = "host-b:1:127.0.0.1:9";

fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|item| item.key.as_str() == key)
        .map(|item| item.value.to_string())
}

fn named(capture: &Capture, name: &str) -> Vec<SpanData> {
    capture.spans().into_iter().filter(|span| span.name == name).collect()
}

fn is_error(span: &SpanData) -> bool {
    matches!(span.status, opentelemetry::trace::Status::Error { .. })
}

#[test]
fn a_forward_has_its_span_under_the_server_span_and_names_it_to_the_peer() {
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
    let capture = Capture::start(Vec::new());
    let traceparent = format!("00-{TRACE_ID}-{PARENT_ID}-01");
    let headers = [("traceparent", traceparent.as_str())];

    // A forward that is done.
    let reply = send(
        &council,
        "POST",
        "/council/v2/generations",
        "{\"model\": \"toy-b\"}",
        &headers,
    );
    assert_eq!(reply.status, 200);
    let forwards = named(&capture, "nagaya.forward");
    assert_eq!(forwards.len(), 1, "one span for one forward");
    let forward = &forwards[0];
    let server = named(&capture, "POST /council/v2/generations")
        .pop()
        .expect("a server span");
    assert_eq!(forward.parent_span_id, server.span_context.span_id());
    assert_eq!(forward.span_context.trace_id().to_string(), TRACE_ID);
    assert_eq!(attribute(forward, "nagaya.peer").as_deref(), Some(PEER));
    assert_eq!(attribute(forward, "outcome").as_deref(), Some("done"));
    assert_eq!(attribute(forward, "http.response.status_code").as_deref(), Some("200"));
    assert!(!is_error(forward));
    // The peer's server span becomes the child of the forward's span.
    let sent = stub.seen()[0].headers["traceparent"].to_str().unwrap().to_owned();
    let want = format!("00-{TRACE_ID}-{}-01", forward.span_context.span_id());
    assert_eq!(sent, want);

    // A text route that runs here makes no forward span.
    let reply = send(
        &council,
        "POST",
        "/nagaya/v1/tokenize",
        "{\"model\": \"toy-b\", \"text\": \"a\"}",
        &[],
    );
    assert_eq!(reply.status, 200);
    assert_eq!(named(&capture, "nagaya.forward").len(), 1);

    // A forward that is refused marks its span as an error.
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
    let reply = send(
        &council,
        "POST",
        "/council/v2/generations",
        "{\"model\": \"toy-b\"}",
        &headers,
    );
    assert_eq!(reply.status, 503);
    let forwards = named(&capture, "nagaya.forward");
    let refused = forwards
        .iter()
        .find(|span| attribute(span, "outcome").as_deref() == Some("refused"));
    let refused = refused.unwrap_or_else(|| panic!("no refused forward in {forwards:?}"));
    assert!(is_error(refused));
    assert_eq!(attribute(refused, "http.response.status_code").as_deref(), Some("503"));
}
