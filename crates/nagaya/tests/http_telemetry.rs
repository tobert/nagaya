//! The HTTP server span and the request duration (spec 9.2, 9.4, 9.5). The test has its own binary, because a
//! tracing call site caches its interest across the threads of one process.

mod council_support;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::Fixture;
use nagaya::http::{Council, Served, router};
use nagaya_telemetry::testing::Capture;
use serde_json::Value;
use tower::ServiceExt;

fn council(f: &Fixture) -> Arc<Council> {
    let served = Served {
        service: f.attach_again(),
        aliases: Vec::new(),
        device: None,
    };
    Arc::new(Council::new(vec![served], None).unwrap())
}

/// Send one request through the router on this thread, so that the capture of this thread sees it.
fn send_with(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: Option<Value>,
    headers: &[(&str, &str)],
) -> (StatusCode, String, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let id = response.headers()["request-id"].to_str().unwrap().to_owned();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, id, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

const TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
const PARENT_ID: &str = "00f067aa0ba902b7";

#[test]
fn a_request_has_a_server_span_under_the_callers_span_and_a_duration_point() {
    let f = Fixture::new();
    let council = council(&f);
    let capture = Capture::start(Vec::new());
    let traceparent = format!("00-{TRACE_ID}-{PARENT_ID}-01");
    let (status, _, _) = send_with(
        &council,
        "GET",
        "/council/v2/identity",
        None,
        &[("traceparent", &traceparent)],
    );
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = send_with(
        &council,
        "GET",
        "/council/v2/pins/nobody",
        None,
        &[("traceparent", &traceparent)],
    );
    assert_eq!(status, StatusCode::NOT_FOUND);
    send_with(&council, "GET", "/nowhere", None, &[("traceparent", &traceparent)]);
    let spans = capture.spans();
    let names: Vec<String> = spans.iter().map(|span| span.name.to_string()).collect();
    for name in ["GET /council/v2/identity", "GET /council/v2/pins/{owner}", "GET"] {
        let span = spans
            .iter()
            .find(|span| span.name == name)
            .unwrap_or_else(|| panic!("no span {name}: {names:?}"));
        assert_eq!(span.span_context.trace_id().to_string(), TRACE_ID);
        assert_eq!(span.parent_span_id.to_string(), PARENT_ID);
    }
    let points = capture.metric("http.server.request.duration");
    let mut seen: Vec<(String, String, u64)> = points
        .iter()
        .map(|point| {
            (
                point.attributes["http.route"].clone(),
                point.attributes["status"].clone(),
                point.count,
            )
        })
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        [
            ("/council/v2/identity".to_owned(), "200".to_owned(), 1),
            ("/council/v2/pins/{owner}".to_owned(), "404".to_owned(), 1),
            ("unmatched".to_owned(), "404".to_owned(), 1),
        ]
    );
    assert!(points.iter().all(|point| point.unit == "s"));
}
