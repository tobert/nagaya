//! The server span and the request duration of a text route (spec 11.4b, 9.2, 9.4). The test has its own binary, because a
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

#[test]
fn a_text_route_has_a_server_span_and_a_duration_point_and_makes_no_job_span() {
    let f = Fixture::new();
    let council = council(&f);
    let capture = Capture::start(Vec::new());
    let body = serde_json::json!({"model": "toy", "text": "hello"});
    let (status, _, _) = send_with(
        &council,
        "POST",
        "/nagaya/v1/tokenize",
        Some(body),
        &[("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")],
    );
    assert_eq!(status, StatusCode::OK);
    let spans = capture.spans();
    let names: Vec<String> = spans.iter().map(|span| span.name.to_string()).collect();
    assert!(names.contains(&"POST /nagaya/v1/tokenize".to_owned()), "{names:?}");
    assert!(
        names.iter().all(|name| name != "nagaya.job"),
        "a text route runs no job: {names:?}"
    );
    let points = capture.metric("http.server.request.duration");
    let routes: Vec<(String, String)> = points
        .iter()
        .map(|point| {
            (
                point.attributes["http.route"].clone(),
                point.attributes["status"].clone(),
            )
        })
        .collect();
    assert_eq!(routes, [("/nagaya/v1/tokenize".to_owned(), "200".to_owned())]);
}
