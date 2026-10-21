//! W3C trace context on a request (spec 9.2).

use nagaya_telemetry::testing::Capture;
use nagaya_telemetry::{INVALID_ATTRIBUTE, ParentStatus, adopt_parent};
use opentelemetry_sdk::trace::SpanData;

const TRACE_ID: &str = "0af7651916cd43dd8448eb211c80319c";
const PARENT_ID: &str = "b7ad6b7169203331";

fn exported(capture: &Capture, name: &str) -> SpanData {
    capture
        .spans()
        .into_iter()
        .find(|span| span.name == name)
        .unwrap_or_else(|| panic!("no span named {name}"))
}

#[test]
fn a_valid_traceparent_makes_the_span_a_child_of_the_caller() {
    let capture = Capture::start(Vec::new());
    let span = tracing::info_span!("nagaya.http");
    let header = format!("00-{TRACE_ID}-{PARENT_ID}-01");
    assert_eq!(adopt_parent(&span, Some(&header), None), ParentStatus::Adopted);
    drop(span);

    let span = exported(&capture, "nagaya.http");
    assert_eq!(span.span_context.trace_id().to_string(), TRACE_ID);
    assert_eq!(span.parent_span_id.to_string(), PARENT_ID);
}

#[test]
fn tracestate_goes_to_the_span_context() {
    let capture = Capture::start(Vec::new());
    let span = tracing::info_span!("nagaya.http");
    let header = format!("00-{TRACE_ID}-{PARENT_ID}-01");
    assert_eq!(
        adopt_parent(&span, Some(&header), Some("vendor=one")),
        ParentStatus::Adopted
    );
    drop(span);

    let span = exported(&capture, "nagaya.http");
    assert_eq!(span.span_context.trace_state().header(), "vendor=one");
}

#[test]
fn an_invalid_traceparent_is_an_attribute_and_not_an_error() {
    let capture = Capture::start(Vec::new());
    let span = tracing::info_span!("nagaya.http");
    assert_eq!(adopt_parent(&span, Some("not-a-header"), None), ParentStatus::Invalid);
    drop(span);

    let span = exported(&capture, "nagaya.http");
    assert!(
        span.parent_span_id == opentelemetry::trace::SpanId::INVALID,
        "an invalid header gives no parent"
    );
    let attribute = span
        .attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == INVALID_ATTRIBUTE)
        .expect("the header is an attribute");
    assert_eq!(attribute.value.as_str(), "not-a-header");
    assert!(
        !span
            .attributes
            .iter()
            .any(|attribute| attribute.key.as_str() == "otel.status_code"),
        "an invalid header is not an error"
    );
}

#[test]
fn no_traceparent_changes_nothing() {
    let capture = Capture::start(Vec::new());
    let span = tracing::info_span!("nagaya.http");
    assert_eq!(adopt_parent(&span, None, Some("vendor=one")), ParentStatus::Absent);
    drop(span);

    let span = exported(&capture, "nagaya.http");
    assert!(span.parent_span_id == opentelemetry::trace::SpanId::INVALID);
    assert!(
        span.attributes
            .iter()
            .all(|attribute| attribute.key.as_str() != INVALID_ATTRIBUTE)
    );
}

#[test]
fn a_child_span_keeps_the_callers_trace() {
    let capture = Capture::start(Vec::new());
    let request = tracing::info_span!("nagaya.http");
    let header = format!("00-{TRACE_ID}-{PARENT_ID}-01");
    adopt_parent(&request, Some(&header), None);
    request.in_scope(|| {
        tracing::info_span!("nagaya.job").in_scope(|| {});
    });
    drop(request);

    let job = exported(&capture, "nagaya.job");
    let http = exported(&capture, "nagaya.http");
    assert_eq!(job.span_context.trace_id().to_string(), TRACE_ID);
    assert_eq!(job.parent_span_id, http.span_context.span_id());
}
