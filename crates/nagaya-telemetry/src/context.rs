//! W3C trace context from a request (spec 9.2).

use std::collections::HashMap;

use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::TraceContextExt;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// The attribute that holds a `traceparent` header that the nagaya could not use (spec 9.2).
pub const INVALID_ATTRIBUTE: &str = "nagaya.traceparent.invalid";

/// What a request did to its span's parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentStatus {
    /// The request had no `traceparent`. The span keeps its own parent.
    Absent,
    /// The span is now a child of the caller's span.
    Adopted,
    /// The header was not valid. The span keeps its own parent and records the header as an attribute.
    Invalid,
}

/// Make `span` a child of the caller's span, from the `traceparent` and `tracestate` headers of a request.
///
/// An invalid header is not an error: the span records the header text in the attribute
/// [`INVALID_ATTRIBUTE`] and the request goes on (spec 9.2).
pub fn adopt_parent(span: &tracing::Span, traceparent: Option<&str>, tracestate: Option<&str>) -> ParentStatus {
    let Some(traceparent) = traceparent else {
        return ParentStatus::Absent;
    };
    let mut carrier = HashMap::new();
    carrier.insert("traceparent".to_owned(), traceparent.to_owned());
    if let Some(tracestate) = tracestate {
        carrier.insert("tracestate".to_owned(), tracestate.to_owned());
    }
    let context = TraceContextPropagator::new().extract(&carrier);
    if !context.span().span_context().is_valid() {
        span.set_attribute(INVALID_ATTRIBUTE, traceparent.to_owned());
        return ParentStatus::Invalid;
    }
    // The setter fails only when the span is disabled, and then there is nothing to link.
    let _ = span.set_parent(context);
    ParentStatus::Adopted
}

/// The `traceparent` header that names `span` as the parent, for a call to another node (spec 9.2, "A forwarded
/// request"). A span with no valid trace context gives `None`, for example when no trace layer records it.
pub fn traceparent_of(span: &tracing::Span) -> Option<String> {
    let context = span.context();
    if !context.span().span_context().is_valid() {
        return None;
    }
    let mut carrier: HashMap<String, String> = HashMap::new();
    TraceContextPropagator::new().inject_context(&context, &mut carrier);
    carrier.remove("traceparent")
}
