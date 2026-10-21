//! The sampling rule (spec 9.3). It is kaijutsu's rule with the nagaya's prefixes.

use opentelemetry::trace::{Link, SamplingDecision, SamplingResult, SpanKind, TraceContextExt, TraceId, TraceState};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::trace::{Sampler, ShouldSample};

/// The rate for spans whose name has none of the full-rate prefixes.
pub const DEFAULT_RATE: f64 = 0.1;

/// The prefixes with a full rate. Each ends with a dot, so that `nagaya_x` does not match `nagaya.`.
pub const FULL_RATE_PREFIXES: [&str; 2] = ["nagaya.", "council."];

/// The rate for a span, chosen by its name (spec 9.3).
///
/// The prefixes carry the dot on purpose. kaijutsu once swept a span named `drift_queue` into the `drift.` rate.
pub fn sampling_rate(name: &str) -> f64 {
    if FULL_RATE_PREFIXES.iter().any(|prefix| name.starts_with(prefix)) {
        1.0
    } else {
        DEFAULT_RATE
    }
}

/// The sampler of the nagaya (spec 9.3).
///
/// It keeps a span whose parent is sampled, keeps an error, and keeps the rest at the rate of [`sampling_rate`].
#[derive(Clone, Debug, Default)]
pub struct NagayaSampler;

impl ShouldSample for NagayaSampler {
    fn should_sample(
        &self,
        parent_context: Option<&Context>,
        trace_id: TraceId,
        name: &str,
        span_kind: &SpanKind,
        attributes: &[KeyValue],
        links: &[Link],
    ) -> SamplingResult {
        if let Some(context) = parent_context {
            let parent = context.span();
            let parent = parent.span_context();
            if parent.is_sampled() {
                return SamplingResult {
                    decision: SamplingDecision::RecordAndSample,
                    attributes: Vec::new(),
                    trace_state: parent.trace_state().clone(),
                };
            }
        }
        let is_error = attributes.iter().any(|attribute| {
            (attribute.key.as_str() == "otel.status_code" && attribute.value.as_str() == "ERROR")
                || (attribute.key.as_str() == "error" && attribute.value.as_str() == "true")
        });
        if is_error {
            return SamplingResult {
                decision: SamplingDecision::RecordAndSample,
                attributes: Vec::new(),
                trace_state: TraceState::default(),
            };
        }
        Sampler::TraceIdRatioBased(sampling_rate(name)).should_sample(
            parent_context,
            trace_id,
            name,
            span_kind,
            attributes,
            links,
        )
    }
}
