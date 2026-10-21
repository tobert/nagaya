//! The sampling rule of spec 9.3.

use nagaya_telemetry::{NagayaSampler, sampling_rate};
use opentelemetry::trace::{
    SamplingDecision, SpanContext, SpanId, SpanKind, TraceContextExt, TraceFlags, TraceId, TraceState, Tracer,
    TracerProvider,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, ShouldSample};

/// The ratio sampler reads the low 64 bits of the trace id. All ones is above every rate under 1.
const DROPPED: u128 = (1 << 64) | u64::MAX as u128;

fn decide(parent: Option<&Context>, trace_id: u128, name: &str, attributes: &[KeyValue]) -> SamplingDecision {
    NagayaSampler
        .should_sample(
            parent,
            TraceId::from(trace_id),
            name,
            &SpanKind::Internal,
            attributes,
            &[],
        )
        .decision
}

fn remote_parent(flags: TraceFlags) -> Context {
    let span = SpanContext::new(TraceId::from(7), SpanId::from(9), flags, true, TraceState::default());
    Context::new().with_remote_span_context(span)
}

#[test]
fn nagaya_and_council_prefixes_keep_every_span() {
    for name in ["nagaya.job", "nagaya.quantum", "nagaya.start", "council.decide"] {
        assert_eq!(sampling_rate(name), 1.0, "{name}");
    }
}

#[test]
fn other_names_keep_one_span_in_ten() {
    for name in ["GET /v1/contexts", "rpc", "job", "unknown"] {
        assert_eq!(sampling_rate(name), 0.1, "{name}");
    }
}

#[test]
fn a_prefix_needs_its_dot() {
    // kaijutsu once kept `drift_queue` at the full rate of `drift.`. The nagaya keeps the dot.
    assert_eq!(sampling_rate("nagaya_job"), 0.1);
    assert_eq!(sampling_rate("nagayajob"), 0.1);
    assert_eq!(sampling_rate("council_queue"), 0.1);
}

#[test]
fn a_span_with_a_full_rate_name_is_sampled_for_any_trace_id() {
    for trace_id in [
        1u128,
        u128::MAX,
        0xdead_beef << 64,
        0x1234_5678_9abc_def0_1234_5678_9abc_def0,
    ] {
        assert_eq!(
            decide(None, trace_id, "nagaya.quantum", &[]),
            SamplingDecision::RecordAndSample
        );
    }
}

#[test]
fn a_default_name_is_sampled_for_about_one_trace_in_ten() {
    let kept = (0u64..10_000)
        .filter(|index| {
            // The golden-ratio step spreads the low 64 bits over the whole range that the ratio sampler reads.
            let low = index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            decide(None, (1u128 << 64) | u128::from(low), "GET /v1/contexts", &[]) == SamplingDecision::RecordAndSample
        })
        .count();
    assert!((700..=1300).contains(&kept), "kept {kept} of 10000");
}

#[test]
fn a_sampled_parent_keeps_the_child_whatever_its_name() {
    let parent = remote_parent(TraceFlags::SAMPLED);
    let dropped = DROPPED;
    assert_eq!(decide(None, dropped, "child", &[]), SamplingDecision::Drop);
    assert_eq!(
        decide(Some(&parent), dropped, "child", &[]),
        SamplingDecision::RecordAndSample
    );
}

#[test]
fn an_unsampled_parent_does_not_keep_the_child() {
    let parent = remote_parent(TraceFlags::default());
    let dropped = DROPPED;
    assert_eq!(decide(None, dropped, "child", &[]), SamplingDecision::Drop);
    assert_eq!(decide(Some(&parent), dropped, "child", &[]), SamplingDecision::Drop);
}

#[test]
fn an_error_is_always_kept() {
    let dropped = DROPPED;
    assert_eq!(decide(None, dropped, "child", &[]), SamplingDecision::Drop);
    let status = [KeyValue::new("otel.status_code", "ERROR")];
    assert_eq!(
        decide(None, dropped, "child", &status),
        SamplingDecision::RecordAndSample
    );
    let flag = [KeyValue::new("error", "true")];
    assert_eq!(decide(None, dropped, "child", &flag), SamplingDecision::RecordAndSample);
    let fine = [KeyValue::new("otel.status_code", "OK")];
    assert_eq!(decide(None, dropped, "child", &fine), SamplingDecision::Drop);
}

#[test]
fn the_provider_exports_a_full_rate_span_and_drops_a_default_span() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_sampler(NagayaSampler)
        .with_simple_exporter(exporter.clone())
        .build();
    let tracer = provider.tracer("test");
    let dropped = DROPPED;
    for name in ["nagaya.job", "default"] {
        let span = tracer
            .span_builder(name)
            .with_trace_id(TraceId::from(dropped))
            .start(&tracer);
        drop(span);
    }
    let names: Vec<String> = exporter
        .get_finished_spans()
        .unwrap()
        .into_iter()
        .map(|span| span.name.to_string())
        .collect();
    assert_eq!(names, ["nagaya.job"]);
}
