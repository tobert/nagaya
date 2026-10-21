//! The group thread records each quantum that it runs (spec 9.2, 9.4, 9.5). The tests run a council build and a
//! generation on the fake engine through the process, and read the spans and the points from memory.

mod council_support;

use std::time::{Duration, Instant};

use council_support::{ENGINE, Fixture, Script, TermFactory};
use gogi_proto::context::Context;
use gogi_proto::generation::GenerationRequest;
use nagaya::council::ServiceConfig;
use nagaya_telemetry::testing::{Capture, Point, SpanData};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

fn ctx(messages: Vec<Value>) -> Value {
    json!({"thinking": true, "tools": [{"name": "sh", "input_schema": {"type": "object"}}], "messages": messages})
}

fn opening() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a coding agent. Use the shell tool."}),
        json!({"role": "user", "content": "Delete the build directory, then rebuild.", "snap": true}),
    ]
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|item| item.key.as_str() == key)
        .map(|item| item.value.to_string())
}

fn spans_named(capture: &Capture, name: &str) -> Vec<SpanData> {
    capture.spans().into_iter().filter(|span| span.name == name).collect()
}

fn count(points: &[Point]) -> u64 {
    points.iter().map(|point| point.count).sum()
}

/// A fixture whose engine favors a short reply, a build of the context, and a generation after it.
///
/// The group thread runs after the answer of the caller, so the last quantum of a job ends a little later. This
/// function waits until each quantum that started has ended, and each job has ended.
fn run(capture: &Capture) {
    let script = Script::default();
    let factory = TermFactory {
        script: script.clone(),
        ..TermFactory::toy()
    };
    let config = ServiceConfig::new("toy", ENGINE);
    let f = Fixture::with_factory(factory, 1 << 20, None, config);
    let value = ctx(opening());
    let prompt = nagaya_council::family::Family::render(&ToyFamily::new(), &context(&value), true)
        .unwrap()
        .tokens
        .len();
    let mut reply: Vec<u32> = "ok".bytes().map(u32::from).collect();
    reply.push(gogi_toy::END);
    script.set(prompt, reply);
    f.service.build(&context(&value), false, Default::default()).unwrap();
    let request: GenerationRequest = serde_json::from_value(
        json!({"model": "toy", "context": value, "max_tokens": 20, "sample": {"temperature": 1.0, "seed": 7}}),
    )
    .unwrap();
    f.service.generate(&request, None).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let started = count(&capture.metric("nagaya.quantum.wait.duration"));
        let ended = count(&capture.metric("nagaya.quantum.run.duration"));
        let jobs = spans_named(capture, "nagaya.job").len();
        if started == ended && started > 0 && jobs >= 2 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the quanta never ended: {started} started, {ended} ended, {jobs} jobs"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_council_build_records_a_span_for_each_quantum_that_is_not_decode() {
    let capture = Capture::start(Vec::new());
    run(&capture);

    let quanta = spans_named(&capture, "nagaya.quantum");
    assert!(!quanta.is_empty(), "the group thread made no nagaya.quantum span");
    let kinds: Vec<String> = quanta.iter().map(|span| attribute(span, "kind").unwrap()).collect();
    assert!(
        kinds.iter().any(|kind| kind == "copy"),
        "the build starts with a copy: {kinds:?}"
    );
    assert!(
        kinds.iter().any(|kind| kind == "prefill"),
        "the build feeds the context: {kinds:?}"
    );
    assert!(
        !kinds.iter().any(|kind| kind == "decode"),
        "a decode quantum makes no span (spec 9.2): {kinds:?}"
    );
    for span in &quanta {
        assert_eq!(attribute(span, "engine").as_deref(), Some(ENGINE));
        assert_eq!(attribute(span, "device_group").as_deref(), Some("0"));
        assert_eq!(attribute(span, "class").as_deref(), Some("interactive"));
        assert_eq!(attribute(span, "segments").as_deref(), Some("1"));
        assert_eq!(attribute(span, "outcome").as_deref(), Some("done"));
        assert!(
            attribute(span, "wait_ms").is_some() && attribute(span, "run_ms").is_some(),
            "{span:?}"
        );
    }
}

#[test]
fn the_two_durations_record_each_quantum_with_the_labels_of_the_span() {
    let capture = Capture::start(Vec::new());
    run(&capture);

    let spans = spans_named(&capture, "nagaya.quantum").len() as u64;
    let wait = capture.metric("nagaya.quantum.wait.duration");
    let run = capture.metric("nagaya.quantum.run.duration");
    for points in [&wait, &run] {
        assert!(!points.is_empty(), "no quantum duration points");
        for point in points {
            assert_eq!(point.unit, "ms");
            assert_eq!(point.attributes["engine"], ENGINE);
            assert_eq!(point.attributes["device_group"], "0");
            assert_eq!(point.attributes["class"], "interactive");
        }
    }
    let decode = |points: &[Point]| -> u64 {
        count(
            &points
                .iter()
                .filter(|p| p.attributes["kind"] == "decode")
                .cloned()
                .collect::<Vec<_>>(),
        )
    };
    assert!(decode(&run) > 0, "the generation runs decode quanta: {run:?}");
    // Every quantum records both durations. Every quantum that is not decode also has a span.
    assert_eq!(count(&wait), count(&run));
    assert_eq!(count(&run) - decode(&run), spans);
}

#[test]
fn a_job_span_names_its_engine_class_and_request_and_is_the_parent_of_its_quanta() {
    let capture = Capture::start(Vec::new());
    run(&capture);

    let jobs = spans_named(&capture, "nagaya.job");
    // The service asks the engine for its identity when it attaches. The generation builds its context first, as a job
    // of its own. Thus the requests repeat.
    let mut requests: Vec<String> = jobs.iter().map(|span| attribute(span, "request").unwrap()).collect();
    requests.sort();
    requests.dedup();
    assert_eq!(requests, ["attach", "build", "generation"]);
    for job in &jobs {
        assert_eq!(attribute(job, "engine").as_deref(), Some(ENGINE));
        assert_eq!(attribute(job, "class").as_deref(), Some("interactive"));
    }
    let quanta = spans_named(&capture, "nagaya.quantum");
    for quantum in &quanta {
        assert!(
            jobs.iter()
                .any(|job| job.span_context.span_id() == quantum.parent_span_id),
            "a quantum span has a job span as its parent: {quantum:?}"
        );
    }
    for job in &jobs {
        let owned = quanta
            .iter()
            .filter(|q| q.parent_span_id == job.span_context.span_id())
            .count();
        assert!(
            owned > 0,
            "the job {:?} owns no quantum span",
            attribute(job, "request")
        );
    }
}

#[test]
fn a_job_span_is_the_child_of_the_span_that_submitted_the_job() {
    let capture = Capture::start(Vec::new());
    let f = Fixture::new();
    let value = ctx(opening());
    // The sampler keeps the spans with the prefix `nagaya.` only (spec 9.3), so the test request has that prefix.
    let request = tracing::info_span!("nagaya.test.request");
    request.in_scope(|| f.service.build(&context(&value), false, Default::default()).unwrap());
    drop(request);

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let spans = capture.spans();
        let parent = spans.iter().find(|span| span.name == "nagaya.test.request");
        let job = spans
            .iter()
            .find(|span| span.name == "nagaya.job" && attribute(span, "request").as_deref() == Some("build"));
        if let (Some(parent), Some(job)) = (parent, job) {
            assert_eq!(job.parent_span_id, parent.span_context.span_id());
            assert_eq!(job.span_context.trace_id(), parent.span_context.trace_id());
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the build job span never ended: {} spans",
            spans.len()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

struct Spanning;

impl nagaya::jobs::JobBody for Spanning {
    fn run(&mut self, _kind: nagaya_abi::Kind, _engines: &mut [Box<dyn nagaya_abi::Engine>]) -> nagaya::jobs::Ran {
        tracing::info_span!("engine.pass").in_scope(|| {});
        nagaya::jobs::Ran::Done
    }
}

#[test]
fn the_body_of_a_quantum_runs_inside_the_span_of_the_quantum() {
    let capture = Capture::start(Vec::new());
    let mut jobs = nagaya::jobs::GroupJobs::new(0, "toy");
    jobs.add(nagaya::jobs::JobSpec {
        class: nagaya_sched::Class::Batch,
        priority: 0,
        deadline: None,
        first: nagaya_abi::Kind::Prefill,
        body: Box::new(Spanning),
    });
    while !jobs.idle() {
        jobs.run_one(&mut []);
    }
    let spans = capture.spans();
    let quantum = spans
        .iter()
        .find(|span| span.name == "nagaya.quantum")
        .expect("a quantum span");
    let pass = spans
        .iter()
        .find(|span| span.name == "engine.pass")
        .expect("a pass span");
    assert_eq!(pass.parent_span_id, quantum.span_context.span_id());
}

#[test]
fn a_stop_that_cancels_the_jobs_still_records_the_quantum_that_ran() {
    let capture = Capture::start(Vec::new());
    let mut jobs = nagaya::jobs::GroupJobs::new(0, "toy");
    jobs.add(nagaya::jobs::JobSpec {
        class: nagaya_sched::Class::Batch,
        priority: 0,
        deadline: None,
        first: nagaya_abi::Kind::Prefill,
        body: Box::new(Spanning),
    });
    assert!(jobs.run_one(&mut []));
    assert_eq!(jobs.cancel_all(), 1);

    let run = capture.metric("nagaya.quantum.run.duration");
    assert_eq!(run.iter().map(|point| point.count).sum::<u64>(), 1, "{run:?}");
    assert!(spans_named(&capture, "nagaya.quantum").len() == 1);
    let jobs = spans_named(&capture, "nagaya.job");
    assert_eq!(jobs.len(), 1);
    assert_eq!(attribute(&jobs[0], "outcome").as_deref(), Some("cancelled"));
}
