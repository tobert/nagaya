//! Checks of the attribute `engine` on the telemetry of jobs (spec 9.2, "The engine of a job"). A job span, its
//! quantum spans and the quantum metrics name the engine of the job, not the joined name of the group.

#![allow(dead_code)]

use std::time::{Duration, Instant};

use nagaya_telemetry::testing::{Capture, SpanData};

fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|item| item.key.as_str() == key)
        .map(|item| item.value.to_string())
}

fn spans_named(capture: &Capture, name: &str) -> Vec<SpanData> {
    capture.spans().into_iter().filter(|span| span.name == name).collect()
}

fn count(capture: &Capture, metric: &str) -> u64 {
    capture.metric(metric).iter().map(|point| point.count).sum()
}

/// Wait until `jobs` job spans ended, and each quantum that started has ended.
///
/// A job span ends after the body tells the test that it is done, so the test waits here.
pub fn wait_for_jobs(capture: &Capture, jobs: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let ended = spans_named(capture, "nagaya.job").len();
        let started = count(capture, "nagaya.quantum.wait.duration");
        let ran = count(capture, "nagaya.quantum.run.duration");
        if ended >= jobs && started == ran && ran > 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the jobs never ended: {ended} of {jobs} job spans, {started} quanta started, {ran} ended"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Check that the job spans name exactly the engines `expected`, each at least once. Each quantum span names the
/// engine of its job, and each point of the two quantum durations names one of `expected`.
///
/// `with_spans` names the engines whose jobs run a quantum that is not decode, so that they have quantum spans.
pub fn assert_own_engines(capture: &Capture, expected: &[&str], with_spans: &[&str]) {
    let jobs = spans_named(capture, "nagaya.job");
    let mut named: Vec<String> = jobs
        .iter()
        .map(|job| attribute(job, "engine").expect("a job span has the attribute engine"))
        .collect();
    named.sort();
    named.dedup();
    let mut wanted: Vec<String> = expected.iter().map(|name| (*name).to_owned()).collect();
    wanted.sort();
    assert_eq!(named, wanted, "the job spans name the engine of each job");

    let quanta = spans_named(capture, "nagaya.quantum");
    let mut quantum_engines: Vec<String> = Vec::new();
    for quantum in &quanta {
        let job = jobs
            .iter()
            .find(|job| job.span_context.span_id() == quantum.parent_span_id)
            .unwrap_or_else(|| panic!("a quantum span has a job span as its parent: {quantum:?}"));
        let engine = attribute(quantum, "engine");
        assert_eq!(
            engine,
            attribute(job, "engine"),
            "a quantum span names the engine of its job"
        );
        quantum_engines.extend(engine);
    }
    for name in with_spans {
        assert!(
            quantum_engines.iter().any(|engine| engine == name),
            "no quantum span names engine `{name}`: {quantum_engines:?}"
        );
    }

    for metric in ["nagaya.quantum.wait.duration", "nagaya.quantum.run.duration"] {
        let points = capture.metric(metric);
        for point in &points {
            assert!(
                expected.contains(&point.attributes["engine"].as_str()),
                "a point of {metric} names engine `{}`, not one of {expected:?}",
                point.attributes["engine"]
            );
        }
        for name in expected {
            assert!(
                points.iter().any(|point| point.attributes["engine"] == *name),
                "no point of {metric} names engine `{name}`: {points:?}"
            );
        }
    }
}
