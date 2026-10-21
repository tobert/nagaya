//! `ObservedScheduler` records the metrics and the spans of a quantum from the real step path (spec 9.2, 9.4, 9.5).

use std::collections::BTreeMap;

use nagaya_abi::Kind;
use nagaya_sched::{Class, Completion, Event, Job, JobId, JobState, ObservedScheduler, Quantum, Scheduler, StepError};
use nagaya_telemetry::testing::{Capture, Point, SpanData};
use opentelemetry::KeyValue;

fn job(id: JobId, class: Class, arrival_ms: u64, engine: u32, quanta: &[Kind]) -> Job {
    Job {
        id,
        class,
        priority: 0,
        arrival_ms,
        deadline_ms: None,
        engine,
        quanta: quanta.iter().copied().collect(),
        admitted: false,
        inputs_present: true,
        state: JobState::Ready,
        balance_ms: 0,
        cancel_requested: false,
        checkpoint_interval_ms: None,
    }
}

fn names() -> BTreeMap<u32, String> {
    BTreeMap::from([(0, "qwen".to_owned()), (1, "toy".to_owned())])
}

fn observed() -> ObservedScheduler {
    ObservedScheduler::new(Scheduler::default(), names())
}

fn step(scheduler: ObservedScheduler, event: Event) -> (ObservedScheduler, Option<Quantum>) {
    scheduler.step(event).expect("the event fits the state")
}

fn arrive(scheduler: ObservedScheduler, now_ms: u64, job: Job) -> ObservedScheduler {
    step(scheduler, Event::Arrive { now_ms, job }).0
}

fn admit(scheduler: ObservedScheduler, now_ms: u64, job: JobId) -> (ObservedScheduler, Option<Quantum>) {
    step(scheduler, Event::Admit { now_ms, job })
}

fn complete(scheduler: ObservedScheduler, now_ms: u64, duration_ms: u64) -> (ObservedScheduler, Option<Quantum>) {
    step(
        scheduler,
        Event::Complete {
            now_ms,
            duration_ms,
            outcome: Completion::Done,
        },
    )
}

fn labels(point: &Point) -> (String, String, String, String) {
    (
        point.attributes["engine"].clone(),
        point.attributes["device_group"].clone(),
        point.attributes["kind"].clone(),
        point.attributes["class"].clone(),
    )
}

fn label(engine: &str, group: &str, kind: &str, class: &str) -> (String, String, String, String) {
    (engine.to_owned(), group.to_owned(), kind.to_owned(), class.to_owned())
}

fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|item: &&KeyValue| item.key.as_str() == key)
        .map(|item| item.value.to_string())
}

#[test]
fn the_wait_runs_from_ready_to_the_start_of_the_quantum() {
    let capture = Capture::start(Vec::new());
    let scheduler = arrive(observed(), 100, job(1, Class::Interactive, 100, 0, &[Kind::Prefill]));
    let (scheduler, next) = admit(scheduler, 130, 1);
    assert!(next.is_some(), "the admission starts the quantum");
    let (_, _) = complete(scheduler, 150, 20);

    let wait = capture.metric("nagaya.quantum.wait.duration");
    assert_eq!(wait.len(), 1, "{wait:?}");
    assert_eq!((wait[0].count, wait[0].value), (1, 30.0));
    assert_eq!(labels(&wait[0]), label("qwen", "0", "prefill", "interactive"));
}

#[test]
fn the_run_time_is_the_duration_of_the_complete_event() {
    let capture = Capture::start(Vec::new());
    let scheduler = arrive(observed(), 0, job(1, Class::Batch, 0, 1, &[Kind::Decode]));
    let (scheduler, _) = admit(scheduler, 5, 1);
    complete(scheduler, 90, 77);

    let run = capture.metric("nagaya.quantum.run.duration");
    assert_eq!(run.len(), 1, "{run:?}");
    assert_eq!((run[0].count, run[0].value), (1, 77.0));
    assert_eq!(labels(&run[0]), label("toy", "1", "decode", "batch"));
}

#[test]
fn a_job_that_waits_for_another_job_counts_the_wait_of_its_own_quantum() {
    let capture = Capture::start(Vec::new());
    let mut first = job(1, Class::Interactive, 0, 0, &[Kind::Prefill]);
    first.admitted = true;
    let mut second = job(2, Class::Batch, 0, 0, &[Kind::Prefill]);
    second.admitted = true;
    let scheduler = arrive(observed(), 0, first);
    let (scheduler, started) = step(
        scheduler,
        Event::Arrive {
            now_ms: 10,
            job: second,
        },
    );
    assert!(started.is_none(), "the first job started at its arrival");
    let (scheduler, next) = complete(scheduler, 60, 50);
    assert_eq!(next.map(|quantum| quantum.segments), Some(vec![2]));
    complete(scheduler, 100, 40);

    let mut waits = capture.metric("nagaya.quantum.wait.duration");
    waits.sort_by_key(|point| point.attributes["class"].clone());
    // The interactive job started the moment it arrived. The batch job arrived at 10 and started at 60.
    assert_eq!(waits.iter().map(|point| point.value).collect::<Vec<_>>(), [50.0, 0.0]);
    assert_eq!(waits[0].attributes["class"], "batch");
}

#[test]
fn a_job_that_returns_to_ready_waits_from_the_end_of_its_last_quantum() {
    let capture = Capture::start(Vec::new());
    let mut waiting = job(1, Class::Batch, 0, 0, &[Kind::Prefill, Kind::Prefill]);
    waiting.admitted = true;
    let mut urgent = job(2, Class::Interactive, 0, 0, &[Kind::Prefill]);
    urgent.admitted = true;
    let scheduler = arrive(observed(), 0, waiting);
    // The batch job runs its first quantum from 0 to 10. Then the interactive job arrives and runs from 10 to 40.
    let scheduler = arrive(scheduler, 5, urgent);
    let (scheduler, next) = complete(scheduler, 10, 10);
    assert_eq!(next.map(|quantum| quantum.segments), Some(vec![2]));
    let (scheduler, next) = complete(scheduler, 40, 30);
    assert_eq!(next.map(|quantum| quantum.segments), Some(vec![1]));
    complete(scheduler, 60, 20);

    let mut waits = capture.metric("nagaya.quantum.wait.duration");
    waits.sort_by_key(|point| point.attributes["class"].clone());
    // Batch quanta: the first waited 0, and the second waited from 10 (the end of the first) to 40.
    assert_eq!(waits[0].attributes["class"], "batch");
    assert_eq!((waits[0].count, waits[0].value), (2, 30.0));
    // The interactive quantum arrived at 5 and started at 10.
    assert_eq!((waits[1].count, waits[1].value), (1, 5.0));
}

#[test]
fn a_prefill_quantum_makes_a_span_with_its_labels_and_times() {
    let capture = Capture::start(Vec::new());
    let scheduler = arrive(observed(), 100, job(1, Class::Interactive, 100, 0, &[Kind::Prefill]));
    let (scheduler, _) = admit(scheduler, 130, 1);
    complete(scheduler, 150, 20);

    let spans = capture.spans();
    let span = spans
        .iter()
        .find(|span| span.name == "nagaya.quantum")
        .expect("a quantum span");
    for (key, value) in [
        ("engine", "qwen"),
        ("device_group", "0"),
        ("kind", "prefill"),
        ("class", "interactive"),
        ("segments", "1"),
        ("wait_ms", "30"),
        ("run_ms", "20"),
        ("outcome", "done"),
    ] {
        assert_eq!(attribute(span, key).as_deref(), Some(value), "attribute {key}");
    }
}

#[test]
fn a_decode_quantum_makes_metrics_and_no_span() {
    let capture = Capture::start(Vec::new());
    let scheduler = arrive(observed(), 0, job(1, Class::Interactive, 0, 0, &[Kind::Decode]));
    let (scheduler, _) = admit(scheduler, 1, 1);
    complete(scheduler, 9, 8);

    assert!(
        capture.spans().iter().all(|span| span.name != "nagaya.quantum"),
        "one span for each decode would bury a trace (spec 9.2)"
    );
    assert_eq!(capture.metric("nagaya.quantum.run.duration").len(), 1);
    assert_eq!(capture.metric("nagaya.quantum.wait.duration").len(), 1);
}

#[test]
fn a_failed_quantum_marks_its_span_as_an_error() {
    let capture = Capture::start(Vec::new());
    let scheduler = arrive(observed(), 0, job(1, Class::Batch, 0, 0, &[Kind::Prefill]));
    let (scheduler, _) = admit(scheduler, 1, 1);
    step(
        scheduler,
        Event::Complete {
            now_ms: 9,
            duration_ms: 8,
            outcome: Completion::Failed,
        },
    );

    let spans = capture.spans();
    let span = spans
        .iter()
        .find(|span| span.name == "nagaya.quantum")
        .expect("a quantum span");
    assert_eq!(attribute(span, "outcome").as_deref(), Some("failed"));
    assert!(
        matches!(span.status, opentelemetry::trace::Status::Error { .. }),
        "{:?}",
        span.status
    );
}

#[test]
fn a_pass_with_two_segments_counts_one_quantum_and_the_wait_of_its_lead_segment() {
    let capture = Capture::start(Vec::new());
    let engines = vec![nagaya_sched::Engine {
        id: 0,
        segments_exact: true,
        max_segments: 4,
    }];
    let mut lead = job(1, Class::Batch, 4, 0, &[Kind::Prefill]);
    lead.admitted = true;
    lead.priority = 1;
    let mut follower = job(2, Class::Batch, 0, 0, &[Kind::Prefill]);
    follower.admitted = true;
    let state = Scheduler {
        jobs: vec![lead, follower],
        engines,
        running: None,
    };
    let scheduler = ObservedScheduler::new(state, names());
    let (scheduler, next) = step(scheduler, Event::Tick { now_ms: 10 });
    assert_eq!(next.map(|quantum| quantum.segments), Some(vec![1, 2]));
    complete(scheduler, 30, 20);

    let waits = capture.metric("nagaya.quantum.wait.duration");
    assert_eq!(waits.len(), 1, "{waits:?}");
    // The lead segment is job 1. It was ready from 4 and started at 10.
    assert_eq!((waits[0].count, waits[0].value), (1, 6.0));
    let spans = capture.spans();
    let span = spans
        .iter()
        .find(|span| span.name == "nagaya.quantum")
        .expect("a quantum span");
    assert_eq!(attribute(span, "segments").as_deref(), Some("2"));
}

#[test]
fn the_wrapped_step_gives_the_state_and_the_quantum_of_the_plain_step() {
    let _capture = Capture::start(Vec::new());
    let events = vec![
        Event::Arrive {
            now_ms: 0,
            job: job(1, Class::Interactive, 0, 0, &[Kind::Prefill, Kind::Decode]),
        },
        Event::Admit { now_ms: 3, job: 1 },
        Event::Complete {
            now_ms: 9,
            duration_ms: 6,
            outcome: Completion::Done,
        },
        Event::Tick { now_ms: 12 },
    ];
    let mut plain = Scheduler::default();
    let mut wrapped = observed();
    for event in events {
        let (next_plain, quantum_plain) = nagaya_sched::step(plain, event.clone()).unwrap();
        let (next_wrapped, quantum_wrapped) = wrapped.step(event).unwrap();
        assert_eq!(quantum_plain, quantum_wrapped);
        assert_eq!(&next_plain, next_wrapped.state());
        plain = next_plain;
        wrapped = next_wrapped;
    }
}

#[test]
fn an_event_that_does_not_fit_is_an_error_and_records_nothing() {
    let capture = Capture::start(Vec::new());
    let error = observed()
        .step(Event::Complete {
            now_ms: 1,
            duration_ms: 1,
            outcome: Completion::Done,
        })
        .map(|_| ())
        .unwrap_err();
    assert_eq!(error, StepError::NothingRunning);
    assert!(capture.points().is_empty());
}

#[test]
#[should_panic(expected = "device group 7")]
fn a_quantum_on_a_group_with_no_engine_name_is_a_defect() {
    let scheduler = observed();
    let mut stray = job(1, Class::Batch, 0, 7, &[Kind::Prefill]);
    stray.admitted = true;
    scheduler.step(Event::Arrive { now_ms: 0, job: stray }).unwrap();
}
