//! The scheduler with its telemetry (spec 9.2, 9.4).
//!
//! [`step`] stays a pure function. [`ObservedScheduler`] calls it, and then records what the step showed: the wait of a
//! quantum, its run time, and its span.

use std::collections::BTreeMap;

use nagaya_abi::Kind;
use nagaya_telemetry::{QuantumLabels, instruments};
use tracing::Span;

use crate::{Class, Completion, EngineId, Event, JobId, JobState, Quantum, Scheduler, StepError, step};

/// A scheduler that records the metrics and the spans of its quanta.
///
/// Use it in place of [`step`] on the path that runs a device group. It gives the same states and the same quanta.
///
/// The wait of a quantum is the time from the moment its lead segment became ready to the moment the quantum started.
/// The lead segment is the first of `Quantum::segments`. A job is ready when it arrives, and again when a quantum
/// of it ends and more quanta remain. A job that the ledger has not admitted yet is ready, so its wait includes the
/// time until the admission.
///
/// Spec 9.2: a decode quantum makes metrics only. Every other kind makes the span `nagaya.quantum`.
pub struct ObservedScheduler {
    state: Scheduler,
    engines: BTreeMap<EngineId, String>,
    ready_since: BTreeMap<JobId, u64>,
    span: Option<Span>,
}

impl ObservedScheduler {
    /// Wrap `state`. `engines` gives the engine name of each device group that a quantum can run on.
    ///
    /// A job that is ready now counts its wait from its `arrival_ms`.
    pub fn new(state: Scheduler, engines: BTreeMap<EngineId, String>) -> Self {
        let ready_since = state
            .jobs
            .iter()
            .filter(|job| job.state == JobState::Ready)
            .map(|job| (job.id, job.arrival_ms))
            .collect();
        Self {
            state,
            engines,
            ready_since,
            span: None,
        }
    }

    /// The scheduler state.
    pub fn state(&self) -> &Scheduler {
        &self.state
    }

    /// Take one step, as [`step`] does, and record it.
    ///
    /// # Panics
    ///
    /// Panics when a quantum runs on a device group that `new` gave no engine name. That is a defect in the caller.
    pub fn step(mut self, event: Event) -> Result<(Self, Option<Quantum>), StepError> {
        let now = event_time(&event);
        let finished = match &event {
            Event::Complete {
                duration_ms, outcome, ..
            } => self
                .state
                .running
                .clone()
                .map(|quantum| (quantum, *duration_ms, *outcome)),
            _ => None,
        };

        let (state, next) = step(self.state, event)?;
        self.state = state;

        if let Some((quantum, duration_ms, outcome)) = finished {
            self.record_end(&quantum, duration_ms, outcome);
        }
        for job in &self.state.jobs {
            let dispatched = next.as_ref().is_some_and(|quantum| quantum.segments.contains(&job.id));
            if job.state == JobState::Ready || dispatched {
                self.ready_since.entry(job.id).or_insert(now);
            }
        }
        if let Some(quantum) = &next {
            self.record_start(quantum, now);
        }
        let jobs = &self.state.jobs;
        self.ready_since
            .retain(|id, _| jobs.iter().any(|job| job.id == *id && job.state == JobState::Ready));
        Ok((self, next))
    }

    fn engine_name(&self, group: EngineId) -> &str {
        self.engines.get(&group).unwrap_or_else(|| {
            panic!("device group {group} has no engine name: give one for each group in `ObservedScheduler::new`")
        })
    }

    fn record_start(&mut self, quantum: &Quantum, now: u64) {
        let lead = quantum.segments.first().expect("a quantum holds a segment");
        let ready = self
            .ready_since
            .get(lead)
            .expect("the lead segment of a quantum was ready");
        // Event times do not go back. A caller that sends an earlier time gets a wait of zero.
        let wait_ms = now.saturating_sub(*ready);
        let engine = self.engine_name(quantum.engine).to_owned();
        instruments().quantum_wait(&labels(&engine, quantum), wait_ms as f64);
        self.span = (quantum.kind != Kind::Decode).then(|| {
            tracing::info_span!(
                "nagaya.quantum",
                engine = engine.as_str(),
                device_group = quantum.engine,
                kind = kind_label(quantum.kind),
                class = class_label(quantum.class),
                segments = quantum.segments.len(),
                wait_ms,
                run_ms = tracing::field::Empty,
                outcome = tracing::field::Empty,
                otel.status_code = tracing::field::Empty,
            )
        });
    }

    fn record_end(&mut self, quantum: &Quantum, duration_ms: u64, outcome: Completion) {
        let engine = self.engine_name(quantum.engine);
        instruments().quantum_run(&labels(engine, quantum), duration_ms as f64);
        if let Some(span) = self.span.take() {
            span.record("run_ms", duration_ms);
            match outcome {
                Completion::Done => span.record("outcome", "done"),
                Completion::Failed => {
                    span.record("outcome", "failed");
                    span.record("otel.status_code", "ERROR")
                }
            };
        }
    }
}

fn labels<'a>(engine: &'a str, quantum: &Quantum) -> QuantumLabels<'a> {
    QuantumLabels {
        engine,
        device_group: quantum.engine,
        kind: kind_label(quantum.kind),
        class: class_label(quantum.class),
    }
}

fn event_time(event: &Event) -> u64 {
    match event {
        Event::Arrive { now_ms, .. }
        | Event::Admit { now_ms, .. }
        | Event::Complete { now_ms, .. }
        | Event::Cancel { now_ms, .. }
        | Event::Tick { now_ms } => *now_ms,
    }
}

/// The attribute value for a quantum kind.
fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::Read => "read",
        Kind::Prefill => "prefill",
        Kind::Decode => "decode",
        Kind::Embed => "embed",
        Kind::Copy => "copy",
        Kind::Step => "step",
        Kind::Train => "train",
    }
}

/// The attribute value for a class.
fn class_label(class: Class) -> &'static str {
    match class {
        Class::Interactive => "interactive",
        Class::Batch => "batch",
        Class::Filler => "filler",
    }
}
