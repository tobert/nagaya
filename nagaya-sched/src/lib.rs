//! Jobs, quanta, classes, and the order of ready quanta.
//!
//! Stub: the tests fail until the scheduler is built.

use std::collections::VecDeque;

use nagaya_abi::Kind;

/// One job.
pub type JobId = u64;

/// One device group.
pub type EngineId = u32;

/// One engine's scheduling facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engine {
    /// The device group.
    pub id: EngineId,
    /// True when the engine allows segments in one pass.
    pub segments_exact: bool,
}

/// How urgently a job's quanta run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Interactive work.
    Interactive,
    /// Batch work.
    Batch,
    /// Filler work.
    Filler,
}

/// The state of a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    /// The job can run its next quantum.
    Ready,
    /// The job has a quantum that runs.
    Running,
    /// The job finished.
    Done,
    /// A client cancelled the job.
    Cancelled,
    /// A quantum failed.
    Failed,
    /// The job passed its deadline.
    Expired,
}

impl JobState {
    /// True when the job is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobState::Done | JobState::Cancelled | JobState::Failed | JobState::Expired
        )
    }
}

/// One job and its remaining quanta.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    /// The job id.
    pub id: JobId,
    /// The job class.
    pub class: Class,
    /// The integer priority.
    pub priority: i64,
    /// The arrival time in milliseconds.
    pub arrival_ms: u64,
    /// The deadline in milliseconds, when the job has one.
    pub deadline_ms: Option<u64>,
    /// The device group.
    pub engine: EngineId,
    /// The kinds of the remaining quanta.
    pub quanta: VecDeque<Kind>,
    /// True when the ledger admitted the next quantum.
    pub admitted: bool,
    /// True when the inputs of the next quantum are present.
    pub inputs_present: bool,
    /// The job state.
    pub state: JobState,
    /// The owed balance in milliseconds.
    pub balance_ms: i64,
    /// True when a client cancelled a running job.
    pub cancel_requested: bool,
    /// The filler checkpoint interval in milliseconds.
    pub checkpoint_interval_ms: Option<u64>,
}

/// One quantum to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quantum {
    /// The device group.
    pub engine: EngineId,
    /// The quantum class.
    pub class: Class,
    /// The quantum kind.
    pub kind: Kind,
    /// The jobs that share the pass.
    pub segments: Vec<JobId>,
}

/// The scheduler state.
///
/// One scheduler serves one device group in the process (spec 3.2, D1:
/// one thread per device group). A test state may carry more than one
/// engine to check that a pass never mixes engines; the gate is per
/// scheduler.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scheduler {
    /// The jobs.
    pub jobs: Vec<Job>,
    /// The engines and their segment rules.
    pub engines: Vec<Engine>,
    /// The quantum that runs, when one does.
    pub running: Option<Quantum>,
}

/// How a quantum ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    /// The quantum finished.
    Done,
    /// The quantum failed.
    Failed,
}

/// One event for the scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A job arrives.
    Arrive {
        /// The time in milliseconds.
        now_ms: u64,
        /// The job.
        job: Job,
    },
    /// The ledger admits a job's next quantum.
    Admit {
        /// The time in milliseconds.
        now_ms: u64,
        /// The job.
        job: JobId,
    },
    /// The running quantum ends.
    Complete {
        /// The time in milliseconds.
        now_ms: u64,
        /// The quantum duration in milliseconds.
        duration_ms: u64,
        /// How the quantum ended.
        outcome: Completion,
    },
    /// A client cancels a job.
    Cancel {
        /// The time in milliseconds.
        now_ms: u64,
        /// The job.
        job: JobId,
    },
    /// Time passes with no other event.
    Tick {
        /// The time in milliseconds.
        now_ms: u64,
    },
}

impl Event {
    fn now(&self) -> u64 {
        match self {
            Event::Arrive { now_ms, .. }
            | Event::Admit { now_ms, .. }
            | Event::Complete { now_ms, .. }
            | Event::Cancel { now_ms, .. }
            | Event::Tick { now_ms } => *now_ms,
        }
    }
}

/// Take one step: one event in, one new state and one next quantum out.
pub fn step(mut state: Scheduler, event: Event) -> (Scheduler, Option<Quantum>) {
    let now = event.now();
    match event {
        Event::Arrive { job, .. } => state.jobs.push(job),
        Event::Admit { job, .. } => {
            if let Some(job) = state.jobs.iter_mut().find(|candidate| candidate.id == job) {
                job.admitted = true;
            }
        }
        Event::Complete {
            duration_ms, outcome, ..
        } => {
            if let Some(running) = state.running.take() {
                state.complete(running, duration_ms, outcome);
            }
        }
        Event::Cancel { job, .. } => {
            if let Some(job) = state.jobs.iter_mut().find(|candidate| candidate.id == job) {
                if job.state == JobState::Running {
                    job.cancel_requested = true;
                } else if !job.state.is_terminal() {
                    job.state = JobState::Cancelled;
                    job.quanta.clear();
                }
            }
        }
        Event::Tick { .. } => {}
    }

    if state.running.is_some() {
        return (state, None);
    }
    state.expire(now);
    let next = state.choose();
    (state, next)
}

impl Scheduler {
    fn complete(&mut self, running: Quantum, duration_ms: u64, outcome: Completion) {
        if running.class == Class::Interactive {
            let present: Vec<(JobId, u64)> = self
                .jobs
                .iter()
                .filter(|job| job.class == Class::Interactive && !job.state.is_terminal())
                .map(|job| (job.id, job.arrival_ms))
                .collect();
            let count = present.len() as u128;
            if count > 0 {
                let duration = duration_ms as u128;
                let segment_count = running.segments.len() as u128;
                let base = duration.checked_div(segment_count).unwrap_or(0);
                let remainder = if segment_count == 0 {
                    0
                } else {
                    duration % segment_count
                };
                let earliest_segment = running
                    .segments
                    .iter()
                    .min_by_key(|id| {
                        self.jobs
                            .iter()
                            .find(|job| job.id == **id)
                            .map(|job| (job.arrival_ms, job.id))
                            .unwrap_or((u64::MAX, **id))
                    })
                    .copied();
                let mut deltas: Vec<(JobId, i64)> = Vec::new();
                for (id, _) in &present {
                    let delta = if running.segments.contains(id) {
                        let charge = if Some(*id) == earliest_segment {
                            base + remainder
                        } else {
                            base
                        };
                        -((charge * (count - 1)).checked_div(count).unwrap_or(0) as i64)
                    } else {
                        duration.checked_div(count).unwrap_or(0) as i64
                    };
                    deltas.push((*id, delta));
                }
                let total: i64 = deltas.iter().map(|(_, delta)| *delta).sum();
                if total != 0
                    && let Some((earliest, _)) = present.iter().min_by_key(|(id, arrival)| (*arrival, *id))
                    && let Some((_, delta)) = deltas.iter_mut().find(|(id, _)| id == earliest)
                {
                    *delta -= total;
                }
                for (id, delta) in deltas {
                    if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
                        job.balance_ms = (job.balance_ms + delta).clamp(-30_000, 30_000);
                    }
                }
            }
        }

        for id in &running.segments {
            if let Some(job) = self.jobs.iter_mut().find(|job| job.id == *id) {
                if outcome == Completion::Failed {
                    job.state = JobState::Failed;
                    job.quanta.clear();
                } else if job.cancel_requested {
                    job.state = JobState::Cancelled;
                    job.quanta.clear();
                } else if job.quanta.is_empty() {
                    job.state = JobState::Done;
                } else {
                    job.state = JobState::Ready;
                }
            }
        }
    }

    fn expire(&mut self, now_ms: u64) {
        for job in &mut self.jobs {
            if job.state == JobState::Ready && job.deadline_ms.is_some_and(|deadline| deadline <= now_ms) {
                job.state = JobState::Expired;
                job.quanta.clear();
            }
        }
    }

    fn choose(&mut self) -> Option<Quantum> {
        let (engine, class, kind, segments) = {
            let ready: Vec<&Job> = self
                .jobs
                .iter()
                .filter(|job| {
                    job.state == JobState::Ready && job.inputs_present && job.admitted && !job.quanta.is_empty()
                })
                .collect();
            if ready.is_empty() {
                return None;
            }
            let class = if ready.iter().any(|job| job.class == Class::Interactive) {
                Class::Interactive
            } else if ready.iter().any(|job| job.class == Class::Batch) {
                Class::Batch
            } else {
                Class::Filler
            };
            let mut candidates: Vec<&Job> = ready.into_iter().filter(|job| job.class == class).collect();
            if class == Class::Interactive {
                candidates.sort_by(|left, right| {
                    right
                        .priority
                        .cmp(&left.priority)
                        .then(right.balance_ms.cmp(&left.balance_ms))
                        .then(left.arrival_ms.cmp(&right.arrival_ms))
                });
            } else {
                candidates.sort_by(|left, right| {
                    right
                        .priority
                        .cmp(&left.priority)
                        .then(left.arrival_ms.cmp(&right.arrival_ms))
                });
            }
            let best = candidates.first().copied()?;
            let engine = best.engine;
            let engine_exact = self
                .engines
                .iter()
                .find(|info| info.id == engine)
                .map(|info| info.segments_exact)
                .unwrap_or(false);
            let kind = *best.quanta.front()?;
            let mut segments = vec![best.id];
            if engine_exact {
                for candidate in &candidates[1..] {
                    if candidate.engine == engine {
                        segments.push(candidate.id);
                    }
                }
            }
            (engine, class, kind, segments)
        };

        let running = Quantum {
            engine,
            class,
            kind,
            segments,
        };
        for id in &running.segments {
            if let Some(job) = self.jobs.iter_mut().find(|job| job.id == *id) {
                job.quanta.pop_front();
                job.state = JobState::Running;
            }
        }
        self.running = Some(running.clone());
        Some(running)
    }
}
