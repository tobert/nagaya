//! Jobs, quanta, classes, and the order of ready quanta.
//!
//! The scheduler is a state machine. The function [`step`] takes the state and one [`Event`]. It gives the new
//! state and the next [`Quantum`]. An event that does not fit the state is an error (spec 8.5).

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
    /// The most segments in one pass (`Limits.max_segments`).
    pub max_segments: u32,
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

impl Class {
    /// The attribute value for the class (spec 9.4).
    pub fn label(self) -> &'static str {
        match self {
            Class::Interactive => "interactive",
            Class::Batch => "batch",
            Class::Filler => "filler",
        }
    }
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
    /// The attribute value for the state.
    pub fn label(self) -> &'static str {
        match self {
            JobState::Ready => "ready",
            JobState::Running => "running",
            JobState::Done => "done",
            JobState::Cancelled => "cancelled",
            JobState::Failed => "failed",
            JobState::Expired => "expired",
        }
    }

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
/// one thread per device group). This is the D1 reading. A test state may
/// carry more than one engine to check that a pass never mixes engines;
/// the gate is per scheduler.
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

/// An event that does not fit the state.
///
/// The step function takes the state by value and drops it on an error. The caller must not continue with a
/// scheduler that has seen a wrong event: the caller reports the error and stops (spec 8.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepError {
    /// An `Arrive` event names a job id that the scheduler already holds.
    DuplicateJob(JobId),
    /// An `Arrive` event brings a job that is not `Ready`.
    ArrivalNotReady(JobId, JobState),
    /// An `Admit` or `Cancel` event names a job that the scheduler does not hold.
    UnknownJob(JobId),
    /// A `Complete` event arrives while no quantum runs.
    NothingRunning,
    /// A caller appends quanta to a job that has no quantum that runs.
    AppendToIdleJob(JobId, JobState),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::DuplicateJob(id) => write!(f, "job {id} arrived twice: the job id must be new"),
            StepError::ArrivalNotReady(id, state) => {
                write!(f, "job {id} arrived in state {state:?}: an arriving job must be Ready")
            }
            StepError::UnknownJob(id) => write!(f, "job {id} is not in the scheduler: send Arrive first"),
            StepError::NothingRunning => write!(f, "a quantum ended but none was running: send Complete only once"),
            StepError::AppendToIdleJob(id, state) => write!(
                f,
                "job {id} is in state {state:?}: append quanta only while a quantum of the job runs"
            ),
        }
    }
}

impl std::error::Error for StepError {}

/// Take one step: one event in, one new state and one next quantum out.
///
/// An event that does not fit the state returns a [`StepError`].
pub fn step(mut state: Scheduler, event: Event) -> Result<(Scheduler, Option<Quantum>), StepError> {
    let now = event.now();
    match event {
        Event::Arrive { job, .. } => {
            if state.jobs.iter().any(|held| held.id == job.id) {
                return Err(StepError::DuplicateJob(job.id));
            }
            if job.state != JobState::Ready {
                return Err(StepError::ArrivalNotReady(job.id, job.state));
            }
            state.jobs.push(job);
        }
        Event::Admit { job, .. } => {
            let held = state
                .jobs
                .iter_mut()
                .find(|candidate| candidate.id == job)
                .ok_or(StepError::UnknownJob(job))?;
            held.admitted = true;
        }
        Event::Complete {
            duration_ms, outcome, ..
        } => {
            let running = state.running.take().ok_or(StepError::NothingRunning)?;
            state.complete(running, duration_ms, outcome);
        }
        Event::Cancel { job, .. } => {
            let held = state
                .jobs
                .iter_mut()
                .find(|candidate| candidate.id == job)
                .ok_or(StepError::UnknownJob(job))?;
            if held.state == JobState::Running {
                held.cancel_requested = true;
            } else if !held.state.is_terminal() {
                held.state = JobState::Cancelled;
                held.quanta.clear();
            }
        }
        Event::Tick { .. } => {}
    }

    if state.running.is_some() {
        return Ok((state, None));
    }
    state.expire(now);
    let next = state.choose();
    Ok((state, next))
}

impl Scheduler {
    /// Add quanta to the end of a job, while one of its quanta runs.
    ///
    /// A job whose later quanta depend on its first one, such as a build after its lookup, calls this before the
    /// `Complete` event of the running quantum. A refusal changes nothing.
    pub fn append_quanta(&mut self, job: JobId, kinds: &[Kind]) -> Result<(), StepError> {
        let held = self
            .jobs
            .iter_mut()
            .find(|candidate| candidate.id == job)
            .ok_or(StepError::UnknownJob(job))?;
        if held.state != JobState::Running {
            return Err(StepError::AppendToIdleJob(job, held.state));
        }
        held.quanta.extend(kinds.iter().copied());
        Ok(())
    }

    /// Put quanta before the quanta that a running job has, in the given order. An interval copy uses it: the copy
    /// holds the state at the end of the quantum that ran (spec 6.13).
    pub fn prepend_quanta(&mut self, job: JobId, kinds: &[Kind]) -> Result<(), StepError> {
        let held = self
            .jobs
            .iter_mut()
            .find(|candidate| candidate.id == job)
            .ok_or(StepError::UnknownJob(job))?;
        if held.state != JobState::Running {
            return Err(StepError::AppendToIdleJob(job, held.state));
        }
        for kind in kinds.iter().rev() {
            held.quanta.push_front(*kind);
        }
        Ok(())
    }

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
                self.charge_segments(&running, duration_ms, &present);
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

    /// Charge each segment of a pass on its own (spec 5.4).
    ///
    /// A segment with the share `c` lowers its balance by `c * (n - 1) / n` and raises the balance of each other
    /// present job by `c / n`. Both round down. The rounding remainder goes to the earliest present job, so the sum
    /// stays zero. Then each balance is clamped.
    fn charge_segments(&mut self, running: &Quantum, duration_ms: u64, present: &[(JobId, u64)]) {
        let count = present.len() as u128;
        let duration = duration_ms as u128;
        let segment_count = running.segments.len() as u128;
        if segment_count == 0 {
            return;
        }
        let base = duration / segment_count;
        let remainder = duration % segment_count;
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
        let mut deltas: Vec<(JobId, i128)> = present.iter().map(|(id, _)| (*id, 0)).collect();
        for segment in &running.segments {
            let share = if Some(*segment) == earliest_segment {
                base + remainder
            } else {
                base
            };
            let fall = (share * (count - 1) / count) as i128;
            let rise = (share / count) as i128;
            for (id, delta) in &mut deltas {
                if id == segment {
                    *delta -= fall;
                } else {
                    *delta += rise;
                }
            }
        }
        let total: i128 = deltas.iter().map(|(_, delta)| *delta).sum();
        if let Some((earliest, _)) = present.iter().min_by_key(|(id, arrival)| (*arrival, *id))
            && let Some((_, delta)) = deltas.iter_mut().find(|(id, _)| id == earliest)
        {
            *delta -= total;
        }
        for (id, delta) in deltas {
            if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
                job.balance_ms = (job.balance_ms as i128 + delta).clamp(-30_000, 30_000) as i64;
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
            // An engine that is not listed takes one segment in a pass.
            let segment_limit = match self.engines.iter().find(|info| info.id == engine) {
                None => 1,
                Some(info) => {
                    assert!(
                        info.max_segments > 0,
                        "engine {engine} has max_segments 0, so no pass can run"
                    );
                    if info.segments_exact {
                        info.max_segments as usize
                    } else {
                        1
                    }
                }
            };
            let kind = *best.quanta.front()?;
            let mut segments = vec![best.id];
            for candidate in &candidates[1..] {
                if segments.len() >= segment_limit {
                    break;
                }
                // A pass holds segments of one kind (spec 5.7).
                if candidate.engine == engine && candidate.quanta.front() == Some(&kind) {
                    segments.push(candidate.id);
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

mod interval;

mod observed;

pub use interval::{ContextId, CopyDue, IntervalCopies, IntervalError, IntervalPolicy};
pub use observed::ObservedScheduler;
