//! The jobs of a device group: the scheduler's state machine, driven on the group thread (spec 3.2, 5).
//!
//! A [`JobBody`] holds the work of one request. The group thread asks [`GroupJobs`] for one quantum at a time. The
//! scheduler chooses the quantum (`ObservedScheduler::step`), and the body runs it. The scheduler never sees the body.
//!
//! The group thread records what it runs (spec 9.2, 9.4, 9.6). Each job has a span `nagaya.job`. Each quantum that is
//! not decode has a span `nagaya.quantum` under it, and each quantum has its two durations. A quantum that fails
//! logs the change of the machine's counters.

use std::collections::BTreeMap;
use std::time::Instant;

use nagaya_abi::{Engine, Kind, PassError, PassOut, Segment};
use nagaya_sched::{Class, Completion, Event, Item, Job, JobId, JobState, ObservedScheduler, Scheduler};
use tracing::Span;

use crate::evidence::{self, Machine};

/// What a body reports after one quantum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ran {
    /// The quantum ran. The job has no new quantum to add.
    Done,
    /// The quantum ran, and the job has these quanta to add after the ones it has.
    More(Vec<Kind>),
    /// The quantum ran, and the job runs these quanta before the ones it has, for example an interval copy (spec
    /// 6.13).
    First(Vec<Kind>),
    /// The quantum failed. The job ends, and the body has told its caller why.
    Failed,
    /// The quantum ran, and the body cancels its job, for example because its client went away. The job ends as
    /// cancelled (spec 5.9).
    Cancelled,
}

/// The work of one job.
pub trait JobBody: Send {
    /// Run the quantum of the given kind. `engines` are the engines of the group, in profile order.
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran;

    /// Hear that the job reached a terminal state. The group calls it once, before it drops the body. A job can end
    /// with no quantum that ran, for example when it passed its deadline (spec 5.8).
    fn ended(&mut self, _state: JobState) {}

    /// The kind of request that the job serves, for the span `nagaya.job` (spec 9.2). The set of values is bounded.
    fn request(&self) -> &'static str {
        "job"
    }

    /// The item of a quantum of `kind` on an engine with stages (engine-abi D1). A `Block` is a pass that the group
    /// runs stage by stage through [`JobBody::begin_pass`] and [`JobBody::end_pass`]. The default is a `Whole` copy
    /// quantum, which [`JobBody::run`] runs while it holds every device.
    fn item(&self, kind: Kind) -> Item {
        Item::Whole(kind)
    }

    /// Begin a pass of `kind` on an engine with stages: give its segments. A body whose `item` gives a `Block`
    /// implements it. An error is `Ran::Failed` or `Ran::Cancelled`, and the body told its caller why.
    fn begin_pass(&mut self, kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        panic!("a {kind:?} quantum of this job is not a pass; its item must not be a Block")
    }

    /// End the oldest pass that began, with what its last stage gave. The passes of one job end in order.
    fn end_pass(&mut self, kind: Kind, _result: Result<PassOut, PassError>, _engines: &mut [Box<dyn Engine>]) -> Ran {
        panic!("a {kind:?} quantum of this job is not a pass; its item must not be a Block")
    }
}

/// A job that a request brings to a group.
pub struct JobSpec {
    /// The class of the job (spec 5.3).
    pub class: Class,
    /// The integer priority (spec 5.4).
    pub priority: i64,
    /// The time after which the job runs no more quanta (spec 5.8).
    pub deadline: Option<Instant>,
    /// The kind of the first quantum. The body adds the others when it runs the first one.
    pub first: Kind,
    /// The engine of the job: its place in the engines of the group, in profile order. A job for an engine that is
    /// not loaded waits for a swap (spec 3.3a).
    pub engine: usize,
    /// The work.
    pub body: Box<dyn JobBody>,
}

/// The state of the jobs of one group.
pub struct GroupJobs {
    group: u32,
    /// The names of the engines of the group, as the attribute `engine` of spans and metrics.
    engine: String,
    machine: Machine,
    epoch: Instant,
    scheduler: ObservedScheduler,
    bodies: BTreeMap<JobId, Box<dyn JobBody>>,
    /// The engine of each job that has not ended, as its place in the group (spec 3.3a).
    engines: BTreeMap<JobId, usize>,
    /// The span `nagaya.job` of each job that has not ended.
    spans: BTreeMap<JobId, Span>,
    /// The jobs that arrived since the last quantum. Their inputs become present when the next quantum is chosen.
    fresh: Vec<JobId>,
    /// The quantum that ran last, until the scheduler learns that it ended. The end event chooses the next quantum.
    finished: Option<(u64, Completion)>,
    next_id: JobId,
}

impl GroupJobs {
    /// Make the jobs of a group. The scheduler serves one device group (spec 5.5).
    ///
    /// `engine` is the attribute `engine` of the group. A group with several engines gives their names in profile
    /// order, joined with `+`. The set of values stays bounded by the profile.
    pub fn new(group: u32, engine: impl Into<String>) -> Self {
        let engine = engine.into();
        Self {
            group,
            machine: Machine::real(),
            epoch: Instant::now(),
            scheduler: ObservedScheduler::new(Scheduler::default(), BTreeMap::from([(group, engine.clone())])),
            engine,
            bodies: BTreeMap::new(),
            engines: BTreeMap::new(),
            spans: BTreeMap::new(),
            fresh: Vec::new(),
            finished: None,
            next_id: 1,
        }
    }

    /// Read the counters of `machine` before a quantum (spec 9.6). The default is the machine that runs this process.
    pub fn machine(mut self, machine: Machine) -> Self {
        self.machine = machine;
        self
    }

    /// True when no job waits or runs.
    pub fn idle(&self) -> bool {
        self.bodies.is_empty() && self.finished.is_none()
    }

    /// True when a job of the engine at `engine` in the group has not ended (spec 3.3a).
    pub fn holds(&self, engine: usize) -> bool {
        self.engines.values().any(|held| *held == engine)
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn event(&mut self, event: Event) {
        // A wrong event is a defect of this driver, and the state is gone (spec 8.5).
        let chosen = self
            .scheduler
            .step(event)
            .expect("the group driver sent an event that does not fit");
        assert!(
            chosen.is_none() || self.scheduler.state().running == chosen,
            "the scheduler chose a quantum that it does not hold as running"
        );
    }

    /// Add a job. It waits for the next call of [`GroupJobs::run_one`]. The span of the job is the child of the span
    /// that is current.
    pub fn add(&mut self, spec: JobSpec) -> JobId {
        self.add_under(spec, Span::current())
    }

    /// Add a job as [`GroupJobs::add`] does. The span of the job is the child of `parent`: the span of the request that
    /// brought the job (spec 9.2).
    pub fn add_under(&mut self, spec: JobSpec, parent: Span) -> JobId {
        let id = self.next_id;
        self.next_id += 1;
        let now_ms = self.now_ms();
        let deadline_ms = spec
            .deadline
            .map(|deadline| deadline.saturating_duration_since(self.epoch).as_millis() as u64);
        let job = Job {
            id,
            class: spec.class,
            priority: spec.priority,
            arrival_ms: now_ms,
            deadline_ms,
            engine: self.group,
            quanta: [spec.first].into_iter().collect(),
            admitted: false,
            inputs_present: false,
            state: JobState::Ready,
            balance_ms: 0,
            cancel_requested: false,
            checkpoint_interval_ms: None,
        };
        let span = tracing::info_span!(
            parent: &parent,
            "nagaya.job",
            engine = self.engine.as_str(),
            class = spec.class.label(),
            request = spec.body.request(),
            outcome = tracing::field::Empty,
        );
        self.scheduler.set_job_span(id, span.clone());
        self.spans.insert(id, span);
        self.event(Event::Arrive { now_ms, job });
        // The ledger admitted the context before the job came, and a pass never allocates (spec 4.3).
        self.event(Event::Admit { now_ms, job: id });
        self.bodies.insert(id, spec.body);
        self.engines.insert(id, spec.engine);
        self.fresh.push(id);
        id
    }

    /// Choose and run one quantum. It gives false when no quantum was ready.
    ///
    /// The choice happens here, after every arrival so far, so a job that arrived while the last quantum ran can go
    /// first (spec 5.3). The end of the last quantum reaches the scheduler here too.
    pub fn run_one(&mut self, engines: &mut [Box<dyn Engine>]) -> bool {
        for id in std::mem::take(&mut self.fresh) {
            if let Some(job) = self.scheduler.state_mut().jobs.iter_mut().find(|job| job.id == id) {
                job.inputs_present = true;
            }
        }
        let now_ms = self.now_ms();
        match self.finished.take() {
            Some((duration_ms, outcome)) => self.event(Event::Complete {
                now_ms,
                duration_ms,
                outcome,
            }),
            None => self.event(Event::Tick { now_ms }),
        }
        self.forget_ended_jobs();
        let Some(quantum) = self.scheduler.state().running.clone() else {
            return false;
        };
        assert!(
            quantum.segments.len() == 1,
            "the group driver runs one segment in a pass; the scheduler joined {}",
            quantum.segments.len()
        );
        let id = quantum.segments[0];
        let body = self.bodies.get_mut(&id).expect("a chosen job has a body");
        // Reading the counters costs four files, so a decode quantum skips it (spec 9.6).
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        let span = self.scheduler.running_span().cloned();
        let started = Instant::now();
        // The spans of the engine nest under the span of the quantum (spec 9.2).
        let ran = match &span {
            Some(span) => span.in_scope(|| body.run(quantum.kind, engines)),
            None => body.run(quantum.kind, engines),
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        if matches!(ran, Ran::Failed) {
            let failure = format!(
                "{} quantum of a {} request on engine {}",
                format!("{:?}", quantum.kind).to_lowercase(),
                body.request(),
                self.engine
            );
            match watch {
                Some(watch) => watch.failed(&failure),
                None => evidence::failed_unwatched(&failure),
            };
        }
        let outcome = match ran {
            Ran::Done => Completion::Done,
            Ran::More(kinds) => {
                self.scheduler
                    .state_mut()
                    .append_quanta(id, &kinds)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            Ran::First(kinds) => {
                self.scheduler
                    .state_mut()
                    .prepend_quanta(id, &kinds)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            Ran::Failed => Completion::Failed,
            Ran::Cancelled => {
                // The quantum ran to its end. A cancel of a running job takes effect when it completes (spec 5.9).
                self.event(Event::Cancel {
                    now_ms: self.now_ms(),
                    job: id,
                });
                Completion::Done
            }
        };
        self.finished = Some((duration_ms, outcome));
        true
    }

    /// Drop the jobs that reached a terminal state, and their bodies. A job that leaves takes its balance (spec 5.4).
    fn forget_ended_jobs(&mut self) {
        let ended: Vec<(JobId, JobState)> = self
            .scheduler
            .state()
            .jobs
            .iter()
            .filter(|job| job.state.is_terminal())
            .map(|job| (job.id, job.state))
            .collect();
        self.scheduler.state_mut().jobs.retain(|job| !job.state.is_terminal());
        for (id, state) in ended {
            if let Some(span) = self.spans.remove(&id) {
                span.record("outcome", state.label());
            }
            let mut body = self.bodies.remove(&id).expect("a job that ended has a body");
            self.engines.remove(&id);
            body.ended(state);
        }
    }

    /// Cancel every job that has not ended (spec 10.3, 5.9). It gives the number of jobs that it dropped.
    ///
    /// A body that drops its result sender tells its caller that the job ended with no result.
    pub fn cancel_all(&mut self) -> usize {
        let count = self.bodies.len();
        self.bodies.clear();
        self.engines.clear();
        self.fresh.clear();
        for (_, span) in std::mem::take(&mut self.spans) {
            span.record("outcome", JobState::Cancelled.label());
        }
        // The last quantum ran. Its end still counts, although no job follows it.
        self.scheduler.abandon(self.finished.take());
        count
    }
}
