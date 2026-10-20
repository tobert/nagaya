//! The jobs of a device group: the scheduler's state machine, driven on the group thread (spec 3.2, 5).
//!
//! A [`JobBody`] holds the work of one request. The group thread asks [`GroupJobs`] for one quantum at a time. The
//! scheduler chooses the quantum (`nagaya_sched::step`), and the body runs it. The scheduler never sees the body.

use std::collections::BTreeMap;
use std::time::Instant;

use nagaya_abi::{Engine, Kind};
use nagaya_sched::{Class, Completion, Event, Job, JobId, JobState, Scheduler, step};

/// What a body reports after one quantum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ran {
    /// The quantum ran. The job has no new quantum to add.
    Done,
    /// The quantum ran, and the job has these quanta to add after the ones it has.
    More(Vec<Kind>),
    /// The quantum failed. The job ends, and the body has told its caller why.
    Failed,
}

/// The work of one job.
pub trait JobBody: Send {
    /// Run the quantum of the given kind. `engines` are the engines of the group, in profile order.
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran;
}

/// A job that a request brings to a group.
pub struct JobSpec {
    /// The class of the job (spec 5.3).
    pub class: Class,
    /// The integer priority (spec 5.4).
    pub priority: i64,
    /// The kind of the first quantum. The body adds the others when it runs the first one.
    pub first: Kind,
    /// The work.
    pub body: Box<dyn JobBody>,
}

/// The state of the jobs of one group.
pub struct GroupJobs {
    group: u32,
    epoch: Instant,
    state: Scheduler,
    bodies: BTreeMap<JobId, Box<dyn JobBody>>,
    /// The jobs that arrived since the last quantum. Their inputs become present when the next quantum is chosen.
    fresh: Vec<JobId>,
    /// The quantum that ran last, until the scheduler learns that it ended. The end event chooses the next quantum.
    finished: Option<(u64, Completion)>,
    next_id: JobId,
}

impl GroupJobs {
    /// Make the jobs of a group. The scheduler serves one device group (spec 5.5).
    pub fn new(group: u32) -> Self {
        Self {
            group,
            epoch: Instant::now(),
            state: Scheduler::default(),
            bodies: BTreeMap::new(),
            fresh: Vec::new(),
            finished: None,
            next_id: 1,
        }
    }

    /// True when no job waits or runs.
    pub fn idle(&self) -> bool {
        self.bodies.is_empty() && self.finished.is_none()
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn event(&mut self, event: Event) {
        let state = std::mem::take(&mut self.state);
        // A wrong event is a defect of this driver, and the state is gone (spec 8.5).
        let (state, chosen) = step(state, event).expect("the group driver sent an event that does not fit");
        assert!(
            chosen.is_none() || state.running == chosen,
            "the scheduler chose a quantum that it does not hold as running"
        );
        self.state = state;
    }

    /// Add a job. It waits for the next call of [`GroupJobs::run_one`].
    pub fn add(&mut self, spec: JobSpec) -> JobId {
        let id = self.next_id;
        self.next_id += 1;
        let now_ms = self.now_ms();
        let job = Job {
            id,
            class: spec.class,
            priority: spec.priority,
            arrival_ms: now_ms,
            deadline_ms: None,
            engine: self.group,
            quanta: [spec.first].into_iter().collect(),
            admitted: false,
            inputs_present: false,
            state: JobState::Ready,
            balance_ms: 0,
            cancel_requested: false,
            checkpoint_interval_ms: None,
        };
        self.event(Event::Arrive { now_ms, job });
        // The ledger admitted the context before the job came, and a pass never allocates (spec 4.3).
        self.event(Event::Admit { now_ms, job: id });
        self.bodies.insert(id, spec.body);
        self.fresh.push(id);
        id
    }

    /// Choose and run one quantum. It gives false when no quantum was ready.
    ///
    /// The choice happens here, after every arrival so far, so a job that arrived while the last quantum ran can go
    /// first (spec 5.3). The end of the last quantum reaches the scheduler here too.
    pub fn run_one(&mut self, engines: &mut [Box<dyn Engine>]) -> bool {
        for id in std::mem::take(&mut self.fresh) {
            if let Some(job) = self.state.jobs.iter_mut().find(|job| job.id == id) {
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
        let Some(quantum) = self.state.running.clone() else {
            return false;
        };
        assert!(
            quantum.segments.len() == 1,
            "the group driver runs one segment in a pass; the scheduler joined {}",
            quantum.segments.len()
        );
        let id = quantum.segments[0];
        let body = self.bodies.get_mut(&id).expect("a chosen job has a body");
        let started = Instant::now();
        let ran = body.run(quantum.kind, engines);
        let duration_ms = started.elapsed().as_millis() as u64;
        let outcome = match ran {
            Ran::Done => Completion::Done,
            Ran::More(kinds) => {
                self.state
                    .append_quanta(id, &kinds)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            Ran::Failed => Completion::Failed,
        };
        self.finished = Some((duration_ms, outcome));
        true
    }

    /// Drop the jobs that reached a terminal state, and their bodies. A job that leaves takes its balance (spec 5.4).
    fn forget_ended_jobs(&mut self) {
        let ended: Vec<JobId> = self
            .state
            .jobs
            .iter()
            .filter(|job| job.state.is_terminal())
            .map(|job| job.id)
            .collect();
        self.state.jobs.retain(|job| !job.state.is_terminal());
        for id in ended {
            self.bodies.remove(&id);
        }
    }

    /// Cancel every job that has not ended (spec 10.3, 5.9). It gives the number of jobs that it dropped.
    ///
    /// A body that drops its result sender tells its caller that the job ended with no result.
    pub fn cancel_all(&mut self) -> usize {
        let count = self.bodies.len();
        self.bodies.clear();
        self.fresh.clear();
        self.finished = None;
        self.state = Scheduler::default();
        count
    }
}
