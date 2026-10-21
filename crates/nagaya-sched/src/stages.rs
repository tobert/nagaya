//! The scheduler of a device group whose engine declares stages (engine-abi D1, spec 5.1).
//!
//! It is a state machine like [`crate::step`]. [`stage_step`] takes the state and one [`StageEvent`]. It gives the
//! new state and the quanta that start, at most one for each device. A job holds its slot with a lease from its first
//! quantum to its end (spec 5.6). The lease lives in the job's body, so no quantum of another job writes the slot
//! (D1 rule 3). The scheduler orders the quanta of each job:
//!
//! 1. Stage s + 1 of a block is ready only after stage s of that block ends and writes the residual.
//! 2. Stage s of block k + 1 is ready only after stage s of block k ends.
//! 3. A restore is a copy quantum on the device of each stage. It comes before the block after it, by rule 2.
//!    **Proposed** (claude-opus-5-5, engine-abi D1).
//!
//! The rules of "The stage path in v0" (engine-abi D1, Proposed) also hold. A block holds a residual buffer from the
//! start of stage s to the end of stage s + 1, and stage s waits for a free buffer. A copy quantum that is not a
//! restore holds every device. Between quanta on each device, the ready quantum of the highest rank for that device
//! starts (spec 5.3, 5.4): the class first, then the order in the class.

use std::collections::{BTreeMap, VecDeque};

use nagaya_abi::Kind;

use crate::{Class, Completion, EngineId, JobId, JobState, charges, clamp_balance, order_in_class};

/// One device of a group: an index into the device group (spec 2.4).
pub type Device = u32;

/// The scheduling facts of an engine with stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedEngine {
    /// The device group.
    pub id: EngineId,
    /// The device of each stage, in stage order (`Limits::stages`).
    pub devices: Vec<Device>,
    /// The residual buffers at each boundary between two stages (engine-abi D1: two).
    pub buffers: u32,
    /// The longest that one stage quantum may run, in milliseconds (`Limits::max_quantum_ms`, D8).
    pub max_quantum_ms: u64,
}

/// One unit of the work of a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    /// A pass: one stage quantum on the device of each stage, in stage order.
    Block(Kind),
    /// A restore: one copy quantum on the device of each stage.
    Restore,
    /// A copy quantum that reads or writes the parts of every stage. It holds every device of the group.
    Whole(Kind),
}

/// What one quantum runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// One stage of a block.
    Stage {
        /// The serial number of the block in its job.
        block: u64,
        /// The stage.
        stage: usize,
    },
    /// The copy of one stage's parts of a restore.
    Restore {
        /// The serial number of the restore in its job.
        block: u64,
        /// The stage.
        stage: usize,
    },
    /// A copy quantum that holds every device.
    Whole,
}

/// One quantum that starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageQuantum {
    /// The job.
    pub job: JobId,
    /// The class of the job.
    pub class: Class,
    /// The kind of the quantum (spec 5.2). A restore is a copy.
    pub kind: Kind,
    /// What the quantum runs.
    pub work: Work,
    /// The devices that the quantum holds.
    pub devices: Vec<Device>,
}

/// One job of a group with stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageJob {
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
    /// The items that have not started.
    pub items: VecDeque<Item>,
    /// True when the ledger admitted the job's calls.
    pub admitted: bool,
    /// True when the inputs of the job are present.
    pub inputs_present: bool,
    /// The job state. `Running` while any quantum of the job runs.
    pub state: JobState,
    /// The owed balance in milliseconds.
    pub balance_ms: i64,
    /// True when a client cancelled the job while a quantum of it ran.
    pub cancel_requested: bool,
}

impl StageJob {
    /// A job that is ready, with its first items.
    pub fn new(id: JobId, class: Class, arrival_ms: u64, items: impl IntoIterator<Item = Item>) -> Self {
        Self {
            id,
            class,
            priority: 0,
            arrival_ms,
            deadline_ms: None,
            items: items.into_iter().collect(),
            admitted: true,
            inputs_present: true,
            state: JobState::Ready,
            balance_ms: 0,
            cancel_requested: false,
        }
    }
}

/// A block or a restore that started and has a stage that has not ended.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Flight {
    block: u64,
    kind: Kind,
    restore: bool,
    running: Vec<bool>,
    done: Vec<bool>,
}

/// A quantum that runs, and when it started.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Running {
    quantum: StageQuantum,
    started_ms: u64,
}

/// The state of the scheduler of one group with stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageScheduler {
    engine: StagedEngine,
    jobs: Vec<StageJob>,
    flights: BTreeMap<JobId, VecDeque<Flight>>,
    running: BTreeMap<Device, Running>,
    next_block: u64,
}

/// One event for the scheduler of a group with stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageEvent {
    /// A job arrives.
    Arrive {
        /// The time in milliseconds.
        now_ms: u64,
        /// The job.
        job: StageJob,
    },
    /// The ledger admits a job's calls.
    Admit {
        /// The time in milliseconds.
        now_ms: u64,
        /// The job.
        job: JobId,
    },
    /// The quantum that runs on a device ends. A quantum that holds every device ends once, on any of them.
    Complete {
        /// The time in milliseconds.
        now_ms: u64,
        /// The device.
        device: Device,
        /// The run time in milliseconds.
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

impl StageEvent {
    fn now(&self) -> u64 {
        match self {
            StageEvent::Arrive { now_ms, .. }
            | StageEvent::Admit { now_ms, .. }
            | StageEvent::Complete { now_ms, .. }
            | StageEvent::Cancel { now_ms, .. }
            | StageEvent::Tick { now_ms } => *now_ms,
        }
    }
}

/// An event that does not fit the state. The caller reports it and stops (spec 8.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageError {
    /// An `Arrive` event names a job id that the scheduler already holds.
    DuplicateJob(JobId),
    /// An `Arrive` event brings a job that is not `Ready`.
    ArrivalNotReady(JobId, JobState),
    /// An event names a job that the scheduler does not hold.
    UnknownJob(JobId),
    /// A `Complete` event names a device on which no quantum runs.
    NothingRunning(Device),
    /// A caller adds items to a job that has no quantum that runs.
    AppendToIdleJob(JobId, JobState),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageError::DuplicateJob(id) => write!(f, "job {id} arrived twice: the job id must be new"),
            StageError::ArrivalNotReady(id, state) => {
                write!(f, "job {id} arrived in state {state:?}: an arriving job must be Ready")
            }
            StageError::UnknownJob(id) => write!(f, "job {id} is not in the scheduler: send Arrive first"),
            StageError::NothingRunning(device) => {
                write!(
                    f,
                    "a quantum ended on device {device}, but none runs there: send Complete only once"
                )
            }
            StageError::AppendToIdleJob(id, state) => write!(
                f,
                "job {id} is in state {state:?}: add items only while a quantum of the job runs"
            ),
        }
    }
}

impl std::error::Error for StageError {}

/// One candidate for a device: the job, what it would run, and whether it starts a new item.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    job: usize,
    kind: Kind,
    work: Work,
    /// True when the quantum starts the front item of the job.
    starts_item: bool,
}

impl StageScheduler {
    /// Make the scheduler of one engine with stages. The engine has at least one stage, and each stage has its own
    /// device (`Limits::check`).
    pub fn new(engine: StagedEngine) -> Self {
        assert!(!engine.devices.is_empty(), "an engine with stages has a stage");
        assert!(engine.buffers > 0, "a boundary has at least one residual buffer");
        for (index, device) in engine.devices.iter().enumerate() {
            assert!(
                !engine.devices[..index].contains(device),
                "two stages run on device {device}; each stage has its own device"
            );
        }
        Self {
            engine,
            jobs: Vec::new(),
            flights: BTreeMap::new(),
            running: BTreeMap::new(),
            next_block: 0,
        }
    }

    /// The engine facts.
    pub fn engine(&self) -> &StagedEngine {
        &self.engine
    }

    /// The jobs.
    pub fn jobs(&self) -> &[StageJob] {
        &self.jobs
    }

    /// The jobs, to change between two steps. The caller marks inputs present and drops the jobs that ended. It
    /// changes no job state, because the next step reads it.
    pub fn jobs_mut(&mut self) -> &mut Vec<StageJob> {
        &mut self.jobs
    }

    /// The quantum that runs on a device.
    pub fn running_on(&self, device: Device) -> Option<&StageQuantum> {
        self.running.get(&device).map(|running| &running.quantum)
    }

    /// True when no quantum runs on any device.
    pub fn idle(&self) -> bool {
        self.running.is_empty()
    }

    /// The quanta that run longer than `max_quantum_ms` at `now_ms`, one for each device (D8, spec 7.4). Each stage
    /// quantum has the bound of its own.
    pub fn past_bound(&self, now_ms: u64) -> Vec<(Device, &StageQuantum)> {
        self.running
            .iter()
            .filter(|(_, running)| now_ms.saturating_sub(running.started_ms) > self.engine.max_quantum_ms)
            .map(|(device, running)| (*device, &running.quantum))
            .collect()
    }

    /// Add items to the end of a job, while a quantum of it runs. A refusal changes nothing.
    pub fn append_items(&mut self, job: JobId, items: &[Item]) -> Result<(), StageError> {
        let held = self
            .jobs
            .iter_mut()
            .find(|candidate| candidate.id == job)
            .ok_or(StageError::UnknownJob(job))?;
        if held.state != JobState::Running {
            return Err(StageError::AppendToIdleJob(job, held.state));
        }
        held.items.extend(items.iter().copied());
        Ok(())
    }

    fn stages(&self) -> usize {
        self.engine.devices.len()
    }

    fn job_runs(&self, job: JobId) -> bool {
        self.running.values().any(|running| running.quantum.job == job)
    }

    /// The residual buffers that the blocks hold at the boundary after stage `stage`.
    fn buffers_held(&self, stage: usize) -> u32 {
        self.flights
            .values()
            .flatten()
            .filter(|flight| {
                !flight.restore && (flight.running[stage] || flight.done[stage]) && !flight.done[stage + 1]
            })
            .count() as u32
    }

    /// True when stage `stage` of a block may start now, by its buffer.
    fn buffer_free(&self, stage: usize) -> bool {
        stage + 1 >= self.stages() || self.buffers_held(stage) < self.engine.buffers
    }

    /// True when every earlier block or restore of the job ended stage `stage` (rule 2).
    fn earlier_done(flights: &VecDeque<Flight>, before: usize, stage: usize) -> bool {
        flights.iter().take(before).all(|flight| flight.done[stage])
    }

    /// The candidates of one job for one device. A whole quantum is a candidate on every device.
    fn candidates(&self, index: usize, device: Device, out: &mut Vec<Candidate>) {
        let job = &self.jobs[index];
        let empty = VecDeque::new();
        let flights = self.flights.get(&job.id).unwrap_or(&empty);
        for (position, flight) in flights.iter().enumerate() {
            for stage in 0..self.stages() {
                if self.engine.devices[stage] != device || flight.done[stage] || flight.running[stage] {
                    continue;
                }
                let order = flight.restore || stage == 0 || flight.done[stage - 1];
                if order && Self::earlier_done(flights, position, stage) && (flight.restore || self.buffer_free(stage))
                {
                    let work = if flight.restore {
                        Work::Restore {
                            block: flight.block,
                            stage,
                        }
                    } else {
                        Work::Stage {
                            block: flight.block,
                            stage,
                        }
                    };
                    out.push(Candidate {
                        job: index,
                        kind: flight.kind,
                        work,
                        starts_item: false,
                    });
                }
            }
        }
        let Some(item) = job.items.front() else {
            return;
        };
        match *item {
            Item::Block(kind) => {
                let stage = 0;
                if self.engine.devices[stage] == device
                    && Self::earlier_done(flights, flights.len(), stage)
                    && self.buffer_free(stage)
                {
                    out.push(Candidate {
                        job: index,
                        kind,
                        work: Work::Stage {
                            block: self.next_block,
                            stage,
                        },
                        starts_item: true,
                    });
                }
            }
            Item::Restore => {
                if let Some(stage) = self.engine.devices.iter().position(|candidate| *candidate == device)
                    && Self::earlier_done(flights, flights.len(), stage)
                {
                    out.push(Candidate {
                        job: index,
                        kind: Kind::Copy,
                        work: Work::Restore {
                            block: self.next_block,
                            stage,
                        },
                        starts_item: true,
                    });
                }
            }
            Item::Whole(kind) => {
                if flights.is_empty() && !self.job_runs(job.id) {
                    out.push(Candidate {
                        job: index,
                        kind,
                        work: Work::Whole,
                        starts_item: true,
                    });
                }
            }
        }
    }

    /// The order of two candidates: the class, the order in the class, then the order of the work in the job.
    fn before(&self, left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
        let (a, b) = (&self.jobs[left.job], &self.jobs[right.job]);
        let rank = |class: Class| match class {
            Class::Interactive => 0,
            Class::Batch => 1,
            Class::Filler => 2,
        };
        rank(a.class)
            .cmp(&rank(b.class))
            .then_with(|| {
                order_in_class(
                    a.class,
                    (a.priority, a.balance_ms, a.arrival_ms),
                    (b.priority, b.balance_ms, b.arrival_ms),
                )
            })
            .then(a.id.cmp(&b.id))
            .then(left.starts_item.cmp(&right.starts_item))
    }

    fn eligible(job: &StageJob) -> bool {
        !job.state.is_terminal() && job.admitted && job.inputs_present && !job.cancel_requested
    }

    /// Start the best ready quantum on each free device, in device order.
    fn choose(&mut self, now_ms: u64) -> Vec<StageQuantum> {
        let mut started = Vec::new();
        let mut devices: Vec<Device> = self.engine.devices.clone();
        devices.sort_unstable();
        for device in devices {
            if self.running.contains_key(&device) {
                continue;
            }
            let mut candidates = Vec::new();
            for index in 0..self.jobs.len() {
                if Self::eligible(&self.jobs[index]) {
                    self.candidates(index, device, &mut candidates);
                }
            }
            let Some(best) = candidates.into_iter().min_by(|left, right| self.before(left, right)) else {
                continue;
            };
            if best.work == Work::Whole {
                // A whole quantum waits for every device. A free device starts no quantum of a lower rank before it.
                if self.running.is_empty() {
                    started.push(self.start(best, now_ms));
                    break;
                }
                continue;
            }
            started.push(self.start(best, now_ms));
        }
        started
    }

    fn start(&mut self, candidate: Candidate, now_ms: u64) -> StageQuantum {
        let stages = self.stages();
        let job = &mut self.jobs[candidate.job];
        if candidate.starts_item {
            let item = job.items.pop_front().expect("a candidate that starts an item has one");
            let (kind, restore) = match item {
                Item::Block(kind) => (kind, false),
                Item::Restore => (Kind::Copy, true),
                Item::Whole(_) => (candidate.kind, false),
            };
            if !matches!(item, Item::Whole(_)) {
                self.flights.entry(job.id).or_default().push_back(Flight {
                    block: self.next_block,
                    kind,
                    restore,
                    running: vec![false; stages],
                    done: vec![false; stages],
                });
                self.next_block += 1;
            }
        }
        job.state = JobState::Running;
        let devices = match candidate.work {
            Work::Whole => {
                let mut devices = self.engine.devices.clone();
                devices.sort_unstable();
                devices
            }
            Work::Stage { block, stage } | Work::Restore { block, stage } => {
                let flight = self
                    .flights
                    .get_mut(&job.id)
                    .and_then(|flights| flights.iter_mut().find(|flight| flight.block == block))
                    .expect("a stage quantum has its block in flight");
                flight.running[stage] = true;
                vec![self.engine.devices[stage]]
            }
        };
        let quantum = StageQuantum {
            job: job.id,
            class: job.class,
            kind: candidate.kind,
            work: candidate.work,
            devices: devices.clone(),
        };
        for device in devices {
            self.running.insert(
                device,
                Running {
                    quantum: quantum.clone(),
                    started_ms: now_ms,
                },
            );
        }
        quantum
    }

    fn complete(&mut self, device: Device, duration_ms: u64, outcome: Completion) -> Result<(), StageError> {
        let running = self.running.remove(&device).ok_or(StageError::NothingRunning(device))?;
        let quantum = running.quantum;
        for other in &quantum.devices {
            self.running.remove(other);
        }
        if quantum.class == Class::Interactive {
            let present: Vec<(JobId, u64)> = self
                .jobs
                .iter()
                .filter(|job| job.class == Class::Interactive && !job.state.is_terminal())
                .map(|job| (job.id, job.arrival_ms))
                .collect();
            let arrival = self
                .jobs
                .iter()
                .find(|job| job.id == quantum.job)
                .map_or(u64::MAX, |job| job.arrival_ms);
            for (id, delta) in charges(&[(quantum.job, arrival)], duration_ms, &present) {
                if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
                    job.balance_ms = clamp_balance(job.balance_ms, delta);
                }
            }
        }
        if let Work::Stage { block, stage } | Work::Restore { block, stage } = quantum.work
            && let Some(flights) = self.flights.get_mut(&quantum.job)
        {
            if let Some(flight) = flights.iter_mut().find(|flight| flight.block == block) {
                flight.running[stage] = false;
                flight.done[stage] = true;
            }
            while flights
                .front()
                .is_some_and(|flight| flight.done.iter().all(|done| *done))
            {
                flights.pop_front();
            }
        }
        let runs = self.job_runs(quantum.job);
        let Some(job) = self.jobs.iter_mut().find(|job| job.id == quantum.job) else {
            return Ok(());
        };
        if job.state.is_terminal() {
            return Ok(());
        }
        if outcome == Completion::Failed {
            job.state = JobState::Failed;
            job.items.clear();
            self.flights.remove(&job.id);
            return Ok(());
        }
        if runs {
            return Ok(());
        }
        let in_flight = self.flights.get(&job.id).is_some_and(|flights| !flights.is_empty());
        job.state = if job.cancel_requested {
            job.items.clear();
            self.flights.remove(&job.id);
            JobState::Cancelled
        } else if job.items.is_empty() && !in_flight {
            JobState::Done
        } else {
            JobState::Ready
        };
        Ok(())
    }

    fn expire(&mut self, now_ms: u64) {
        for index in 0..self.jobs.len() {
            let id = self.jobs[index].id;
            let runs = self.job_runs(id);
            let job = &mut self.jobs[index];
            if job.state == JobState::Ready && !runs && job.deadline_ms.is_some_and(|deadline| deadline <= now_ms) {
                job.state = JobState::Expired;
                job.items.clear();
                self.flights.remove(&id);
            }
        }
    }
}

/// Take one step: one event in, the new state and the quanta that start out.
///
/// An event that does not fit the state returns a [`StageError`], and the state is gone (spec 8.5).
pub fn stage_step(
    mut state: StageScheduler,
    event: StageEvent,
) -> Result<(StageScheduler, Vec<StageQuantum>), StageError> {
    let now = event.now();
    match event {
        StageEvent::Arrive { job, .. } => {
            if state.jobs.iter().any(|held| held.id == job.id) {
                return Err(StageError::DuplicateJob(job.id));
            }
            if job.state != JobState::Ready {
                return Err(StageError::ArrivalNotReady(job.id, job.state));
            }
            state.jobs.push(job);
        }
        StageEvent::Admit { job, .. } => {
            state
                .jobs
                .iter_mut()
                .find(|candidate| candidate.id == job)
                .ok_or(StageError::UnknownJob(job))?
                .admitted = true;
        }
        StageEvent::Complete {
            device,
            duration_ms,
            outcome,
            ..
        } => state.complete(device, duration_ms, outcome)?,
        StageEvent::Cancel { job, .. } => {
            let runs = state.job_runs(job);
            let held = state
                .jobs
                .iter_mut()
                .find(|candidate| candidate.id == job)
                .ok_or(StageError::UnknownJob(job))?;
            if runs {
                held.cancel_requested = true;
            } else if !held.state.is_terminal() {
                held.state = JobState::Cancelled;
                held.items.clear();
                state.flights.remove(&job);
            }
        }
        StageEvent::Tick { .. } => {}
    }
    state.expire(now);
    let started = state.choose(now);
    Ok((state, started))
}
