//! A device group whose engine declares stages (engine-abi D1, spec 3.2, 7.4).
//!
//! The group thread runs the one scheduler of the group ([`StageScheduler`]). It has one stage thread for each stage,
//! and each stage thread holds the [`StageHandle`] of its device. The group thread sends each stage quantum to the
//! thread of its device and gets the end back on its own channel, so the two devices run at the same time.
//!
//! A copy quantum holds every device, and it runs on the group thread through [`StagedEngine`]. That type is the
//! `Engine` of the job bodies. It sends each part call to the stage thread of the part and waits for the answer. A
//! stage thread runs one call at a time, in order, so a call waits for the stage quantum before it.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nagaya_abi::{
    Capability, Coverage, CoverageSpan, DeviceId, Engine, Identity, Kind, Limits, PartError, PartId, Parts, PassError,
    PassErrorKind, PassOut, Residual, Segment, SlotId, Slots, StageHandle, StageOut, State, Tap, WriteResult,
};
use nagaya_sched::{
    Completion, Device, Item, JobId, JobState, StageEvent, StageJob, StageQuantum, StageScheduler,
    StagedEngine as Facts, Work, stage_step,
};
use nagaya_telemetry::{QuantumLabels, instruments};
use tracing::Span;

use super::{Event, GroupMessage, OpenedEngine, ParkProgress, WatchMessage, guard, park_group, pin, spawn_named};
use crate::jobs::{JobBody, JobSpec, Ran};
use crate::park::StoreMessage;

/// The links of a group with stages to the rest of the process.
pub(super) struct StagedLinks {
    /// The machine devices of the engine, by group index (spec 2.4).
    pub(super) devices: Vec<DeviceId>,
    pub(super) events: Sender<Event>,
    pub(super) heartbeats: Sender<WatchMessage>,
    pub(super) messages: Receiver<GroupMessage>,
    pub(super) own: Sender<GroupMessage>,
    pub(super) store: Option<Sender<StoreMessage>>,
    pub(super) accepting: Arc<AtomicBool>,
    pub(super) cpus: Option<Vec<u32>>,
}

/// A call that a stage thread makes on its handle for the group thread.
type Call = Box<dyn FnOnce(&mut dyn StageHandle) + Send>;

/// The work of a stage thread.
enum StageWork {
    /// A part call of a copy quantum. The closure sends its own answer.
    Call(Call),
    /// One stage quantum of a block.
    Pass {
        job: JobId,
        block: u64,
        stage: usize,
        segments: Arc<Vec<Segment>>,
        residual: Residual,
    },
}

/// The end of one stage quantum, from its stage thread.
pub(crate) struct StageDone {
    job: JobId,
    block: u64,
    stage: usize,
    result: Result<StageOut, PassError>,
    residual: Residual,
    ms: u64,
}

/// The stage threads of a group, one for each stage, in stage order.
struct StageThreads {
    senders: Vec<Sender<StageWork>>,
    handles: Vec<JoinHandle<()>>,
}

impl StageThreads {
    fn start(
        group: usize,
        handles: Vec<Box<dyn StageHandle + Send>>,
        devices: &[DeviceId],
        links: &StagedLinks,
        interval: Duration,
    ) -> Self {
        let mut senders = Vec::with_capacity(handles.len());
        let mut threads = Vec::with_capacity(handles.len());
        for (stage, handle) in handles.into_iter().enumerate() {
            let (work_tx, work_rx) = mpsc::channel();
            let device = *links.devices.get(devices[stage] as usize).unwrap_or_else(|| {
                panic!(
                    "stage {stage} names group device {}, which the group lacks",
                    devices[stage]
                )
            });
            let events = links.events.clone();
            let done = links.own.clone();
            let heartbeats = links.heartbeats.clone();
            let cpus = links.cpus.clone();
            let _ = heartbeats.send(WatchMessage::StartStage {
                group,
                stage,
                devices: vec![device],
            });
            let thread = spawn_named(format!("nagaya-stg-{group}-{stage}"), move || {
                guard("stage", &events, || {
                    // Spec 3.2b gives each card the CPUs near it. That split is not built, so a stage thread takes the
                    // pool set of its group thread (spec 3.2a).
                    pin("stage", cpus.as_deref());
                    stage_thread(group, stage, handle, work_rx, done, heartbeats, interval);
                });
            });
            senders.push(work_tx);
            threads.push(thread);
        }
        Self {
            senders,
            handles: threads,
        }
    }
}

/// The loop of one stage thread. It beats its own heartbeat between its quanta (spec 7.4).
fn stage_thread(
    group: usize,
    stage: usize,
    mut handle: Box<dyn StageHandle + Send>,
    work: Receiver<StageWork>,
    done: Sender<GroupMessage>,
    heartbeats: Sender<WatchMessage>,
    interval: Duration,
) {
    let beat = || WatchMessage::BeatStage {
        group,
        stage,
        at: Instant::now(),
    };
    let _ = heartbeats.send(beat());
    loop {
        match work.recv_timeout(interval) {
            Ok(StageWork::Call(call)) => call(handle.as_mut()),
            Ok(StageWork::Pass {
                job,
                block,
                stage: named,
                segments,
                mut residual,
            }) => {
                assert_eq!(
                    named, stage,
                    "the group sent the quantum of stage {named} to stage {stage}"
                );
                let started = Instant::now();
                let result = handle.stage_pass(&segments, &mut residual);
                let ms = started.elapsed().as_millis() as u64;
                let end = StageDone {
                    job,
                    block,
                    stage,
                    result,
                    residual,
                    ms,
                };
                if done.send(GroupMessage::StageDone(end)).is_err() {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            // The group thread dropped the stage: the group stops.
            Err(RecvTimeoutError::Disconnected) => return,
        }
        if heartbeats.send(beat()).is_err() {
            return;
        }
    }
}

/// The engine of a group with stages, as the job bodies and the park see it. Each call that touches a device goes to
/// the stage thread of that device. `identity`, `limits` and `last_error` stay on the group thread (engine-abi D1).
pub(crate) struct StagedEngine {
    inner: Box<dyn Engine>,
    limits: Limits,
    senders: Vec<Sender<StageWork>>,
}

impl StagedEngine {
    /// Run `call` on the thread of `stage`, and wait for its answer. A stage thread that stopped is a crash (spec 8.5).
    fn call<R: Send + 'static>(
        &self,
        stage: usize,
        call: impl FnOnce(&mut dyn StageHandle) -> R + Send + 'static,
    ) -> R {
        let (answer, answered) = mpsc::sync_channel(1);
        self.senders[stage]
            .send(StageWork::Call(Box::new(move |handle| {
                let _ = answer.send(call(handle));
            })))
            .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
        answered
            .recv()
            .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped during a call"))
    }

    fn stage_of(&self, part: PartId) -> Option<usize> {
        self.limits.parts.get(part).map(|part| part.stage as usize)
    }
}

impl Engine for StagedEngine {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// A pass of an engine with stages runs stage by stage on the stage threads. This call refuses it.
    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(PassError {
            kind: PassErrorKind::Refused,
            message: "a pass of an engine with stages runs through the stage threads, as a block of the scheduler"
                .to_owned(),
        })
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.limits
            .capabilities
            .contains(&Capability::Slots)
            .then_some(self as &mut dyn Slots)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.limits
            .capabilities
            .contains(&Capability::Coverage)
            .then_some(self as &mut dyn Coverage)
    }
}

impl Parts for StagedEngine {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let stage = self.stage_of(part).ok_or(PartError::Refused)?;
        let len = out.len();
        let (result, bytes) = self.call(stage, move |handle| {
            let mut bytes = vec![0u8; len];
            let result = match handle.parts() {
                Some(parts) => parts.part_read(state, part, &mut bytes),
                None => Err(PartError::Refused),
            };
            (result, bytes)
        });
        out.copy_from_slice(&bytes);
        result
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        let Some(stage) = self.stage_of(part) else {
            return WriteResult::Refused;
        };
        let bytes = bytes.to_vec();
        self.call(stage, move |handle| match handle.parts() {
            Some(parts) => parts.part_write(state, part, &bytes),
            None => WriteResult::Refused,
        })
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let stage = self.stage_of(part).ok_or(PartError::Refused)?;
        let len = out.len();
        let (result, bytes) = self.call(stage, move |handle| {
            let mut bytes = vec![0u8; len];
            let result = match handle.parts() {
                Some(parts) => parts.cells_read(state, part, from, n, &mut bytes),
                None => Err(PartError::Refused),
            };
            (result, bytes)
        });
        out.copy_from_slice(&bytes);
        result
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let Some(stage) = self.stage_of(part) else {
            return WriteResult::Refused;
        };
        let bytes = bytes.to_vec();
        self.call(stage, move |handle| match handle.parts() {
            Some(parts) => parts.cells_write(state, part, from, n, &bytes),
            None => WriteResult::Refused,
        })
    }

    /// Set the rows of every stage, in stage order. A refusal after a stage that took the count is a failure, because
    /// the state changed (spec 6.8).
    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        for stage in 0..self.senders.len() {
            let result = self.call(stage, move |handle| match handle.parts() {
                Some(parts) => parts.set_n(state, n),
                None => WriteResult::Refused,
            });
            match result {
                WriteResult::Ok => {}
                WriteResult::Refused if stage == 0 => return WriteResult::Refused,
                _ => return WriteResult::Failed,
            }
        }
        WriteResult::Ok
    }
}

impl Slots for StagedEngine {
    /// The highest row of the slot in any stage.
    fn slot_high(&self, slot: SlotId) -> u64 {
        (0..self.senders.len())
            .map(|stage| {
                self.call(stage, move |handle| {
                    handle
                        .slots()
                        .unwrap_or_else(|| panic!("stage {stage} has no slot capability, but the engine declares it"))
                        .slot_high(slot)
                })
            })
            .max()
            .unwrap_or(0)
    }

    /// Clear the slot in every stage, in stage order. A refusal after a stage that cleared is a failure (spec 6.8).
    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        for stage in 0..self.senders.len() {
            let result = self.call(stage, move |handle| match handle.slots() {
                Some(slots) => slots.slot_clear(slot),
                None => WriteResult::Refused,
            });
            match result {
                WriteResult::Ok => {}
                WriteResult::Refused if stage == 0 => return WriteResult::Refused,
                _ => return WriteResult::Failed,
            }
        }
        WriteResult::Ok
    }
}

impl Coverage for StagedEngine {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        let stage = self
            .stage_of(part)
            .unwrap_or_else(|| panic!("the store asked for the coverage of part {part}, which the engine lacks"));
        let bytes = bytes.to_vec();
        self.call(stage, move |handle| {
            handle
                .coverage()
                .unwrap_or_else(|| panic!("stage {stage} has no coverage capability, but the engine declares it"))
                .coverage(state, part, &bytes)
        })
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        let stage = self.stage_of(coverage.part).ok_or("unknown part")?;
        let coverage = *coverage;
        self.call(stage, move |handle| match handle.coverage() {
            Some(fit) => fit.fits(&coverage, slot),
            None => Err(format!("stage {stage} has no coverage capability")),
        })
    }
}

/// A block that began: its kind, its segments, and its residual between two stages.
struct Block {
    kind: Kind,
    segments: Arc<Vec<Segment>>,
    residual: Option<Residual>,
}

/// The jobs of a group with stages, on its scheduler.
struct StagedJobs {
    group: u32,
    engine: String,
    epoch: Instant,
    scheduler: Option<StageScheduler>,
    bodies: BTreeMap<JobId, Box<dyn JobBody>>,
    spans: BTreeMap<JobId, Span>,
    blocks: BTreeMap<(JobId, u64), Block>,
    /// The blocks of each job in the order they began. Their passes end in this order.
    order: BTreeMap<JobId, VecDeque<u64>>,
    /// The residual buffers that no block holds: the term `residual` (engine-abi D1).
    free: Vec<Residual>,
    /// The quanta that started and that the group has not sent to a device yet.
    started: VecDeque<StageQuantum>,
    stages: usize,
    next_id: JobId,
}

impl StagedJobs {
    fn new(group: u32, engine: String, limits: &Limits) -> Self {
        let devices: Vec<Device> = limits.stages.iter().map(|stage| stage.device).collect();
        let buffers = 2;
        let rows = limits.residual_rows() as usize;
        let row_bytes = limits.residual_row_bytes as usize;
        Self {
            group,
            engine,
            epoch: Instant::now(),
            scheduler: Some(StageScheduler::new(Facts {
                id: group,
                devices,
                buffers,
                max_quantum_ms: u64::from(limits.max_quantum_ms),
            })),
            bodies: BTreeMap::new(),
            spans: BTreeMap::new(),
            blocks: BTreeMap::new(),
            order: BTreeMap::new(),
            free: (0..buffers).map(|_| Residual::new(row_bytes, rows)).collect(),
            started: VecDeque::new(),
            stages: limits.stages.len(),
            next_id: 1,
        }
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn scheduler(&self) -> &StageScheduler {
        self.scheduler
            .as_ref()
            .expect("the scheduler is in place between steps")
    }

    fn idle(&self) -> bool {
        self.bodies.is_empty() && self.scheduler().idle()
    }

    /// Take one step. A wrong event is a defect of this driver (spec 8.5).
    fn event(&mut self, event: StageEvent) {
        let state = self.scheduler.take().expect("the scheduler is in place between steps");
        let (state, started) = stage_step(state, event).expect("the group driver sent an event that does not fit");
        self.scheduler = Some(state);
        self.started.extend(started);
    }

    fn add(&mut self, spec: JobSpec, parent: Span, engines: &mut [Box<dyn Engine>], threads: &StageThreads) {
        let id = self.next_id;
        self.next_id += 1;
        let now_ms = self.now_ms();
        let mut job = StageJob::new(id, spec.class, now_ms, [spec.body.item(spec.first)]);
        job.priority = spec.priority;
        job.deadline_ms = spec
            .deadline
            .map(|deadline| deadline.saturating_duration_since(self.epoch).as_millis() as u64);
        let span = tracing::info_span!(
            parent: &parent,
            "nagaya.job",
            engine = self.engine.as_str(),
            class = spec.class.label(),
            request = spec.body.request(),
            outcome = tracing::field::Empty,
        );
        self.spans.insert(id, span);
        self.bodies.insert(id, spec.body);
        self.event(StageEvent::Arrive { now_ms, job });
        self.pump(engines, threads);
    }

    /// Send each started quantum to its device, or run it here when it holds every device.
    fn pump(&mut self, engines: &mut [Box<dyn Engine>], threads: &StageThreads) {
        while let Some(quantum) = self.started.pop_front() {
            match quantum.work {
                Work::Whole => self.run_whole(&quantum, engines),
                Work::Stage { block, stage: 0 } => self.begin_block(&quantum, block, engines, threads),
                Work::Stage { block, stage } => {
                    let job = quantum.job;
                    let entry = self.blocks.get_mut(&(job, block)).expect("a later stage has its block");
                    let residual = entry.residual.take().expect("the stage before wrote the residual");
                    threads.senders[stage]
                        .send(StageWork::Pass {
                            job,
                            block,
                            stage,
                            segments: Arc::clone(&entry.segments),
                            residual,
                        })
                        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
                }
                Work::Restore { .. } => unreachable!("the job path makes no restore quantum; it restores in a copy"),
            }
            self.forget_ended_jobs();
        }
    }

    fn run_whole(&mut self, quantum: &StageQuantum, engines: &mut [Box<dyn Engine>]) {
        let started = Instant::now();
        let ran = match self.bodies.get_mut(&quantum.job) {
            Some(body) => {
                let span = self.spans.get(&quantum.job).cloned().unwrap_or_else(Span::none);
                span.in_scope(|| body.run(quantum.kind, engines))
            }
            None => Ran::Cancelled,
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        self.record(quantum, duration_ms);
        let outcome = self.apply(quantum.job, ran);
        let device = quantum.devices[0];
        self.event(StageEvent::Complete {
            now_ms: self.now_ms(),
            device,
            duration_ms,
            outcome,
        });
    }

    fn begin_block(
        &mut self,
        quantum: &StageQuantum,
        block: u64,
        engines: &mut [Box<dyn Engine>],
        threads: &StageThreads,
    ) {
        let job = quantum.job;
        let begun = match self.bodies.get_mut(&job) {
            Some(body) => body.begin_pass(quantum.kind, engines),
            None => Err(Ran::Cancelled),
        };
        let segments = match begun {
            Ok(segments) => Arc::new(segments),
            Err(ran) => {
                let outcome = self.apply(job, ran);
                self.event(StageEvent::Complete {
                    now_ms: self.now_ms(),
                    device: quantum.devices[0],
                    duration_ms: 0,
                    outcome,
                });
                return;
            }
        };
        // The scheduler starts a block only when a buffer is free, so the pool is never empty here.
        let residual = self.free.pop().expect("a block that starts has a free residual buffer");
        self.blocks.insert(
            (job, block),
            Block {
                kind: quantum.kind,
                segments: Arc::clone(&segments),
                residual: None,
            },
        );
        self.order.entry(job).or_default().push_back(block);
        threads.senders[0]
            .send(StageWork::Pass {
                job,
                block,
                stage: 0,
                segments,
                residual,
            })
            .unwrap_or_else(|_| panic!("the thread of stage 0 stopped"));
    }

    /// The end of one stage quantum.
    fn stage_done(&mut self, done: StageDone, engines: &mut [Box<dyn Engine>], threads: &StageThreads) {
        let StageDone {
            job,
            block,
            stage,
            result,
            residual,
            ms,
        } = done;
        let last = stage + 1 == self.stages;
        let entry_kind = self.blocks.get(&(job, block)).map(|entry| entry.kind);
        let outcome = match (&result, last) {
            (Ok(StageOut::Residual { .. }), false) => {
                let entry = self
                    .blocks
                    .get_mut(&(job, block))
                    .expect("a block in flight has its entry");
                entry.residual = Some(residual);
                Completion::Done
            }
            (Ok(StageOut::Last(_)), true) | (Err(_), _) => {
                self.blocks.remove(&(job, block));
                self.free.push(residual);
                let front = self.order.get(&job).and_then(|order| order.front()).copied();
                if let Some(order) = self.order.get_mut(&job) {
                    order.retain(|serial| *serial != block);
                }
                let result = match result {
                    Ok(StageOut::Last(out)) => Ok(out),
                    Ok(StageOut::Residual { .. }) => unreachable!("the last stage gives the outputs"),
                    Err(error) => Err(error),
                };
                let failed = result.is_err();
                match (self.bodies.get_mut(&job), front == Some(block)) {
                    // The oldest pass of the job ended, so its body hears it. An error of a later block fails the
                    // job, and the body hears of the failure when it ends.
                    (Some(body), true) => {
                        let kind = entry_kind.expect("a block in flight has its entry");
                        let ran = body.end_pass(kind, result, engines);
                        self.apply(job, ran)
                    }
                    _ if failed => Completion::Failed,
                    _ => Completion::Done,
                }
            }
            (Ok(out), _) => panic!(
                "stage {stage} of {} gave {} for block {block}",
                self.stages,
                match out {
                    StageOut::Residual { .. } => "a residual",
                    StageOut::Last(_) => "outputs",
                }
            ),
        };
        if let Some(kind) = entry_kind {
            let labels = QuantumLabels {
                engine: &self.engine,
                device_group: self.group,
                kind: kind_label(kind),
                class: self
                    .scheduler()
                    .jobs()
                    .iter()
                    .find(|candidate| candidate.id == job)
                    .map_or("batch", |candidate| candidate.class.label()),
            };
            instruments().quantum_run(&labels, ms as f64);
        }
        let device = self.scheduler().engine().devices[stage];
        self.event(StageEvent::Complete {
            now_ms: self.now_ms(),
            device,
            duration_ms: ms,
            outcome,
        });
        self.forget_ended_jobs();
        self.pump(engines, threads);
    }

    fn record(&self, quantum: &StageQuantum, duration_ms: u64) {
        let labels = QuantumLabels {
            engine: &self.engine,
            device_group: self.group,
            kind: kind_label(quantum.kind),
            class: quantum.class.label(),
        };
        instruments().quantum_run(&labels, duration_ms as f64);
    }

    /// Apply what a body reported to the scheduler, and give the end of the quantum.
    fn apply(&mut self, job: JobId, ran: Ran) -> Completion {
        match ran {
            Ran::Done => Completion::Done,
            Ran::More(kinds) => {
                let body = self.bodies.get(&job).expect("a body that adds quanta lives");
                let items: Vec<Item> = kinds.iter().map(|kind| body.item(*kind)).collect();
                self.scheduler
                    .as_mut()
                    .expect("the scheduler is in place between steps")
                    .append_items(job, &items)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            Ran::First(_) => panic!("an engine with stages declares no checkpoint grain, so no job makes a copy first"),
            Ran::Failed => Completion::Failed,
            Ran::Cancelled => {
                // The quantum ran to its end. The cancel takes effect when it completes (spec 5.9).
                self.event(StageEvent::Cancel {
                    now_ms: self.now_ms(),
                    job,
                });
                Completion::Done
            }
        }
    }

    fn forget_ended_jobs(&mut self) {
        let ended: Vec<(JobId, JobState)> = self
            .scheduler()
            .jobs()
            .iter()
            .filter(|job| job.state.is_terminal())
            .map(|job| (job.id, job.state))
            .collect();
        if ended.is_empty() {
            return;
        }
        self.scheduler
            .as_mut()
            .expect("the scheduler is in place between steps")
            .jobs_mut()
            .retain(|job| !job.state.is_terminal());
        for (id, state) in ended {
            if let Some(span) = self.spans.remove(&id) {
                span.record("outcome", state.label());
            }
            self.order.remove(&id);
            if let Some(mut body) = self.bodies.remove(&id) {
                body.ended(state);
            }
        }
    }

    /// Cancel every job (spec 10.3, 5.9). The stage quanta that run go to their end. It gives the number of jobs.
    fn cancel_all(&mut self) -> usize {
        let ids: Vec<JobId> = self.bodies.keys().copied().collect();
        self.bodies.clear();
        for id in &ids {
            if self.scheduler().jobs().iter().any(|job| job.id == *id) {
                self.event(StageEvent::Cancel {
                    now_ms: self.now_ms(),
                    job: *id,
                });
            }
        }
        self.started.clear();
        for (_, span) in std::mem::take(&mut self.spans) {
            span.record("outcome", JobState::Cancelled.label());
        }
        let _ = self.scheduler.as_mut().map(|state| {
            state.jobs_mut().retain(|job| !job.state.is_terminal());
        });
        ids.len()
    }
}

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

/// Open the stage threads of the engine, and serve the group until it stops.
pub(super) fn serve(
    group: usize,
    mut engines: Vec<Box<dyn Engine>>,
    opened: Vec<OpenedEngine>,
    links: StagedLinks,
    interval: Duration,
) {
    let refuse = |events: &Sender<Event>, name: &str, message: String| {
        let _ = events.send(Event::OpenFailed {
            group,
            engine: name.to_owned(),
            kind: crate::admission::OpenErrorKind::Other,
            message,
        });
    };
    let name = opened[0].name.clone();
    if engines.len() != 1 {
        refuse(
            &links.events,
            &name,
            "an engine with stages has a device group of its own; the profile shares its devices".to_owned(),
        );
        return;
    }
    let mut inner = engines.pop().expect("one engine");
    let limits = inner.limits();
    if limits.stages.len() != 2 {
        refuse(
            &links.events,
            &name,
            format!(
                "the process runs an engine of two stages; this one has {}. One residual buffer serves a block",
                limits.stages.len()
            ),
        );
        return;
    }
    let handles = match inner.stages() {
        Some(stages) => stages.take_handles(),
        None => Vec::new(),
    };
    if handles.len() != limits.stages.len() {
        refuse(
            &links.events,
            &name,
            format!(
                "the engine declares {} stages and gave {} stage handles",
                limits.stages.len(),
                handles.len()
            ),
        );
        return;
    }
    let devices: Vec<DeviceId> = limits.stages.iter().map(|stage| stage.device).collect();
    let threads = StageThreads::start(group, handles, &devices, &links, interval);
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(StagedEngine {
        inner,
        limits: limits.clone(),
        senders: threads.senders.clone(),
    })];
    let names = vec![name.clone()];
    let _ = links.events.send(Event::Opened { group, engines: opened });
    let mut jobs = StagedJobs::new(group as u32, name, &limits);
    let mut cancelled = 0;
    let beat = || WatchMessage::Beat {
        group,
        at: Instant::now(),
    };
    loop {
        match links.messages.recv_timeout(interval) {
            Ok(GroupMessage::Quantum(quantum)) => {
                if links.accepting.load(Ordering::SeqCst) {
                    quantum.finish(&mut engines);
                } else {
                    quantum.cancel();
                    cancelled += 1;
                }
            }
            Ok(GroupMessage::Job(spec, parent)) => jobs.add(spec, parent, &mut engines, &threads),
            Ok(GroupMessage::StageDone(done)) => jobs.stage_done(done, &mut engines, &threads),
            Ok(GroupMessage::Park(reply)) => {
                cancelled += jobs.cancel_all();
                // A park call waits on each stage thread for the stage quantum before it.
                let _ = reply.send(ParkProgress::Draining { ended: Instant::now() });
                let mut report = park_group(&names, &mut engines, links.store.as_ref());
                report.cancelled = cancelled;
                let _ = reply.send(ParkProgress::Parked(report));
                break;
            }
            Ok(GroupMessage::Stop) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if !links.accepting.load(Ordering::SeqCst) && !jobs.idle() {
            cancelled += jobs.cancel_all();
        }
        if links.heartbeats.send(beat()).is_err() {
            break;
        }
    }
    // The stage threads stop when their senders drop. Each one ends its quantum first.
    drop(engines);
    let StageThreads { senders, handles } = threads;
    drop(senders);
    for handle in handles {
        let _ = handle.join();
    }
}
