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
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nagaya_abi::{
    Capability, Coverage, CoverageSpan, DeviceId, Engine, Identity, Kind, Limits, PartError, PartId, Parts, PassError,
    PassErrorKind, PassOut, Residual, Segment, SlotId, Slots, StageHandle, StageOut, State, Tap, WriteResult,
};
use nagaya_sched::{
    Class, Completion, Device, Item, JobId, JobState, StageEvent, StageJob, StageQuantum, StageScheduler,
    StagedEngine as Facts, Work, stage_step,
};
use nagaya_telemetry::{QuantumLabels, instruments};
use tracing::Span;

use super::in_place::{self, Exit, Member, Plan, Unloaded};
use super::{GroupLinks, GroupMessage, ParkProgress, WatchMessage, guard, park_group, pin, spawn_named};
use crate::evidence::{self, Machine, Watch};
use crate::jobs::{JobBody, JobSpec, Ran};

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
        span: Option<Span>,
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
    /// Start a thread for each stage. `machine` holds the machine devices of the engine by group index, and `devices`
    /// holds the group index of each stage (spec 2.4).
    fn start(
        group: usize,
        handles: Vec<Box<dyn StageHandle + Send>>,
        machine: &[DeviceId],
        devices: &[DeviceId],
        links: &GroupLinks,
        interval: Duration,
    ) -> Self {
        let mut senders = Vec::with_capacity(handles.len());
        let mut threads = Vec::with_capacity(handles.len());
        for (stage, handle) in handles.into_iter().enumerate() {
            let (work_tx, work_rx) = mpsc::channel();
            let device = *machine.get(devices[stage] as usize).unwrap_or_else(|| {
                panic!(
                    "stage {stage} names group device {}, which the group lacks",
                    devices[stage]
                )
            });
            let events = links.events.clone();
            let done = links.own.clone();
            let heartbeats = links.heartbeats.clone();
            let cpus = links.sets.as_ref().map(|sets| sets.device(device).to_vec());
            let _ = heartbeats.send(WatchMessage::StartStage {
                group,
                stage,
                devices: vec![device],
            });
            let thread = spawn_named(format!("nagaya-stg-{group}-{stage}"), move || {
                guard("stage", &events, || {
                    // A stage thread takes the CPUs of its card (spec 3.2b).
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
                span,
            }) => {
                assert_eq!(
                    named, stage,
                    "the group sent the quantum of stage {named} to stage {stage}"
                );
                let started = Instant::now();
                let result = match &span {
                    Some(span) => span.in_scope(|| handle.stage_pass(&segments, &mut residual)),
                    None => handle.stage_pass(&segments, &mut residual),
                };
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

/// A block that began: its kind, its class, its request, its segments, and its residual between two stages.
struct Block {
    kind: Kind,
    class: Class,
    request: &'static str,
    segments: Arc<Vec<Segment>>,
    residual: Option<Residual>,
}

/// A stage quantum that runs on a stage thread, with its telemetry and failure watch.
struct RunningStage {
    kind: Kind,
    class: Class,
    request: &'static str,
    watch: Option<Watch>,
    span: Option<Span>,
}

/// The jobs of a group with stages, on its scheduler.
struct StagedJobs {
    group: u32,
    engine: String,
    machine: Machine,
    epoch: Instant,
    scheduler: Option<StageScheduler>,
    bodies: BTreeMap<JobId, Box<dyn JobBody>>,
    spans: BTreeMap<JobId, Span>,
    requests: BTreeMap<JobId, &'static str>,
    ready_since: BTreeMap<JobId, u64>,
    stage_ready_since: BTreeMap<(JobId, u64, usize), u64>,
    running_stages: BTreeMap<(JobId, u64, usize), RunningStage>,
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
    fn new(group: u32, engine: String, limits: &Limits, machine: Machine) -> Self {
        let devices: Vec<Device> = limits.stages.iter().map(|stage| stage.device).collect();
        let buffers = 2;
        let rows = limits.residual_rows() as usize;
        let row_bytes = limits.residual_row_bytes as usize;
        Self {
            group,
            engine,
            machine,
            epoch: Instant::now(),
            scheduler: Some(StageScheduler::new(Facts {
                id: group,
                devices,
                buffers,
                max_quantum_ms: u64::from(limits.max_quantum_ms),
            })),
            bodies: BTreeMap::new(),
            spans: BTreeMap::new(),
            requests: BTreeMap::new(),
            ready_since: BTreeMap::new(),
            stage_ready_since: BTreeMap::new(),
            running_stages: BTreeMap::new(),
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
        let request = spec.body.request();
        let span = tracing::info_span!(
            parent: &parent,
            "nagaya.job",
            engine = self.engine.as_str(),
            class = spec.class.label(),
            request,
            outcome = tracing::field::Empty,
        );
        self.spans.insert(id, span);
        self.requests.insert(id, request);
        self.ready_since.insert(id, now_ms);
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
                    let now_ms = self.now_ms();
                    let ready = self.stage_ready_since.remove(&(job, block, stage)).unwrap_or(now_ms);
                    let wait_ms = now_ms.saturating_sub(ready);
                    let span = self.start_quantum(&quantum, wait_ms);
                    let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
                    let entry = self.blocks.get_mut(&(job, block)).expect("a later stage has its block");
                    let residual = entry.residual.take().expect("the stage before wrote the residual");
                    self.running_stages.insert(
                        (job, block, stage),
                        RunningStage {
                            kind: entry.kind,
                            class: entry.class,
                            request: entry.request,
                            watch,
                            span: span.clone(),
                        },
                    );
                    threads.senders[stage]
                        .send(StageWork::Pass {
                            job,
                            block,
                            stage,
                            segments: Arc::clone(&entry.segments),
                            residual,
                            span,
                        })
                        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
                }
                Work::Restore { .. } => unreachable!("the job path makes no restore quantum; it restores in a copy"),
                Work::On { .. } => unreachable!("the group makes no quantum on one device until it loads two engines"),
            }
            self.forget_ended_jobs();
        }
    }

    fn run_whole(&mut self, quantum: &StageQuantum, engines: &mut [Box<dyn Engine>]) {
        let now_ms = self.now_ms();
        let ready = self.ready_since.get(&quantum.job).copied().unwrap_or(now_ms);
        let wait_ms = now_ms.saturating_sub(ready);
        let span = self.start_quantum(quantum, wait_ms);
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        let request = self.requests.get(&quantum.job).copied().unwrap_or("job");
        let started = Instant::now();
        let ran = match self.bodies.get_mut(&quantum.job) {
            Some(body) => match &span {
                Some(span) => span.in_scope(|| body.run(quantum.kind, engines)),
                None => body.run(quantum.kind, engines),
            },
            None => Ran::Cancelled,
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        if matches!(ran, Ran::Failed) {
            self.log_failure(quantum.kind, request, watch);
        }
        let outcome = self.apply(quantum.job, ran);
        self.end_quantum(quantum.kind, quantum.class, duration_ms, outcome, span);
        let done_ms = self.now_ms();
        self.ready_since.insert(quantum.job, done_ms);
        let device = quantum.devices[0];
        self.event(StageEvent::Complete {
            now_ms: done_ms,
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
        let now_ms = self.now_ms();
        let ready = self.ready_since.get(&job).copied().unwrap_or(now_ms);
        let wait_ms = now_ms.saturating_sub(ready);
        let request = self.requests.get(&job).copied().unwrap_or("job");
        let begun = match self.bodies.get_mut(&job) {
            Some(body) => body.begin_pass(quantum.kind, engines),
            None => Err(Ran::Cancelled),
        };
        let segments = match begun {
            Ok(segments) => Arc::new(segments),
            Err(ran) => {
                let span = self.start_quantum(quantum, wait_ms);
                let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
                if matches!(ran, Ran::Failed) {
                    self.log_failure(quantum.kind, request, watch);
                }
                let outcome = self.apply(job, ran);
                self.end_quantum(quantum.kind, quantum.class, 0, outcome, span);
                self.event(StageEvent::Complete {
                    now_ms: self.now_ms(),
                    device: quantum.devices[0],
                    duration_ms: 0,
                    outcome,
                });
                return;
            }
        };
        let span = self.start_quantum(quantum, wait_ms);
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        // The scheduler starts a block only when a buffer is free, so the pool is never empty here.
        let residual = self.free.pop().expect("a block that starts has a free residual buffer");
        self.blocks.insert(
            (job, block),
            Block {
                kind: quantum.kind,
                class: quantum.class,
                request,
                segments: Arc::clone(&segments),
                residual: None,
            },
        );
        self.running_stages.insert(
            (job, block, 0),
            RunningStage {
                kind: quantum.kind,
                class: quantum.class,
                request,
                watch,
                span: span.clone(),
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
                span,
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
        let now_ms = self.now_ms();
        let last = stage + 1 == self.stages;
        let running = self.running_stages.remove(&(job, block, stage));
        let entry_kind = self
            .blocks
            .get(&(job, block))
            .map(|entry| entry.kind)
            .or_else(|| running.as_ref().map(|item| item.kind));
        if stage == 0 {
            self.ready_since.insert(job, now_ms);
        }
        let outcome = match (&result, last) {
            (Ok(StageOut::Residual { .. }), false) => {
                let entry = self
                    .blocks
                    .get_mut(&(job, block))
                    .expect("a block in flight has its entry");
                entry.residual = Some(residual);
                self.stage_ready_since.insert((job, block, stage + 1), now_ms);
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
                self.ready_since.insert(job, now_ms);
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
        if let Some(running) = running {
            if outcome == Completion::Failed {
                self.log_failure(running.kind, running.request, running.watch);
            }
            self.end_quantum(running.kind, running.class, ms, outcome, running.span);
        }
        let device = self.scheduler().engine().devices[stage];
        self.event(StageEvent::Complete {
            now_ms,
            device,
            duration_ms: ms,
            outcome,
        });
        self.forget_ended_jobs();
        self.pump(engines, threads);
    }

    fn start_quantum(&self, quantum: &StageQuantum, wait_ms: u64) -> Option<Span> {
        let labels = QuantumLabels {
            engine: &self.engine,
            device_group: self.group,
            kind: kind_label(quantum.kind),
            class: quantum.class.label(),
        };
        instruments().quantum_wait(&labels, wait_ms as f64);
        let parent = self.spans.get(&quantum.job);
        (quantum.kind != Kind::Decode).then(|| {
            let make = || {
                tracing::info_span!(
                    "nagaya.quantum",
                    engine = self.engine.as_str(),
                    device_group = self.group,
                    kind = kind_label(quantum.kind),
                    class = quantum.class.label(),
                    segments = 1usize,
                    wait_ms,
                    run_ms = tracing::field::Empty,
                    outcome = tracing::field::Empty,
                    otel.status_code = tracing::field::Empty,
                )
            };
            match parent {
                Some(parent) => parent.in_scope(make),
                None => make(),
            }
        })
    }

    fn end_quantum(&self, kind: Kind, class: Class, duration_ms: u64, outcome: Completion, span: Option<Span>) {
        let labels = QuantumLabels {
            engine: &self.engine,
            device_group: self.group,
            kind: kind_label(kind),
            class: class.label(),
        };
        instruments().quantum_run(&labels, duration_ms as f64);
        if let Some(span) = span {
            span.record("run_ms", duration_ms);
            match outcome {
                Completion::Done => {
                    span.record("outcome", "done");
                }
                Completion::Failed => {
                    span.record("outcome", "failed");
                    span.record("otel.status_code", "ERROR");
                }
            }
        }
    }

    fn log_failure(&self, kind: Kind, request: &'static str, watch: Option<Watch>) {
        let failure = format!(
            "{} quantum of a {} request on engine {}",
            kind_label(kind),
            request,
            self.engine
        );
        match watch {
            Some(watch) => watch.failed(&failure),
            None => evidence::failed_unwatched(&failure),
        };
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
            self.requests.remove(&id);
            self.ready_since.remove(&id);
            self.stage_ready_since.retain(|(job, _, _), _| *job != id);
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
        self.requests.clear();
        self.ready_since.clear();
        self.stage_ready_since.clear();
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

/// Serve the engine with stages at `index` of the group until the group stops or swaps (engine-abi D1, spec 3.3a).
///
/// The engine loads alone, so each job for another engine of the group waits, and the first one plans a swap. When the
/// engine holds no job, the loop unloads it while its stage threads still run, ends the threads, and gives the swap.
pub(super) fn serve(
    group: usize,
    index: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    waiting: &mut Vec<(JobSpec, Span)>,
    links: &GroupLinks,
    interval: Duration,
) -> Exit {
    let name = members[index].name().to_owned();
    let limits = members[index].job.limits.clone();
    let handles = std::mem::take(&mut members[index].handles);
    assert_eq!(
        handles.len(),
        limits.stages.len(),
        "the open of engine `{name}` gave a handle for each stage"
    );
    let machine: Vec<DeviceId> = members[index].job.devices.iter().map(|device| device.id).collect();
    let devices: Vec<DeviceId> = limits.stages.iter().map(|stage| stage.device).collect();
    let threads = StageThreads::start(group, handles, &machine, &devices, links, interval);
    let inner = std::mem::replace(&mut engines[index], Unloaded::of(&members[index]));
    engines[index] = Box::new(StagedEngine {
        inner,
        limits: limits.clone(),
        senders: threads.senders.clone(),
    });
    let mut jobs = StagedJobs::new(group as u32, name.clone(), &limits, links.machine.clone());
    // The jobs that waited for this engine run now. The others wait for the next swap.
    for (spec, parent) in std::mem::take(waiting) {
        if spec.engine == index {
            jobs.add(spec, parent, engines, &threads);
        } else {
            waiting.push((spec, parent));
        }
    }
    let mut pending: Option<Plan> = None;
    let mut cancelled = 0;
    let beat = || WatchMessage::Beat {
        group,
        at: Instant::now(),
    };
    let exit = loop {
        match links.messages.recv_timeout(interval) {
            Ok(GroupMessage::Quantum(quantum)) => {
                if links.accepting.load(Ordering::SeqCst) {
                    quantum.finish(engines);
                } else {
                    quantum.cancel();
                    cancelled += 1;
                }
            }
            Ok(GroupMessage::Job(spec, parent)) => {
                assert!(
                    spec.engine < members.len(),
                    "a job names engine {} of group {group}, which has {} engines",
                    spec.engine,
                    members.len()
                );
                if spec.engine == index && pending.is_none() {
                    jobs.add(spec, parent, engines, &threads);
                } else {
                    // This loop takes one message at a time and plans after it, so a later job waits (3.3a).
                    waiting.push((spec, parent));
                }
            }
            Ok(GroupMessage::StageDone(done)) => jobs.stage_done(done, engines, &threads),
            Ok(GroupMessage::Park(reply)) => {
                cancelled += jobs.cancel_all() + waiting.len();
                waiting.clear();
                // A park call waits on each stage thread for the stage quantum before it.
                let _ = reply.send(ParkProgress::Draining { ended: Instant::now() });
                let mut report = park_group(
                    std::slice::from_ref(&name),
                    &mut engines[index..=index],
                    links.store.as_ref(),
                );
                report.cancelled = cancelled;
                let _ = reply.send(ParkProgress::Parked(report));
                break Exit::Stop;
            }
            Ok(GroupMessage::Stop) | Err(RecvTimeoutError::Disconnected) => break Exit::Stop,
            Err(RecvTimeoutError::Timeout) => {}
        }
        let serving = links.accepting.load(Ordering::SeqCst);
        if !serving && !jobs.idle() {
            cancelled += jobs.cancel_all();
        }
        if serving
            && pending.is_none()
            && let Some((spec, _)) = waiting.first()
        {
            pending = Some(in_place::plan(spec.engine, members, &links.ledger));
        }
        if !jobs.idle() {
            members[index].last_busy = Instant::now();
        }
        if serving && pending.is_none() && waiting.is_empty() {
            pending = in_place::idle_plan(members, Instant::now(), &links.ledger);
        }
        if serving && pending.is_some() && jobs.idle() {
            // The park reads the parts through the stage threads, so the engine unloads before the threads end.
            in_place::unload(index, members, engines, links.store.as_ref(), links.fd_store.as_ref());
            break Exit::Swap(pending.take().expect("a pending swap"));
        }
        if links.heartbeats.send(beat()).is_err() {
            break Exit::Stop;
        }
    };
    // The engine holds clones of the senders, so it drops first. A swap unloaded it already. Each stage thread ends
    // its quantum, then stops when its sender drops.
    if members[index].loaded {
        drop(std::mem::replace(&mut engines[index], Unloaded::of(&members[index])));
    }
    let StageThreads { senders, handles } = threads;
    drop(senders);
    for handle in handles {
        let _ = handle.join();
    }
    exit
}
