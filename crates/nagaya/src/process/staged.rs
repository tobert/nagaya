//! A device group whose engine declares stages, or a group of more than one device (engine-abi D1, spec 3.2, 7.4).
//!
//! The group thread runs the one scheduler of the group ([`StageScheduler`]). It has one stage thread for each stage,
//! and each stage thread holds the [`StageHandle`] of its device. The group thread sends each stage quantum to the
//! thread of its device and gets the end back on its own channel, so the two devices run at the same time.
//!
//! A group of more than one device with no engine with stages loaded runs the same loop with one device thread for
//! each device, and no handle (spec 3.2, "One device thread per device"). An engine with one device opens on the
//! thread of its device, and its quanta run there, so engines on separate devices run at once.
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
use nagaya_telemetry::{EngineState, QuantumLabels, instruments};
use tracing::Span;

use super::in_place::{self, Exit, Member, Plan, Unloaded};
use super::loads::{self, Ask};
use super::{
    Event, GroupJob, GroupLinks, GroupMessage, LedgerMessage, OpenBeat, OpenedEngine, ParkProgress, ParkReport,
    WatchMessage, guard, handoff, park, park_group, pin, spawn_named,
};
use crate::arena::FdStore;
use crate::evidence::{self, Machine, Watch};
use crate::jobs::{JobBody, JobSpec, Ran};
use crate::park::StoreMessage;

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
    /// Open an engine with one device on this thread, and keep it here (engine-abi "How a group loads both"). The
    /// job comes back with the answer.
    Open {
        index: usize,
        count: usize,
        job: Box<GroupJob>,
        fd_store: Option<FdStore>,
        reply: Sender<(Box<GroupJob>, Result<OpenedEngine, OpenFailure>)>,
    },
    /// Run one quantum of a job of an engine on this thread. The body comes back in [`HostedDone`].
    Run {
        job: JobId,
        kind: Kind,
        body: Box<dyn JobBody>,
        span: Option<Span>,
    },
    /// Park the states of an engine on this thread, and drop it, which closes it (spec 3.3a).
    Unload {
        index: usize,
        name: String,
        store: Option<Sender<StoreMessage>>,
        reply: Sender<ParkReport>,
    },
}

/// Why an open on a stage or device thread failed: the kind, which decides the exit code (spec 10.4), and the message.
type OpenFailure = (super::OpenErrorKind, String);

/// The end of a quantum of an engine that a stage thread holds.
pub(crate) struct HostedDone {
    job: JobId,
    kind: Kind,
    body: Box<dyn JobBody>,
    ran: Ran,
    ms: u64,
}

/// A place in the engines of a thread for an engine that is not on that thread. A call here is a defect, and it
/// panics.
struct Elsewhere {
    name: String,
    limits: Option<Limits>,
    place: &'static str,
}

impl Elsewhere {
    /// The place of a loaded engine on a stage thread, as the group thread sees it.
    fn hosted(member: &Member) -> Box<dyn Engine> {
        Box::new(Self {
            name: member.name().to_owned(),
            limits: Some(member.job.limits.clone()),
            place: "it runs on the stage thread of its device",
        })
    }

    /// The place of each other engine of the group, as a stage thread sees it.
    fn absent(index: usize) -> Box<dyn Engine> {
        Box::new(Self {
            name: format!("{index} of the group"),
            limits: None,
            place: "it is not on this stage thread",
        })
    }
}

impl Engine for Elsewhere {
    fn identity(&self) -> Identity {
        panic!("the identity of engine {} was read here, but {}", self.name, self.place)
    }

    fn limits(&self) -> Limits {
        match &self.limits {
            Some(limits) => limits.clone(),
            None => panic!("the limits of engine {} were read here, but {}", self.name, self.place),
        }
    }

    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        panic!("a pass reached engine {} here, but {}", self.name, self.place)
    }

    fn last_error(&self) -> Option<String> {
        panic!(
            "the last error of engine {} was read here, but {}",
            self.name, self.place
        )
    }
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
    /// holds the group index of each stage (spec 2.4). A device thread has no handle, and its name says `dev`.
    fn start(
        group: usize,
        handles: Vec<Option<Box<dyn StageHandle + Send>>>,
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
            let kind = if handle.is_some() { "stg" } else { "dev" };
            let thread = spawn_named(format!("nagaya-{kind}-{group}-{stage}"), move || {
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

/// The handle of a stage thread. A device thread has none, so a stage call there is a defect.
fn own(handle: &mut Option<Box<dyn StageHandle + Send>>, stage: usize) -> &mut dyn StageHandle {
    handle
        .as_deref_mut()
        .map(|handle| handle as &mut dyn StageHandle)
        .unwrap_or_else(|| panic!("a stage call reached device thread {stage}, which holds no engine with stages"))
}

/// The loop of one stage thread. It beats its own heartbeat between its quanta (spec 7.4).
fn stage_thread(
    group: usize,
    stage: usize,
    mut handle: Option<Box<dyn StageHandle + Send>>,
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
    // The engines with one device that this thread holds, at their places in the group (engine-abi "How a group loads
    // both"). They drop with the thread, which closes them.
    let mut hosted: Vec<Box<dyn Engine>> = Vec::new();
    loop {
        match work.recv_timeout(interval) {
            Ok(StageWork::Call(call)) => call(own(&mut handle, stage)),
            Ok(StageWork::Open {
                index,
                count,
                mut job,
                fd_store,
                reply,
            }) => {
                // The open blocks this thread for seconds, so a helper beats for it (spec 7.4).
                let beat_guard = OpenBeat::stage(group, stage, heartbeats.clone(), interval);
                let opened = in_place::open(&mut job, fd_store.as_ref());
                drop(beat_guard);
                let answer = match opened {
                    Ok((engine, handles, opened)) => {
                        assert!(handles.is_empty(), "an engine with one device gives no stage handle");
                        while hosted.len() < count {
                            hosted.push(Elsewhere::absent(hosted.len()));
                        }
                        hosted[index] = engine;
                        Ok(opened)
                    }
                    Err(failure) => Err(failure),
                };
                if reply.send((job, answer)).is_err() {
                    return;
                }
            }
            Ok(StageWork::Run {
                job,
                kind,
                mut body,
                span,
            }) => {
                let started = Instant::now();
                let ran = match &span {
                    Some(span) => span.in_scope(|| body.run(kind, &mut hosted)),
                    None => body.run(kind, &mut hosted),
                };
                let end = HostedDone {
                    job,
                    kind,
                    body,
                    ran,
                    ms: started.elapsed().as_millis() as u64,
                };
                if done.send(GroupMessage::HostedDone(end)).is_err() {
                    return;
                }
            }
            Ok(StageWork::Unload {
                index,
                name,
                store,
                reply,
            }) => {
                let mut report = ParkReport::default();
                park::park_engine(&name, hosted[index].as_mut(), store.as_ref(), &mut report);
                // The drop closes the engine (engine-abi "`close` in Rust").
                hosted[index] = Elsewhere::absent(index);
                if reply.send(report).is_err() {
                    return;
                }
            }
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
                    Some(span) => span.in_scope(|| own(&mut handle, stage).stage_pass(&segments, &mut residual)),
                    None => own(&mut handle, stage).stage_pass(&segments, &mut residual),
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

/// Where the engine of a job is, which decides where its quanta run.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// The engine with stages: its passes run stage by stage.
    Stages,
    /// An engine on the stage or device thread of this device.
    On(Device),
    /// Another engine on the group thread, such as an engine of more devices with no stages. Its quanta hold every
    /// device.
    Group,
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
    /// The engine of each job, as its place in the group.
    engine_of: BTreeMap<JobId, usize>,
    /// The device of each job of an engine on a stage thread (engine-abi "How a group loads both").
    on: BTreeMap<JobId, Device>,
    /// The jobs of an engine on the group thread that is not the engine with stages. Each of their quanta holds
    /// every device and runs there, where the engine is.
    whole: std::collections::BTreeSet<JobId>,
    /// The quanta of those jobs that run on a stage thread. Their bodies are there.
    running_on: BTreeMap<JobId, RunningStage>,
}

impl StagedJobs {
    /// The jobs of a group. `limits` are the limits of its engine with stages; with none, the devices are those of
    /// the device threads, and no job has a block.
    fn new(group: u32, engine: String, limits: Option<&Limits>, devices: &[Device], machine: Machine) -> Self {
        let buffers = 2;
        let (rows, row_bytes, stages, max_quantum_ms) = match limits {
            Some(limits) => (
                limits.residual_rows() as usize,
                limits.residual_row_bytes as usize,
                limits.stages.len(),
                u64::from(limits.max_quantum_ms),
            ),
            None => (0, 0, 0, u64::MAX),
        };
        Self {
            group,
            engine,
            machine,
            epoch: Instant::now(),
            scheduler: Some(StageScheduler::new(Facts {
                id: group,
                devices: devices.to_vec(),
                buffers,
                max_quantum_ms,
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
            stages,
            next_id: 1,
            engine_of: BTreeMap::new(),
            on: BTreeMap::new(),
            whole: std::collections::BTreeSet::new(),
            running_on: BTreeMap::new(),
        }
    }

    /// True when a job of the engine at `member` has not ended.
    fn holds(&self, member: usize) -> bool {
        self.engine_of.values().any(|engine| *engine == member)
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

    /// Add a job of the engine at `place`. A job of an engine on a stage thread holds only the device of that thread
    /// with each quantum.
    fn add(
        &mut self,
        spec: JobSpec,
        parent: Span,
        place: Place,
        engines: &mut [Box<dyn Engine>],
        threads: &StageThreads,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        let now_ms = self.now_ms();
        match place {
            Place::Stages => {}
            Place::On(device) => {
                self.on.insert(id, device);
            }
            Place::Group => {
                self.whole.insert(id);
            }
        }
        let item = self.item(id, spec.first, spec.body.as_ref());
        self.engine_of.insert(id, spec.engine);
        let mut job = StageJob::new(id, spec.class, now_ms, [item]);
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
                Work::On { device } => self.send_on(&quantum, device, engines, threads),
            }
            self.forget_ended_jobs();
        }
    }

    /// Send a quantum of an engine on a stage thread to that thread, with the body of its job.
    fn send_on(
        &mut self,
        quantum: &StageQuantum,
        device: Device,
        engines: &mut [Box<dyn Engine>],
        threads: &StageThreads,
    ) {
        let job = quantum.job;
        let Some(body) = self.bodies.remove(&job) else {
            // The job was cancelled before its quantum started, so the quantum ends here at once.
            return self.run_whole(quantum, engines);
        };
        let now_ms = self.now_ms();
        let ready = self.ready_since.get(&job).copied().unwrap_or(now_ms);
        let span = self.start_quantum(quantum, now_ms.saturating_sub(ready));
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        let request = self.requests.get(&job).copied().unwrap_or("job");
        self.running_on.insert(
            job,
            RunningStage {
                kind: quantum.kind,
                class: quantum.class,
                request,
                watch,
                span: span.clone(),
            },
        );
        let stage = self
            .scheduler()
            .engine()
            .devices
            .iter()
            .position(|candidate| *candidate == device)
            .expect("the scheduler starts a quantum on one device only on a device of a stage");
        threads.senders[stage]
            .send(StageWork::Run {
                job,
                kind: quantum.kind,
                body,
                span,
            })
            .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
    }

    /// The end of a quantum of an engine on a stage thread.
    fn hosted_done(&mut self, done: HostedDone, engines: &mut [Box<dyn Engine>], threads: &StageThreads) {
        let HostedDone {
            job,
            kind,
            mut body,
            ran,
            ms,
        } = done;
        let device = *self.on.get(&job).expect("a quantum on one device has its device");
        let running = self.running_on.remove(&job);
        if self.scheduler().jobs().iter().any(|held| held.id == job) {
            self.bodies.insert(job, body);
        } else {
            body.ended(JobState::Cancelled);
        }
        let outcome = if self.bodies.contains_key(&job) {
            self.apply(job, ran)
        } else {
            Completion::Done
        };
        if let Some(running) = running {
            if outcome == Completion::Failed {
                self.log_failure(kind, running.request, running.watch);
            }
            self.end_quantum(kind, running.class, ms, outcome, running.span);
        }
        let now_ms = self.now_ms();
        self.ready_since.insert(job, now_ms);
        self.event(StageEvent::Complete {
            now_ms,
            device,
            duration_ms: ms,
            outcome,
        });
        self.forget_ended_jobs();
        self.pump(engines, threads);
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

    /// The item of a quantum of `kind` of a job, by the place of its engine.
    fn item(&self, job: JobId, kind: Kind, body: &dyn JobBody) -> Item {
        match self.on.get(&job) {
            Some(device) => Item::On(kind, *device),
            None if self.whole.contains(&job) => Item::Whole(kind),
            None => body.item(kind),
        }
    }

    /// Apply what a body reported to the scheduler, and give the end of the quantum.
    fn apply(&mut self, job: JobId, ran: Ran) -> Completion {
        match ran {
            Ran::Done => Completion::Done,
            Ran::More(kinds) | Ran::First(kinds) if kinds.is_empty() => Completion::Done,
            Ran::More(kinds) => {
                let body = self.bodies.get(&job).expect("a body that adds quanta lives");
                let items: Vec<Item> = kinds.iter().map(|kind| self.item(job, *kind, body.as_ref())).collect();
                self.scheduler
                    .as_mut()
                    .expect("the scheduler is in place between steps")
                    .append_items(job, &items)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            // A copy first, such as an interval copy (spec 6.13), on the place of the job's engine.
            Ran::First(kinds) => {
                let body = self.bodies.get(&job).expect("a body that adds quanta lives");
                let items: Vec<Item> = kinds.iter().map(|kind| self.item(job, *kind, body.as_ref())).collect();
                self.scheduler
                    .as_mut()
                    .expect("the scheduler is in place between steps")
                    .prepend_items(job, &items)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
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
            self.engine_of.remove(&id);
            self.on.remove(&id);
            self.whole.remove(&id);
            if let Some(mut body) = self.bodies.remove(&id) {
                body.ended(state);
            }
        }
    }

    /// Cancel every job (spec 10.3, 5.9). The stage quanta that run go to their end. It gives the number of jobs.
    fn cancel_all(&mut self) -> usize {
        // A job whose quantum runs on a stage or device thread has its body there, so it is in `running_on`.
        let mut ids: Vec<JobId> = self.bodies.keys().chain(self.running_on.keys()).copied().collect();
        ids.sort_unstable();
        ids.dedup();
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

/// The engines with one device that the stage threads of a serve loop hold, by their places in the group, with the
/// stage of each (engine-abi "How a group loads both").
type Hosted = BTreeMap<usize, usize>;

/// The stage whose device is the one device of `member`, or `None` when the member cannot run on a stage thread.
fn stage_of(member: &Member, machine: &[DeviceId], devices: &[DeviceId]) -> Option<usize> {
    if member.staged() || member.job.devices.len() != 1 {
        return None;
    }
    let id = member.job.devices[0].id;
    devices
        .iter()
        .position(|device| machine.get(*device as usize) == Some(&id))
}

/// Park and close the engine at `member` on the thread of its stage (spec 3.3a). A part that cannot be parked is a
/// crash (spec 8.5).
fn unload_hosted(
    member: usize,
    stage: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    threads: &StageThreads,
    links: &GroupLinks,
) {
    let name = members[member].name().to_owned();
    links.states.record(&name, EngineState::Draining);
    let (reply, answer) = mpsc::channel();
    threads.senders[stage]
        .send(StageWork::Unload {
            index: member,
            name: name.clone(),
            store: links.store.clone(),
            reply,
        })
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
    let report = answer
        .recv()
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped during an unload"));
    if !report.failures.is_empty() {
        panic!(
            "the swap cannot park the states of engine `{name}`: {}",
            report.failures.join("; ")
        );
    }
    // An fd in systemd's store holds the device memory of its part (spec 3.3a).
    handoff::remove_arena(&members[member].job, links.fd_store.as_ref());
    engines[member] = Unloaded::of(&members[member]);
    members[member].loaded = false;
    links.states.record(&name, EngineState::Unloaded);
    tracing::info!(
        "engine `{name}` unloaded from stage {stage}: parked {} states",
        report.states
    );
}

/// Open the engine at `member` on the thread of `stage`, and mark it ready (engine-abi "How a group loads both").
fn open_hosted(
    member: usize,
    stage: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    threads: &StageThreads,
    links: &GroupLinks,
) -> Result<OpenedEngine, OpenFailure> {
    let engine = members[member].name().to_owned();
    links.states.record(&engine, EngineState::Loading);
    let shell = members[member].job.shell();
    let job = Box::new(std::mem::replace(&mut members[member].job, shell));
    let (reply, answer) = mpsc::channel();
    threads.senders[stage]
        .send(StageWork::Open {
            index: member,
            count: members.len(),
            job,
            fd_store: links.fd_store.clone(),
            reply,
        })
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
    let (job, opened) = answer
        .recv()
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped during an open"));
    members[member].job = *job;
    let opened = opened?;
    members[member].loaded = true;
    members[member].last_busy = Instant::now();
    engines[member] = Elsewhere::hosted(&members[member]);
    links.states.opened(&engine, &opened.identity);
    links.states.record(&engine, EngineState::Ready);
    Ok(opened)
}

/// Run a swap that keeps the engine with stages loaded: unload engines on stage threads, then open engines on the
/// stage threads of their devices (engine-abi "How a group loads both"). A failure is a crash (spec 8.5).
#[allow(clippy::too_many_arguments)]
fn swap_beside(
    group: usize,
    plan: Plan,
    stages: &BTreeMap<usize, usize>,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    hosted: &mut Hosted,
    threads: &StageThreads,
    links: &GroupLinks,
    interval: Duration,
) {
    let started = Instant::now();
    let name = |index: &usize| members[*index].name().to_owned();
    let unloaded: Vec<String> = plan.unload.iter().map(name).collect();
    let loading: Vec<String> = plan.load.iter().map(name).collect();
    let span = tracing::info_span!(
        "nagaya.swap",
        group,
        unload = unloaded.join(",").as_str(),
        load = loading.join(",").as_str(),
        unload_ms = tracing::field::Empty,
        load_ms = tracing::field::Empty,
    );
    let _entered = span.enter();
    // An open blocks this thread for seconds, so a helper beats for the group (spec 7.4).
    let beat = OpenBeat::start(group, links.heartbeats.clone(), interval);
    for &member in &plan.unload {
        match hosted.remove(&member) {
            Some(stage) => unload_hosted(member, stage, members, engines, threads, links),
            // The group thread unloaded it before this loop began (`Entry::Swap`).
            None => assert!(
                !members[member].loaded,
                "a swap beside the stages unloads engine `{}`, which is on no stage thread",
                members[member].name()
            ),
        }
    }
    span.record("unload_ms", started.elapsed().as_millis() as u64);
    let (reply, answer) = mpsc::channel();
    links
        .ledger
        .send(LedgerMessage::Swap {
            load: loading.clone(),
            unload: unloaded.clone(),
            reply,
        })
        .expect("the ledger thread stopped");
    if let Err(error) = answer.recv().expect("the ledger thread stopped") {
        panic!("the ledger refused the swap to {}: {error}", loading.join(", "));
    }
    let opening = Instant::now();
    let mut loaded = Vec::with_capacity(plan.load.len());
    for &member in &plan.load {
        let stage = stages[&member];
        let engine = members[member].name().to_owned();
        let opened = open_hosted(member, stage, members, engines, threads, links)
            .unwrap_or_else(|(_, message)| panic!("the swap cannot open engine `{engine}`: {message}"));
        hosted.insert(member, stage);
        loaded.push(opened);
    }
    span.record("load_ms", opening.elapsed().as_millis() as u64);
    drop(beat);
    tracing::info!(
        "swap beside the stages of group {group}: unloaded [{}], loaded [{}] in {} ms",
        unloaded.join(", "),
        loading.join(", "),
        started.elapsed().as_millis()
    );
    let _ = links.events.send(Event::Swapped {
        group,
        unloaded,
        loaded,
    });
}

/// How a serve loop begins.
pub(super) enum Entry {
    /// The engines are loaded as the loop finds them.
    Serve,
    /// The group thread unloaded the engines of the plan. The loop loads its engines on their device threads.
    Swap(Plan),
    /// The start of the process: the loop opens these resident engines on their device threads, then sends the
    /// `Opened` event with the engines that opened before it.
    Start {
        open: Vec<usize>,
        opened: Vec<OpenedEngine>,
    },
}

/// The machine devices of a group, by group index: the devices of its members, in machine order.
pub(super) fn group_devices(members: &[Member]) -> Vec<DeviceId> {
    let mut devices: Vec<DeviceId> = members
        .iter()
        .flat_map(|member| member.job.devices.iter().map(|device| device.id))
        .collect();
    devices.sort_unstable();
    devices.dedup();
    devices
}

/// Serve the group until it stops or swaps (engine-abi D1, spec 3.2, 3.3a). `index` is the loaded engine with stages,
/// or `None` for device threads.
///
/// An engine with one device loads beside it when the ledger admits both. Its engine opens on the stage thread of its
/// device, and its quanta run there (engine-abi "How a group loads both"). A load or an unload plans a swap (spec
/// 3.3b). A plan that unloads the engine with stages waits until the engine holds no job. Then the loop unloads the
/// engines on the stage threads and the engine itself while the threads still run, ends the threads, and gives the
/// swap. With no engine with stages, the threads are device threads, and an engine with one device loads on its
/// device thread in the same way (spec 3.2).
#[allow(clippy::too_many_arguments)]
pub(super) fn serve(
    group: usize,
    index: Option<usize>,
    entry: Entry,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    asks: &mut VecDeque<Ask>,
    links: &GroupLinks,
    interval: Duration,
) -> Exit {
    let names: Vec<String> = members.iter().map(|member| member.name().to_owned()).collect();
    let (name, machine, devices, limits) = match index {
        Some(index) => {
            let limits = members[index].job.limits.clone();
            let machine: Vec<DeviceId> = members[index].job.devices.iter().map(|device| device.id).collect();
            let devices: Vec<DeviceId> = limits.stages.iter().map(|stage| stage.device).collect();
            (members[index].name().to_owned(), machine, devices, Some(limits))
        }
        None => {
            let machine = group_devices(members);
            let devices: Vec<DeviceId> = (0..machine.len() as DeviceId).collect();
            (names.join("+"), machine, devices, None)
        }
    };
    let handles: Vec<Option<Box<dyn StageHandle + Send>>> = match index {
        Some(index) => {
            let handles = std::mem::take(&mut members[index].handles);
            assert_eq!(
                handles.len(),
                devices.len(),
                "the open of engine `{name}` gave a handle for each stage"
            );
            handles.into_iter().map(Some).collect()
        }
        None => devices.iter().map(|_| None).collect(),
    };
    let threads = StageThreads::start(group, handles, &machine, &devices, links, interval);
    if let (Some(index), Some(limits)) = (index, &limits) {
        let inner = std::mem::replace(&mut engines[index], Unloaded::of(&members[index]));
        engines[index] = Box::new(StagedEngine {
            inner,
            limits: limits.clone(),
            senders: threads.senders.clone(),
        });
    }
    // The stage of each engine with one device on a device of a stage.
    let stages: BTreeMap<usize, usize> = (0..members.len())
        .filter_map(|member| stage_of(&members[member], &machine, &devices).map(|stage| (member, stage)))
        .collect();
    let mut hosted: Hosted = BTreeMap::new();
    match entry {
        Entry::Serve => {}
        Entry::Swap(plan) => swap_beside(
            group,
            plan,
            &stages,
            members,
            engines,
            &mut hosted,
            &threads,
            links,
            interval,
        ),
        Entry::Start { open, mut opened } => {
            let beat = OpenBeat::start(group, links.heartbeats.clone(), interval);
            for member in open {
                let stage = *stages
                    .get(&member)
                    .unwrap_or_else(|| panic!("engine `{}` has no device thread", names[member]));
                match open_hosted(member, stage, members, engines, &threads, links) {
                    Ok(engine) => {
                        hosted.insert(member, stage);
                        opened.push(engine);
                    }
                    Err((kind, message)) => {
                        drop(beat);
                        let _ = links.events.send(Event::OpenFailed {
                            group,
                            engine: names[member].clone(),
                            kind,
                            message,
                        });
                        return end_threads(group, index, members, engines, hosted, threads, links, Exit::Stop);
                    }
                }
            }
            drop(beat);
            let _ = links.events.send(Event::Opened { group, engines: opened });
        }
    }
    let mut jobs = StagedJobs::new(
        group as u32,
        name.clone(),
        limits.as_ref(),
        &devices,
        links.machine.clone(),
    );
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
                // A planned swap set the states of its engines, so a job for an engine that unloads ends here.
                let engine = members[spec.engine].name().to_owned();
                match links.states.get(&engine) {
                    // An engine on a stage or device thread holds its device. Another engine, such as an engine of
                    // more than one device that loaded on the group thread, holds every device.
                    Some(EngineState::Ready) => {
                        let place = match hosted.get(&spec.engine) {
                            Some(stage) => Place::On(devices[*stage]),
                            None if Some(spec.engine) == index => Place::Stages,
                            None => Place::Group,
                        };
                        jobs.add(spec, parent, place, engines, &threads);
                    }
                    state => super::refuse(spec, &engine, state),
                }
            }
            Ok(GroupMessage::Ask(ask)) => asks.push_back(ask),
            Ok(GroupMessage::StageDone(done)) => jobs.stage_done(done, engines, &threads),
            Ok(GroupMessage::HostedDone(done)) => jobs.hosted_done(done, engines, &threads),
            Ok(GroupMessage::Park(reply)) => {
                cancelled += jobs.cancel_all();
                // An ask that waits gets no reply. Its caller hears that the group stopped.
                asks.clear();
                // A park call waits on each stage thread for the stage quantum before it.
                let _ = reply.send(ParkProgress::Draining { ended: Instant::now() });
                // The engines on the group thread park here: the engine with stages, or an engine of more devices.
                let local: Vec<usize> = (0..members.len())
                    .filter(|member| members[*member].loaded && !hosted.contains_key(member))
                    .collect();
                let mut report = super::ParkReport::default();
                for &member in &local {
                    let part = park_group(
                        std::slice::from_ref(&names[member]),
                        &mut engines[member..=member],
                        links.store.as_ref(),
                    );
                    report.merge(part);
                }
                // Each engine on a stage thread parks there, after the quantum that runs on it.
                for (member, stage) in std::mem::take(&mut hosted) {
                    let (answer, parked) = mpsc::channel();
                    let engine = members[member].name().to_owned();
                    let sent = threads.senders[stage].send(StageWork::Unload {
                        index: member,
                        name: engine.clone(),
                        store: links.store.clone(),
                        reply: answer,
                    });
                    report.engines += 1;
                    match sent.ok().and_then(|()| parked.recv().ok()) {
                        Some(parked) => {
                            report.states += parked.states;
                            report.bytes += parked.bytes;
                            if parked.failures.is_empty() {
                                report.parked_engines.push(engine);
                            }
                            report.failures.extend(parked.failures);
                        }
                        None => report
                            .failures
                            .push(format!("the thread of stage {stage} stopped before the park")),
                    }
                    engines[member] = Unloaded::of(&members[member]);
                    members[member].loaded = false;
                }
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
        // A plan that does not keep the engine with stages beside its hosted engines unloads every loaded engine. A
        // resident hosted engine then loads again on the group thread, so the group reaches the state of its profile
        // (spec 3.3). An engine with stages loads alone, so a plan that loads one loads no other engine.
        // With no engine with stages, the same holds for the engines on the group thread: a plan that does not keep
        // them unloads them.
        let local: Vec<usize> = (0..members.len())
            .filter(|member| members[*member].loaded && !hosted.contains_key(member))
            .collect();
        let widen = |mut plan: Plan, members: &[Member]| {
            let beside = local.iter().all(|member| !plan.unload.contains(member))
                && plan.unload.iter().all(|member| hosted.contains_key(member))
                && plan.load.iter().all(|member| stages.contains_key(member));
            if !beside {
                let loads_staged = plan.load.iter().any(|member| members[*member].staged());
                for member in hosted.keys().copied().chain(local.iter().copied()) {
                    if !plan.unload.contains(&member) {
                        plan.unload.push(member);
                    }
                    let again = Some(member) != index && members[member].resident() && !members[member].retired;
                    if again && !loads_staged && !plan.load.contains(&member) {
                        plan.load.push(member);
                    }
                }
            }
            plan
        };
        if serving && pending.is_none() {
            pending = loads::next_plan(asks, members, &links.ledger, &links.states, widen);
        }
        let now = Instant::now();
        for member in local.iter().copied().chain(hosted.keys().copied()) {
            if jobs.holds(member) {
                members[member].last_busy = now;
            }
        }
        if serving && pending.is_none() && asks.is_empty() {
            pending = in_place::idle_plan(members, now, &links.ledger).map(|plan| widen(plan, members));
            if let Some(plan) = &pending {
                loads::mark(plan, members, &links.states);
            }
        }
        let beside = pending.as_ref().is_some_and(|plan| {
            local.iter().all(|member| !plan.unload.contains(member))
                && plan.unload.iter().all(|member| hosted.contains_key(member))
                && plan.load.iter().all(|member| stages.contains_key(member))
        });
        if serving && beside {
            let plan = pending.as_ref().expect("a pending swap");
            if plan.unload.iter().all(|member| !jobs.holds(*member)) {
                let plan = pending.take().expect("a pending swap");
                swap_beside(
                    group,
                    plan,
                    &stages,
                    members,
                    engines,
                    &mut hosted,
                    &threads,
                    links,
                    interval,
                );
            }
        } else if serving && pending.is_some() && jobs.idle() {
            // The engines on the stage threads and the engine with stages unload while the threads still run. The
            // ledger hears of each of them in the swap.
            let mut plan = pending.take().expect("a pending swap");
            for (member, stage) in std::mem::take(&mut hosted) {
                unload_hosted(member, stage, members, engines, &threads, links);
                if !plan.unload.contains(&member) {
                    plan.unload.push(member);
                }
            }
            for &member in &local {
                if !plan.unload.contains(&member) {
                    plan.unload.push(member);
                }
                in_place::unload(
                    member,
                    members,
                    engines,
                    links.store.as_ref(),
                    links.fd_store.as_ref(),
                    &links.states,
                );
            }
            break Exit::Swap(plan);
        }
        if links.heartbeats.send(beat()).is_err() {
            break Exit::Stop;
        }
    };
    end_threads(group, index, members, engines, hosted, threads, links, exit)
}

/// End the threads of a serve loop, and give `exit`. The engine with stages holds clones of the senders, so it drops
/// first; a swap unloaded it already. Each thread ends its quantum, then stops when its sender drops, and the engines
/// that it holds drop with it.
#[allow(clippy::too_many_arguments)]
fn end_threads(
    group: usize,
    index: Option<usize>,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    hosted: Hosted,
    threads: StageThreads,
    links: &GroupLinks,
    exit: Exit,
) -> Exit {
    if let Some(index) = index
        && members[index].loaded
    {
        drop(std::mem::replace(&mut engines[index], Unloaded::of(&members[index])));
    }
    for member in hosted.into_keys() {
        engines[member] = Unloaded::of(&members[member]);
        members[member].loaded = false;
    }
    let StageThreads { senders, handles } = threads;
    drop(senders);
    for (stage, handle) in handles.into_iter().enumerate() {
        let _ = handle.join();
        // A stage thread that ended beats no more, so its watch ends with it (spec 7.4).
        let _ = links.heartbeats.send(WatchMessage::StopStage { group, stage });
    }
    exit
}
