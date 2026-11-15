//! A device group whose engine declares stages, or a group of more than one device (engine-abi D1, spec 3.2, 7.4).
//!
//! The group thread runs the one scheduler of the group ([`StageScheduler`]). It has one stage thread for each stage,
//! and each stage thread holds the [`StageHandle`] of its device. The group thread sends each stage quantum to the
//! thread of its device and gets the end back on its own channel, so the two devices run at the same time.
//!
//! A group of more than one device has one device thread for each device, for the life of the group (spec 3.2, "The
//! device threads live as long as the group"). The engine with stages lends the handle of each stage to the thread of
//! its device while it is loaded. An engine with one device opens on the thread of its device and stays there, so
//! engines on separate devices run at once, and a swap of the engine with stages never moves them (spec 3.3a).
//!
//! A swap of engines with one device runs on their threads (spec 3.3a, "Single-card swaps"). The group thread sends
//! each unload and open to the thread of its device, and it serves the other devices until the answers come back
//! ([`Swapping`]). The swap of an engine of more than one device still runs on the group thread.
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
use crate::jobs::{JobBody, JobSpec, Ran, run_joined_bodies};
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
        reply: Answer<Opening>,
    },
    /// Run one quantum of a job of an engine on this thread. The body comes back in [`HostedDone`].
    Run {
        job: JobId,
        kind: Kind,
        body: Box<dyn JobBody>,
        span: Option<Span>,
    },
    /// Run one joined decode quantum of several jobs of engines on this thread, each with the place of its engine.
    /// The bodies come back in [`HostedJoinedDone`] (engine-abi "LFM3", "Batched passes").
    RunJoined {
        lead: JobId,
        kind: Kind,
        jobs: Vec<(JobId, usize, Box<dyn JobBody>)>,
        span: Option<Span>,
    },
    /// Take the handle of a stage from the engine with stages that opened. The stage is the one whose device is the
    /// device of this thread.
    Attach {
        stage: usize,
        handle: Box<dyn StageHandle + Send>,
    },
    /// Drop the handle of the engine with stages that unloaded, and answer when it is gone.
    Detach(Sender<()>),
    /// Park the states of an engine on this thread, and drop it, which closes it (spec 3.3a).
    Unload {
        index: usize,
        name: String,
        store: Option<Sender<StoreMessage>>,
        reply: Answer<ParkReport>,
    },
}

/// The job of an engine that a stage thread opened, with the result of the open.
type Opening = (Box<GroupJob>, Result<OpenedEngine, OpenFailure>);

/// Where a stage thread sends the answer of an open or an unload.
enum Answer<T> {
    /// A caller waits for the answer on this channel.
    Wait(Sender<T>),
    /// The answer is a step of a swap that runs while the group thread serves (spec 3.3a, "Single-card swaps").
    Swap,
}

/// The end of one unload or one open of a swap on a stage or device thread, for the group thread.
pub(crate) enum SwapStep {
    /// The thread parked and closed the engine at `member`.
    Unloaded { member: usize, report: ParkReport },
    /// The thread opened the engine at `member`, or failed to open it. The box keeps the group message small.
    Opened { member: usize, opening: Box<Opening> },
}

/// Why an open on a stage or device thread failed: the kind, which decides the exit code (spec 10.4), and the message.
type OpenFailure = (super::OpenErrorKind, String);

/// The end of a joined decode quantum on a device thread: the bodies, and the result of each job.
pub(crate) struct HostedJoinedDone {
    lead: JobId,
    kind: Kind,
    jobs: Vec<(JobId, usize, Box<dyn JobBody>)>,
    results: Vec<(JobId, Ran)>,
    ms: u64,
}

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

    /// The place of an engine on a device thread with these limits, for a test of the group thread.
    #[cfg(test)]
    fn hosted_with(limits: Limits) -> Box<dyn Engine> {
        Box::new(Self {
            name: "a test engine".to_owned(),
            limits: Some(limits),
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

/// The device threads of a group of more than one device, one for each device in machine order (spec 3.2). They live
/// as long as the group, so an engine with one device stays on its thread through each swap.
///
/// The place of a thread is the position of its machine device in the group's sorted list. It is not a stage: stage
/// `s` of an engine with stages runs on the thread of machine device `devices[s]` of its profile entry (spec 2.4).
pub(super) struct StageThreads {
    senders: Vec<Sender<StageWork>>,
    handles: Vec<JoinHandle<()>>,
    /// The engines with one device on the threads, by their places in the group, with the thread of each.
    hosted: Hosted,
    /// The engine with stages whose handles the threads hold.
    attached: Option<usize>,
}

impl StageThreads {
    /// Start a thread for each device of the group. `machine` holds the machine devices of the group in machine
    /// order. A thread has no handle until an engine with stages loads.
    pub(super) fn start(group: usize, machine: &[DeviceId], links: &GroupLinks, interval: Duration) -> Self {
        let mut senders = Vec::with_capacity(machine.len());
        let mut threads = Vec::with_capacity(machine.len());
        for (place, &device) in machine.iter().enumerate() {
            let (work_tx, work_rx) = mpsc::channel();
            let events = links.events.clone();
            let done = links.own.clone();
            let heartbeats = links.heartbeats.clone();
            let cpus = links.sets.as_ref().map(|sets| sets.device(device).to_vec());
            let _ = heartbeats.send(WatchMessage::StartStage {
                group,
                stage: place,
                devices: vec![device],
            });
            let thread = spawn_named(format!("nagaya-dev-{group}-{place}"), move || {
                guard("stage", &events, || {
                    // A device thread takes the CPUs of its card (spec 3.2b).
                    pin("stage", cpus.as_deref());
                    stage_thread(group, place, work_rx, done, heartbeats, interval);
                });
            });
            senders.push(work_tx);
            threads.push(thread);
        }
        Self {
            senders,
            handles: threads,
            hosted: Hosted::new(),
            attached: None,
        }
    }

    /// Give the handle of each stage of the engine at `index` to the thread of its device. `places` holds the
    /// thread of each stage, in stage order.
    fn attach(&mut self, index: usize, handles: Vec<Box<dyn StageHandle + Send>>, places: &[usize]) {
        assert!(
            self.attached.is_none(),
            "engine {index} attaches its stages while another engine with stages holds the threads"
        );
        for (stage, handle) in handles.into_iter().enumerate() {
            let place = places[stage];
            self.senders[place]
                .send(StageWork::Attach { stage, handle })
                .unwrap_or_else(|_| panic!("device thread {place} stopped"));
        }
        self.attached = Some(index);
    }

    /// Take back the handles of the engine with stages after its unload, and wait until each is dropped. The drop frees
    /// the device memory of the stage before the next open (spec 3.3a).
    fn detach(&mut self) {
        for (stage, sender) in self.senders.iter().enumerate() {
            let (reply, answer) = mpsc::channel();
            sender
                .send(StageWork::Detach(reply))
                .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
            answer
                .recv()
                .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped during a detach"));
        }
        self.attached = None;
    }

    /// End the threads when the group stops. Each thread ends its quantum, then stops when its sender drops, and the
    /// engines and the handle that it holds drop with it.
    pub(super) fn end(self, group: usize, links: &GroupLinks) {
        let StageThreads { senders, handles, .. } = self;
        drop(senders);
        for (stage, handle) in handles.into_iter().enumerate() {
            let _ = handle.join();
            // A thread that ended beats no more, so its watch ends with it (spec 7.4).
            let _ = links.heartbeats.send(WatchMessage::StopStage { group, stage });
        }
    }
}

/// The handle of a stage thread. A device thread has none, so a stage call there is a defect.
fn own(handle: &mut Option<(usize, Box<dyn StageHandle + Send>)>, place: usize) -> &mut dyn StageHandle {
    handle
        .as_mut()
        .map(|(_, handle)| handle.as_mut() as &mut dyn StageHandle)
        .unwrap_or_else(|| panic!("a stage call reached device thread {place}, which holds no engine with stages"))
}

/// The loop of the device thread at `place`. It beats its own heartbeat between its quanta (spec 7.4).
fn stage_thread(
    group: usize,
    place: usize,
    work: Receiver<StageWork>,
    done: Sender<GroupMessage>,
    heartbeats: Sender<WatchMessage>,
    interval: Duration,
) {
    let stage = place;
    let beat = || WatchMessage::BeatStage {
        group,
        stage,
        at: Instant::now(),
    };
    // The stage of the engine with stages on this device, with its handle.
    let mut handle: Option<(usize, Box<dyn StageHandle + Send>)> = None;
    let _ = heartbeats.send(beat());
    // The engines with one device that this thread holds, at their places in the group (engine-abi "How a group loads
    // both"). They drop with the thread, which closes them.
    let mut hosted: Vec<Box<dyn Engine>> = Vec::new();
    loop {
        match work.recv_timeout(interval) {
            Ok(StageWork::Call(call)) => call(own(&mut handle, place)),
            Ok(StageWork::Attach {
                stage: held,
                handle: new,
            }) => {
                assert!(
                    handle.is_none(),
                    "device thread {place} takes the handle of stage {held} while it holds one"
                );
                handle = Some((held, new));
            }
            Ok(StageWork::Detach(reply)) => {
                // The drop closes the stage of the engine with stages on this device (engine-abi "`close` in Rust").
                // It frees device memory and can take seconds, so a helper beats for this thread (spec 7.4).
                let beat_guard = OpenBeat::stage(group, stage, heartbeats.clone(), interval);
                drop(handle.take());
                drop(beat_guard);
                if reply.send(()).is_err() {
                    return;
                }
            }
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
                let sent = match reply {
                    Answer::Wait(reply) => reply.send((job, answer)).is_ok(),
                    Answer::Swap => {
                        let step = SwapStep::Opened {
                            member: index,
                            opening: Box::new((job, answer)),
                        };
                        done.send(GroupMessage::SwapStep(step)).is_ok()
                    }
                };
                if !sent {
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
            Ok(StageWork::RunJoined {
                lead,
                kind,
                mut jobs,
                span,
            }) => {
                let started = Instant::now();
                let results = match &span {
                    Some(span) => span.in_scope(|| run_joined_bodies(kind, &mut jobs, &mut hosted)),
                    None => run_joined_bodies(kind, &mut jobs, &mut hosted),
                };
                let end = HostedJoinedDone {
                    lead,
                    kind,
                    jobs,
                    results,
                    ms: started.elapsed().as_millis() as u64,
                };
                if done.send(GroupMessage::HostedJoinedDone(end)).is_err() {
                    return;
                }
            }
            Ok(StageWork::Unload {
                index,
                name,
                store,
                reply,
            }) => {
                // The park and the close block this thread, so a helper beats for it (spec 7.4).
                let beat_guard = OpenBeat::stage(group, stage, heartbeats.clone(), interval);
                let mut report = ParkReport::default();
                park::park_engine(&name, hosted[index].as_mut(), store.as_ref(), &mut report);
                // The drop closes the engine (engine-abi "`close` in Rust").
                hosted[index] = Elsewhere::absent(index);
                drop(beat_guard);
                let sent = match reply {
                    Answer::Wait(reply) => reply.send(report).is_ok(),
                    Answer::Swap => done
                        .send(GroupMessage::SwapStep(SwapStep::Unloaded { member: index, report }))
                        .is_ok(),
                };
                if !sent {
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
                let held = handle.as_ref().map(|(held, _)| *held);
                assert_eq!(
                    held,
                    Some(named),
                    "the group sent the quantum of stage {named} to device thread {place}"
                );
                let started = Instant::now();
                let result = match &span {
                    Some(span) => span.in_scope(|| own(&mut handle, place).stage_pass(&segments, &mut residual)),
                    None => own(&mut handle, place).stage_pass(&segments, &mut residual),
                };
                let ms = started.elapsed().as_millis() as u64;
                let end = StageDone {
                    job,
                    block,
                    stage: named,
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
    /// The sender of the device thread of each stage, in stage order.
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
    /// The engine of the job, as its place in the group.
    engine: usize,
    kind: Kind,
    class: Class,
    request: &'static str,
    segments: Arc<Vec<Segment>>,
    residual: Option<Residual>,
}

/// A stage quantum that runs on a stage thread, with its telemetry and failure watch.
struct RunningStage {
    /// The engine of the job, as its place in the group. The quantum names it (spec 9.2, "The engine of a job").
    engine: usize,
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
    /// The names of the engines of the group, in profile order. A job span, its quanta and their metrics name the
    /// engine of the job (spec 9.2, "The engine of a job").
    names: Vec<String>,
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
    /// The engines on device threads, each with its device and the most jobs of a joined decode quantum: its
    /// `max_segments` when it has a feed rule, else 1 (engine-abi "LFM3", "Batched passes").
    decode_limits: BTreeMap<usize, (Device, u32)>,
    /// The other jobs of each joined decode quantum that runs, by its lead job.
    joined: BTreeMap<JobId, Vec<JobId>>,
    /// The jobs that a cancel counted. A cancelled job stays in `running_on` until its quantum ends, so a later
    /// cancel does not count it again.
    cancelled: std::collections::BTreeSet<JobId>,
}

impl StagedJobs {
    /// The jobs of a group. `limits` are the limits of its engine with stages; with none, the devices are those of
    /// the device threads, and no job has a block.
    fn new(group: u32, names: Vec<String>, limits: Option<&Limits>, devices: &[Device], machine: Machine) -> Self {
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
            names,
            machine,
            epoch: Instant::now(),
            scheduler: Some(StageScheduler::new(Facts {
                id: group,
                devices: devices.to_vec(),
                buffers,
                max_quantum_ms,
                decode_joins: BTreeMap::new(),
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
            decode_limits: BTreeMap::new(),
            joined: BTreeMap::new(),
            cancelled: std::collections::BTreeSet::new(),
        }
    }

    /// The device thread of `stage`. The scheduler's device of a stage is the place of its thread.
    fn place_of(&self, stage: usize) -> usize {
        self.scheduler().engine().devices[stage] as usize
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
                self.learn_decode_limit(spec.engine, device, engines);
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
            engine = self.names[spec.engine].as_str(),
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
                    let engine = self
                        .blocks
                        .get(&(job, block))
                        .expect("a later stage has its block")
                        .engine;
                    let span = self.start_quantum(&quantum, engine, wait_ms);
                    let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
                    let place = self.place_of(stage);
                    let entry = self.blocks.get_mut(&(job, block)).expect("a later stage has its block");
                    let residual = entry.residual.take().expect("the stage before wrote the residual");
                    self.running_stages.insert(
                        (job, block, stage),
                        RunningStage {
                            engine,
                            kind: entry.kind,
                            class: entry.class,
                            request: entry.request,
                            watch,
                            span: span.clone(),
                        },
                    );
                    threads.senders[place]
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

    /// Learn the decode limit of the engine at `engine` on `device`, and give the scheduler the limit of each device:
    /// the most of its engines (engine-abi "LFM3", "Batched passes"). A device thread runs alone any job that cannot
    /// join, so a limit that an unload left behind is safe.
    fn learn_decode_limit(&mut self, engine: usize, device: Device, engines: &[Box<dyn Engine>]) {
        let limits = engines[engine].limits();
        let limit = if crate::jobs::joins_decode(&limits) {
            limits.max_segments
        } else {
            1
        };
        self.decode_limits.insert(engine, (device, limit));
        let mut joins: BTreeMap<Device, u32> = BTreeMap::new();
        for (device, limit) in self.decode_limits.values() {
            if *limit > 1 {
                let most = joins.entry(*device).or_insert(1);
                *most = (*most).max(*limit);
            }
        }
        self.scheduler
            .as_mut()
            .expect("the scheduler is in place between steps")
            .set_decode_joins(joins);
    }

    /// Send a joined decode quantum to its device thread, with the body of each job. A job that a client cancelled
    /// before the quantum started has no body; it ends as cancelled when the quantum ends.
    fn send_joined(&mut self, quantum: &StageQuantum, device: Device, threads: &StageThreads) {
        let now_ms = self.now_ms();
        let ready = self.ready_since.get(&quantum.job).copied().unwrap_or(now_ms);
        let lead = self.engine_of_job(quantum.job);
        let span = self.start_quantum(quantum, lead, now_ms.saturating_sub(ready));
        let mut jobs = Vec::new();
        for job in std::iter::once(quantum.job).chain(quantum.joined.iter().copied()) {
            let engine = self.engine_of_job(job);
            let request = self.requests.get(&job).copied().unwrap_or("job");
            self.running_on.insert(
                job,
                RunningStage {
                    engine,
                    kind: quantum.kind,
                    class: quantum.class,
                    request,
                    watch: None,
                    span: if job == quantum.job { span.clone() } else { None },
                },
            );
            if let Some(body) = self.bodies.remove(&job) {
                jobs.push((job, engine, body));
            }
        }
        assert!(
            self.scheduler().engine().devices.contains(&device),
            "the scheduler started a quantum on device {device}, which has no device thread"
        );
        let place = device as usize;
        threads.senders[place]
            .send(StageWork::RunJoined {
                lead: quantum.job,
                kind: quantum.kind,
                jobs,
                span,
            })
            .unwrap_or_else(|_| panic!("device thread {place} stopped"));
        self.joined.insert(quantum.job, quantum.joined.clone());
    }

    /// The end of a joined decode quantum on a device thread. Each job ends or goes on by its own result, and the
    /// quantum ends once on its device.
    fn hosted_joined_done(&mut self, done: HostedJoinedDone, engines: &mut [Box<dyn Engine>], threads: &StageThreads) {
        let HostedJoinedDone {
            lead,
            kind,
            jobs,
            results,
            ms,
        } = done;
        let others = self.joined.remove(&lead).expect("the lead names its joined jobs");
        let all: Vec<JobId> = std::iter::once(lead).chain(others).collect();
        let device = *self.on.get(&lead).expect("a quantum on one device has its device");
        // A job that sent a body has a result. A job with no body was cancelled before the quantum started.
        let sent: std::collections::BTreeSet<JobId> = jobs.iter().map(|(job, ..)| *job).collect();
        for (job, _, body) in jobs {
            if self.scheduler().jobs().iter().any(|held| held.id == job) {
                self.bodies.insert(job, body);
            } else {
                let mut body = body;
                body.ended(JobState::Cancelled);
            }
        }
        let mut lead_running = None;
        for &job in &all {
            let running = self.running_on.remove(&job);
            let ran = match results.iter().find(|(id, _)| *id == job) {
                Some((_, ran)) => ran.clone(),
                None => {
                    assert!(
                        !sent.contains(&job),
                        "job {job} sent its body to the joined quantum, and the device thread gave no result for it"
                    );
                    Ran::Cancelled
                }
            };
            if !self.bodies.contains_key(&job) {
                // A cancelled job: its quantum ends with it.
            } else if matches!(ran, Ran::Failed) {
                if let Some(running) = &running {
                    self.log_failure(running.engine, kind, running.request, None);
                }
                self.scheduler
                    .as_mut()
                    .expect("the scheduler is in place between steps")
                    .fail_job(job)
                    .expect("a job of the joined quantum runs");
            } else {
                self.apply(job, ran);
            }
            if job == lead {
                lead_running = running;
            }
        }
        if let Some(running) = lead_running {
            self.end_quantum(running.engine, kind, running.class, ms, Completion::Done, running.span);
        }
        let now_ms = self.now_ms();
        for &job in &all {
            self.ready_since.insert(job, now_ms);
        }
        self.event(StageEvent::Complete {
            now_ms,
            device,
            duration_ms: ms,
            outcome: Completion::Done,
        });
        self.forget_ended_jobs();
        self.pump(engines, threads);
    }

    /// Send a quantum of an engine on a stage thread to that thread, with the body of its job.
    fn send_on(
        &mut self,
        quantum: &StageQuantum,
        device: Device,
        engines: &mut [Box<dyn Engine>],
        threads: &StageThreads,
    ) {
        if !quantum.joined.is_empty() {
            return self.send_joined(quantum, device, threads);
        }
        let job = quantum.job;
        let Some(body) = self.bodies.remove(&job) else {
            // The job was cancelled before its quantum started, so the quantum ends here at once.
            return self.run_whole(quantum, engines);
        };
        let now_ms = self.now_ms();
        let ready = self.ready_since.get(&job).copied().unwrap_or(now_ms);
        let engine = self.engine_of_job(job);
        let span = self.start_quantum(quantum, engine, now_ms.saturating_sub(ready));
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        let request = self.requests.get(&job).copied().unwrap_or("job");
        self.running_on.insert(
            job,
            RunningStage {
                engine,
                kind: quantum.kind,
                class: quantum.class,
                request,
                watch,
                span: span.clone(),
            },
        );
        // A device of the scheduler is the place of its device thread.
        assert!(
            self.scheduler().engine().devices.contains(&device),
            "the scheduler started a quantum on device {device}, which has no device thread"
        );
        let place = device as usize;
        threads.senders[place]
            .send(StageWork::Run {
                job,
                kind: quantum.kind,
                body,
                span,
            })
            .unwrap_or_else(|_| panic!("device thread {place} stopped"));
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
        // A refused job ends as failed, and it is not a failed quantum (spec 9.6).
        let refused = matches!(ran, Ran::Refused);
        let outcome = if self.bodies.contains_key(&job) {
            self.apply(job, ran)
        } else {
            Completion::Done
        };
        if let Some(running) = running {
            if refused {
                self.log_refusal(running.engine, kind, running.request);
            } else if outcome == Completion::Failed {
                self.log_failure(running.engine, kind, running.request, running.watch);
            }
            self.end_quantum(running.engine, kind, running.class, ms, outcome, running.span);
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
        let engine = self.engine_of_job(quantum.job);
        let span = self.start_quantum(quantum, engine, wait_ms);
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
        match ran {
            Ran::Failed => self.log_failure(engine, quantum.kind, request, watch),
            Ran::Refused => self.log_refusal(engine, quantum.kind, request),
            _ => {}
        }
        let outcome = self.apply(quantum.job, ran);
        self.end_quantum(engine, quantum.kind, quantum.class, duration_ms, outcome, span);
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
        let engine = self.engine_of_job(job);
        let begun = match self.bodies.get_mut(&job) {
            Some(body) => body.begin_pass(quantum.kind, engines),
            None => Err(Ran::Cancelled),
        };
        let segments = match begun {
            Ok(segments) => Arc::new(segments),
            Err(ran) => {
                let span = self.start_quantum(quantum, engine, wait_ms);
                let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
                if matches!(ran, Ran::Failed) {
                    self.log_failure(engine, quantum.kind, request, watch);
                }
                let outcome = self.apply(job, ran);
                self.end_quantum(engine, quantum.kind, quantum.class, 0, outcome, span);
                self.event(StageEvent::Complete {
                    now_ms: self.now_ms(),
                    device: quantum.devices[0],
                    duration_ms: 0,
                    outcome,
                });
                return;
            }
        };
        let span = self.start_quantum(quantum, engine, wait_ms);
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        // The scheduler starts a block only when a buffer is free, so the pool is never empty here.
        let residual = self.free.pop().expect("a block that starts has a free residual buffer");
        self.blocks.insert(
            (job, block),
            Block {
                engine,
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
                engine,
                kind: quantum.kind,
                class: quantum.class,
                request,
                watch,
                span: span.clone(),
            },
        );
        self.order.entry(job).or_default().push_back(block);
        threads.senders[self.place_of(0)]
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
                self.log_failure(running.engine, running.kind, running.request, running.watch);
            }
            self.end_quantum(running.engine, running.kind, running.class, ms, outcome, running.span);
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

    /// The engine of a job that has not ended, as its place in the group.
    fn engine_of_job(&self, job: JobId) -> usize {
        *self
            .engine_of
            .get(&job)
            .unwrap_or_else(|| panic!("job {job} started a quantum, but the group holds no engine for it"))
    }

    /// Record the wait of a quantum of a job of the engine at `engine`, and open its span (spec 9.2).
    fn start_quantum(&self, quantum: &StageQuantum, engine: usize, wait_ms: u64) -> Option<Span> {
        let engine = self.names[engine].as_str();
        let labels = QuantumLabels {
            engine,
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
                    engine,
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

    fn end_quantum(
        &self,
        engine: usize,
        kind: Kind,
        class: Class,
        duration_ms: u64,
        outcome: Completion,
        span: Option<Span>,
    ) {
        let labels = QuantumLabels {
            engine: &self.names[engine],
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
                Completion::Refused => {
                    span.record("outcome", "refused");
                }
            }
        }
    }

    /// Log a refused job at the level `info` (spec 9.6).
    fn log_refusal(&self, engine: usize, kind: Kind, request: &'static str) {
        evidence::refused(&format!(
            "{} quantum of a {} request on engine {}",
            kind_label(kind),
            request,
            self.names[engine]
        ));
    }

    fn log_failure(&self, engine: usize, kind: Kind, request: &'static str, watch: Option<Watch>) {
        let failure = format!(
            "{} quantum of a {} request on engine {}",
            kind_label(kind),
            request,
            self.names[engine]
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
            Ran::Refused => Completion::Refused,
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
            self.cancelled.remove(&id);
            if let Some(mut body) = self.bodies.remove(&id) {
                body.ended(state);
            }
        }
    }

    /// Cancel every job (spec 10.3, 5.9). The stage quanta that run go to their end. It gives the number of jobs
    /// that no cancel counted before.
    fn cancel_all(&mut self) -> usize {
        // A job whose quantum runs on a stage or device thread has its body there, so it is in `running_on`.
        let mut ids: Vec<JobId> = self.bodies.keys().chain(self.running_on.keys()).copied().collect();
        ids.sort_unstable();
        ids.dedup();
        ids.retain(|id| !self.cancelled.contains(id));
        self.cancelled.extend(ids.iter().copied());
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

/// The place of the device thread of the one device of `member`, or `None` when the member cannot run on a device
/// thread.
fn stage_of(member: &Member, machine: &[DeviceId]) -> Option<usize> {
    if member.staged() || member.job.devices.len() != 1 {
        return None;
    }
    let id = member.job.devices[0].id;
    machine.iter().position(|device| *device == id)
}

/// The place of the device thread of each stage, in stage order. Stage `s` runs on group index `stages[s].device`,
/// which is machine device `own[stages[s].device]` (spec 2.4). `machine` gives the place of each machine device.
fn places_of_stages(stages: &[nagaya_abi::Stage], own: &[DeviceId], machine: &[DeviceId]) -> Vec<usize> {
    stages
        .iter()
        .map(|stage| {
            let id = own[stage.device as usize];
            machine
                .iter()
                .position(|device| *device == id)
                .unwrap_or_else(|| panic!("stage device {id} is not a device of the group {machine:?}"))
        })
        .collect()
}

/// Send the unload of the engine at `member` to the thread of its stage, and mark the engine draining (spec 3.3a).
fn send_unload(
    member: usize,
    stage: usize,
    members: &[Member],
    threads: &StageThreads,
    links: &GroupLinks,
    reply: Answer<ParkReport>,
) {
    let name = members[member].name().to_owned();
    links.states.record(&name, EngineState::Draining);
    threads.senders[stage]
        .send(StageWork::Unload {
            index: member,
            name,
            store: links.store.clone(),
            reply,
        })
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
}

/// Mark the engine at `member` unloaded after its thread parked and closed it. A part that cannot be parked is a
/// crash (spec 8.5).
fn end_unload(
    member: usize,
    stage: usize,
    report: ParkReport,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    links: &GroupLinks,
) {
    let name = members[member].name().to_owned();
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

/// Park and close the engine at `member` on the thread of its stage, and wait for the end (spec 3.3a).
fn unload_hosted(
    member: usize,
    stage: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    threads: &StageThreads,
    links: &GroupLinks,
) {
    let (reply, answer) = mpsc::channel();
    send_unload(member, stage, members, threads, links, Answer::Wait(reply));
    let report = answer
        .recv()
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped during an unload"));
    end_unload(member, stage, report, members, engines, links);
}

/// Send the open of the engine at `member` to the thread of `stage`, and mark the engine loading. The job of the
/// engine goes with the open, and a shell holds its place until the answer.
fn send_open(
    member: usize,
    stage: usize,
    members: &mut [Member],
    threads: &StageThreads,
    links: &GroupLinks,
    reply: Answer<Opening>,
) {
    links.states.record(members[member].name(), EngineState::Loading);
    let shell = members[member].job.shell();
    let job = Box::new(std::mem::replace(&mut members[member].job, shell));
    threads.senders[stage]
        .send(StageWork::Open {
            index: member,
            count: members.len(),
            job,
            fd_store: links.fd_store.clone(),
            reply,
        })
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped"));
}

/// Take back the job of the engine at `member` after its open, and mark the engine ready when it opened.
fn end_open(
    member: usize,
    (job, opened): Opening,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    links: &GroupLinks,
) -> Result<OpenedEngine, OpenFailure> {
    members[member].job = *job;
    let opened = opened?;
    let engine = members[member].name().to_owned();
    members[member].loaded = true;
    engines[member] = Elsewhere::hosted(&members[member]);
    links.states.opened(&engine, &opened.identity);
    links.states.record(&engine, EngineState::Ready);
    Ok(opened)
}

/// Open the engine at `member` on the thread of `stage`, and wait for the end (engine-abi "How a group loads both").
fn open_hosted(
    member: usize,
    stage: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    threads: &StageThreads,
    links: &GroupLinks,
) -> Result<OpenedEngine, OpenFailure> {
    let (reply, answer) = mpsc::channel();
    send_open(member, stage, members, threads, links, Answer::Wait(reply));
    let opening = answer
        .recv()
        .unwrap_or_else(|_| panic!("the thread of stage {stage} stopped during an open"));
    end_open(member, opening, members, engines, links)
}

/// A swap that runs on the stage or device threads while the group thread serves (spec 3.3a, "Single-card swaps").
/// It keeps the engine with stages, when one is loaded (engine-abi "How a group loads both").
///
/// The ledger took the swap when the group planned it. Each unload goes to the thread of its device. When every
/// unload ended, the group charges the swap in the ledger, on its one thread. Then each open goes to the thread of its device. A failure is a crash (spec 8.5). The group starts
/// no other swap until this one ends, so the ledger sees the swaps in order.
struct Swapping {
    group: usize,
    plan: Plan,
    /// The stage of each engine with one device on a device of a stage.
    stages: BTreeMap<usize, usize>,
    /// The unloads that run, as the stage of each engine.
    unloads: BTreeMap<usize, usize>,
    /// The opens that run, as the stage of each engine. It is empty until the unloads end.
    opens: BTreeMap<usize, usize>,
    /// The engines that opened.
    loaded: BTreeMap<usize, OpenedEngine>,
    started: Instant,
    opening: Option<Instant>,
    span: Span,
}

impl Swapping {
    /// Start the swap of `plan`: send each unload of an engine on a stage thread. An engine of the plan on no stage
    /// thread unloaded on the group thread before this serve loop began (`Entry::Swap`).
    #[allow(clippy::too_many_arguments)]
    fn start(
        group: usize,
        plan: Plan,
        stages: &BTreeMap<usize, usize>,
        members: &mut [Member],
        hosted: &Hosted,
        threads: &StageThreads,
        links: &GroupLinks,
    ) -> Option<Self> {
        let names = |list: &[usize]| {
            list.iter()
                .map(|index| members[*index].name().to_owned())
                .collect::<Vec<_>>()
                .join(",")
        };
        let span = tracing::info_span!(
            "nagaya.swap",
            group,
            unload = names(&plan.unload).as_str(),
            load = names(&plan.load).as_str(),
            unload_ms = tracing::field::Empty,
            load_ms = tracing::field::Empty,
        );
        let mut unloads = BTreeMap::new();
        for &member in &plan.unload {
            match hosted.get(&member) {
                Some(&stage) => {
                    send_unload(member, stage, members, threads, links, Answer::Swap);
                    unloads.insert(member, stage);
                }
                None => assert!(
                    !members[member].loaded,
                    "a swap on the stage threads unloads engine `{}`, which is on no stage thread",
                    members[member].name()
                ),
            }
        }
        let swapping = Self {
            group,
            plan,
            stages: stages.clone(),
            unloads,
            opens: BTreeMap::new(),
            loaded: BTreeMap::new(),
            started: Instant::now(),
            opening: None,
            span,
        };
        swapping.advance(members, threads, links)
    }

    /// Take the end of one step from a stage thread, and give the swap when it has more steps.
    fn step(
        mut self,
        step: SwapStep,
        hosted: &mut Hosted,
        members: &mut [Member],
        engines: &mut [Box<dyn Engine>],
        threads: &StageThreads,
        links: &GroupLinks,
    ) -> Option<Self> {
        match step {
            SwapStep::Unloaded { member, report } => {
                let stage = self.unloads.remove(&member).unwrap_or_else(|| {
                    panic!("a stage thread unloaded engine {member}, which the swap does not unload")
                });
                hosted.remove(&member);
                end_unload(member, stage, report, members, engines, links);
            }
            SwapStep::Opened { member, opening } => {
                let stage = self
                    .opens
                    .remove(&member)
                    .unwrap_or_else(|| panic!("a stage thread opened engine {member}, which the swap does not load"));
                let engine = members[member].name().to_owned();
                let opened = end_open(member, *opening, members, engines, links)
                    .unwrap_or_else(|(_, message)| panic!("the swap cannot open engine `{engine}`: {message}"));
                hosted.insert(member, stage);
                self.loaded.insert(member, opened);
            }
        }
        self.advance(members, threads, links)
    }

    /// Start the opens when the unloads ended, and end the swap when the opens ended.
    fn advance(mut self, members: &mut [Member], threads: &StageThreads, links: &GroupLinks) -> Option<Self> {
        if !self.unloads.is_empty() {
            return Some(self);
        }
        if self.opening.is_none() {
            self.span.record("unload_ms", self.started.elapsed().as_millis() as u64);
            self.admit(members, links);
            self.opening = Some(Instant::now());
            for &member in &self.plan.load {
                let stage = *self.stages.get(&member).unwrap_or_else(|| {
                    panic!(
                        "a swap on the stage threads loads engine `{}`, which has no stage thread",
                        members[member].name()
                    )
                });
                send_open(member, stage, members, threads, links, Answer::Swap);
                self.opens.insert(member, stage);
            }
        }
        if !self.opens.is_empty() {
            return Some(self);
        }
        self.end(members, links);
        None
    }

    /// Charge the swap in the ledger. A refusal is a crash (spec 8.5): the ledger took the plan, and each later charge
    /// had to fit after it.
    fn admit(&self, members: &[Member], links: &GroupLinks) {
        let name = |index: &usize| members[*index].name().to_owned();
        let (reply, answer) = mpsc::channel();
        let loading: Vec<String> = self.plan.load.iter().map(name).collect();
        links
            .ledger
            .send(LedgerMessage::Swap {
                load: loading.clone(),
                unload: self.plan.unload.iter().map(name).collect(),
                reply,
            })
            .expect("the ledger thread stopped");
        if let Err(error) = answer.recv().expect("the ledger thread stopped") {
            panic!("the ledger refused the swap to {}: {error}", loading.join(", "));
        }
    }

    /// Report the swap, with the engines that opened in the order of the plan.
    fn end(mut self, members: &[Member], links: &GroupLinks) {
        let opening = self.opening.expect("a swap ends after its opens");
        self.span.record("load_ms", opening.elapsed().as_millis() as u64);
        let name = |index: &usize| members[*index].name().to_owned();
        let unloaded: Vec<String> = self.plan.unload.iter().map(name).collect();
        let loading: Vec<String> = self.plan.load.iter().map(name).collect();
        let loaded: Vec<OpenedEngine> = self
            .plan
            .load
            .iter()
            .map(|member| self.loaded.remove(member).expect("each engine of the swap opened"))
            .collect();
        tracing::info!(
            parent: &self.span,
            "swap on the threads of group {}: unloaded [{}], loaded [{}] in {} ms",
            self.group,
            unloaded.join(", "),
            loading.join(", "),
            self.started.elapsed().as_millis()
        );
        let _ = links.events.send(Event::Swapped {
            group: self.group,
            unloaded,
            loaded,
        });
    }
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
/// or `None`. `threads` are the device threads of the group, which outlive the loop.
///
/// An engine with one device loads beside the engine with stages when the ledger admits both. It opens on the thread
/// of its device, and its quanta run there (engine-abi "How a group loads both"). A load or an unload plans a swap
/// (spec 3.3b). A swap of engines with one device runs on their threads while the loop serves, and the loop plans no
/// other swap until it ends (spec 3.3a, "Single-card swaps"). A swap that loads or unloads the engine with stages, or
/// an engine of more devices, waits until the group holds no job. Then the loop unloads the engines that the swap
/// names and gives the swap to the group thread. The other engines stay on their threads (spec 3.3a, "No moves").
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
    threads: &mut StageThreads,
) -> Exit {
    let names: Vec<String> = members.iter().map(|member| member.name().to_owned()).collect();
    let machine = group_devices(members);
    let order: Vec<DeviceId> = (0..machine.len() as DeviceId).collect();
    // The scheduler's device of each stage is the place of its device thread. With no engine with stages, the
    // devices are the places in order.
    let (limits, places) = match index {
        Some(index) => {
            let limits = members[index].job.limits.clone();
            let own: Vec<DeviceId> = members[index].job.devices.iter().map(|device| device.id).collect();
            let mut sorted = own.clone();
            sorted.sort_unstable();
            let stages: Vec<DeviceId> = limits.stages.iter().map(|stage| stage.device).collect();
            // A stage runs on the device thread of its device, so the stages cover the devices of the group, each
            // once. Admission refuses another shape (spec 3.2).
            assert!(
                sorted == machine && stages == order,
                "engine `{}` has stages on group indexes {stages:?} of machine devices {own:?}, and its group has \
                 machine devices {machine:?}: an engine with stages spans its whole group, one stage on each device",
                names[index]
            );
            let places = places_of_stages(&limits.stages, &own, &machine);
            (Some(limits), places)
        }
        None => (None, (0..machine.len()).collect()),
    };
    let devices: Vec<DeviceId> = places.iter().map(|place| *place as DeviceId).collect();
    if let (Some(index), Some(limits)) = (index, &limits)
        && threads.attached != Some(index)
    {
        let handles = std::mem::take(&mut members[index].handles);
        assert_eq!(
            handles.len(),
            devices.len(),
            "the open of engine `{}` gave a handle for each stage",
            names[index]
        );
        threads.attach(index, handles, &places);
        let inner = std::mem::replace(&mut engines[index], Unloaded::of(&members[index]));
        engines[index] = Box::new(StagedEngine {
            inner,
            limits: limits.clone(),
            senders: places.iter().map(|place| threads.senders[*place].clone()).collect(),
        });
    }
    // The device thread of each engine with one device.
    let stages: BTreeMap<usize, usize> = (0..members.len())
        .filter_map(|member| stage_of(&members[member], &machine).map(|stage| (member, stage)))
        .collect();
    let mut hosted: Hosted = std::mem::take(&mut threads.hosted);
    // The swap that runs on the stage or device threads, while this thread serves (spec 3.3a, "Single-card swaps").
    let mut swapping: Option<Swapping> = None;
    match entry {
        Entry::Serve => {}
        Entry::Swap(plan) => {
            swapping = Swapping::start(group, plan, &stages, members, &hosted, threads, links);
        }
        Entry::Start { open, mut opened } => {
            let beat = OpenBeat::start(group, links.heartbeats.clone(), interval);
            for member in open {
                let stage = *stages
                    .get(&member)
                    .unwrap_or_else(|| panic!("engine `{}` has no device thread", names[member]));
                match open_hosted(member, stage, members, engines, threads, links) {
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
                        threads.hosted = hosted;
                        return Exit::Stop;
                    }
                }
            }
            drop(beat);
            let _ = links.events.send(Event::Opened { group, engines: opened });
        }
    }
    let mut jobs = StagedJobs::new(
        group as u32,
        names.clone(),
        limits.as_ref(),
        &devices,
        links.machine.clone(),
    );
    let mut pending: Option<Plan> = None;
    // A swap on the group thread waits for the jobs in flight. New jobs of the engines that stay wait here, so a steady
    // stream cannot hold the swap off (spec 3.3a, "The wait before an unload"). They go back to the group after it.
    let mut held: Vec<(JobSpec, Span)> = Vec::new();
    let mut hold = false;
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
                    Some(EngineState::Ready) if hold => held.push((spec, parent)),
                    Some(EngineState::Ready) => {
                        let place = match hosted.get(&spec.engine) {
                            Some(place) => Place::On(*place as DeviceId),
                            None if Some(spec.engine) == index => Place::Stages,
                            None => Place::Group,
                        };
                        jobs.add(spec, parent, place, engines, threads);
                    }
                    state => super::refuse(spec, &engine, state),
                }
            }
            Ok(GroupMessage::Ask(ask)) => loads::take_ask(ask, asks, &links.ledger),
            Ok(GroupMessage::StageDone(done)) => jobs.stage_done(done, engines, threads),
            Ok(GroupMessage::HostedDone(done)) => jobs.hosted_done(done, engines, threads),
            Ok(GroupMessage::HostedJoinedDone(done)) => jobs.hosted_joined_done(done, engines, threads),
            Ok(GroupMessage::SwapStep(step)) => {
                let running = swapping.take().expect("a swap step comes from a swap that runs");
                swapping = running.step(step, &mut hosted, members, engines, threads, links);
            }
            Ok(GroupMessage::Park(reply)) => {
                cancelled += jobs.cancel_all() + held.len();
                held.clear();
                // An ask that waits gets no reply. Its caller hears that the group stopped.
                asks.clear();
                // A swap that runs ends first, so each engine is on one thread for the park. An open can take seconds,
                // so this thread beats while it waits (spec 7.4).
                while let Some(running) = swapping.take() {
                    swapping = Some(running);
                    let _ = links.heartbeats.send(beat());
                    match links.messages.recv_timeout(interval) {
                        Ok(GroupMessage::SwapStep(step)) => {
                            let running = swapping.take().expect("a swap runs");
                            swapping = running.step(step, &mut hosted, members, engines, threads, links);
                        }
                        // The process stops, so a quantum or a job that comes now is cancelled, as at a stop.
                        Ok(GroupMessage::Quantum(quantum)) => {
                            quantum.cancel();
                            cancelled += 1;
                        }
                        Ok(GroupMessage::Job(..)) => cancelled += 1,
                        // An ask gets no reply, and the ends of the cancelled jobs change nothing.
                        Ok(_) | Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
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
                        reply: Answer::Wait(answer),
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
        if !serving && !held.is_empty() {
            cancelled += held.len();
            held.clear();
        }
        // The engines on the group thread: the engine with stages, or an engine of more than one device.
        let local: Vec<usize> = (0..members.len())
            .filter(|member| members[*member].loaded && !hosted.contains_key(member))
            .collect();
        // The group plans no swap while one runs, so the ledger sees the swaps in order (spec 3.3a).
        let free = swapping.is_none();
        if serving && free && pending.is_none() {
            pending = loads::next_plan(asks, members, &links.ledger, &links.states);
        }
        let beside = pending.as_ref().is_some_and(|plan| {
            local.iter().all(|member| !plan.unload.contains(member))
                && plan.unload.iter().all(|member| hosted.contains_key(member))
                && plan.load.iter().all(|member| stages.contains_key(member))
        });
        if serving && free && beside {
            let plan = pending.as_ref().expect("a pending swap");
            if plan.unload.iter().all(|member| !jobs.holds(*member)) {
                let plan = pending.take().expect("a pending swap");
                swapping = Swapping::start(group, plan, &stages, members, &hosted, threads, links);
            }
        } else if serving && free && pending.is_some() && jobs.idle() {
            // The engines that the plan names unload here, on their threads, and the group thread runs the rest of the
            // swap. Each other engine stays loaded on its device thread (spec 3.3a, "No moves").
            let plan = pending.take().expect("a pending swap");
            for &member in &plan.unload {
                if let Some(stage) = hosted.remove(&member) {
                    unload_hosted(member, stage, members, engines, threads, links);
                }
            }
            for &member in &plan.unload {
                if !local.contains(&member) {
                    continue;
                }
                in_place::unload(
                    member,
                    members,
                    engines,
                    links.store.as_ref(),
                    links.fd_store.as_ref(),
                    &links.states,
                );
                if Some(member) == index {
                    // The stages of the engine close on their threads before the next open (spec 3.3a). This thread
                    // waits for each, so a helper beats for it (spec 7.4).
                    let beat_guard = OpenBeat::start(group, links.heartbeats.clone(), interval);
                    threads.detach();
                    drop(beat_guard);
                }
            }
            // The held jobs wait in the group's own channel for the end of the swap.
            for (spec, parent) in held.drain(..) {
                let _ = links.own.send(GroupMessage::Job(spec, parent));
            }
            break Exit::Swap(plan);
        }
        hold = serving && free && pending.is_some() && !beside;
        if links.heartbeats.send(beat()).is_err() {
            break Exit::Stop;
        }
    };
    threads.hosted = hosted;
    exit
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A body that the test never runs: its quantum stays on the device thread.
    struct Held;

    impl JobBody for Held {
        fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
            unreachable!("the quantum of this job stays on its device thread")
        }

        fn unavailable(&mut self, engine: &str, state: crate::notify::EngineState) {
            panic!(
                "engine `{engine}` is {}, and this test does not ask for it",
                state.as_str()
            );
        }
    }

    /// The jobs of a group of one device, with the thread of that device as a channel that the test holds.
    fn jobs_on_one_device() -> (StagedJobs, StageThreads, Receiver<StageWork>) {
        let jobs = StagedJobs::new(0, vec!["left".to_owned()], None, &[0], Machine::at("/nonexistent", 1));
        let (sender, work) = mpsc::channel();
        let threads = StageThreads {
            senders: vec![sender],
            handles: Vec::new(),
            hosted: Hosted::new(),
            attached: None,
        };
        (jobs, threads, work)
    }

    /// A stop can cancel the jobs of a group twice: once when the group sees that the process refuses work, and again
    /// at the park. A job whose quantum runs on a device thread counts once (spec 10.3).
    #[test]
    fn a_second_cancel_does_not_count_a_job_whose_quantum_runs_on_a_device_thread() {
        let (mut jobs, threads, work) = jobs_on_one_device();
        let spec = JobSpec {
            class: Class::Interactive,
            priority: 0,
            deadline: None,
            first: Kind::Decode,
            engine: 0,
            body: Box::new(Held),
        };
        // The group thread holds a place for each engine of the group; this one has the limits of a plain fake.
        let mut engines: Vec<Box<dyn Engine>> =
            vec![Elsewhere::hosted_with(nagaya_abi::fake::FakeAlways::new().limits())];
        jobs.add(spec, Span::none(), Place::On(0), &mut engines, &threads);
        assert!(
            matches!(work.try_recv(), Ok(StageWork::Run { job: 1, .. })),
            "the quantum went to the device thread"
        );
        assert_eq!(jobs.cancel_all(), 1);
        assert!(
            !jobs.idle(),
            "the quantum still runs, so the group cancels again on its next pass"
        );
        assert_eq!(jobs.cancel_all(), 0);
    }
}
