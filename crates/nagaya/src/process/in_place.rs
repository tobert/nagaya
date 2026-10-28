//! The swap in place on a group thread (spec 3.3a).
//!
//! A group keeps every engine of its profile entries, in profile order. An engine that is not loaded has an
//! [`Unloaded`] place. A load or an unload of the process plans a swap with the ledger thread (spec 3.3b). When the
//! engines that must unload hold no job, the group parks and closes them, admits the new set of loaded engines, and
//! opens the engines of the plan.

use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};

use nagaya_abi::{Capability, Engine, Identity, Limits, PassError, PassOut, Segment, StageHandle, Tap};
use nagaya_ledger::LedgerError;
use nagaya_telemetry::{EngineState, instruments};

use super::loads::EngineStates;

use super::{Event, GroupJob, LedgerMessage, OpenBeat, OpenedEngine, ParkReport, WatchMessage, handoff, park};
use crate::admission::{AdmitError, OpenErrorKind};
use crate::arena::FdStore;
use crate::park::StoreMessage;
use crate::profile::{DEFAULT_IDLE_UNLOAD_S, Residency};

/// One engine of a group: what opens it, and whether it is loaded.
pub(super) struct Member {
    pub(super) job: GroupJob,
    pub(super) loaded: bool,
    /// The stage handles of a loaded engine with stages, from its open until its stage threads take them.
    pub(super) handles: Vec<Box<dyn StageHandle + Send>>,
    /// The last time that the engine held a job, or its load. The idle time of an on-demand engine counts from it.
    pub(super) last_busy: Instant,
}

impl Member {
    pub(super) fn name(&self) -> &str {
        &self.job.placement.name
    }

    /// True when the engine runs as stages on stage threads (engine-abi D1).
    pub(super) fn staged(&self) -> bool {
        self.job.limits.capabilities.contains(&Capability::Stages)
    }

    /// True for an engine that the profile makes resident (spec 3.3a).
    pub(super) fn resident(&self) -> bool {
        self.job.placement.residency == Residency::Resident
    }

    /// The time with no job after which an on-demand engine unloads (spec 3.3a).
    pub(super) fn idle_time(&self) -> Duration {
        Duration::from_secs(self.job.placement.idle_unload_s.unwrap_or(DEFAULT_IDLE_UNLOAD_S))
    }
}

/// The place of an engine that is not loaded (spec 3.3a).
///
/// A job runs only on a loaded engine, so a call here other than `limits` is a defect of the group, and it panics.
pub(super) struct Unloaded {
    name: String,
    limits: Limits,
}

impl Unloaded {
    pub(super) fn of(member: &Member) -> Box<dyn Engine> {
        Box::new(Self {
            name: member.name().to_owned(),
            limits: member.job.limits.clone(),
        })
    }
}

impl Engine for Unloaded {
    fn identity(&self) -> Identity {
        panic!("the identity of engine `{}` was read while it is not loaded", self.name)
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        panic!("a pass reached engine `{}` while it is not loaded", self.name)
    }

    fn last_error(&self) -> Option<String> {
        panic!(
            "the last error of engine `{}` was read while it is not loaded",
            self.name
        )
    }
}

/// The engines of one swap, by their place in the group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub(super) load: Vec<usize>,
    pub(super) unload: Vec<usize>,
}

/// How a serve loop of a group ended.
pub(super) enum Exit {
    /// The group stops.
    Stop,
    /// The group swaps by the plan. A serve loop of stages unloaded its own engine before it ended.
    Swap(Plan),
}

/// Ask the ledger thread whether a swap fits. It changes nothing.
fn check(ledger: &Sender<LedgerMessage>, load: Vec<String>, unload: Vec<String>) -> Result<(), AdmitError> {
    let (reply, answer) = mpsc::channel();
    ledger
        .send(LedgerMessage::CheckSwap { load, unload, reply })
        .expect("the ledger thread stopped");
    answer.recv().expect("the ledger thread stopped")
}

/// Plan the swap that loads `target` (spec 3.3a).
///
/// An engine with stages loads after the group unloads every other engine (engine-abi "How a group loads both").
/// Another engine first tries to load beside the loaded engines. Then the group unloads its engines with stages, and
/// then its other loaded engines, the last in profile order first, until the target fits. Admission checked that the
/// target fits with its whole group unloaded, so a plan that still does not fit is a defect.
pub(super) fn plan(target: usize, members: &[Member], ledger: &Sender<LedgerMessage>) -> Plan {
    let loaded: Vec<usize> = (0..members.len()).filter(|index| members[*index].loaded).collect();
    let mut unload: Vec<usize> = if members[target].staged() {
        loaded.clone()
    } else {
        Vec::new()
    };
    let staged = loaded.iter().copied().filter(|index| members[*index].staged());
    let others = loaded.iter().rev().copied().filter(|index| !members[*index].staged());
    let mut candidates = staged
        .chain(others)
        .filter(|index| !unload.contains(index))
        .collect::<Vec<_>>()
        .into_iter();
    let name = |index: &usize| members[*index].name().to_owned();
    loop {
        match check(ledger, vec![name(&target)], unload.iter().map(name).collect()) {
            Ok(()) => {
                return Plan {
                    load: vec![target],
                    unload,
                };
            }
            Err(AdmitError::Ledger(LedgerError::Refused(refusal))) => match candidates.next() {
                Some(index) => unload.push(index),
                None => panic!(
                    "engine `{}` does not fit with every other engine of its group unloaded, and admission let it \
                     in: {refusal:?}",
                    name(&target)
                ),
            },
            Err(error) => panic!(
                "the ledger cannot check the swap to engine `{}`: {error}",
                name(&target)
            ),
        }
    }
}

/// The swap back of an on-demand engine that held no job for its idle time, or `None` (spec 3.3a).
pub(super) fn idle_plan(members: &[Member], now: Instant, ledger: &Sender<LedgerMessage>) -> Option<Plan> {
    let idle = (0..members.len()).find(|index| {
        let member = &members[*index];
        member.loaded && !member.resident() && now.duration_since(member.last_busy) >= member.idle_time()
    })?;
    Some(back_plan(idle, members, ledger))
}

/// The swap that unloads the on-demand engine at `idle`, after its idle time or by an unload (spec 3.3a, 3.3b).
///
/// The swap unloads that engine and loads each resident engine of the group that is not loaded. When the resident
/// engines do not fit beside the engines that stay loaded, or an engine with stages would not load alone, the swap only
/// unloads.
pub(super) fn back_plan(idle: usize, members: &[Member], ledger: &Sender<LedgerMessage>) -> Plan {
    let residents: Vec<usize> = (0..members.len())
        .filter(|index| !members[*index].loaded && members[*index].resident())
        .collect();
    let others_stay = (0..members.len()).any(|index| index != idle && members[index].loaded);
    let alone = !residents.iter().any(|index| members[*index].staged()) || (!others_stay && residents.len() == 1);
    let name = |index: &usize| members[*index].name().to_owned();
    let fits = alone
        && (residents.is_empty() || check(ledger, residents.iter().map(name).collect(), vec![name(&idle)]).is_ok());
    Plan {
        load: if fits { residents } else { Vec::new() },
        unload: vec![idle],
    }
}

/// The ways out of the group thread that a swap uses.
pub(super) struct SwapLinks<'a> {
    pub(super) events: &'a Sender<Event>,
    pub(super) heartbeats: &'a Sender<WatchMessage>,
    pub(super) ledger: &'a Sender<LedgerMessage>,
    pub(super) store: Option<&'a Sender<StoreMessage>>,
    pub(super) fd_store: Option<&'a FdStore>,
    pub(super) states: &'a EngineStates,
}

/// Run a swap on the group thread (spec 3.3a). A failure is a crash, because the group no longer has a known set of
/// loaded engines (spec 8.5).
pub(super) fn swap(
    group: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    plan: Plan,
    links: SwapLinks<'_>,
    interval: Duration,
) {
    let started = Instant::now();
    let unloaded: Vec<String> = plan
        .unload
        .iter()
        .map(|index| members[*index].name().to_owned())
        .collect();
    let loading: Vec<String> = plan
        .load
        .iter()
        .map(|index| members[*index].name().to_owned())
        .collect();
    let span = tracing::info_span!(
        "nagaya.swap",
        group,
        unload = unloaded.join(",").as_str(),
        load = loading.join(",").as_str(),
        unload_ms = tracing::field::Empty,
        load_ms = tracing::field::Empty,
    );
    let _entered = span.enter();
    // A park and an open can take seconds, and they block this thread. The guard beats for the group (spec 7.4).
    let beat = OpenBeat::start(group, links.heartbeats.clone(), interval);
    let instruments = instruments();
    for &index in &plan.unload {
        // A serve loop of stages unloads its engine while its stage threads still run.
        if members[index].loaded {
            unload(index, members, engines, links.store, links.fd_store, links.states);
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
    for &index in &plan.load {
        let name = members[index].name().to_owned();
        links.states.record(&name, EngineState::Loading);
        let (engine, handles, opened) = open(&mut members[index].job, links.fd_store)
            .unwrap_or_else(|(_, message)| panic!("the swap cannot open engine `{name}`: {message}"));
        engines[index] = engine;
        members[index].handles = handles;
        members[index].loaded = true;
        members[index].last_busy = Instant::now();
        links.states.record(&name, EngineState::Ready);
        if let Some(handoff) = opened.handoff {
            instruments.arena_handoff(&name, handoff.status());
        }
        loaded.push(opened);
    }
    span.record("load_ms", opening.elapsed().as_millis() as u64);
    drop(beat);
    tracing::info!(
        "swap on group {group}: unloaded [{}], loaded [{}] in {} ms",
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

/// The stage handles and the opened engine of [`open`].
type Open = (Box<dyn Engine>, Vec<Box<dyn StageHandle + Send>>, OpenedEngine);

/// Unload the engine at `index`: park its states, remove its arena fds from systemd's store, and drop it (spec 3.3a).
///
/// An engine with stages parks through its stage threads, so its serve loop calls this before the threads end. A
/// part that cannot be parked is a crash (spec 8.5).
pub(super) fn unload(
    index: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    store: Option<&Sender<StoreMessage>>,
    fd_store: Option<&FdStore>,
    states: &EngineStates,
) {
    let name = members[index].name().to_owned();
    states.record(&name, EngineState::Draining);
    let mut report = ParkReport::default();
    park::park_engine(&name, engines[index].as_mut(), store, &mut report);
    if !report.failures.is_empty() {
        panic!(
            "the swap cannot park the states of engine `{name}`: {}",
            report.failures.join("; ")
        );
    }
    // An fd in systemd's store holds the device memory of its part (spec 3.3a).
    handoff::remove_arena(&members[index].job, fd_store);
    // The drop closes the engine (engine-abi "`close` in Rust").
    engines[index] = Unloaded::of(&members[index]);
    members[index].handles.clear();
    members[index].loaded = false;
    states.record(&name, EngineState::Unloaded);
    tracing::info!(
        "engine `{name}` unloaded: parked {} states ({} bytes)",
        report.states,
        report.bytes
    );
}

/// Open the engine of `job` and check its limits against the admitted limits, as the start does (spec 10.2, 10.4).
///
/// An engine with stages gives its stage handles here: the process runs an engine of two stages, with one handle for
/// each stage (engine-abi D1).
pub(super) fn open(job: &mut GroupJob, fd_store: Option<&FdStore>) -> Result<Open, (OpenErrorKind, String)> {
    let (mut engine, handoff) = handoff::open(job, fd_store).map_err(|error| (error.kind, error.to_string()))?;
    let opened_limits = engine.limits();
    if opened_limits != job.limits {
        // A part size is the shape and the element type of a part. Any other difference is a crash (spec 10.4).
        return Err(if opened_limits.parts != job.limits.parts {
            (
                OpenErrorKind::PartSize,
                "a part size after open does not equal the admitted part size".to_owned(),
            )
        } else {
            (
                OpenErrorKind::Other,
                "limits() after open do not equal the admitted limits".to_owned(),
            )
        });
    }
    // The rules of the stages and the feed rule (engine-abi D1, D11). An engine that breaks one does not serve.
    if let Err(message) = opened_limits.check() {
        return Err((
            OpenErrorKind::Other,
            format!("the limits of the engine break a rule: {message}"),
        ));
    }
    let mut handles = Vec::new();
    if opened_limits.capabilities.contains(&Capability::Stages) {
        if opened_limits.stages.len() != 2 {
            return Err((
                OpenErrorKind::Other,
                format!(
                    "the process runs an engine of two stages; this one has {}. One residual buffer serves a block",
                    opened_limits.stages.len()
                ),
            ));
        }
        handles = engine.stages().map(|stages| stages.take_handles()).unwrap_or_default();
        if handles.len() != opened_limits.stages.len() {
            return Err((
                OpenErrorKind::Other,
                format!(
                    "the engine declares {} stages and gave {} stage handles",
                    opened_limits.stages.len(),
                    handles.len()
                ),
            ));
        }
    }
    let opened = OpenedEngine {
        name: job.placement.name.clone(),
        thread: thread::current().id(),
        identity: engine.identity(),
        limits: opened_limits,
        handoff,
    };
    Ok((engine, handles, opened))
}
