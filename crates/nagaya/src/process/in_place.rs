//! The swap in place on a group thread (spec 3.3a).
//!
//! A group keeps every engine of its profile entries, in profile order. An engine that is not loaded has an
//! [`Unloaded`] place. A job for such an engine waits, and the group plans a swap with the ledger thread. When the
//! engines that must unload hold no job, the group parks and closes them, admits the new set of loaded engines, and
//! opens the engine of the job.

use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};

use nagaya_abi::{Capability, Engine, Identity, Limits, PassError, PassOut, Segment, Tap};
use nagaya_ledger::LedgerError;
use nagaya_telemetry::{EngineState, instruments};

use super::{Event, GroupJob, LedgerMessage, OpenBeat, OpenedEngine, ParkReport, WatchMessage, handoff, park};
use crate::admission::{AdmitError, OpenErrorKind};
use crate::arena::FdStore;
use crate::park::StoreMessage;

/// One engine of a group: what opens it, and whether it is loaded.
pub(super) struct Member {
    pub(super) job: GroupJob,
    pub(super) loaded: bool,
}

impl Member {
    pub(super) fn name(&self) -> &str {
        &self.job.placement.name
    }

    /// True when the engine runs as stages on stage threads (engine-abi D1).
    pub(super) fn staged(&self) -> bool {
        self.job.limits.capabilities.contains(&Capability::Stages)
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
/// An engine with stages shares its cards with no other loaded engine, so it unloads the others, and the others unload
/// it. Then the group unloads its loaded engines, the last in profile order first, until the target fits. Admission
/// checked that the target fits with its whole group unloaded, so a plan that still does not fit is a defect.
pub(super) fn plan(target: usize, members: &[Member], ledger: &Sender<LedgerMessage>) -> Plan {
    let loaded: Vec<usize> = (0..members.len()).filter(|index| members[*index].loaded).collect();
    let mut unload: Vec<usize> = if members[target].staged() {
        loaded.clone()
    } else {
        loaded
            .iter()
            .copied()
            .filter(|index| members[*index].staged())
            .collect()
    };
    let mut candidates = loaded
        .iter()
        .rev()
        .copied()
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

/// The ways out of the group thread that a swap uses.
pub(super) struct SwapLinks<'a> {
    pub(super) events: &'a Sender<Event>,
    pub(super) heartbeats: &'a Sender<WatchMessage>,
    pub(super) ledger: &'a Sender<LedgerMessage>,
    pub(super) store: Option<&'a Sender<StoreMessage>>,
    pub(super) fd_store: Option<&'a FdStore>,
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
        let name = members[index].name().to_owned();
        instruments.engine_state(&name, EngineState::Draining);
        let mut report = ParkReport::default();
        park::park_engine(&name, engines[index].as_mut(), links.store, &mut report);
        if !report.failures.is_empty() {
            panic!(
                "the swap cannot park the states of engine `{name}`: {}",
                report.failures.join("; ")
            );
        }
        // An fd in systemd's store holds the device memory of its part (spec 3.3a).
        handoff::remove_arena(&members[index].job, links.fd_store);
        // The drop closes the engine (engine-abi "`close` in Rust").
        engines[index] = Unloaded::of(&members[index]);
        members[index].loaded = false;
        instruments.engine_state(&name, EngineState::Unloaded);
        tracing::info!(
            "engine `{name}` unloaded: parked {} states ({} bytes)",
            report.states,
            report.bytes
        );
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
        instruments.engine_state(&name, EngineState::Loading);
        let (engine, opened) = open(&mut members[index].job, links.fd_store)
            .unwrap_or_else(|(_, message)| panic!("the swap cannot open engine `{name}`: {message}"));
        engines[index] = engine;
        members[index].loaded = true;
        instruments.engine_state(&name, EngineState::Ready);
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

/// Open the engine of `job` and check its limits against the admitted limits, as the start does (spec 10.2, 10.4).
pub(super) fn open(
    job: &mut GroupJob,
    fd_store: Option<&FdStore>,
) -> Result<(Box<dyn Engine>, OpenedEngine), (OpenErrorKind, String)> {
    let (engine, handoff) = handoff::open(job, fd_store).map_err(|error| (error.kind, error.to_string()))?;
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
    let opened = OpenedEngine {
        name: job.placement.name.clone(),
        thread: thread::current().id(),
        identity: engine.identity(),
        limits: opened_limits,
        handoff,
    };
    Ok((engine, opened))
}
