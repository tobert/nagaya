//! Explicit loads and the state of each engine (spec 3.3, 3.3a, 3.3b).
//!
//! A request never starts a load. A client asks the process to load or unload an engine, and the group thread of the
//! engine plans the swap. The group threads write the state of each engine, and the routes read it.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};

use nagaya_telemetry::{EngineState, instruments};

use super::in_place::{self, Member, Plan};
use super::{GroupMessage, LedgerMessage};
use crate::profile::{EnginePlacement, Residency};

/// The state of each engine of the profile, by name (spec 3.3, 9.4).
///
/// The group threads write it, and the routes read it. A route can read a state that a swap changes a moment later,
/// so the group checks the state of each job again (spec 3.3a, "A job for an engine that is not ready").
#[derive(Clone, Debug, Default)]
pub struct EngineStates(
    Arc<Mutex<BTreeMap<String, EngineState>>>,
    /// The identity of each engine at its last open since the start. A served model shows its hashes from it.
    Arc<Mutex<BTreeMap<String, nagaya_abi::Identity>>>,
    /// The count of the opens of each engine since the start. A worker that keeps a table on the device of an engine
    /// knows from it that a new open cleared the table (engine-abi "Control vectors").
    Arc<Mutex<BTreeMap<String, u64>>>,
);

impl EngineStates {
    /// The identity of the engine at its last open since the start, or `None` before its first open.
    pub fn identity(&self, engine: &str) -> Option<nagaya_abi::Identity> {
        self.1
            .lock()
            .expect("the engine identities lock is poisoned")
            .get(engine)
            .cloned()
    }

    /// The count of the opens of the engine since the start: 0 before its first open.
    pub fn opens(&self, engine: &str) -> u64 {
        self.2
            .lock()
            .expect("the engine opens lock is poisoned")
            .get(engine)
            .copied()
            .unwrap_or(0)
    }

    /// Keep the identity of an engine that opened, and count the open.
    pub(super) fn opened(&self, engine: &str, identity: &nagaya_abi::Identity) {
        *self
            .2
            .lock()
            .expect("the engine opens lock is poisoned")
            .entry(engine.to_owned())
            .or_default() += 1;
        self.1
            .lock()
            .expect("the engine identities lock is poisoned")
            .insert(engine.to_owned(), identity.clone());
    }

    /// The state of the engine with this name, or `None` for a name that is not in the profile.
    pub fn get(&self, engine: &str) -> Option<EngineState> {
        self.0
            .lock()
            .expect("the engine states lock is poisoned")
            .get(engine)
            .copied()
    }

    /// Set the state of an engine. The start records its own states as metrics (`lifecycle`), so this records none.
    pub(super) fn set(&self, engine: &str, state: EngineState) {
        self.0
            .lock()
            .expect("the engine states lock is poisoned")
            .insert(engine.to_owned(), state);
    }

    /// Set the state of an engine in a swap, and record it as the metric `nagaya.engine.state` (spec 9.4).
    pub(super) fn record(&self, engine: &str, state: EngineState) {
        self.set(engine, state);
        instruments().engine_state(engine, state);
    }
}

/// What a group did with a load or an unload (spec 3.3b).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadReply {
    /// The group planned a swap. It unloads the engines of `unload`, then it loads the engines of `load`.
    Started {
        /// The engines that unload, in the order of the swap.
        unload: Vec<String>,
        /// The engines that load, in the order of the swap.
        load: Vec<String>,
    },
    /// No swap is necessary: the engine has this state, or a planned swap gives it this state.
    Already(EngineState),
}

/// The answer to a release or a reclaim of a device (spec 3.3b, "The release of a device").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceReply {
    /// The groups planned the swaps: the engines that unload for a release, or that load for a reclaim, in the
    /// order of the profile.
    Started(Vec<String>),
    /// No swap is necessary. The device has the state that the call asks for.
    Already,
}

/// Why a load or an unload did not start (spec 3.3b). Nothing changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// No engine of the profile has this name.
    Unknown(String),
    /// The engine is pinned. A pinned engine leaves only by a profile change (spec 3.3a, "Explicit layout").
    Pinned(String),
    /// The engine has no `close`, so a swap cannot unload it (spec 3.3a).
    NoClose(String),
    /// The `unload` list of a load is wrong: it names the engine itself, or an engine of another device group.
    BadUnload(String),
    /// A profile change waits for the ledger, so the load must wait too (spec 3.3a). The client sends it again.
    Busy(String),
    /// The engine does not fit beside the engines that stay loaded (spec 3.3a, "Explicit layout").
    InTheWay {
        /// The engine that does not load.
        engine: String,
        /// Why it does not fit: the shortfall of each short budget, or the rule of an engine with stages.
        shortfall: String,
        /// The loaded engines that are not pinned. A load can name them in `unload`.
        in_the_way: Vec<String>,
        /// The pinned engines that stay loaded.
        pinned: Vec<String>,
    },
    /// The engine names a released device, which takes no load until a reclaim (spec 3.3b, "The release of a
    /// device").
    Released {
        /// The engine that does not load.
        engine: String,
        /// The released device.
        device: nagaya_abi::DeviceId,
    },
    /// No engine of the profile names the device, so it has nothing to release (spec 3.3b).
    NoDevice(nagaya_abi::DeviceId),
    /// The process is stopping and takes no new work (spec 10.3 step 1).
    Stopping,
    /// The group thread stopped.
    Stopped,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Unknown(name) => write!(
                formatter,
                "no engine of the profile has the name `{name}`; the profile names each engine"
            ),
            LoadError::Pinned(name) => write!(
                formatter,
                "engine `{name}` is pinned, so no load and no unload removes it; change the profile to remove it"
            ),
            LoadError::NoClose(name) => write!(
                formatter,
                "engine `{name}` has no `close`, so a swap cannot unload it; change the profile, which restarts"
            ),
            LoadError::BadUnload(why) => write!(formatter, "{why}; name other loaded engines of the same group"),
            LoadError::Released { engine, device } => write!(
                formatter,
                "engine `{engine}` names device {device}, which is released; reclaim the device first with \
                 POST /nagaya/v1/devices/{device}/reclaim"
            ),
            LoadError::NoDevice(device) => write!(
                formatter,
                "no engine of the profile names device {device}; the profile names the devices of each engine"
            ),
            LoadError::Busy(why) => write!(formatter, "nothing changed: {why}; send the load again when it ends"),
            LoadError::InTheWay {
                engine,
                shortfall,
                in_the_way,
                pinned,
            } => {
                write!(formatter, "engine `{engine}` does not fit: {shortfall}")?;
                let list = |names: &[String]| {
                    names
                        .iter()
                        .map(|name| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                if !in_the_way.is_empty() {
                    write!(
                        formatter,
                        "; the loaded engines in the way are {}, and a load can name them in `unload`",
                        list(in_the_way)
                    )?;
                }
                if !pinned.is_empty() {
                    write!(formatter, "; the pinned engines {} stay loaded", list(pinned))?;
                }
                Ok(())
            }
            LoadError::Stopping => write!(formatter, "the process is stopping"),
            LoadError::Stopped => write!(formatter, "the group of the engine stopped"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A load, an unload or a profile change of one engine of a group, by its place in the group.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Request {
    /// Load the engine after the engines of the list unload, in one swap (spec 3.3a, "Explicit layout").
    Load(usize, Vec<usize>),
    Unload(usize),
    /// Unload the engine for good: the new profile lacks it (spec 3.3, "unload").
    Retire(usize),
    /// Take the residency, the pin and the share of a new profile entry (spec 3.3, "residency" and "keep").
    Place(usize, Box<EnginePlacement>),
}

/// What an ask asks of the engine with a name.
pub(super) enum Asking {
    /// Load the engine after the engines with these names unload.
    Load(Vec<String>),
    Unload,
    Retire,
    Place(Box<EnginePlacement>),
}

/// A load or an unload that waits for its group to plan it, with the sender of its reply.
pub(super) struct Ask {
    pub(super) request: Request,
    pub(super) reply: Sender<Result<LoadReply, LoadError>>,
}

/// Send a load or an unload to the group of `engine`, and wait for the reply of the group.
pub(super) fn ask(
    groups: &[Sender<GroupMessage>],
    group_engines: &[Vec<String>],
    accepting: bool,
    engine: &str,
    asking: Asking,
) -> Result<LoadReply, LoadError> {
    let (group, index) = group_engines
        .iter()
        .enumerate()
        .find_map(|(group, names)| names.iter().position(|name| name == engine).map(|index| (group, index)))
        .ok_or_else(|| LoadError::Unknown(engine.to_owned()))?;
    if !accepting {
        return Err(LoadError::Stopping);
    }
    let request = match asking {
        Asking::Load(names) => {
            // The list is checked here, before the group sees it, so a loaded engine cannot hide a bad list behind
            // its 200 (spec 3.3b). A repeat would unload one engine twice.
            let mut unload = Vec::with_capacity(names.len());
            for name in names {
                match group_engines[group].iter().position(|other| *other == name) {
                    Some(other) if other == index => {
                        return Err(LoadError::BadUnload(format!(
                            "the load of `{engine}` names the engine itself in `unload`"
                        )));
                    }
                    Some(other) if unload.contains(&other) => {
                        return Err(LoadError::BadUnload(format!(
                            "the load of `{engine}` names `{name}` twice in `unload`"
                        )));
                    }
                    Some(other) => unload.push(other),
                    None if group_engines.iter().any(|names| names.contains(&name)) => {
                        return Err(LoadError::BadUnload(format!(
                            "the load of `{engine}` names `{name}` in `unload`, and `{name}` is in another device group"
                        )));
                    }
                    None => return Err(LoadError::Unknown(name)),
                }
            }
            Request::Load(index, unload)
        }
        Asking::Unload => Request::Unload(index),
        Asking::Retire => Request::Retire(index),
        Asking::Place(entry) => Request::Place(index, entry),
    };
    let (reply, answer) = mpsc::channel();
    groups[group]
        .send(GroupMessage::Ask(Ask { request, reply }))
        .map_err(|_| LoadError::Stopped)?;
    answer.recv().map_err(|_| LoadError::Stopped)?
}

/// Plan a swap that unloads one engine and loads none. The ledger counts it as a swap that runs until its charge after
/// the drop, so a profile change waits for it (spec 3.3a). The ledger takes it with no check, because fewer engines
/// leave more room. The unload runs even if the ledger did not answer, because it frees memory.
fn unload_only(index: usize, members: &[Member], ledger: &Sender<LedgerMessage>) -> Plan {
    let name = members[index].name().to_owned();
    if let Err(error) = in_place::take(ledger, Vec::new(), vec![name.clone()]) {
        tracing::error!("the ledger did not take the unload of engine `{name}`, and the unload runs: {error}");
    }
    Plan {
        load: Vec::new(),
        unload: vec![index],
    }
}

/// Take an ask that comes to a group. A load while a profile change runs is busy at once (spec 3.3a). Each other ask
/// waits in order of arrival.
///
/// A load must not wait behind the retire of a change. That retire waits for the jobs of its engine, and a client
/// can hold such a job until its load returns. A change cannot begin while a swap waits, so the answer stays true
/// until this group plans the load.
pub(super) fn take_ask(ask: Ask, asks: &mut VecDeque<Ask>, ledger: &Sender<LedgerMessage>) {
    if matches!(ask.request, Request::Load(..)) {
        let (reply, answer) = mpsc::channel();
        if ledger.send(LedgerMessage::Changing(reply)).is_err() {
            let _ = ask.reply.send(Err(LoadError::Stopped));
            return;
        }
        match answer.recv() {
            Ok(true) => {
                let _ = ask.reply.send(Err(LoadError::Busy(
                    "a profile change waits for the ledger, and no load runs until the ledger takes it".to_owned(),
                )));
                return;
            }
            Ok(false) => {}
            Err(_) => {
                let _ = ask.reply.send(Err(LoadError::Stopped));
                return;
            }
        }
    }
    asks.push_back(ask);
}

/// Answer the asks that wait, in order of arrival, until one plans a swap (spec 3.3b). The group calls it only when
/// it has no planned swap.
///
/// The swap unloads only the engines that its ask names (spec 3.3a, "No moves"). The states of the engines of the
/// swap change here: an engine that unloads is `draining`, and an engine that loads is `loading`.
pub(super) fn next_plan(
    asks: &mut VecDeque<Ask>,
    members: &mut [Member],
    ledger: &Sender<LedgerMessage>,
    states: &EngineStates,
) -> Option<Plan> {
    while let Some(Ask { request, reply }) = asks.pop_front() {
        let plan = match request {
            Request::Load(index, _) | Request::Unload(index) | Request::Place(index, _) | Request::Retire(index)
                if members[index].retired =>
            {
                let answer = match request {
                    Request::Retire(_) => Ok(LoadReply::Already(EngineState::Unloaded)),
                    _ => Err(LoadError::Unknown(members[index].name().to_owned())),
                };
                let _ = reply.send(answer);
                continue;
            }
            Request::Retire(index) => {
                members[index].retired = true;
                if !members[index].loaded {
                    let _ = reply.send(Ok(LoadReply::Already(EngineState::Unloaded)));
                    continue;
                }
                unload_only(index, members, ledger)
            }
            Request::Place(index, entry) => {
                let placement = &mut members[index].job.placement;
                placement.residency = entry.residency;
                placement.pinned = entry.pinned;
                placement.share = entry.share;
                if members[index].loaded || entry.residency != Residency::Resident {
                    let state = if members[index].loaded {
                        EngineState::Ready
                    } else {
                        EngineState::Unloaded
                    };
                    let _ = reply.send(Ok(LoadReply::Already(state)));
                    continue;
                }
                // A profile change loads a new resident engine only beside the loaded engines (spec 3.3).
                match in_place::plan(index, &[], members, ledger) {
                    Ok(plan) => plan,
                    Err(error) => {
                        tracing::warn!(
                            "the profile makes engine `{}` resident, and it stays unloaded: {error}",
                            members[index].name()
                        );
                        let _ = reply.send(Ok(LoadReply::Already(EngineState::Unloaded)));
                        continue;
                    }
                }
            }
            Request::Load(index, _) if members[index].loaded => {
                let _ = reply.send(Ok(LoadReply::Already(EngineState::Ready)));
                continue;
            }
            Request::Load(index, named) => match in_place::plan(index, &named, members, ledger) {
                Ok(plan) => plan,
                Err(error) => {
                    let _ = reply.send(Err(error));
                    continue;
                }
            },
            Request::Unload(index) if members[index].pinned() => {
                let _ = reply.send(Err(LoadError::Pinned(members[index].name().to_owned())));
                continue;
            }
            Request::Unload(index) if !members[index].closes() => {
                let _ = reply.send(Err(LoadError::NoClose(members[index].name().to_owned())));
                continue;
            }
            Request::Unload(index) if !members[index].loaded => {
                let _ = reply.send(Ok(LoadReply::Already(EngineState::Unloaded)));
                continue;
            }
            Request::Unload(index) => unload_only(index, members, ledger),
        };
        mark(&plan, members, states);
        let name = |index: &usize| members[*index].name().to_owned();
        let _ = reply.send(Ok(LoadReply::Started {
            unload: plan.unload.iter().map(name).collect(),
            load: plan.load.iter().map(name).collect(),
        }));
        return Some(plan);
    }
    None
}

/// Set the states of a planned swap: each engine that unloads is `draining`, and each engine that loads is `loading`.
pub(super) fn mark(plan: &Plan, members: &[Member], states: &EngineStates) {
    for &index in &plan.unload {
        states.record(members[index].name(), EngineState::Draining);
    }
    for &index in &plan.load {
        states.record(members[index].name(), EngineState::Loading);
    }
}
