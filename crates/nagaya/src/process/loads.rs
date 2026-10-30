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

    /// Keep the identity of an engine that opened.
    pub(super) fn opened(&self, engine: &str, identity: &nagaya_abi::Identity) {
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
            let mut unload = Vec::with_capacity(names.len());
            for name in names {
                match group_engines[group].iter().position(|other| *other == name) {
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
                Plan {
                    load: Vec::new(),
                    unload: vec![index],
                }
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
            Request::Unload(index) => Plan {
                load: Vec::new(),
                unload: vec![index],
            },
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
