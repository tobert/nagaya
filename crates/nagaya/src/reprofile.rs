//! The actions of a profile change, and whether the change needs a restart (spec 3.3).
//!
//! The comparison is by engine name. `keep`, `residency` and `unload` run in place. `load` and `reload` restart the
//! process, because the engines of a device group are fixed at start (spec 3.3, "Which actions run in place").
//!
//! The main loop owns the active profile. The profile routes send it calls, and [`Changer::tick`] answers them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use nagaya_abi::Capability;
use nagaya_ledger::LedgerError;
use serde::Serialize;

use crate::admission::AdmitError;
use crate::process::{LoadError, Ports, Process};
use crate::profile::{EnginePlacement, PlacementProfile, TierSpec};
use crate::swap::{Shortfall, Swap, SwapError};

/// What a profile change does to one engine (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// The entry is the same, or only its `share` changes.
    Keep,
    /// Only `residency` or `pinned` changes.
    Residency,
    /// The entry is new.
    Load,
    /// The entry is gone.
    Unload,
    /// `kind`, `devices`, `config`, `family` or `pin_identity` changes.
    Reload,
}

/// The action of one engine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EngineAction {
    /// The name of the engine.
    pub engine: String,
    /// The action.
    pub action: Action,
}

/// A profile change: the action of each engine, and whether the process must restart.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Change {
    /// The engines of the new profile in its order, then the engines that are gone, in the order of the active profile.
    pub actions: Vec<EngineAction>,
    /// True when the change needs a restart with exit 4.
    pub restart: bool,
}

/// The change from `active` to `new`. `closes` tells whether an engine of the active profile declares `close`.
///
/// A restart is necessary for a `load`, a `reload`, an `unload` of an engine without `close`, and a new `[store]`
/// table (spec 3.3).
pub fn change(active: &PlacementProfile, new: &PlacementProfile, closes: impl Fn(&str) -> bool) -> Change {
    let mut actions: Vec<EngineAction> = new
        .engines
        .iter()
        .map(|entry| EngineAction {
            engine: entry.name.clone(),
            action: match active.engines.iter().find(|old| old.name == entry.name) {
                None => Action::Load,
                Some(old) => compare(old, entry),
            },
        })
        .collect();
    actions.extend(
        active
            .engines
            .iter()
            .filter(|old| new.engines.iter().all(|entry| entry.name != old.name))
            .map(|old| EngineAction {
                engine: old.name.clone(),
                action: Action::Unload,
            }),
    );
    let restart = active.store != new.store
        || actions.iter().any(|entry| match entry.action {
            Action::Load | Action::Reload => true,
            Action::Unload => !closes(&entry.engine),
            Action::Keep | Action::Residency => false,
        });
    Change { actions, restart }
}

/// The action of an entry that is in both profiles.
fn compare(old: &EnginePlacement, new: &EnginePlacement) -> Action {
    let opens_the_same = old.kind == new.kind
        && old.devices == new.devices
        && old.config == new.config
        && old.family == new.family
        && old.pin_identity == new.pin_identity;
    if !opens_the_same {
        Action::Reload
    } else if old.residency != new.residency || old.pinned != new.pinned {
        Action::Residency
    } else {
        Action::Keep
    }
}

/// A call of the profile routes to the main loop, which owns the active profile (spec 3.3).
pub enum ProfileCall {
    /// Give the active profile.
    Get(Sender<PlacementProfile>),
    /// Change the active profile.
    Put(PlacementProfile, Sender<PutAnswer>),
}

/// The answer to a `PUT` of a profile (spec 3.3, "The profile routes").
#[derive(Clone, Debug, PartialEq)]
pub enum PutAnswer {
    /// The profile equals the active profile. Nothing changed.
    Unchanged(Change),
    /// The profile persisted. With `restart`, the process stops with exit 4; else the change runs in place.
    Accepted(Change),
    /// The profile breaks a rule of 2.4 or 3.3a. Nothing changed.
    Invalid(String),
    /// The profile does not fit its budgets. Nothing changed.
    Refused {
        /// Why.
        reason: String,
        /// The bytes that each short budget lacks.
        shortfall: Vec<Shortfall>,
    },
    /// A change in place runs now. Nothing changed.
    Busy,
}

/// The profile changes of a running process. The main loop owns it and calls [`Changer::tick`] (spec 3.3).
pub struct Changer {
    calls: Receiver<ProfileCall>,
    /// True while a change in place runs on its own thread.
    applying: Arc<AtomicBool>,
}

impl Changer {
    /// A changer, and the sender that the routes use.
    pub fn new() -> (Self, Sender<ProfileCall>) {
        let (calls_tx, calls) = mpsc::channel();
        let changer = Self {
            calls,
            applying: Arc::new(AtomicBool::new(false)),
        };
        (changer, calls_tx)
    }

    /// True while a change in place runs.
    pub fn applying(&self) -> bool {
        self.applying.load(Ordering::SeqCst)
    }

    /// Answer each call that waits. Gives true when an accepted profile needs a restart: the main loop then stops with
    /// exit 4. An error to persist the profile is an error here, and the process stops (spec 8.5).
    pub fn tick(&mut self, swap: &mut Swap, process: &Process) -> Result<bool, SwapError> {
        let mut restart = false;
        while let Ok(call) = self.calls.try_recv() {
            match call {
                ProfileCall::Get(reply) => {
                    let _ = reply.send(swap.active().clone());
                }
                ProfileCall::Put(profile, reply) => {
                    let answer = self.put(swap, process, profile)?;
                    restart |= matches!(&answer, PutAnswer::Accepted(change) if change.restart);
                    let _ = reply.send(answer);
                    if restart {
                        // The process stops now. A later call waits for the next start.
                        break;
                    }
                }
            }
        }
        Ok(restart)
    }

    fn put(&self, swap: &mut Swap, process: &Process, profile: PlacementProfile) -> Result<PutAnswer, SwapError> {
        if !profile.engines.is_empty() && profile.store.is_none() {
            return Ok(PutAnswer::Invalid(
                "the profile has engines and no `store.nvme` budget; add a [store] table".to_owned(),
            ));
        }
        let active = swap.active().clone();
        if profile == active {
            return Ok(PutAnswer::Unchanged(change(&active, &profile, |_| true)));
        }
        match process.check_profile(&profile) {
            Ok(_) => {}
            Err(AdmitError::Ledger(LedgerError::Refused(refusal))) => {
                let shortfall = refusal
                    .shortfall
                    .into_iter()
                    .map(|(tiers, bytes)| Shortfall {
                        tiers: tiers.into_iter().map(TierSpec::from).collect(),
                        bytes,
                    })
                    .collect();
                return Ok(PutAnswer::Refused {
                    reason: "the profile does not fit its budgets".to_owned(),
                    shortfall,
                });
            }
            Err(error) => return Ok(PutAnswer::Invalid(error.to_string())),
        }
        if self.applying() {
            return Ok(PutAnswer::Busy);
        }
        // The active profile was admitted at start, so its check gives the limits of each engine.
        let planned = process
            .check_profile(&active)
            .unwrap_or_else(|error| panic!("the active profile does not pass its own check: {error}"));
        let closes = |name: &str| {
            planned
                .engines
                .iter()
                .find(|engine| engine.name == name)
                .is_some_and(|engine| engine.limits.capabilities.contains(&Capability::Close))
        };
        let change = change(&active, &profile, closes);
        swap.accept(profile.clone())?;
        tracing::info!(
            "profile accepted: {}{}",
            change
                .actions
                .iter()
                .map(|entry| format!("{} {:?}", entry.engine, entry.action))
                .collect::<Vec<_>>()
                .join(", "),
            if change.restart { "; the process restarts" } else { "" }
        );
        if !change.restart {
            self.applying.store(true, Ordering::SeqCst);
            let ports = process.ports();
            let applying = Arc::clone(&self.applying);
            let in_place = change.clone();
            std::thread::Builder::new()
                .name("nagaya-reprofile".to_owned())
                .spawn(move || {
                    apply(&ports, &profile, &in_place);
                    applying.store(false, Ordering::SeqCst);
                })
                .expect("cannot start the thread of a profile change");
        }
        Ok(PutAnswer::Accepted(change))
    }
}

/// Run a change in place (spec 3.3): retire the engines that are gone, wait until the ledger counts them unloaded,
/// admit the new profile, then give each engine its new entry. A stop ends the change. Another failure stops the
/// process (spec 8.5).
fn apply(ports: &Ports, profile: &PlacementProfile, change: &Change) {
    let gone: Vec<&str> = change
        .actions
        .iter()
        .filter(|entry| entry.action == Action::Unload)
        .map(|entry| entry.engine.as_str())
        .collect();
    for engine in &gone {
        match ports.retire(engine) {
            Ok(_) => {}
            Err(LoadError::Stopping | LoadError::Stopped) => return,
            Err(error) => return ports.fail(format!("the profile change cannot retire `{engine}`: {error}")),
        }
    }
    // A swap that retires an engine tells the ledger after the unload. The new profile lacks the engine, so the ledger
    // takes it only after that.
    loop {
        let Some(loaded) = ports.loaded_engines() else { return };
        if gone.iter().all(|engine| !loaded.contains(*engine)) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if let Err(error) = ports.reprofile(profile) {
        return ports.fail(format!(
            "the ledger refused the profile that its check admitted: {error}"
        ));
    }
    for entry in &profile.engines {
        match ports.place(entry) {
            Ok(_) => {}
            Err(LoadError::Stopping | LoadError::Stopped) => return,
            Err(error) => return ports.fail(format!("the profile change cannot place `{}`: {error}", entry.name)),
        }
    }
    tracing::info!("the profile change in place ended");
}
