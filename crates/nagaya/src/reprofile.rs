//! The actions of a profile change, and whether the change needs a restart (spec 3.3).
//!
//! The comparison is by engine name. `keep`, `residency` and `unload` run in place. `load` and `reload` restart the
//! process, because the engines of a device group are fixed at start (spec 3.3, "Which actions run in place").
//!
//! The main loop owns the active profile. The profile routes send it calls, and [`Changer::tick`] answers them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use nagaya_abi::Capability;
use nagaya_ledger::LedgerError;
use serde::Serialize;

use crate::admission::{AdmitError, Admitted};
use crate::cluster::Cluster;
use crate::process::{LoadError, Ports, Process};
use crate::profile::{EnginePlacement, PlacementProfile, TierSpec};
use crate::swap::{Candidate, Outcome, Refusal, Shortfall, Swap, SwapError};

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
    /// `kind`, `devices`, `config`, `family` or `lab` changes.
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
        && old.lab == new.lab;
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
    /// Publish a desired profile for another host (spec 2.4, "The routes of a profile for another host").
    Desire(Desire),
}

/// A desired profile for another host, which the main loop publishes in `profile:desired:<host>` (spec 2.4).
pub struct Desire {
    /// The node name of the target.
    pub host: String,
    /// The version that the client picked.
    pub version: u64,
    /// The profile as the client sent it. The target checks it (spec 2.4).
    pub profile: serde_json::Value,
    /// Where the answer goes.
    pub reply: Sender<DesireAnswer>,
}

/// The answer to a desired profile for another host (spec 2.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DesireAnswer {
    /// The node wrote the key. The target handles it later.
    Published,
    /// No live node has the name of the host.
    UnknownHost,
    /// A live node holds this version or a higher one for the host.
    Conflict {
        /// The highest version of a live `profile:desired:<host>`.
        highest: u64,
    },
    /// The value is past the bound of a gossip value (spec 2.3).
    TooLarge(String),
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
    /// The process stops and takes no new admission. Nothing changed (spec 10.3 step 1).
    Stopping,
}

/// The profile changes of a running process. The main loop owns it and calls [`Changer::tick`] (spec 3.3).
pub struct Changer {
    calls: Receiver<ProfileCall>,
    /// True while a change in place runs on its own thread.
    applying: Arc<AtomicBool>,
    /// The cluster from gossip. A profile that adds a name of a live peer is refused (spec 2.3).
    cluster: Option<Arc<Cluster>>,
    /// The desired profiles for other hosts that wait for the main loop, which owns the gossip (spec 2.4).
    desires: Vec<Desire>,
}

impl Changer {
    /// A changer, and the sender that the routes use.
    pub fn new() -> (Self, Sender<ProfileCall>) {
        let (calls_tx, calls) = mpsc::channel();
        let changer = Self {
            calls,
            applying: Arc::new(AtomicBool::new(false)),
            cluster: None,
            desires: Vec::new(),
        };
        (changer, calls_tx)
    }

    /// Refuse a profile that adds a served model name of a live peer of `cluster` (spec 2.3).
    pub fn with_cluster(mut self, cluster: Arc<Cluster>) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// The refusal of a profile whose plans add a served model name that a live peer serves, or `None` (spec 2.3,
    /// "The check of a name"). A name is added when no engine of the active profile gives it.
    fn clash(&self, process: &Process, active: &PlacementProfile, planned: &Admitted) -> Option<String> {
        let state = self.cluster.as_ref()?.state();
        let names = |admitted: &Admitted| -> Vec<String> {
            admitted
                .engines
                .iter()
                .flat_map(|engine| engine.served.clone().unwrap_or_default())
                .collect()
        };
        let new = names(planned);
        if new.iter().all(|name| state.served_by_peer(name).is_none()) {
            return None;
        }
        let old = names(
            &process
                .check_profile(active)
                .unwrap_or_else(|error| panic!("the active profile does not pass its own check: {error}")),
        );
        new.iter().filter(|name| !old.contains(name)).find_map(|name| {
            state.served_by_peer(name).map(|peer| {
                format!(
                    "the profile adds the served model name `{name}`, which node {} serves (spec 2.3); give the model \
                     another name, for example with the host name in it",
                    peer.node
                )
            })
        })
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
                ProfileCall::Desire(desire) => self.desires.push(desire),
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

    /// The desired profiles for other hosts that wait. The main loop publishes each one and answers it.
    pub fn take_desires(&mut self) -> Vec<Desire> {
        std::mem::take(&mut self.desires)
    }

    /// Handle a desired profile for this host with the code of a local `PUT` (spec 2.4, "The routes of a profile for
    /// another host"). The node records the version, also for a refusal. A refusal of a running change is `busy`. A
    /// stopping process handles nothing, so the version stays a candidate.
    pub fn desire(&mut self, swap: &mut Swap, process: &Process, candidate: Candidate) -> Result<Outcome, SwapError> {
        let version = candidate.version;
        let refused = |reason: String, shortfall: Vec<Shortfall>| {
            Outcome::Refused(Refusal {
                version,
                reason,
                shortfall,
            })
        };
        let outcome = match serde_json::from_value::<PlacementProfile>(candidate.profile) {
            Err(error) => refused(format!("the profile does not parse: {error}"), Vec::new()),
            Ok(profile) => match self.put(swap, process, profile)? {
                PutAnswer::Unchanged(_) => Outcome::Unchanged,
                PutAnswer::Accepted(change) => Outcome::Accepted(change),
                PutAnswer::Invalid(reason) => refused(reason, Vec::new()),
                PutAnswer::Refused { reason, shortfall } => refused(reason, shortfall),
                PutAnswer::Busy => refused("busy".to_owned(), Vec::new()),
                PutAnswer::Stopping => return Ok(Outcome::Stopping),
            },
        };
        swap.record(version)?;
        Ok(outcome)
    }

    fn put(&self, swap: &mut Swap, process: &Process, profile: PlacementProfile) -> Result<PutAnswer, SwapError> {
        if !profile.engines.is_empty() && profile.store.is_none() {
            return Ok(PutAnswer::Invalid(
                "the profile has engines and no `store.nvme` budget; add a [store] table".to_owned(),
            ));
        }
        // The lab tables follow the rules of the start, so an accepted change never stops a restart (docs/lab.md).
        if let Err(reason) = crate::serving::lab_files(&profile) {
            return Ok(PutAnswer::Invalid(reason));
        }
        let active = swap.active().clone();
        if profile == active {
            return Ok(PutAnswer::Unchanged(change(&active, &profile, |_| true)));
        }
        if process.stopping() {
            return Ok(PutAnswer::Stopping);
        }
        let planned = match process.check_profile(&profile) {
            Ok(planned) => planned,
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
        };
        if let Some(clash) = self.clash(process, &active, &planned) {
            return Ok(PutAnswer::Invalid(clash));
        }
        if self.applying() {
            return Ok(PutAnswer::Busy);
        }
        // A swap that waits or runs makes the change busy (spec 3.3a). From here, no load is
        // planned until the ledger takes the new profile.
        match process.begin_change(&profile) {
            Ok(_) => {}
            Err(AdmitError::Busy(_)) => return Ok(PutAnswer::Busy),
            Err(error) => return Ok(PutAnswer::Invalid(error.to_string())),
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
        if let Err(error) = swap.accept(profile.clone()) {
            process.end_change();
            return Err(error);
        }
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
            let in_place = change.clone();
            spawn_change(process.ports(), Arc::clone(&self.applying), move |ports| {
                apply(ports, &profile, &in_place)
            });
        }
        Ok(PutAnswer::Accepted(change))
    }
}

/// Run `change` on the thread `nagaya-reprofile`, and set `applying` false at its end. The change logs to the
/// subscriber of the caller, so a test that reads the log sees it too.
fn spawn_change(ports: Ports, applying: Arc<AtomicBool>, change: impl FnOnce(&Ports) + Send + 'static) {
    let dispatch = tracing::dispatcher::get_default(tracing::Dispatch::clone);
    std::thread::Builder::new()
        .name("nagaya-reprofile".to_owned())
        .spawn(move || {
            tracing::dispatcher::with_default(&dispatch, || {
                // A panic of the change stops the process (spec 8.5). Else every later PUT and load stays busy.
                ports.guard("nagaya-reprofile", || change(&ports));
            });
            applying.store(false, Ordering::SeqCst);
        })
        .expect("cannot start the thread of a profile change");
}

/// Run `change` on the thread of a change in place, as [`Changer::tick`] does. Only the `fake` feature builds this
/// function, so a test can give a change that panics.
#[cfg(feature = "fake")]
pub fn spawn_test_change(ports: Ports, applying: Arc<AtomicBool>, change: impl FnOnce(&Ports) + Send + 'static) {
    spawn_change(ports, applying, change);
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
    // The ledger takes the new profile when no swap waits or runs, and each gone engine dropped and holds no context
    // (spec 3.3a). A council request can end its context after the drop of its engine.
    let gone: Vec<String> = gone.iter().map(|engine| (*engine).to_owned()).collect();
    let mut told: Option<Instant> = None;
    loop {
        match ports.try_reprofile(profile, &gone) {
            None => return,
            Some(Ok(Ok(_))) => break,
            Some(Ok(Err(why))) => {
                // The log names the wait at its start and every 10 s after (spec 9.1).
                if told.is_none_or(|at| at.elapsed() >= Duration::from_secs(10)) {
                    tracing::info!("the profile change waits for {why} before the ledger takes the profile");
                    told = Some(Instant::now());
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Some(Err(error)) => {
                return ports.fail(format!(
                    "the ledger refused the profile that its check admitted: {error}"
                ));
            }
        }
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
