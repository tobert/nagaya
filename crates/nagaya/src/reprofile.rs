//! The actions of a profile change, and whether the change needs a restart (spec 3.3).
//!
//! The comparison is by engine name. `keep`, `residency` and `unload` run in place. `load` and `reload` restart the
//! process, because the engines of a device group are fixed at start (spec 3.3, "Which actions run in place").

use serde::Serialize;

use crate::profile::{EnginePlacement, PlacementProfile};

/// What a profile change does to one engine (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// The entry is the same, or only its `share` changes.
    Keep,
    /// Only `residency` or `idle_unload_s` changes.
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
    } else if old.residency != new.residency || old.idle_unload_s != new.idle_unload_s {
        Action::Residency
    } else {
        Action::Keep
    }
}
