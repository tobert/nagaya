//! Admission of a placement profile before any engine opens (spec 2.4).

use std::collections::BTreeMap;

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Limits, Tier};
use nagaya_ledger::{Admission, Budget, EnginePlan, Ledger, LedgerError, Profile};

use crate::profile::PlacementProfile;

/// Why an engine cannot give its limits before it opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanError {
    /// What the engine could not plan.
    pub message: String,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for PlanError {}

/// The planner of one engine kind. It reads no weights (spec 2.4).
pub trait Plan {
    /// Give the engine's limits for a config.
    fn plan(&self, config: &toml::Table) -> Result<Limits, PlanError>;
}

/// The planners by engine kind.
#[derive(Default)]
pub struct Registry {
    planners: BTreeMap<String, Box<dyn Plan>>,
}

impl Registry {
    /// Make an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a planner for an engine kind.
    pub fn register(&mut self, kind: &str, planner: Box<dyn Plan>) {
        self.planners.insert(kind.to_owned(), planner);
    }

    /// The planner for an engine kind.
    fn planner(&self, kind: &str) -> Option<&dyn Plan> {
        self.planners.get(kind).map(Box::as_ref)
    }

    /// Make a registry with the fake engine.
    pub fn with_fake() -> Self {
        let mut registry = Self::new();
        registry.register("fake", Box::new(FakePlan));
        registry
    }
}

/// Why a profile cannot be admitted.
#[derive(Debug)]
pub enum AdmitError {
    /// The profile names an engine kind with no planner.
    UnknownKind {
        /// The engine name.
        engine: String,
        /// The engine kind.
        kind: String,
    },
    /// An engine cannot plan.
    Plan {
        /// The engine name.
        engine: String,
        /// The planner's message.
        message: String,
    },
    /// The ledger refused the profile.
    Ledger(LedgerError),
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmitError::UnknownKind { engine, kind } => {
                write!(formatter, "engine `{engine}` has kind `{kind}`, which has no planner")
            }
            AdmitError::Plan { engine, message } => {
                write!(formatter, "engine `{engine}` cannot plan: {message}")
            }
            AdmitError::Ledger(error) => {
                write!(formatter, "the profile cannot be admitted: {error}")
            }
        }
    }
}

impl std::error::Error for AdmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AdmitError::Ledger(error) => Some(error),
            AdmitError::UnknownKind { .. } | AdmitError::Plan { .. } => None,
        }
    }
}

/// The fake engine's planner. The fake engine reads no weights and declares no memory terms.
pub struct FakePlan;

impl Plan for FakePlan {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }
}

/// Admit a placement profile with the ledger, before any engine opens.
pub fn admit(profile: &PlacementProfile, registry: &Registry, ledger: &mut Ledger) -> Result<Admission, AdmitError> {
    let budgets: Vec<Budget> = profile
        .budgets
        .iter()
        .map(|budget| Budget {
            tiers: budget.tiers.iter().map(|tier| Tier::from(*tier)).collect(),
            bytes: budget.bytes,
        })
        .collect();
    let mut engines = Vec::with_capacity(profile.engines.len());
    for engine in &profile.engines {
        let planner = registry.planner(&engine.kind).ok_or_else(|| AdmitError::UnknownKind {
            engine: engine.name.clone(),
            kind: engine.kind.clone(),
        })?;
        let limits = planner.plan(&engine.config).map_err(|error| AdmitError::Plan {
            engine: engine.name.clone(),
            message: error.to_string(),
        })?;
        engines.push(EnginePlan {
            name: engine.name.clone(),
            limits,
            share: engine.share,
        });
    }
    ledger.admit(&Profile { budgets, engines }).map_err(AdmitError::Ledger)
}
