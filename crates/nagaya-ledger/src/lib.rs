//! Budgets, terms, admission, shares, and reclaim order.

use std::collections::BTreeMap;

use nagaya_abi::{DeviceId, Lifetime, Limits, Tier};
use nagaya_telemetry::{RefusalGate, TierLabel, instruments};

/// One budget: a byte target for a set of tiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Budget {
    /// The tiers that share this budget.
    pub tiers: Vec<Tier>,
    /// The byte target.
    pub bytes: u64,
}

/// One engine in a placement profile.
#[derive(Clone, Debug, PartialEq)]
pub struct EnginePlan {
    /// The engine name.
    pub name: String,
    /// The result of `plan(config)`.
    pub limits: Limits,
    /// The machine devices of the engine, in the order of the profile's `devices` list (spec 2.4).
    ///
    /// A term tier `device(i)` names the position `i` in this list.
    pub devices: Vec<DeviceId>,
    /// The engine's share of spare memory.
    pub share: u64,
}

/// A placement profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// The budgets.
    pub budgets: Vec<Budget>,
    /// The engines and their shares.
    pub engines: Vec<EnginePlan>,
}

/// The bytes that one engine got.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineAllocation {
    /// The engine name.
    pub name: String,
    /// The mandatory bytes, one entry per term.
    pub mandatory: Vec<(Tier, u64)>,
    /// The elastic bytes, one entry per term.
    pub elastic: Vec<(String, u64)>,
}

/// The result of a fitting profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    /// The engine allocations.
    pub engines: Vec<EngineAllocation>,
    /// The bytes left unassigned, one entry per budget.
    pub unassigned: Vec<(Vec<Tier>, u64)>,
}

/// A profile that does not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The shortfall in bytes for each short budget, with the tiers it covers.
    pub shortfall: Vec<(Vec<Tier>, u64)>,
}

/// A ledger error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// The profile does not fit its budgets.
    Refused(Refusal),
    /// The profile is not valid.
    Invalid(String),
    /// A new context does not fit. The shortfall is in bytes for each short budget.
    ContextRefused(Refusal),
    /// A staging copy does not fit (spec 4.2). The shortfall is in bytes for each short budget.
    StagingRefused(Refusal),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::Refused(refusal) => {
                write!(formatter, "the profile does not fit: ")?;
                write_shortfall(formatter, refusal)
            }
            LedgerError::Invalid(message) => write!(formatter, "{message}"),
            LedgerError::ContextRefused(refusal) => {
                write!(formatter, "the new context does not fit: ")?;
                write_shortfall(formatter, refusal)
            }
            LedgerError::StagingRefused(refusal) => {
                write!(formatter, "the copy of the state does not fit in memory: ")?;
                write_shortfall(formatter, refusal)?;
                write!(
                    formatter,
                    "; try again when other requests end, or ask for a smaller context"
                )
            }
        }
    }
}

fn write_shortfall(formatter: &mut std::fmt::Formatter<'_>, refusal: &Refusal) -> std::fmt::Result {
    for (index, (tiers, bytes)) in refusal.shortfall.iter().enumerate() {
        let names: Vec<String> = tiers
            .iter()
            .map(|tier| match tier {
                Tier::Device(id) => format!("device {id}"),
                Tier::Host => "host".to_owned(),
            })
            .collect();
        let separator = if index == 0 { "" } else { "; " };
        write!(
            formatter,
            "{separator}the budget of {} is short by {bytes} bytes",
            names.join(", ")
        )?;
    }
    Ok(())
}

impl std::error::Error for LedgerError {}

/// The machine's ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    budgets: Vec<Budget>,
    engines: Vec<EngineAllocation>,
    /// The bytes that no term holds, one entry per budget (spec 4.4 step 4). A new context takes from them.
    unassigned: Vec<u128>,
    /// The terms with the lifetime `context` of each admitted engine, by engine name (spec 4.2).
    context_terms: BTreeMap<String, Vec<(String, Tier, u64)>>,
    /// The live contexts of each engine.
    contexts: BTreeMap<String, u64>,
    /// The staging bytes that are charged now, by the index of the budget (spec 4.2).
    staged: BTreeMap<usize, u128>,
}

impl Ledger {
    /// Make an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// The bytes that no term holds, one entry per budget.
    pub fn unassigned(&self) -> Vec<(Vec<Tier>, u64)> {
        self.budgets
            .iter()
            .zip(&self.unassigned)
            .map(|(budget, left)| (budget.tiers.clone(), (*left).min(u64::MAX as u128) as u64))
            .collect()
    }

    /// The live contexts of an engine.
    pub fn live_contexts(&self, engine: &str) -> u64 {
        self.contexts.get(engine).copied().unwrap_or(0)
    }

    /// Admit a new context of an engine (spec 4.2, 4.3).
    ///
    /// The ledger charges every term of the engine with the lifetime `context` against the unassigned bytes of its
    /// budget. A refusal changes nothing.
    pub fn admit_context(&mut self, engine: &str) -> Result<(), LedgerError> {
        let result = self.admit_context_unrecorded(engine);
        match &result {
            Ok(()) => self.record_context_bytes(engine),
            Err(LedgerError::ContextRefused(refusal)) => record_refusal(engine, refusal),
            Err(_) => {}
        }
        result
    }

    fn admit_context_unrecorded(&mut self, engine: &str) -> Result<(), LedgerError> {
        let terms = self
            .context_terms
            .get(engine)
            .ok_or_else(|| LedgerError::Invalid(format!("engine {engine} is not admitted: admit its profile first")))?;
        let mut need = vec![0u128; self.budgets.len()];
        for (_, tier, bytes) in terms {
            need[budget_index_in(&self.budgets, *tier)?] += *bytes as u128;
        }
        let shortfall: Vec<(Vec<Tier>, u64)> = self
            .budgets
            .iter()
            .enumerate()
            .filter(|(index, _)| need[*index] > self.unassigned[*index])
            .map(|(index, budget)| {
                let bytes = (need[index] - self.unassigned[index]).min(u64::MAX as u128) as u64;
                (budget.tiers.clone(), bytes)
            })
            .collect();
        if !shortfall.is_empty() {
            return Err(LedgerError::ContextRefused(Refusal { shortfall }));
        }
        for (index, bytes) in need.iter().enumerate() {
            self.unassigned[index] -= bytes;
        }
        *self.contexts.entry(engine.to_owned()).or_insert(0) += 1;
        Ok(())
    }

    /// Release the bytes of an ended context. A release with no live context is an error.
    pub fn release_context(&mut self, engine: &str) -> Result<(), LedgerError> {
        let live = self.contexts.get(engine).copied().unwrap_or(0);
        if live == 0 {
            return Err(LedgerError::Invalid(format!(
                "engine {engine} has no live context to release: release a context once"
            )));
        }
        let terms = self.context_terms.get(engine).cloned().unwrap_or_default();
        for (_, tier, bytes) in terms {
            let index = budget_index_in(&self.budgets, tier)?;
            self.unassigned[index] += bytes as u128;
        }
        self.contexts.insert(engine.to_owned(), live - 1);
        self.record_context_bytes(engine);
        Ok(())
    }

    fn record_context_bytes(&self, engine: &str) {
        let live = self.contexts.get(engine).copied().unwrap_or(0);
        let Some(terms) = self.context_terms.get(engine) else {
            return;
        };
        let instruments = instruments();
        for (name, tier, bytes) in terms {
            instruments.ledger_bytes(tier_label(*tier), engine, name, bytes.saturating_mul(live));
        }
    }

    /// Charge staging bytes of an engine in one tier (spec 4.2).
    ///
    /// Staging is a copy of engine state in host memory, for example a mark or the buffer of a snapshot. The ledger
    /// takes the bytes from the unassigned bytes of the budget of the tier. A refusal changes nothing. The caller
    /// releases the same bytes with [`Ledger::release_staging`] when it drops the copy.
    pub fn admit_staging(&mut self, engine: &str, tier: Tier, bytes: u64) -> Result<(), LedgerError> {
        let result = self.admit_staging_unrecorded(engine, tier, bytes);
        if let Err(LedgerError::StagingRefused(refusal)) = &result {
            record_refusal(engine, refusal);
        }
        result
    }

    fn admit_staging_unrecorded(&mut self, engine: &str, tier: Tier, bytes: u64) -> Result<(), LedgerError> {
        if !self.context_terms.contains_key(engine) {
            return Err(LedgerError::Invalid(format!(
                "engine {engine} is not admitted: admit its profile first"
            )));
        }
        let index = budget_index_in(&self.budgets, tier)?;
        if u128::from(bytes) > self.unassigned[index] {
            let short = (u128::from(bytes) - self.unassigned[index]).min(u64::MAX as u128) as u64;
            return Err(LedgerError::StagingRefused(Refusal {
                shortfall: vec![(self.budgets[index].tiers.clone(), short)],
            }));
        }
        self.unassigned[index] -= u128::from(bytes);
        *self.staged.entry(index).or_insert(0) += u128::from(bytes);
        Ok(())
    }

    /// Release staging bytes. The bytes must be charged now: a release of more is an error that changes nothing.
    pub fn release_staging(&mut self, tier: Tier, bytes: u64) -> Result<(), LedgerError> {
        let index = budget_index_in(&self.budgets, tier)?;
        let staged = self.staged.get(&index).copied().unwrap_or(0);
        if u128::from(bytes) > staged {
            return Err(LedgerError::Invalid(format!(
                "a release of {bytes} staging bytes is more than the {staged} bytes that are charged: release what was charged"
            )));
        }
        self.unassigned[index] += u128::from(bytes);
        self.staged.insert(index, staged - u128::from(bytes));
        Ok(())
    }

    /// Admit a profile. A refusal changes nothing.
    ///
    /// The ledger records `nagaya.ledger.bytes` for each granted term, or `nagaya.ledger.refusals` (spec 9.4).
    pub fn admit(&mut self, profile: &Profile) -> Result<Admission, LedgerError> {
        let result = self.admit_unrecorded(profile);
        match &result {
            Ok(admission) => record_admission(profile, admission),
            Err(LedgerError::Refused(refusal)) => record_refusal(PROFILE_ENGINE, refusal),
            Err(LedgerError::Invalid(_) | LedgerError::ContextRefused(_) | LedgerError::StagingRefused(_)) => {}
        }
        result
    }

    /// Give what `admit` would give for a profile, with no change and no record (spec 3.3a).
    ///
    /// A swap asks it whether an engine fits beside the engines that stay loaded, before it unloads any of them.
    pub fn check(&self, profile: &Profile) -> Result<Admission, LedgerError> {
        self.clone().admit_unrecorded(profile)
    }

    fn admit_unrecorded(&mut self, profile: &Profile) -> Result<Admission, LedgerError> {
        if profile.budgets.iter().any(|budget| budget.tiers.is_empty()) {
            return Err(LedgerError::Invalid("budget has no tiers".to_owned()));
        }
        let mut mandatory = vec![0u128; profile.budgets.len()];
        let mut minimums = vec![0u128; profile.budgets.len()];
        for engine in &profile.engines {
            // A context term is charged at `ctx_new`, not at `open` (spec 4.2).
            for term in engine
                .limits
                .memory
                .mandatory
                .iter()
                .filter(|term| term.lifetime != Lifetime::Context)
            {
                let index = budget_index(profile, machine_tier(engine, term.tier)?)?;
                mandatory[index] += term.bytes as u128;
            }
            for term in &engine.limits.memory.elastic {
                if term.unit_bytes == 0 || term.minimum > term.maximum {
                    return Err(LedgerError::Invalid(format!("bad elastic term {}", term.name)));
                }
                let index = budget_index(profile, machine_tier(engine, term.tier)?)?;
                minimums[index] += term.minimum as u128 * term.unit_bytes as u128;
            }
        }

        let mut shortfall = Vec::new();
        for (index, budget) in profile.budgets.iter().enumerate() {
            let required = mandatory[index] + minimums[index];
            if required > budget.bytes as u128 {
                let bytes = (required - budget.bytes as u128).min(u64::MAX as u128) as u64;
                shortfall.push((budget.tiers.clone(), bytes));
            }
        }
        if !shortfall.is_empty() {
            return Err(LedgerError::Refused(Refusal { shortfall }));
        }

        let spare: Vec<u128> = (0..profile.budgets.len())
            .map(|index| profile.budgets[index].bytes as u128 - mandatory[index] - minimums[index])
            .collect();
        let total_share: u128 = profile.engines.iter().map(|engine| engine.share as u128).sum();

        let mut engines = Vec::new();
        let mut allocated = vec![0u128; profile.budgets.len()];
        for engine in &profile.engines {
            let mandatory_allocation = machine_terms(engine, |lifetime| lifetime != Lifetime::Context)?;
            let mut elastic_allocation = Vec::new();
            for term in &engine.limits.memory.elastic {
                let index = budget_index(profile, machine_tier(engine, term.tier)?)?;
                let engine_minimum: u128 = engine
                    .limits
                    .memory
                    .elastic
                    .iter()
                    .filter(|other| {
                        machine_tier(engine, other.tier).and_then(|tier| budget_index(profile, tier)) == Ok(index)
                    })
                    .map(|other| other.minimum as u128 * other.unit_bytes as u128)
                    .sum();
                let engine_part = (spare[index] * engine.share as u128)
                    .checked_div(total_share)
                    .unwrap_or(0);
                let term_minimum = term.minimum as u128 * term.unit_bytes as u128;
                let term_spare = (engine_part * term_minimum).checked_div(engine_minimum).unwrap_or(0);
                let total = term_minimum + term_spare;
                let units = (total / term.unit_bytes as u128).min(term.maximum as u128);
                let bytes = units * term.unit_bytes as u128;
                allocated[index] += bytes;
                elastic_allocation.push((term.name.clone(), bytes.min(u64::MAX as u128) as u64));
            }
            engines.push(EngineAllocation {
                name: engine.name.clone(),
                mandatory: mandatory_allocation,
                elastic: elastic_allocation,
            });
        }

        let mut left: Vec<u128> = (0..profile.budgets.len())
            .map(|index| {
                (profile.budgets[index].bytes as u128)
                    .saturating_sub(mandatory[index])
                    .saturating_sub(allocated[index])
            })
            .collect();

        let mut context_terms = BTreeMap::new();
        for engine in &profile.engines {
            let terms = named_machine_terms(engine, |lifetime| lifetime == Lifetime::Context)?;
            context_terms.insert(engine.name.clone(), terms);
        }

        // The charges that live now stay charged. A later release finds its charge, and the unassigned bytes stay
        // true. Decided by senpai 2026-10-20 night (vetoable); see spec 4.2 "Staging".
        let mut live = vec![0u128; profile.budgets.len()];
        for (engine, count) in &self.contexts {
            if *count == 0 {
                continue;
            }
            let Some(terms) = context_terms.get(engine) else {
                return Err(LedgerError::Invalid(format!(
                    "engine {engine} has {count} live contexts and is absent from the new profile: release its contexts first, or keep the engine"
                )));
            };
            for (_, tier, bytes) in terms {
                live[budget_index(profile, *tier)?] += *bytes as u128 * *count as u128;
            }
        }
        let mut staged = BTreeMap::new();
        for (old, bytes) in &self.staged {
            let mut target = None;
            for tier in &self.budgets[*old].tiers {
                let index = budget_index(profile, *tier)?;
                if target.is_some_and(|found| found != index) {
                    return Err(LedgerError::Invalid(
                        "a budget with live staging would split across two budgets: end the staging, or keep the budget"
                            .to_owned(),
                    ));
                }
                target = Some(index);
            }
            let index = target.expect("an admitted budget has a tier");
            live[index] += *bytes;
            *staged.entry(index).or_insert(0) += *bytes;
        }
        let shortfall: Vec<(Vec<Tier>, u64)> = (0..profile.budgets.len())
            .filter(|index| live[*index] > left[*index])
            .map(|index| {
                let bytes = (live[index] - left[index]).min(u64::MAX as u128) as u64;
                (profile.budgets[index].tiers.clone(), bytes)
            })
            .collect();
        if !shortfall.is_empty() {
            return Err(LedgerError::Refused(Refusal { shortfall }));
        }
        for (index, bytes) in live.iter().enumerate() {
            left[index] -= bytes;
        }
        let unassigned: Vec<(Vec<Tier>, u64)> = profile
            .budgets
            .iter()
            .zip(&left)
            .map(|(budget, left)| (budget.tiers.clone(), (*left).min(u64::MAX as u128) as u64))
            .collect();

        self.budgets = profile.budgets.clone();
        self.engines = engines.clone();
        self.unassigned = left;
        self.context_terms = context_terms;
        self.staged = staged;
        Ok(Admission { engines, unassigned })
    }
}

/// The `engine` label of a refusal of a whole profile: no one engine is short, the budget is.
const PROFILE_ENGINE: &str = "all";

fn tier_label(tier: Tier) -> TierLabel {
    match tier {
        Tier::Device(id) => TierLabel::Device(id),
        Tier::Host => TierLabel::Host,
    }
}

/// Count one refusal for each short budget. The label `tier` names all the tiers of that budget.
fn record_refusal(engine: &str, refusal: &Refusal) {
    for (tiers, _) in &refusal.shortfall {
        let labels: Vec<TierLabel> = tiers.iter().copied().map(tier_label).collect();
        instruments().ledger_refusal(&labels, engine, RefusalGate::Budget);
    }
}

/// Record the bytes of each term that the admission granted. A term of the lifetime `context` waits for `ctx_new`.
fn record_admission(profile: &Profile, admission: &Admission) {
    let instruments = instruments();
    for (plan, allocation) in profile.engines.iter().zip(&admission.engines) {
        let terms = plan
            .limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.lifetime != Lifetime::Context);
        for (term, (tier, bytes)) in terms.zip(&allocation.mandatory) {
            instruments.ledger_bytes(tier_label(*tier), &plan.name, &term.name, *bytes);
        }
        for (term, (name, bytes)) in plan.limits.memory.elastic.iter().zip(&allocation.elastic) {
            let tier =
                machine_tier(plan, term.tier).expect("the ledger checked each tier when it admitted the profile");
            instruments.ledger_bytes(tier_label(tier), &plan.name, name, *bytes);
        }
    }
}

/// Map the tier of a term to the machine tier (spec 2.4). A term names `device(i)`, a group index.
///
/// The machine device is `devices[i]` of the engine. A group index beyond the list is an error.
fn machine_tier(engine: &EnginePlan, tier: Tier) -> Result<Tier, LedgerError> {
    match tier {
        Tier::Host => Ok(Tier::Host),
        Tier::Device(index) => engine
            .devices
            .get(index as usize)
            .map(|device| Tier::Device(*device))
            .ok_or_else(|| {
                LedgerError::Invalid(format!(
                    "engine {} has a term on device({index}), but it has {} devices: fix the engine's limits or its profile",
                    engine.name,
                    engine.devices.len()
                ))
            }),
    }
}

/// The mandatory terms of an engine whose lifetime the filter accepts, with machine tiers.
fn machine_terms(engine: &EnginePlan, accept: impl Fn(Lifetime) -> bool) -> Result<Vec<(Tier, u64)>, LedgerError> {
    engine
        .limits
        .memory
        .mandatory
        .iter()
        .filter(|term| accept(term.lifetime))
        .map(|term| Ok((machine_tier(engine, term.tier)?, term.bytes)))
        .collect()
}

/// The mandatory terms of an engine whose lifetime the filter accepts, with term names and machine tiers.
fn named_machine_terms(
    engine: &EnginePlan,
    accept: impl Fn(Lifetime) -> bool,
) -> Result<Vec<(String, Tier, u64)>, LedgerError> {
    engine
        .limits
        .memory
        .mandatory
        .iter()
        .filter(|term| accept(term.lifetime))
        .map(|term| Ok((term.name.clone(), machine_tier(engine, term.tier)?, term.bytes)))
        .collect()
}

fn budget_index(profile: &Profile, tier: Tier) -> Result<usize, LedgerError> {
    budget_index_in(&profile.budgets, tier)
}

fn budget_index_in(budgets: &[Budget], tier: Tier) -> Result<usize, LedgerError> {
    let mut found = None;
    for (index, budget) in budgets.iter().enumerate() {
        if budget.tiers.contains(&tier) {
            if found.is_some() {
                return Err(LedgerError::Invalid("tier has two budgets".to_owned()));
            }
            found = Some(index);
        }
    }
    found.ok_or_else(|| LedgerError::Invalid("term tier has no budget".to_owned()))
}
