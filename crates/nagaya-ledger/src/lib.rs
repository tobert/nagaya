//! Budgets, terms, admission, shares, and reclaim order.

use nagaya_abi::{Limits, Tier};

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
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::Refused(refusal) => {
                write!(formatter, "the profile does not fit: {:?}", refusal.shortfall)
            }
            LedgerError::Invalid(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for LedgerError {}

/// The machine's ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    budgets: Vec<Budget>,
    engines: Vec<EngineAllocation>,
}

impl Ledger {
    /// Make an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit a profile. A refusal changes nothing.
    pub fn admit(&mut self, profile: &Profile) -> Result<Admission, LedgerError> {
        if profile.budgets.iter().any(|budget| budget.tiers.is_empty()) {
            return Err(LedgerError::Invalid("budget has no tiers".to_owned()));
        }
        let mut mandatory = vec![0u128; profile.budgets.len()];
        let mut minimums = vec![0u128; profile.budgets.len()];
        for engine in &profile.engines {
            for term in &engine.limits.memory.mandatory {
                let index = budget_index(profile, term.tier)?;
                mandatory[index] += term.bytes as u128;
            }
            for term in &engine.limits.memory.elastic {
                if term.unit_bytes == 0 || term.minimum > term.maximum {
                    return Err(LedgerError::Invalid(format!("bad elastic term {}", term.name)));
                }
                let index = budget_index(profile, term.tier)?;
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
            let mandatory_allocation: Vec<(Tier, u64)> = engine
                .limits
                .memory
                .mandatory
                .iter()
                .map(|term| (term.tier, term.bytes))
                .collect();
            let mut elastic_allocation = Vec::new();
            for term in &engine.limits.memory.elastic {
                let index = budget_index(profile, term.tier)?;
                let engine_minimum: u128 = engine
                    .limits
                    .memory
                    .elastic
                    .iter()
                    .filter(|other| budget_index(profile, other.tier) == Ok(index))
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

        let unassigned: Vec<(Vec<Tier>, u64)> = profile
            .budgets
            .iter()
            .enumerate()
            .map(|(index, budget)| {
                let left = (budget.bytes as u128)
                    .saturating_sub(mandatory[index])
                    .saturating_sub(allocated[index]);
                (budget.tiers.clone(), left.min(u64::MAX as u128) as u64)
            })
            .collect();

        self.budgets = profile.budgets.clone();
        self.engines = engines.clone();
        Ok(Admission { engines, unassigned })
    }
}

fn budget_index(profile: &Profile, tier: Tier) -> Result<usize, LedgerError> {
    let mut found = None;
    for (index, budget) in profile.budgets.iter().enumerate() {
        if budget.tiers.contains(&tier) {
            if found.is_some() {
                return Err(LedgerError::Invalid("tier has two budgets".to_owned()));
            }
            found = Some(index);
        }
    }
    found.ok_or_else(|| LedgerError::Invalid("term tier has no budget".to_owned()))
}
