//! The suite report: every gate, its outcome, and the profiles for the admission gate.

use std::fmt;

use nagaya_abi::{ChainLimits, Determinism, Engine, Inputs, Lifetime, Limits, MandatoryTerm, MemoryLimits, Tier};
use nagaya_ledger::{Budget, EnginePlan, Ledger, Profile};

use crate::gate::{
    Outcome, admission_refusal_changes_nothing, park_and_restore, quantum_identity, quantum_isolation,
    quantum_resume_fit, quantum_same_boundaries, seal_only_at_boundaries,
};

/// One gate in a suite run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateResult {
    /// The gate name.
    pub name: &'static str,
    /// The gate result.
    pub outcome: Outcome,
}

/// The results of one suite run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Suite {
    /// One entry per gate.
    pub gates: Vec<GateResult>,
}

impl Suite {
    /// True when no gate failed. A gate that does not apply is not a failure.
    pub fn passed(&self) -> bool {
        self.gates.iter().all(|gate| !gate.outcome.failed())
    }

    /// The failed gates.
    pub fn failures(&self) -> impl Iterator<Item = &GateResult> {
        self.gates.iter().filter(|gate| gate.outcome.failed())
    }

    /// The gates that do not apply to this engine.
    pub fn not_applicable(&self) -> impl Iterator<Item = &GateResult> {
        self.gates
            .iter()
            .filter(|gate| matches!(gate.outcome, Outcome::NotApplicable(_)))
    }
}

impl fmt::Display for Suite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for gate in &self.gates {
            writeln!(formatter, "{}: {}", gate.name, gate.outcome)?;
        }
        Ok(())
    }
}

/// Run every gate of spec 8.3 on fresh engines from `make`.
pub fn run_all<E: Engine>(make: &mut dyn FnMut() -> E) -> Suite {
    let (fitting, refused) = refusal_profiles();
    let gates = vec![
        GateResult {
            name: "park_and_restore",
            outcome: park_and_restore(&mut make()),
        },
        GateResult {
            name: "seal_only_at_boundaries",
            outcome: seal_only_at_boundaries(&mut make()),
        },
        GateResult {
            name: "quantum_identity",
            outcome: quantum_identity(make),
        },
        GateResult {
            name: "quantum_isolation",
            outcome: quantum_isolation(make),
        },
        GateResult {
            name: "quantum_same_boundaries",
            outcome: quantum_same_boundaries(make),
        },
        GateResult {
            name: "quantum_resume_fit",
            outcome: quantum_resume_fit(&mut make()),
        },
        GateResult {
            name: "admission_refusal_changes_nothing",
            outcome: admission_refusal_changes_nothing(&mut Ledger::new(), &fitting, &refused),
        },
    ];
    Suite { gates }
}

fn gate_limits(mandatory_bytes: u64) -> Limits {
    Limits {
        capabilities: Vec::new(),
        parts: Vec::new(),
        inputs: Inputs::Tokens,
        devices: Vec::new(),
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        slots: Vec::new(),
        max_quantum_ms: 0,
        memory: MemoryLimits {
            mandatory: vec![MandatoryTerm {
                name: "gate".to_owned(),
                tier: Tier::Host,
                bytes: mandatory_bytes,
                lifetime: Lifetime::Engine,
                kinds: Vec::new(),
            }],
            elastic: Vec::new(),
        },
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        segments_exact: true,
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        taps: Vec::new(),
    }
}

fn gate_profile(budget_bytes: u64, mandatory_bytes: u64) -> Profile {
    Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: budget_bytes,
        }],
        engines: vec![EnginePlan {
            name: "gate".to_owned(),
            limits: gate_limits(mandatory_bytes),
            share: 1,
        }],
    }
}

/// One profile that fits 16 bytes, and one that needs 1 byte in a 0-byte budget.
fn refusal_profiles() -> (Profile, Profile) {
    (gate_profile(16, 1), gate_profile(0, 1))
}
