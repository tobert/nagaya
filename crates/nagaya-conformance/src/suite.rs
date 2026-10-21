//! The suite report: every gate, its outcome, and the profiles for the admission gate.

use std::fmt;

use nagaya_abi::{
    ChainLimits, Determinism, ElasticTerm, Engine, Inputs, Lifetime, Limits, MandatoryTerm, MemoryLimits, Tier,
};
use nagaya_ledger::{Budget, EnginePlan, Ledger, Profile};

use crate::gate::{
    Outcome, admission_refusal_changes_nothing, park_and_restore, quantum_identity, quantum_isolation,
    quantum_resume_fit, quantum_same_boundaries, seal_only_at_boundaries, segments_exact_bits,
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
    // A temporary in a `vec![...]` lives until the end of the whole statement. Each gate that takes one engine gets
    // its own statement, so that its engine is dropped before the next gate. An engine with a large pool (mk: 12
    // GiB) cannot hold the engines of all gates at once.
    let park = park_and_restore(&mut make());
    let seal = seal_only_at_boundaries(&mut make());
    let identity = quantum_identity(make);
    let isolation = quantum_isolation(make);
    let boundaries = quantum_same_boundaries(make);
    let resume_fit = quantum_resume_fit(&mut make());
    let admission = admission_gate();
    let segments = segments_exact_bits(make);
    let gates = vec![
        GateResult {
            name: "park_and_restore",
            outcome: park,
        },
        GateResult {
            name: "seal_only_at_boundaries",
            outcome: seal,
        },
        GateResult {
            name: "quantum_identity",
            outcome: identity,
        },
        GateResult {
            name: "quantum_isolation",
            outcome: isolation,
        },
        GateResult {
            name: "quantum_same_boundaries",
            outcome: boundaries,
        },
        GateResult {
            name: "quantum_resume_fit",
            outcome: resume_fit,
        },
        GateResult {
            name: "admission_refusal_changes_nothing",
            outcome: admission,
        },
        GateResult {
            name: "segments_exact_bits",
            outcome: segments,
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
        checkpoint_grain: None,
        feed: None,
        residual_row_bytes: 0,
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
            devices: vec![0],
            share: 1,
        }],
    }
}

/// One profile that fits 16 bytes, and one that needs 1 byte in a 0-byte budget.
fn refusal_profiles() -> (Profile, Profile) {
    (gate_profile(16, 1), gate_profile(0, 1))
}

fn elastic_limits() -> Limits {
    let mut limits = gate_limits(0);
    limits.memory.elastic = vec![ElasticTerm {
        name: "gate-cache".to_owned(),
        tier: Tier::Host,
        unit_bytes: 8,
        minimum: 2,
        maximum: 4,
        kinds: Vec::new(),
    }];
    limits
}

fn elastic_profile(budget_bytes: u64) -> Profile {
    Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: budget_bytes,
        }],
        engines: vec![EnginePlan {
            name: "gate".to_owned(),
            limits: elastic_limits(),
            devices: vec![0],
            share: 1,
        }],
    }
}

/// Two 8-byte units fit 16 bytes and do not fit 8 bytes.
fn elastic_refusal_profiles() -> (Profile, Profile) {
    (elastic_profile(16), elastic_profile(8))
}

/// Run the admission gate on the mandatory path first, then on the elastic path.
fn admission_gate() -> Outcome {
    let (fitting, refused) = refusal_profiles();
    let outcome = admission_refusal_changes_nothing(&mut Ledger::new(), &fitting, &refused);
    if outcome.failed() {
        return outcome;
    }
    let (fitting, refused) = elastic_refusal_profiles();
    admission_refusal_changes_nothing(&mut Ledger::new(), &fitting, &refused)
}
