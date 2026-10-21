//! The conformance suite: the properties of spec 8.3, generic over the `Engine` trait.
//!
//! The suite has these gates:
//!
//! - `park_and_restore` (quantum condition 3);
//! - `seal_only_at_boundaries`;
//! - `quantum_identity` (condition 1);
//! - `quantum_isolation` (condition 5);
//! - `quantum_same_boundaries` (conditions 2 and 7);
//! - `quantum_resume_fit` (condition 4);
//! - `admission_refusal_changes_nothing` (spec 4.3);
//! - `segments_exact_bits`: for an engine that declares `segments_exact`, a segment gives the same bits in passes
//!   with one, two and the most segments, and in one pass or two (spec 5.7);
//! - `stage_path_bits`: for an engine with stages, `Engine::pass` and the stage path give the same outputs and parts
//!   (engine-abi D1);
//! - `feed_rule_bits`: for an engine with a feed rule, a span fed after a restore gives the bits of one job
//!   (engine-abi D11).
//!
//! Condition 6 needs a lab-only state that changes bits. The suite cannot see such a state, so no gate checks it.
//! The scheduler owns condition 5 between two stages of one block; the gate checks one pass against another state.
//!
//! Each gate takes the engine. `run_all` makes a fresh engine for each gate. Milestone 8 calls it on mk.

mod compare;
mod gate;
mod stages;
mod suite;

pub use compare::{Difference, Failure, compare_bytes, compare_f32_rows};
pub use gate::{
    Admit, Outcome, admission_refusal_changes_nothing, park_and_restore, quantum_identity, quantum_isolation,
    quantum_resume_fit, quantum_same_boundaries, seal_only_at_boundaries, segments_exact_bits,
};
pub use stages::{feed_rule_bits, stage_path_bits};
pub use suite::{GateResult, Suite, run_all};
