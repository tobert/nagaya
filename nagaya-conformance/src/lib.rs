//! The conformance suite: the properties of spec 8.3, generic over the `Engine` trait.

mod compare;
mod gate;

pub use compare::{Difference, Failure, compare_bytes, compare_f32_rows};
pub use gate::{
    Admit, Outcome, admission_refusal_changes_nothing, park_and_restore, quantum_identity, quantum_isolation,
    quantum_resume_fit, quantum_same_boundaries, seal_only_at_boundaries,
};
