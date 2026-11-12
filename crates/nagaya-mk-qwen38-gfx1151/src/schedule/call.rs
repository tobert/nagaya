//! The plan of a pass, as the Rust pass build asks for it (`pass.rs`).
//!
//! An error of the schedule is a defect of the pass build, so it stops the program with the message.

use super::{Plan, plain, plan};
use crate::stage::Stage;

/// The plan of a pass: the schedule with `elide`, else each stage in build order with a barrier after it.
pub(crate) fn pass_plan(stages: &[Stage], window_starts: &[u32], elide: bool, tail: bool) -> Plan {
    if !elide {
        return plain(stages.len() as u32);
    }
    plan(stages, window_starts, tail).unwrap_or_else(|error| panic!("the schedule of the test stages failed: {error}"))
}
