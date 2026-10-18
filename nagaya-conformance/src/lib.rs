//! The conformance suite: the properties of spec 8.3, generic over the `Engine` trait.

mod compare;

pub use compare::{Difference, Failure, compare_bytes, compare_f32_rows};
