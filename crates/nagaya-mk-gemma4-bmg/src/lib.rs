//! The Gemma 4 26B-A4B engine on two Arc Pro B70 cards: a copy of mkviolet's engine (milestone 15s, step 4).
//!
//! Amy, 2026-10-21: "copy code in, we own all of it 100%". The copy keeps mkviolet's bits. `engine` holds the `Engine`
//! trait with two stages (step 5), and `spec` holds its limits and rules. The nagaya's stage threads take the place of
//! mkviolet's card locks and `pipeline_stage`. The kernels are OpenCL C in `kernels/`, and `build.rs` compiles them
//! with `ocloc` for `bmg-g31` in the crate `nagaya-mk-bmg`, which this engine shares with the speech engine.
//! The parts that touch a card need the feature `xe`. With no feature, the crate builds and tests with no card.

// The copy keeps mkviolet's code as it was, so these lints wait for the port of each part.
#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::len_without_is_empty,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::missing_safety_doc,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

#[cfg(feature = "xe")]
pub mod engine;
pub mod gemma4;
#[cfg(feature = "xe")]
pub use nagaya_mk_bmg::level_zero;
pub use nagaya_mk_bmg::protocol;
pub mod safetensors;
pub mod sharding;
pub mod spec;

/// The kernel binary of this engine for `bmg-g31`, which `nagaya-mk-bmg` compiled from its `kernels/megakernel.cl`.
#[cfg(feature = "xe")]
pub const MEGAKERNEL_BIN: &[u8] = nagaya_mk_bmg::GEMMA4_KERNEL;

/// The name of the kernel's entry point in `MEGAKERNEL_BIN`.
pub const MEGAKERNEL_ENTRY: &str = nagaya_mk_bmg::KERNEL_ENTRY;
