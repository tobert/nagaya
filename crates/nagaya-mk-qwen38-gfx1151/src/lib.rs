//! The Qwen3.8 engine: its HIP kernels in `hip/`, the Rust host code around them, and the `Engine` trait on top.
//!
//! `build.rs` compiles the kernels and the C++ runtime layer, and the crate links them statically, under the feature
//! `hip`. The Rust host makes every decision (milestone 15 step 6): `gguf.rs` reads the shards, `arena.rs` fills the
//! weight arena, `context.rs` holds the contexts, `ple.rs` gathers the PLE rows, `pass.rs` builds the stages,
//! `schedule.rs` orders them, and `host.rs` runs the pass. The three shard paths come from `QWEN38_SHARDS`. One worker
//! thread makes every engine call, because the HIP device is the calling thread's current device. The engine is never
//! closed: the HIP teardown segfaults in the process (spec 3.3).

mod arena;
mod binding;
mod context;
mod dequant;
mod engine;
mod gguf;
#[cfg(test)]
mod gguf_shard_tests;
mod host;
#[cfg(test)]
mod layout;
mod pass;
mod ple;
mod schedule;
mod spec;
mod stage;
mod worker;

pub use engine::{Hold, Qwen38Engine, describe, hold, open};
pub use spec::{CONTEXT_CLASSES, ContextClass, MAX_CONTEXT};
pub use worker::{ClassReading, Qwen38Error, Qwen38Paths};

/// The limits that `open` gives for an arena of `weights_bytes`. It reads no weights and opens no device.
///
/// The engine knows the arena size only after the weights load. The caller of `plan` supplies the size, and `open`
/// gives the real one in `limits()`. The two must be equal (spec 2.4).
pub fn plan(weights_bytes: u64) -> nagaya_abi::Limits {
    spec::limits(weights_bytes)
}
