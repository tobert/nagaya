//! The mk engine: mk's HIP code in `hip/`, an FFI binding to it, and the `Engine` trait on top.
//!
//! `build.rs` compiles the engine, and the crate links it statically, under the feature `hip`. The three shard paths
//! come from `MK_SHARDS`. One worker thread makes every mk call, because mk's HIP device is the calling thread's
//! current device. The engine is never closed: mk's HIP teardown segfaults in the process (spec 3.3).

mod binding;
mod dequant;
mod engine;
mod gguf;
#[cfg(test)]
mod gguf_shard_tests;
mod spec;
mod table;
mod worker;

pub use engine::{MkEngine, open};
pub use spec::{CONTEXT_CLASSES, ContextClass, MAX_CONTEXT};
pub use worker::{ClassReading, MkError, MkPaths};

/// The limits that `open` gives for an arena of `weights_bytes`. It reads no weights and opens no device.
///
/// mk knows the arena size only after the weights load. The caller of `plan` supplies the size, and `open` gives the
/// real one in `limits()`. The two must be equal (spec 2.4).
pub fn plan(weights_bytes: u64) -> nagaya_abi::Limits {
    spec::limits(weights_bytes)
}
