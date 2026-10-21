//! The mk engine: an FFI binding to `libmkengine.so` and the `Engine` trait on top.
//!
//! The library is loaded from `MK_ENGINE_LIB` at run time. The three shard paths come from `MK_SHARDS`. One worker
//! thread makes every mk call, because mk's HIP device is the calling thread's current device. The engine is never
//! closed: mk's HIP teardown segfaults in the process (spec 3.3).

mod binding;
mod engine;
mod gguf;
mod spec;
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
