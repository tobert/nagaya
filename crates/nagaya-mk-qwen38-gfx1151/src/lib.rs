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
pub use worker::{MkError, MkPaths};
