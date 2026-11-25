//! The support of the retrieval test: the needles, the haystack, the snapshot files and, with
//! the feature `cuda`, the calls of the engine. The host parts need no GPU, so `retrieval_host.rs` tests them.

#![allow(dead_code)]

pub mod haystack;
pub mod needles;
pub mod snapshot;

#[cfg(feature = "cuda")]
pub mod gpu;
