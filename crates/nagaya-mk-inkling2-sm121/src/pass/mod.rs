//! The pass of Inkling 2 on the GB10: the task ring of a run, a decode step and a prefill block (milestone 37 step 10).
//!
//! `layout` holds the plan of the runs and the byte layout of the buffers. It needs no GPU.

#[cfg(feature = "cuda")]
pub mod arena;
pub mod desc;
#[cfg(feature = "cuda")]
pub mod dev;
pub mod layout;
#[cfg(feature = "cuda")]
pub mod run;
#[cfg(feature = "cuda")]
pub mod slot;
#[cfg(feature = "cuda")]
pub mod weights;
