//! The op tests of mk: each op kernel of the engine, called through its C ABI from a library of its own.
//!
//! These tests port mk's Python tests in `tests/ops` at mk commit `555a2a3`. A test module has the name of its mk file.
//! `build.rs` builds the op libraries under the feature `hip`. Run with `--test-threads=1`, with `megakernel.service`
//! and `nagaya.service` inactive, and under the GPU lock of the machine.

#![cfg(feature = "hip")]

mod fixture;
mod inputs;
mod ops_lib;

mod child;

mod abi;
mod attn;
mod attn_a;
mod gdn;
mod hc;
mod moe;
mod qmm;
mod tiles;
