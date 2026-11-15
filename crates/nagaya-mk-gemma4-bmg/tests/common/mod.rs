//! The files of the golden tests, and the rule for a machine that has no copy of them.
//!
//! The tests are mkviolet's `src/golden.rs`. Each check is its own test file, so it runs in its own process: device
//! memory is free only when the process exits, and two engines do not fit on the cards (mkviolet `golden.rs`).
//!
//! An absent file skips the test, as for the Qwen3.8 GGUF (AGENTS "Tests that need a model file"). The variable
//! `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. A path that exists but is not readable fails it.
#![allow(dead_code)]

use nagaya_mk_gemma4_bmg::level_zero::L0Cluster;
use nagaya_mk_gemma4_bmg::sharding::Checkpoint;
use nagaya_mk_gemma4_bmg::{MEGAKERNEL_BIN, MEGAKERNEL_ENTRY};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const MODEL_VARIABLE: &str = "GEMMA4_MODEL_DIR";
pub const GOLDEN_VARIABLE: &str = "GEMMA4_GOLDEN_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";
/// The directory of the reference files when `GEMMA4_GOLDEN_DIR` is not set: under the workspace's `target/`.
pub const GOLDEN_DEFAULT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/gemma4-golden");

/// Relative L2 error per op. Measured worst case on 16 tokens is 7.6e-7: every op is f32
/// arithmetic on identical BF16 weights, so differences come from summation order.
pub const OP_TOL: f64 = 1e-5;
/// Last-token logits after all 30 layers: max absolute difference. Measured 3.4e-5 on the
/// 16-token chat prompt.
pub const LOGIT_TOL: f32 = 1e-3;
/// The same over 1270 tokens, where attention sums grow f32 drift: measured 5.7e-3. With
/// the sliding window disabled the difference is 3.6, so this still catches a mask error.
pub const LONG_LOGIT_TOL: f32 = 0.02;
/// Over 1270 tokens, f32 summation-order differences flip a few top-8 picks where HF's 8th
/// and 9th router probabilities nearly tie; the first flip measured has a gap of 1.5e-8.
pub const ROUTE_TIE: f32 = 1e-5;
/// A later position can first diverge after an earlier flip reaches it through attention,
/// on a wider gap: measured worst 3.3e-4 (54 of 1270 positions diverge somewhere).
pub const ROUTE_ORIGIN_TOL: f32 = 1e-3;
/// Layer 14's output over the last block after those flips; measured 3.3e-4.
pub const LONG_L14_TOL: f64 = 2e-3;

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

fn test_name() -> String {
    std::thread::current().name().unwrap_or("a test").to_string()
}

/// Check that `path` exists as a file or a directory. An absent path gives the skip line, or panics when
/// `NAGAYA_REQUIRE_MODELS=1`. Any other error panics.
fn check(path: &Path, what: &str, how: &str) -> Option<String> {
    match std::fs::metadata(path) {
        Ok(_) => None,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let missing = format!("{what} {} does not exist", path.display());
            if required() {
                panic!("{missing}. {REQUIRE_VARIABLE} is set. {how}");
            }
            Some(format!("skipped: {}: {missing}; {how}", test_name()))
        }
        Err(e) => panic!("cannot read {what} {}: {e}", path.display()),
    }
}

/// The Gemma 4 checkpoint directory from `GEMMA4_MODEL_DIR`, or `None` when it is not set.
fn model_dir_configured() -> Option<PathBuf> {
    std::env::var_os(MODEL_VARIABLE).map(PathBuf::from)
}

/// The directory of the reference files.
pub fn golden_dir() -> PathBuf {
    PathBuf::from(std::env::var_os(GOLDEN_VARIABLE).unwrap_or_else(|| GOLDEN_DEFAULT.into()))
}

/// True when the checkpoint or one of the reference files `golden` is absent, and the test must return. It prints
/// one line to stderr. Call it first in each test.
pub fn absent(golden: &[&str]) -> bool {
    let how = format!("set {MODEL_VARIABLE} to the Gemma 4 checkpoint directory");
    let mut skip = match model_dir_configured() {
        None if required() => panic!("{MODEL_VARIABLE} is not set and {REQUIRE_VARIABLE} is set. {how}."),
        None => Some(format!("skipped: {}: {MODEL_VARIABLE} is not set; {how}", test_name())),
        Some(dir) => check(&dir.join("config.json"), "the Gemma 4 config", &how),
    };
    for name in golden {
        if skip.is_some() {
            break;
        }
        let how = if name.ends_with(".safetensors") {
            format!("write it with tools/gemma4_golden.sh, or set {GOLDEN_VARIABLE}")
        } else {
            format!("write it with GEMMA4_KV_BASELINE=write and the layout to compare with, or set {GOLDEN_VARIABLE}")
        };
        skip = check(&golden_dir().join(name), "the reference file", &how);
    }
    match skip {
        Some(line) => {
            eprintln!("{line}");
            true
        }
        None => false,
    }
}

/// The checkpoint directory. A test that calls it has called `absent` before.
pub fn model_dir() -> PathBuf {
    model_dir_configured().expect("the test must call `absent` first")
}

/// The two cards of `NAGAYA_TEST_CARDS`, with the megakernel loaded.
pub fn cluster() -> L0Cluster {
    let cards = nagaya_mk_bmg::test_cards::cards(2);
    L0Cluster::init_at(MEGAKERNEL_BIN, MEGAKERNEL_ENTRY, &cards).expect("the two cards of the test open")
}

/// A reference file of tools/gemma4_reference.py.
pub fn reference(name: &str) -> Checkpoint {
    let dir = golden_dir();
    let path = dir.join(name);
    Checkpoint::open_files(&dir, std::slice::from_ref(&path))
        .unwrap_or_else(|e| panic!("{} (run tools/gemma4_golden.sh): {}", path.display(), e))
}

pub fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((*g - *w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

pub fn ref_ids(r: &Checkpoint) -> Vec<u32> {
    r.f32("input_ids").unwrap().0.iter().map(|&v| v as u32).collect()
}

pub fn top1(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 0..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Each token's 8 expert slots of OP_MOE_DOWN's output added in slot order, as OP_FFN_OUT does.
pub fn sum_slots(slots: &[f32]) -> Vec<f32> {
    use nagaya_mk_gemma4_bmg::gemma4::HIDDEN;
    use nagaya_mk_gemma4_bmg::protocol::TOP_K;
    slots
        .chunks(TOP_K * HIDDEN)
        .flat_map(|t| (0..HIDDEN).map(move |i| (0..TOP_K).fold(0.0f32, |s, p| s + t[p * HIDDEN + i])))
        .collect()
}
