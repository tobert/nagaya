//! The conformance suite of spec 8.3 on the GB10. The engine has parts, slots, coverage and a feed
//! rule, and no seal, no stages and no `segments_exact`, as the LFM3 engine has none. So the gates that apply are the
//! park, the identity, the isolation, the fit, the admission and the feed rule, and the four gates of the list below do
//! not apply.
//!
//! Each gate opens a new engine, and an open reads the whole checkpoint, so the run takes minutes.
//!
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_engine_conformance -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod engine_support;

use std::time::Duration;

use nagaya_conformance::run_all;
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine;

use engine_support::{config, model_dir, open, watchdog};

#[test]
fn the_conformance_suite_passes_on_the_gpu() {
    let Some(dir) = model_dir("cuda_engine_conformance") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(3 * 3600), "the conformance suite");
    // Two slots, so the park and the isolation have a second state. The feed gate holds 1,536 tokens.
    let config = config(&dir, &[2048, 2048], CacheFormat::F32);
    let suite = run_all(&mut || -> Inkling2Sm121Engine { open(config.clone()) });
    println!("{suite}");
    assert!(suite.passed(), "{suite}");
    let names: Vec<&str> = suite.not_applicable().map(|gate| gate.name).collect();
    assert_eq!(
        names,
        vec![
            "seal_only_at_boundaries",
            "quantum_same_boundaries",
            "segments_exact_bits",
            "stage_path_bits"
        ],
        "{suite}"
    );
}
