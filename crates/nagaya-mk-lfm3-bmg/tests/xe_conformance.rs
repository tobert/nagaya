//! The conformance suite of spec 8.3 on a B70 card, for both models. The engine has parts, slots, coverage and a feed
//! rule, and no seal, no stages and no `segments_exact` (engine-abi "The first LFM3 binding"). So the gates that apply
//! are the park, the identity, the isolation, the fit, the admission and the feed rule.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_conformance -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use nagaya_conformance::run_all;

#[test]
fn the_conformance_suite_passes_on_the_card() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        // Two slots, so the park and the isolation have a second state. The feed gate holds 1,536 tokens.
        let suite = run_all(&mut || common::open(which, 2, 512));
        println!("{which:?}:\n{suite}");
        assert!(suite.passed(), "{which:?}: {suite}");
        let names: Vec<&str> = suite.not_applicable().map(|gate| gate.name).collect();
        assert_eq!(
            names,
            vec![
                "seal_only_at_boundaries",
                "quantum_same_boundaries",
                "segments_exact_bits",
                "stage_path_bits"
            ],
            "{which:?}: {suite}"
        );
    }
}
