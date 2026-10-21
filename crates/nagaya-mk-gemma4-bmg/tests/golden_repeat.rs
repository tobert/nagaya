//! Five runs of the same tokens give the same bits.
//!
//! mkviolet's `src/golden.rs`, copied with no change in behavior (milestone 15s, step 4).
#![cfg(feature = "xe")]
// `!(err < tol)` fails on NaN too, as mkviolet meant. The other lints wait for the port, as in lib.rs.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop
)]

mod common;

use common::*;
use nagaya_mk_gemma4_bmg::gemma4::*;

/// Every op sums in a fixed order, so the same tokens must give bit-identical logits on every
/// run. A stale read across work-groups (a missing fence) shows up here as a difference:
/// five runs of a 600-token prefill in blocks of 256 and 24 one-token steps.
#[test]
fn golden_repeat_identical() {
    if common::absent(&[]) {
        return;
    }
    let cl = common::cluster();
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[1024], 256).unwrap();
    let prompt: Vec<u32> = (0..600u32).map(|i| (1000 + i * 7919) % 200000).collect();
    let run = || {
        let (mut logits, _) = engine.forward(&cl, &prompt, 0).unwrap();
        let mut all: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
        for i in 0..24 {
            let next = top1(&logits) as u32;
            logits = engine.forward(&cl, &[next], prompt.len() + i).unwrap().0;
            all.extend(logits.iter().map(|v| v.to_bits()));
        }
        all
    };
    let first = run();
    for r in 1..5 {
        let again = run();
        let diff = first.iter().zip(&again).filter(|(a, b)| a != b).count();
        println!("run {}: {} of {} logits differ from run 0", r, diff, first.len());
        assert_eq!(diff, 0, "run {} differs from run 0", r);
    }
}
