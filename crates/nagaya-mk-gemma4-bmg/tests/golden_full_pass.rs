//! All 30 layers on both cards against the HF reference logits.
//!
//! A golden check of the first Gemma 4 engine, copied with no change in behavior.
#![cfg(feature = "xe")]
// `!(err < tol)` fails on NaN too, as the first engine meant. The other lints wait for the port, as in lib.rs.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop
)]

mod common;

use common::*;
use nagaya_mk_gemma4_bmg::gemma4::*;

/// All 30 layers on both cards: the residual leaving card 0 and the last-token logits
/// match the reference, whole-prompt and in smaller blocks.
#[test]
fn golden_full_pass_logits() {
    if common::absent(&["chat.safetensors"]) {
        return;
    }
    let r = reference("chat.safetensors");
    let ids = ref_ids(&r);
    let want = r.f32("logits_last").unwrap().0;
    let want_l14 = r.f32("L14.out").unwrap().0;
    let cl = common::cluster();
    let engine = Gemma4Engine::load(&cl, &common::model_dir(), 256, 32).unwrap();
    println!("loaded in {} ms", engine.load_ms);
    let mut failures = Vec::new();
    for block in [32usize, 5, 1] {
        let (logits, timing) = engine.forward_blocks(&cl, &ids, 0, block).unwrap();
        let max_diff = logits
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (g1, w1) = (top1(&logits), top1(&want));
        println!(
            "block {:>2}: top1 gpu {} ({:.4}) ref {} ({:.4}), max |logit diff| {:.4e}, {} blocks, card0 {} us, card1 {} us",
            block, g1, logits[g1], w1, want[w1], max_diff, timing.blocks, timing.card_us[0], timing.card_us[1]
        );
        if g1 != w1 || !(max_diff < LOGIT_TOL) {
            failures.push(format!(
                "block {}: top1 {} vs {}, max diff {:.4e}",
                block, g1, w1, max_diff
            ));
        }
    }
    // Card 0's residual after the last whole-prompt block is layer 14's output.
    let (_, _) = engine.forward_blocks(&cl, &ids, 0, 32).unwrap();
    let got_l14 = engine.shards[0].read(&cl.cards[0], Buf::X, 0, ids.len() * HIDDEN);
    let e14 = rel_err(&got_l14, &want_l14);
    println!("layer 14 output rel_err {:.3e}", e14);
    if !(e14 < 1e-3) {
        failures.push(format!("layer 14 output rel_err {:.3e}", e14));
    }
    assert!(failures.is_empty(), "full-pass mismatches:\n{}", failures.join("\n"));
}
