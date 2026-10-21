//! A prompt past the sliding window against the HF reference.
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
use nagaya_mk_gemma4_bmg::protocol::TOP_K;

/// A 1270-token prompt, past the 1024-token sliding window, in blocks of 64: every
/// layer's output at the last 64 positions and the last-token logits match the reference.
#[test]
fn golden_long_context_sliding_window() {
    if common::absent(&["long.safetensors"]) {
        return;
    }
    let r = reference("long.safetensors");
    let ids = ref_ids(&r);
    assert!(
        ids.len() > SLIDING_WINDOW as usize + 64,
        "the prompt must pass the sliding window"
    );
    let want = r.f32("logits_last").unwrap().0;
    let cl = common::cluster();
    let engine = Gemma4Engine::load(&cl, &common::model_dir(), 2048, 64).unwrap();
    *engine.route_log.lock().unwrap() = Some(Vec::new());
    let (logits, timing) = engine.forward(&cl, &ids, 0).unwrap();
    let max_diff = logits
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (g1, w1) = (top1(&logits), top1(&want));
    println!(
        "{} tokens in {} blocks: wall {} us (card0 {} us, card1 {} us); top1 gpu {} ref {}, max |logit diff| {:.4e}",
        ids.len(),
        timing.blocks,
        timing.wall_us,
        timing.card_us[0],
        timing.card_us[1],
        g1,
        w1,
        max_diff
    );
    // The last block's residual on card 0 is layer 14's output for the last n tokens.
    let n_last = (ids.len() - 1) % 64 + 1;
    let got = engine.shards[0].read(&cl.cards[0], Buf::X, 0, n_last * HIDDEN);
    let full = r.f32("L14.out").unwrap().0;
    let e14 = rel_err(&got, &full[(ids.len() - n_last) * HIDDEN..]);
    println!("layer 14 output, last {} positions: rel_err {:.3e}", n_last, e14);
    let base = ids.len() - n_last;
    let mut per: Vec<(f64, usize)> = (0..n_last)
        .map(|t| {
            (
                rel_err(
                    &got[t * HIDDEN..(t + 1) * HIDDEN],
                    &full[(base + t) * HIDDEN..(base + t + 1) * HIDDEN],
                ),
                base + t,
            )
        })
        .collect();
    per.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("worst positions: {:?}", &per[..5.min(per.len())]);
    // Routing at every position and layer: a pick that differs from HF's must sit on a
    // near-tie between HF's 8th and 9th router probabilities.
    let log = engine.route_log.lock().unwrap().take().unwrap();
    let mut mismatches = Vec::new();
    for (l, pos, picks) in log {
        let want_idx = r.f32(&format!("L{}.route_index", l)).unwrap().0;
        let p9 = r.f32(&format!("L{}.route_p9", l)).unwrap().0;
        let mut g: Vec<u32> = picks.to_vec();
        let mut w: Vec<u32> = want_idx[pos * TOP_K..(pos + 1) * TOP_K]
            .iter()
            .map(|&v| v as u32)
            .collect();
        g.sort();
        w.sort();
        if g != w {
            mismatches.push((l, pos, p9[pos * 9 + 7] - p9[pos * 9 + 8]));
        }
    }
    // Each position's lowest mismatching layer is where it first diverged; later layers of
    // that token drift from there. The earliest divergence in the prompt can only come from
    // summation order, so it must be an exact near-tie.
    mismatches.sort_by_key(|m| (m.1, m.0));
    let mut origins: Vec<(usize, usize, f32)> = Vec::new();
    for m in &mismatches {
        if origins.last().map(|o| o.1) != Some(m.1) {
            origins.push(*m);
        }
    }
    let worst_origin = origins.iter().map(|o| o.2).fold(0.0f32, f32::max);
    println!(
        "routing: {} mismatched (layer, position) picks over {} tokens x 30 layers; {} positions diverge; first {:?}; worst first-divergence gap {:.2e}",
        mismatches.len(),
        ids.len(),
        origins.len(),
        origins.first(),
        worst_origin
    );
    if let Some(first) = origins.first() {
        assert!(
            first.2 < ROUTE_TIE,
            "the prompt's first routing difference is not a near-tie: {:?}",
            first
        );
    }
    assert!(
        worst_origin < ROUTE_ORIGIN_TOL,
        "a position first diverges on a routing gap of {:.2e}",
        worst_origin
    );
    let usage = engine.expert_usage();
    println!(
        "distinct experts picked per layer over {} tokens: {:?}",
        ids.len(),
        usage.iter().map(|u| u.0).collect::<Vec<_>>()
    );
    println!(
        "fewest picks of any expert per layer: {:?}",
        usage.iter().map(|u| u.1).collect::<Vec<_>>()
    );
    // The last token again on its own: one-token attention, split into key chunks, over a
    // sliding window the prompt has passed and a global span of every earlier position.
    let last = ids.len() - 1;
    let (one, _) = engine.forward(&cl, &ids[last..], last).unwrap();
    let one_diff = one.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let one_vs_block = one
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "last token alone: top1 gpu {} ref {}, max |logit diff| {:.4e} from HF, {:.4e} from the block pass",
        top1(&one),
        w1,
        one_diff,
        one_vs_block
    );
    assert_eq!(top1(&one), w1, "one-token top-1 differs");
    assert!(one_diff < LONG_LOGIT_TOL, "one-token max logit diff {:.4e}", one_diff);
    assert_eq!(g1, w1, "top-1 token differs");
    assert!(max_diff < LONG_LOGIT_TOL, "max logit diff {:.4e}", max_diff);
    assert!(e14 < LONG_L14_TOL, "layer 14 rel_err {:.3e}", e14);
}
