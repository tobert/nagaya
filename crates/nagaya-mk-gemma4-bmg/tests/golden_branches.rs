//! Two branches in one block against separate passes.
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

/// Two continuations of one prefix in one block: a decoy, then the prompt's real last 3
/// tokens. The second must match the HF reference's last-token logits, so it must ignore
/// the decoy's rows and keep its own positions; the decoy must match a plain forward pass.
#[test]
fn golden_branches_match_separate_passes() {
    if common::absent(&["chat.safetensors"]) {
        return;
    }
    let r = reference("chat.safetensors");
    let ids = ref_ids(&r);
    let want = r.f32("logits_last").unwrap().0;
    let cl = common::cluster();
    let engine = Gemma4Engine::load(&cl, &common::model_dir(), 256, 32).unwrap();
    let split = ids.len() - 3;
    let decoy = vec![818u32, 5279, 529, 7001];
    let real = ids[split..].to_vec();
    let (out, timing) = engine
        .forward_branches(&cl, &ids[..split], &[decoy.clone(), real], 0)
        .unwrap();
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let d_real = diff(&out[1], &want);
    let mut plain = ids[..split].to_vec();
    plain.extend(&decoy);
    let (decoy_alone, _) = engine.forward(&cl, &plain, 0).unwrap();
    let d_decoy = diff(&out[0], &decoy_alone);
    println!(
        "branches in {} block(s), card0 {} us, card1 {} us: real vs HF max |diff| {:.3e} (top1 {} vs {}); decoy vs plain pass {:.3e}",
        timing.blocks,
        timing.card_us[0],
        timing.card_us[1],
        d_real,
        top1(&out[1]),
        top1(&want),
        d_decoy
    );
    assert_eq!(top1(&out[1]), top1(&want));
    assert!(d_real < LOGIT_TOL, "real branch vs HF {:.3e}", d_real);
    assert!(d_decoy < LOGIT_TOL, "decoy branch vs plain pass {:.3e}", d_decoy);
}
