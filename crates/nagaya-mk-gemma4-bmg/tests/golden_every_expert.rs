//! Every routed expert of every layer against the HF reference.
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
use nagaya_mk_gemma4_bmg::protocol::{MoERouteDecision, TOP_K};
use nagaya_mk_gemma4_bmg::sharding::Checkpoint;

/// Every one of the 128 routed experts in every layer, on the GPU's grouped MoE ops,
/// against HF's Gemma4TextExperts: token t picks experts 8t..8t+7 with weights 1/8 + s/100,
/// on the layer's real pre_feedforward_layernorm_2 output.
#[test]
fn golden_every_expert() {
    if common::absent(&["chat.safetensors"]) {
        return;
    }
    let r = reference("chat.safetensors");
    let cl = common::cluster();
    let ckpt = Checkpoint::open_dir(&common::model_dir()).unwrap();
    let card0 = CardShard::new(&cl.cards[0], &ckpt, CARD_LAYERS[0].clone(), false, 64, 16).unwrap();
    let card1 = CardShard::new(&cl.cards[1], &ckpt, CARD_LAYERS[1].clone(), false, 64, 16).unwrap();
    let mut failures = Vec::new();
    let mut worst = 0.0f64;
    for l in 0..NUM_LAYERS {
        let (shard, card) = if CARD_LAYERS[0].contains(&l) {
            (&card0, &cl.cards[0])
        } else {
            (&card1, &cl.cards[1])
        };
        let x = r.f32(&format!("L{}.sweep_in", l)).unwrap().0;
        let idx = r.f32(&format!("L{}.sweep_index", l)).unwrap().0;
        let w = r.f32(&format!("L{}.sweep_weights", l)).unwrap().0;
        let want = r.f32(&format!("L{}.sweep_out", l)).unwrap().0;
        let n = 16;
        let labels = unsafe { shard.build_ring(n, 0, &[], None) };
        shard.write(card, Buf::Pre2, 0, &x);
        let routes = shard.buf_ptr(Buf::Routes(l), 0) as *mut MoERouteDecision;
        for t in 0..n {
            let mut d = MoERouteDecision::default();
            for k in 0..TOP_K {
                d.top_k_indices[k] = idx[t * TOP_K + k] as u32;
                d.top_k_weights[k] = w[t * TOP_K + k];
            }
            unsafe { *routes.add(t) = d };
        }
        let start = labels.iter().position(|s| s == &format!("L{}.moe_plan", l)).unwrap();
        let end = labels.iter().position(|s| s == &format!("L{}.moe_down", l)).unwrap() + 1;
        unsafe { shard.run_range(card, start, end) };
        let got = sum_slots(&shard.read(card, Buf::MoeOut, 0, n * TOP_K * HIDDEN));
        let err = rel_err(&got, &want);
        let worst_token = (0..n)
            .map(|t| {
                (
                    rel_err(&got[t * HIDDEN..(t + 1) * HIDDEN], &want[t * HIDDEN..(t + 1) * HIDDEN]),
                    t,
                )
            })
            .fold((0.0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
        println!(
            "L{} all 128 experts: rel_err {:.3e}, worst token {} (experts {}..{}) {:.3e}",
            l,
            err,
            worst_token.1,
            worst_token.1 * 8,
            worst_token.1 * 8 + 7,
            worst_token.0
        );
        worst = worst.max(worst_token.0);
        if !(worst_token.0 < OP_TOL) {
            failures.push(format!(
                "L{} experts {}..{} rel_err {:.3e}",
                l,
                worst_token.1 * 8,
                worst_token.1 * 8 + 7,
                worst_token.0
            ));
        }
    }
    println!(
        "all 128 experts in all {} layers: worst token rel_err {:.3e}",
        NUM_LAYERS, worst
    );
    assert!(failures.is_empty(), "expert mismatches:\n{}", failures.join("\n"));
}
