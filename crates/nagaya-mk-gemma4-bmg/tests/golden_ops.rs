//! Each op of layers 0 and 5 against the HF reference.
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

/// Each op of layers 0 (sliding) and 5 (global) on the reference's own layer input, run
/// on card 0 up to that op and compared with the reference's tensor for it.
#[test]
fn golden_ops_layers_0_and_5() {
    if common::absent(&["chat.safetensors"]) {
        return;
    }
    let r = reference("chat.safetensors");
    let ids = ref_ids(&r);
    let s = ids.len();
    let cl = common::cluster();
    let card = &cl.cards[0];
    let ckpt = Checkpoint::open_dir(&common::model_dir()).unwrap();
    let shard = CardShard::new(card, &ckpt, 0..6, false, 64, 32).unwrap();
    let labels = unsafe { shard.build_ring(s, 0, &[], None) };
    let d = HIDDEN;
    let mut worst = 0.0f64;
    let mut failures = Vec::new();

    for l in [0usize, 5] {
        let g = layer_geom(l);
        let (hd, kv) = (g.head_dim, g.kv_heads);
        let get = |name: &str| r.f32(&format!("L{}.{}", l, name)).unwrap().0;
        let input = get("in");
        let start = labels.iter().position(|x| x == &format!("L{}.attn_norm", l)).unwrap();
        let add = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>();
        let mut checks: Vec<(&str, Buf, usize, Vec<f32>)> = vec![
            ("attn_norm", Buf::Norm, s * d, get("attn_norm")),
            ("q_proj", Buf::Q, s * HEADS * hd, get("q_proj")),
            ("k_proj", Buf::K(l), s * kv * hd, get("k_proj")),
            ("v_norm", Buf::V(l), s * kv * hd, get("v_norm")),
            ("q_norm_rope", Buf::Q, s * HEADS * hd, get("q_rope")),
            ("k_norm_rope", Buf::K(l), s * kv * hd, get("k_rope")),
            ("attn_pv_sum", Buf::Attn, s * HEADS * hd, get("attn")),
            ("o_proj", Buf::Tmp, s * d, get("o_proj")),
            ("attn_out", Buf::X, s * d, add(&input, &get("post_attn_norm"))),
            ("attn_out", Buf::Norm, s * d, get("pre_ffn_norm")),
            ("attn_out", Buf::Pre2, s * d, get("pre_ffn_norm_2")),
            ("mlp", Buf::Tmp, s * d, get("mlp")),
            ("moe_down", Buf::MoeOut, s * TOP_K * d, get("experts")),
            ("out", Buf::X, s * d, get("out")),
        ];
        if !g.global {
            checks.insert(3, ("v_proj", Buf::V(l), s * kv * hd, get("v_proj")));
        }
        for (op, buf, count, want) in checks.iter_mut() {
            let end = labels.iter().position(|x| x == &format!("L{}.{}", l, op)).unwrap() + 1;
            shard.write(card, Buf::X, 0, &input);
            unsafe { shard.run_range(card, start, end) };
            let mut got = shard.read(card, *buf, 0, *count);
            if *op == "moe_down" {
                got = sum_slots(&got);
            }
            let err = rel_err(&got, want);
            worst = worst.max(err);
            println!(
                "L{} {:<16} {:<8} rel_err {:.3e}",
                l,
                op,
                format!("{:?}", buf).split('(').next().unwrap(),
                err
            );
            if !(err < OP_TOL) {
                failures.push(format!("L{} {} rel_err {:.3e}", l, op, err));
            }
        }

        // Routing must pick the same experts with the same weights.
        let end = labels.iter().position(|x| x == &format!("L{}.router", l)).unwrap() + 1;
        shard.write(card, Buf::X, 0, &input);
        unsafe { shard.run_range(card, start, end) };
        let routes: Vec<MoERouteDecision> = unsafe {
            std::slice::from_raw_parts(shard.buf_ptr(Buf::Routes(l), 0) as *const MoERouteDecision, s).to_vec()
        };
        let want_idx = get("router_index");
        let want_w = get("router_weights");
        let mut idx_mismatch = 0;
        let mut w_err = 0.0f32;
        for t in 0..s {
            for k in 0..TOP_K {
                if routes[t].top_k_indices[k] as f32 != want_idx[t * TOP_K + k] {
                    idx_mismatch += 1;
                }
                w_err = w_err.max((routes[t].top_k_weights[k] - want_w[t * TOP_K + k]).abs());
            }
        }
        println!(
            "L{} router           index mismatches {} / {}, max weight diff {:.3e}",
            l,
            idx_mismatch,
            s * TOP_K,
            w_err
        );
        if idx_mismatch > 0 || w_err > 1e-4 {
            failures.push(format!(
                "L{} router: {} index mismatches, weight diff {:.3e}",
                l, idx_mismatch, w_err
            ));
        }
    }
    println!("worst op rel_err {:.3e}", worst);
    assert!(failures.is_empty(), "golden op mismatches:\n{}", failures.join("\n"));
}
