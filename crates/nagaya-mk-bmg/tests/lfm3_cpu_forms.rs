//! The CPU forms of the LFM3 ops against the fixture of plain torch (`tools/lfm3_op_fixture.py`). The card test
//! `xe_lfm3_ops.rs` compares the kernels with these forms, so a wrong form would hide a wrong kernel. They need no card.

mod common;

use common::lfm3::*;

#[test]
fn the_short_convolution_matches_the_reference_with_its_state() {
    let fx = Fixture::load();
    for (t, seed) in CONV_CASES {
        let (w, bcx, state) = conv_inputs(t, seed);
        let (y, next) = short_conv(&bcx, &w, &state, t, CONV_D, CONV_TAPS);
        let (ey, es) = (
            rel(&y, fx.get(&format!("short_conv.{t}.y"))),
            rel(&next, fx.get(&format!("short_conv.{t}.state"))),
        );
        assert!(ey < 1e-6, "{t} positions: the output has a relative error of {ey}");
        assert_eq!(es, 0.0, "{t} positions: the state is the last inputs, to the bit");
    }
}

#[test]
fn the_partial_rotation_matches_the_zero_tail_layout_of_the_reference() {
    let fx = Fixture::load();
    for (name, theta, pairs, seed) in ROPE_CASES {
        let (x, w) = rope_inputs(seed);
        let got = norm_rope(&x, &w, ROPE_HEADS, ROPE_D, ROPE_P0, theta, pairs, ROPE_EPS, 1.0);
        let err = rel(&got, fx.get(&format!("rope.{name}")));
        assert!(err < 1e-3, "the {name} layers: relative error {err}");
    }
}

#[test]
fn the_attention_matches_the_sliding_mask_of_transformers() {
    let fx = Fixture::load();
    let (q, k, v) = attn_inputs();
    for (name, window) in ATTN_CASES {
        let mut got = Vec::new();
        for t in fx.rows(name) {
            for h in 0..ATTN_H {
                got.extend(attend_row(&q, &k, &v, t, h, ATTN_H, ATTN_KV, ATTN_D, window));
            }
        }
        let err = rel(&got, fx.get(&format!("attention.{name}")));
        assert!(err < 1e-5, "the {name} layers: relative error {err}");
    }
}

#[test]
fn the_head_gate_matches_the_sigmoid_of_the_reference() {
    let fx = Fixture::load();
    let (a, g) = gate_inputs();
    let got = head_gate(&a, &g, GATE_H, GATE_D, GATE_STRIDE, GATE_OFFSET);
    let err = rel(&got, fx.get("head_gate"));
    assert!(err < 1e-6, "relative error {err}");
}

/// The CPU form of `OP_CONTROL` keeps the definitions of engine-abi "Control vectors": an ablate of 1 leaves no part
/// along the vector, a clamp sets that part to its strength, an ablate of 0.5 halves it, and an add moves it by the
/// strength times |v|. The rows of the case have a part along the vector to start with.
///
/// Falsified by an ablate that subtracts `s * (h . u) * v` (the vector, not the unit vector).
#[test]
fn the_control_form_keeps_the_definition_of_each_operation() {
    use nagaya_mk_bmg::protocol::ControlOp;
    let (h, v) = control_inputs();
    let before = along(&h, &v, CONTROL_K);
    let norm = v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    assert!(
        before.iter().all(|a| a.abs() > 1.0),
        "each row starts with a part along v: {before:?}"
    );
    for (op, s) in CONTROL_CASES {
        let after = along(&control(&h, &v, CONTROL_K, op, s), &v, CONTROL_K);
        for (b, a) in before.iter().zip(&after) {
            let want = match op {
                ControlOp::Add => b + s as f64 * norm,
                ControlOp::Ablate => b * (1.0 - s as f64),
                ControlOp::Clamp => s as f64,
            };
            assert!(
                (a - want).abs() < 1e-3,
                "{op:?} {s}: the part along v is {a}, not {want}"
            );
        }
    }
    let zeros = vec![0.0f32; CONTROL_K];
    assert_eq!(
        control(&h, &zeros, CONTROL_K, ControlOp::Clamp, 8.0),
        h,
        "a vector of zeros keeps each row"
    );
}
