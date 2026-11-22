//! The builders of the layer-part ops of Inkling 2 (`nagaya_mk_bmg::tasks`), with no card: each builder puts its
//! arguments in the fields that `kernels/inkling2_layer_ops.cl` reads, and it refuses a task that the op cannot run.

use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::tasks::{self, ConvTaps, HeadClip};
use nagaya_mk_inkling2_bmg::host::ops::{HEAD_DIM, HEADS, HIDDEN, KV_HEADS, QK_CLIP};

const H: u32 = HIDDEN as u32;
/// Buffers far apart, so that no two of them overlap.
const A: u64 = 1 << 40;
const B: u64 = 2 << 40;
const C: u64 = 3 << 40;
const X: u64 = 4 << 40;

fn panics(f: impl FnOnce() -> MkTaskDescriptor + std::panic::UnwindSafe) -> String {
    let err = std::panic::catch_unwind(f).expect_err("the builder accepted the task");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn the_attention_residual_puts_the_counts_the_plane_and_the_scaling_in_its_fields() {
    let t = tasks::attn_residual(A, 3, B, X, 4, C, 37, H, 512);
    assert_eq!(t.opcode, OP_ATTN_RESIDUAL);
    assert_eq!((t.a, t.b, t.c, t.x), (A, B, C, X));
    assert_eq!((t.m, t.n, t.k, t.p0), (3 | 4 << 16, 37, H, 512));
    // The scaling of the host form, 2304^-0.5, as bits.
    assert_eq!(f32::from_bits(t.layer), (HIDDEN as f32).powf(-0.5));
}

#[test]
fn the_attention_residual_refuses_counts_and_widths_that_the_op_cannot_hold() {
    assert!(panics(|| tasks::attn_residual(A, 5, B, X, 1, C, 1, H, 1)).contains("at most 4 block outputs"));
    assert!(panics(|| tasks::attn_residual(A, 1, B, X, 0, C, 1, H, 1)).contains("1 to 4 queries"));
    assert!(panics(|| tasks::attn_residual(A, 1, B, X, 5, C, 1, H, 1)).contains("1 to 4 queries"));
    assert!(panics(|| tasks::attn_residual(A, 1, B, X, 1, C, 1, 2300, 1)).contains("steps of 16"));
    assert!(panics(|| tasks::attn_residual(A, 1, B, X, 1, C, 8, H, 7)).contains("8 positions"));
    assert!(panics(|| tasks::attn_residual(A, 1, B, X, 1, C, 0, H, 1)).contains("one position"));
}

#[test]
fn the_attention_residual_refuses_outputs_over_its_inputs() {
    // The second output plane starts in the partial rows.
    let plane = 4 * 8 * H as u64;
    let msg = panics(|| tasks::attn_residual(A, 2, B - plane, X, 2, B - plane, 8, H, 8));
    assert!(msg.contains("overlap"), "{msg}");
    // An output plane in the block outputs.
    let msg = panics(|| tasks::attn_residual(A, 2, B, X, 1, A + plane, 8, H, 8));
    assert!(msg.contains("overlap"), "{msg}");
    // The next plane after the last block output is free.
    tasks::attn_residual(A, 2, B, X, 1, A + 2 * plane, 8, H, 8);
    // The op reads its rows in blocks of 16-byte aligned values.
    for (a, b, x, c) in [(A + 8, B, X, C), (A, B + 8, X, C), (A, B, X + 8, C), (A, B, X, C + 8)] {
        let msg = panics(move || tasks::attn_residual(a, 1, b, x, 1, c, 8, H, 8));
        assert!(msg.contains("multiple of 16 bytes"), "{msg}");
    }
}

#[test]
fn the_message_gate_runs_in_place_and_refuses_an_output_over_the_message_or_the_weights() {
    // An address above 4 GiB, so that the gate must carry both halves of g.
    const G: u64 = (5 << 40) | 0x1234_5670;
    let [down, gate] = tasks::message_gate(A, B, X, A, G, 5, H);
    assert_eq!(down.opcode, OP_MESSAGE_GATE_DOWN);
    assert_eq!((down.a, down.b, down.c, down.x, down.n, down.k), (A, B, G, X, 5, H));
    assert_eq!(gate.opcode, OP_MESSAGE_GATE);
    assert_eq!((gate.a, gate.b, gate.c, gate.x, gate.n, gate.k), (A, B, A, X, 5, H));
    assert_eq!(
        gate.p0 as u64 | (gate.layer as u64) << 32,
        G,
        "the gate reads g at p0 | layer << 32"
    );
    assert!(panics(|| tasks::message_gate(A, B, X, B + 4, G, 5, H)[0]).contains("overlap"));
    assert!(panics(|| tasks::message_gate(A, B, X, X + 64, G, 5, H)[0]).contains("overlap"));
    assert!(panics(|| tasks::message_gate(A, B, X, A + 4, G, 5, H)[0]).contains("not h"));
    assert!(panics(|| tasks::message_gate(A, B, X, C, G, 5, 100)[0]).contains("steps of 16"));
    // The rows g must not overlap h, the messages, the output or the weights: 5 rows of 64 floats from G.
    for g in [A + 4, B + 4, C + 4, X + 64] {
        let msg = panics(move || tasks::message_gate(A, B, X, C, g, 5, H)[0]);
        assert!(msg.contains("rows g overlap"), "{msg}");
    }
    let [_, gate] = tasks::message_gate(A, B, X, C, C - 5 * 64 * 4, 5, H);
    assert_eq!(gate.c, C, "the rows g may end where the output starts");
    // The op reads its rows in blocks of 16-byte aligned values.
    for (h, m, w, y, g) in [
        (A + 8, B, X, A + 8, G),
        (A, B + 8, X, A, G),
        (A, B, X + 8, A, G),
        (A, B, X, C + 8, G),
    ] {
        let msg = panics(move || tasks::message_gate(h, m, w, y, g, 5, H)[0]);
        assert!(msg.contains("multiple of 16 bytes"), "{msg}");
    }
    let msg = panics(|| tasks::message_gate(A, B, X, A, G + 8, 5, H)[0]);
    assert!(msg.contains("multiple of 16 bytes"), "{msg}");
}

#[test]
fn the_head_clip_encodes_its_mode_its_bound_and_its_log_scale() {
    let q = tasks::head_clip(
        A,
        A,
        5,
        HEADS as u32,
        HEAD_DIM as u32,
        HeadClip::L2 {
            scale: B,
            bound: QK_CLIP,
        },
        Some(9000),
    );
    assert_eq!(q.opcode, OP_HEAD_CLIP);
    assert_eq!((q.a, q.b, q.c, q.n, q.m, q.k, q.p0), (A, B, A, 5, 24, 128, 9000));
    assert_eq!((q.layer, f32::from_bits(q.x as u32)), (256, QK_CLIP));
    let k = tasks::head_clip(
        A,
        C,
        5,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        HeadClip::L2 {
            scale: 0,
            bound: QK_CLIP,
        },
        None,
    );
    assert_eq!((k.b, k.c, k.layer, k.p0), (0, C, 0, 0));
    let v = tasks::head_clip(A, A, 5, KV_HEADS as u32, HEAD_DIM as u32, HeadClip::Rms, None);
    assert_eq!((v.b, v.layer), (0, 1));
}

#[test]
fn the_head_clip_refuses_a_log_scale_on_a_norm_and_heads_it_cannot_sum() {
    assert!(panics(|| tasks::head_clip(A, A, 1, 8, 128, HeadClip::Rms, Some(0))).contains("log scale"));
    assert!(panics(|| tasks::head_clip(A, A, 1, 8, 8, HeadClip::Rms, None)).contains("steps of 16"));
    let bad = HeadClip::L2 { scale: 0, bound: 0.0 };
    assert!(panics(move || tasks::head_clip(A, A, 1, 8, 128, bad, None)).contains("positive"));
    assert!(panics(|| tasks::head_clip(A, A + 4, 2, 8, 128, HeadClip::Rms, None)).contains("overlap"));
}

#[test]
fn the_output_gate_runs_in_place_and_refuses_an_output_over_the_gates() {
    let t = tasks::output_gate(A, B, A, 5, 3072);
    assert_eq!(t.opcode, OP_OUTPUT_GATE);
    assert_eq!((t.a, t.b, t.c, t.n, t.k), (A, B, A, 5, 3072));
    assert!(panics(|| tasks::output_gate(A, B, B, 5, 3072)).contains("gates"));
    assert!(panics(|| tasks::output_gate(A, B, A + 4, 5, 3072)).contains("not the input"));
}

#[test]
fn the_convolution_packs_its_strides_and_its_mode() {
    let t = tasks::residual_conv(A, 1056, ConvTaps::Dynamic(B), X, 6656, C, 37, 1024, 100, false);
    assert_eq!(t.opcode, OP_RESIDUAL_CONV);
    assert_eq!((t.a, t.b, t.c, t.x), (A, B, C, X));
    assert_eq!(
        (t.m, t.n, t.k, t.p0, t.layer, t.flags),
        (1056 | 6656 << 16, 37, 1024, 100, 1, 0)
    );
    let f = tasks::residual_conv(A, H, ConvTaps::Fixed(B), X, H, C, 1, H, 0, true);
    assert_eq!((f.m, f.layer, f.flags), (H | H << 16, 0, FLAG_ACCUMULATE));
}

#[test]
fn the_convolution_refuses_rows_without_their_taps_and_buffers_that_overlap() {
    // A dynamic row holds the 1,024 channels and the 32 taps of the 8 heads.
    let msg = panics(|| tasks::residual_conv(A, 1055, ConvTaps::Dynamic(B), X, 6656, C, 1, 1024, 0, false));
    assert!(msg.contains("1056 floats"), "{msg}");
    assert!(
        panics(|| tasks::residual_conv(A, 1008, ConvTaps::Dynamic(B), X, 1008, C, 1, 1008, 0, false)).contains("128")
    );
    assert!(panics(|| tasks::residual_conv(A, 70000, ConvTaps::Fixed(B), X, H, C, 1, H, 0, false)).contains("65535"));
    assert!(
        panics(|| tasks::residual_conv(A, H, ConvTaps::Fixed(B), X, H - 16, C, 1, H, 0, false)).contains("state row")
    );
    assert!(
        panics(|| tasks::residual_conv(A, H, ConvTaps::Fixed(B), X, H, X + 4, 1, H, 0, false))
            .contains("the output overlaps the state")
    );
    assert!(
        panics(|| tasks::residual_conv(A, H, ConvTaps::Fixed(B), X, H, A, 1, H, 0, false))
            .contains("the output overlaps the inputs")
    );
    assert!(
        panics(|| tasks::residual_conv(A, H, ConvTaps::Fixed(B), A + 4, H, C, 1, H, 0, false))
            .contains("the state overlaps")
    );
}

#[test]
fn the_router_names_its_layer_and_refuses_decisions_over_its_logits() {
    let t = tasks::sigmoid_router(A, B, C, 37, 19);
    assert_eq!(t.opcode, OP_SIGMOID_ROUTER);
    assert_eq!((t.a, t.b, t.c, t.n, t.layer), (A, B, C, 37, 19));
    assert!(panics(|| tasks::sigmoid_router(A, B, A + 64, 37, 0)).contains("overlap"));
    assert!(panics(|| tasks::sigmoid_router(A, B, C, 0, 0)).contains("one position"));
}

#[test]
fn the_layer_ops_have_the_opcodes_150_to_155() {
    assert_eq!(
        [
            OP_ATTN_RESIDUAL,
            OP_MESSAGE_GATE,
            OP_HEAD_CLIP,
            OP_OUTPUT_GATE,
            OP_RESIDUAL_CONV,
            OP_SIGMOID_ROUTER
        ],
        [150, 151, 152, 153, 154, 155]
    );
}
