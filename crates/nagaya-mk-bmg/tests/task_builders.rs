//! The task builders refuse the inputs that their kernels cannot take. They need no card.

use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::tasks;

/// A call of `tasks::biattn` with head size `d` and score scale `scale`.
fn biattn(d: u32, scale: f32) -> MkTaskDescriptor {
    tasks::biattn(0, 0, 0, 0, 4, 2, 1, d, 0, scale)
}

/// `OP_BIATTN` keeps 32 values of each head in each of 16 lanes, so a head of 16 to 512 values in 16s fits.
#[test]
fn biattn_takes_heads_of_16_to_512_values() {
    for d in [16, 256, 512] {
        assert_eq!(biattn(d, 1.0).p0, d);
    }
}

/// A head above 512 values writes past the arrays of the kernel (`kernels/embeddinggemma2_ops.cl`).
#[test]
#[should_panic(expected = "OP_BIATTN takes a head of 16 to 512 values in 16s")]
fn biattn_refuses_a_head_above_512() {
    biattn(528, 1.0);
}

/// A head that is not a multiple of 16 leaves its last values unwritten.
#[test]
#[should_panic(expected = "OP_BIATTN takes a head of 16 to 512 values in 16s")]
fn biattn_refuses_a_head_that_is_not_a_multiple_of_16() {
    biattn(100, 1.0);
}

/// The kernel reads the bits 0 as a scale of 1.0, so a scale of 0.0 would become 1.0.
#[test]
#[should_panic(expected = "OP_BIATTN reads a scale of 0.0 as 1.0")]
fn biattn_refuses_a_scale_of_zero() {
    biattn(256, 0.0);
}

/// `pad_end` sets `FLAG_PAD_END` on a convolution and keeps the rest of the task.
#[test]
fn pad_end_marks_a_convolution_and_keeps_its_fields() {
    use nagaya_mk_bmg::protocol::{FLAG_ACCUMULATE, FLAG_PAD_END};
    let conv = tasks::conv2d(1, 2, 3, 4, 1, 8, 6, 64, 32, 3, 2, 1, true, true);
    let marked = tasks::pad_end(conv);
    assert_eq!(marked.flags, conv.flags | FLAG_PAD_END);
    assert_ne!(marked.flags & FLAG_ACCUMULATE, 0);
    assert_eq!(
        (marked.layer, marked.n, marked.a, marked.c),
        (conv.layer, conv.n, conv.a, conv.c)
    );
    let gemm = tasks::gemm(1, 2, 3, 4, 1, 8, 6, 64, 32, (3, 2, 1), 3, false, false);
    assert_eq!(tasks::pad_end(gemm).flags, gemm.flags | FLAG_PAD_END);
}

/// The other ops do not read `FLAG_PAD_END`, so a mark on one of them is a defect of the graph.
#[test]
#[should_panic(expected = "FLAG_PAD_END marks OP_CONV2D or OP_GEMM")]
fn pad_end_refuses_an_op_that_does_not_pad() {
    tasks::pad_end(tasks::upsample2x(0, 0, 1, 8, 8, 64));
}

/// A call of `tasks::short_conv` over 4 positions at separate addresses.
fn short_conv(d: u32, taps: u32) -> MkTaskDescriptor {
    tasks::short_conv(0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 4, d, taps, false)
}

/// The message of the panic of `f`.
fn panic_message(f: impl FnOnce() -> MkTaskDescriptor + std::panic::UnwindSafe) -> String {
    let caught = std::panic::catch_unwind(f).expect_err("the builder took its inputs");
    match caught.downcast::<String>() {
        Ok(message) => *message,
        Err(caught) => caught.downcast::<&str>().map(|m| m.to_string()).unwrap_or_default(),
    }
}

/// `OP_SHORT_CONV` puts the taps in `layer`, the channels in `k`, and the state in `x`.
#[test]
fn short_conv_puts_its_fields_where_the_kernel_reads_them() {
    use nagaya_mk_bmg::protocol::{FLAG_F32, OP_SHORT_CONV};
    let task = short_conv(2048, 7);
    assert_eq!(task.opcode, OP_SHORT_CONV);
    assert_eq!(
        (task.a, task.b, task.x, task.c),
        (0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000)
    );
    assert_eq!((task.n, task.k, task.layer, task.flags), (4, 2048, 7, 0));
    let f32_weights = tasks::short_conv(0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 1, 16, 2, true);
    assert_eq!(f32_weights.flags, FLAG_F32);
}

/// A lane takes one channel, and a group of 16 lanes takes 16 channels, so the channels come in 16s.
#[test]
#[should_panic(expected = "OP_SHORT_CONV takes channels in 16s")]
fn short_conv_refuses_channels_that_are_not_in_16s() {
    short_conv(40, 7);
}

/// The kernel keeps at most 8 taps in its window, and one tap has no state.
#[test]
fn short_conv_refuses_1_tap_and_9_taps() {
    for taps in [1, 9] {
        let message = panic_message(move || short_conv(64, taps));
        assert!(message.contains("OP_SHORT_CONV takes 2 to 8 taps"), "{message}");
    }
}

/// The op reads the in_proj rows and the state while it writes the output and the state, so they must not overlap.
#[test]
#[should_panic(expected = "OP_SHORT_CONV: the output overlaps the in_proj rows")]
fn short_conv_refuses_an_output_over_its_input() {
    // 4 positions of 3 * 64 floats from 0x10_0000; the output starts inside them.
    tasks::short_conv(0x10_0000, 0x20_0000, 0x30_0000, 0x10_0100, 4, 64, 7, false);
}

#[test]
#[should_panic(expected = "OP_SHORT_CONV: the output overlaps the state")]
fn short_conv_refuses_an_output_over_its_state() {
    tasks::short_conv(0x10_0000, 0x20_0000, 0x30_0000, 0x30_0010, 4, 64, 7, false);
}

#[test]
#[should_panic(expected = "OP_SHORT_CONV: the state overlaps the in_proj rows")]
fn short_conv_refuses_a_state_in_its_input() {
    tasks::short_conv(0x10_0000, 0x20_0000, 0x10_0040, 0x40_0000, 4, 64, 7, false);
}

#[test]
#[should_panic(expected = "OP_SHORT_CONV takes at least one position")]
fn short_conv_refuses_no_positions() {
    tasks::short_conv(0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 0, 64, 7, false);
}

/// `partial_rotation` puts the rotated pairs in `layer` of a rotation and keeps the rest of the task.
#[test]
fn partial_rotation_marks_a_rotation_and_keeps_its_fields() {
    let rope = tasks::qk_rope(1, 2, 32, 16, 128, 9, true, true, 1e-5, 0.5);
    let partial = tasks::partial_rotation(rope, 32);
    assert_eq!(partial.layer, 32);
    assert_eq!(
        (partial.flags, partial.k, partial.x, partial.p0),
        (rope.flags, rope.k, rope.x, rope.p0)
    );
}

/// A norm with no rotation does not read the pairs, so a mark on it is a defect of the graph.
#[test]
#[should_panic(expected = "partial_rotation marks an OP_RMSNORM with FLAG_ROPE")]
fn partial_rotation_refuses_a_norm_with_no_rotation() {
    tasks::partial_rotation(tasks::rmsnorm(1, 2, 3, 4, 2048, 1e-5, false), 32);
}

/// The kernel reads 0 as "as the flags say", and a head has only d / 2 pairs.
#[test]
fn partial_rotation_refuses_0_pairs_and_more_than_half_the_head() {
    for pairs in [0, 65] {
        let rope = tasks::qk_rope(1, 2, 32, 16, 128, 9, false, true, 1e-5, 1.0);
        let message = panic_message(move || tasks::partial_rotation(rope, pairs));
        assert!(message.contains("a head of 128 values has 1 to 64 pairs"), "{message}");
    }
}

/// `OP_HEAD_GATE` reads the gates of a position from a row of `stride` values.
#[test]
fn head_gate_puts_its_fields_where_the_kernel_reads_them() {
    let task = tasks::head_gate(1, 2, 3, 5, 16, 128, 2064);
    assert_eq!(task.opcode, nagaya_mk_bmg::protocol::OP_HEAD_GATE);
    assert_eq!(
        (task.a, task.b, task.c, task.n, task.m, task.k, task.p0),
        (1, 2, 3, 5, 16, 128, 2064)
    );
}

/// A row of gates holds one gate for each head.
#[test]
#[should_panic(expected = "OP_HEAD_GATE reads rows of at least 16 gates")]
fn head_gate_refuses_rows_shorter_than_the_heads() {
    tasks::head_gate(1, 2, 3, 5, 16, 128, 8);
}
