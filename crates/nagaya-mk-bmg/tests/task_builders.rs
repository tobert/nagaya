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
