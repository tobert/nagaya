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
