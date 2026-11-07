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

/// The split attention gives each slot size the same bits (spec 6.7): its count of key segments reads no slot size.
/// The scratch bound holds for each block of up to 512 tokens and each head shape of the engines.
///
/// Falsified by a bound of 2^20 + N*H*D floats, and by a count that drops the cap of two key tiles a segment.
#[test]
fn the_key_segments_need_no_slot_size_and_fit_the_scratch_bound() {
    for (heads, d) in [(16u32, 512u32), (16, 256), (16, 128), (32, 64)] {
        let bound = tasks::attn_pv_scratch_floats(512, heads as usize, d as usize);
        for n in 1..=512u32 {
            for p0 in [0u32, 1, 31, 256, 4096, 65_000] {
                let segs = tasks::attn_pv_segments(heads, n, d, p0 + n);
                assert!(segs >= 1);
                assert!(
                    segs <= ((p0 + n).div_ceil(16) / 2).max(1),
                    "{heads} heads of {d}, {n} tokens after {p0}: {segs} segments for too few key tiles"
                );
                let need = segs as usize * (n * heads * d) as usize;
                assert!(
                    need <= bound,
                    "{heads} heads of {d}, {n} tokens after {p0}: {segs} segments need {need} floats, past {bound}"
                );
            }
        }
    }
    // The case of the defect of 2026-11-06: 64 tokens after 256 keys, 16 heads of 256 values.
    assert_eq!(tasks::attn_pv_segments(16, 64, 256, 320), 8);
}

/// A call of `tasks::subsample` over 9 time rows of 128 mel rows, with `cin` channels in and 256 out.
fn subsample(cin: u32, silu: bool) -> MkTaskDescriptor {
    tasks::subsample(0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 9, 128, cin, 256, silu)
}

/// `OP_SUBSAMPLE` reads the plane from `a`, the taps from `b`, the bias from `x`, and silu from `layer`.
#[test]
fn subsample_puts_its_fields_where_the_kernel_reads_them() {
    let task = subsample(1, true);
    assert_eq!(task.opcode, nagaya_mk_bmg::protocol::OP_SUBSAMPLE);
    assert_eq!(
        (
            task.a, task.b, task.x, task.c, task.n, task.k, task.p0, task.m, task.layer
        ),
        (0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 9, 128, 1, 256, 1)
    );
    let depthwise = subsample(256, false);
    assert_eq!((depthwise.p0, depthwise.layer), (256, 0));
}

/// The output plane is half of each side, rounded up: `ceil(n / 2)` by `ceil(k / 2)` by `m`.
#[test]
fn subsample_gives_half_of_each_side_rounded_up() {
    assert_eq!(tasks::subsample_out(149, 128), (75, 64));
    assert_eq!(tasks::subsample_out(75, 64), (38, 32));
    assert_eq!(tasks::subsample_out(38, 32), (19, 16));
    assert_eq!(tasks::subsample_out(1, 1), (1, 1));
}

/// A channel in reads channel 0, and a depthwise convolution reads its own channel. Other counts have no meaning.
#[test]
#[should_panic(expected = "OP_SUBSAMPLE reads 1 channel or 256 channels, and it got 16")]
fn subsample_refuses_another_count_of_channels_in() {
    subsample(16, false);
}

/// The kernel reads the input while it writes the output, so they must not overlap.
#[test]
#[should_panic(expected = "OP_SUBSAMPLE: the output overlaps the input")]
fn subsample_refuses_an_output_over_its_input() {
    tasks::subsample(0x10_0000, 0x20_0000, 0x30_0000, 0x10_0100, 9, 128, 1, 256, false);
}

#[test]
#[should_panic(expected = "OP_SUBSAMPLE takes at least one time row and one mel row")]
fn subsample_refuses_an_empty_plane() {
    tasks::subsample(0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 0, 128, 1, 256, false);
}

/// A call of `tasks::group_attn` with head size `d`, `n` rows in groups of `group`, and score scale `scale`.
fn group_attn(d: u32, n: u32, group: u32, scale: f32) -> MkTaskDescriptor {
    tasks::group_attn(0, 0, 0, 0, n, 32, 8, d, group, scale)
}

/// The depth decoder of LFM3 attends over groups of 12 rows with heads of 32 values.
#[test]
fn group_attn_takes_whole_groups_of_rows() {
    let task = group_attn(32, 48, 12, 0.5);
    assert_eq!((task.n, task.k, task.p0, task.m), (48, 12, 32, 32 | 8 << 16));
    assert_eq!(task.layer, 0.5f32.to_bits());
}

/// A part of a group would attend across the end of the rows.
#[test]
#[should_panic(expected = "OP_LFM3_GROUP_ATTN takes whole groups")]
fn group_attn_refuses_a_part_of_a_group() {
    group_attn(32, 20, 12, 1.0);
}

/// The kernel reads a group of 0 as every row.
#[test]
#[should_panic(expected = "OP_LFM3_GROUP_ATTN takes whole groups")]
fn group_attn_refuses_a_group_of_zero() {
    group_attn(32, 12, 0, 1.0);
}

#[test]
#[should_panic(expected = "OP_LFM3_GROUP_ATTN takes a head of 16 to 512 values in 16s")]
fn group_attn_refuses_a_head_that_is_not_a_multiple_of_16() {
    group_attn(40, 12, 12, 1.0);
}

#[test]
#[should_panic(expected = "OP_LFM3_GROUP_ATTN reads a scale of 0.0 as 1.0")]
fn group_attn_refuses_a_scale_of_zero() {
    group_attn(32, 12, 12, 0.0);
}

/// The polar form reads the magnitudes and the angles of a row, so the row must hold both.
#[test]
#[should_panic(expected = "OP_LFM3_POLAR reads 641 magnitudes and 641 angles")]
fn polar_refuses_a_row_shorter_than_its_bins() {
    tasks::polar(0, 0, 6, 641, 1280);
}

/// The inverse FFT of 1,280 samples takes 641 bins of each row, and its table holds a cosine and a sine for each
/// sample.
#[test]
fn irfft_takes_half_the_samples_and_one_bin() {
    let task = tasks::irfft(1, 2, 3, 4, 6, 1280);
    assert_eq!(
        (task.a, task.b, task.x, task.c, task.n, task.k, task.p0),
        (1, 2, 3, 4, 6, 1280, 641)
    );
    assert!(std::panic::catch_unwind(|| tasks::irfft(0, 0, 0, 0, 1, 1281)).is_err());
    let table = tasks::irfft_table(8);
    assert_eq!(table.len(), 16);
    assert_eq!((table[0], table[1]), (1.0, 0.0));
    assert_eq!(
        table[4], 6.123_234e-17_f64 as f32,
        "j = 2 is a quarter turn: the cosine is the f64 rounding"
    );
    assert_eq!(table[5], 1.0);
}

/// `OP_VISION_ROPE` rotates in place, with the grid in `p0` and eps and theta in `x`.
#[test]
fn vision_rope_puts_its_fields_where_the_kernel_reads_them() {
    use nagaya_mk_bmg::protocol::OP_VISION_ROPE;
    let task = tasks::vision_rope(0x10_0000, 0x20_0000, 308 * 16, 16, 64, 14, 22, 1e-5, 1e6);
    assert_eq!(task.opcode, OP_VISION_ROPE);
    assert_eq!((task.a, task.b, task.c), (0x10_0000, 0x20_0000, 0x10_0000));
    assert_eq!((task.n, task.m, task.k, task.p0), (308 * 16, 16, 64, 22 | 14 << 16));
    assert_eq!(f32::from_bits(task.x as u32), 1e-5);
    assert_eq!(f32::from_bits((task.x >> 32) as u32), 1e6);
}

/// The merge takes 2 by 2 patches, so each side of the grid is even, and the rows are whole frames.
#[test]
fn vision_rope_refuses_an_odd_grid_and_part_of_a_frame() {
    let odd = panic_message(|| tasks::vision_rope(0, 0, 15 * 22 * 16, 16, 64, 15, 22, 1e-5, 1e6));
    assert!(odd.contains("OP_VISION_ROPE takes a grid with even sides"), "{odd}");
    let part = panic_message(|| tasks::vision_rope(0, 0, 300 * 16, 16, 64, 14, 22, 1e-5, 1e6));
    assert!(part.contains("whole frames of 308 patches of 16 heads"), "{part}");
    let head = panic_message(|| tasks::vision_rope(0, 0, 308 * 16, 16, 62, 14, 22, 1e-5, 1e6));
    assert!(head.contains("a multiple of 4 values"), "{head}");
}

/// `OP_VISION_CONV` reads the in_proj rows from `a`, the taps from `b`, and the grid from `p0`.
#[test]
fn vision_conv_puts_its_fields_where_the_kernel_reads_them() {
    use nagaya_mk_bmg::protocol::OP_VISION_CONV;
    let task = tasks::vision_conv(0x10_0000, 0x20_0000, 0x4000_0000, 468, 1024, 18, 26);
    assert_eq!(task.opcode, OP_VISION_CONV);
    assert_eq!((task.a, task.b, task.c), (0x10_0000, 0x20_0000, 0x4000_0000));
    assert_eq!((task.n, task.k, task.p0), (468, 1024, 26 | 18 << 16));
}

/// A lane takes one channel of 16, a lane reads its neighbours, and the rows are whole frames.
#[test]
fn vision_conv_refuses_what_its_kernel_cannot_take() {
    let channels = panic_message(|| tasks::vision_conv(0, 0x100_0000, 0x200_0000, 16, 40, 4, 4));
    assert!(channels.contains("OP_VISION_CONV takes channels in 16s"), "{channels}");
    let frames = panic_message(|| tasks::vision_conv(0, 0x100_0000, 0x200_0000, 20, 64, 4, 4));
    assert!(frames.contains("whole frames of 16 patches"), "{frames}");
    let grid = panic_message(|| tasks::vision_conv(0, 0x100_0000, 0x200_0000, 12, 64, 4, 3));
    assert!(grid.contains("a grid with even sides"), "{grid}");
    // 16 rows of 3 * 64 floats from 0; the output starts inside them.
    let over = panic_message(|| tasks::vision_conv(0, 0x100_0000, 0x100, 16, 64, 4, 4));
    assert!(over.contains("the output overlaps the in_proj rows"), "{over}");
}

/// The opcodes of the vision ops in `protocol.rs` are the ones that `kernels/megakernel.cl` defines, in the range 100
/// to 109 of the vision tower (milestone 24 step 7).
#[test]
fn the_vision_opcodes_of_the_host_are_the_opcodes_of_the_kernel() {
    use nagaya_mk_bmg::protocol::{OP_VISION_CONV, OP_VISION_ROPE};
    let source = include_str!("../kernels/megakernel.cl");
    for (name, code) in [("OP_VISION_ROPE", OP_VISION_ROPE), ("OP_VISION_CONV", OP_VISION_CONV)] {
        let line = source
            .lines()
            .find(|l| l.split_whitespace().take(2).eq(["#define", name]))
            .unwrap_or_else(|| panic!("megakernel.cl does not define {name}"));
        let value: u16 = line.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert_eq!(value, code, "{name}");
        assert!((100..110).contains(&code), "{name} is {code}");
    }
}

/// `OP_CONTROL` changes `n` rows of `k` values in place: `a` is the residual, `b` the vector of the layer, `p0` the
/// operation (1 add, 2 ablate, 3 clamp), and `m` the bits of the strength (engine-abi "Control vectors").
///
/// Falsified by a builder that puts the strength in `k`.
#[test]
fn control_puts_its_fields_where_the_kernel_reads_them() {
    use nagaya_mk_bmg::protocol::{ControlOp, OP_CONTROL};
    for (op, code) in [(ControlOp::Add, 1), (ControlOp::Ablate, 2), (ControlOp::Clamp, 3)] {
        let task = tasks::control(10, 20, 7, 2048, op, -1.5);
        assert_eq!(task.opcode, OP_CONTROL);
        assert_eq!((task.a, task.b, task.n, task.k, task.p0), (10, 20, 7, 2048, code));
        assert_eq!(f32::from_bits(task.m), -1.5);
    }
    assert_eq!(tasks::control(10, 20, 7, 2048, ControlOp::Clamp, 0.0).m, 0);
}

/// An add or an ablate with strength 0 changes nothing, so the engine skips it, and the builder refuses it. Thus the
/// plain bits never depend on the sign of a zero.
#[test]
#[should_panic(expected = "an add or an ablate with strength 0 changes nothing")]
fn control_refuses_an_item_that_changes_nothing() {
    tasks::control(10, 20, 7, 2048, nagaya_mk_bmg::protocol::ControlOp::Ablate, 0.0);
}

/// A strength that is not finite has no meaning for the residual.
#[test]
#[should_panic(expected = "a control strength is finite")]
fn control_refuses_a_strength_that_is_not_finite() {
    tasks::control(10, 20, 7, 2048, nagaya_mk_bmg::protocol::ControlOp::Add, f32::NAN);
}
