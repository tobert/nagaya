//! The fields of the vision ops of Inkling 2 (`nagaya_mk_bmg::tasks`), and the calls that each builder refuses. The
//! kernels read these fields (`kernels/inkling2_vision_ops.cl`), so a field in the wrong place is a defect that
//! no card is needed to see.

use nagaya_mk_bmg::protocol::{
    OP_INKLING2_VISION_ATTN, OP_INKLING2_VISION_EMBED, OP_INKLING2_VISION_ROPE, OP_VISION_ATTN,
};
use nagaya_mk_bmg::tasks;

#[test]
fn the_embed_task_names_its_rows_its_table_its_taps_and_its_grid() {
    let t = tasks::inkling2_vision_embed(0x1000, 0x2000, 0x3000, 0x10_0000, 3, 6, 1024);
    assert_eq!(t.opcode, OP_INKLING2_VISION_EMBED);
    assert_eq!((t.a, t.b, t.x, t.c), (0x1000, 0x2000, 0x3000, 0x10_0000));
    assert_eq!((t.n, t.k, t.p0), (72, 1024, 3 | 6 << 16));
    // The largest image of the size rule.
    let t = tasks::inkling2_vision_embed(0, 0, 0, 1 << 40, 103, 103, 1024);
    assert_eq!(t.n, 4 * 103 * 103);
}

#[test]
fn the_rope_task_names_both_planes_the_table_and_the_columns() {
    let t = tasks::inkling2_vision_rope(0x1000, 0x2000, 0x3000, 73, 1024, 6);
    assert_eq!(t.opcode, OP_INKLING2_VISION_ROPE);
    assert_eq!((t.a, t.b, t.x, t.n, t.k, t.p0), (0x1000, 0x2000, 0x3000, 73, 1024, 6));
}

#[test]
fn the_attention_is_the_vision_attention_of_lfm3_under_its_own_opcode() {
    let t = tasks::inkling2_vision_attn(1, 2, 3, 4, 73, 16, 16, 64, 0.125);
    let mut lfm3 = tasks::vision_attn(1, 2, 3, 4, 73, 16, 16, 64, 0.125);
    assert_eq!((t.opcode, lfm3.opcode), (OP_INKLING2_VISION_ATTN, OP_VISION_ATTN));
    lfm3.opcode = OP_INKLING2_VISION_ATTN;
    assert_eq!(format!("{t:?}"), format!("{lfm3:?}"));
}

#[test]
#[should_panic(expected = "1 to 103 image tokens on each side")]
fn an_embed_of_a_side_past_the_rotary_table_is_refused() {
    tasks::inkling2_vision_embed(0, 0, 0, 1 << 40, 104, 1, 1024);
}

#[test]
#[should_panic(expected = "overlaps the patch rows")]
fn an_embed_that_writes_over_its_patch_rows_is_refused() {
    tasks::inkling2_vision_embed(0x1000, 0, 0, 0x1000, 1, 1, 1024);
}

#[test]
#[should_panic(expected = "heads of 64 values")]
fn a_rope_of_rows_that_are_not_heads_of_64_is_refused() {
    tasks::inkling2_vision_rope(1, 2, 3, 5, 1000, 1);
}

#[test]
#[should_panic(expected = "whole lines of 2 image tokens")]
fn a_rope_of_rows_that_are_not_a_class_row_and_whole_lines_is_refused() {
    tasks::inkling2_vision_rope(1, 2, 3, 8, 1024, 2);
}

#[test]
#[should_panic(expected = "two buffers")]
fn a_rope_of_one_plane_twice_is_refused() {
    tasks::inkling2_vision_rope(1, 1, 3, 5, 1024, 1);
}
