//! The plan of the tower with no card: the rotary table, the tasks of one image, and the bytes of the buffers.

use super::*;
use crate::host::vision::{MAX_PATCH_SIDE, patch_position};
use crate::weights::Count;
use nagaya_mk_bmg::protocol::{
    MAX_TASKS_PER_CARD, OP_COPY, OP_INKLING2_VISION_ATTN, OP_INKLING2_VISION_EMBED, OP_INKLING2_VISION_ROPE, OP_MATMUL,
};

/// Weights and buffers at fixed fake addresses, so that a test can name each pointer of a task.
fn fake() -> (Weights, Bufs) {
    let mut next = 0x1000u64;
    let mut z = |bytes: usize| {
        let at = next;
        next += bytes.next_multiple_of(64) as u64;
        at
    };
    let lin = |z: &mut dyn FnMut(usize) -> u64| Linear {
        weight: z(64),
        bias: z(64),
    };
    let mut w = Weights {
        patch: z(64),
        position: z(64),
        class_row: z(64),
        ln_pre: Norm {
            weight: z(64),
            bias: z(64),
        },
        proj_in: z(64),
        proj_out: z(64),
        rope: z(64),
        layers: Vec::new(),
    };
    for _ in 0..LAYERS {
        w.layers.push(Layer {
            ln_1: Norm {
                weight: z(64),
                bias: z(64),
            },
            ln_2: Norm {
                weight: z(64),
                bias: z(64),
            },
            ls_1: z(64),
            ls_2: z(64),
            q: lin(&mut z),
            k: lin(&mut z),
            v: lin(&mut z),
            out: lin(&mut z),
            fc: lin(&mut z),
            proj: lin(&mut z),
        });
    }
    let b = Bufs::new(&mut z, 18);
    (w, b)
}

#[test]
fn the_rotary_table_holds_the_cosine_and_the_sine_of_each_angle_of_the_host_form() {
    let table = rope_table();
    assert_eq!(table.len(), 2 * ROPE_POSITIONS * ROPE_FREQS);
    assert_eq!(ROPE_POSITIONS, 207);
    assert_eq!(ROPE_POSITIONS, nagaya_mk_bmg::tasks::INKLING2_VISION_POSITIONS);
    assert_eq!(ROPE_FREQS, nagaya_mk_bmg::tasks::INKLING2_VISION_FREQS);
    let (cos, sin) = table.split_at(ROPE_POSITIONS * ROPE_FREQS);
    // Each pair of each patch of the largest image reads the angle of the host form at the place the op reads.
    for line in 0..=MAX_PATCH_SIDE {
        for column in [0, 1, 17, MAX_PATCH_SIDE] {
            for j in 0..HEAD_DIM / 2 {
                let p = if j < ROPE_FREQS { column } else { line };
                let at = p * ROPE_FREQS + j % ROPE_FREQS;
                let a = rope_angle(j, line, column);
                assert_eq!(
                    cos[at].to_bits(),
                    a.cos().to_bits(),
                    "cos of pair {j} at ({line}, {column})"
                );
                assert_eq!(
                    sin[at].to_bits(),
                    a.sin().to_bits(),
                    "sin of pair {j} at ({line}, {column})"
                );
            }
        }
    }
}

#[test]
fn the_tasks_of_one_image_follow_the_host_form_and_write_only_the_buffers_and_the_output() {
    let (w, b) = fake();
    let g = Grid { rows: 3, columns: 6 };
    let (rows, out) = (0x10_0000_0000u64, 0x20_0000_0000u64);
    let axes_at = 0x30_0000_0000u64;
    let run = tower(&w, &b, g, rows, axes_at, out);
    assert_eq!(run.tasks.len(), TOWER_TASKS);
    assert_eq!(TOWER_TASKS, 444);
    // Two images of one call fit one ring.
    const { assert!(2 * TOWER_TASKS < MAX_TASKS_PER_CARD) };
    let stages: Vec<Stage> = run.marks.iter().map(|m| m.stage).collect();
    let mut want = vec![Stage::PatchEmbed, Stage::Embeddings, Stage::PreLayernorm];
    want.extend((0..LAYERS).map(Stage::Layer));
    want.push(Stage::Projector);
    assert_eq!(stages, want);
    assert_eq!(run.marks.last().unwrap().tasks, TOWER_TASKS);
    assert_eq!(
        (
            run.marks[0].rows,
            run.marks[1].rows,
            run.marks[25].rows,
            run.marks[26].rows
        ),
        (72, 73, 73, 18)
    );
    let written = [b.x, b.h, b.q, b.k, b.v, b.m, out];
    for (i, t) in run.tasks.iter().enumerate() {
        assert!(
            written.contains(&t.c) || t.opcode == OP_INKLING2_VISION_ROPE,
            "task {i} writes {:#x}",
            t.c
        );
        if t.opcode == OP_INKLING2_VISION_ROPE {
            assert_eq!((t.a, t.b, t.x, t.n, t.k, t.p0), (b.q, b.k, w.rope, 73, 1024, 6));
        }
    }
    let first = &run.tasks[0];
    assert_eq!(
        (first.opcode, first.a, first.n, first.m, first.k),
        (OP_MATMUL, rows, 72, 1024, 1200)
    );
    let class = &run.tasks[1];
    assert_eq!(
        (class.opcode, class.a, class.c, class.n),
        (OP_COPY, w.class_row, b.x, 1024)
    );
    let embed = &run.tasks[2];
    assert_eq!(embed.opcode, OP_INKLING2_VISION_EMBED);
    assert_eq!(
        (embed.a, embed.b, embed.x, embed.c, embed.p0),
        (b.q, w.position, axes_at, b.x, 3 | 6 << 16)
    );
    let attn: Vec<_> = run
        .tasks
        .iter()
        .filter(|t| t.opcode == OP_INKLING2_VISION_ATTN)
        .collect();
    assert_eq!(attn.len(), LAYERS);
    assert_eq!(
        (attn[0].n, attn[0].m, attn[0].p0, attn[0].layer),
        (73, 16 | 16 << 16, 64, 0.125f32.to_bits())
    );
    // The projector reads the rows after the class row as 18 rows of 4,096, and writes 18 rows of 2,304 at out.
    let proj_in = &run.tasks[TOWER_TASKS - 3];
    assert_eq!(
        (proj_in.a, proj_in.n, proj_in.m, proj_in.k),
        (b.x + 4096, 18, 4096, 4096)
    );
    let proj_out = &run.tasks[TOWER_TASKS - 1];
    assert_eq!((proj_out.c, proj_out.n, proj_out.m, proj_out.k), (out, 18, 2304, 4096));
    // The patch order of the embed op and of the rotation is the host's.
    assert_eq!(patch_position(5, g), (0, 3));
}

#[test]
fn the_buffers_hold_the_largest_image_of_the_size_rule() {
    assert_eq!(MAX_TOKENS, 10_609);
    assert_eq!(tower_rows(MAX_TOKENS), 42_437);
    assert_eq!(Bufs::bytes(MAX_TOKENS), 5 * 42_437 * 1024 * 4 + 42_437 * 4096 * 4);
    assert_eq!(Bufs::bytes(MAX_TOKENS), 1_564_397_568);
    assert_eq!(rows_bytes(MAX_TOKENS), 203_692_800);
    assert_eq!(out_bytes(MAX_TOKENS), 97_772_544);
    let mut count = Count::default();
    let b = Bufs::new(&mut |bytes| count.alloc(bytes), MAX_TOKENS);
    assert_eq!(count.bytes, Bufs::bytes(MAX_TOKENS));
    assert_eq!(b.max_tokens, MAX_TOKENS);
}

#[test]
#[should_panic(expected = "hold 18 image tokens, and the image has 19")]
fn an_image_larger_than_the_buffers_is_a_defect_of_the_caller() {
    let (w, b) = fake();
    tower(&w, &b, Grid { rows: 1, columns: 19 }, 0, 0, 0);
}

#[test]
fn the_axes_hold_the_taps_and_the_weights_of_the_host_form_for_each_line_and_column() {
    let g = Grid { rows: 3, columns: 6 };
    let a = axes(g);
    assert_eq!(a.len(), 16 * (6 + 12));
    assert_eq!(a.len(), nagaya_mk_bmg::tasks::inkling2_vision_axes_bytes(3, 6));
    let word = |i: usize| u32::from_le_bytes(a[4 * i..4 * i + 4].try_into().unwrap());
    for (base, size) in [(0usize, 6usize), (6, 12)] {
        for index in 0..size {
            let [(t0, w0), (t1, w1)] = axis_taps(index, size);
            let e = 4 * (base + index);
            assert_eq!(
                [word(e), word(e + 1)],
                [t0 as u32, t1 as u32],
                "the taps of {index} of {size}"
            );
            assert_eq!(
                [word(e + 2), word(e + 3)],
                [w0.to_bits(), w1.to_bits()],
                "the weights of {index} of {size}"
            );
        }
    }
    let largest = Grid {
        rows: 103,
        columns: 103,
    };
    assert_eq!(axes(largest).len(), AXES_BYTES);
}
