//! The builder of the attention of Inkling 2 puts each field where `kernels/inkling2_attn_ops.cl` reads it, and it
//! refuses the blocks that the kernel cannot take. It needs no card.

use nagaya_mk_bmg::protocol::{OP_INKLING2_ATTN, OP_INKLING2_ATTN_SUM};
use nagaya_mk_bmg::tasks::{self, Inkling2Attention};

/// A block of `n` positions from `p0` on a ring of `ring` rows (0: cells of a slot of 4,096 positions).
fn block(n: u32, p0: u32, ring: u32, chunk_keys: u32) -> Inkling2Attention {
    Inkling2Attention {
        q: 0x1000,
        kv: 0x2000,
        rel: 0x3000,
        proj: 0x4000,
        sinks: 0x5000,
        out: 0x6000,
        n,
        p0,
        max_seq: 4096,
        ring,
        chunk_keys,
    }
}

/// The value of `#define <name>` in a kernel source.
fn define(source: &str, name: &str) -> u32 {
    let line = source
        .lines()
        .find(|l| l.split_whitespace().take(2).eq(["#define", name]))
        .unwrap_or_else(|| panic!("the kernel does not define {name}"));
    line.split_whitespace().nth(2).unwrap().parse().unwrap()
}

/// Falsified by a builder that puts the relative states in `x` and proj in `c`.
#[test]
fn the_builder_puts_each_field_where_the_kernel_reads_it() {
    let [attn, sum] = block(37, 600, 0, 2048).tasks();
    assert_eq!(attn.opcode, OP_INKLING2_ATTN);
    assert_eq!((attn.a, attn.b, attn.c, attn.x), (0x1000, 0x2000, 0x3000, 0x4000));
    assert_eq!((attn.m, attn.n, attn.k, attn.p0, attn.layer), (4096, 37, 2048, 600, 0));
    assert_eq!(sum.opcode, OP_INKLING2_ATTN_SUM);
    assert_eq!((sum.a, sum.c), (0x5000, 0x6000));
    assert_eq!((sum.n, sum.k, sum.p0, sum.layer), (37, 2048, 600, 0));
    let [attn, sum] = block(1, 5000, 2048, 256).tasks();
    assert_eq!((attn.layer, sum.layer, attn.k, sum.k), (2048, 2048, 256, 256));
}

/// The opcodes of `protocol.rs` are the ones that `kernels/megakernel.cl` defines, in the range 160 to 169 of this
/// lane. The constants of the helpers match the host form.
#[test]
fn the_opcodes_and_constants_of_the_host_are_those_of_the_kernel() {
    let source = include_str!("../kernels/megakernel.cl");
    for (name, code) in [
        ("OP_INKLING2_ATTN", OP_INKLING2_ATTN),
        ("OP_INKLING2_ATTN_SUM", OP_INKLING2_ATTN_SUM),
    ] {
        assert_eq!(define(source, name), code as u32, "{name}");
        assert!((160..170).contains(&code), "{name} is {code}");
    }
    let helpers = include_str!("../kernels/inkling2_attn_helpers.cl");
    assert_eq!(define(helpers, "INKLING2_ATTN_KEYS"), tasks::INKLING2_ATTN_KEYS);
    assert_eq!(
        define(helpers, "INKLING2_ATTN_TEAM_FLOATS") as usize,
        tasks::INKLING2_ATTN_TEAM_FLOATS
    );
    assert_eq!(define(helpers, "INKLING2_WINDOW"), tasks::INKLING2_WINDOW);
}

/// The kernel scales q . k by the f32 of 128^-0.5 as a literal. The host form computes `128f32.powf(-0.5)`.
///
/// Falsified by the literal 0x3DB504F4 in the kernel.
#[test]
fn the_score_scale_of_the_kernel_is_the_scale_of_the_host_form() {
    assert_eq!(128f32.powf(-0.5).to_bits(), tasks::INKLING2_ATTN_SCALE_BITS);
    let literal = format!("as_float(0x{:08X}u)", tasks::INKLING2_ATTN_SCALE_BITS);
    assert!(
        include_str!("../kernels/inkling2_attn_helpers.cl").contains(&literal),
        "the kernel does not scale by {literal}"
    );
}

/// The chunks sit on a grid of absolute key positions: chunk c holds keys c * chunk_keys onward.
///
/// Falsified by a first chunk that ignores the window of a ring.
#[test]
fn the_chunks_of_a_block_sit_on_the_absolute_grid() {
    // A full layer reads from key 0.
    assert_eq!(block(37, 0, 0, 2048).chunks(), (0, 1));
    assert_eq!(block(88, 512, 0, 128).chunks(), (0, 5));
    assert_eq!(block(1, 4095, 0, 2048).chunks(), (0, 2));
    // A sliding layer reads from position p0 - 1023: 777 here, in chunk 3 of 256 keys. The last key, 2311, is in
    // chunk 9.
    assert_eq!(block(512, 1800, 2048, 256).chunks(), (3, 7));
    assert_eq!(block(1, 1023, 2048, 256).chunks(), (0, 4));
    assert_eq!(block(1, 1024, 2048, 256).chunks(), (0, 5));
    assert_eq!(block(1, 1025, 2048, 256).chunks(), (0, 5));
    assert_eq!(block(1, 1280, 2048, 256).chunks(), (1, 5));
}

/// The scratch holds a max, a sum and 128 values for each chunk, position and head, then the scratch of each team.
#[test]
fn the_scratch_of_a_block_holds_its_chunks_and_the_teams() {
    let b = block(512, 1800, 2048, 256);
    assert_eq!(
        b.scratch_floats(),
        (7 * 512 * 24 * 130usize).next_multiple_of(16) + 512 * tasks::INKLING2_ATTN_TEAM_FLOATS
    );
    // The bound of a slot covers each block of the slot with the default chunks. In a short slot the sliding blocks
    // take the most chunks: a block of 512 from position 1,278 reads keys 255 to 1,789, in 7 chunks of 256.
    for max_seq in [2048u32, 32768] {
        let bound = tasks::inkling2_attn_scratch_floats(512, max_seq as usize);
        for ring in [0, 2048] {
            for n in [1, 16, 300, 512] {
                for p0 in [0, 1, 1023, 1278, 1500, 2047, 4000, 8191, 30000, max_seq - n] {
                    if ring == 0 && p0 + n > max_seq {
                        continue;
                    }
                    let b = Inkling2Attention {
                        max_seq,
                        chunk_keys: tasks::inkling2_chunk_keys(ring),
                        ..block(n, p0, ring, 128)
                    };
                    assert!(
                        b.scratch_floats() <= bound,
                        "max_seq {max_seq} n {n} p0 {p0} ring {ring}"
                    );
                }
            }
        }
    }
}

/// A sliding layer has small chunks, so a decode step spreads over more teams. A full layer has larger chunks, so the
/// partial sums of a block of a long slot fit the scratch.
#[test]
fn the_default_chunks_depend_only_on_the_kind_of_layer() {
    assert_eq!(tasks::inkling2_chunk_keys(2048), tasks::INKLING2_SLIDING_CHUNK_KEYS);
    assert_eq!(tasks::inkling2_chunk_keys(0), tasks::INKLING2_FULL_CHUNK_KEYS);
}

#[test]
#[should_panic(expected = "OP_INKLING2_ATTN takes 1 to 512 positions")]
fn the_builder_refuses_an_empty_block() {
    block(0, 0, 0, 2048).tasks();
}

/// The kernel keeps the scratch of a block of 512 at most, the largest block of the engine.
#[test]
#[should_panic(expected = "OP_INKLING2_ATTN takes 1 to 512 positions")]
fn the_builder_refuses_a_block_above_512() {
    block(513, 0, 0, 2048).tasks();
}

/// A round of 128 keys must not cross the end of a chunk.
#[test]
#[should_panic(expected = "a chunk of OP_INKLING2_ATTN holds a whole count of rounds of 128 keys")]
fn the_builder_refuses_a_chunk_that_is_not_a_multiple_of_128() {
    block(16, 0, 0, 200).tasks();
}

#[test]
#[should_panic(expected = "a chunk of OP_INKLING2_ATTN holds a whole count of rounds of 128 keys")]
fn the_builder_refuses_a_chunk_of_no_keys() {
    block(16, 0, 0, 0).tasks();
}

/// The keys of a block and the window before its first position must have rows of their own in a ring.
///
/// Falsified by a check of `ring >= n` alone.
#[test]
#[should_panic(expected = "a ring of OP_INKLING2_ATTN holds the window and the block")]
fn the_builder_refuses_a_ring_that_cannot_hold_the_window_and_the_block() {
    block(512, 3000, 1408, 256).tasks();
}

/// A round of 128 keys must not cross the end of a ring.
#[test]
#[should_panic(expected = "a ring of OP_INKLING2_ATTN has a whole count of rounds of 128 rows")]
fn the_builder_refuses_a_ring_that_is_not_a_multiple_of_128() {
    block(16, 3000, 2000, 256).tasks();
}

/// Cells hold one row for each position of the slot.
#[test]
#[should_panic(expected = "the block of OP_INKLING2_ATTN ends past the cells of the slot")]
fn the_builder_refuses_a_block_past_the_cells_of_the_slot() {
    block(16, 4090, 0, 2048).tasks();
}

/// The kernel sums positions in `u32` and takes distances as `int`, so a block past 2^24 positions could wrap.
///
/// Falsified by a builder with no limit of the positions.
#[test]
#[should_panic(expected = "past the 16777216 positions that the kernel counts")]
fn the_builder_refuses_a_block_past_the_positions_that_the_kernel_counts() {
    block(2, (1 << 24) - 1, 2048, 256).tasks();
}

/// The sizes of the scratch check the block first, so a chunk of no keys is a refusal, not a division by zero.
#[test]
#[should_panic(expected = "a chunk of OP_INKLING2_ATTN holds a whole count of rounds of 128 keys")]
fn the_scratch_of_a_chunk_of_no_keys_is_a_refusal() {
    block(16, 0, 0, 0).scratch_floats();
}

/// The last position that the kernel counts is a block of its own.
#[test]
fn the_builder_takes_the_last_position_that_the_kernel_counts() {
    let [attn, _] = block(1, (1 << 24) - 1, 2048, 256).tasks();
    assert_eq!(attn.p0, (1 << 24) - 1);
}
