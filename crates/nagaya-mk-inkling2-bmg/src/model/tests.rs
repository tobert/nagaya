use super::*;
use nagaya_mk_bmg::protocol::{OP_ATTN_RESIDUAL, OP_COPY, OP_EMBED, OP_INKLING2_ATTN, OP_MATMUL, OP_MESSAGE_GATE};

use crate::host::weights::CYCLE_END;
use crate::spec::{DEFAULT_EXTRA_RUNS, MAX_BLOCK, runs};

/// Fake addresses of buffers that do not overlap, so that a task names which buffer it reads.
fn counter() -> impl FnMut(usize) -> u64 {
    let mut next = 1u64 << 40;
    move |bytes| {
        let at = next;
        next += spec::rounded(bytes) as u64;
        at
    }
}

/// Weights at fake addresses: field `f` of layer `l` is `(l + 1) << 32 | f`.
fn weights() -> Weights {
    let layers = (0..LAYERS)
        .map(|l| {
            let a = |f: u64| ((l as u64 + 1) << 32) | f;
            Layer {
                attn_queries: if l == 0 { 0 } else { a(1) },
                mlp_query: a(2),
                norms: if is_full(l) { [a(3), a(4), a(5)] } else { [a(3); 3] },
                mlp_norm: a(6),
                wq: a(7),
                wr: a(8),
                sink: a(9),
                gate_attn: a(10),
                wk: a(11),
                wv: a(12),
                wo: a(13),
                q_scale: a(14),
                rel_scale: a(15),
                rel_proj: a(16),
                k_conv: a(17),
                v_conv: a(18),
                attn_conv: a(19),
                mlp_conv: a(20),
                router: a(21),
                router_bias: a(22),
                gates: if (CYCLE_START..CYCLE_END).contains(&l) {
                    [a(23), a(24), a(25), a(26)]
                } else {
                    [0; 4]
                },
                w13: a(27),
                w2: a(28),
                mxfp8: l >= 20,
            }
        })
        .collect();
    Weights {
        embed: 1,
        embed_norm: 2,
        head: 3,
        norm: 4,
        output_query: 5,
        message_queries: 6 << 20,
        layers,
    }
}

struct Fixture {
    w: Weights,
    bufs: Bufs,
    slot: SlotMemory,
    runs: Vec<Run>,
}

fn fixture(seq: usize) -> Fixture {
    let mut z = counter();
    Fixture {
        w: weights(),
        bufs: Bufs::new(&mut z, MAX_BLOCK),
        slot: SlotMemory::new(&mut z, seq, DEFAULT_EXTRA_RUNS),
        runs: runs(3),
    }
}

impl Fixture {
    fn pass(&self, pos0: usize, n: usize, logit_rows: Option<&[usize]>) -> Pass {
        pass(
            &self.w,
            &self.bufs,
            TaskRing {
                slot: &self.slot,
                pos0,
                n,
                runs: &self.runs,
                upto: self.runs.len(),
                logit_rows,
                after_embed: &[],
            },
        )
    }
}

fn plane(bufs: &Bufs, at: u64, p: usize) -> u64 {
    bufs.plane(at, p, HIDDEN)
}

#[test]
fn the_buffers_take_the_bytes_of_the_plan() {
    for max_block in [1, 37, 512] {
        let mut total = 0u64;
        Bufs::new(
            &mut |bytes| {
                total += spec::rounded(bytes) as u64;
                0
            },
            max_block,
        );
        assert_eq!(total, spec::bufs_bytes(max_block), "a block of {max_block}");
    }
    assert_eq!(&Bufs::sink_ids_of(2), &[0, 1, 2, 4, 5, 6]);
}

#[test]
fn a_pass_has_the_embedding_48_runs_and_the_head_of_each_logit_row() {
    let f = fixture(32_768);
    let p = f.pass(0, 5, Some(&[4, 0]));
    assert_eq!(p.runs.len(), 48);
    assert_eq!(p.head.iter().map(|t| t.opcode).collect::<Vec<_>>(), [OP_EMBED, 1]);
    assert_eq!((p.head[0].a, p.head[0].b), (f.bufs.ids, f.w.embed));
    let heads: Vec<&MkTaskDescriptor> = p.tail.iter().filter(|t| t.opcode == OP_MATMUL).collect();
    assert_eq!(heads.len(), 2);
    for (i, (t, row)) in heads.iter().zip([4usize, 0]).enumerate() {
        assert_eq!((t.n, t.m, t.b), (1, HEAD_ROWS as u32, f.w.head));
        assert_eq!(t.a, f.bufs.e_norm + (row * HIDDEN * 4) as u64);
        assert_eq!(t.c, f.bufs.logits + (i * HEAD_ROWS * 4) as u64);
        assert_ne!(t.flags & FLAG_SOFTCAP_30, 0, "the head applies the softcap of 30");
    }
    // The output residual reads the 4 block outputs and the last row.
    assert_eq!((p.tail[0].opcode, p.tail[0].m), (OP_ATTN_RESIDUAL, 4 | 1 << 16));
    assert!(f.pass(0, 5, None).tail.is_empty());
}

#[test]
fn each_cycle_after_the_first_starts_from_the_row_after_layer_5() {
    let f = fixture(32_768);
    let p = f.pass(0, 5, Some(&[4]));
    let b = &f.bufs;
    let words = (5 * HIDDEN) as u32;
    let copy = |a, c| (OP_COPY, a, c, words);
    let of = |t: &MkTaskDescriptor| (t.opcode, t.a, t.c, t.n);
    // Run 5 (layer 5) keeps the row after it.
    assert_eq!(of(p.runs[5].last().unwrap()), copy(b.x, b.start));
    for (r, run) in p.runs.iter().enumerate() {
        let starts = f.runs[r].cycle.is_some_and(|c| c >= 2) && f.runs[r].layer == CYCLE_START;
        let head: Vec<_> = run.iter().take(3).map(of).collect();
        let want = [
            copy(plane(b, b.blocks, 2), b.previous),
            copy(b.x, b.y),
            copy(b.start, b.x),
        ];
        assert_eq!(head == want, starts, "run {r}");
    }
    // Only runs 18 and 30 start a cycle.
    let starts: Vec<usize> = (0..48)
        .filter(|r| p.runs[*r].iter().take(3).map(of).collect::<Vec<_>>()[2] == copy(b.start, b.x))
        .collect();
    assert_eq!(starts, [18, 30]);
}

#[test]
fn the_first_layer_of_a_block_keeps_its_input_as_a_block_output() {
    let f = fixture(32_768);
    let p = f.pass(0, 5, None);
    let b = &f.bufs;
    for (r, run) in p.runs.iter().enumerate() {
        let l = f.runs[r].layer;
        let kept: Vec<u64> = run
            .iter()
            .filter(|t| t.opcode == OP_COPY && t.a == b.x && t.c != b.y && t.c != b.start)
            .map(|t| t.c)
            .collect();
        if l.is_multiple_of(6) {
            assert_eq!(kept, [plane(b, b.blocks, l / 6)], "run {r} of layer {l}");
        } else {
            assert!(kept.is_empty(), "run {r} of layer {l}");
        }
        // The attention input reads the block outputs before this layer's, and the expert input reads them after.
        let mixes: Vec<&MkTaskDescriptor> = run
            .iter()
            .filter(|t| t.opcode == OP_ATTN_RESIDUAL && t.a == b.blocks)
            .collect();
        let after = l / 6 + 1;
        let before = if l.is_multiple_of(6) { l / 6 } else { after };
        let queries = if is_full(l) { 3 } else { 1 };
        if l == 0 {
            assert_eq!(mixes.len(), 1, "layer 0 reads the embedding row");
        } else {
            assert_eq!(mixes[0].m, before as u32 | queries << 16, "run {r} of layer {l}");
        }
        assert_eq!(mixes.last().unwrap().m, after as u32 | 1 << 16, "run {r} of layer {l}");
        assert_eq!(mixes.last().unwrap().x, f.w.layers[l].mlp_query);
    }
}

#[test]
fn the_runs_of_later_cycles_read_the_messages_of_the_cycle_before() {
    let f = fixture(32_768);
    let p = f.pass(0, 5, None);
    let b = &f.bufs;
    for (r, run) in p.runs.iter().enumerate() {
        let Run { layer: l, cycle, .. } = f.runs[r];
        let gates: Vec<&MkTaskDescriptor> = run.iter().filter(|t| t.opcode == OP_MESSAGE_GATE).collect();
        let message = run.iter().find(|t| t.opcode == OP_ATTN_RESIDUAL && t.a == b.previous);
        if cycle.is_some_and(|c| c >= 2) {
            let m = message.expect("a later cycle mixes its messages");
            assert_eq!((m.b, m.c, m.m), (b.y, b.m, 1 | 4 << 16), "run {r}");
            assert_eq!(m.x, f.w.message_queries + ((l - CYCLE_START) * 4 * HIDDEN * 4) as u64);
            // The query, key and value gates, then the gate of the expert input with message 3.
            assert_eq!(gates.len(), 4, "run {r}");
            let lw = &f.w.layers[l];
            let mut used: Vec<u64> = gates.iter().map(|g| g.x).collect();
            used.sort();
            let mut want = lw.gates.to_vec();
            want.sort();
            assert_eq!(used, want, "run {r}: each gate once");
            for g in &gates {
                let i = lw.gates.iter().position(|w| *w == g.x).unwrap();
                assert_eq!(g.b, plane(b, b.m, i), "run {r}: gate {i} reads message {i}");
            }
            assert_eq!((gates[3].a, gates[3].c), (b.e, b.e));
        } else {
            assert!(message.is_none() && gates.is_empty(), "run {r}");
        }
    }
}

#[test]
fn a_block_that_crosses_the_end_of_a_ring_moves_its_spill_before_the_attention() {
    let f = fixture(32_768);
    let p = f.pass(2000, 100, None);
    for (r, run) in p.runs.iter().enumerate() {
        let s = f.runs[r].state;
        let at = run.iter().position(|t| t.opcode == OP_INKLING2_ATTN).unwrap();
        let spills: Vec<&MkTaskDescriptor> = run[..at]
            .iter()
            .filter(|t| t.opcode == OP_COPY && (t.c == f.slot.planes[s] || t.c == f.slot.row(s, true, 0)))
            .collect();
        if is_full(f.runs[r].layer) {
            assert!(spills.is_empty(), "run {r}");
            assert_eq!(run[at].layer, 0, "a full run attends to cells");
        } else {
            assert_eq!(spills.len(), 2, "run {r}");
            for (t, value) in spills.iter().zip([false, true]) {
                assert_eq!(t.n, (52 * KV_ROW) as u32);
                assert_eq!(t.a, f.slot.row(s, value, 0) + (RING_ROWS * KV_ROW * 4) as u64);
            }
            assert_eq!(run[at].layer, RING_ROWS as u32);
        }
    }
    // A block inside the ring moves nothing.
    let inside = f.pass(1000, 100, None);
    assert!(
        inside
            .runs
            .iter()
            .all(|run| !run.iter().any(|t| t.opcode == OP_COPY && t.n == (52 * KV_ROW) as u32))
    );
}

#[test]
fn a_short_slot_holds_a_ring_that_does_not_wrap() {
    let f = fixture(64);
    let s = 0;
    assert_eq!(f.slot.ring(s), RING_ROWS);
    assert_eq!(f.slot.ring(5), 0);
    assert_eq!(
        f.slot.row(s, true, 3),
        f.slot.planes[s] + ((3072 + 3) * KV_ROW * 4) as u64
    );
    assert_eq!(
        f.slot.row(5, true, 3),
        f.slot.planes[5] + ((64 + 3) * KV_ROW * 4) as u64
    );
    assert_eq!(f.slot.runs(s, false, 0..64), [(f.slot.planes[s], 64)]);
    let big = fixture(32_768);
    assert_eq!(
        big.slot.runs(0, false, 2000..2100),
        [(big.slot.row(0, false, 2000), 48), (big.slot.planes[0], 52)]
    );
}

#[test]
fn the_launches_hold_whole_runs_and_every_task_in_order() {
    let f = fixture(32_768);
    for (pos0, n) in [(0, 512), (2000, 100), (40, 1)] {
        let p = f.pass(pos0, n, Some(&[n - 1]));
        let all: Vec<MkTaskDescriptor> = p
            .head
            .iter()
            .chain(p.runs.iter().flatten())
            .chain(&p.tail)
            .copied()
            .collect();
        assert_eq!(all.len(), p.len());
        for limit in [LAUNCH_TASKS, 200, 60] {
            let launches = p.launches(limit);
            assert!(launches.iter().all(|l| l.len() <= limit && !l.is_empty()));
            let joined: Vec<MkTaskDescriptor> = launches.concat();
            assert_eq!(joined.len(), all.len());
            assert!(
                joined
                    .iter()
                    .zip(&all)
                    .all(|(a, b)| a.opcode == b.opcode && a.a == b.a && a.c == b.c),
                "the launches keep the order"
            );
            // A cut falls between two runs: each launch but the first starts with the first task of a run.
            let mut at = launches[0].len();
            let starts: Vec<usize> = std::iter::once(p.head.len())
                .chain(p.runs.iter().scan(p.head.len(), |acc, r| {
                    *acc += r.len();
                    Some(*acc)
                }))
                .collect();
            for l in &launches[1..] {
                assert!(starts.contains(&at), "a cut at task {at} is inside a run");
                at += l.len();
            }
        }
        eprintln!(
            "a pass of {n} tokens from {pos0}: {} tasks, {} launches",
            p.len(),
            p.launches(LAUNCH_TASKS).len()
        );
    }
}

/// The copies of the image rows follow the norm of the embedding, so they replace the normed rows of the
/// placeholders, and the first run reads them (engine-abi "The choices of the pass", "The place").
#[test]
fn the_tasks_after_the_embedding_follow_its_norm_before_the_first_run() {
    let f = fixture(32_768);
    let places = [tasks::copy(0x10, f.bufs.x, 7), tasks::copy(0x20, f.bufs.x + 4, 9)];
    let p = pass(
        &f.w,
        &f.bufs,
        TaskRing {
            slot: &f.slot,
            pos0: 0,
            n: 5,
            runs: &f.runs,
            upto: f.runs.len(),
            logit_rows: Some(&[4]),
            after_embed: &places,
        },
    );
    let head: Vec<(u16, u64, u64)> = p.head.iter().map(|t| (t.opcode, t.a, t.c)).collect();
    assert_eq!(
        head,
        [
            (OP_EMBED, f.bufs.ids, f.bufs.o),
            (1, f.bufs.o, f.bufs.x),
            (OP_COPY, 0x10, f.bufs.x),
            (OP_COPY, 0x20, f.bufs.x + 4)
        ]
    );
    assert_eq!(p.len(), f.pass(0, 5, Some(&[4])).len() + 2);
}
