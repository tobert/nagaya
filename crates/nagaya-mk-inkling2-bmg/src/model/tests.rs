use super::*;
use nagaya_mk_bmg::protocol::{
    OP_ATTN_RESIDUAL, OP_COPY, OP_EMBED, OP_HEAD_CLIP, OP_INKLING2_ATTN, OP_INKLING2_ATTN_SUM, OP_INKLING2_CONTROL,
    OP_MATMUL, OP_MESSAGE_GATE, OP_MESSAGE_GATE_DOWN, OP_RESIDUAL_CONV, OP_SIGMOID_ROUTER,
};

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

/// Weights at fake addresses: field `f` of layer `l` is `(l + 1) << 32 | f << 8`. Each one is a multiple of 256
/// bytes, as the ops that read rows in blocks require.
fn weights() -> Weights {
    let layers = (0..LAYERS)
        .map(|l| {
            let a = |f: u64| ((l as u64 + 1) << 32) | f << 8;
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
        embed: 1 << 8,
        embed_norm: 2 << 8,
        head: 3 << 8,
        norm: 4 << 8,
        output_query: 5 << 8,
        message_queries: 6 << 20,
        layers,
    }
}

struct Fixture {
    w: Weights,
    bufs: Bufs,
    slot: SlotMemory,
    runs: Vec<Run>,
    taps: TapBufs,
}

fn fixture(seq: usize) -> Fixture {
    let mut z = counter();
    Fixture {
        w: weights(),
        bufs: Bufs::new(&mut z, MAX_BLOCK),
        slot: SlotMemory::new(&mut z, seq, DEFAULT_EXTRA_RUNS),
        runs: runs(3),
        taps: TapBufs::new(&mut z, DEFAULT_EXTRA_RUNS),
    }
}

impl Fixture {
    fn pass(&self, pos0: usize, n: usize, logit_rows: Option<&[usize]>) -> Pass {
        self.lab(pos0, n, logit_rows, &[], Taps::default())
    }

    /// A pass with control items and taps.
    fn lab(&self, pos0: usize, n: usize, logit_rows: Option<&[usize]>, control: &[Control], named: Taps) -> Pass {
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
                control,
                taps: named.any().then_some(TapRing {
                    named,
                    bufs: &self.taps,
                }),
            },
        )
    }

    /// True when a task writes into the buffers of the taps.
    fn writes_taps(&self, task: &MkTaskDescriptor) -> bool {
        let start = self.taps.residual;
        let end = self.taps.routes + (16 * self.taps.runs * ROUTE_WORDS * 4) as u64;
        (start..end).contains(&task.c)
    }
}

const ALL: Taps = Taps {
    residual: true,
    fold: true,
    blocks: true,
    messages: true,
    gates: true,
    routes: true,
};

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
            // Each gate reads g from the task just before it, which writes it into the router rows.
            for (i, t) in run.iter().enumerate().filter(|(_, t)| t.opcode == OP_MESSAGE_GATE) {
                let down = &run[i - 1];
                assert_eq!(down.opcode, OP_MESSAGE_GATE_DOWN, "run {r}: the task before a gate");
                assert_eq!((down.a, down.b, down.x, down.n), (t.a, t.b, t.x, t.n), "run {r}");
                assert_eq!(down.c, b.router, "run {r}: g goes in the router rows");
                assert_eq!(
                    t.p0 as u64 | (t.layer as u64) << 32,
                    b.router,
                    "run {r}: the gate reads g there"
                );
            }
            let downs = run.iter().filter(|t| t.opcode == OP_MESSAGE_GATE_DOWN).count();
            assert_eq!(downs, 4, "run {r}: one first step for each gate");
            // The router rows hold g only until the router of the layer writes them.
            let last_gate = run.iter().rposition(|t| t.opcode == OP_MESSAGE_GATE).unwrap();
            let router = run
                .iter()
                .position(|t| t.c == b.router && t.opcode == OP_MATMUL)
                .unwrap();
            assert!(
                last_gate < router,
                "run {r}: the router writes its rows after the last gate"
            );
        } else {
            assert!(message.is_none() && gates.is_empty(), "run {r}");
            assert!(run.iter().all(|t| t.opcode != OP_MESSAGE_GATE_DOWN), "run {r}");
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
            control: &[],
            taps: None,
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

/// The buffers of the taps take the bytes of the plan, in its order.
#[test]
fn the_buffers_of_the_taps_take_the_bytes_of_the_plan() {
    for extra in [24, 36, 48] {
        let mut total = 0u64;
        let t = TapBufs::new(
            &mut |bytes| {
                total += spec::rounded(bytes) as u64;
                total
            },
            extra,
        );
        assert_eq!(total, spec::tap_bytes(extra), "a reserve of {extra}");
        assert_eq!(
            (t.runs, t.cycles, t.message_rows()),
            (24 + extra, 1 + extra / 12, extra)
        );
    }
}

/// A tap only adds copies into its own buffers: with them taken out, a tapped pass is the plain pass, task for task
/// (engine-abi "The lab of Inkling 2", "Bits").
#[test]
fn a_pass_with_every_tap_is_the_plain_pass_and_copies_into_the_taps() {
    let f = fixture(32_768);
    let rows = [4usize, 1];
    let plain = f.pass(0, 5, Some(&rows));
    let tapped = f.lab(0, 5, Some(&rows), &[], ALL);
    let all = |p: &Pass| -> Vec<MkTaskDescriptor> {
        let mut out = p.head.clone();
        p.runs.iter().for_each(|r| out.extend_from_slice(r));
        out.extend_from_slice(&p.tail);
        out
    };
    let (plain, tapped) = (all(&plain), all(&tapped));
    let (copies, rest): (Vec<_>, Vec<_>) = tapped.iter().partition(|t| f.writes_taps(t));
    assert!(plain.iter().all(|t| !f.writes_taps(t)), "a plain pass writes no tap");
    let rest: Vec<MkTaskDescriptor> = rest.into_iter().copied().collect();
    let debug = |list: &[MkTaskDescriptor]| list.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>();
    assert_eq!(debug(&rest), debug(&plain), "the tapped pass with its copies taken out");
    assert!(copies.iter().all(|t| t.opcode == OP_COPY));
    // For each of 2 logit rows: 48 rows of the residual and 48 routes, 6 block outputs, and 24 layers of 4 messages
    // and 4 gates.
    assert_eq!(copies.len(), 2 * (48 + 48 + 6 + 24 * 4 + 24 * 4));
}

/// The copies of each tap read the logit row of the buffer that holds what the tap gives, at the place of the run.
#[test]
fn each_tap_copies_its_row_after_the_task_that_writes_it() {
    let f = fixture(32_768);
    let rows = [4usize, 1];
    let one = |named: Taps| f.lab(0, 5, Some(&rows), &[], named);
    let row = |at: u64, r: usize| at + (r * HIDDEN * 4) as u64;
    let residual = one(Taps {
        residual: true,
        ..Taps::default()
    });
    for (r, run) in residual.runs.iter().enumerate() {
        let n = run.len();
        // The row after the run: after its last task that writes `x`, which is the convolution after the experts,
        // or the copy of a cycle start of layer 5.
        let last_x = run
            .iter()
            .rposition(|t| t.c == f.bufs.x && t.opcode == OP_RESIDUAL_CONV)
            .unwrap();
        let copies: Vec<(u64, u64, u32)> = run[last_x + 1..]
            .iter()
            .filter(|t| f.writes_taps(t))
            .map(|t| (t.a, t.c, t.n))
            .collect();
        assert_eq!(
            copies,
            [
                (row(f.bufs.x, 4), f.taps.residual_row(0, r), HIDDEN as u32),
                (row(f.bufs.x, 1), f.taps.residual_row(1, r), HIDDEN as u32)
            ],
            "run {r} of {n} tasks"
        );
    }
    let routes = one(Taps {
        routes: true,
        ..Taps::default()
    });
    for (r, run) in routes.runs.iter().enumerate() {
        let at = run.iter().position(|t| t.opcode == OP_SIGMOID_ROUTER).unwrap();
        let words = (ROUTE_WORDS * 4) as u64;
        assert_eq!(
            (run[at + 1].a, run[at + 1].c, run[at + 1].n),
            (f.bufs.decisions + 4 * words, f.taps.route_row(0, r), ROUTE_WORDS as u32),
            "run {r}"
        );
    }
    let messages = one(Taps {
        messages: true,
        ..Taps::default()
    });
    // Run 18 is layer 6 of cycle 2, the first run with messages; run 47 is layer 23 and has none.
    for (r, g) in [(18usize, 0usize), (29, 11), (30, 12), (41, 23)] {
        let run = &messages.runs[r];
        let at = run
            .iter()
            .position(|t| t.opcode == OP_ATTN_RESIDUAL && t.a == f.bufs.previous)
            .unwrap();
        let copies: Vec<(u64, u64)> = run[at + 1..at + 9].iter().map(|t| (t.a, t.c)).collect();
        let want: Vec<(u64, u64)> = (0..4)
            .flat_map(|k| {
                let src = plane(&f.bufs, f.bufs.m, k);
                [
                    (row(src, 4), f.taps.message_row(f.taps.messages, 0, g, k)),
                    (row(src, 1), f.taps.message_row(f.taps.messages, 1, g, k)),
                ]
            })
            .collect();
        assert_eq!(copies, want, "run {r}");
    }
    assert!(
        messages.runs[..18]
            .iter()
            .chain(&messages.runs[42..])
            .flatten()
            .all(|t| !f.writes_taps(t))
    );
    let gates = one(Taps {
        gates: true,
        ..Taps::default()
    });
    for (r, g) in [(18usize, 0usize), (23, 5), (41, 23)] {
        let run = &gates.runs[r];
        let taps: Vec<(u64, u64)> = run.iter().filter(|t| f.writes_taps(t)).map(|t| (t.a, t.c)).collect();
        let mut want = Vec::new();
        for k in 0..3 {
            let src = plane(&f.bufs, f.bufs.a, k);
            want.push((row(src, 4), f.taps.message_row(f.taps.gates, 0, g, k)));
            want.push((row(src, 1), f.taps.message_row(f.taps.gates, 1, g, k)));
        }
        want.push((row(f.bufs.e, 4), f.taps.message_row(f.taps.gates, 0, g, 3)));
        want.push((row(f.bufs.e, 1), f.taps.message_row(f.taps.gates, 1, g, 3)));
        assert_eq!(taps, want, "run {r}");
        // The gates of the query, the key and the value come after the last message gate before the norms, and the
        // gate of the expert input after its own message gate.
        let last_gate = run
            .iter()
            .position(|t| t.opcode == OP_MESSAGE_GATE && t.c == f.bufs.e)
            .unwrap();
        assert_eq!(run[last_gate + 1].c, f.taps.message_row(f.taps.gates, 0, g, 3));
    }
    let blocks = one(Taps {
        blocks: true,
        ..Taps::default()
    });
    // Runs 17, 29 and 41 end cycles 1, 2 and 3, and the tail copies the other three block outputs.
    for (r, b) in [(17usize, 2usize), (29, 3), (41, 4)] {
        let taps: Vec<(u64, u64)> = blocks.runs[r]
            .iter()
            .filter(|t| f.writes_taps(t))
            .map(|t| (t.a, t.c))
            .collect();
        let src = plane(&f.bufs, f.bufs.blocks, 2);
        assert_eq!(
            taps,
            [
                (row(src, 4), f.taps.block_row(0, b)),
                (row(src, 1), f.taps.block_row(1, b))
            ],
            "run {r}"
        );
    }
    let tail: Vec<(u64, u64)> = blocks
        .tail
        .iter()
        .filter(|t| f.writes_taps(t))
        .map(|t| (t.a, t.c))
        .collect();
    let mut want = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        for (p, b) in [(0, 0), (1, 1), (3, 5)] {
            want.push((row(plane(&f.bufs, f.bufs.blocks, p), *r), f.taps.block_row(i, b)));
        }
    }
    assert_eq!(tail, want);
}

/// Each control item acts after each run of each layer that it holds, before the taps read the row and before a
/// cycle starts from it. An add of strength 0 gets no task, and a clamp of strength 0 does (engine-abi "Control
/// vectors").
#[test]
fn a_control_item_acts_after_each_run_of_its_layers() {
    let f = fixture(32_768);
    let item = |op, strength, layers: Option<Vec<(u32, f32)>>| Control {
        vector: 7 << 36,
        op,
        strength,
        layers,
    };
    let count = |p: &Pass| {
        p.runs
            .iter()
            .flatten()
            .filter(|t| t.opcode == OP_INKLING2_CONTROL)
            .count()
    };
    let tapped = Taps {
        residual: true,
        ..Taps::default()
    };
    let add = f.lab(0, 5, Some(&[4]), &[item(ControlOp::Add, 2.0, None)], tapped);
    assert_eq!(count(&add), 48);
    for (r, (run, spec)) in add.runs.iter().zip(&f.runs).enumerate() {
        let at = run.iter().position(|t| t.opcode == OP_INKLING2_CONTROL).unwrap();
        let t = &run[at];
        assert_eq!(
            (t.a, t.b, t.n, t.k, t.p0, f32::from_bits(t.m)),
            (
                f.bufs.x,
                (7 << 36) + (spec.layer * HIDDEN * 4) as u64,
                5,
                HIDDEN as u32,
                ControlOp::Add as u32,
                2.0
            ),
            "run {r}"
        );
        // After the convolution of the experts, before the tap and the copy of a cycle start.
        assert_eq!(run[at - 1].opcode, OP_RESIDUAL_CONV, "run {r}");
        assert!(f.writes_taps(&run[at + 1]), "run {r}");
        if r == 5 {
            assert_eq!((run[at + 2].a, run[at + 2].c), (f.bufs.x, f.bufs.start));
        }
    }
    let plain = f.pass(0, 5, Some(&[4]));
    for zero in [ControlOp::Add, ControlOp::Ablate] {
        let runs = |p: &Pass| format!("{:?}", p.runs);
        assert_eq!(
            runs(&f.lab(0, 5, Some(&[4]), &[item(zero, 0.0, None)], Taps::default())),
            runs(&plain)
        );
    }
    assert_eq!(
        count(&f.lab(0, 5, Some(&[4]), &[item(ControlOp::Clamp, 0.0, None)], Taps::default())),
        48
    );
    // Layer 8 runs in each of the 3 cycles; layer 20 once. Layer 2 at strength 0 adds nothing.
    let layered = f.lab(
        0,
        5,
        Some(&[4]),
        &[item(ControlOp::Ablate, 1.0, Some(vec![(2, 0.0), (8, 3.0), (20, -1.0)]))],
        Taps::default(),
    );
    let acts: Vec<(usize, f32)> = layered
        .runs
        .iter()
        .zip(&f.runs)
        .flat_map(|(run, spec)| {
            run.iter()
                .filter(|t| t.opcode == OP_INKLING2_CONTROL)
                .map(move |t| (spec.layer, f32::from_bits(t.m)))
        })
        .collect();
    assert_eq!(acts, [(8, 3.0), (8, 3.0), (8, 3.0), (20, -1.0)]);
}

/// Each matmul of a pass takes the tiled path, so a row of one token sums in the order of 2 to 8 rows (milestone 36
/// item 10). Then a decode row gets the same bits in a batch of 1 to 8 rows.
#[test]
fn each_matmul_of_a_pass_takes_the_tiled_path() {
    let f = fixture(32_768);
    for n in [1, 8, 512] {
        let p = f.pass(0, n, Some(&[0]));
        let all: Vec<&MkTaskDescriptor> = p.head.iter().chain(p.runs.iter().flatten()).chain(&p.tail).collect();
        let matmuls: Vec<&&MkTaskDescriptor> = all.iter().filter(|t| t.opcode == OP_MATMUL).collect();
        assert_eq!(
            matmuls.len(),
            48 * 8 + 1,
            "a block of {n}: 8 matmuls in each run, and one head"
        );
        for t in matmuls {
            assert_ne!(t.flags & FLAG_TILED, 0, "a block of {n}: a matmul of {} rows", t.m);
        }
    }
}

/// Slots for a batch, at addresses that do not overlap the fixture: one of 32,768 positions with a ring that wraps,
/// one short slot, and another long one.
fn batch_slots() -> Vec<SlotMemory> {
    let mut next = 1u64 << 46;
    let mut z = move |bytes: usize| {
        let at = next;
        next += spec::rounded(bytes) as u64;
        at
    };
    [32_768, 2_048, 32_768]
        .into_iter()
        .map(|seq| SlotMemory::new(&mut z, seq, DEFAULT_EXTRA_RUNS))
        .collect()
}

/// The ops that a context runs on its own slot: the convolutions, the clips, the attention and the copies of a
/// ring's spill.
fn slot_op(t: &MkTaskDescriptor, slot: &SlotMemory) -> bool {
    t.opcode == OP_INKLING2_ATTN_SUM || [t.a, t.b, t.c, t.x].into_iter().any(|at| inside(slot, at))
}

/// True when `at` is in the memory of `slot`. The counter gives a slot one span: the plane and the conv part of each
/// state in turn.
fn inside(slot: &SlotMemory, at: u64) -> bool {
    (slot.planes[0]..*slot.conv.last().unwrap() + spec::CONV_BYTES).contains(&at)
}

/// A clip with the log scale of a full layer, which reads the position of its row.
fn log_clip(t: &MkTaskDescriptor) -> bool {
    t.opcode == OP_HEAD_CLIP && t.layer & 256 != 0
}

/// What a task of a slot op does, without the rows of the buffers that it reads: its op, its sizes, its positions and
/// the addresses that it reads or writes in the slot.
fn slot_shape(t: &MkTaskDescriptor, slot: &SlotMemory) -> (u16, u32, u32, u32, u32, u32, u16, [u64; 4]) {
    let keep = |at: u64| if inside(slot, at) { at } else { 0 };
    (
        t.opcode,
        t.n,
        t.m,
        t.k,
        t.p0,
        t.layer,
        t.flags,
        [t.a, t.b, t.c, t.x].map(keep),
    )
}

fn tasks_of(p: &Pass) -> Vec<MkTaskDescriptor> {
    p.head
        .iter()
        .chain(p.runs.iter().flatten())
        .chain(&p.tail)
        .copied()
        .collect()
}

/// A batch of one row is the pass of one token alone, task for task (milestone 36 item 11).
#[test]
fn a_batch_of_one_row_is_the_pass_of_its_token_alone() {
    let f = fixture(32_768);
    for pos in [0, 1_024, 5_000] {
        let alone = f.pass(pos, 1, Some(&[0]));
        let rows = [(&f.slot, pos)];
        let one = batch(
            &f.w,
            &f.bufs,
            Batch {
                rows: &rows,
                runs: &f.runs,
                control: &[],
            },
        );
        assert_eq!(
            format!("{:?}", tasks_of(&one)),
            format!("{:?}", tasks_of(&alone)),
            "a row at {pos}"
        );
    }
}

/// Each row of a batch runs the slot ops of its pass alone on its own slot, at its own position. The other ops run
/// once over all the rows, with the weights of the pass alone (engine-abi "A batch of decode rows").
#[test]
fn each_row_of_a_batch_runs_the_slot_ops_of_its_pass_alone() {
    let f = fixture(32_768);
    let slots = batch_slots();
    let rows: Vec<(&SlotMemory, usize)> = slots.iter().zip([5_000usize, 1_500, 0]).collect();
    let got = tasks_of(&batch(
        &f.w,
        &f.bufs,
        Batch {
            rows: &rows,
            runs: &f.runs,
            control: &[],
        },
    ));
    let k = rows.len();
    let mut shared = Vec::new();
    for (i, (slot, pos)) in rows.iter().enumerate() {
        let alone = tasks_of(&pass(
            &f.w,
            &f.bufs,
            TaskRing {
                slot,
                pos0: *pos,
                n: 1,
                runs: &f.runs,
                upto: f.runs.len(),
                logit_rows: Some(&[0]),
                after_embed: &[],
                control: &[],
                taps: None,
            },
        ));
        let want: Vec<_> = alone
            .iter()
            .filter(|t| slot_op(t, slot))
            .map(|t| slot_shape(t, slot))
            .collect();
        // The clips with the log scale of row i read its queries and its position, as the pass alone does.
        let clips = |ts: &[MkTaskDescriptor], row: usize| -> Vec<(u32, u32, u64)> {
            ts.iter()
                .filter(|t| {
                    log_clip(t)
                        && (t.a == f.bufs.q + (row * HEADS * HEAD_DIM * 4) as u64
                            || t.a == f.bufs.rel + (row * HEADS * D_REL * 4) as u64)
                })
                .map(|t| (t.n, t.p0, t.b))
                .collect()
        };
        assert_eq!(clips(&got, i), clips(&alone, 0), "the log clips of row {i}");
        assert_eq!(
            clips(&alone, 0).len(),
            2 * 8,
            "two log clips in each of the 8 full runs"
        );
        // Each other row's slot ops name its own slot, so the filter of this slot finds only the ops of row i.
        let mine: Vec<&MkTaskDescriptor> = got
            .iter()
            .filter(|t| {
                if t.opcode == OP_INKLING2_ATTN_SUM {
                    t.c == f.bufs.attn + (i * HEADS * HEAD_DIM * 4) as u64
                } else {
                    slot_op(t, slot)
                }
            })
            .collect();
        let have: Vec<_> = mine.iter().map(|t| slot_shape(t, slot)).collect();
        assert_eq!(have.len(), want.len(), "row {i} at {pos}");
        assert_eq!(have, want, "row {i} at {pos}");
        // The slot ops of the row read and write row i of the buffers.
        for t in mine.iter().filter(|t| t.opcode == OP_INKLING2_ATTN) {
            assert_eq!(
                t.a,
                f.bufs.q + (i * HEADS * HEAD_DIM * 4) as u64,
                "the queries of row {i}"
            );
            assert_eq!(
                t.c,
                f.bufs.rel + (i * HEADS * D_REL * 4) as u64,
                "the bias states of row {i}"
            );
        }
        // The convolutions and the clips read and write row i of the buffers: the row of the pass alone, i rows on.
        let alone_ops: Vec<&MkTaskDescriptor> = alone.iter().filter(|t| slot_op(t, slot)).collect();
        for (g, a) in mine.iter().zip(&alone_ops) {
            let (input, output) = match g.opcode {
                OP_RESIDUAL_CONV => (g.m & 0xFFFF, g.k),
                OP_HEAD_CLIP => (g.m * g.k, g.m * g.k),
                _ => continue,
            };
            let moved =
                |have: u64, want: u64, width: u32| inside(slot, want) || have == want + (i * width as usize * 4) as u64;
            assert!(
                moved(g.a, a.a, input) && moved(g.c, a.c, output),
                "op {} of row {i} reads or writes another row",
                g.opcode
            );
        }
        for t in mine.iter().filter(|t| t.opcode == OP_INKLING2_ATTN_SUM) {
            assert_eq!(t.a, f.bufs.sinks + (i * HEADS * 4) as u64, "the sinks of row {i}");
        }
        if i == 0 {
            shared = alone
                .into_iter()
                .filter(|t| !slot_op(t, slot) && !log_clip(t))
                .collect();
        }
    }
    // The ops of the weights run once over the k rows, in the order of the pass alone. The pass alone has one head;
    // the batch has one for each row.
    let ops: Vec<&MkTaskDescriptor> = got
        .iter()
        .filter(|t| !log_clip(t) && !rows.iter().any(|(slot, _)| slot_op(t, slot)))
        .collect();
    let alone_heads = shared
        .iter()
        .filter(|t| t.opcode == OP_MATMUL && t.m == HEAD_ROWS as u32)
        .count();
    assert_eq!(alone_heads, 1);
    let heads: Vec<&&MkTaskDescriptor> = ops
        .iter()
        .filter(|t| t.opcode == OP_MATMUL && t.m == HEAD_ROWS as u32)
        .collect();
    assert_eq!(heads.len(), k, "one head for each row");
    for (i, t) in heads.iter().enumerate() {
        assert_eq!(
            (t.n, t.a),
            (1, f.bufs.e_norm + (i * HIDDEN * 4) as u64),
            "the head of row {i}"
        );
        assert_ne!(t.flags & FLAG_TILED, 0);
    }
    let body: Vec<&&MkTaskDescriptor> = ops
        .iter()
        .filter(|t| !(t.opcode == OP_MATMUL && t.m == HEAD_ROWS as u32))
        .collect();
    let alone_body: Vec<&MkTaskDescriptor> = shared
        .iter()
        .filter(|t| !(t.opcode == OP_MATMUL && t.m == HEAD_ROWS as u32))
        .collect();
    assert_eq!(body.len(), alone_body.len());
    for (b, a) in body.iter().zip(&alone_body) {
        assert_eq!((b.opcode, b.b, b.flags), (a.opcode, a.b, a.flags));
        if b.opcode == OP_MATMUL {
            assert_eq!(
                (b.n, b.m, b.k),
                (k as u32, a.m, a.k),
                "a matmul runs once over the rows"
            );
        }
    }
}
