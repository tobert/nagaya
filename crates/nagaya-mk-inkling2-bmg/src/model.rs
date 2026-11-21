//! The ring of one pass of the text model, and the device buffers that it uses.
//!
//! The ring follows `host::forward` and engine-abi "The ring of one run" for one block of `n` tokens at the positions
//! `pos0..pos0 + n` of one slot: the normed embedding, then each run of `spec::runs`, then the output residual, the
//! final norm and the head with the softcap of 30 for each logit row. The rows of the images of the block replace the
//! normed embedding rows of their placeholders, with no norm after them. Each run of a cycle after the first reads the
//! messages of the cycle before it.
//!
//! The ring is a list of runs, so that a pass whose tasks do not fit one launch runs as several, cut between two runs
//! (engine-abi "A pass in several launches"). The state of the block stays in the buffers between two launches.

use nagaya_mk_bmg::protocol::{FLAG_SOFTCAP_30, MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::tasks::{self, ConvTaps, HeadClip, Inkling2Attention};

use crate::host::ops::{D_REL, EPS, EXPERTS, HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, QK_CLIP, REL_CLIP};
use crate::host::weights::{BLOCK, CYCLE_START, LAYERS, is_full};
use crate::spec::{
    self, CONV_ROW, HEAD_ROWS, KV_ROW, MAX_LOGIT_ROWS, RING_ROWS, Run, SINK_ROWS, plane_rows, state_layer,
};
use crate::weights::{Layer, Weights};

/// The width of a query row and of an attention row: 24 heads of 128.
const Q_ROW: usize = HEADS * HEAD_DIM;
/// The width of the input row of a key or value convolution on a full layer: 1,024 values, then the taps of 8 heads.
const DYNAMIC_ROW: usize = KV_ROW + KV_HEADS * 4;
/// The columns of a conv part: the key, the value, the attention output and the expert output.
const CONV_KEY: usize = 0;
const CONV_VALUE: usize = KV_ROW;
const CONV_ATTN: usize = 2 * KV_ROW;
const CONV_MLP: usize = 2 * KV_ROW + HIDDEN;
/// The values of one row of a sink copy of `OP_EMBED`: 8 of the 32 sink values of a position.
const SINK_COPY: usize = 8;

const _: () = assert!(CONV_MLP + HIDDEN == CONV_ROW && HEADS == 3 * SINK_COPY && SINK_ROWS == 4 * SINK_COPY);

/// The most tasks of one launch: a ring ends with `OP_HALT`.
pub const LAUNCH_TASKS: usize = MAX_TASKS_PER_CARD - 1;

/// The activation buffers of a block of at most `max_block` tokens, all `f32` but the ids. A plane holds `max_block`
/// rows of one kind; `[k]` planes follow each other.
#[derive(Clone, Copy, Debug)]
pub struct Bufs {
    pub max_block: usize,
    /// The ids, as `u32`.
    pub ids: u64,
    /// The row ids of the sink copy: row `i` of 8 values takes row `4 (i / 3) + i % 3` of the rows of 32.
    pub sink_ids: u64,
    /// The residual `x`.
    pub x: u64,
    /// The block outputs `B`: 4 planes.
    pub blocks: u64,
    /// The block output of the cycle before, `P`: row 2 of `B` at the end of that cycle.
    pub previous: u64,
    /// The last residual of the cycle before, `y`.
    pub y: u64,
    /// The residual after layer 5, from which each cycle starts.
    pub start: u64,
    /// The attention residuals `a_q`, `a_k`, `a_v`: 3 planes.
    pub a: u64,
    /// The 4 messages: 4 planes.
    pub m: u64,
    /// The normed inputs `h_q`, `h_k`, `h_v`: 3 planes.
    pub h: u64,
    pub q: u64,
    /// The relative states: `[n][24][16]`.
    pub rel: u64,
    /// The sinks as the matmul gives them, `[n][32]`, and as the attention reads them, `[n][24]`.
    pub sinks32: u64,
    pub sinks: u64,
    /// The output gate before its sigmoid: `[n][3072]`.
    pub gate: u64,
    /// The inputs of the key and value convolutions: `[n][1056]` with the taps of a full layer.
    pub k_in: u64,
    pub v_in: u64,
    /// The keys and values after their convolutions, before the clip and the norm: `[n][1024]`.
    pub k_conv: u64,
    pub v_conv: u64,
    /// The attention after its output gate: `[n][3072]`.
    pub attn: u64,
    /// The output projection: `[n][2304]`.
    pub o: u64,
    /// The expert input `e`, and the output residual after the last run.
    pub e: u64,
    /// The norm of `e`, and the final norm.
    pub e_norm: u64,
    pub router: u64,
    pub decisions: u64,
    pub plan: u64,
    /// The rows of the experts in plan order: `[8 n][2304]`, `[8 n][1152]`, and the slot rows `[8 n][2304]`.
    pub gathered: u64,
    pub act: u64,
    pub slots: u64,
    /// The sum of the 8 picks: `[n][2304]`.
    pub moe: u64,
    /// The logits of each logit row: `[16][200064]`.
    pub logits: u64,
}

impl Bufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count. The names follow
    /// `spec::block_buffers`, in its order.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, max_block: usize) -> Self {
        let mut list = spec::block_buffers(max_block).into_iter();
        let mut next = |name: &str| {
            let (have, floats) = list.next().expect("spec::block_buffers names each buffer");
            assert_eq!(have, name, "spec::block_buffers names the buffers in the order of Bufs");
            z(floats * 4)
        };
        Self {
            max_block,
            ids: next("ids"),
            sink_ids: next("sink ids"),
            x: next("x"),
            blocks: next("B"),
            previous: next("P"),
            y: next("y"),
            start: next("x after layer 5"),
            a: next("a_q, a_k, a_v"),
            m: next("m"),
            h: next("h_q, h_k, h_v"),
            q: next("query"),
            rel: next("bias states"),
            sinks32: next("sinks of the matmul"),
            sinks: next("sinks"),
            gate: next("output gate"),
            k_in: next("key input"),
            v_in: next("value input"),
            k_conv: next("key"),
            v_conv: next("value"),
            attn: next("attention"),
            o: next("o"),
            e: next("e"),
            e_norm: next("norm of e"),
            router: next("router"),
            decisions: next("decisions"),
            plan: next("plan"),
            gathered: next("gathered rows"),
            act: next("activations"),
            slots: next("down rows"),
            moe: next("f"),
            logits: next("logits"),
        }
    }

    /// Row `t` of plane `p` of a buffer of planes of `width` values.
    fn plane(&self, at: u64, p: usize, width: usize) -> u64 {
        at + (p * self.max_block * width * 4) as u64
    }

    /// The ids of the sink copy for a block of `max_block` positions.
    pub fn sink_ids_of(max_block: usize) -> Vec<u32> {
        (0..3 * max_block).map(|i| (4 * (i / 3) + i % 3) as u32).collect()
    }
}

/// The device memory of one slot: the planes and the conv part of each state. A key plane comes first, and its value
/// plane follows it.
#[derive(Clone, Debug)]
pub struct SlotMemory {
    /// The positions of the slot.
    pub seq: usize,
    /// The planes of each state.
    pub planes: Vec<u64>,
    /// The conv part of each state: 3 rows of 6,656 values, oldest first.
    pub conv: Vec<u64>,
}

impl SlotMemory {
    /// The memory of a slot of `seq` positions with `extra` entries of the reserve, from `z`.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, seq: usize, extra: usize) -> Self {
        let mut planes = Vec::new();
        let mut conv = Vec::new();
        for s in 0..LAYERS + extra {
            planes.push(z(2 * plane_rows(seq, s) * KV_ROW * 4));
            conv.push(z(spec::CONV_BYTES as usize));
        }
        Self { seq, planes, conv }
    }

    /// The entries of the reserve.
    pub fn extra(&self) -> usize {
        self.planes.len() - LAYERS
    }

    /// The ring rows of state `s`, or 0 for cells. Every sliding state holds a ring on the card (engine-abi "The host
    /// spec", "Slot sizes").
    pub fn ring(&self, s: usize) -> usize {
        if is_full(state_layer(s)) { 0 } else { RING_ROWS }
    }

    /// The device row of position `pos` in state `s`.
    fn row_of(&self, s: usize, pos: usize) -> usize {
        match self.ring(s) {
            0 => pos,
            ring => pos % ring,
        }
    }

    /// The address of the key row (`value` false) or the value row of position `pos` of state `s`.
    pub fn row(&self, s: usize, value: bool, pos: usize) -> u64 {
        let plane = if value { plane_rows(self.seq, s) } else { 0 };
        self.planes[s] + ((plane + self.row_of(s, pos)) * KV_ROW * 4) as u64
    }

    /// The runs of device rows that hold positions `rows` of state `s`: one run, or two where a ring wraps. Each run is
    /// the address of its first row and its count of rows.
    pub fn runs(&self, s: usize, value: bool, rows: std::ops::Range<usize>) -> Vec<(u64, usize)> {
        let ring = self.ring(s);
        let n = rows.len();
        if ring > 0 && rows.start % ring + n > ring {
            let head = ring - rows.start % ring;
            vec![
                (self.row(s, value, rows.start), head),
                (self.row(s, value, rows.start + head), n - head),
            ]
        } else {
            vec![(self.row(s, value, rows.start), n)]
        }
    }
}

/// What one ring runs: a block of `n` tokens at `pos0` of one slot, and how far it goes.
#[derive(Clone, Copy, Debug)]
pub struct TaskRing<'a> {
    pub slot: &'a SlotMemory,
    pub pos0: usize,
    pub n: usize,
    /// The runs of the pass, in order (`spec::runs`).
    pub runs: &'a [Run],
    /// The runs that the ring runs, from the first. A probe of the oracle stops early.
    pub upto: usize,
    /// The block rows whose logits the ring gives. With `None`, the ring stops after the runs.
    pub logit_rows: Option<&'a [usize]>,
    /// The tasks after the norm of the embedding and before the first run: the copies of the image rows over the rows
    /// of their placeholders (engine-abi "The choices of the pass").
    pub after_embed: &'a [MkTaskDescriptor],
}

/// The tasks of a pass: the tasks before the first run, the tasks of each run, and the tasks after the last run.
#[derive(Clone, Debug, Default)]
pub struct Pass {
    pub head: Vec<MkTaskDescriptor>,
    pub runs: Vec<Vec<MkTaskDescriptor>>,
    pub tail: Vec<MkTaskDescriptor>,
}

impl Pass {
    /// The launches of the pass: whole runs, at most `limit` tasks each. The head goes with the first run, and the tail
    /// with the last. A run longer than `limit` is a defect of the builder.
    pub fn launches(&self, limit: usize) -> Vec<Vec<MkTaskDescriptor>> {
        let mut out: Vec<Vec<MkTaskDescriptor>> = vec![self.head.clone()];
        for run in &self.runs {
            assert!(
                run.len() <= limit,
                "a run of {} tasks does not fit a launch of {limit}",
                run.len()
            );
            if out.last().unwrap().len() + run.len() > limit {
                out.push(Vec::new());
            }
            out.last_mut().unwrap().extend_from_slice(run);
        }
        if out.last().unwrap().len() + self.tail.len() > limit {
            out.push(Vec::new());
        }
        out.last_mut().unwrap().extend_from_slice(&self.tail);
        out.retain(|launch| !launch.is_empty());
        out
    }

    /// The count of tasks.
    pub fn len(&self) -> usize {
        self.head.len() + self.runs.iter().map(Vec::len).sum::<usize>() + self.tail.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The tasks of one pass. The ids are in `bufs.ids`. After the pass, `bufs.x` holds the residual after the runs that
/// ran, `bufs.e_norm` the final norm, and row `i` of `bufs.logits` the logits of logit row `i` after the softcap.
pub fn pass(w: &Weights, bufs: &Bufs, ring: TaskRing<'_>) -> Pass {
    let TaskRing {
        slot,
        pos0,
        n,
        runs,
        upto,
        logit_rows,
        after_embed,
    } = ring;
    assert!(
        upto <= runs.len() && (upto == runs.len() || logit_rows.is_none()),
        "a ring gives logits only after every run"
    );
    assert!(
        n >= 1 && n <= bufs.max_block && pos0 + n <= slot.seq,
        "positions {pos0} to {} do not fit a block of {} in a slot of {}; the engine checks the pass first",
        pos0 + n,
        bufs.max_block,
        slot.seq
    );
    let cycles = runs.iter().filter_map(|r| r.cycle).max().unwrap_or(1);
    let (n32, h32) = (n as u32, HIDDEN as u32);
    let mut embed = vec![
        tasks::embed(bufs.ids, w.embed, bufs.o, n32, h32, 0, false, false),
        tasks::rmsnorm(bufs.o, w.embed_norm, bufs.x, n32, h32, EPS, false),
    ];
    embed.extend_from_slice(after_embed);
    let mut pass = Pass {
        head: embed,
        ..Pass::default()
    };
    for run in runs.iter().take(upto) {
        pass.runs.push(run_tasks(w, bufs, slot, pos0, n, *run, cycles));
    }
    if let Some(rows) = logit_rows {
        assert!(
            rows.len() <= MAX_LOGIT_ROWS,
            "a pass gives at most {MAX_LOGIT_ROWS} logit rows"
        );
        let t = &mut pass.tail;
        // The output residual over the 4 block outputs and the last row, then the final norm.
        t.push(tasks::attn_residual(
            bufs.blocks,
            4,
            bufs.x,
            w.output_query,
            1,
            bufs.e,
            n32,
            h32,
            bufs.max_block as u32,
        ));
        t.push(tasks::rmsnorm(bufs.e, w.norm, bufs.e_norm, n32, h32, EPS, false));
        // Each logit row is its own matmul of one row, so its bits do not depend on the other logit rows.
        for (i, row) in rows.iter().enumerate() {
            assert!(*row < n, "logit row {row} is outside the block of {n}");
            let mut head = tasks::matmul(
                bufs.e_norm + (row * HIDDEN * 4) as u64,
                w.head,
                bufs.logits + (i * HEAD_ROWS * 4) as u64,
                1,
                HEAD_ROWS as u32,
                h32,
                false,
            );
            head.flags |= FLAG_SOFTCAP_30;
            t.push(head);
        }
    }
    pass
}

/// The tasks of one run of one layer (engine-abi "The ring of one run").
fn run_tasks(
    w: &Weights,
    bufs: &Bufs,
    slot: &SlotMemory,
    pos0: usize,
    n: usize,
    run: Run,
    cycles: usize,
) -> Vec<MkTaskDescriptor> {
    let Run {
        layer: l,
        cycle,
        state: s,
    } = run;
    let lw: &Layer = &w.layers[l];
    let full = is_full(l);
    let messages = cycle.is_some_and(|c| c >= 2);
    let (n32, h32, mb) = (n as u32, HIDDEN as u32, bufs.max_block as u32);
    let hidden_plane = |at: u64, p: usize| bufs.plane(at, p, HIDDEN);
    let words = |rows: usize, width: usize| (rows * width) as u32;
    let block = l / BLOCK;
    let first_of_block = l.is_multiple_of(BLOCK);
    let mut t = Vec::new();

    // A cycle after the first starts from the row after layer 5, and reads the messages of the cycle before it.
    if messages && l == CYCLE_START {
        t.push(tasks::copy(
            hidden_plane(bufs.blocks, 2),
            bufs.previous,
            words(n, HIDDEN),
        ));
        t.push(tasks::copy(bufs.x, bufs.y, words(n, HIDDEN)));
        t.push(tasks::copy(bufs.start, bufs.x, words(n, HIDDEN)));
    }

    // 1. The attention residual of the query, key and value inputs. Layer 0 reads the embedding row.
    let queries = if full { 3 } else { 1 };
    let mut inputs = [bufs.x; 3];
    if l > 0 {
        let blocks = if first_of_block { block } else { block + 1 };
        t.push(tasks::attn_residual(
            bufs.blocks,
            blocks as u32,
            bufs.x,
            lw.attn_queries,
            queries,
            bufs.a,
            n32,
            h32,
            mb,
        ));
        inputs = if full {
            [0, 1, 2].map(|p| hidden_plane(bufs.a, p))
        } else {
            [hidden_plane(bufs.a, 0); 3]
        };
    }
    // 2. The messages, and the gates of the query, key and value inputs.
    if messages {
        let li = l - CYCLE_START;
        let queries = w.message_queries + (li * 4 * HIDDEN * 4) as u64;
        t.push(tasks::attn_residual(
            bufs.previous,
            1,
            bufs.y,
            queries,
            4,
            bufs.m,
            n32,
            h32,
            mb,
        ));
        let a = [0, 1, 2].map(|p| hidden_plane(bufs.a, p));
        let m = |i: usize| hidden_plane(bufs.m, i);
        if full {
            for i in 0..3 {
                t.push(tasks::message_gate(a[i], m(i), lw.gates[i], a[i], n32, h32));
            }
        } else {
            // A sliding layer has one residual: the key and the value gate it into planes 1 and 2 first.
            t.push(tasks::message_gate(a[0], m(1), lw.gates[1], a[1], n32, h32));
            t.push(tasks::message_gate(a[0], m(2), lw.gates[2], a[2], n32, h32));
            t.push(tasks::message_gate(a[0], m(0), lw.gates[0], a[0], n32, h32));
        }
        inputs = a;
    }
    // 3. The norms.
    let h = [0, 1, 2].map(|p| hidden_plane(bufs.h, p));
    let normed = if inputs[0] == inputs[1] && inputs[1] == inputs[2] {
        t.push(tasks::rmsnorm(inputs[0], lw.norms[0], h[0], n32, h32, EPS, false));
        [h[0]; 3]
    } else {
        for i in 0..3 {
            t.push(tasks::rmsnorm(inputs[i], lw.norms[i], h[i], n32, h32, EPS, false));
        }
        h
    };
    // 4. The projections. The sinks come out of rows of 32 into rows of 24.
    let kv_in = if full { DYNAMIC_ROW } else { KV_ROW } as u32;
    let q32 = Q_ROW as u32;
    t.push(tasks::matmul(normed[0], lw.wq, bufs.q, n32, q32, h32, false));
    t.push(tasks::matmul(
        normed[0],
        lw.wr,
        bufs.rel,
        n32,
        (HEADS * D_REL) as u32,
        h32,
        false,
    ));
    t.push(tasks::matmul(
        normed[0],
        lw.sink,
        bufs.sinks32,
        n32,
        SINK_ROWS as u32,
        h32,
        false,
    ));
    t.push(tasks::embed(
        bufs.sink_ids,
        bufs.sinks32,
        bufs.sinks,
        3 * n32,
        SINK_COPY as u32,
        0,
        true,
        false,
    ));
    t.push(tasks::matmul(normed[0], lw.gate_attn, bufs.gate, n32, q32, h32, false));
    t.push(tasks::matmul(normed[1], lw.wk, bufs.k_in, n32, kv_in, h32, false));
    t.push(tasks::matmul(normed[2], lw.wv, bufs.v_in, n32, kv_in, h32, false));
    // 5. The clips with the log scale of a full layer, the convolutions, and the keys and values into the slot.
    let log_p0 = full.then_some(pos0 as u32);
    t.push(tasks::head_clip(
        bufs.q,
        bufs.q,
        n32,
        HEADS as u32,
        HEAD_DIM as u32,
        HeadClip::L2 {
            scale: lw.q_scale,
            bound: QK_CLIP,
        },
        log_p0,
    ));
    t.push(tasks::head_clip(
        bufs.rel,
        bufs.rel,
        n32,
        HEADS as u32,
        D_REL as u32,
        HeadClip::L2 {
            scale: lw.rel_scale,
            bound: REL_CLIP,
        },
        log_p0,
    ));
    let taps = |at: u64| {
        if full {
            ConvTaps::Dynamic(at)
        } else {
            ConvTaps::Fixed(at)
        }
    };
    let conv = slot.conv[s];
    let column = |c: usize| conv + (c * 4) as u64;
    let (kv32, row) = (KV_ROW as u32, CONV_ROW as u32);
    t.push(tasks::residual_conv(
        bufs.k_in,
        kv_in,
        taps(lw.k_conv),
        column(CONV_KEY),
        row,
        bufs.k_conv,
        n32,
        kv32,
        pos0 as u32,
        false,
    ));
    t.push(tasks::head_clip(
        bufs.k_conv,
        slot.row(s, false, pos0),
        n32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        HeadClip::L2 {
            scale: 0,
            bound: QK_CLIP,
        },
        None,
    ));
    t.push(tasks::residual_conv(
        bufs.v_in,
        kv_in,
        taps(lw.v_conv),
        column(CONV_VALUE),
        row,
        bufs.v_conv,
        n32,
        kv32,
        pos0 as u32,
        false,
    ));
    t.push(tasks::head_clip(
        bufs.v_conv,
        slot.row(s, true, pos0),
        n32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        HeadClip::Rms,
        None,
    ));
    // The rows past the end of a ring move to its start, bit for bit, before the attention reads them.
    let ring = slot.ring(s);
    if ring > 0 && pos0 % ring + n > ring {
        let spill = words(pos0 % ring + n - ring, KV_ROW);
        for value in [false, true] {
            let start = slot.row(s, value, 0);
            let past = start + (ring * KV_ROW * 4) as u64;
            t.push(tasks::copy(past, start, spill));
        }
    }
    // 6. The attention, the output gate, the output projection and the convolution after the attention.
    let attention = Inkling2Attention {
        q: bufs.q,
        kv: slot.planes[s],
        rel: bufs.rel,
        proj: lw.rel_proj,
        sinks: bufs.sinks,
        out: bufs.attn,
        n: n32,
        p0: pos0 as u32,
        max_seq: slot.seq as u32,
        ring: ring as u32,
        chunk_keys: tasks::inkling2_chunk_keys(ring as u32),
    };
    t.extend(attention.tasks());
    t.push(tasks::output_gate(bufs.attn, bufs.gate, bufs.attn, n32, q32));
    t.push(tasks::matmul(bufs.attn, lw.wo, bufs.o, n32, h32, q32, false));
    // 7. On the first layer of a block, `x` is a new block output and `r = o`. Else `r = x + o`.
    if first_of_block {
        t.push(tasks::copy(bufs.x, hidden_plane(bufs.blocks, block), words(n, HIDDEN)));
    }
    t.push(tasks::residual_conv(
        bufs.o,
        h32,
        ConvTaps::Fixed(lw.attn_conv),
        column(CONV_ATTN),
        row,
        bufs.x,
        n32,
        h32,
        pos0 as u32,
        !first_of_block,
    ));
    // 8. The expert input, its gate, the norm, the router, the experts and the convolution after them: `x = r + f`.
    t.push(tasks::attn_residual(
        bufs.blocks,
        block as u32 + 1,
        bufs.x,
        lw.mlp_query,
        1,
        bufs.e,
        n32,
        h32,
        mb,
    ));
    if messages {
        t.push(tasks::message_gate(
            bufs.e,
            hidden_plane(bufs.m, 3),
            lw.gates[3],
            bufs.e,
            n32,
            h32,
        ));
    }
    t.push(tasks::rmsnorm(bufs.e, lw.mlp_norm, bufs.e_norm, n32, h32, EPS, false));
    t.push(tasks::matmul(
        bufs.e_norm,
        lw.router,
        bufs.router,
        n32,
        EXPERTS as u32,
        h32,
        false,
    ));
    t.push(tasks::sigmoid_router(
        bufs.router,
        lw.router_bias,
        bufs.decisions,
        n32,
        l as u32,
    ));
    t.push(tasks::moe_plan(bufs.decisions, bufs.plan, n32));
    t.push(tasks::moe_gather(bufs.e_norm, bufs.gathered, bufs.plan, n32, h32));
    let i32_ = INTERMEDIATE as u32;
    t.push(tasks::expert_up(
        bufs.gathered,
        lw.w13,
        bufs.act,
        bufs.plan,
        i32_,
        h32,
        lw.mxfp8,
    ));
    t.push(tasks::expert_down(
        bufs.act, lw.w2, bufs.slots, bufs.plan, h32, i32_, lw.mxfp8,
    ));
    t.push(tasks::expert_sum(bufs.slots, bufs.moe, n32, h32));
    t.push(tasks::residual_conv(
        bufs.moe,
        h32,
        ConvTaps::Fixed(lw.mlp_conv),
        column(CONV_MLP),
        row,
        bufs.x,
        n32,
        h32,
        pos0 as u32,
        true,
    ));
    // Each cycle starts again from the row after layer 5.
    if l == CYCLE_START - 1 && cycles > 1 {
        t.push(tasks::copy(bufs.x, bufs.start, words(n, HIDDEN)));
    }
    t
}

#[cfg(test)]
mod tests;
