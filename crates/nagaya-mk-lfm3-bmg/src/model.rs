//! The ring of one pass of the backbone, and the device buffers that it uses.
//!
//! The ring follows `Lfm3Model.forward` of `modeling_lfm3.py` (LFM3-Omni-2B at `21d75123`) for one block of `n` tokens
//! at the positions `pos0..pos0 + n` of one slot:
//!
//! - the embedding rows, with no scale, and the rows of the media encoders over the rows of their placeholders;
//! - for each layer: `operator_norm`, the conv or the attention, `post_operator_norm` added to the residual, then
//!   `ffn_norm`, the SwiGLU MLP and `post_ffn_norm` added to the residual (modeling_lfm3.py:472-498);
//! - `embedding_norm`, and the head, which is the embedding table, for each logit row.
//!
//! An attention layer writes its keys and values into the slot, normalizes and turns them in place, attends, and
//! gates each head with `sigmoid(gate_proj(h))` (modeling_lfm3.py:245-272). A conv layer reads and writes the conv
//! state of the slot (`OP_SHORT_CONV`).

use nagaya_mk_bmg::protocol::{ControlOp, FLAG_ACCUMULATE, FLAG_TILED, MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::tasks;

use crate::spec::{Backbone, HEAD_DIM, LayerKind, MAX_BLOCK, MAX_LOGIT_ROWS, plane_rows, ring_of};
use crate::weights::{Weights, rounded};

/// The heads whose chunks the one-token attention keeps in the scratch of the ring (`attn_pv_segments`).
const SCRATCH_HEADS: usize = 16;

/// The floats of the scratch of the ring for slots of up to `max_seq` positions. One-token attention keeps the chunks
/// of 32 keys, each a max, a sum and an output of up to 512 values (`ATTN_CHUNK`, megakernel.cl). The split attention
/// of a block keeps its partial sums (`tasks::attn_pv_scratch_floats`).
pub fn scratch_floats(max_seq: usize) -> usize {
    let one_token = SCRATCH_HEADS * (max_seq / 32 + 1) * (512 + 2);
    one_token.max(tasks::attn_pv_scratch_floats(MAX_BLOCK, SCRATCH_HEADS, HEAD_DIM))
}

/// The activation buffers of a block of at most `max_block` tokens. Each row is one token.
#[derive(Clone, Copy, Debug)]
pub struct Bufs {
    /// The ids, as `u32`.
    pub ids: u64,
    /// The residual stream: `[n][hidden]`.
    pub x: u64,
    /// The normed rows: `[n][hidden]`.
    pub h: u64,
    /// The rows B, C and x of a conv layer: `[n][3 * hidden]`.
    pub bcx: u64,
    /// The output of an operator before its projection: `[n][hidden]`.
    pub y: u64,
    /// The queries: `[n][heads * head_dim]`.
    pub q: u64,
    /// The gate of each head: `[n][heads]`.
    pub gates: u64,
    /// The keys and the values of a batch before their copies to the slots: `[n][kv_heads * head_dim]` each.
    pub k: u64,
    pub v: u64,
    /// The output of a projection before its norm: `[n][hidden]`.
    pub o: u64,
    /// The SwiGLU rows: `[n][intermediate]`.
    pub g: u64,
    /// The scores of block attention: `heads` rows for each token, as long as the largest slot.
    pub scores: u64,
    /// The logits of each logit row: `[MAX_LOGIT_ROWS][vocab_rows]`.
    pub logits: u64,
    /// The residual tap (spec 13.1): the residual after each layer at each logit row, `[MAX_LOGIT_ROWS][layers][hidden]`.
    pub taps: u64,
    /// The lens (spec 13.1): the final norm of the tapped row of each layer of one logit row, `[layers][hidden]`.
    pub lens_h: u64,
    /// The lens: the logits of each layer of one logit row, `[layers][vocab_rows]`.
    pub lens: u64,
}

impl Bufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, b: &Backbone, max_block: usize, max_seq: usize) -> Self {
        let f = |values: usize| values * max_block * 4;
        Self {
            ids: z(max_block * 4),
            x: z(f(b.hidden)),
            h: z(f(b.hidden)),
            bcx: z(f(3 * b.hidden)),
            y: z(f(b.hidden)),
            q: z(f(b.heads * b.head_dim)),
            gates: z(f(b.heads)),
            k: z(f(b.kv_row())),
            v: z(f(b.kv_row())),
            o: z(f(b.hidden)),
            g: z(f(b.intermediate)),
            scores: z(tasks::score_floats(
                b.heads as u32,
                max_block as u32,
                (max_seq - max_block.min(max_seq)) as u32,
            ) * 4),
            logits: z(MAX_LOGIT_ROWS * b.vocab_rows * 4),
            taps: z(MAX_LOGIT_ROWS * b.layers.len() * b.hidden * 4),
            lens_h: z(b.layers.len() * b.hidden * 4),
            lens: z(b.layers.len() * b.vocab_rows * 4),
        }
    }
}

/// The device bytes of the buffers, with no card: the plan (spec 2.4).
pub fn bufs_bytes(b: &Backbone, max_block: usize, max_seq: usize) -> usize {
    let mut total = 0usize;
    Bufs::new(
        &mut |bytes| {
            total += rounded(bytes);
            0
        },
        b,
        max_block,
        max_seq,
    );
    total
}

/// The device memory of one slot: the address of each layer's state. A conv layer's address is its conv state
/// `[taps - 1][hidden]`. An attention layer's address is its key plane, and its value plane follows it.
#[derive(Clone, Debug)]
pub struct SlotMemory {
    /// The positions of the slot.
    pub seq: usize,
    pub layers: Vec<u64>,
}

impl SlotMemory {
    /// The memory of a slot of `seq` positions, from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, b: &Backbone, seq: usize) -> Self {
        let layers = b
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Conv => z(b.conv_state() * 4),
                _ => z(2 * plane_rows(seq, *kind) * b.kv_row() * 4),
            })
            .collect();
        Self { seq, layers }
    }

    /// The ring rows of layer `l` in this slot, or 0 when its rows are positions.
    pub fn ring(&self, b: &Backbone, l: usize) -> usize {
        if b.layers[l] == LayerKind::Sliding {
            ring_of(self.seq)
        } else {
            0
        }
    }

    /// The address of the key row (`value` false) or the value row of position `pos` of attention layer `l`.
    pub fn row(&self, b: &Backbone, l: usize, value: bool, pos: usize) -> u64 {
        let plane = if value { plane_rows(self.seq, b.layers[l]) } else { 0 };
        let ring = self.ring(b, l);
        let at = if ring > 0 { pos % ring } else { pos };
        self.layers[l] + ((plane + at) * b.kv_row() * 4) as u64
    }

    /// The runs of device rows that hold positions `rows` of attention layer `l`: one run, or two where a ring wraps.
    /// Each run is the address of its first row and its count of rows.
    pub fn runs(&self, b: &Backbone, l: usize, value: bool, rows: std::ops::Range<usize>) -> Vec<(u64, usize)> {
        let ring = self.ring(b, l);
        let n = rows.len();
        if ring > 0 && rows.start % ring + n > ring {
            let head = ring - rows.start % ring;
            vec![
                (self.row(b, l, value, rows.start), head),
                (self.row(b, l, value, rows.start + head), n - head),
            ]
        } else {
            vec![(self.row(b, l, value, rows.start), n)]
        }
    }
}

/// One control item of a run (engine-abi "Control vectors"): the address of layer 0 of its vector in the table, the
/// op and the strength. Layer `l` of the vector follows at `l * hidden` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control {
    pub vector: u64,
    pub op: ControlOp,
    pub strength: f32,
}

/// What one ring runs: a block of `n` tokens at `pos0` of one slot, and how far it goes.
#[derive(Clone, Copy, Debug)]
pub struct Run<'a> {
    pub slot: &'a SlotMemory,
    pub pos0: usize,
    pub n: usize,
    /// The layers that the ring runs, from layer 0. A probe of the oracle stops early.
    pub layers: usize,
    /// The block rows whose logits the ring gives, after the final norm. With `None`, the ring stops after the layers.
    pub logit_rows: Option<&'a [usize]>,
    /// The tasks that run after the embedding rows: the copies of the media rows over the rows of their placeholders.
    pub after_embed: &'a [MkTaskDescriptor],
    /// The control items, in order. Each one changes every row of the block after each layer.
    pub control: &'a [Control],
    /// True when the ring copies the residual after each layer at each logit row into `bufs.taps` (spec 13.1).
    pub tap: bool,
    /// True when the ring also gives the lens of each tapped row in `bufs.lens`: the final norm, then the head (spec
    /// 13.1). It needs the tap and one logit row.
    pub lens: bool,
}

fn accumulate(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_ACCUMULATE;
    task
}

/// A weight op on the tiled path: one row sums in the order of 2 to 8 rows, so a context's bits do not depend on the
/// other segments of a pass (engine-abi "LFM3", "Batched passes").
fn tiled(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_TILED;
    task
}

/// The tasks of one run. The ids are in `bufs.ids`. After the run, `bufs.x` holds the residual stream after the layers
/// that ran, `bufs.h` holds the final norm, and `bufs.logits` row `i` holds the logits of logit row `i`.
pub fn pass(b: &Backbone, w: &Weights, bufs: &Bufs, run: Run<'_>) -> Vec<MkTaskDescriptor> {
    let Run {
        slot,
        pos0,
        n,
        layers,
        logit_rows,
        after_embed,
        control,
        tap,
        lens,
    } = run;
    assert!(
        layers <= b.layers.len() && (layers == b.layers.len() || logit_rows.is_none()),
        "a run gives logits only after every layer"
    );
    assert!(
        n >= 1 && pos0 + n <= slot.seq,
        "positions {pos0} to {} do not fit a slot of {}; the engine checks the pass first",
        pos0 + n,
        slot.seq
    );
    assert!(
        !tap || logit_rows.is_some(),
        "the residual tap reads the logit rows, so a run with the tap gives logits"
    );
    assert!(
        !lens || (tap && logit_rows.is_some_and(|rows| rows.len() == 1)),
        "the lens reads the tapped rows of one logit row; the engine checks the pass first"
    );
    let (n32, h32, i32_) = (n as u32, b.hidden as u32, b.intermediate as u32);
    let (heads, kv, d) = (b.heads as u32, b.kv_heads as u32, b.head_dim as u32);
    let mut t = vec![tasks::embed(bufs.ids, w.embed, bufs.x, n32, h32, 0, false, false)];
    t.extend_from_slice(after_embed);
    for (l, lw) in w.layers.iter().enumerate().take(layers) {
        let kind = b.layers[l];
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n32, h32, b.eps, false));
        match kind {
            LayerKind::Conv => {
                t.push(tiled(tasks::matmul(
                    bufs.h,
                    lw.in_proj,
                    bufs.bcx,
                    n32,
                    3 * h32,
                    h32,
                    false,
                )));
                t.push(tasks::short_conv(
                    bufs.bcx,
                    lw.conv,
                    slot.layers[l],
                    bufs.y,
                    n32,
                    h32,
                    b.taps as u32,
                    false,
                ));
                t.push(tiled(tasks::matmul(bufs.y, lw.out_proj, bufs.o, n32, h32, h32, false)));
            }
            LayerKind::Full | LayerKind::Sliding => {
                let (rope, window) = match kind {
                    LayerKind::Full => (b.full_rope, 0u32),
                    _ => (b.sliding_rope, b.window as u32),
                };
                let full = kind == LayerKind::Full;
                let (kr, vr) = (slot.row(b, l, false, pos0), slot.row(b, l, true, pos0));
                t.push(tiled(tasks::matmul(bufs.h, lw.q, bufs.q, n32, heads * d, h32, false)));
                t.push(tiled(tasks::matmul(bufs.h, lw.k, kr, n32, kv * d, h32, false)));
                t.push(tiled(tasks::matmul(bufs.h, lw.v, vr, n32, kv * d, h32, false)));
                t.push(tiled(tasks::matmul(
                    bufs.h, lw.gate, bufs.gates, n32, heads, h32, false,
                )));
                // The queries carry the scale head_dim^-0.5, because the attention ops do not scale.
                let scale = (b.head_dim as f32).powf(-0.5);
                t.push(tasks::partial_rotation(
                    tasks::qk_rope(
                        bufs.q,
                        lw.q_norm,
                        n32 * heads,
                        heads,
                        d,
                        pos0 as u32,
                        full,
                        true,
                        b.eps,
                        scale,
                    ),
                    rope.pairs as u32,
                ));
                t.push(tasks::partial_rotation(
                    tasks::qk_rope(kr, lw.k_norm, n32 * kv, kv, d, pos0 as u32, full, true, b.eps, 1.0),
                    rope.pairs as u32,
                ));
                let ring = slot.ring(b, l);
                if ring > 0 && pos0 % ring + n > ring {
                    // The rows past the end of the ring move to its start, bit for bit, before attention reads them.
                    let spill = ((pos0 % ring + n - ring) * b.kv_row()) as u32;
                    for value in [false, true] {
                        let start = slot.row(b, l, value, 0);
                        let past = start + (ring * b.kv_row() * 4) as u64;
                        t.push(tasks::copy(past, start, spill));
                    }
                }
                let base = slot.layers[l];
                let mut attend: Vec<MkTaskDescriptor> = if n == 1 {
                    tasks::attend_one(bufs.q, base, bufs.y, heads, kv, d, pos0 as u32, slot.seq as u32, window).to_vec()
                } else {
                    tasks::attend_block(
                        bufs.q,
                        base,
                        bufs.scores,
                        bufs.y,
                        n32,
                        heads,
                        kv,
                        d,
                        pos0 as u32,
                        slot.seq as u32,
                        window,
                    )
                    .to_vec()
                };
                for task in &mut attend {
                    task.layer = ring as u32;
                }
                t.extend(attend);
                t.push(tasks::head_gate(bufs.y, bufs.gates, bufs.y, n32, heads, d, heads));
                t.push(tiled(tasks::matmul(bufs.y, lw.o, bufs.o, n32, h32, heads * d, false)));
            }
        }
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_operator_norm,
            bufs.x,
            n32,
            h32,
            b.eps,
            false,
        )));
        t.push(tasks::rmsnorm(bufs.x, lw.ffn_norm, bufs.h, n32, h32, b.eps, false));
        t.push(tiled(tasks::swiglu(bufs.h, lw.w13, bufs.g, n32, i32_, h32)));
        t.push(tiled(tasks::matmul(bufs.g, lw.w2, bufs.o, n32, h32, i32_, false)));
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_ffn_norm,
            bufs.x,
            n32,
            h32,
            b.eps,
            false,
        )));
        // The control items change the residual before the next layer reads it. An add or an ablate of strength 0
        // changes no bit, so it gets no task (engine-abi "Control vectors").
        let layer_vector = (l * b.hidden * 4) as u64;
        for item in control {
            if item.strength == 0.0 && item.op != ControlOp::Clamp {
                continue;
            }
            t.push(tasks::control(
                bufs.x,
                item.vector + layer_vector,
                n32,
                h32,
                item.op,
                item.strength,
            ));
        }
        if tap {
            let rows = logit_rows.expect("the tap has logit rows");
            for (i, row) in rows.iter().enumerate() {
                t.push(tasks::copy(
                    bufs.x + (row * b.hidden * 4) as u64,
                    bufs.taps + ((i * b.layers.len() + l) * b.hidden * 4) as u64,
                    h32,
                ));
            }
        }
    }
    if let Some(rows) = logit_rows {
        assert!(
            rows.len() <= MAX_LOGIT_ROWS,
            "a run gives at most {MAX_LOGIT_ROWS} logit rows"
        );
        t.push(tasks::rmsnorm(bufs.x, w.norm, bufs.h, n32, h32, b.eps, false));
        // Each logit row is its own head matmul of one row, so the bits of a row do not depend on the count of rows.
        for (i, row) in rows.iter().enumerate() {
            assert!(*row < n, "logit row {row} is outside the block of {n}");
            t.push(tasks::matmul(
                bufs.h + (row * b.hidden * 4) as u64,
                w.embed,
                bufs.logits + (i * b.vocab_rows * 4) as u64,
                1,
                b.vocab_rows as u32,
                h32,
                false,
            ));
        }
        // The lens runs after the head, into its own buffers, so it changes no bit of the logits. One matmul of the
        // rows of every layer reads the table once.
        if lens {
            let l32 = b.layers.len() as u32;
            t.push(tasks::rmsnorm(bufs.taps, w.norm, bufs.lens_h, l32, h32, b.eps, false));
            t.push(tasks::matmul(
                bufs.lens_h,
                w.embed,
                bufs.lens,
                l32,
                b.vocab_rows as u32,
                h32,
                false,
            ));
        }
    }
    assert!(t.len() < MAX_TASKS_PER_CARD, "the pass has {} tasks", t.len());
    t
}

/// What a batched ring runs: one decode row on each of several slots, with one control list for all of them (engine-abi
/// "LFM3", "Batched passes"). The ids are in `bufs.ids`, one for each row.
#[derive(Clone, Copy, Debug)]
pub struct Batch<'a> {
    /// The slot of each row and the position of its token.
    pub rows: &'a [(&'a SlotMemory, usize)],
    pub control: &'a [Control],
}

/// The tasks of a batch. The weight ops run once over the rows, on the tiled path. Each row has its own convolution,
/// K and V rows, rotation and attention on its own slot, as a pass of one row has. Thus a row gets the bits of a pass
/// alone. `bufs.logits` row `i` holds the logits of row `i`.
pub fn batch(b: &Backbone, w: &Weights, bufs: &Bufs, batch: Batch<'_>) -> Vec<MkTaskDescriptor> {
    let Batch { rows, control } = batch;
    let n = rows.len();
    assert!(
        (1..=MAX_LOGIT_ROWS).contains(&n),
        "a batch has 1 to {MAX_LOGIT_ROWS} rows, not {n}; the engine checks the pass first"
    );
    for (slot, pos) in rows {
        assert!(*pos < slot.seq, "position {pos} does not fit a slot of {}", slot.seq);
    }
    let (n32, h32, i32_) = (n as u32, b.hidden as u32, b.intermediate as u32);
    let (heads, kv, d) = (b.heads as u32, b.kv_heads as u32, b.head_dim as u32);
    let (hidden, q_row, kv_row) = (b.hidden * 4, b.heads * b.head_dim * 4, b.kv_row() * 4);
    let mut t = vec![tasks::embed(bufs.ids, w.embed, bufs.x, n32, h32, 0, false, false)];
    for (l, lw) in w.layers.iter().enumerate() {
        let kind = b.layers[l];
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n32, h32, b.eps, false));
        match kind {
            LayerKind::Conv => {
                t.push(tiled(tasks::matmul(
                    bufs.h,
                    lw.in_proj,
                    bufs.bcx,
                    n32,
                    3 * h32,
                    h32,
                    false,
                )));
                for (i, (slot, _)) in rows.iter().enumerate() {
                    t.push(tasks::short_conv(
                        bufs.bcx + (i * 3 * hidden) as u64,
                        lw.conv,
                        slot.layers[l],
                        bufs.y + (i * hidden) as u64,
                        1,
                        h32,
                        b.taps as u32,
                        false,
                    ));
                }
                t.push(tiled(tasks::matmul(bufs.y, lw.out_proj, bufs.o, n32, h32, h32, false)));
            }
            LayerKind::Full | LayerKind::Sliding => {
                let (rope, window) = match kind {
                    LayerKind::Full => (b.full_rope, 0u32),
                    _ => (b.sliding_rope, b.window as u32),
                };
                let full = kind == LayerKind::Full;
                t.push(tiled(tasks::matmul(bufs.h, lw.q, bufs.q, n32, heads * d, h32, false)));
                t.push(tiled(tasks::matmul(bufs.h, lw.k, bufs.k, n32, kv * d, h32, false)));
                t.push(tiled(tasks::matmul(bufs.h, lw.v, bufs.v, n32, kv * d, h32, false)));
                t.push(tiled(tasks::matmul(
                    bufs.h, lw.gate, bufs.gates, n32, heads, h32, false,
                )));
                // The queries carry the scale head_dim^-0.5, because the attention ops do not scale.
                let scale = (b.head_dim as f32).powf(-0.5);
                let ring = slot_ring(b, rows, l);
                for (i, (slot, pos)) in rows.iter().enumerate() {
                    let (kr, vr) = (slot.row(b, l, false, *pos), slot.row(b, l, true, *pos));
                    // A row of one token never wraps a ring, so its rows go straight to their places.
                    t.push(tasks::copy(bufs.k + (i * kv_row) as u64, kr, kv * d));
                    t.push(tasks::copy(bufs.v + (i * kv_row) as u64, vr, kv * d));
                    let q = bufs.q + (i * q_row) as u64;
                    t.push(tasks::partial_rotation(
                        tasks::qk_rope(q, lw.q_norm, heads, heads, d, *pos as u32, full, true, b.eps, scale),
                        rope.pairs as u32,
                    ));
                    t.push(tasks::partial_rotation(
                        tasks::qk_rope(kr, lw.k_norm, kv, kv, d, *pos as u32, full, true, b.eps, 1.0),
                        rope.pairs as u32,
                    ));
                    let mut attend = tasks::attend_one(
                        q,
                        slot.layers[l],
                        bufs.y + (i * hidden) as u64,
                        heads,
                        kv,
                        d,
                        *pos as u32,
                        slot.seq as u32,
                        window,
                    );
                    for task in &mut attend {
                        task.layer = ring[i] as u32;
                    }
                    t.extend(attend);
                }
                t.push(tasks::head_gate(bufs.y, bufs.gates, bufs.y, n32, heads, d, heads));
                t.push(tiled(tasks::matmul(bufs.y, lw.o, bufs.o, n32, h32, heads * d, false)));
            }
        }
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_operator_norm,
            bufs.x,
            n32,
            h32,
            b.eps,
            false,
        )));
        t.push(tasks::rmsnorm(bufs.x, lw.ffn_norm, bufs.h, n32, h32, b.eps, false));
        t.push(tiled(tasks::swiglu(bufs.h, lw.w13, bufs.g, n32, i32_, h32)));
        t.push(tiled(tasks::matmul(bufs.g, lw.w2, bufs.o, n32, h32, i32_, false)));
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_ffn_norm,
            bufs.x,
            n32,
            h32,
            b.eps,
            false,
        )));
        // One control list for every row. The op changes each row on its own, so a row has the bits of a pass alone.
        let layer_vector = (l * b.hidden * 4) as u64;
        for item in control {
            if item.strength == 0.0 && item.op != ControlOp::Clamp {
                continue;
            }
            t.push(tasks::control(
                bufs.x,
                item.vector + layer_vector,
                n32,
                h32,
                item.op,
                item.strength,
            ));
        }
    }
    t.push(tasks::rmsnorm(bufs.x, w.norm, bufs.h, n32, h32, b.eps, false));
    // Each row is its own head matmul of one row, as in a pass of one row.
    for i in 0..n {
        t.push(tasks::matmul(
            bufs.h + (i * hidden) as u64,
            w.embed,
            bufs.logits + (i * b.vocab_rows * 4) as u64,
            1,
            b.vocab_rows as u32,
            h32,
            false,
        ));
    }
    assert!(t.len() < MAX_TASKS_PER_CARD, "the batch has {} tasks", t.len());
    t
}

/// The ring rows of layer `l` of the slot of each row.
fn slot_ring(b: &Backbone, rows: &[(&SlotMemory, usize)], l: usize) -> Vec<usize> {
    rows.iter().map(|(slot, _)| slot.ring(b, l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::tests::backbone;
    use nagaya_mk_bmg::protocol::{
        OP_ATTN, OP_ATTN_SCORES, OP_CONTROL, OP_COPY, OP_EMBED, OP_GEGLU, OP_MATMUL, OP_SHORT_CONV,
    };

    fn weights(b: &Backbone) -> Weights {
        Weights {
            embed: 0x1000,
            norm: 0x2000,
            layers: vec![crate::weights::Layer::default(); b.layers.len()],
        }
    }

    fn memory(b: &Backbone, seq: usize) -> SlotMemory {
        memory_at(b, seq, 1 << 40)
    }

    /// The memory of a slot from `base`, so that several slots of a test do not overlap.
    fn memory_at(b: &Backbone, seq: usize, base: u64) -> SlotMemory {
        let mut next = base;
        SlotMemory::new(
            &mut |bytes| {
                let at = next;
                next += rounded(bytes) as u64;
                at
            },
            b,
            seq,
        )
    }

    fn bufs(b: &Backbone) -> Bufs {
        let mut next = 1u64 << 32;
        Bufs::new(
            &mut |bytes| {
                let at = next;
                next += rounded(bytes) as u64;
                at
            },
            b,
            512,
            4096,
        )
    }

    fn count(t: &[MkTaskDescriptor], op: u16) -> usize {
        t.iter().filter(|task| task.opcode == op).count()
    }

    #[test]
    fn a_pass_runs_each_layer_and_one_head_matmul_for_each_logit_row() {
        let b = backbone();
        let (w, bufs, slot) = (weights(&b), bufs(&b), memory(&b, 1024));
        let rows = [0usize, 4, 7];
        let t = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 0,
                n: 8,
                layers: b.layers.len(),
                logit_rows: Some(&rows),
                after_embed: &[],
                control: &[],
                tap: false,
                lens: false,
            },
        );
        assert_eq!(t[0].opcode, OP_EMBED);
        assert_eq!(count(&t, OP_SHORT_CONV), 13);
        assert_eq!(count(&t, OP_ATTN_SCORES), 11);
        // Each conv layer: in_proj, out_proj, w2. Each attention layer: q, k, v, gate, o, w2. Then 3 heads.
        assert_eq!(count(&t, OP_MATMUL), 13 * 3 + 11 * 6 + 3);
        let heads: Vec<&MkTaskDescriptor> = t.iter().rev().take(3).collect();
        for (i, task) in heads.iter().rev().enumerate() {
            assert_eq!((task.opcode, task.n, task.m), (OP_MATMUL, 1, 128_000));
            assert_eq!(task.a, bufs.h + (rows[i] * 2048 * 4) as u64);
            assert_eq!(task.c, bufs.logits + (i * 128_000 * 4) as u64);
        }
    }

    /// Each matrix multiply and SwiGLU of the backbone takes the tiled path, so a decode row has the bits of a row in
    /// any batch of 1 to 8 (engine-abi "LFM3", "Batched passes"). The heads are 128,000 rows tall, so one row already
    /// takes that path.
    #[test]
    fn each_matmul_and_swiglu_of_the_backbone_takes_the_tiled_path() {
        let b = backbone();
        let (w, bufs, slot) = (weights(&b), bufs(&b), memory(&b, 1024));
        let rows = [0usize];
        let t = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 5,
                n: 1,
                layers: b.layers.len(),
                logit_rows: Some(&rows),
                after_embed: &[],
                control: &[],
                tap: false,
                lens: false,
            },
        );
        let weight_ops: Vec<&MkTaskDescriptor> = t
            .iter()
            .filter(|task| (task.opcode == OP_MATMUL && task.m != 128_000) || task.opcode == OP_GEGLU)
            .collect();
        assert_eq!(weight_ops.len(), 13 * 3 + 11 * 6 + 24);
        for task in weight_ops {
            assert_ne!(task.flags & FLAG_TILED, 0, "op {} of {} rows", task.opcode, task.m);
        }
    }

    /// A batch of three rows on three slots: the weight ops run once over 3 rows, and each row has its own convolution,
    /// K and V rows, rotation and attention on its own slot, then its own head (engine-abi "LFM3", "Batched passes").
    #[test]
    fn a_batch_runs_the_weight_ops_once_and_each_row_on_its_own_slot() {
        let b = backbone();
        let (w, bufs) = (weights(&b), bufs(&b));
        let slots = [
            memory_at(&b, 1024, 1 << 40),
            memory_at(&b, 1024, 2 << 40),
            memory_at(&b, 1024, 3 << 40),
        ];
        let at = [5usize, 300, 1000];
        let rows: Vec<(&SlotMemory, usize)> = slots.iter().zip(at).collect();
        let t = batch(
            &b,
            &w,
            &bufs,
            Batch {
                rows: &rows,
                control: &[],
            },
        );
        assert_eq!(t[0].opcode, OP_EMBED);
        assert_eq!(t[0].n, 3);
        // The weight ops: 3 rows each, on the tiled path. Then one head matmul for each row.
        let heads: Vec<&MkTaskDescriptor> = t.iter().filter(|task| task.m == 128_000).collect();
        let weight_ops: Vec<&MkTaskDescriptor> = t
            .iter()
            .filter(|task| (task.opcode == OP_MATMUL && task.m != 128_000) || task.opcode == OP_GEGLU)
            .collect();
        assert_eq!(weight_ops.len(), 13 * 3 + 11 * 6 + 24);
        assert!(
            weight_ops
                .iter()
                .all(|task| task.n == 3 && task.flags & FLAG_TILED != 0)
        );
        assert_eq!(heads.len(), 3);
        for (i, task) in heads.iter().enumerate() {
            assert_eq!((task.n, task.a), (1, bufs.h + (i * 2048 * 4) as u64));
            assert_eq!(task.c, bufs.logits + (i * 128_000 * 4) as u64);
        }
        for (l, kind) in b.layers.iter().enumerate() {
            let on = |slot: &SlotMemory, pos: usize, value: bool| slot.row(&b, l, value, pos);
            match kind {
                LayerKind::Conv => {
                    let convs: Vec<(u64, u64, u32)> = t
                        .iter()
                        .filter(|task| task.opcode == OP_SHORT_CONV && slots.iter().any(|s| s.layers[l] == task.x))
                        .map(|task| (task.x, task.a, task.n))
                        .collect();
                    let want: Vec<(u64, u64, u32)> = (0..3)
                        .map(|i| (slots[i].layers[l], bufs.bcx + (i * 3 * 2048 * 4) as u64, 1))
                        .collect();
                    assert_eq!(convs, want, "layer {l}");
                }
                _ => {
                    for (i, slot) in slots.iter().enumerate() {
                        for value in [false, true] {
                            assert!(
                                t.iter()
                                    .any(|task| task.opcode == OP_COPY && task.c == on(slot, at[i], value)),
                                "layer {l} row {i}: the copy of its {} row",
                                if value { "value" } else { "key" }
                            );
                        }
                        let attend = t
                            .iter()
                            .find(|task| task.opcode == OP_ATTN && task.b == slot.layers[l])
                            .unwrap_or_else(|| panic!("layer {l} row {i} attends on its slot"));
                        assert_eq!(
                            (attend.p0, attend.a),
                            (at[i] as u32, bufs.q + (i * 16 * 128 * 4) as u64)
                        );
                    }
                }
            }
        }
        assert_eq!(t.iter().filter(|task| task.opcode == OP_ATTN).count(), 11 * 3);
    }

    /// Weights with a distinct address for the norm of each layer, so a test finds the accumulate of each layer.
    fn weights_with_norms(b: &Backbone) -> Weights {
        let mut w = weights(b);
        for (l, lw) in w.layers.iter_mut().enumerate() {
            lw.post_operator_norm = 0x10_0000 + l as u64 * 0x100;
            lw.post_ffn_norm = 0x20_0000 + l as u64 * 0x100;
        }
        w
    }

    #[test]
    fn the_control_items_run_in_order_after_each_layer_and_skip_an_add_or_an_ablate_of_strength_0() {
        let b = backbone();
        let (w, bufs, slot) = (weights_with_norms(&b), bufs(&b), memory(&b, 1024));
        let control = [
            Control {
                vector: 0xC000_0000,
                op: ControlOp::Add,
                strength: 4.0,
            },
            Control {
                vector: 0xD000_0000,
                op: ControlOp::Ablate,
                strength: 0.0,
            },
            Control {
                vector: 0xD000_0000,
                op: ControlOp::Add,
                strength: -0.0,
            },
            Control {
                vector: 0xE000_0000,
                op: ControlOp::Clamp,
                strength: 0.0,
            },
        ];
        let run = |control: &[Control]| {
            pass(
                &b,
                &w,
                &bufs,
                Run {
                    slot: &slot,
                    pos0: 0,
                    n: 8,
                    layers: b.layers.len(),
                    logit_rows: Some(&[7]),
                    after_embed: &[],
                    control,
                    tap: false,
                    lens: false,
                },
            )
        };
        let plain = run(&[]);
        assert_eq!(count(&plain, OP_CONTROL), 0);
        let t = run(&control);
        assert_eq!(
            count(&t, OP_CONTROL),
            2 * b.layers.len(),
            "the add 4 and the clamp 0 of each layer"
        );
        // The x of a post_ffn accumulate is the residual; find each one by its norm.
        let ends: Vec<usize> = t
            .iter()
            .enumerate()
            .filter(|(_, task)| {
                w.layers.iter().any(|lw| lw.post_ffn_norm == task.b) && task.flags & FLAG_ACCUMULATE != 0
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(ends.len(), b.layers.len());
        for (l, end) in ends.iter().enumerate() {
            let row = (l * 2048 * 4) as u64;
            let (add, clamp) = (&t[end + 1], &t[end + 2]);
            assert_eq!(
                (add.opcode, add.a, add.b, add.n, add.k),
                (OP_CONTROL, bufs.x, 0xC000_0000 + row, 8, 2048)
            );
            assert_eq!((add.p0, f32::from_bits(add.m)), (ControlOp::Add as u32, 4.0));
            assert_eq!(
                (clamp.opcode, clamp.b, clamp.p0),
                (OP_CONTROL, 0xE000_0000 + row, ControlOp::Clamp as u32)
            );
            assert_ne!(t[end + 3].opcode, OP_CONTROL, "layer {l} has two control tasks");
        }
        // With the items gone, the tasks are those of the plain pass.
        let left: Vec<MkTaskDescriptor> = t.iter().copied().filter(|task| task.opcode != OP_CONTROL).collect();
        assert_eq!(left.len(), plain.len());
        assert!(
            left.iter()
                .zip(&plain)
                .all(|(a, p)| format!("{a:?}") == format!("{p:?}"))
        );
    }

    #[test]
    fn the_tap_copies_the_residual_after_the_items_of_each_layer_at_each_logit_row() {
        let b = backbone();
        let (w, bufs, slot) = (weights_with_norms(&b), bufs(&b), memory(&b, 1024));
        let rows = [2usize, 7];
        let control = [Control {
            vector: 0xC000_0000,
            op: ControlOp::Clamp,
            strength: 8.0,
        }];
        let t = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 0,
                n: 8,
                layers: b.layers.len(),
                logit_rows: Some(&rows),
                after_embed: &[],
                control: &control,
                tap: true,
                lens: false,
            },
        );
        let items: Vec<usize> = t
            .iter()
            .enumerate()
            .filter(|(_, task)| task.opcode == OP_CONTROL)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(items.len(), b.layers.len());
        for (l, at) in items.iter().enumerate() {
            for (i, row) in rows.iter().enumerate() {
                let copy = &t[at + 1 + i];
                assert_eq!(copy.opcode, OP_COPY, "layer {l} row {i}");
                assert_eq!(copy.a, bufs.x + (row * 2048 * 4) as u64);
                assert_eq!(copy.c, bufs.taps + ((i * 24 + l) * 2048 * 4) as u64);
                assert_eq!(copy.n, 2048);
            }
        }
        assert_eq!(count(&t, OP_COPY), 24 * rows.len());
        // With no tap, the run has no copy.
        let untapped = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 0,
                n: 8,
                layers: b.layers.len(),
                logit_rows: Some(&rows),
                after_embed: &[],
                control: &control,
                tap: false,
                lens: false,
            },
        );
        assert_eq!(count(&untapped, OP_COPY), 0);
    }

    /// The lens (spec 13.1): after the head, one norm and one head matmul take the tapped row of each layer of the one
    /// logit row into the buffers of the lens. The tasks before them are the tasks of the tap alone.
    #[test]
    fn the_lens_takes_the_final_norm_and_the_head_of_each_tapped_row_after_the_head() {
        let b = backbone();
        let (w, bufs, slot) = (weights_with_norms(&b), bufs(&b), memory(&b, 1024));
        let run = |tap: bool, lens: bool| {
            pass(
                &b,
                &w,
                &bufs,
                Run {
                    slot: &slot,
                    pos0: 0,
                    n: 8,
                    layers: b.layers.len(),
                    logit_rows: Some(&[7]),
                    after_embed: &[],
                    control: &[],
                    tap,
                    lens,
                },
            )
        };
        let tapped = run(true, false);
        let lensed = run(true, true);
        assert_eq!(lensed.len(), tapped.len() + 2);
        assert!(
            lensed
                .iter()
                .zip(&tapped)
                .all(|(a, t)| format!("{a:?}") == format!("{t:?}")),
            "the lens adds tasks after the head and changes none before"
        );
        let (norm, head) = (&lensed[tapped.len()], &lensed[tapped.len() + 1]);
        assert_eq!(
            (norm.opcode, norm.a, norm.b, norm.c, norm.n, norm.k),
            (
                nagaya_mk_bmg::protocol::OP_RMSNORM,
                bufs.taps,
                w.norm,
                bufs.lens_h,
                24,
                2048
            )
        );
        assert_eq!(norm.x, b.eps.to_bits() as u64, "the final norm with its epsilon");
        assert_eq!(
            (head.opcode, head.a, head.b, head.c, head.n, head.m, head.k),
            (OP_MATMUL, bufs.lens_h, w.embed, bufs.lens, 24, 128_000, 2048)
        );
        assert_ne!(bufs.lens_h, bufs.h, "the lens has its own buffers");
    }

    #[test]
    #[should_panic(expected = "one logit row")]
    fn a_lens_of_two_logit_rows_is_a_defect() {
        let b = backbone();
        let (w, bufs, slot) = (weights(&b), bufs(&b), memory(&b, 1024));
        pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 0,
                n: 8,
                layers: b.layers.len(),
                logit_rows: Some(&[6, 7]),
                after_embed: &[],
                control: &[],
                tap: true,
                lens: true,
            },
        );
    }

    #[test]
    fn the_tap_buffer_holds_every_layer_of_each_logit_row() {
        let b = backbone();
        let with = bufs_bytes(&b, 512, 4096);
        let bufs = bufs(&b);
        assert!(bufs.taps > bufs.logits);
        let mut sizes = Vec::new();
        Bufs::new(
            &mut |bytes| {
                sizes.push(bytes);
                0
            },
            &b,
            512,
            4096,
        );
        // The taps, then the normed rows and the logit rows of the lens of one logit row.
        assert_eq!(
            sizes[sizes.len() - 3..],
            [16 * 24 * 2048 * 4, 24 * 2048 * 4, 24 * 128_000 * 4]
        );
        assert_eq!(with, sizes.iter().map(|s| rounded(*s)).sum::<usize>());
    }

    #[test]
    fn the_media_rows_replace_the_embedding_rows_before_the_first_layer() {
        let b = backbone();
        let (w, bufs, slot) = (weights(&b), bufs(&b), memory(&b, 1024));
        let place = tasks::copy(0xAA00, bufs.x + 5 * 2048 * 4, 77 * 2048);
        let t = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 0,
                n: 90,
                layers: b.layers.len(),
                logit_rows: Some(&[89]),
                after_embed: &[place],
                control: &[],
                tap: false,
                lens: false,
            },
        );
        assert_eq!((t[0].opcode, t[1].opcode), (OP_EMBED, OP_COPY));
        assert_eq!((t[1].a, t[1].c, t[1].n), (place.a, place.c, place.n));
        assert_eq!(
            t[2].opcode,
            nagaya_mk_bmg::protocol::OP_RMSNORM,
            "layer 0 starts after the copy"
        );
    }

    #[test]
    fn one_token_takes_the_one_token_attention_and_a_ring_wraps_with_a_copy() {
        let b = backbone();
        let (w, bufs) = (weights(&b), bufs(&b));
        let slot = memory(&b, 4096);
        let one = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 100,
                n: 1,
                layers: b.layers.len(),
                logit_rows: Some(&[0]),
                after_embed: &[],
                control: &[],
                tap: false,
                lens: false,
            },
        );
        assert_eq!((count(&one, OP_ATTN), count(&one, OP_ATTN_SCORES)), (11, 0));
        assert_eq!(count(&one, OP_COPY), 0);
        // A block of 100 at position 2000 crosses the end of the ring of 2048 rows in the 5 sliding layers.
        let wrap = pass(
            &b,
            &w,
            &bufs,
            Run {
                slot: &slot,
                pos0: 2000,
                n: 100,
                layers: b.layers.len(),
                logit_rows: None,
                after_embed: &[],
                control: &[],
                tap: false,
                lens: false,
            },
        );
        let copies: Vec<&MkTaskDescriptor> = wrap.iter().filter(|task| task.opcode == OP_COPY).collect();
        assert_eq!(copies.len(), 10);
        assert!(copies.iter().all(|task| task.n == 52 * 512));
        let rings: Vec<u32> = wrap
            .iter()
            .filter(|task| task.opcode == OP_ATTN_SCORES)
            .map(|task| task.layer)
            .collect();
        assert_eq!(rings, vec![0, 2048, 0, 2048, 0, 2048, 0, 2048, 0, 2048, 0]);
    }

    #[test]
    fn a_ring_row_is_its_position_modulo_the_ring_and_a_run_splits_at_the_end() {
        let b = backbone();
        let slot = memory(&b, 4096);
        let sliding = 4;
        let row = (b.kv_row() * 4) as u64;
        assert_eq!(slot.row(&b, sliding, false, 2049), slot.layers[sliding] + row);
        let plane = (2048 + 1024) as u64 * row;
        assert_eq!(slot.row(&b, sliding, true, 0), slot.layers[sliding] + plane);
        assert_eq!(
            slot.runs(&b, sliding, false, 2040..2060),
            vec![(slot.layers[sliding] + 2040 * row, 8), (slot.layers[sliding], 12)]
        );
        let full = 2;
        assert_eq!(slot.row(&b, full, true, 3000), slot.layers[full] + (4096 + 3000) * row);
    }
}
