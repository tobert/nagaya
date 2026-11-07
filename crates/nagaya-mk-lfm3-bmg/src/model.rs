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

use nagaya_mk_bmg::protocol::{FLAG_ACCUMULATE, MAX_TASKS_PER_CARD, MkTaskDescriptor};
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
    /// The output of a projection before its norm: `[n][hidden]`.
    pub o: u64,
    /// The SwiGLU rows: `[n][intermediate]`.
    pub g: u64,
    /// The scores of block attention: `heads` rows for each token, as long as the largest slot.
    pub scores: u64,
    /// The logits of each logit row: `[MAX_LOGIT_ROWS][vocab_rows]`.
    pub logits: u64,
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
            o: z(f(b.hidden)),
            g: z(f(b.intermediate)),
            scores: z(tasks::score_floats(
                b.heads as u32,
                max_block as u32,
                (max_seq - max_block.min(max_seq)) as u32,
            ) * 4),
            logits: z(MAX_LOGIT_ROWS * b.vocab_rows * 4),
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
}

fn accumulate(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_ACCUMULATE;
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
    let (n32, h32, i32_) = (n as u32, b.hidden as u32, b.intermediate as u32);
    let (heads, kv, d) = (b.heads as u32, b.kv_heads as u32, b.head_dim as u32);
    let mut t = vec![tasks::embed(bufs.ids, w.embed, bufs.x, n32, h32, 0, false, false)];
    t.extend_from_slice(after_embed);
    for (l, lw) in w.layers.iter().enumerate().take(layers) {
        let kind = b.layers[l];
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n32, h32, b.eps, false));
        match kind {
            LayerKind::Conv => {
                t.push(tasks::matmul(bufs.h, lw.in_proj, bufs.bcx, n32, 3 * h32, h32, false));
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
                t.push(tasks::matmul(bufs.y, lw.out_proj, bufs.o, n32, h32, h32, false));
            }
            LayerKind::Full | LayerKind::Sliding => {
                let (rope, window) = match kind {
                    LayerKind::Full => (b.full_rope, 0u32),
                    _ => (b.sliding_rope, b.window as u32),
                };
                let full = kind == LayerKind::Full;
                let (kr, vr) = (slot.row(b, l, false, pos0), slot.row(b, l, true, pos0));
                t.push(tasks::matmul(bufs.h, lw.q, bufs.q, n32, heads * d, h32, false));
                t.push(tasks::matmul(bufs.h, lw.k, kr, n32, kv * d, h32, false));
                t.push(tasks::matmul(bufs.h, lw.v, vr, n32, kv * d, h32, false));
                t.push(tasks::matmul(bufs.h, lw.gate, bufs.gates, n32, heads, h32, false));
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
                t.push(tasks::matmul(bufs.y, lw.o, bufs.o, n32, h32, heads * d, false));
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
        t.push(tasks::swiglu(bufs.h, lw.w13, bufs.g, n32, i32_, h32));
        t.push(tasks::matmul(bufs.g, lw.w2, bufs.o, n32, h32, i32_, false));
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_ffn_norm,
            bufs.x,
            n32,
            h32,
            b.eps,
            false,
        )));
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
    }
    assert!(t.len() < MAX_TASKS_PER_CARD, "the pass has {} tasks", t.len());
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::tests::backbone;
    use nagaya_mk_bmg::protocol::{OP_ATTN, OP_ATTN_SCORES, OP_COPY, OP_EMBED, OP_MATMUL, OP_SHORT_CONV};

    fn weights(b: &Backbone) -> Weights {
        Weights {
            embed: 0x1000,
            norm: 0x2000,
            layers: vec![crate::weights::Layer::default(); b.layers.len()],
        }
    }

    fn memory(b: &Backbone, seq: usize) -> SlotMemory {
        let mut next = 1u64 << 40;
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
