//! The forward of the Inkling 2 text model on the host: the runs of a span of positions, and the state that a
//! context holds between two spans.
//!
//! The order follows `Inkling2LoopTextModel.forward` (lines 1359 to 1443): layers 0 to 5, then each cycle of layers
//! 6 to 17, then layers 18 to 23, then the output residual, the norm, the head and the softcap.

use super::ops::{self, CONV_TAPS, HEAD_DIM, HEADS, HIDDEN, KV_HEADS, REL_EXTENT, TOP_K};
use super::weights::{BLOCK, CYCLE_END, CYCLE_START, KvConv, LAYERS, Layer, Matrix, Weights};
use rayon::prelude::*;
use std::path::Path;

/// The width of the keys and the values of a run: 8 heads of 128.
pub const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
/// The count of cycles that a context runs when it names none (engine-abi "Inkling 2").
pub const DEFAULT_CYCLES: usize = 3;

/// The layer of each run, in the order of the runs: layers 0 to 5, `cycles` times layers 6 to 17, layers 18 to 23.
pub fn run_layers(cycles: usize) -> Vec<usize> {
    assert!(cycles >= 1, "run_layers: a context runs 1 cycle or more, not {cycles}");
    let mut out: Vec<usize> = (0..CYCLE_START).collect();
    for _ in 0..cycles {
        out.extend(CYCLE_START..CYCLE_END);
    }
    out.extend(CYCLE_END..LAYERS);
    out
}

/// The keys or the values of one run: one row of `KV_WIDTH` for each held position, oldest first. A sliding run
/// holds the last `REL_EXTENT` positions, and a full run holds all.
pub struct KvRows {
    pub window: Option<usize>,
    /// The position of the first held row.
    pub first: usize,
    pub rows: Vec<f32>,
}

impl KvRows {
    fn new(window: Option<usize>) -> Self {
        Self {
            window,
            first: 0,
            rows: Vec::new(),
        }
    }

    /// The count of held rows.
    pub fn len(&self) -> usize {
        self.rows.len() / KV_WIDTH
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Append the rows of a span, then drop the rows that are out of the window.
    fn push(&mut self, rows: &[f32]) {
        self.rows.extend_from_slice(rows);
        if let Some(window) = self.window {
            let extra = self.len().saturating_sub(window);
            self.rows.drain(..extra * KV_WIDTH);
            self.first += extra;
        }
    }
}

/// The first position that the query at `pos` reads, when the run holds the rows from `first`. A sliding layer
/// reads a key when `pos - key < window` (`sliding_window_overlay` of transformers), and a full layer reads all.
pub fn visible_from(pos: usize, first: usize, window: Option<usize>) -> usize {
    let lowest = window.map_or(first, |w| (pos + 1).saturating_sub(w));
    assert!(
        lowest >= first,
        "the query at {pos} reads the key at {lowest}, and the run holds keys from {first}"
    );
    lowest
}

/// The state of one run: its keys and values, and the 3 last inputs of its four convolutions, oldest first.
pub struct RunState {
    pub layer: usize,
    pub keys: KvRows,
    pub values: KvRows,
    /// The key, the value, the attention output and the expert output: 3 rows of 1,024, 1,024, 2,304 and 2,304.
    pub conv: [Vec<f32>; 4],
}

/// The state of one context: the count of cycles, the count of positions that it holds, and the state of each run.
pub struct Context {
    pub cycles: usize,
    pub position: usize,
    pub runs: Vec<RunState>,
}

impl Context {
    /// An empty context of `cycles` cycles. The convolution states start as zeros, as in the reference.
    pub fn new(cycles: usize) -> Self {
        let runs = run_layers(cycles)
            .into_iter()
            .map(|layer| {
                let window = if super::weights::is_full(layer) {
                    None
                } else {
                    Some(REL_EXTENT)
                };
                RunState {
                    layer,
                    keys: KvRows::new(window),
                    values: KvRows::new(window),
                    conv: [KV_WIDTH, KV_WIDTH, HIDDEN, HIDDEN].map(|c| vec![0f32; (CONV_TAPS - 1) * c]),
                }
            })
            .collect();
        Self {
            cycles,
            position: 0,
            runs,
        }
    }
}

/// The rows that a forward can keep for a gate, each `T x HIDDEN` for a span of `T` positions.
#[derive(Default)]
pub struct Trace {
    /// The row after each run, in the order of the runs.
    pub runs: Vec<Vec<f32>>,
    /// The message of each layer of the cycles after the first, `T x 4 x HIDDEN`.
    pub messages: Vec<Vec<f32>>,
    /// The output of each message gate, in call order: query, key, value, expert input of each layer.
    pub gates: Vec<Vec<f32>>,
    /// The row after the output residual and the final norm.
    pub final_norm: Vec<f32>,
}

/// The text model on the host.
pub struct Model {
    pub weights: Weights,
}

/// `xs W^T` for a `bf16` matrix: `n x rows` from `n x cols`. The rows of `W` run in parallel.
fn linear(w: &Weights, m: &Matrix, xs: &[f32]) -> Vec<f32> {
    let n = xs.len() / m.cols;
    assert_eq!(
        xs.len(),
        n * m.cols,
        "linear: {} inputs for rows of {}",
        xs.len(),
        m.cols
    );
    let bytes = w.checkpoint.of(m);
    let chunk = 16;
    let mut transposed = vec![0f32; m.rows * n];
    transposed.par_chunks_mut(chunk * n).enumerate().for_each(|(c, out)| {
        let mut row = vec![0f32; m.cols];
        for (i, o) in out.chunks_exact_mut(n).enumerate() {
            let r = c * chunk + i;
            ops::widen_bf16_row(&bytes[2 * r * m.cols..2 * (r + 1) * m.cols], &mut row);
            for (t, v) in o.iter_mut().enumerate() {
                *v = ops::dot(&row, &xs[t * m.cols..(t + 1) * m.cols]);
            }
        }
    });
    transpose(&transposed, m.rows, n)
}

/// The transpose of a `rows x cols` matrix.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

fn row(a: &[f32], t: usize, width: usize) -> &[f32] {
    &a[t * width..(t + 1) * width]
}

impl Model {
    /// Map the checkpoint at `path`.
    pub fn open(path: &Path) -> Result<Self, String> {
        Ok(Self {
            weights: Weights::open(path)?,
        })
    }

    /// The normed embedding rows of `ids` (`Inkling2LoopNormedEmbedding`, line 1246).
    pub fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let w = &self.weights;
        let bytes = w.checkpoint.of(&w.embed);
        let mut out = vec![0f32; ids.len() * HIDDEN];
        let mut raw = vec![0f32; HIDDEN];
        for (t, id) in ids.iter().enumerate() {
            let id = *id as usize;
            assert!(
                id < w.embed.rows,
                "the token {id} is outside the vocabulary of {} entries",
                w.embed.rows
            );
            ops::widen_bf16_row(&bytes[2 * id * HIDDEN..2 * (id + 1) * HIDDEN], &mut raw);
            ops::rms_norm(&raw, &w.embed_norm, &mut out[t * HIDDEN..(t + 1) * HIDDEN]);
        }
        out
    }

    /// Run the span `ids` at the next positions of `ctx`, and return the logits of the span rows `logit_rows`.
    /// Each logit row has `LOGITS` values after the softcap of 30.
    pub fn forward(
        &self,
        ctx: &mut Context,
        ids: &[u32],
        logit_rows: &[usize],
        mut trace: Option<&mut Trace>,
    ) -> Vec<Vec<f32>> {
        assert!(!ids.is_empty(), "forward: a span has one position or more");
        let t = ids.len();
        let p0 = ctx.position;
        let mut x = self.embed(ids);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut run = 0;
        let mut step = |x: &mut Vec<f32>,
                        blocks: &mut Vec<Vec<f32>>,
                        messages: Option<&[Vec<f32>; 4]>,
                        ctx: &mut Context,
                        trace: &mut Option<&mut Trace>| {
            let state = &mut ctx.runs[run];
            let layer = &self.weights.layers[state.layer];
            let out = self.layer(layer, state, x, blocks, messages, p0, trace);
            // A block boundary keeps the input of the layer as a block output (line 1374).
            if layer.index.is_multiple_of(BLOCK) {
                blocks.push(std::mem::take(x));
            }
            if let Some(tr) = trace.as_deref_mut() {
                tr.runs.push(out.clone());
            }
            *x = out;
            run += 1;
        };
        for _ in 0..CYCLE_START {
            step(&mut x, &mut blocks, None, ctx, &mut trace);
        }
        let (pre_x, pre_blocks) = (x.clone(), blocks.clone());
        for _ in CYCLE_START..CYCLE_END {
            step(&mut x, &mut blocks, None, ctx, &mut trace);
        }
        for _cycle in 1..ctx.cycles {
            // The message of the cycle before: its block output after layer 11 and its row after layer 17 (line 1400).
            let previous_blocks = blocks[pre_blocks.len() + 1..].to_vec();
            let previous_x = std::mem::replace(&mut x, pre_x.clone());
            blocks = pre_blocks.clone();
            for li in 0..CYCLE_END - CYCLE_START {
                let messages = self.messages(&previous_blocks, &previous_x, li);
                if let Some(tr) = trace.as_deref_mut() {
                    let mut flat = Vec::with_capacity(4 * t * HIDDEN);
                    for p in 0..t {
                        for m in &messages {
                            flat.extend_from_slice(row(m, p, HIDDEN));
                        }
                    }
                    tr.messages.push(flat);
                }
                step(&mut x, &mut blocks, Some(&messages), ctx, &mut trace);
            }
        }
        for _ in CYCLE_END..LAYERS {
            step(&mut x, &mut blocks, None, ctx, &mut trace);
        }
        assert_eq!(
            run,
            ctx.runs.len(),
            "forward: {run} runs for a context of {} runs",
            ctx.runs.len()
        );
        ctx.position += t;
        // The output residual and the final norm (lines 1442 and 1443).
        let w = &self.weights;
        let mut final_norm = vec![0f32; t * HIDDEN];
        let mut mixed = vec![Vec::new()];
        for p in 0..t {
            let blk: Vec<&[f32]> = blocks.iter().map(|b| row(b, p, HIDDEN)).collect();
            ops::attention_residual(&blk, row(&x, p, HIDDEN), &[&w.output_query], &mut mixed);
            ops::rms_norm(&mixed[0], &w.norm, &mut final_norm[p * HIDDEN..(p + 1) * HIDDEN]);
        }
        let logits = logit_rows
            .iter()
            .map(|r| {
                let mut out = linear(w, &w.unembed, row(&final_norm, *r, HIDDEN));
                for v in &mut out {
                    *v = ops::softcap(*v, ops::LOGIT_SOFTCAP);
                }
                out
            })
            .collect();
        if let Some(tr) = trace {
            tr.final_norm = final_norm;
        }
        logits
    }

    /// The four messages of cycle layer `li`, each `T x HIDDEN` (`Inkling2LoopMessageAttentionResidual`, line 158).
    fn messages(&self, previous_blocks: &[Vec<f32>], previous_x: &[f32], li: usize) -> [Vec<f32>; 4] {
        let t = previous_x.len() / HIDDEN;
        let q = &self.weights.message_queries[li];
        let queries: Vec<&[f32]> = q.iter().map(|v| v.as_slice()).collect();
        let mut out: [Vec<f32>; 4] = Default::default();
        let mut mixed = vec![Vec::new(); 4];
        for p in 0..t {
            let blk: Vec<&[f32]> = previous_blocks.iter().map(|b| row(b, p, HIDDEN)).collect();
            ops::attention_residual(&blk, row(previous_x, p, HIDDEN), &queries, &mut mixed);
            for (o, m) in out.iter_mut().zip(&mixed) {
                o.extend_from_slice(m);
            }
        }
        out
    }

    /// One run of one layer over a span (the decoder layers, lines 856 to 1149).
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        w: &Layer,
        state: &mut RunState,
        x: &[f32],
        blocks: &[Vec<f32>],
        messages: Option<&[Vec<f32>; 4]>,
        p0: usize,
        trace: &mut Option<&mut Trace>,
    ) -> Vec<f32> {
        let t = x.len() / HIDDEN;
        let block_rows = |p: usize| -> Vec<&[f32]> { blocks.iter().map(|b| row(b, p, HIDDEN)).collect() };
        // The inputs of the query, the key and the value.
        let mut inputs: [Vec<f32>; 3] = [vec![0f32; t * HIDDEN], vec![0f32; t * HIDDEN], vec![0f32; t * HIDDEN]];
        let mut gated: [Vec<f32>; 3] = Default::default();
        let queries: Vec<&[f32]> = w.attn_queries.iter().map(|q| q.as_slice()).collect();
        let mut mixed = vec![Vec::new(); queries.len()];
        let mut g = vec![0f32; HIDDEN];
        for p in 0..t {
            let xp = row(x, p, HIDDEN);
            let parts: [&[f32]; 3] = if queries.is_empty() {
                [xp, xp, xp]
            } else {
                ops::attention_residual(&block_rows(p), xp, &queries, &mut mixed);
                if w.full {
                    [&mixed[0], &mixed[1], &mixed[2]]
                } else {
                    [&mixed[0], &mixed[0], &mixed[0]]
                }
            };
            for (i, part) in parts.iter().enumerate() {
                let src: &[f32] = match messages {
                    Some(m) => {
                        let gw = &w.gates[i];
                        ops::message_gate(part, row(&m[i], p, HIDDEN), &gw.down, &gw.forget, &gw.gain, &mut g);
                        gated[i].extend_from_slice(&g);
                        &g
                    }
                    None => part,
                };
                let norm = if w.full { &w.input_norms[i] } else { &w.input_norms[0] };
                ops::rms_norm(src, norm, &mut inputs[i][p * HIDDEN..(p + 1) * HIDDEN]);
            }
        }
        if messages.is_some()
            && let Some(tr) = trace.as_deref_mut()
        {
            tr.gates.extend(gated);
        }
        let attn = self.attention(w, state, &inputs, p0);
        // The post-attention convolution (line 899).
        let mut residual = vec![0f32; t * HIDDEN];
        for p in 0..t {
            let o = &mut residual[p * HIDDEN..(p + 1) * HIDDEN];
            ops::short_conv(&state.conv[2], row(&attn, p, HIDDEN), &w.attn_conv, o);
            ops::conv_push(&mut state.conv[2], row(&attn, p, HIDDEN));
        }
        // At a block boundary the residual restarts from the attention output, and the input becomes a block
        // output (lines 901 to 905).
        let boundary = w.index.is_multiple_of(BLOCK);
        let mut local: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
        if boundary {
            local.push(x);
        } else {
            for (r, v) in residual.iter_mut().zip(x) {
                *r += v;
            }
        }
        let mut mlp_in = vec![0f32; t * HIDDEN];
        let mut mixed = vec![Vec::new()];
        let mut mlp_gated = Vec::new();
        for p in 0..t {
            let blk: Vec<&[f32]> = local.iter().map(|b| row(b, p, HIDDEN)).collect();
            ops::attention_residual(&blk, row(&residual, p, HIDDEN), &[&w.mlp_query], &mut mixed);
            let src: &[f32] = match messages {
                Some(m) => {
                    let gw = &w.gates[3];
                    ops::message_gate(&mixed[0], row(&m[3], p, HIDDEN), &gw.down, &gw.forget, &gw.gain, &mut g);
                    mlp_gated.extend_from_slice(&g);
                    &g
                }
                None => &mixed[0],
            };
            ops::rms_norm(src, &w.mlp_norm, &mut mlp_in[p * HIDDEN..(p + 1) * HIDDEN]);
        }
        if messages.is_some()
            && let Some(tr) = trace.as_deref_mut()
        {
            tr.gates.push(mlp_gated);
        }
        let moe = self.experts(w, &mlp_in);
        let mut out = vec![0f32; t * HIDDEN];
        let mut c = vec![0f32; HIDDEN];
        for p in 0..t {
            ops::short_conv(&state.conv[3], row(&moe, p, HIDDEN), &w.mlp_conv, &mut c);
            ops::conv_push(&mut state.conv[3], row(&moe, p, HIDDEN));
            for ((o, r), v) in out[p * HIDDEN..(p + 1) * HIDDEN]
                .iter_mut()
                .zip(row(&residual, p, HIDDEN))
                .zip(&c)
            {
                *o = r + v;
            }
        }
        out
    }

    /// The attention of one run over a span: the projections, the convolutions, the clips, the relative bias, the
    /// sinks, the output gate and the output projection (lines 598 to 636 and 691 to 727).
    fn attention(&self, w: &Layer, state: &mut RunState, inputs: &[Vec<f32>; 3], p0: usize) -> Vec<f32> {
        let wt = &self.weights;
        let t = inputs[0].len() / HIDDEN;
        let qd = HEADS * HEAD_DIM;
        let (q_in, k_in, v_in) = (&inputs[0], &inputs[1], &inputs[2]);
        let mut q = linear(wt, &w.wq, q_in);
        let mut rel = linear(wt, &w.wr, q_in);
        let sinks = linear(wt, &w.sink, q_in);
        let gate = linear(wt, &w.gate_attn, q_in);
        let mut profiles = vec![0f32; t * HEADS * REL_EXTENT];
        for p in 0..t {
            let qp = &mut q[p * qd..(p + 1) * qd];
            ops::scaled_l2_clip(qp, &w.q_scale, HEAD_DIM, ops::QK_CLIP);
            let rp = &mut rel[p * HEADS * ops::D_REL..(p + 1) * HEADS * ops::D_REL];
            ops::scaled_l2_clip(rp, &w.rel_scale, ops::D_REL, ops::REL_CLIP);
            if w.full {
                let s = ops::log_scale(p0 + p);
                qp.iter_mut().chain(rp.iter_mut()).for_each(|v| *v *= s);
            }
            ops::relative_logits(
                rp,
                &w.rel_proj,
                &mut profiles[p * HEADS * REL_EXTENT..(p + 1) * HEADS * REL_EXTENT],
            );
        }
        let keys = self.kv(w, &w.k_conv, &mut state.conv[0], k_in, true);
        let values = self.kv(w, &w.v_conv, &mut state.conv[1], v_in, false);
        // The held rows, then the rows of the span, at their positions.
        let held = state.keys.len();
        let first = state.keys.first;
        assert_eq!(
            first + held,
            p0,
            "attention: the held rows end at position {}, not at {p0}",
            first + held
        );
        let all_k: Vec<f32> = state.keys.rows.iter().chain(&keys).copied().collect();
        let all_v: Vec<f32> = state.values.rows.iter().chain(&values).copied().collect();
        let window = state.keys.window;
        let mut out = vec![0f32; t * qd];
        out.par_chunks_mut(HEAD_DIM).enumerate().for_each(|(i, o)| {
            let (p, h) = (i / HEADS, i % HEADS);
            let pos = p0 + p;
            let kvh = h / (HEADS / KV_HEADS);
            let profile = &profiles[(p * HEADS + h) * REL_EXTENT..(p * HEADS + h + 1) * REL_EXTENT];
            let lowest = visible_from(pos, first, window);
            let mut ks = Vec::new();
            let mut vs = Vec::new();
            let mut bias = Vec::new();
            for j in lowest..=pos {
                let at = (j - first) * KV_WIDTH + kvh * HEAD_DIM;
                ks.push(&all_k[at..at + HEAD_DIM]);
                vs.push(&all_v[at..at + HEAD_DIM]);
                bias.push(ops::relative_bias(profile, (pos - j) as i64));
            }
            let qh = &q[p * qd + h * HEAD_DIM..p * qd + (h + 1) * HEAD_DIM];
            ops::attend(qh, &ks, &vs, &bias, sinks[p * HEADS + h], o);
            // The output gate: a sigmoid of the query input (line 541).
            for (v, g) in o.iter_mut().zip(&gate[p * qd + h * HEAD_DIM..]) {
                *v *= ops::sigmoid(*g);
            }
        });
        state.keys.push(&keys);
        state.values.push(&values);
        linear(wt, &w.wo, &out)
    }

    /// The keys or the values of a span: the projection, the convolution, then the clip of a key or the norm of a
    /// value for each head (`Inkling2LoopKeyValueProjection`, lines 523 to 528).
    fn kv(&self, w: &Layer, conv: &KvConv, conv_state: &mut [f32], input: &[f32], key: bool) -> Vec<f32> {
        let t = input.len() / HIDDEN;
        let m = if key { &w.wk } else { &w.wv };
        let lin = linear(&self.weights, m, input);
        let mut out = vec![0f32; t * KV_WIDTH];
        let mut taps = vec![0f32; KV_HEADS * CONV_TAPS];
        for p in 0..t {
            let xp = row(&lin, p, KV_WIDTH);
            let o = &mut out[p * KV_WIDTH..(p + 1) * KV_WIDTH];
            match conv {
                KvConv::Fixed(weight) => ops::short_conv(conv_state, xp, weight, o),
                KvConv::Dynamic { static_w, proj } => {
                    ops::matvec(proj, row(input, p, HIDDEN), &mut taps);
                    ops::dynamic_conv(conv_state, xp, static_w, &taps, o);
                }
            }
            ops::conv_push(conv_state, xp);
            for head in o.as_chunks_mut::<HEAD_DIM>().0 {
                if key {
                    ops::l2_clip(head, ops::QK_CLIP);
                } else {
                    ops::rms_norm_unweighted(head);
                }
            }
        }
        out
    }

    /// The routed experts of a span (`Inkling2LoopSparseMoeBlock`, line 820). The sum of the experts is in
    /// ascending expert order, as `index_add_` of the reference.
    fn experts(&self, w: &Layer, xs: &[f32]) -> Vec<f32> {
        let t = xs.len() / HIDDEN;
        let mut assigned: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ops::EXPERTS];
        let mut logits = vec![0f32; ops::EXPERTS];
        for p in 0..t {
            ops::matvec(&w.router, row(xs, p, HIDDEN), &mut logits);
            let chosen: [(usize, f32); TOP_K] = ops::route(&logits, &w.router_bias);
            for (e, weight) in chosen {
                assigned[e].push((p, weight));
            }
        }
        let hit: Vec<usize> = (0..ops::EXPERTS).filter(|e| !assigned[*e].is_empty()).collect();
        let parts: Vec<Vec<f32>> = hit
            .par_iter()
            .map(|e| {
                let list = &assigned[*e];
                let input: Vec<f32> = list
                    .iter()
                    .flat_map(|(p, _)| row(xs, *p, HIDDEN).iter().copied())
                    .collect();
                let both = self.expert_rows(w.index, *e, false, &input, HIDDEN);
                let n = list.len();
                let inner = ops::INTERMEDIATE;
                let mut act = vec![0f32; n * inner];
                for i in 0..n {
                    let b = &both[i * 2 * inner..(i + 1) * 2 * inner];
                    ops::swiglu(&b[..inner], &b[inner..], &mut act[i * inner..(i + 1) * inner]);
                }
                let mut down = self.expert_rows(w.index, *e, true, &act, inner);
                for (i, (_, weight)) in list.iter().enumerate() {
                    for v in &mut down[i * HIDDEN..(i + 1) * HIDDEN] {
                        *v *= weight;
                    }
                }
                down
            })
            .collect();
        let mut out = vec![0f32; t * HIDDEN];
        for (e, part) in hit.iter().zip(&parts) {
            for (i, (p, _)) in assigned[*e].iter().enumerate() {
                for (o, v) in out[p * HIDDEN..(p + 1) * HIDDEN].iter_mut().zip(row(part, i, HIDDEN)) {
                    *o += v;
                }
            }
        }
        out
    }

    /// `xs M^T` for one expert matrix, `n x rows` from `n x cols`. Each row is expanded once for all `n` inputs.
    fn expert_rows(&self, layer: usize, expert: usize, down: bool, xs: &[f32], cols: usize) -> Vec<f32> {
        let rows = if down { HIDDEN } else { 2 * ops::INTERMEDIATE };
        let n = xs.len() / cols;
        let chunk = 64;
        let mut transposed = vec![0f32; rows * n];
        transposed.par_chunks_mut(chunk * n).enumerate().for_each(|(c, out)| {
            let mut buf = vec![0f32; cols];
            for (i, o) in out.chunks_exact_mut(n).enumerate() {
                self.weights.expert_row(layer, expert, down, c * chunk + i, &mut buf);
                for (t, v) in o.iter_mut().enumerate() {
                    *v = ops::dot(&buf, &xs[t * cols..(t + 1) * cols]);
                }
            }
        });
        transpose(&transposed, rows, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sliding_run_holds_the_last_rows_of_its_window() {
        let mut rows = KvRows::new(Some(3));
        let span: Vec<f32> = (0..5 * KV_WIDTH).map(|i| (i / KV_WIDTH) as f32).collect();
        rows.push(&span[..2 * KV_WIDTH]);
        assert_eq!((rows.first, rows.len()), (0, 2));
        rows.push(&span[2 * KV_WIDTH..]);
        assert_eq!((rows.first, rows.len()), (2, 3));
        assert_eq!(rows.rows[0], 2.0);
        let mut full = KvRows::new(None);
        full.push(&span);
        assert_eq!((full.first, full.len()), (0, 5));
    }

    #[test]
    fn a_sliding_query_reads_the_last_1024_positions() {
        assert_eq!(visible_from(5, 0, Some(REL_EXTENT)), 0);
        assert_eq!(visible_from(1023, 0, Some(REL_EXTENT)), 0);
        assert_eq!(visible_from(1024, 0, Some(REL_EXTENT)), 1);
        assert_eq!(visible_from(2000, 977, Some(REL_EXTENT)), 977);
        assert_eq!(visible_from(2000, 0, None), 0);
        // A run that dropped a key that the query reads is a defect, not a shorter window.
        assert!(std::panic::catch_unwind(|| visible_from(2000, 978, Some(REL_EXTENT))).is_err());
    }

    #[test]
    fn the_runs_go_through_the_cycles_in_order() {
        let three = run_layers(3);
        assert_eq!(three.len(), 48);
        assert_eq!(&three[..7], &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(&three[17..19], &[17, 6]);
        assert_eq!(&three[41..], &[17, 18, 19, 20, 21, 22, 23]);
        assert_eq!(run_layers(1).len(), 24);
        assert_eq!(run_layers(5).len(), 72);
    }
}
