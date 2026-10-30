//! The graph of the text tower as one ring of tasks, and the device buffers that it uses.
//!
//! The ring follows `EmbeddingGemma2TextModel.forward` of transformers 5.21.0 for one input of `n` tokens: the scaled
//! embeddings, the per-layer inputs, 24 layers, the final norm and `embedding_projection`. The mean over the positions
//! runs on the host (engine-abi "Embedding").

use nagaya_mk_bmg::protocol::{FLAG_ACCUMULATE, MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::tasks;

use crate::spec::*;
use crate::weights::{Weights, rounded};

/// The activation buffers of one input of at most `max_tokens` tokens. Each row is one token.
#[derive(Clone, Copy, Debug)]
pub struct Bufs {
    /// The ids, as `u32`.
    pub ids: u64,
    /// The residual stream: `[n][512]`.
    pub x: u64,
    /// The normed rows, and the gate of the per-layer input: `[n][512]`.
    pub h: u64,
    /// The query rows: `[n][4 * 512]` at most.
    pub q: u64,
    pub k: u64,
    pub v: u64,
    /// The output of the attention: `[n][4 * 512]` at most.
    pub att: u64,
    /// The output of a projection before its norm: `[n][512]`.
    pub o: u64,
    /// The GeGLU rows of the MLP: `[n][2048]`.
    pub g: u64,
    /// The per-layer inputs of every layer: `[n][24][512]`.
    pub ple: u64,
    /// The last output: `[n][768]`.
    pub out: u64,
}

impl Bufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, max_tokens: usize) -> Self {
        let f = |values: usize| values * max_tokens * 4;
        let wide = HEADS * FULL.head_dim.max(SLIDING.head_dim);
        let kv = (FULL.kv_heads * FULL.head_dim).max(SLIDING.kv_heads * SLIDING.head_dim);
        Self {
            ids: z(max_tokens * 4),
            x: z(f(HIDDEN)),
            h: z(f(HIDDEN)),
            q: z(f(wide)),
            k: z(f(kv)),
            v: z(f(kv)),
            att: z(f(wide)),
            o: z(f(HIDDEN)),
            g: z(f(INTERMEDIATE)),
            ple: z(f(LAYERS * PER_LAYER)),
            out: z(f(DIMENSION)),
        }
    }
}

/// The device bytes of the buffers of `max_tokens` tokens, with no card: the plan (spec 2.4).
pub fn bufs_bytes(max_tokens: usize) -> usize {
    let mut total = 0usize;
    Bufs::new(
        &mut |bytes| {
            total += rounded(bytes);
            0
        },
        max_tokens,
    );
    total
}

fn accumulate(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_ACCUMULATE;
    task
}

/// The tasks of one input of `n` tokens, whose ids are in `b.ids`. The last output goes to `b.out`.
pub fn pass(w: &Weights, b: &Bufs, n: usize) -> Vec<MkTaskDescriptor> {
    pass_until(w, b, n, LAYERS, true)
}

/// The tasks of one input up to the end of `layers` layers, with the final norm and the projection when `tail`. Then
/// `b.x` holds the residual stream after those layers. A probe of the oracle stops early this way.
pub fn pass_until(w: &Weights, b: &Bufs, n: usize, layers: usize, tail: bool) -> Vec<MkTaskDescriptor> {
    assert!(
        layers <= LAYERS && (layers == LAYERS || !tail),
        "a pass ends after {layers} layers with no tail"
    );
    assert!(
        (1..=MAX_TOKENS).contains(&n),
        "a pass of {n} tokens is outside 1..={MAX_TOKENS}; the engine checks the input first"
    );
    let n32 = n as u32;
    let h32 = HIDDEN as u32;
    let mut t = vec![
        tasks::embed(b.ids, w.embed, b.x, n32, h32, 0, false, false),
        tasks::scale(b.x, b.x, n32, h32, embed_scale()),
        // The per-layer inputs come from the scaled embeddings only.
        tasks::matmul(b.x, w.ple_proj, b.ple, n32, (LAYERS * PER_LAYER) as u32, h32, false),
        tasks::scale(b.ple, b.ple, n32, (LAYERS * PER_LAYER) as u32, per_layer_scale()),
        tasks::rmsnorm(
            b.ple,
            w.ple_norm,
            b.ple,
            (n * LAYERS) as u32,
            PER_LAYER as u32,
            EPS,
            false,
        ),
    ];
    for (l, lw) in w.layers.iter().enumerate().take(layers) {
        let a = attn(l);
        let (d, kv) = (a.head_dim as u32, a.kv_heads as u32);
        let heads = HEADS as u32;
        let full = a == FULL;
        t.push(tasks::rmsnorm(b.x, lw.input_norm, b.h, n32, h32, EPS, false));
        t.push(tasks::matmul(b.h, lw.q, b.q, n32, heads * d, h32, false));
        t.push(tasks::matmul(b.h, lw.k, b.k, n32, kv * d, h32, false));
        t.push(tasks::matmul(b.h, lw.v, b.v, n32, kv * d, h32, false));
        t.push(tasks::qk_rope(
            b.q,
            lw.q_norm,
            n32 * heads,
            heads,
            d,
            0,
            full,
            true,
            EPS,
            1.0,
        ));
        t.push(tasks::qk_rope(b.k, lw.k_norm, n32 * kv, kv, d, 0, full, true, EPS, 1.0));
        // The value norm has no scale.
        t.push(tasks::rmsnorm(b.v, 0, b.v, n32 * kv, d, EPS, false));
        t.push(tasks::biattn(
            b.q,
            b.k,
            b.v,
            b.att,
            n32,
            heads,
            kv,
            d,
            a.window as u32,
            1.0,
        ));
        t.push(tasks::matmul(b.att, lw.o, b.o, n32, h32, heads * d, false));
        t.push(accumulate(tasks::rmsnorm(
            b.o,
            lw.post_attn_norm,
            b.x,
            n32,
            h32,
            EPS,
            false,
        )));
        t.push(tasks::rmsnorm(b.x, lw.pre_ff_norm, b.h, n32, h32, EPS, false));
        t.push(tasks::geglu(b.h, lw.gate_up, b.g, n32, INTERMEDIATE as u32, h32));
        t.push(tasks::matmul(b.g, lw.down, b.o, n32, h32, INTERMEDIATE as u32, false));
        t.push(accumulate(tasks::rmsnorm(
            b.o,
            lw.post_ff_norm,
            b.x,
            n32,
            h32,
            EPS,
            false,
        )));
        // The third block: the gate against this layer's slice of the per-layer inputs.
        t.push(tasks::matmul(b.x, lw.ple_gate, b.h, n32, PER_LAYER as u32, h32, false));
        let slice = b.ple + (l * PER_LAYER * 4) as u64;
        t.push(tasks::gelu_mul(
            b.h,
            slice,
            b.h,
            n32,
            PER_LAYER as u32,
            (LAYERS * PER_LAYER) as u32,
        ));
        t.push(tasks::matmul(b.h, lw.ple_proj, b.o, n32, h32, PER_LAYER as u32, false));
        t.push(accumulate(tasks::rmsnorm(b.o, lw.ple_norm, b.x, n32, h32, EPS, false)));
        t.push(tasks::scale(b.x, b.x, n32, h32, lw.scalar));
    }
    if tail {
        t.push(tasks::rmsnorm(b.x, w.norm, b.h, n32, h32, EPS, false));
        t.push(tasks::matmul(
            b.h,
            w.projection,
            b.out,
            n32,
            DIMENSION as u32,
            h32,
            false,
        ));
    }
    assert!(t.len() < MAX_TASKS_PER_CARD, "the pass has {} tasks", t.len());
    t
}

/// Why the engine refuses `ids`, or `None` when it can run them.
pub fn ids_problem(ids: &[u32]) -> Option<String> {
    if ids.is_empty() {
        Some("the input has no ids; give at least one".to_owned())
    } else if ids.len() > MAX_TOKENS {
        Some(format!(
            "the input has {} ids, and the engine takes at most {MAX_TOKENS}; split the input",
            ids.len()
        ))
    } else {
        ids.iter()
            .find(|id| **id as usize >= VOCAB)
            .map(|id| format!("the id {id} is outside the vocabulary of {VOCAB} ids"))
    }
}

/// The mean of `n` rows of `DIMENSION` values: each column summed in f64 in row order, then divided and rounded to
/// f32. The host computes it, so the order of the sum is fixed (engine-abi "Embedding", "Determinism").
pub fn mean(rows: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(rows.len(), n * DIMENSION, "the last output has {} values", rows.len());
    (0..DIMENSION)
        .map(|j| ((0..n).map(|t| rows[t * DIMENSION + j] as f64).sum::<f64>() / n as f64) as f32)
        .collect()
}
