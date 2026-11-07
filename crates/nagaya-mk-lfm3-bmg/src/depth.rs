//! The depth decoder of Omni on the card: its weights, its buffers, and the ring of one pass (milestone 24 step 8).
//!
//! The ring follows `Lfm3OmniDepthDecoder.forward` (modeling_lfm3_omni.py:254-281) for one frame. `depth_linear`
//! projects the hidden row of the backbone to one row of 1,024 values for each codebook. Each row adds the code
//! embedding of a revealed codebook or the mask embedding. Then 3 full attention layers in both directions run over
//! the 12 rows, with no sandwich norms (modeling_lfm3.py:445-498), and the final norm. Each codebook that the pass
//! picks gets the norm of its table and its logits from the same table (modeling_lfm3_omni.py:215-227).

use nagaya_mk_bmg::protocol::{MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView};
use nagaya_mk_bmg::tasks;

use crate::codes::{CODEBOOKS, CODES};
use crate::weights::{Upload, rounded};

/// The width of the trunk (`depth_config.hidden_size`).
pub const HIDDEN: usize = 1024;
pub const HEADS: usize = 32;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 32;
pub const INTERMEDIATE: usize = 2816;
pub const LAYERS: usize = 3;
/// The eps of each norm (`depth_config.norm_eps`).
pub const EPS: f32 = 1e-5;
/// The width of the hidden row of the backbone.
pub const BACKBONE: usize = 2048;
/// The row of the code table that holds the mask embedding: after the 2,017 rows of each of the 12 codebooks.
pub const MASK_ROW: u32 = (CODEBOOKS * CODES) as u32;

/// The prefix of the tensors of the depth decoder in `model.safetensors` of Omni.
pub const PREFIX: &str = "model.depth_decoder.";
/// The table of the codebooks of the backbone: a fed frame is the sum of 12 of its rows.
pub const CODEBOOK_TABLE: &str = "model.codebook_embedding.weight";

/// The weights of one layer of the trunk.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layer {
    pub operator_norm: u64,
    pub ffn_norm: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub gate: u64,
    /// The gate rows (`w1`), then the up rows (`w3`).
    pub w13: u64,
    pub w2: u64,
}

/// The weights of the depth decoder on the card, and the codebook table of the backbone.
#[derive(Clone, Debug)]
pub struct Weights {
    /// `depth_linear`: `[12 * 1024][2048]`, and its bias.
    pub linear: u64,
    pub linear_bias: u64,
    /// The table of each codebook, one after the other, then the mask embedding: `[12 * 2017 + 1][1024]`. The table
    /// of codebook `g` is also its head.
    pub codes: u64,
    /// The norm of each codebook before its head: `[12][1024]`.
    pub norms: u64,
    pub layers: Vec<Layer>,
    /// The final norm of the trunk.
    pub norm: u64,
    /// The codebook table of the backbone: `[12 * 2017][2048]`.
    pub codebook: u64,
}

impl Weights {
    /// The address of the table of codebook `g`.
    pub fn table(&self, g: usize) -> u64 {
        self.codes + (g * CODES * HIDDEN * 2) as u64
    }
}

/// The tensors of the depth decoder and the codebook table, with their shapes, by their whole names.
pub fn tensors() -> Vec<(String, Vec<usize>)> {
    let p = PREFIX;
    let (h, i) = (HIDDEN, INTERMEDIATE);
    let mut all = vec![
        (format!("{p}depth_linear.weight"), vec![CODEBOOKS * h, BACKBONE]),
        (format!("{p}depth_linear.bias"), vec![CODEBOOKS * h]),
        (format!("{p}mask_embedding"), vec![h]),
        (format!("{p}model.embedding_norm.weight"), vec![h]),
        (CODEBOOK_TABLE.to_owned(), vec![CODEBOOKS * CODES, BACKBONE]),
    ];
    for g in 0..CODEBOOKS {
        all.push((format!("{p}depth_embeddings.{g}.embedding.weight"), vec![CODES, h]));
        all.push((format!("{p}depth_embeddings.{g}.embedding_norm.weight"), vec![h]));
    }
    for l in 0..LAYERS {
        for (name, shape) in [
            ("operator_norm.weight", vec![h]),
            ("ffn_norm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, h]),
            ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, h]),
            ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, h]),
            ("self_attn.o_proj.weight", vec![h, HEADS * HEAD_DIM]),
            ("self_attn.q_norm.weight", vec![HEAD_DIM]),
            ("self_attn.k_norm.weight", vec![HEAD_DIM]),
            ("self_attn.gate_proj.weight", vec![HEADS, h]),
            ("feed_forward.w1.weight", vec![i, h]),
            ("feed_forward.w3.weight", vec![i, h]),
            ("feed_forward.w2.weight", vec![h, i]),
        ] {
            all.push((format!("{p}model.layers.{l}.{name}"), shape));
        }
    }
    all
}

/// Check that `view` holds each tensor of the depth decoder as BF16 with its shape. The message names the tensor.
pub fn check(view: &SafeTensorsView<'_>) -> Result<(), String> {
    for (name, shape) in tensors() {
        let meta = view
            .tensors
            .get(&name)
            .ok_or_else(|| format!("the checkpoint of Omni has no tensor {name}"))?;
        if meta.dtype != Dtype::BF16 || meta.shape != shape {
            return Err(format!(
                "tensor {name} is {:?} {:?}, and the depth decoder needs BF16 {shape:?}",
                meta.dtype, meta.shape
            ));
        }
    }
    let known: Vec<String> = tensors().into_iter().map(|(n, _)| n).collect();
    let mut other: Vec<&String> = view
        .tensors
        .keys()
        .filter(|name| name.starts_with(PREFIX) && !known.contains(name))
        .collect();
    other.sort();
    match other.first() {
        Some(name) => Err(format!(
            "the checkpoint has the tensor {name}, which the depth decoder does not use"
        )),
        None => Ok(()),
    }
}

/// Load the depth decoder of `view` into `sink`. A checkpoint that fails `check` uploads nothing.
pub fn load(sink: &mut dyn Upload, view: &SafeTensorsView<'_>) -> Result<Weights, String> {
    check(view)?;
    let bytes = |name: &str| -> &[u8] { view.tensor_bytes(name).expect("check found each tensor") };
    let mut put = |names: &[String]| -> u64 {
        let parts: Vec<&[u8]> = names.iter().map(|n| bytes(n)).collect();
        let at = sink.alloc(parts.iter().map(|p| p.len()).sum());
        if sink.writes() {
            let mut offset = 0u64;
            for part in parts {
                sink.write(at + offset, part);
                offset += part.len() as u64;
            }
        }
        at
    };
    let p = PREFIX;
    let linear = put(&[format!("{p}depth_linear.weight")]);
    let linear_bias = put(&[format!("{p}depth_linear.bias")]);
    let mut table: Vec<String> = (0..CODEBOOKS)
        .map(|g| format!("{p}depth_embeddings.{g}.embedding.weight"))
        .collect();
    table.push(format!("{p}mask_embedding"));
    let codes = put(&table);
    let norms: Vec<String> = (0..CODEBOOKS)
        .map(|g| format!("{p}depth_embeddings.{g}.embedding_norm.weight"))
        .collect();
    let norms = put(&norms);
    let mut layers = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let n = |name: &str| format!("{p}model.layers.{l}.{name}");
        layers.push(Layer {
            operator_norm: put(&[n("operator_norm.weight")]),
            ffn_norm: put(&[n("ffn_norm.weight")]),
            q: put(&[n("self_attn.q_proj.weight")]),
            k: put(&[n("self_attn.k_proj.weight")]),
            v: put(&[n("self_attn.v_proj.weight")]),
            o: put(&[n("self_attn.o_proj.weight")]),
            q_norm: put(&[n("self_attn.q_norm.weight")]),
            k_norm: put(&[n("self_attn.k_norm.weight")]),
            gate: put(&[n("self_attn.gate_proj.weight")]),
            w13: put(&[n("feed_forward.w1.weight"), n("feed_forward.w3.weight")]),
            w2: put(&[n("feed_forward.w2.weight")]),
        });
    }
    let norm = put(&[format!("{p}model.embedding_norm.weight")]);
    let codebook = put(&[CODEBOOK_TABLE.to_owned()]);
    Ok(Weights {
        linear,
        linear_bias,
        codes,
        norms,
        layers,
        norm,
        codebook,
    })
}

/// The buffers of one pass of the depth decoder. Each row is one codebook.
#[derive(Clone, Copy, Debug)]
pub struct Bufs {
    /// The row of the code table for each codebook, as `u32`: `g * 2017 + code`, or `MASK_ROW`.
    pub ids: u64,
    /// The residual stream: `[12][1024]`.
    pub x: u64,
    /// The normed rows: `[12][1024]`.
    pub h: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    /// The gate of each head: `[12][32]`.
    pub gates: u64,
    /// The output of the attention: `[12][1024]`.
    pub y: u64,
    /// The SwiGLU rows: `[12][2816]`.
    pub g: u64,
    /// The norm of each codebook before its head: `[12][1024]`.
    pub normed: u64,
    /// The logits of each picked codebook: `[12][2017]`, in the order of the picks.
    pub logits: u64,
    /// The rows of a fed frame: the row of the codebook table for each codebook, as `u32`.
    pub frame_ids: u64,
    /// A hidden row of the backbone that a probe of the oracle writes: `[2048]`.
    pub probe: u64,
}

impl Bufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64) -> Self {
        let rows = |values: usize| CODEBOOKS * values * 4;
        Self {
            ids: z(CODEBOOKS * 4),
            x: z(rows(HIDDEN)),
            h: z(rows(HIDDEN)),
            q: z(rows(HEADS * HEAD_DIM)),
            k: z(rows(KV_HEADS * HEAD_DIM)),
            v: z(rows(KV_HEADS * HEAD_DIM)),
            gates: z(rows(HEADS)),
            y: z(rows(HEADS * HEAD_DIM)),
            g: z(rows(INTERMEDIATE)),
            normed: z(rows(HIDDEN)),
            logits: z(rows(CODES)),
            frame_ids: z(CODEBOOKS * 4),
            probe: z(BACKBONE * 4),
        }
    }

    /// The device bytes of the buffers, with no card.
    pub fn bytes() -> usize {
        let mut total = 0usize;
        Self::new(&mut |bytes| {
            total += rounded(bytes);
            0
        });
        total
    }
}

/// The rows of the code table for one pass: the code of each revealed codebook, and the mask row for the others.
pub fn ids(codes: &[u32], revealed: &[bool]) -> Vec<u32> {
    assert_eq!(codes.len(), CODEBOOKS, "a frame has 12 codes");
    assert_eq!(revealed.len(), CODEBOOKS, "a frame has 12 codebooks");
    (0..CODEBOOKS)
        .map(|g| {
            if revealed[g] {
                assert!((codes[g] as usize) < CODES, "code {} is past the codebook", codes[g]);
                (g * CODES) as u32 + codes[g]
            } else {
                MASK_ROW
            }
        })
        .collect()
}

/// The rows of the codebook table of the backbone for a fed frame: `g * 2017 + code` for each codebook.
pub fn frame_ids(codes: &[u32]) -> Vec<u32> {
    assert_eq!(codes.len(), CODEBOOKS, "a frame has 12 codes");
    codes
        .iter()
        .enumerate()
        .map(|(g, code)| {
            assert!((*code as usize) < CODES, "code {code} is past the codebook");
            (g * CODES) as u32 + code
        })
        .collect()
}

/// The tasks of one pass of the depth decoder on the hidden row at `hidden`. The rows of the code table are in
/// `bufs.ids`. After the pass, `bufs.x` holds the input of the trunk until the first layer, then the trunk output is
/// in `bufs.h`, and row `i` of `bufs.logits` holds the logits of codebook `picks[i]`.
pub fn pass(w: &Weights, bufs: &Bufs, hidden: u64, picks: &[usize]) -> Vec<MkTaskDescriptor> {
    assert!(picks.len() <= CODEBOOKS, "a pass picks at most 12 codebooks");
    let (n, h) = (CODEBOOKS as u32, HIDDEN as u32);
    let mut t = vec![
        tasks::matmul(hidden, w.linear, bufs.x, 1, n * h, BACKBONE as u32, false),
        tasks::bias(w.linear_bias, bufs.x, 1, n * h, false),
        tasks::embed(bufs.ids, w.codes, bufs.x, n, h, 0, false, true),
    ];
    t.extend(trunk(w, bufs));
    for (row, g) in picks.iter().enumerate() {
        assert!(*g < CODEBOOKS, "codebook {g} does not exist");
        let at = |base: u64, values: usize, r: usize| base + (r * values * 4) as u64;
        t.push(tasks::rmsnorm(
            at(bufs.h, HIDDEN, *g),
            w.norms + (g * HIDDEN * 2) as u64,
            at(bufs.normed, HIDDEN, *g),
            1,
            h,
            EPS,
            false,
        ));
        t.push(tasks::matmul(
            at(bufs.normed, HIDDEN, *g),
            w.table(*g),
            at(bufs.logits, CODES, row),
            1,
            CODES as u32,
            h,
            false,
        ));
    }
    assert!(t.len() < MAX_TASKS_PER_CARD, "the pass has {} tasks", t.len());
    t
}

/// The tasks of the trunk on `bufs.x`: 3 layers and the final norm into `bufs.h`.
pub fn trunk(w: &Weights, bufs: &Bufs) -> Vec<MkTaskDescriptor> {
    let n = CODEBOOKS as u32;
    let (h, i) = (HIDDEN as u32, INTERMEDIATE as u32);
    let (heads, kv, d) = (HEADS as u32, KV_HEADS as u32, HEAD_DIM as u32);
    let mut t = Vec::new();
    for lw in &w.layers {
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n, h, EPS, false));
        t.push(tasks::matmul(bufs.h, lw.q, bufs.q, n, heads * d, h, false));
        t.push(tasks::matmul(bufs.h, lw.k, bufs.k, n, kv * d, h, false));
        t.push(tasks::matmul(bufs.h, lw.v, bufs.v, n, kv * d, h, false));
        t.push(tasks::matmul(bufs.h, lw.gate, bufs.gates, n, heads, h, false));
        // The queries carry the scale head_dim^-0.5, because the attention op does not scale. Each position of the 12
        // rows is its codebook (modeling_lfm3.py:674-676), so the rows from position 0 turn as the reference turns them.
        let scale = (HEAD_DIM as f32).powf(-0.5);
        t.push(tasks::qk_rope(
            bufs.q,
            lw.q_norm,
            n * heads,
            heads,
            d,
            0,
            true,
            true,
            EPS,
            scale,
        ));
        t.push(tasks::qk_rope(
            bufs.k,
            lw.k_norm,
            n * kv,
            kv,
            d,
            0,
            true,
            true,
            EPS,
            1.0,
        ));
        t.push(tasks::group_attn(
            bufs.q, bufs.k, bufs.v, bufs.y, n, heads, kv, d, n, 1.0,
        ));
        t.push(tasks::head_gate(bufs.y, bufs.gates, bufs.y, n, heads, d, heads));
        t.push(tasks::matmul(bufs.y, lw.o, bufs.x, n, h, heads * d, true));
        t.push(tasks::rmsnorm(bufs.x, lw.ffn_norm, bufs.h, n, h, EPS, false));
        t.push(tasks::swiglu(bufs.h, lw.w13, bufs.g, n, i, h));
        t.push(tasks::matmul(bufs.g, lw.w2, bufs.x, n, h, i, true));
    }
    t.push(tasks::rmsnorm(bufs.x, w.norm, bufs.h, n, h, EPS, false));
    t
}

/// The tasks that put the embedding of a fed frame into row 0 of `x`: the sum of the 12 rows of the codebook table
/// that `bufs.frame_ids` names, in codebook order (modeling_lfm3_omni.py:596-597).
pub fn frame_embedding(w: &Weights, bufs: &Bufs, x: u64) -> Vec<MkTaskDescriptor> {
    (0..CODEBOOKS)
        .map(|g| {
            tasks::embed(
                bufs.frame_ids + (g * 4) as u64,
                w.codebook,
                x,
                1,
                BACKBONE as u32,
                0,
                false,
                g > 0,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nagaya_mk_bmg::protocol::{FLAG_ACCUMULATE, OP_EMBED, OP_LFM3_GROUP_ATTN, OP_MATMUL, OP_RMSNORM};

    fn weights() -> Weights {
        Weights {
            linear: 0x1000,
            linear_bias: 0x2000,
            codes: 1 << 30,
            norms: 0x3000,
            layers: vec![Layer::default(); LAYERS],
            norm: 0x4000,
            codebook: 1 << 34,
        }
    }

    fn bufs() -> Bufs {
        let mut next = 1u64 << 40;
        Bufs::new(&mut |bytes| {
            let at = next;
            next += rounded(bytes) as u64;
            at
        })
    }

    #[test]
    fn a_pass_projects_adds_the_codes_runs_three_layers_and_one_head_for_each_pick() {
        let (w, b) = (weights(), bufs());
        let t = pass(&w, &b, 0x9000, &[7, 2]);
        assert_eq!(
            (t[0].opcode, t[0].a, t[0].n, t[0].m, t[0].k),
            (OP_MATMUL, 0x9000, 1, 12 * 1024, 2048)
        );
        assert_eq!((t[2].opcode, t[2].a, t[2].n, t[2].k), (OP_EMBED, b.ids, 12, 1024));
        assert_ne!(t[2].flags & FLAG_ACCUMULATE, 0, "the codes add to the projection");
        let attn: Vec<&MkTaskDescriptor> = t.iter().filter(|x| x.opcode == OP_LFM3_GROUP_ATTN).collect();
        assert_eq!(attn.len(), 3);
        assert!(
            attn.iter()
                .all(|x| x.n == 12 && x.k == 12 && x.p0 == 32 && x.m == 32 | 8 << 16)
        );
        let heads: Vec<&MkTaskDescriptor> = t.iter().rev().take(4).collect();
        // The last pick: the norm of codebook 2 and its table into logit row 1.
        assert_eq!((heads[0].opcode, heads[0].b, heads[0].m), (OP_MATMUL, w.table(2), 2017));
        assert_eq!(heads[0].c, b.logits + (2017 * 4) as u64);
        assert_eq!((heads[1].opcode, heads[1].b), (OP_RMSNORM, w.norms + 2 * 1024 * 2));
        assert_eq!(heads[1].a, b.h + (2 * 1024 * 4) as u64);
        assert_eq!(heads[2].b, w.table(7));
        assert_eq!(heads[2].c, b.logits);
    }

    #[test]
    fn the_rows_of_the_code_table_mask_each_hidden_codebook() {
        let codes: Vec<u32> = (0..12).map(|g| g * 100).collect();
        let mut revealed = vec![false; 12];
        revealed[3] = true;
        revealed[11] = true;
        let rows = ids(&codes, &revealed);
        assert_eq!(rows[3], 3 * 2017 + 300);
        assert_eq!(rows[11], 11 * 2017 + 1100);
        assert!(rows.iter().enumerate().all(|(g, r)| revealed[g] || *r == MASK_ROW));
        assert_eq!(MASK_ROW, 24_204);
        assert_eq!(frame_ids(&codes)[5], 5 * 2017 + 500);
    }

    #[test]
    fn a_fed_frame_sums_twelve_rows_in_codebook_order() {
        let (w, b) = (weights(), bufs());
        let t = frame_embedding(&w, &b, 0x7000);
        assert_eq!(t.len(), 12);
        for (g, task) in t.iter().enumerate() {
            assert_eq!(
                (task.opcode, task.a, task.b, task.c, task.n, task.k),
                (OP_EMBED, b.frame_ids + 4 * g as u64, w.codebook, 0x7000, 1, 2048)
            );
            assert_eq!(task.flags & FLAG_ACCUMULATE != 0, g > 0);
        }
    }
}
