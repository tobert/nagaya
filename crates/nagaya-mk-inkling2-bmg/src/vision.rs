//! The vision tower of Inkling 2 on the card: its weights in an upload sink, its buffers, and the tasks of one
//! image (engine-abi "The images of Inkling 2", "The choices of the tower"). None of this needs a card.
//!
//! The tasks of [`tower`] follow the host form (`host::vision::Vision::tower` and `project`) step by step. The pass
//! puts the patch rows of an image at `rows`, runs the tasks, and reads `T` projector rows of 2,304 values at `out`.
//! The tower of each image runs alone, so an image has the same bits in a call of one image or of many.

use std::ops::Range;

use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::safetensors::SafeTensorsView;
use nagaya_mk_bmg::tasks;

use crate::host::ops::HIDDEN;
use crate::host::vision::{
    GRID, Grid, HEAD_DIM, HEADS, LAYERS, LN_EPS, MAX_PATCH_SIDE, MAX_TOKENS, MERGE, MLP, PATCH_VALUES, PROJ_HIDDEN,
    ROPE_POSITIONS, WIDTH, axis_taps, rope_angle,
};
use crate::spec::rounded;
use crate::weights::Upload;

/// The frequencies of one axis of a head: the columns of the rotary table.
pub const ROPE_FREQS: usize = HEAD_DIM / 4;
/// The bytes of the weights of the tower on the card, with the rotary table: the term `vision`
/// (`tests/vision_weights.rs` counts them).
pub const WEIGHTS_BYTES: usize = 636_825_472;
/// The tasks of one tower run: 4 before the layers, 19 in each layer, and 3 of the projector.
pub const TOWER_TASKS: usize = 4 + 19 * LAYERS + 3;

/// A LayerNorm on the card: `f32` weight and bias.
#[derive(Clone, Copy, Debug, Default)]
pub struct Norm {
    pub weight: u64,
    pub bias: u64,
}

/// A linear map on the card: a `bf16` matrix, and its `bf16` bias or 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct Linear {
    pub weight: u64,
    pub bias: u64,
}

/// The weights of one layer of the tower on the card.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layer {
    pub ln_1: Norm,
    pub ln_2: Norm,
    /// The layer scales, `f32`.
    pub ls_1: u64,
    pub ls_2: u64,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub fc: Linear,
    pub proj: Linear,
}

/// The weights of the tower and the projector on the card: the term `vision`.
#[derive(Clone, Debug, Default)]
pub struct Weights {
    /// The patch embedding, `bf16 [1024][1200]`.
    pub patch: u64,
    /// The table of 32 x 32 positions, `bf16 [1024][1024]`.
    pub position: u64,
    /// `class_embedding + cls_position_embedding` in `f32`, as the host form adds them.
    pub class_row: u64,
    pub ln_pre: Norm,
    pub layers: Vec<Layer>,
    /// The projector, `bf16 [4096][4096]` and `[2304][4096]`, with no bias.
    pub proj_in: u64,
    pub proj_out: u64,
    /// The cosines `[207][16]`, then the sines, from [`rope_table`].
    pub rope: u64,
}

fn visual(name: &str) -> String {
    format!("model.visual.{name}")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The table of the rotary embedding: the cosine of each position `p` of `0..207` and frequency `k` of `0..16`,
/// then the sine, as the host form computes the angle `p * 10000^(-k / 16)` and its cosine and sine in `f32`.
pub fn rope_table() -> Vec<f32> {
    let angles: Vec<f32> = (0..ROPE_POSITIONS)
        .flat_map(|p| (0..ROPE_FREQS).map(move |k| rope_angle(k, 0, p)))
        .collect();
    let mut table: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    table.extend(angles.iter().map(|a| a.sin()));
    table
}

/// The tensors of one load and its sink.
struct Loader<'s, 'v> {
    sink: &'s mut dyn Upload,
    view: &'s SafeTensorsView<'v>,
}

impl<'v> Loader<'_, 'v> {
    fn bf16(&self, name: &str, shape: &[usize]) -> Result<&'v [u8], String> {
        self.view
            .bf16_tensor(&visual(name), shape)
            .map_err(|e| format!("the vision tower cannot load: {e}. The file is not Inkling 2, or it is damaged."))
    }

    fn put(&mut self, bytes: &[u8]) -> u64 {
        let at = self.sink.alloc(bytes.len());
        if self.sink.writes() {
            self.sink.write(at, bytes);
        }
        at
    }

    /// A `bf16` tensor as the checkpoint holds it.
    fn raw(&mut self, name: &str, shape: &[usize]) -> Result<u64, String> {
        let bytes = self.bf16(name, shape)?;
        Ok(self.put(bytes))
    }

    fn widened(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        Ok(self
            .bf16(name, shape)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect())
    }

    /// A `bf16` vector widened to `f32`: each value is exact.
    fn wide(&mut self, name: &str, len: usize) -> Result<u64, String> {
        let values = self.widened(name, &[len])?;
        Ok(self.put(&f32_bytes(&values)))
    }

    fn norm(&mut self, name: &str) -> Result<Norm, String> {
        Ok(Norm {
            weight: self.wide(&format!("{name}.weight"), WIDTH)?,
            bias: self.wide(&format!("{name}.bias"), WIDTH)?,
        })
    }

    fn linear(&mut self, name: &str, rows: usize, cols: usize) -> Result<Linear, String> {
        Ok(Linear {
            weight: self.raw(&format!("{name}.weight"), &[rows, cols])?,
            bias: self.raw(&format!("{name}.bias"), &[rows])?,
        })
    }
}

/// Load the 422 tensors of `model.visual.` of `view` into `sink`, and the rotary table. A tensor with another dtype
/// or shape is an error.
pub fn load(sink: &mut dyn Upload, view: &SafeTensorsView<'_>) -> Result<Weights, String> {
    let mut ld = Loader { sink, view };
    let emb = "backbone.embeddings";
    let class = ld.widened(&format!("{emb}.class_embedding"), &[WIDTH])?;
    let class_pos = ld.widened(&format!("{emb}.cls_position_embedding"), &[WIDTH])?;
    let class_row: Vec<f32> = class.iter().zip(&class_pos).map(|(a, b)| a + b).collect();
    let mut layers = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let p = format!("backbone.layers.{l}");
        layers.push(Layer {
            ln_1: ld.norm(&format!("{p}.ln_1"))?,
            ln_2: ld.norm(&format!("{p}.ln_2"))?,
            ls_1: ld.wide(&format!("{p}.ls_1.gamma"), WIDTH)?,
            ls_2: ld.wide(&format!("{p}.ls_2.gamma"), WIDTH)?,
            q: ld.linear(&format!("{p}.self_attn.q_proj"), WIDTH, WIDTH)?,
            k: ld.linear(&format!("{p}.self_attn.k_proj"), WIDTH, WIDTH)?,
            v: ld.linear(&format!("{p}.self_attn.v_proj"), WIDTH, WIDTH)?,
            out: ld.linear(&format!("{p}.self_attn.out_proj"), WIDTH, WIDTH)?,
            fc: ld.linear(&format!("{p}.mlp.c_fc"), MLP, WIDTH)?,
            proj: ld.linear(&format!("{p}.mlp.c_proj"), WIDTH, MLP)?,
        });
    }
    Ok(Weights {
        patch: ld.raw(&format!("{emb}.patch_embedding.weight"), &[WIDTH, PATCH_VALUES])?,
        position: ld.raw(&format!("{emb}.position_embedding"), &[GRID * GRID, WIDTH])?,
        class_row: ld.put(&f32_bytes(&class_row)),
        ln_pre: ld.norm("backbone.ln_pre")?,
        layers,
        proj_in: ld.raw("proj_in.weight", &[PROJ_HIDDEN, MERGE * MERGE * WIDTH])?,
        proj_out: ld.raw("proj_out.weight", &[HIDDEN, PROJ_HIDDEN])?,
        rope: ld.put(&f32_bytes(&rope_table())),
    })
}

/// The taps and the weights of the position embedding of an image, which `OP_INKLING2_VISION_EMBED` reads: for each
/// patch line, then for each patch column, the two taps as `u32` and their two weights as `f32`, little-endian. The
/// rules are those of the host form (`host::vision::axis_taps`), so the op gives the bits of the host form.
pub fn axes(g: Grid) -> Vec<u8> {
    let (lines, columns) = (MERGE * g.rows, MERGE * g.columns);
    let mut out = Vec::with_capacity(16 * (lines + columns));
    for (size, count) in [(lines, lines), (columns, columns)] {
        for index in 0..count {
            let [(t0, w0), (t1, w1)] = axis_taps(index, size);
            for word in [t0 as u32, t1 as u32, w0.to_bits(), w1.to_bits()] {
                out.extend(word.to_le_bytes());
            }
        }
    }
    out
}

/// The bytes of [`axes`] for the largest image of the size rule.
pub const AXES_BYTES: usize = 16 * 2 * MAX_PATCH_SIDE;

/// The rows of the tower of an image of `tokens` image tokens: the class row and 4 patches for each token.
pub fn tower_rows(tokens: usize) -> usize {
    MERGE * MERGE * tokens + 1
}

/// The bytes of the patch rows of an image of `tokens` image tokens, `f32`.
pub fn rows_bytes(tokens: usize) -> usize {
    MERGE * MERGE * tokens * PATCH_VALUES * 4
}

/// The bytes of the projector rows of an image of `tokens` image tokens, `f32`.
pub fn out_bytes(tokens: usize) -> usize {
    tokens * HIDDEN * 4
}

/// The buffers of the tower, each `f32`, for images of at most `max_tokens` image tokens. Each step uses them again
/// (see [`tower`]): `x` holds the residual, `h` a norm or the attention, `q`, `k` and `v` the projections, and `m`
/// the MLP.
#[derive(Clone, Copy, Debug)]
pub struct Bufs {
    pub x: u64,
    pub h: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub m: u64,
    pub max_tokens: usize,
}

impl Bufs {
    /// The bytes of each buffer for `max_tokens`: five of `WIDTH` and one of `MLP` values in each row.
    pub fn sizes(max_tokens: usize) -> [usize; 6] {
        let rows = tower_rows(max_tokens);
        let w = rows * WIDTH * 4;
        [w, w, w, w, w, rows * MLP * 4]
    }

    /// The bytes of all buffers for `max_tokens`.
    pub fn bytes(max_tokens: usize) -> usize {
        Self::sizes(max_tokens).iter().sum()
    }

    /// The buffers from `z`, which gives device memory of zeros. The largest image of the size rule needs
    /// `max_tokens` = `MAX_TOKENS`.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, max_tokens: usize) -> Self {
        assert!(
            (1..=MAX_TOKENS).contains(&max_tokens),
            "the buffers of the tower hold 1 to {MAX_TOKENS} image tokens, not {max_tokens}"
        );
        let [x, h, q, k, v, m] = Self::sizes(max_tokens).map(&mut *z);
        Self {
            x,
            h,
            q,
            k,
            v,
            m,
            max_tokens,
        }
    }
}

/// The device memory of the images of a pass (engine-abi "The choices of the pass"): the buffers of the tower, the
/// patch rows, the taps and the projector rows of one image of the largest size, and the media rows of a block.
#[derive(Clone, Copy, Debug)]
pub struct PassBufs {
    pub tower: Bufs,
    /// The patch rows of one image, `f32`.
    pub rows: u64,
    /// The taps of the position embedding of one image ([`axes`]).
    pub axes: u64,
    /// The projector rows of one image, `f32 [T][2304]`: the rows that `media::HeldRows` names.
    pub out: u64,
    /// The media rows of a block, `f32 [max_block][2304]`: the projector rows of the placeholders of the block.
    pub media: u64,
    pub max_block: usize,
}

impl PassBufs {
    /// The bytes of each buffer for a block of `max_block` tokens, after the buffers of the tower: the patch rows, the
    /// taps, the projector rows and the media rows.
    fn sizes(max_block: usize) -> [usize; 4] {
        [
            rows_bytes(MAX_TOKENS),
            AXES_BYTES,
            out_bytes(MAX_TOKENS),
            max_block * HIDDEN * 4,
        ]
    }

    /// The device bytes of all the buffers for a block of `max_block` tokens: a part of the term `work`.
    pub fn bytes(max_block: usize) -> u64 {
        let tower: usize = Bufs::sizes(MAX_TOKENS).iter().map(|b| rounded(*b)).sum();
        let rest: usize = Self::sizes(max_block).iter().map(|b| rounded(*b)).sum();
        (tower + rest) as u64
    }

    /// The buffers from `z`, which gives device memory of zeros.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, max_block: usize) -> Self {
        let tower = Bufs::new(z, MAX_TOKENS);
        let [rows, axes, out, media] = Self::sizes(max_block).map(&mut *z);
        Self {
            tower,
            rows,
            axes,
            out,
            media,
            max_block,
        }
    }

    /// The copy of the projector rows `take` of the held image into the media rows from `block_row`.
    pub fn take(&self, take: Range<usize>, block_row: usize) -> MkTaskDescriptor {
        assert!(
            !take.is_empty() && block_row + take.len() <= self.max_block,
            "the rows {take:?} do not fit the media rows of a block of {} from row {block_row}",
            self.max_block
        );
        let row = (HIDDEN * 4) as u64;
        tasks::copy(
            self.out + take.start as u64 * row,
            self.media + block_row as u64 * row,
            (take.len() * HIDDEN) as u32,
        )
    }

    /// The copy of the media rows `rows` of the block over the rows of the block in `x`, the normed embedding.
    pub fn place(&self, x: u64, rows: Range<usize>) -> MkTaskDescriptor {
        assert!(
            !rows.is_empty() && rows.end <= self.max_block,
            "the rows {rows:?} are not rows of a block of {}",
            self.max_block
        );
        let row = (HIDDEN * 4) as u64;
        tasks::copy(
            self.media + rows.start as u64 * row,
            x + rows.start as u64 * row,
            (rows.len() * HIDDEN) as u32,
        )
    }
}

/// A stage of the tower that a gate reads: the stage, the count of tasks that write it, and its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mark {
    pub stage: Stage,
    /// The stage is in place after the first `tasks` tasks of the run.
    pub tasks: usize,
    /// The buffer of the stage, its rows and the values of a row.
    pub at: u64,
    pub rows: usize,
    pub width: usize,
}

/// The stages of the tower, as the host form traces them (`host::vision::VisionTrace`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The patch embedding alone, `4T` rows.
    PatchEmbed,
    /// The class row, then the patches with their position embedding.
    Embeddings,
    /// After `ln_pre`.
    PreLayernorm,
    /// After layer `l`.
    Layer(usize),
    /// The projector rows, `T` rows of 2,304 values.
    Projector,
}

/// The tasks of one tower run, and the stages that a gate can read.
#[derive(Clone, Debug)]
pub struct Tower {
    pub tasks: Vec<MkTaskDescriptor>,
    pub marks: Vec<Mark>,
}

/// The tasks of the tower and the projector for one image of grid `g`: the patch rows at `rows` give the projector
/// rows at `out`. `axes` holds [`axes`] of `g`, which the caller writes with the rows. The tasks write only `b` and
/// `out`.
pub fn tower(w: &Weights, b: &Bufs, g: Grid, rows: u64, axes: u64, out: u64) -> Tower {
    let t = g.tokens();
    assert!(
        (1..=b.max_tokens).contains(&t),
        "the buffers of the tower hold {} image tokens, and the image has {t}",
        b.max_tokens
    );
    assert_eq!(w.layers.len(), LAYERS, "the tower has {LAYERS} layers");
    let n = tower_rows(t) as u32;
    let p = n - 1;
    let (wd, mlp) = (WIDTH as u32, MLP as u32);
    let eps = LN_EPS as f32;
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let mut tk = Vec::with_capacity(TOWER_TASKS);
    let mut marks = Vec::new();
    let mut mark = |tk: &Vec<MkTaskDescriptor>, stage, at, rows: u32, width: usize| {
        marks.push(Mark {
            stage,
            tasks: tk.len(),
            at,
            rows: rows as usize,
            width,
        })
    };
    // The patch embedding into q, then the class row and the positions into x, then ln_pre in place.
    tk.push(tasks::matmul(rows, w.patch, b.q, p, wd, PATCH_VALUES as u32, false));
    mark(&tk, Stage::PatchEmbed, b.q, p, WIDTH);
    tk.push(tasks::copy(w.class_row, b.x, wd));
    tk.push(tasks::inkling2_vision_embed(
        b.q,
        w.position,
        axes,
        b.x,
        g.rows as u32,
        g.columns as u32,
        wd,
    ));
    mark(&tk, Stage::Embeddings, b.x, n, WIDTH);
    tk.push(tasks::layernorm(b.x, w.ln_pre.weight, w.ln_pre.bias, b.x, n, wd, eps));
    mark(&tk, Stage::PreLayernorm, b.x, n, WIDTH);
    let linear = |tk: &mut Vec<MkTaskDescriptor>, l: &Linear, x: u64, y: u64, m: u32, k: u32| {
        tk.push(tasks::matmul(x, l.weight, y, n, m, k, false));
        tk.push(tasks::bias(l.bias, y, n, m, false));
    };
    for (i, l) in w.layers.iter().enumerate() {
        // x + ls_1 * out(attn(rope(q(h)), rope(k(h)), v(h))), with h = ln_1(x). The attention goes into h.
        tk.push(tasks::layernorm(b.x, l.ln_1.weight, l.ln_1.bias, b.h, n, wd, eps));
        linear(&mut tk, &l.q, b.h, b.q, wd, wd);
        linear(&mut tk, &l.k, b.h, b.k, wd, wd);
        linear(&mut tk, &l.v, b.h, b.v, wd, wd);
        tk.push(tasks::inkling2_vision_rope(b.q, b.k, w.rope, n, wd, g.columns as u32));
        tk.push(tasks::inkling2_vision_attn(
            b.q,
            b.k,
            b.v,
            b.h,
            n,
            HEADS as u32,
            HEADS as u32,
            HEAD_DIM as u32,
            scale,
        ));
        linear(&mut tk, &l.out, b.h, b.q, wd, wd);
        tk.push(tasks::scale_add(b.q, l.ls_1, b.x, n, wd));
        // x + ls_2 * c_proj(gelu(c_fc(ln_2(x)))).
        tk.push(tasks::layernorm(b.x, l.ln_2.weight, l.ln_2.bias, b.h, n, wd, eps));
        linear(&mut tk, &l.fc, b.h, b.m, mlp, wd);
        tk.push(tasks::gelu(b.m, b.m, n, mlp));
        linear(&mut tk, &l.proj, b.m, b.q, wd, mlp);
        tk.push(tasks::scale_add(b.q, l.ls_2, b.x, n, wd));
        mark(&tk, Stage::Layer(i), b.x, n, WIDTH);
    }
    // The class row goes, and the 4 rows of each token are one row of 4,096 values: the rows after row 0 of x.
    let merged = b.x + (WIDTH * 4) as u64;
    let (t32, hidden) = (t as u32, PROJ_HIDDEN as u32);
    tk.push(tasks::matmul(
        merged,
        w.proj_in,
        b.m,
        t32,
        hidden,
        (MERGE * MERGE * WIDTH) as u32,
        false,
    ));
    tk.push(tasks::gelu(b.m, b.m, t32, hidden));
    tk.push(tasks::matmul(b.m, w.proj_out, out, t32, HIDDEN as u32, hidden, false));
    mark(&tk, Stage::Projector, out, t32, HIDDEN);
    assert_eq!(tk.len(), TOWER_TASKS, "the tower has {TOWER_TASKS} tasks");
    Tower { tasks: tk, marks }
}

#[cfg(test)]
mod tests;
