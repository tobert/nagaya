//! The checkpoint of Inkling 2 on the host: the names of `model.llm.*`, their dtypes and shapes, and the views of
//! the forward.
//!
//! The big matrices stay `bf16` in the mapped file, and each op widens a row when it reads it. The small tensors are
//! widened to `f32` at load. A tensor with another dtype or shape is an error, not a guess.

use super::ops::{
    CONV_TAPS, D_REL, EXPERTS, GATE_RANK, HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, REL_EXTENT, widen_bf16_row,
};
use crate::formats::{self, FormatError};
use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors, TensorMeta};
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

/// The count of layers.
pub const LAYERS: usize = 24;
/// The first layer of a cycle, and the layer after its last.
pub const CYCLE_START: usize = 6;
pub const CYCLE_END: usize = 18;
/// The layers of one block of the attention residuals.
pub const BLOCK: usize = 6;
/// The rows of the unembedding that are logits (`unpadded_vocab_size`). The checkpoint holds 201,024 rows.
pub const LOGITS: usize = 200_058;
pub const VOCAB: usize = 201_024;
/// The layers of the NVFP4 experts. The layers after them hold MXFP8.
pub const NVFP4_LAYERS: usize = 20;

/// True for the layers that attend to all positions: 5, 11, 17 and 23.
pub fn is_full(layer: usize) -> bool {
    layer % BLOCK == BLOCK - 1
}

/// The mapped checkpoint and its table of tensors.
pub struct Checkpoint {
    map: MappedSafeTensors,
    data_offset: usize,
    tensors: HashMap<String, TensorMeta>,
}

/// A `bf16` matrix of the checkpoint, row-major, by its byte range in the file.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub range: Range<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl Checkpoint {
    /// Map the file. It reads only the header.
    pub fn open(path: &Path) -> Result<Self, String> {
        let map = MappedSafeTensors::open(path)?;
        let view = map.view()?;
        let data_offset = view.data_offset;
        let tensors = view.tensors;
        Ok(Self {
            map,
            data_offset,
            tensors,
        })
    }

    fn meta(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&TensorMeta, String> {
        let meta = self.tensors.get(name).ok_or_else(|| {
            format!("the checkpoint holds no tensor {name}. The file is not Inkling 2, or it is damaged.")
        })?;
        if meta.dtype != dtype || meta.shape != shape {
            return Err(format!(
                "the tensor {name} is {:?} {:?}, and the engine needs {dtype:?} {shape:?}. The file is not Inkling 2.",
                meta.dtype, meta.shape
            ));
        }
        Ok(meta)
    }

    /// True when the checkpoint holds the tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The bytes of the tensor `name`, which must have `dtype` and `shape`.
    pub fn bytes(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&[u8], String> {
        let meta = self.meta(name, dtype, shape)?;
        Ok(&self.map.as_bytes()[self.data_offset + meta.data_start..self.data_offset + meta.data_end])
    }

    /// The `bf16` matrix `name` of `rows x cols`.
    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix, String> {
        let meta = self.meta(name, Dtype::BF16, &[rows, cols])?;
        let start = self.data_offset + meta.data_start;
        Ok(Matrix {
            range: start..start + 2 * rows * cols,
            rows,
            cols,
        })
    }

    /// The bytes of a matrix.
    pub fn of(&self, m: &Matrix) -> &[u8] {
        &self.map.as_bytes()[m.range.clone()]
    }

    /// The tensor `name` widened to `f32`. The checkpoint holds it as `BF16` or `F32`; both widen exactly.
    pub fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let dtype = self
            .tensors
            .get(name)
            .map(|m| m.dtype)
            .ok_or_else(|| format!("the checkpoint holds no tensor {name}. The file is not Inkling 2."))?;
        let count: usize = shape.iter().product();
        let mut out = vec![0f32; count];
        match dtype {
            Dtype::BF16 => widen_bf16_row(self.bytes(name, Dtype::BF16, shape)?, &mut out),
            Dtype::F32 => {
                for (o, b) in out
                    .iter_mut()
                    .zip(self.bytes(name, Dtype::F32, shape)?.as_chunks::<4>().0)
                {
                    *o = f32::from_le_bytes(*b);
                }
            }
            other => {
                return Err(format!(
                    "the tensor {name} is {other:?}, and the engine needs BF16 or F32."
                ));
            }
        }
        Ok(out)
    }
}

/// The convolution of the key or the value of a layer.
pub enum KvConv {
    /// The sliding layers: fixed taps, `C x 4` (`k_sconv.weight`).
    Fixed(Vec<f32>),
    /// The full layers: fixed taps by lag, `4 x C` (`k_sconv.static_w`), and the projection of the layer input to the
    /// taps of each head, `KV_HEADS * 4 x HIDDEN` (`k_sconv.weight_proj`).
    Dynamic { static_w: Vec<f32>, proj: Vec<f32> },
}

/// One message gate: `controllers.<attn|k|v|mlp>`.
pub struct MessageGateWeights {
    pub down: Vec<f32>,
    pub forget: Vec<f32>,
    pub gain: Vec<f32>,
}

/// The format of the experts of a layer.
pub enum ExpertFormat {
    /// Layers 0 to 19. The factor of each expert, for `w13` and for `w2` (`reference_expert_factor`).
    Nvfp4 { w13_factor: Vec<f32>, w2_factor: Vec<f32> },
    /// Layers 20 to 23.
    Mxfp8,
}

/// The experts of one layer: the byte ranges of the values and the scales of `w13` and `w2`.
pub struct Experts {
    pub format: ExpertFormat,
    pub w13: Range<usize>,
    pub w13_scale: Range<usize>,
    pub w2: Range<usize>,
    pub w2_scale: Range<usize>,
}

/// The weights of one layer.
pub struct Layer {
    pub index: usize,
    pub full: bool,
    /// The queries of the attention input: none on layer 0, 3 on a full layer (query, key, value), else 1.
    pub attn_queries: Vec<Vec<f32>>,
    pub mlp_query: Vec<f32>,
    /// The norm of the attention input of a sliding layer (`attn_norm`), or the three norms of a full layer
    /// (`q_input_norm`, `k_input_norm`, `v_input_norm`).
    pub input_norms: Vec<Vec<f32>>,
    pub mlp_norm: Vec<f32>,
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub gate_attn: Matrix,
    pub sink: Matrix,
    pub wr: Matrix,
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    pub rel_proj: Vec<f32>,
    pub k_conv: KvConv,
    pub v_conv: KvConv,
    pub attn_conv: Vec<f32>,
    pub mlp_conv: Vec<f32>,
    pub router: Vec<f32>,
    pub router_bias: Vec<f32>,
    /// The gates of a cycle layer, in the order query, key, value, expert input. Empty outside the cycles.
    pub gates: Vec<MessageGateWeights>,
    pub experts: Experts,
}

/// All the weights of the text model.
pub struct Weights {
    pub checkpoint: Checkpoint,
    pub embed: Matrix,
    pub embed_norm: Vec<f32>,
    pub unembed: Matrix,
    pub norm: Vec<f32>,
    pub output_query: Vec<f32>,
    /// The message queries of each cycle layer: query, key, value, expert input.
    pub message_queries: Vec<[Vec<f32>; 4]>,
    pub layers: Vec<Layer>,
}

fn llm(name: &str) -> String {
    format!("model.llm.{name}")
}

impl Weights {
    /// Map the checkpoint at `path` and read the small tensors.
    pub fn open(path: &Path) -> Result<Self, String> {
        let ck = Checkpoint::open(path)?;
        let embed = ck.matrix(&llm("embed.weight"), VOCAB, HIDDEN)?;
        let embed_norm = ck.f32s(&llm("embed_norm.weight"), &[HIDDEN])?;
        // Only the first LOGITS rows are logits (`Inkling2UnpadVocab`).
        let mut unembed = ck.matrix(&llm("unembed.weight"), VOCAB, HIDDEN)?;
        unembed.rows = LOGITS;
        unembed.range.end = unembed.range.start + 2 * LOGITS * HIDDEN;
        let norm = ck.f32s(&llm("norm.weight"), &[HIDDEN])?;
        let output_query = ck.f32s(&llm("attn_res.output_query"), &[HIDDEN])?;
        let message_queries = merge_message_queries(
            &ck.f32s(
                &llm("middle_cycle_cross_pass_queries"),
                &[2 * (CYCLE_END - CYCLE_START), HIDDEN],
            )?,
            &ck.f32s(
                &llm("middle_cycle_cross_pass_kv_queries"),
                &[2 * (CYCLE_END - CYCLE_START), HIDDEN],
            )?,
        );
        let layers = (0..LAYERS).map(|l| layer(&ck, l)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            checkpoint: ck,
            embed,
            embed_norm,
            unembed,
            norm,
            output_query,
            message_queries,
            layers,
        })
    }
}

/// The message queries of each cycle layer, in the order query, key, value, expert input
/// (`Inkling2LoopMergeMessageQueries` in `core_model_loading.py`). Row `2l` of `queries` is the query of layer `l`
/// and row `2l + 1` its expert input. Row `2l` of `kv_queries` is the key, and row `2l + 1` the value.
pub fn merge_message_queries(queries: &[f32], kv_queries: &[f32]) -> Vec<[Vec<f32>; 4]> {
    let row = |t: &[f32], r: usize| t[r * HIDDEN..(r + 1) * HIDDEN].to_vec();
    (0..queries.len() / HIDDEN / 2)
        .map(|l| {
            [
                row(queries, 2 * l),
                row(kv_queries, 2 * l),
                row(kv_queries, 2 * l + 1),
                row(queries, 2 * l + 1),
            ]
        })
        .collect()
}

fn layer(ck: &Checkpoint, l: usize) -> Result<Layer, String> {
    let p = |name: &str| llm(&format!("layers.{l}.{name}"));
    let full = is_full(l);
    let qd = HEADS * HEAD_DIM;
    let kvd = KV_HEADS * HEAD_DIM;
    // `attn_res.attn_queries.N` is the attention input of layer N + 1 (`Inkling2IndexOffsetWeightRename`).
    let attn_queries = if l == 0 {
        Vec::new()
    } else if full {
        let q = ck.f32s(&llm(&format!("attn_res.attn_queries.{}", l - 1)), &[3, HIDDEN])?;
        q.as_chunks::<HIDDEN>().0.iter().map(|r| r.to_vec()).collect()
    } else {
        vec![ck.f32s(&llm(&format!("attn_res.attn_queries.{}", l - 1)), &[HIDDEN])?]
    };
    let input_norms = if full {
        ["q_input_norm", "k_input_norm", "v_input_norm"]
            .iter()
            .map(|n| ck.f32s(&p(&format!("{n}.weight")), &[HIDDEN]))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![ck.f32s(&p("attn_norm.weight"), &[HIDDEN])?]
    };
    let conv = |which: &str| -> Result<KvConv, String> {
        if full {
            Ok(KvConv::Dynamic {
                static_w: ck.f32s(&p(&format!("attn.{which}_sconv.static_w")), &[CONV_TAPS, kvd])?,
                proj: ck.f32s(
                    &p(&format!("attn.{which}_sconv.weight_proj")),
                    &[KV_HEADS * CONV_TAPS, HIDDEN],
                )?,
            })
        } else {
            Ok(KvConv::Fixed(ck.f32s(
                &p(&format!("attn.{which}_sconv.weight")),
                &[kvd, 1, CONV_TAPS],
            )?))
        }
    };
    let gates = if (CYCLE_START..CYCLE_END).contains(&l) {
        ["attn", "k", "v", "mlp"]
            .iter()
            .map(|g| {
                Ok(MessageGateWeights {
                    down: ck.f32s(&p(&format!("controllers.{g}.down_weight")), &[GATE_RANK, 2 * HIDDEN])?,
                    forget: ck.f32s(&p(&format!("controllers.{g}.forget_weight")), &[HIDDEN, GATE_RANK])?,
                    gain: ck.f32s(&p(&format!("controllers.{g}.gain_weight")), &[HIDDEN, GATE_RANK])?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        Vec::new()
    };
    Ok(Layer {
        index: l,
        full,
        attn_queries,
        mlp_query: ck.f32s(&llm(&format!("attn_res.mlp_queries.{l}")), &[HIDDEN])?,
        input_norms,
        mlp_norm: ck.f32s(&p("mlp_norm.weight"), &[HIDDEN])?,
        wq: ck.matrix(&p("attn.wq_du.weight"), qd, HIDDEN)?,
        wk: ck.matrix(&p("attn.wk_dv.weight"), kvd, HIDDEN)?,
        wv: ck.matrix(&p("attn.wv_dv.weight"), kvd, HIDDEN)?,
        wo: ck.matrix(&p("attn.wo_ud.weight"), HIDDEN, qd)?,
        gate_attn: ck.matrix(&p("attn.gate_attn.weight"), qd, HIDDEN)?,
        sink: ck.matrix(&p("attn.attn_sink_proj.weight"), HEADS, HIDDEN)?,
        wr: ck.matrix(&p("attn.wr_du.weight"), HEADS * D_REL, HIDDEN)?,
        q_scale: ck.f32s(&p("attn.q_l2_clip.scale"), &[HEADS, HEAD_DIM])?,
        rel_scale: ck.f32s(&p("attn.rel_l2_clip.scale"), &[HEADS, D_REL])?,
        rel_proj: ck.f32s(&p("attn.rel_logits_proj.proj"), &[D_REL, REL_EXTENT])?,
        k_conv: conv("k")?,
        v_conv: conv("v")?,
        attn_conv: ck.f32s(&p("attn_sconv.weight"), &[HIDDEN, 1, CONV_TAPS])?,
        mlp_conv: ck.f32s(&p("mlp_sconv.weight"), &[HIDDEN, 1, CONV_TAPS])?,
        router: ck.f32s(&p("mlp.gate.weight"), &[EXPERTS, HIDDEN])?,
        router_bias: ck.f32s(&p("mlp.gate.bias"), &[EXPERTS])?,
        gates,
        experts: experts(ck, l)?,
    })
}

fn experts(ck: &Checkpoint, l: usize) -> Result<Experts, String> {
    let p = |name: &str| llm(&format!("layers.{l}.mlp.experts.{name}"));
    let range = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<Range<usize>, String> {
        let bytes = ck.bytes(&p(name), dtype, shape)?;
        let start = bytes.as_ptr() as usize - ck.map.as_bytes().as_ptr() as usize;
        Ok(start..start + bytes.len())
    };
    let rows13 = 2 * INTERMEDIATE;
    if l < NVFP4_LAYERS {
        let factors = |w: &str| -> Result<Vec<f32>, String> {
            let amax = ck.f32s(&p(&format!("{w}_weight_amax")), &[EXPERTS])?;
            let den = ck.f32s(&p(&format!("{w}_weight_outer_scale_denominator")), &[1])?[0];
            Ok(amax.iter().map(|a| formats::reference_expert_factor(*a, den)).collect())
        };
        Ok(Experts {
            format: ExpertFormat::Nvfp4 {
                w13_factor: factors("w13")?,
                w2_factor: factors("w2")?,
            },
            w13: range("w13_weight", Dtype::U8, &[EXPERTS, rows13, HIDDEN / 2])?,
            w13_scale: range("w13_weight_scale", Dtype::F8E4M3, &[EXPERTS, rows13, HIDDEN / 16])?,
            w2: range("w2_weight", Dtype::U8, &[EXPERTS, HIDDEN, INTERMEDIATE / 2])?,
            w2_scale: range("w2_weight_scale", Dtype::F8E4M3, &[EXPERTS, HIDDEN, INTERMEDIATE / 16])?,
        })
    } else {
        Ok(Experts {
            format: ExpertFormat::Mxfp8,
            w13: range("w13_weight", Dtype::F8E4M3, &[EXPERTS, rows13, HIDDEN])?,
            w13_scale: range("w13_weight_scale", Dtype::U8, &[EXPERTS, rows13, HIDDEN / 32])?,
            w2: range("w2_weight", Dtype::F8E4M3, &[EXPERTS, HIDDEN, INTERMEDIATE])?,
            w2_scale: range("w2_weight_scale", Dtype::U8, &[EXPERTS, HIDDEN, INTERMEDIATE / 32])?,
        })
    }
}

impl Weights {
    /// Expand row `row` of one expert matrix to `f32`. `down` chooses `w2`, else `w13`. For `w13`, `row` is a row of
    /// the loaded matrix: the gate rows, then the up rows (`formats::gate_up_source_row`).
    pub fn expert_row(&self, layer: usize, expert: usize, down: bool, row: usize, out: &mut [f32]) {
        let e = &self.layers[layer].experts;
        let bytes = self.checkpoint.map.as_bytes();
        let (rows, cols, values, scales) = if down {
            (HIDDEN, INTERMEDIATE, &e.w2, &e.w2_scale)
        } else {
            (2 * INTERMEDIATE, HIDDEN, &e.w13, &e.w13_scale)
        };
        assert_eq!(out.len(), cols, "expert_row: {} values for a row of {cols}", out.len());
        let source = if down {
            row
        } else {
            formats::gate_up_source_row(row, rows)
        };
        let index = expert * rows + source;
        let mut bits = vec![0u16; cols];
        let result: Result<(), FormatError> = match &e.format {
            ExpertFormat::Nvfp4 { w13_factor, w2_factor } => {
                let (vb, sb) = (cols / 2, cols / formats::NVFP4_BLOCK);
                let codes = &bytes[values.start + index * vb..values.start + (index + 1) * vb];
                let sc = &bytes[scales.start + index * sb..scales.start + (index + 1) * sb];
                let factor = if down { w2_factor[expert] } else { w13_factor[expert] };
                formats::expand_nvfp4_row(codes, sc, factor, &mut bits)
            }
            ExpertFormat::Mxfp8 => {
                let sb = cols / formats::MXFP8_BLOCK;
                let vals = &bytes[values.start + index * cols..values.start + (index + 1) * cols];
                let sc = &bytes[scales.start + index * sb..scales.start + (index + 1) * sb];
                formats::expand_mxfp8_row(vals, sc, &mut bits)
            }
        };
        if let Err(err) = result {
            panic!("layer {layer} expert {expert} row {row}: {err}");
        }
        for (o, b) in out.iter_mut().zip(&bits) {
            *o = super::ops::bf16_to_f32(*b);
        }
    }
}
