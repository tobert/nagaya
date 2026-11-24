//! The weights of Inkling 2 Mini on the device, in the forms that the ops read (`cuda` feature).
//!
//! Each dense matrix is repacked for the matmul (`repack_bf16`), the experts for the expert ops, and the small
//! tensors go up in the dtype that their op reads. The widening of `bf16` to `f32` is exact.

use std::path::Path;

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::experts::{ExpertFormat, repack_expert, repack_expert_up};
use nagaya_mk_cuda::proj::{ProjLayout, SINKS, Segment, join};
use nagaya_mk_cuda::repack::{MatmulShape, repack_bf16};
use nagaya_mk_cuda::rows::{gate_down_weights, repack_gate_up};

use super::dev::{Checkpoint, Dev};
use super::layout::{LOGIT_COLUMNS, message_queries_of};
use crate::spec::{
    CONV_TAPS, CYCLE_END, CYCLE_START, EXPERTS, HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_ROW, LAYERS, NVFP4_LAYERS,
    VOCAB, is_full,
};

/// The width of the query projection: 24 heads of 128.
const Q_WIDTH: usize = HEADS * HEAD_DIM;
/// The width of the relative state of a head.
const D_REL: usize = 16;
/// The positions of the relative projection.
const REL_EXTENT: usize = 1024;
/// The message gates of a cycle layer, in the order of their use: the three inputs of the attention, then the mlp.
const GATES: [&str; 4] = ["attn", "k", "v", "mlp"];
/// The layers of the cycle, with their message queries.
const CYCLE_RUNS: usize = CYCLE_END - CYCLE_START;

/// The format of the experts of a layer.
pub fn expert_format(layer: usize) -> ExpertFormat {
    if layer < NVFP4_LAYERS {
        ExpertFormat::Nvfp4
    } else {
        ExpertFormat::Mxfp8
    }
}

/// The experts of one layer: the banks of the up and the down matmul, and what the NVFP4 epilogue needs.
pub struct ExpertBank<'d> {
    pub format: ExpertFormat,
    /// `w13`, repacked with the rows of the gate and the up in one MMA tile (`repack_expert_up`).
    pub up: Dev<'d>,
    /// `w2`.
    pub down: Dev<'d>,
    /// The amax of each expert of `w13` and `w2` (`f32`, 128 each). NVFP4 only.
    pub up_amax: Option<Dev<'d>>,
    pub down_amax: Option<Dev<'d>>,
    /// The outer scale denominators of `w13` and `w2`. NVFP4 only, else 1.
    pub up_denominator: f32,
    pub down_denominator: f32,
}

/// The weights of one layer.
pub struct LayerWeights<'d> {
    pub layer: usize,
    pub full: bool,
    /// `J`: the joined and repacked projection of the attention input (`proj::join`).
    pub j: Dev<'d>,
    /// The norm weights of the q, k and v inputs (`bf16`). A sliding layer has one.
    pub input_norms: Vec<Dev<'d>>,
    /// The scale of the query `[24][128]` and of the relative state `[24][16]` (`bf16`).
    pub q_scale: Dev<'d>,
    pub rel_scale: Dev<'d>,
    /// The convolution weights of k and v (`bf16`): `[1024][1][4]` on a sliding layer, `[4][1024]` on a full layer.
    pub k_conv: Dev<'d>,
    pub v_conv: Dev<'d>,
    /// The relative projection `[16][1024]` (`bf16`).
    pub rel_proj: Dev<'d>,
    /// The output matrix `[2304][3072]`, repacked.
    pub wo: Dev<'d>,
    /// The attention convolution `[2304][1][4]` (`bf16`) and the mlp convolution (`f32`, `[2304][4]`).
    pub attn_conv: Dev<'d>,
    pub mlp_conv: Dev<'d>,
    /// The queries of the attention mix (`f32`, `[Q][2304]`, none on layer 0) and the mlp mix (`f32`, 2304).
    pub attn_queries: Option<Dev<'d>>,
    pub mlp_query: Dev<'d>,
    /// The norm of the mlp input, `f32` (the router and the gates read it as `f32`).
    pub mlp_norm: Dev<'d>,
    /// The router `[128][2304]`, repacked, and its bias (`f32`).
    pub router: Dev<'d>,
    pub router_bias: Dev<'d>,
    /// The message gates of a cycle layer: the 4 down matrices one after the other (`gate_down_weights`), and the 4
    /// forget and gain pairs one after the other (`repack_gate_up`).
    pub gate_down: Option<Dev<'d>>,
    pub gate_up: Option<Dev<'d>>,
    pub experts: ExpertBank<'d>,
}

/// The weights of the whole model.
pub struct DeviceWeights<'d> {
    /// The embedding `[201024][2304]` (`bf16`) and its norm (`f32`).
    pub embed: Dev<'d>,
    pub embed_norm: Dev<'d>,
    /// The first 200,064 rows of the unembedding, repacked. The first 200,058 are logits.
    pub head: Dev<'d>,
    /// The final norm (`bf16`, for the head) and the output query (`f32`).
    pub norm_bf16: Dev<'d>,
    pub output_query: Dev<'d>,
    /// The message queries of each cycle layer: query, key, value, expert input (`f32`, `[4][2304]`).
    pub message_queries: Vec<Dev<'d>>,
    pub layers: Vec<LayerWeights<'d>>,
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
        .collect()
}

/// Repack the experts of a matrix in parallel, one expert after the other in the bank.
fn repack_bank(
    format: ExpertFormat,
    n: usize,
    k: usize,
    up: bool,
    codes: &[u8],
    scales: &[u8],
) -> Result<Vec<u8>, String> {
    let (code_row, scale_row) = (format.code_bytes(k), format.scale_bytes(k));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let per = EXPERTS.div_ceil(threads);
    let results: Vec<Result<Vec<u8>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..EXPERTS)
            .step_by(per)
            .map(|first| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for e in first..(first + per).min(EXPERTS) {
                        let c = &codes[e * n * code_row..(e + 1) * n * code_row];
                        let s = &scales[e * n * scale_row..(e + 1) * n * scale_row];
                        out.extend(if up {
                            repack_expert_up(format, n, k, c, s)?
                        } else {
                            repack_expert(format, n, k, c, s)?
                        });
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("an expert repack thread panicked".to_owned()))
            })
            .collect()
    });
    let mut out = Vec::new();
    for r in results {
        out.extend(r?);
    }
    Ok(out)
}

fn experts<'d>(dev: &'d CudaDevice, ck: &mut Checkpoint, l: usize) -> Result<ExpertBank<'d>, String> {
    let format = expert_format(l);
    let p = |name: &str| format!("model.llm.layers.{l}.mlp.experts.{name}");
    let rows13 = 2 * INTERMEDIATE;
    let (code_dtype, scale_dtype) = match format {
        ExpertFormat::Nvfp4 => ("U8", "F8_E4M3"),
        ExpertFormat::Mxfp8 => ("F8_E4M3", "U8"),
    };
    let shapes = |n: usize, k: usize| {
        (
            vec![EXPERTS, n, format.code_bytes(k)],
            vec![EXPERTS, n, format.scale_bytes(k)],
        )
    };
    let mut bank = |which: &str, n: usize, k: usize, up: bool| -> Result<Dev<'d>, String> {
        let (code_shape, scale_shape) = shapes(n, k);
        let codes = ck.bytes(&p(&format!("{which}_weight")), code_dtype, &code_shape)?;
        let scales = ck.bytes(&p(&format!("{which}_weight_scale")), scale_dtype, &scale_shape)?;
        Dev::upload(dev, &repack_bank(format, n, k, up, &codes, &scales)?)
    };
    let up = bank("w13", rows13, HIDDEN, true)?;
    let down = bank("w2", HIDDEN, INTERMEDIATE, false)?;
    let (up_amax, down_amax, up_den, down_den) = match format {
        ExpertFormat::Nvfp4 => {
            let amax = |ck: &mut Checkpoint, w: &str| -> Result<Dev<'d>, String> {
                Dev::upload_f32(dev, &ck.f32s(&p(&format!("{w}_weight_amax")), &[EXPERTS])?)
            };
            let den = |ck: &mut Checkpoint, w: &str| -> Result<f32, String> {
                Ok(ck.f32s(&p(&format!("{w}_weight_outer_scale_denominator")), &[1])?[0])
            };
            (
                Some(amax(ck, "w13")?),
                Some(amax(ck, "w2")?),
                den(ck, "w13")?,
                den(ck, "w2")?,
            )
        }
        ExpertFormat::Mxfp8 => (None, None, 1.0, 1.0),
    };
    Ok(ExpertBank {
        format,
        up,
        down,
        up_amax,
        down_amax,
        up_denominator: up_den,
        down_denominator: down_den,
    })
}

fn layer<'d>(dev: &'d CudaDevice, ck: &mut Checkpoint, l: usize) -> Result<LayerWeights<'d>, String> {
    let full = is_full(l);
    let p = |name: &str| format!("model.llm.layers.{l}.{name}");
    let llm = |name: &str| format!("model.llm.{name}");
    let bf = |ck: &mut Checkpoint, name: &str, shape: &[usize]| ck.bytes(name, "BF16", shape);
    let kv = KV_ROW;
    // The joined projection.
    let layout = ProjLayout { full };
    let wq = bf(ck, &p("attn.wq_du.weight"), &[Q_WIDTH, HIDDEN])?;
    let wr = bf(ck, &p("attn.wr_du.weight"), &[HEADS * D_REL, HIDDEN])?;
    let sink = bf(ck, &p("attn.attn_sink_proj.weight"), &[SINKS, HIDDEN])?;
    let gate = bf(ck, &p("attn.gate_attn.weight"), &[Q_WIDTH, HIDDEN])?;
    let wk = bf(ck, &p("attn.wk_dv.weight"), &[kv, HIDDEN])?;
    let wv = bf(ck, &p("attn.wv_dv.weight"), &[kv, HIDDEN])?;
    let taps = if full {
        Some((
            bf(ck, &p("attn.k_sconv.weight_proj"), &[8 * CONV_TAPS, HIDDEN])?,
            bf(ck, &p("attn.v_sconv.weight_proj"), &[8 * CONV_TAPS, HIDDEN])?,
        ))
    } else {
        None
    };
    let mut parts: Vec<(Segment, &[u8])> = vec![
        (Segment::Q, &wq),
        (Segment::Rel, &wr),
        (Segment::Sink, &sink),
        (Segment::Gate, &gate),
        (Segment::K, &wk),
        (Segment::V, &wv),
    ];
    if let Some((tk, tv)) = &taps {
        parts.push((Segment::TapsK, tk));
        parts.push((Segment::TapsV, tv));
    }
    let joined = join(&layout, &parts)?;
    let j = Dev::upload(dev, &repack_bf16(&joined, &layout.shape())?)?;
    drop((joined, wq, wr, sink, gate, wk, wv, taps));

    let norm_names: &[&str] = if full {
        &["q_input_norm", "k_input_norm", "v_input_norm"]
    } else {
        &["attn_norm"]
    };
    let input_norms = norm_names
        .iter()
        .map(|n| Dev::upload(dev, &bf(ck, &p(&format!("{n}.weight")), &[HIDDEN])?))
        .collect::<Result<Vec<_>, String>>()?;
    let conv = |ck: &mut Checkpoint, which: &str| -> Result<Dev<'d>, String> {
        let bytes = if full {
            bf(ck, &p(&format!("attn.{which}_sconv.static_w")), &[CONV_TAPS, kv])?
        } else {
            bf(ck, &p(&format!("attn.{which}_sconv.weight")), &[kv, 1, CONV_TAPS])?
        };
        Dev::upload(dev, &bytes)
    };
    let wo = bf(ck, &p("attn.wo_ud.weight"), &[HIDDEN, Q_WIDTH])?;
    let wo = Dev::upload(dev, &repack_bf16(&wo, &MatmulShape::new(HIDDEN, Q_WIDTH)?)?)?;
    let mlp_conv = ck.f32s(&p("mlp_sconv.weight"), &[HIDDEN, 1, CONV_TAPS])?;
    let router = bf(ck, &p("mlp.gate.weight"), &[EXPERTS, HIDDEN])?;
    let router = Dev::upload(dev, &repack_bf16(&router, &MatmulShape::new(EXPERTS, HIDDEN)?)?)?;
    let attn_queries = if l == 0 {
        None
    } else {
        let shape: &[usize] = if full { &[3, HIDDEN] } else { &[HIDDEN] };
        Some(Dev::upload_f32(
            dev,
            &ck.f32s(&llm(&format!("attn_res.attn_queries.{}", l - 1)), shape)?,
        )?)
    };
    let (gate_down, gate_up) = if (CYCLE_START..CYCLE_END).contains(&l) {
        let mut down = Vec::new();
        let mut up = Vec::new();
        for g in GATES {
            down.push(bf(ck, &p(&format!("controllers.{g}.down_weight")), &[64, 2 * HIDDEN])?);
        }
        let refs: Vec<&[u8]> = down.iter().map(Vec::as_slice).collect();
        let down = gate_down_weights(&refs)?;
        for g in GATES {
            let forget = bf(ck, &p(&format!("controllers.{g}.forget_weight")), &[HIDDEN, 64])?;
            let gain = bf(ck, &p(&format!("controllers.{g}.gain_weight")), &[HIDDEN, 64])?;
            up.extend(repack_gate_up(&forget, &gain)?);
        }
        (Some(Dev::upload(dev, &down)?), Some(Dev::upload(dev, &up)?))
    } else {
        (None, None)
    };
    Ok(LayerWeights {
        layer: l,
        full,
        j,
        input_norms,
        q_scale: Dev::upload(dev, &bf(ck, &p("attn.q_l2_clip.scale"), &[HEADS, HEAD_DIM])?)?,
        rel_scale: Dev::upload(dev, &bf(ck, &p("attn.rel_l2_clip.scale"), &[HEADS, D_REL])?)?,
        k_conv: conv(ck, "k")?,
        v_conv: conv(ck, "v")?,
        rel_proj: Dev::upload(dev, &bf(ck, &p("attn.rel_logits_proj.proj"), &[D_REL, REL_EXTENT])?)?,
        wo,
        attn_conv: Dev::upload(dev, &bf(ck, &p("attn_sconv.weight"), &[HIDDEN, 1, CONV_TAPS])?)?,
        mlp_conv: Dev::upload_f32(dev, &mlp_conv)?,
        attn_queries,
        mlp_query: Dev::upload_f32(dev, &ck.f32s(&llm(&format!("attn_res.mlp_queries.{l}")), &[HIDDEN])?)?,
        mlp_norm: Dev::upload_f32(dev, &ck.f32s(&p("mlp_norm.weight"), &[HIDDEN])?)?,
        router,
        router_bias: Dev::upload_f32(dev, &ck.f32s(&p("mlp.gate.bias"), &[EXPERTS])?)?,
        gate_down,
        gate_up,
        experts: experts(dev, ck, l)?,
    })
}

impl<'d> DeviceWeights<'d> {
    /// Read the checkpoint in the folder `dir` and upload every weight of the 24 layers, the embedding and the head.
    /// `progress` gets one line for each layer.
    pub fn load(dev: &'d CudaDevice, dir: &Path, mut progress: impl FnMut(&str)) -> Result<Self, String> {
        let mut ck = Checkpoint::open(dir)?;
        let llm = |name: &str| format!("model.llm.{name}");
        let embed_bytes = ck.bytes(&llm("embed.weight"), "BF16", &[VOCAB, HIDDEN])?;
        let embed = Dev::upload(dev, &embed_bytes)?;
        drop(embed_bytes);
        let unembed = ck.range(
            &llm("unembed.weight"),
            "BF16",
            &[VOCAB, HIDDEN],
            0,
            LOGIT_COLUMNS * HIDDEN * 2,
        )?;
        let head = Dev::upload(dev, &repack_bf16(&unembed, &MatmulShape::new(LOGIT_COLUMNS, HIDDEN)?)?)?;
        drop(unembed);
        let queries = ck.f32s(&llm("middle_cycle_cross_pass_queries"), &[2 * CYCLE_RUNS, HIDDEN])?;
        let kv_queries = ck.f32s(&llm("middle_cycle_cross_pass_kv_queries"), &[2 * CYCLE_RUNS, HIDDEN])?;
        let message_queries = (0..CYCLE_RUNS)
            .map(|l| Dev::upload_f32(dev, &message_queries_of(&queries, &kv_queries, l)))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = ck.bytes(&llm("norm.weight"), "BF16", &[HIDDEN])?;
        let mut weights = Self {
            embed,
            embed_norm: Dev::upload_f32(
                dev,
                &bf16_bytes_to_f32(&ck.bytes(&llm("embed_norm.weight"), "BF16", &[HIDDEN])?),
            )?,
            head,
            norm_bf16: Dev::upload(dev, &norm)?,
            output_query: Dev::upload_f32(dev, &ck.f32s(&llm("attn_res.output_query"), &[HIDDEN])?)?,
            message_queries,
            layers: Vec::with_capacity(LAYERS),
        };
        for l in 0..LAYERS {
            weights.layers.push(layer(dev, &mut ck, l)?);
            progress(&format!("layer {l} loaded"));
        }
        Ok(weights)
    }

    /// The bytes of device memory that the weights hold.
    pub fn bytes(&self) -> usize {
        let mut total = self.embed.bytes()
            + self.embed_norm.bytes()
            + self.head.bytes()
            + self.norm_bf16.bytes()
            + self.output_query.bytes()
            + self.message_queries.iter().map(Dev::bytes).sum::<usize>();
        for l in &self.layers {
            let e = &l.experts;
            total += l.j.bytes()
                + l.input_norms.iter().map(Dev::bytes).sum::<usize>()
                + l.q_scale.bytes()
                + l.rel_scale.bytes()
                + l.k_conv.bytes()
                + l.v_conv.bytes()
                + l.rel_proj.bytes()
                + l.wo.bytes()
                + l.attn_conv.bytes()
                + l.mlp_conv.bytes()
                + l.attn_queries.as_ref().map_or(0, Dev::bytes)
                + l.mlp_query.bytes()
                + l.mlp_norm.bytes()
                + l.router.bytes()
                + l.router_bias.bytes()
                + l.gate_down.as_ref().map_or(0, Dev::bytes)
                + l.gate_up.as_ref().map_or(0, Dev::bytes)
                + e.up.bytes()
                + e.down.bytes()
                + e.up_amax.as_ref().map_or(0, Dev::bytes)
                + e.down_amax.as_ref().map_or(0, Dev::bytes);
        }
        total
    }
}
