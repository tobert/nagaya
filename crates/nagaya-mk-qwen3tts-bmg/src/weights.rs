//! The weights of the engine on its card, and their load into an upload sink.
//!
//! The talker and the code predictor keep their BF16 tensors. The codec keeps its F32 tensors. The 16 codebooks of the
//! codec become one F32 table at load: row `c` of codebook `q` is the output projection of the codebook's embedding,
//! which the reference computes at each decode (`SplitResidualVectorQuantizer.decode`).

use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView};

use crate::spec::*;

/// The bytes of `f32` values.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The `f32` values of little-endian bytes.
pub fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// A tensor of a checkpoint with its dtype and shape checked.
pub(crate) fn tensor<'a>(
    view: &SafeTensorsView<'a>,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<&'a [u8], String> {
    let meta = view
        .tensors
        .get(name)
        .ok_or_else(|| format!("the checkpoint has no tensor {name}; use the Qwen3-TTS VoiceDesign checkpoint"))?;
    if meta.dtype != dtype || meta.shape != shape {
        return Err(format!(
            "tensor {name} is {:?} {:?}, and this engine needs {dtype:?} {shape:?}",
            meta.dtype, meta.shape
        ));
    }
    Ok(view.tensor_bytes(name).expect("parse checked the range"))
}

/// The weights of one layer of the talker or of the code predictor (BF16).
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub ln1: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub o: u64,
    pub ln2: u64,
    /// The gate rows, then the up rows, as one tensor: `OP_GEGLU` reads them so.
    pub gate_up: u64,
    pub down: u64,
}

/// One layer of the codec's transformer (F32).
#[derive(Clone, Copy, Debug)]
pub struct DecLayer {
    pub ln1: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub attn_scale: u64,
    pub ln2: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    pub mlp_scale: u64,
}

/// One upsampling stage of the codec: a transposed convolution, then a ConvNeXt block (F32).
#[derive(Clone, Copy, Debug)]
pub struct Upsample {
    pub ct_w: u64,
    pub ct_b: u64,
    pub dw_w: u64,
    pub dw_b: u64,
    pub ln_w: u64,
    pub ln_b: u64,
    pub pw1_w: u64,
    pub pw1_b: u64,
    pub pw2_w: u64,
    pub pw2_b: u64,
    pub gamma: u64,
}

/// One residual unit of a decoder block: SnakeBeta, a dilated convolution, SnakeBeta, a 1 x 1 convolution (F32).
#[derive(Clone, Copy, Debug)]
pub struct Unit {
    pub dilation: usize,
    pub a1: u64,
    pub b1: u64,
    pub c1_w: u64,
    pub c1_b: u64,
    pub a2: u64,
    pub b2: u64,
    pub c2_w: u64,
    pub c2_b: u64,
}

/// One decoder block: SnakeBeta, a transposed convolution, three residual units (F32).
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub alpha: u64,
    pub beta: u64,
    pub ct_w: u64,
    pub ct_b: u64,
    pub units: [Unit; 3],
}

/// The codec decoder on the card.
#[derive(Clone, Debug)]
pub struct Codec {
    /// The projected codebooks: `[16][2048][512]` F32.
    pub tables: u64,
    pub pre_conv_w: u64,
    pub pre_conv_b: u64,
    pub in_proj_w: u64,
    pub in_proj_b: u64,
    pub layers: Vec<DecLayer>,
    pub norm: u64,
    pub out_proj_w: u64,
    pub out_proj_b: u64,
    pub ups: Vec<Upsample>,
    pub dec0_w: u64,
    pub dec0_b: u64,
    pub blocks: Vec<Block>,
    pub final_alpha: u64,
    pub final_beta: u64,
    pub final_w: u64,
    pub final_b: u64,
}

/// The weights of the engine on its card.
#[derive(Clone, Debug)]
pub struct Weights {
    pub text_embed: u64,
    pub fc1_w: u64,
    pub fc1_b: u64,
    pub fc2_w: u64,
    pub fc2_b: u64,
    pub codec_embed: u64,
    pub layers: Vec<Layer>,
    pub norm: u64,
    pub head: u64,
    pub cp_proj_w: u64,
    pub cp_proj_b: u64,
    pub cp_layers: Vec<Layer>,
    pub cp_norm: u64,
    /// The head of codebook `g + 1` of the code predictor, `[2048][1024]` BF16.
    pub cp_heads: Vec<u64>,
    /// The embedding of codebook `g + 1`, `[2048][2048]` BF16.
    pub cp_embeds: Vec<u64>,
    pub codec: Codec,
}

/// The projected codebooks of the codec: `[16][2048][512]`, as the reference's quantizer decode computes them.
pub fn codebook_tables(view: &SafeTensorsView<'_>) -> Result<Vec<f32>, String> {
    let f = Dtype::F32;
    let mut inputs = Vec::with_capacity(CODEBOOKS);
    for q in 0..CODEBOOKS {
        let (group, layer) = if q == 0 { ("rvq_first", 0) } else { ("rvq_rest", q - 1) };
        let base = format!("decoder.quantizer.{group}");
        let book = format!("{base}.vq.layers.{layer}._codebook");
        inputs.push((
            f32_values(tensor(
                view,
                &format!("{book}.embedding_sum"),
                f,
                &[CODEBOOK_SIZE, VQ_DIM],
            )?),
            f32_values(tensor(view, &format!("{book}.cluster_usage"), f, &[CODEBOOK_SIZE])?),
            f32_values(tensor(
                view,
                &format!("{base}.output_proj.weight"),
                f,
                &[CODEBOOK_DIM, VQ_DIM, 1],
            )?),
        ));
    }
    // Each codebook is independent, so one thread computes each.
    let tables: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let threads: Vec<_> = inputs
            .iter()
            .map(|(sum, usage, proj)| {
                scope.spawn(move || {
                    let mut out = Vec::with_capacity(CODEBOOK_SIZE * CODEBOOK_DIM);
                    for c in 0..CODEBOOK_SIZE {
                        // EuclideanCodebook.decode: embedding_sum / cluster_usage.clamp(min=1e-5).
                        let inv = 1.0 / usage[c].max(1e-5);
                        let e: Vec<f32> = (0..VQ_DIM).map(|i| sum[c * VQ_DIM + i] * inv).collect();
                        for o in 0..CODEBOOK_DIM {
                            let row = &proj[o * VQ_DIM..(o + 1) * VQ_DIM];
                            out.push(row.iter().zip(&e).map(|(w, x)| w * x).sum());
                        }
                    }
                    out
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().expect("a codebook thread panicked"))
            .collect()
    });
    Ok(tables.concat())
}

/// Where the weights go: the card, or a count of the bytes for the plan, which reads no weight bytes (spec 2.4).
pub trait Upload {
    /// Device memory of `bytes` bytes, filled with zeros.
    fn alloc(&mut self, bytes: usize) -> u64;
    /// Copy `data` to the device at `at`.
    fn write(&mut self, at: u64, data: &[u8]);
    /// True when `write` copies. A count copies nothing, so the caller need not compute the data.
    fn writes(&self) -> bool;
    /// Device memory that holds `data`.
    fn upload(&mut self, data: &[u8]) -> u64 {
        let at = self.alloc(data.len());
        self.write(at, data);
        at
    }
}

/// The count of the device bytes of an upload, with no card.
#[derive(Default)]
pub struct Count {
    pub bytes: usize,
}

impl Upload for Count {
    fn alloc(&mut self, bytes: usize) -> u64 {
        self.bytes += crate::model::rounded(bytes);
        0
    }

    fn write(&mut self, _at: u64, _data: &[u8]) {}

    fn writes(&self) -> bool {
        false
    }
}

fn layer(
    sink: &mut dyn Upload,
    view: &SafeTensorsView<'_>,
    prefix: &str,
    hidden: usize,
    inter: usize,
) -> Result<Layer, String> {
    let bf = Dtype::BF16;
    let q_rows = HEADS * HEAD_DIM;
    let kv_rows = KV_HEADS * HEAD_DIM;
    let up = |sink: &mut dyn Upload, name: &str, shape: &[usize]| -> Result<u64, String> {
        Ok(sink.upload(tensor(view, &format!("{prefix}.{name}"), bf, shape)?))
    };
    let ln1 = up(sink, "input_layernorm.weight", &[hidden])?;
    let q = up(sink, "self_attn.q_proj.weight", &[q_rows, hidden])?;
    let k = up(sink, "self_attn.k_proj.weight", &[kv_rows, hidden])?;
    let v = up(sink, "self_attn.v_proj.weight", &[kv_rows, hidden])?;
    let q_norm = up(sink, "self_attn.q_norm.weight", &[HEAD_DIM])?;
    let k_norm = up(sink, "self_attn.k_norm.weight", &[HEAD_DIM])?;
    let o = up(sink, "self_attn.o_proj.weight", &[hidden, q_rows])?;
    let ln2 = up(sink, "post_attention_layernorm.weight", &[hidden])?;
    let down = up(sink, "mlp.down_proj.weight", &[hidden, inter])?;
    let gate = tensor(view, &format!("{prefix}.mlp.gate_proj.weight"), bf, &[inter, hidden])?;
    let up_rows = tensor(view, &format!("{prefix}.mlp.up_proj.weight"), bf, &[inter, hidden])?;
    let gate_up = sink.alloc(gate.len() + up_rows.len());
    sink.write(gate_up, gate);
    sink.write(gate_up + gate.len() as u64, up_rows);
    Ok(Layer {
        ln1,
        q,
        k,
        v,
        q_norm,
        k_norm,
        o,
        ln2,
        gate_up,
        down,
    })
}

/// Put the weights of the checkpoint into `sink`. The talker comes from `model`, the codec from `codec`.
pub fn load(
    sink: &mut dyn Upload,
    model: &SafeTensorsView<'_>,
    codec: &SafeTensorsView<'_>,
) -> Result<Weights, String> {
    let view = model;
    let bf = Dtype::BF16;
    let up = |sink: &mut dyn Upload, name: &str, shape: &[usize]| -> Result<u64, String> {
        Ok(sink.upload(tensor(view, name, bf, shape)?))
    };
    let text_embed = up(sink, "talker.model.text_embedding.weight", &[TEXT_VOCAB, HIDDEN])?;
    let fc1_w = up(sink, "talker.text_projection.linear_fc1.weight", &[HIDDEN, HIDDEN])?;
    let fc1_b = up(sink, "talker.text_projection.linear_fc1.bias", &[HIDDEN])?;
    let fc2_w = up(sink, "talker.text_projection.linear_fc2.weight", &[HIDDEN, HIDDEN])?;
    let fc2_b = up(sink, "talker.text_projection.linear_fc2.bias", &[HIDDEN])?;
    let codec_embed = up(sink, "talker.model.codec_embedding.weight", &[CODEC_VOCAB, HIDDEN])?;
    let norm = up(sink, "talker.model.norm.weight", &[HIDDEN])?;
    let head = up(sink, "talker.codec_head.weight", &[CODEC_VOCAB, HIDDEN])?;
    let cp_proj_w = up(
        sink,
        "talker.code_predictor.small_to_mtp_projection.weight",
        &[CP_HIDDEN, HIDDEN],
    )?;
    let cp_proj_b = up(sink, "talker.code_predictor.small_to_mtp_projection.bias", &[CP_HIDDEN])?;
    let cp_norm = up(sink, "talker.code_predictor.model.norm.weight", &[CP_HIDDEN])?;
    let mut cp_heads = Vec::with_capacity(CODEBOOKS - 1);
    let mut cp_embeds = Vec::with_capacity(CODEBOOKS - 1);
    for g in 0..CODEBOOKS - 1 {
        cp_heads.push(up(
            sink,
            &format!("talker.code_predictor.lm_head.{g}.weight"),
            &[CP_VOCAB, CP_HIDDEN],
        )?);
        cp_embeds.push(up(
            sink,
            &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
            &[CP_VOCAB, HIDDEN],
        )?);
    }
    let mut layers = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        layers.push(layer(sink, view, &format!("talker.model.layers.{l}"), HIDDEN, INTER)?);
    }
    let mut cp_layers = Vec::with_capacity(CP_LAYERS);
    for l in 0..CP_LAYERS {
        cp_layers.push(layer(
            sink,
            view,
            &format!("talker.code_predictor.model.layers.{l}"),
            CP_HIDDEN,
            CP_INTER,
        )?);
    }
    let codec = load_codec(sink, codec)?;
    Ok(Weights {
        text_embed,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        codec_embed,
        layers,
        norm,
        head,
        cp_proj_w,
        cp_proj_b,
        cp_layers,
        cp_norm,
        cp_heads,
        cp_embeds,
        codec,
    })
}

fn load_codec(sink: &mut dyn Upload, view: &SafeTensorsView<'_>) -> Result<Codec, String> {
    let f = Dtype::F32;
    let tables = sink.alloc(CODEBOOKS * CODEBOOK_SIZE * CODEBOOK_DIM * 4);
    if sink.writes() {
        sink.write(tables, &f32_bytes(&codebook_tables(view)?));
    }
    let up = |sink: &mut dyn Upload, name: &str, shape: &[usize]| -> Result<u64, String> {
        Ok(sink.upload(tensor(view, &format!("decoder.{name}"), f, shape)?))
    };
    let pre_conv_w = up(sink, "pre_conv.conv.weight", &[LATENT, CODEBOOK_DIM, 3])?;
    let pre_conv_b = up(sink, "pre_conv.conv.bias", &[LATENT])?;
    let in_proj_w = up(sink, "pre_transformer.input_proj.weight", &[DEC_HIDDEN, LATENT])?;
    let in_proj_b = up(sink, "pre_transformer.input_proj.bias", &[DEC_HIDDEN])?;
    let qd = DEC_HEADS * DEC_HEAD_DIM;
    let mut layers = Vec::with_capacity(DEC_LAYERS);
    for l in 0..DEC_LAYERS {
        let p = format!("pre_transformer.layers.{l}");
        layers.push(DecLayer {
            ln1: up(sink, &format!("{p}.input_layernorm.weight"), &[DEC_HIDDEN])?,
            q: up(sink, &format!("{p}.self_attn.q_proj.weight"), &[qd, DEC_HIDDEN])?,
            k: up(sink, &format!("{p}.self_attn.k_proj.weight"), &[qd, DEC_HIDDEN])?,
            v: up(sink, &format!("{p}.self_attn.v_proj.weight"), &[qd, DEC_HIDDEN])?,
            o: up(sink, &format!("{p}.self_attn.o_proj.weight"), &[DEC_HIDDEN, qd])?,
            attn_scale: up(sink, &format!("{p}.self_attn_layer_scale.scale"), &[DEC_HIDDEN])?,
            ln2: up(sink, &format!("{p}.post_attention_layernorm.weight"), &[DEC_HIDDEN])?,
            gate: up(sink, &format!("{p}.mlp.gate_proj.weight"), &[DEC_INTER, DEC_HIDDEN])?,
            up: up(sink, &format!("{p}.mlp.up_proj.weight"), &[DEC_INTER, DEC_HIDDEN])?,
            down: up(sink, &format!("{p}.mlp.down_proj.weight"), &[DEC_HIDDEN, DEC_INTER])?,
            mlp_scale: up(sink, &format!("{p}.mlp_layer_scale.scale"), &[DEC_HIDDEN])?,
        });
    }
    let norm = up(sink, "pre_transformer.norm.weight", &[DEC_HIDDEN])?;
    let out_proj_w = up(sink, "pre_transformer.output_proj.weight", &[LATENT, DEC_HIDDEN])?;
    let out_proj_b = up(sink, "pre_transformer.output_proj.bias", &[LATENT])?;
    let mut ups = Vec::with_capacity(UPSAMPLING_RATIOS.len());
    for (i, ratio) in UPSAMPLING_RATIOS.iter().enumerate() {
        let p = format!("upsample.{i}");
        ups.push(Upsample {
            ct_w: up(sink, &format!("{p}.0.conv.weight"), &[LATENT, LATENT, *ratio])?,
            ct_b: up(sink, &format!("{p}.0.conv.bias"), &[LATENT])?,
            dw_w: up(sink, &format!("{p}.1.dwconv.conv.weight"), &[LATENT, 1, 7])?,
            dw_b: up(sink, &format!("{p}.1.dwconv.conv.bias"), &[LATENT])?,
            ln_w: up(sink, &format!("{p}.1.norm.weight"), &[LATENT])?,
            ln_b: up(sink, &format!("{p}.1.norm.bias"), &[LATENT])?,
            pw1_w: up(sink, &format!("{p}.1.pwconv1.weight"), &[4 * LATENT, LATENT])?,
            pw1_b: up(sink, &format!("{p}.1.pwconv1.bias"), &[4 * LATENT])?,
            pw2_w: up(sink, &format!("{p}.1.pwconv2.weight"), &[LATENT, 4 * LATENT])?,
            pw2_b: up(sink, &format!("{p}.1.pwconv2.bias"), &[LATENT])?,
            gamma: up(sink, &format!("{p}.1.gamma"), &[LATENT])?,
        });
    }
    let dec0_w = up(sink, "decoder.0.conv.weight", &[DECODER_DIM, LATENT, 7])?;
    let dec0_b = up(sink, "decoder.0.conv.bias", &[DECODER_DIM])?;
    let mut blocks = Vec::with_capacity(UPSAMPLE_RATES.len());
    for (b, rate) in UPSAMPLE_RATES.iter().enumerate() {
        let (cin, cout) = (block_in(b), block_in(b + 1));
        let p = format!("decoder.{}.block", b + 1);
        let mut units = Vec::with_capacity(3);
        for (u, dilation) in [1usize, 3, 9].into_iter().enumerate() {
            let q = format!("{p}.{}", u + 2);
            units.push(Unit {
                dilation,
                a1: up(sink, &format!("{q}.act1.alpha"), &[cout])?,
                b1: up(sink, &format!("{q}.act1.beta"), &[cout])?,
                c1_w: up(sink, &format!("{q}.conv1.conv.weight"), &[cout, cout, 7])?,
                c1_b: up(sink, &format!("{q}.conv1.conv.bias"), &[cout])?,
                a2: up(sink, &format!("{q}.act2.alpha"), &[cout])?,
                b2: up(sink, &format!("{q}.act2.beta"), &[cout])?,
                c2_w: up(sink, &format!("{q}.conv2.conv.weight"), &[cout, cout, 1])?,
                c2_b: up(sink, &format!("{q}.conv2.conv.bias"), &[cout])?,
            });
        }
        blocks.push(Block {
            alpha: up(sink, &format!("{p}.0.alpha"), &[cin])?,
            beta: up(sink, &format!("{p}.0.beta"), &[cin])?,
            ct_w: up(sink, &format!("{p}.1.conv.weight"), &[cin, cout, 2 * rate])?,
            ct_b: up(sink, &format!("{p}.1.conv.bias"), &[cout])?,
            units: [units[0], units[1], units[2]],
        });
    }
    let last = block_in(UPSAMPLE_RATES.len());
    let final_alpha = up(sink, "decoder.5.alpha", &[last])?;
    let final_beta = up(sink, "decoder.5.beta", &[last])?;
    let final_w = up(sink, "decoder.6.conv.weight", &[1, last, 7])?;
    let final_b = up(sink, "decoder.6.conv.bias", &[1])?;
    Ok(Codec {
        tables,
        pre_conv_w,
        pre_conv_b,
        in_proj_w,
        in_proj_b,
        layers,
        norm,
        out_proj_w,
        out_proj_b,
        ups,
        dec0_w,
        dec0_b,
        blocks,
        final_alpha,
        final_beta,
        final_w,
        final_b,
    })
}
