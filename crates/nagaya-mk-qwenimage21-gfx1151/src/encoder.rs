//! The text tower of Qwen3-VL-8B on gfx1151: the encoder of Qwen-Image 2.1 (spec 11.9c).
//!
//! The encoder runs the 36 layers over the template ids. Its output is the input of the final norm, which it does
//! not load (spec 11.9c, "The load"). The conditioning is that output after the cut.

use std::ffi::c_void;
use std::fmt;

use nagaya_mk_hip::attention::{Attention, AttentionShape, Mask};
use nagaya_mk_hip::gemm::{GemmError, Gemms};
use nagaya_mk_hip::hip::{DeviceBuffer, HipError, Stream};
use nagaya_mk_hip::ops::{self, RopePairs};
use nagaya_mk_hip::plan::{DataType, GemmShape};
use nagaya_mk_hip::rope;

use crate::names::{EPS, EncoderNames, FFN, HEAD_DIM, HEADS, HIDDEN, KV_HEADS, LAYERS, THETA, VOCAB};
use crate::safetensors::{Dtype, Safetensors, SafetensorsError};

/// The width of the fused q, k and v projection.
const QKV: usize = (HEADS + 2 * KV_HEADS) * HEAD_DIM;

/// An error of the encoder.
#[derive(Debug)]
pub enum EncoderError {
    File(SafetensorsError),
    Names(String),
    Hip(HipError),
    Gemm(GemmError),
    /// An id is not a row of the embedding table, or there are no ids.
    Ids(String),
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncoderError::File(e) => write!(f, "{e}"),
            EncoderError::Names(e) => write!(f, "the encoder file: {e}"),
            EncoderError::Hip(e) => write!(f, "{e}"),
            EncoderError::Gemm(e) => write!(f, "{e}"),
            EncoderError::Ids(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EncoderError {}

impl From<SafetensorsError> for EncoderError {
    fn from(e: SafetensorsError) -> Self {
        EncoderError::File(e)
    }
}

impl From<HipError> for EncoderError {
    fn from(e: HipError) -> Self {
        EncoderError::Hip(e)
    }
}

impl From<GemmError> for EncoderError {
    fn from(e: GemmError) -> Self {
        EncoderError::Gemm(e)
    }
}

/// The weights of one layer in device memory. q, k and v are one matrix, and so are gate and up.
struct Layer {
    input_norm: DeviceBuffer,
    qkv: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    o: DeviceBuffer,
    post_norm: DeviceBuffer,
    gate_up: DeviceBuffer,
    down: DeviceBuffer,
}

/// The text tower in device memory.
pub struct Encoder {
    /// True when attention runs in the fused flash kernel, false for the GEMMs and the softmax kernel.
    pub flash: bool,
    embed: DeviceBuffer,
    layers: Vec<Layer>,
}

fn at(buffer: &DeviceBuffer, elements: usize) -> *mut c_void {
    buffer.ptr().cast::<u16>().wrapping_add(elements).cast()
}

impl Encoder {
    /// Load the text tower from `file`. It checks every name first, and reads one tensor at a time.
    pub fn load(file: &Safetensors) -> Result<Encoder, EncoderError> {
        let names = EncoderNames::check(file.tensors().keys().map(String::as_str)).map_err(EncoderError::Names)?;
        let read = |name: String, shape: &[usize]| file.read(&name, Dtype::Bf16, shape);
        let upload = |name: String, shape: &[usize]| -> Result<DeviceBuffer, EncoderError> {
            Ok(DeviceBuffer::from_slice(&read(name, shape)?)?)
        };
        let stack = |parts: &[(String, &[usize])]| -> Result<DeviceBuffer, EncoderError> {
            let mut bytes = Vec::new();
            for (name, shape) in parts {
                bytes.extend_from_slice(&read(name.clone(), shape)?);
            }
            Ok(DeviceBuffer::from_slice(&bytes)?)
        };
        let embed = upload(names.embed(), &[VOCAB, HIDDEN])?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let n = |part: &str| names.layer(i, part);
            layers.push(Layer {
                input_norm: upload(n("input_layernorm.weight"), &[HIDDEN])?,
                qkv: stack(&[
                    (n("self_attn.q_proj.weight"), &[HEADS * HEAD_DIM, HIDDEN]),
                    (n("self_attn.k_proj.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN]),
                    (n("self_attn.v_proj.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN]),
                ])?,
                q_norm: upload(n("self_attn.q_norm.weight"), &[HEAD_DIM])?,
                k_norm: upload(n("self_attn.k_norm.weight"), &[HEAD_DIM])?,
                o: upload(n("self_attn.o_proj.weight"), &[HIDDEN, HEADS * HEAD_DIM])?,
                post_norm: upload(n("post_attention_layernorm.weight"), &[HIDDEN])?,
                gate_up: stack(&[
                    (n("mlp.gate_proj.weight"), &[FFN, HIDDEN]),
                    (n("mlp.up_proj.weight"), &[FFN, HIDDEN]),
                ])?,
                down: upload(n("mlp.down_proj.weight"), &[HIDDEN, FFN])?,
            });
        }
        Ok(Encoder {
            flash: false,
            embed,
            layers,
        })
    }

    /// The bytes of the weights in device memory.
    pub fn bytes(&self) -> usize {
        self.embed.bytes()
            + self
                .layers
                .iter()
                .map(|l| {
                    l.input_norm.bytes()
                        + l.qkv.bytes()
                        + l.q_norm.bytes()
                        + l.k_norm.bytes()
                        + l.o.bytes()
                        + l.post_norm.bytes()
                        + l.gate_up.bytes()
                        + l.down.bytes()
                })
                .sum::<usize>()
    }

    /// The scores that the attention of `ids` needs: one head of `ids` by `ids`, times the heads of a key group.
    pub fn attention_capacity(ids: usize) -> usize {
        ids * ids * (HEADS / KV_HEADS)
    }

    /// Run the text tower over `ids`. It gives the input of the final norm: `[ids, 4096]` in bf16, in device memory.
    pub fn encode(
        &self,
        gemms: &mut Gemms,
        attention: &Attention,
        ids: &[u32],
        stream: &Stream,
    ) -> Result<DeviceBuffer, EncoderError> {
        let l = ids.len();
        if l == 0 {
            return Err(EncoderError::Ids("the encoder needs at least one id".to_owned()));
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= VOCAB) {
            return Err(EncoderError::Ids(format!(
                "the id {bad} is not a row of the table of {VOCAB}"
            )));
        }
        let rows = l as u32;
        let ids_device = DeviceBuffer::from_slice(&ids.iter().map(|&id| id as i32).collect::<Vec<_>>())?;
        let positions: Vec<i64> = (0..l as i64).collect();
        let table = DeviceBuffer::from_slice(&rope::half_table(&positions, HEAD_DIM, THETA))?;
        let h = DeviceBuffer::new(l * HIDDEN * 2)?;
        let r = DeviceBuffer::new(l * HIDDEN * 2)?;
        let qkv = DeviceBuffer::new(l * QKV * 2)?;
        let mixed = DeviceBuffer::new(l * HIDDEN * 2)?;
        let gate_up = DeviceBuffer::new(l * 2 * FFN * 2)?;
        let act = DeviceBuffer::new(l * FFN * 2)?;
        let attend = AttentionShape {
            lq: l,
            lk: l,
            heads: HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            q_stride: QKV,
            k_stride: QKV,
            v_stride: QKV,
            out_stride: HIDDEN,
            mask: Mask::Causal { offset: 0 },
        };
        let qkv_shape = GemmShape::linear(rows, HIDDEN as u32, QKV as u32, DataType::Bf16);
        let o_shape = GemmShape::linear(rows, (HEADS * HEAD_DIM) as u32, HIDDEN as u32, DataType::Bf16);
        let up_shape = GemmShape::linear(rows, HIDDEN as u32, (2 * FFN) as u32, DataType::Bf16);
        let down_shape = GemmShape::linear(rows, FFN as u32, HIDDEN as u32, DataType::Bf16);
        // SAFETY: each buffer above holds the rows of its use, and the weights hold the shapes of `load`.
        unsafe {
            ops::gather_rows(self.embed.ptr(), ids_device.ptr(), h.ptr(), l, HIDDEN, stream)?;
            for layer in &self.layers {
                ops::rmsnorm(
                    h.ptr(),
                    HIDDEN,
                    layer.input_norm.ptr(),
                    0.0,
                    r.ptr(),
                    HIDDEN,
                    l,
                    HIDDEN,
                    EPS,
                    stream,
                )?;
                gemms.run(
                    &qkv_shape,
                    1.0,
                    layer.qkv.ptr(),
                    r.ptr(),
                    0.0,
                    qkv.ptr(),
                    qkv.ptr(),
                    stream,
                )?;
                let k = at(&qkv, HEADS * HEAD_DIM);
                let v = at(&qkv, (HEADS + KV_HEADS) * HEAD_DIM);
                let rope = |x, heads, w: &DeviceBuffer| {
                    ops::head_norm_rope(
                        x,
                        QKV,
                        l,
                        heads,
                        HEAD_DIM,
                        w.ptr(),
                        EPS,
                        table.ptr(),
                        RopePairs::Halves,
                        stream,
                    )
                };
                rope(qkv.ptr(), HEADS, &layer.q_norm)?;
                rope(k, KV_HEADS, &layer.k_norm)?;
                if self.flash {
                    nagaya_mk_hip::attention::flash(&attend, qkv.ptr(), k, v, mixed.ptr(), stream)?;
                } else {
                    attention.run(gemms, &attend, qkv.ptr(), k, v, mixed.ptr(), stream)?;
                }
                // The residual adds in the GEMM: h = h + mixed o^T.
                gemms.run(&o_shape, 1.0, layer.o.ptr(), mixed.ptr(), 1.0, h.ptr(), h.ptr(), stream)?;
                ops::rmsnorm(
                    h.ptr(),
                    HIDDEN,
                    layer.post_norm.ptr(),
                    0.0,
                    r.ptr(),
                    HIDDEN,
                    l,
                    HIDDEN,
                    EPS,
                    stream,
                )?;
                gemms.run(
                    &up_shape,
                    1.0,
                    layer.gate_up.ptr(),
                    r.ptr(),
                    0.0,
                    gate_up.ptr(),
                    gate_up.ptr(),
                    stream,
                )?;
                ops::silu_mul(gate_up.ptr(), 2 * FFN, FFN, act.ptr(), FFN, l, stream)?;
                gemms.run(
                    &down_shape,
                    1.0,
                    layer.down.ptr(),
                    act.ptr(),
                    1.0,
                    h.ptr(),
                    h.ptr(),
                    stream,
                )?;
            }
        }
        stream.synchronize()?;
        Ok(h)
    }
}
