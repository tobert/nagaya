//! The weights of the text tower on the card, and their load into an upload sink.
//!
//! Every tensor stays BF16. The gate and up rows of each MLP become one tensor, because `OP_GEGLU` reads the gate rows
//! and then the up rows. The layer scalars stay on the host: each one is the scale of one `OP_SCALE` task.

use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView, bf16_to_f32};

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

/// The device bytes of an allocation: at least 64, in steps of 64.
pub fn rounded(bytes: usize) -> usize {
    bytes.max(64).next_multiple_of(64)
}

/// A sink of the weights: device memory of the card, or a count of its bytes for the plan.
pub trait Upload {
    /// Device memory of `bytes` bytes, filled with zeros.
    fn alloc(&mut self, bytes: usize) -> u64;
    /// Copy `data` to the device at `at`.
    fn write(&mut self, at: u64, data: &[u8]);
    /// True when `write` copies. A count copies nothing, so the caller need not read the data.
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
        self.bytes += rounded(bytes);
        0
    }

    fn write(&mut self, _at: u64, _data: &[u8]) {}

    fn writes(&self) -> bool {
        false
    }
}

/// A BF16 tensor of the text tower with its shape checked.
fn tensor<'a>(view: &SafeTensorsView<'a>, name: &str, shape: &[usize]) -> Result<&'a [u8], String> {
    let full = format!("{PREFIX}{name}");
    let meta = view
        .tensors
        .get(&full)
        .ok_or_else(|| format!("the checkpoint has no tensor {full}; use the checkpoint google/embeddinggemma-2"))?;
    if meta.dtype != Dtype::BF16 || meta.shape != shape {
        return Err(format!(
            "tensor {full} is {:?} {:?}, and this engine needs BF16 {shape:?}",
            meta.dtype, meta.shape
        ));
    }
    Ok(view.tensor_bytes(&full).expect("parse checked the range"))
}

/// The weights of one layer on the card, and its layer scalar.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub input_norm: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub o: u64,
    pub post_attn_norm: u64,
    pub pre_ff_norm: u64,
    /// The gate rows, then the up rows: `[2 * 2048][512]`.
    pub gate_up: u64,
    pub down: u64,
    pub post_ff_norm: u64,
    pub ple_gate: u64,
    pub ple_proj: u64,
    pub ple_norm: u64,
    /// The layer scalar in f32. A count gives 0, because it reads no weight.
    pub scalar: f32,
}

/// The weights of the text tower on the card.
#[derive(Clone, Debug)]
pub struct Weights {
    /// The token table: `[262144][512]`.
    pub embed: u64,
    /// The per-layer projection: `[24 * 512][512]`.
    pub ple_proj: u64,
    pub ple_norm: u64,
    pub layers: Vec<Layer>,
    pub norm: u64,
    /// `embedding_projection`: `[768][512]`.
    pub projection: u64,
}

/// Load the text tower of `view` into `sink`. A tensor with another shape or dtype is an error before any write.
pub fn load(sink: &mut dyn Upload, view: &SafeTensorsView<'_>) -> Result<Weights, String> {
    // Check every tensor first, so that a wrong checkpoint uploads nothing.
    let mut names: Vec<(String, Vec<usize>)> = vec![
        ("embed_tokens.weight".into(), vec![VOCAB, HIDDEN]),
        (
            "ple.per_layer_model_projection.weight".into(),
            vec![LAYERS * PER_LAYER, HIDDEN],
        ),
        ("ple.per_layer_projection_norm.weight".into(), vec![PER_LAYER]),
        ("norm.weight".into(), vec![HIDDEN]),
        ("embedding_projection.weight".into(), vec![DIMENSION, HIDDEN]),
    ];
    for l in 0..LAYERS {
        names.extend(layer_tensors(l));
    }
    for (name, shape) in &names {
        tensor(view, name, shape)?;
    }
    let mut put = |name: &str, shape: &[usize]| -> Result<u64, String> {
        let bytes = tensor(view, name, shape)?;
        if sink.writes() {
            Ok(sink.upload(bytes))
        } else {
            Ok(sink.alloc(bytes.len()))
        }
    };
    let embed = put("embed_tokens.weight", &[VOCAB, HIDDEN])?;
    let ple_proj = put("ple.per_layer_model_projection.weight", &[LAYERS * PER_LAYER, HIDDEN])?;
    let ple_norm = put("ple.per_layer_projection_norm.weight", &[PER_LAYER])?;
    let norm = put("norm.weight", &[HIDDEN])?;
    let projection = put("embedding_projection.weight", &[DIMENSION, HIDDEN])?;
    let mut layers = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let a = attn(l);
        let (q_rows, kv_rows) = (HEADS * a.head_dim, a.kv_heads * a.head_dim);
        let p = format!("layers.{l}.");
        let name = |n: &str| format!("{p}{n}");
        let mut put = |n: &str, shape: &[usize]| -> Result<u64, String> {
            let bytes = tensor(view, &name(n), shape)?;
            if sink.writes() {
                Ok(sink.upload(bytes))
            } else {
                Ok(sink.alloc(bytes.len()))
            }
        };
        let input_norm = put("input_layernorm.weight", &[HIDDEN])?;
        let q = put("self_attn.q_proj.weight", &[q_rows, HIDDEN])?;
        let k = put("self_attn.k_proj.weight", &[kv_rows, HIDDEN])?;
        let v = put("self_attn.v_proj.weight", &[kv_rows, HIDDEN])?;
        let q_norm = put("self_attn.q_norm.weight", &[a.head_dim])?;
        let k_norm = put("self_attn.k_norm.weight", &[a.head_dim])?;
        let o = put("self_attn.o_proj.weight", &[HIDDEN, q_rows])?;
        let post_attn_norm = put("post_attention_layernorm.weight", &[HIDDEN])?;
        let pre_ff_norm = put("pre_feedforward_layernorm.weight", &[HIDDEN])?;
        let down = put("mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE])?;
        let post_ff_norm = put("post_feedforward_layernorm.weight", &[HIDDEN])?;
        let ple_gate = put("ple_block.per_layer_input_gate.weight", &[PER_LAYER, HIDDEN])?;
        let ple_proj = put("ple_block.per_layer_projection.weight", &[HIDDEN, PER_LAYER])?;
        let ple_norm = put("ple_block.post_per_layer_input_norm.weight", &[HIDDEN])?;
        let gate = tensor(view, &name("mlp.gate_proj.weight"), &[INTERMEDIATE, HIDDEN])?;
        let up = tensor(view, &name("mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN])?;
        let gate_up = sink.alloc(gate.len() + up.len());
        let mut scalar = 0.0;
        if sink.writes() {
            sink.write(gate_up, gate);
            sink.write(gate_up + gate.len() as u64, up);
            let bits = tensor(view, &name("layer_scalar"), &[1])?;
            scalar = bf16_to_f32(u16::from_le_bytes([bits[0], bits[1]]));
        }
        layers.push(Layer {
            input_norm,
            q,
            k,
            v,
            q_norm,
            k_norm,
            o,
            post_attn_norm,
            pre_ff_norm,
            gate_up,
            down,
            post_ff_norm,
            ple_gate,
            ple_proj,
            ple_norm,
            scalar,
        });
    }
    Ok(Weights {
        embed,
        ple_proj,
        ple_norm,
        layers,
        norm,
        projection,
    })
}

/// The tensors of layer `l` with their shapes.
fn layer_tensors(l: usize) -> Vec<(String, Vec<usize>)> {
    let a = attn(l);
    let (q_rows, kv_rows) = (HEADS * a.head_dim, a.kv_heads * a.head_dim);
    [
        ("input_layernorm.weight", vec![HIDDEN]),
        ("self_attn.q_proj.weight", vec![q_rows, HIDDEN]),
        ("self_attn.k_proj.weight", vec![kv_rows, HIDDEN]),
        ("self_attn.v_proj.weight", vec![kv_rows, HIDDEN]),
        ("self_attn.q_norm.weight", vec![a.head_dim]),
        ("self_attn.k_norm.weight", vec![a.head_dim]),
        ("self_attn.o_proj.weight", vec![HIDDEN, q_rows]),
        ("post_attention_layernorm.weight", vec![HIDDEN]),
        ("pre_feedforward_layernorm.weight", vec![HIDDEN]),
        ("mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
        ("post_feedforward_layernorm.weight", vec![HIDDEN]),
        ("ple_block.per_layer_input_gate.weight", vec![PER_LAYER, HIDDEN]),
        ("ple_block.per_layer_projection.weight", vec![HIDDEN, PER_LAYER]),
        ("ple_block.post_per_layer_input_norm.weight", vec![HIDDEN]),
        ("layer_scalar", vec![1]),
    ]
    .into_iter()
    .map(|(n, s)| (format!("layers.{l}.{n}"), s))
    .collect()
}
