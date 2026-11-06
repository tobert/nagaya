//! The weights of the backbone on the card, and their check and load into an upload sink.
//!
//! Every tensor stays BF16. The `w1` (gate) and `w3` (up) rows of each MLP become one tensor, because `OP_GEGLU` of
//! the LFM3 binary reads the gate rows and then the up rows. The head is the embedding table
//! (`tie_word_embeddings`), so the card holds it once.

use std::collections::BTreeSet;

use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView};

use crate::spec::{Backbone, LayerKind, Model};

/// Bytes of host memory that one upload copy goes through.
pub const STAGING_BYTES: usize = 64 << 20;

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

/// The weights of one layer on the card. A field of the other operator is 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layer {
    pub operator_norm: u64,
    pub post_operator_norm: u64,
    pub ffn_norm: u64,
    pub post_ffn_norm: u64,
    /// The gate rows (`w1`), then the up rows (`w3`): `[2 * intermediate][hidden]`.
    pub w13: u64,
    /// `w2`: `[hidden][intermediate]`.
    pub w2: u64,
    /// The rows B, C and x: `[3 * hidden][hidden]`.
    pub in_proj: u64,
    /// The taps of each channel: `[hidden][taps]`.
    pub conv: u64,
    pub out_proj: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    /// The gate of each head: `[heads][hidden]`.
    pub gate: u64,
}

/// The weights of the backbone on the card.
#[derive(Clone, Debug)]
pub struct Weights {
    /// The embedding table and the head: `[vocab_rows][hidden]`.
    pub embed: u64,
    /// The norm after the last layer (`embedding_norm`).
    pub norm: u64,
    pub layers: Vec<Layer>,
}

/// The tensors of the backbone with their shapes, by the name after the prefix.
pub fn tensors(b: &Backbone) -> Vec<(String, Vec<usize>)> {
    let (h, i) = (b.hidden, b.intermediate);
    let mut all = vec![
        ("embed_tokens.weight".to_owned(), vec![b.vocab_rows, h]),
        ("embedding_norm.weight".to_owned(), vec![h]),
    ];
    for (l, kind) in b.layers.iter().enumerate() {
        let mut layer = vec![
            ("operator_norm.weight", vec![h]),
            ("post_operator_norm.weight", vec![h]),
            ("ffn_norm.weight", vec![h]),
            ("post_ffn_norm.weight", vec![h]),
            ("feed_forward.w1.weight", vec![i, h]),
            ("feed_forward.w3.weight", vec![i, h]),
            ("feed_forward.w2.weight", vec![h, i]),
        ];
        match kind {
            LayerKind::Conv => layer.extend([
                ("conv.in_proj.weight", vec![3 * h, h]),
                ("conv.conv.weight", vec![h, 1, b.taps]),
                ("conv.out_proj.weight", vec![h, h]),
            ]),
            LayerKind::Full | LayerKind::Sliding => layer.extend([
                ("self_attn.q_proj.weight", vec![b.heads * b.head_dim, h]),
                ("self_attn.k_proj.weight", vec![b.kv_row(), h]),
                ("self_attn.v_proj.weight", vec![b.kv_row(), h]),
                ("self_attn.o_proj.weight", vec![h, b.heads * b.head_dim]),
                ("self_attn.q_norm.weight", vec![b.head_dim]),
                ("self_attn.k_norm.weight", vec![b.head_dim]),
                ("self_attn.gate_proj.weight", vec![b.heads, h]),
            ]),
        }
        all.extend(layer.into_iter().map(|(n, s)| (format!("layers.{l}.{n}"), s)));
    }
    all
}

/// Check the tensors of the backbone in `view`: each one is there as BF16 with its shape, and no other tensor has the
/// prefix of the backbone. A file of `LFM3-1.7B` holds the backbone only. The message names the tensor.
pub fn check(b: &Backbone, view: &SafeTensorsView<'_>) -> Result<(), String> {
    let prefix = b.model.prefix();
    let want = tensors(b);
    for (name, shape) in &want {
        let full = format!("{prefix}{name}");
        let meta = view
            .tensors
            .get(&full)
            .ok_or_else(|| format!("the checkpoint has no tensor {full}"))?;
        if meta.dtype != Dtype::BF16 || meta.shape != *shape {
            return Err(format!(
                "tensor {full} is {:?} {:?}, and this engine needs BF16 {shape:?}",
                meta.dtype, meta.shape
            ));
        }
    }
    let known: BTreeSet<String> = want.iter().map(|(n, _)| format!("{prefix}{n}")).collect();
    let mut other: Vec<&String> = view
        .tensors
        .keys()
        .filter(|name| !known.contains(*name) && (name.starts_with(prefix) || b.model == Model::Text))
        .collect();
    other.sort();
    if let Some(name) = other.first() {
        return Err(format!(
            "the checkpoint has the tensor {name}, which the LFM3 backbone does not use; use the checkpoint of \
             LFM3-1.7B or LFM3-Omni-2B"
        ));
    }
    Ok(())
}

/// Load the backbone of `view` into `sink`. A checkpoint that fails `check` uploads nothing.
pub fn load(sink: &mut dyn Upload, b: &Backbone, view: &SafeTensorsView<'_>) -> Result<Weights, String> {
    check(b, view)?;
    let prefix = b.model.prefix();
    let bytes = |name: &str| -> &[u8] {
        view.tensor_bytes(&format!("{prefix}{name}"))
            .expect("check found each tensor")
    };
    let mut put = |names: &[&str]| -> u64 {
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
    let embed = put(&["embed_tokens.weight"]);
    let norm = put(&["embedding_norm.weight"]);
    let mut layers = Vec::with_capacity(b.layers.len());
    for (l, kind) in b.layers.iter().enumerate() {
        let n = |name: &str| format!("layers.{l}.{name}");
        let mut layer = Layer {
            operator_norm: put(&[&n("operator_norm.weight")]),
            post_operator_norm: put(&[&n("post_operator_norm.weight")]),
            ffn_norm: put(&[&n("ffn_norm.weight")]),
            post_ffn_norm: put(&[&n("post_ffn_norm.weight")]),
            w13: put(&[&n("feed_forward.w1.weight"), &n("feed_forward.w3.weight")]),
            w2: put(&[&n("feed_forward.w2.weight")]),
            ..Layer::default()
        };
        match kind {
            LayerKind::Conv => {
                layer.in_proj = put(&[&n("conv.in_proj.weight")]);
                layer.conv = put(&[&n("conv.conv.weight")]);
                layer.out_proj = put(&[&n("conv.out_proj.weight")]);
            }
            LayerKind::Full | LayerKind::Sliding => {
                layer.q = put(&[&n("self_attn.q_proj.weight")]);
                layer.k = put(&[&n("self_attn.k_proj.weight")]);
                layer.v = put(&[&n("self_attn.v_proj.weight")]);
                layer.o = put(&[&n("self_attn.o_proj.weight")]);
                layer.q_norm = put(&[&n("self_attn.q_norm.weight")]);
                layer.k_norm = put(&[&n("self_attn.k_norm.weight")]);
                layer.gate = put(&[&n("self_attn.gate_proj.weight")]);
            }
        }
        layers.push(layer);
    }
    Ok(Weights { embed, norm, layers })
}
