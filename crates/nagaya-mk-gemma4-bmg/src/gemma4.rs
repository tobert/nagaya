//! Gemma 4 26B-A4B-it text model across both B70s, split by layer: card 0 runs layers 0-14,
//! card 1 runs layers 15-29 and the head (final norm and tied unembedding). Every layer holds
//! all 128 routed experts in BF16.
//!
//! A forward pass runs the prompt in blocks of up to `max_block` tokens. For each block the
//! host writes the scaled token embeddings into card 0's residual, runs card 0, copies the
//! residual to card 1, and runs card 1. The copy goes through host memory, one way; the cards
//! never wait on each other inside a launch, and each keeps its own Level Zero context.
//!
//! The math follows HF transformers' `Gemma4TextDecoderLayer` (see tools/gemma4_reference.py).

#![allow(dead_code)]

#[cfg(feature = "xe")]
use crate::level_zero::{B70Card, L0Cluster};
use crate::protocol::*;
use crate::safetensors::bf16_to_f32;
use crate::sharding::Checkpoint;
#[cfg(feature = "xe")]
use std::collections::HashMap;
#[cfg(feature = "xe")]
use std::ffi::c_void;
use std::ops::Range;
use std::path::Path;
#[cfg(feature = "xe")]
use std::time::Instant;

pub const HIDDEN: usize = 2816;
pub const NUM_LAYERS: usize = 30;
pub const HEADS: usize = 16;
pub const VOCAB: usize = 262144;
pub const SHARED_I: usize = 2112;
pub const MOE_I: usize = 704;
pub const SLIDING_WINDOW: u32 = 1024;
/// A sliding layer of a slot longer than this keeps its KV in a ring of this many rows: a key's
/// row is its position modulo the ring. The ring holds a block's keys and the window before
/// them, and a multiple of 32 so no key tile straddles its end.
pub const KV_RING_ROWS: usize = 4096;
/// Rows past a ring that a block may write before OP_COPY moves them to the ring's start; the
/// kernel's KV_RING_SPILL.
pub const KV_RING_SPILL: usize = 1024;
const _: () = assert!(
    KV_RING_ROWS % 32 == 0
        && KV_RING_ROWS >= MAX_PLAN_TOKENS + SLIDING_WINDOW as usize + 32
        && KV_RING_SPILL >= MAX_PLAN_TOKENS
);

/// The ring rows of layer `l`'s KV in a slot of `seq` positions, or 0 when its rows are
/// positions.
pub fn kv_ring(seq: usize, l: usize) -> usize {
    if !layer_geom(l).global && seq > KV_RING_ROWS {
        KV_RING_ROWS
    } else {
        0
    }
}

/// Rows of each of layer `l`'s K and V planes in a slot of `seq` positions.
fn kv_plane_rows(seq: usize, l: usize) -> usize {
    match kv_ring(seq, l) {
        0 => seq,
        r => r + KV_RING_SPILL,
    }
}

/// The first position of layer `l` a slot of `seq` positions holding `len` must keep to go on
/// from `len`: 0, or with a ring, the last ring's worth.
pub fn kv_keep_from(seq: usize, l: usize, len: usize) -> usize {
    match kv_ring(seq, l) {
        0 => 0,
        r => len.saturating_sub(r),
    }
}

/// Whether a slot of `seq` positions whose ring rows run up to position `high` can go on from
/// `keep`: the window before `keep` must still be in the ring. Past that, the positions are
/// fed again.
pub fn kv_can_rewind(seq: usize, high: usize, keep: usize) -> bool {
    seq <= KV_RING_ROWS || high.saturating_sub(keep) + 16 <= KV_RING_ROWS - SLIDING_WINDOW as usize
}

/// The `high` a ring of `ring` rows has after a snapshot of `len` positions is loaded with
/// positions `need..len`. A snapshot that holds every row from 0 leaves the ring as if written
/// to its end; one that starts later, as if written to `need + ring`, so a rewind never reads a
/// window before `need`.
pub fn kv_high_after_load(need: usize, len: usize, ring: usize) -> usize {
    if need == 0 { len } else { need + ring }
}
/// Most rows one block can take logits for: one per continuation in `forward_branches`.
pub const MAX_LOGIT_ROWS: usize = 16;
const PREFIX: &str = "model.language_model.";
/// The name of the embedding table in the checkpoint.
pub const EMBED_TENSOR: &str = "model.language_model.embed_tokens.weight";

/// Layer split: card 0 holds `CARD_LAYERS[0]`, card 1 holds `CARD_LAYERS[1]` and the head.
pub const CARD_LAYERS: [Range<usize>; 2] = [0..15, 15..30];

#[derive(Clone, Copy, Debug)]
pub struct LayerGeom {
    pub global: bool,
    pub head_dim: usize,
    pub kv_heads: usize,
}

pub fn layer_geom(l: usize) -> LayerGeom {
    if l % 6 == 5 {
        LayerGeom {
            global: true,
            head_dim: 512,
            kv_heads: 2,
        }
    } else {
        LayerGeom {
            global: false,
            head_dim: 256,
            kv_heads: 8,
        }
    }
}

/// Fails unless `config.json` describes the model these constants and kernels implement.
pub fn check_config(model_dir: &Path) -> Result<(), String> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("{}: {}", path.display(), e))?;
    let t = &v["text_config"];
    let want = serde_json::json!({
        "hidden_size": HIDDEN, "num_hidden_layers": NUM_LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": 8, "num_global_key_value_heads": 2, "head_dim": 256,
        "global_head_dim": 512, "intermediate_size": SHARED_I, "moe_intermediate_size": MOE_I,
        "num_experts": NUM_EXPERTS, "top_k_experts": TOP_K, "vocab_size": VOCAB,
        "sliding_window": SLIDING_WINDOW, "final_logit_softcapping": 30.0, "attention_k_eq_v": true,
        "enable_moe_block": true, "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
        "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh", "attention_bias": false,
        "use_double_wide_mlp": false,
    });
    for (key, expected) in want.as_object().unwrap() {
        if &t[key] != expected {
            return Err(format!(
                "{}: text_config.{} is {}, expected {}",
                path.display(),
                key,
                t[key],
                expected
            ));
        }
    }
    let rope = &t["rope_parameters"];
    let rope_ok = rope["full_attention"]["rope_type"] == "proportional"
        && rope["full_attention"]["partial_rotary_factor"] == 0.25
        && rope["full_attention"]["rope_theta"] == 1000000.0
        && rope["sliding_attention"]["rope_type"] == "default"
        && rope["sliding_attention"]["rope_theta"] == 10000.0;
    if !rope_ok {
        return Err(format!("{}: unexpected rope_parameters {}", path.display(), rope));
    }
    let types = t["layer_types"].as_array().ok_or("text_config.layer_types missing")?;
    for (l, ty) in types.iter().enumerate() {
        let want = if layer_geom(l).global {
            "full_attention"
        } else {
            "sliding_attention"
        };
        if ty != want {
            return Err(format!("{}: layer {} is {}, expected {}", path.display(), l, ty, want));
        }
    }
    Ok(())
}

/// Device pointers into one layer's DMA-BUF-exportable weight slab.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerWeights {
    pub slab: u64,
    pub slab_bytes: usize,
    pub dmabuf_fd: i32,
    pub input_ln: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub o: u64,
    /// post_attention_layernorm, pre_feedforward_layernorm, pre_feedforward_layernorm_2, in a row
    pub attn_out_lns: u64,
    pub mlp_gate_up: u64,
    pub mlp_down: u64,
    pub router: u64,
    pub experts_gate_up: u64,
    pub experts_down: u64,
    /// post_feedforward_layernorm_1, _2, and post_feedforward_layernorm, in a row
    pub ffn_out_lns: u64,
    pub layer_scalar: f32,
}

fn align64(x: usize) -> usize {
    (x + 63) & !63
}

const STAGING_BYTES: usize = 64 << 20;
/// Floats of a shard's staging buffer: host-to-device copies up to this size go through it.
const STAGE_FLOATS: usize = 1 << 20;

#[cfg(feature = "xe")]
/// Copies host bytes (such as an mmap of a checkpoint) to device memory through a pinned
/// host USM buffer. Copying straight from file-backed pages loses the device.
struct Uploader {
    staging: *mut u8,
}

#[cfg(feature = "xe")]
impl Uploader {
    fn new(card: &B70Card) -> Self {
        Self {
            staging: card.alloc_host_usm::<u8>(STAGING_BYTES),
        }
    }

    /// Free the staging buffer. The weights are on the card, so the engine needs it no more.
    fn free(self, card: &B70Card) {
        card.free(self.staging as *mut c_void);
    }

    fn copy(&self, card: &B70Card, dst: u64, src: &[u8]) {
        for (i, chunk) in src.chunks(STAGING_BYTES).enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(chunk.as_ptr(), self.staging, chunk.len()) };
            card.memcpy_sync(
                (dst as usize + i * STAGING_BYTES) as *mut c_void,
                self.staging as *const c_void,
                chunk.len(),
            );
        }
    }
}

/// Uploads layer `l` into one slab: each group of tensors starts 64-byte aligned, and tensors
/// within a group are contiguous (the kernels index gate/up and the router pack that way).
/// Layer `l`'s slab: each group of tensors (checkpoint bytes) at its offset, the slab's size,
/// and the layer scalar.
pub(crate) struct LayerLayout<'a> {
    groups: Vec<(&'static str, Vec<&'a [u8]>)>,
    offsets: Vec<usize>,
    pub(crate) total: usize,
    layer_scalar: f32,
}

pub(crate) fn layer_layout(ckpt: &Checkpoint, l: usize) -> Result<LayerLayout<'_>, String> {
    let g = layer_geom(l);
    let p = format!("{}layers.{}.", PREFIX, l);
    let (d, hd, kv) = (HIDDEN, g.head_dim, g.kv_heads);
    let q_dim = HEADS * hd;
    let t = |name: &str, shape: &[usize]| ckpt.bf16(&format!("{}{}", p, name), shape);

    let mut groups: Vec<(&'static str, Vec<&[u8]>)> = vec![
        ("input_ln", vec![t("input_layernorm.weight", &[d])?]),
        ("q", vec![t("self_attn.q_proj.weight", &[q_dim, d])?]),
        ("k", vec![t("self_attn.k_proj.weight", &[kv * hd, d])?]),
        ("q_norm", vec![t("self_attn.q_norm.weight", &[hd])?]),
        ("k_norm", vec![t("self_attn.k_norm.weight", &[hd])?]),
        ("o", vec![t("self_attn.o_proj.weight", &[d, q_dim])?]),
        // OP_ATTN_OUT reads these three norms' weights from one pointer.
        (
            "attn_out_lns",
            vec![
                t("post_attention_layernorm.weight", &[d])?,
                t("pre_feedforward_layernorm.weight", &[d])?,
                t("pre_feedforward_layernorm_2.weight", &[d])?,
            ],
        ),
        (
            "mlp_gate_up",
            vec![
                t("mlp.gate_proj.weight", &[SHARED_I, d])?,
                t("mlp.up_proj.weight", &[SHARED_I, d])?,
            ],
        ),
        ("mlp_down", vec![t("mlp.down_proj.weight", &[d, SHARED_I])?]),
        (
            "router",
            vec![
                t("router.proj.weight", &[NUM_EXPERTS, d])?,
                t("router.scale", &[d])?,
                t("router.per_expert_scale", &[NUM_EXPERTS])?,
            ],
        ),
        (
            "experts_gate_up",
            vec![t("experts.gate_up_proj", &[NUM_EXPERTS, 2 * MOE_I, d])?],
        ),
        ("experts_down", vec![t("experts.down_proj", &[NUM_EXPERTS, d, MOE_I])?]),
        // OP_FFN_OUT reads these three norms' weights from one pointer.
        (
            "ffn_out_lns",
            vec![
                t("post_feedforward_layernorm_1.weight", &[d])?,
                t("post_feedforward_layernorm_2.weight", &[d])?,
                t("post_feedforward_layernorm.weight", &[d])?,
            ],
        ),
    ];
    let v_name = format!("{}self_attn.v_proj.weight", p);
    if g.global {
        if ckpt.contains(&v_name) {
            return Err(format!("layer {} is global (attention_k_eq_v) but has {}", l, v_name));
        }
    } else {
        groups.push(("v", vec![t("self_attn.v_proj.weight", &[kv * hd, d])?]));
    }
    let scalar_bytes = t("layer_scalar", &[1])?;
    let expected_tensors = groups.iter().map(|(_, parts)| parts.len()).sum::<usize>() + 1;
    let found = ckpt.names().filter(|n| n.starts_with(&p)).count();
    if found != expected_tensors {
        return Err(format!(
            "layer {}: checkpoint has {} tensors, the graph uses {}",
            l, found, expected_tensors
        ));
    }

    let mut offsets = Vec::with_capacity(groups.len());
    let mut total = 0usize;
    for (_, parts) in &groups {
        total = align64(total);
        offsets.push(total);
        total += parts.iter().map(|b| b.len()).sum::<usize>();
    }
    let layer_scalar = bf16_to_f32(u16::from_le_bytes([scalar_bytes[0], scalar_bytes[1]]));
    Ok(LayerLayout {
        groups,
        offsets,
        total,
        layer_scalar,
    })
}

/// The head slab: the final norm at offset 0, then the embedding table at the next multiple of 64 bytes. The port
/// gives the head a slab, so that a restart keeps it (spec 10.2). mkviolet uploads the head again at each start.
pub(crate) fn head_layout(ckpt: &Checkpoint) -> Result<LayerLayout<'_>, String> {
    let norm = ckpt.bf16(&format!("{}norm.weight", PREFIX), &[HIDDEN])?;
    let embed = ckpt.bf16(&format!("{}embed_tokens.weight", PREFIX), &[VOCAB, HIDDEN])?;
    let embed_at = align64(norm.len());
    Ok(LayerLayout {
        groups: vec![("final_norm", vec![norm]), ("embed", vec![embed])],
        offsets: vec![0, embed_at],
        total: embed_at + embed.len(),
        layer_scalar: 0.0,
    })
}

/// Checks that a stored slab (a DMA-BUF fd from systemd's fd store) holds layer `l` of this
/// checkpoint: imports it, compares the first and last 64 bytes of every tensor with the
/// checkpoint, and frees the import.
#[cfg(feature = "xe")]
pub fn verify_stored_slab(card: &B70Card, ckpt: &Checkpoint, l: usize, fd: i32) -> Result<(), String> {
    verify_layout(card, &layer_layout(ckpt, l)?, &format!("layer {l}"), fd)
}

/// Checks that a stored head slab holds the final norm and the embedding table of this checkpoint, with the spot
/// check of `verify_stored_slab`.
#[cfg(feature = "xe")]
pub fn verify_stored_head(card: &B70Card, ckpt: &Checkpoint, fd: i32) -> Result<(), String> {
    verify_layout(card, &head_layout(ckpt)?, "the head", fd)
}

/// Import a stored slab of the layout `lay`, compare the first and last 64 bytes of each tensor, and free the
/// import.
#[cfg(feature = "xe")]
fn verify_layout(card: &B70Card, lay: &LayerLayout<'_>, what: &str, fd: i32) -> Result<(), String> {
    let base = card.try_import_vram_from_dmabuf_fd::<u8>(card.context, fd, lay.total)? as u64;
    let mut bad = None;
    'groups: for ((name, parts), &off) in lay.groups.iter().zip(&lay.offsets) {
        let mut at = off;
        for bytes in parts {
            for range in [0..bytes.len().min(64), bytes.len().saturating_sub(64)..bytes.len()] {
                let mut got = vec![0u8; range.len()];
                card.memcpy_sync(
                    got.as_mut_ptr() as *mut c_void,
                    (base + (at + range.start) as u64) as *const c_void,
                    got.len(),
                );
                if got != bytes[range] {
                    bad = Some(format!("{what} {name} differs from the checkpoint"));
                    break 'groups;
                }
            }
            at += bytes.len();
        }
    }
    card.free(base as *mut c_void);
    bad.map_or(Ok(()), Err)
}

/// Layer `l`'s weights in one slab: imported from `stored` (a slab fd verify_stored_slab
/// accepted), or uploaded into a new DMA-BUF exportable slab.
#[cfg(feature = "xe")]
fn upload_layer(
    card: &B70Card,
    up: &Uploader,
    ckpt: &Checkpoint,
    l: usize,
    stored: Option<i32>,
) -> Result<LayerWeights, String> {
    let lay = layer_layout(ckpt, l)?;
    let (base, fd) = match stored {
        Some(fd) => (
            card.import_vram_from_dmabuf_fd::<u8>(card.context, fd, lay.total) as u64,
            fd,
        ),
        None => {
            let (slab, fd) = card.alloc_exportable_vram_dmabuf::<u8>(lay.total);
            (slab as u64, fd)
        }
    };
    let mut w = LayerWeights {
        slab: base,
        slab_bytes: lay.total,
        dmabuf_fd: fd,
        layer_scalar: lay.layer_scalar,
        ..Default::default()
    };
    for ((name, parts), &off) in lay.groups.iter().zip(lay.offsets.iter()) {
        if stored.is_none() {
            let mut at = off;
            for bytes in parts {
                up.copy(card, base + at as u64, bytes);
                at += bytes.len();
            }
        }
        let ptr = base + off as u64;
        match *name {
            "input_ln" => w.input_ln = ptr,
            "q" => w.q = ptr,
            "k" => w.k = ptr,
            "v" => w.v = ptr,
            "q_norm" => w.q_norm = ptr,
            "k_norm" => w.k_norm = ptr,
            "o" => w.o = ptr,
            "attn_out_lns" => w.attn_out_lns = ptr,
            "mlp_gate_up" => w.mlp_gate_up = ptr,
            "mlp_down" => w.mlp_down = ptr,
            "router" => w.router = ptr,
            "experts_gate_up" => w.experts_gate_up = ptr,
            "experts_down" => w.experts_down = ptr,
            "ffn_out_lns" => w.ffn_out_lns = ptr,
            _ => unreachable!(),
        }
    }
    Ok(w)
}

/// Final norm and tied unembedding, on the card holding the last layer.
pub struct Head {
    pub final_norm: u64,
    pub embed: u64,
    /// `[MAX_LOGIT_ROWS, HIDDEN]`
    pub normed: *mut f32,
    /// `[MAX_LOGIT_ROWS, VOCAB]` in host USM
    pub logits: *mut f32,
}

/// The weights of one card's layers, and of the head on the card that holds it. Engines on the same cards share them.
#[cfg(feature = "xe")]
pub struct CardWeights {
    /// The index of the card in its `L0Cluster`.
    pub card: usize,
    pub layers: Range<usize>,
    pub weights: Vec<LayerWeights>,
    /// The final norm and the embedding table on the device, when the card holds the head.
    pub head: Option<(u64, u64)>,
    /// The DMA-BUF fd of the head slab, when the card holds the head.
    pub head_fd: Option<i32>,
    pub vram_bytes: usize,
}

#[cfg(feature = "xe")]
unsafe impl Send for CardWeights {}
#[cfg(feature = "xe")]
unsafe impl Sync for CardWeights {}

#[cfg(feature = "xe")]
impl CardWeights {
    /// Upload the layers `layers` of the checkpoint to `card`, and the head when `with_head`. A slab whose fd
    /// `stored` holds by its part name (`spec::layer_part`, `spec::HEAD_PART`) is imported instead.
    pub fn upload(
        card: &B70Card,
        ckpt: &Checkpoint,
        layers: Range<usize>,
        with_head: bool,
        stored: &HashMap<String, i32>,
    ) -> Result<Self, String> {
        let up = Uploader::new(card);
        let result = Self::upload_with(card, &up, ckpt, layers, with_head, stored);
        up.free(card);
        result
    }

    fn upload_with(
        card: &B70Card,
        up: &Uploader,
        ckpt: &Checkpoint,
        layers: Range<usize>,
        with_head: bool,
        stored: &HashMap<String, i32>,
    ) -> Result<Self, String> {
        let mut weights = Vec::new();
        let mut vram = 0usize;
        for l in layers.clone() {
            let w = upload_layer(card, up, ckpt, l, stored.get(&crate::spec::layer_part(l)).copied())?;
            vram += w.slab_bytes;
            weights.push(w);
        }
        let (head, head_fd) = if with_head {
            let lay = head_layout(ckpt)?;
            let (base, fd) = match stored.get(crate::spec::HEAD_PART).copied() {
                Some(fd) => (
                    card.import_vram_from_dmabuf_fd::<u8>(card.context, fd, lay.total) as u64,
                    fd,
                ),
                None => {
                    let (slab, fd) = card.alloc_exportable_vram_dmabuf::<u8>(lay.total);
                    let base = slab as u64;
                    for ((_, parts), &off) in lay.groups.iter().zip(&lay.offsets) {
                        up.copy(card, base + off as u64, parts[0]);
                    }
                    (base, fd)
                }
            };
            vram += lay.total;
            (Some((base, base + lay.offsets[1] as u64)), Some(fd))
        } else {
            (None, None)
        };
        Ok(Self {
            card: card.index,
            layers,
            weights,
            head,
            head_fd,
            vram_bytes: vram,
        })
    }
}

/// The device bytes of the weights of the layers `layers`, and of the head when `with_head`, from the tensor headers
/// of the checkpoint. It reads no weight bytes.
pub fn weights_device_bytes(ckpt: &Checkpoint, layers: Range<usize>, with_head: bool) -> Result<u64, String> {
    let mut bytes = 0u64;
    for l in layers {
        bytes += layer_layout(ckpt, l)?.total as u64;
    }
    if with_head {
        bytes += head_layout(ckpt)?.total as u64;
    }
    Ok(bytes)
}

/// The device bytes of the K and V planes of one slot of `seq` positions, for the layers `layers`.
pub fn slot_device_bytes(seq: usize, layers: Range<usize>) -> u64 {
    layers
        .map(|l| (2 * kv_plane_rows(seq, l) * layer_geom(l).kv_heads * layer_geom(l).head_dim * 4) as u64)
        .sum()
}

/// The device bytes of one card's work buffers for blocks of `max_block` tokens and slots of up to `max_seq`
/// positions. They are the device allocations of `CardShard::with_weights` that are not KV.
pub fn work_device_bytes(max_block: usize, max_seq: usize, with_head: bool) -> u64 {
    let (b, d) = (max_block, HIDDEN);
    let floats = b * d // x
        + 2 * b * d // norm and pre2
        + 2 * b * HEADS * 512 // q and attn
        + b * d // tmp
        + b * SHARED_I // mlp
        + b * TOP_K * MOE_I // moe_inter
        + 2 * b * TOP_K * d // moe_x and moe_out
        + b * NUM_EXPERTS // router_logits
        + attn_scratch_floats(max_seq, max_block)
        + HEADS * b * attn_row_floats(max_seq)
        + if with_head { MAX_LOGIT_ROWS * d } else { 0 };
    (floats * 4 + MOE_PLAN_BYTES + NUM_SCOREBOARD_EVENTS * 4) as u64
}

/// The host bytes of one card's pinned buffers for blocks of `max_block` tokens over `n_layers` layers. They are the
/// host allocations of `CardShard::with_weights`.
pub fn work_host_bytes(n_layers: usize, max_block: usize, with_head: bool) -> u64 {
    let bytes = n_layers * max_block * std::mem::size_of::<MoERouteDecision>()
        + max_block * 16
        + MAX_TASKS_PER_CARD * std::mem::size_of::<MkTaskDescriptor>()
        + STAGE_FLOATS * 4
        + 32 * std::mem::size_of::<XeCoreStats>()
        + if with_head { MAX_LOGIT_ROWS * VOCAB * 4 } else { 0 };
    bytes as u64
}

/// One card's layers, KV caches, and work buffers.
#[cfg(feature = "xe")]
pub struct CardShard {
    pub card: usize,
    pub layers: Range<usize>,
    pub weights: Vec<LayerWeights>,
    /// Per KV slot, per layer: K plane `[slot_seq, kv, hd]` followed by the V plane. Each
    /// slot is its own sequence (the council's reads, a generation), selected with `set_slot`.
    pub slots: Vec<Vec<u64>>,
    pub slot_seq: Vec<usize>,
    /// Per KV slot, the position after the last row a ring holds: the highest position written
    /// since the slot started over, or what a load left. Its ring holds positions from
    /// `kv_high - ring` only.
    kv_high: Vec<std::sync::atomic::AtomicUsize>,
    slot: std::sync::atomic::AtomicUsize,
    pub max_block: usize,
    pub vram_bytes: usize,
    pub x: *mut f32,
    norm: *mut f32,
    q: *mut f32,
    attn: *mut f32,
    tmp: *mut f32,
    mlp: *mut f32,
    pre2: *mut f32,
    moe_inter: *mut f32,
    /// A block's routed-expert inputs in plan order (OP_MOE_GATHER)
    moe_x: *mut f32,
    moe_out: *mut f32,
    /// `[max_block, 128]` router projections before the RMS factor.
    router_logits: *mut f32,
    plan: *mut u8,
    /// `[layer within shard][max_block]` routing decisions of the last block
    pub routes: *mut MoERouteDecision,
    pub head: Option<Head>,
    /// Per-token (position, gap start, gap end, 0) for a block of continuations; see the kernel.
    meta: *mut [u32; 4],
    meta_on: std::sync::atomic::AtomicBool,
    ring: *mut MkTaskDescriptor,
    scoreboard: *mut u32,
    /// Ops' working space: one-token attention's per-chunk partials.
    scratch: *mut f32,
    /// Multi-token attention's scores, then probabilities: `[head][token][key row]`.
    scores: *mut f32,
    /// Pinned host memory that small host-to-device copies go through.
    stage: *mut f32,
    pub stats: *mut XeCoreStats,
}

#[cfg(feature = "xe")]
unsafe impl Send for CardShard {}
#[cfg(feature = "xe")]
unsafe impl Sync for CardShard {}

/// Named device buffers a golden test can read after a ring prefix runs.
#[derive(Clone, Copy, Debug)]
pub enum Buf {
    X,
    Norm,
    Q,
    Attn,
    Tmp,
    Mlp,
    Pre2,
    MoeInter,
    MoeOut,
    /// K rows of a layer's cache from the block's first position
    K(usize),
    /// V rows of a layer's cache from the block's first position
    V(usize),
    Routes(usize),
    Normed,
    Logits,
}

#[cfg(feature = "xe")]
impl CardShard {
    pub fn new(
        cluster_card: &B70Card,
        ckpt: &Checkpoint,
        layers: Range<usize>,
        with_head: bool,
        max_seq: usize,
        max_block: usize,
    ) -> Result<Self, String> {
        Self::new_slots(
            cluster_card,
            ckpt,
            layers,
            with_head,
            &[max_seq],
            max_block,
            &HashMap::new(),
        )
    }

    /// A shard with one KV slot of `slot_seqs[i]` positions per entry.
    pub fn new_slots(
        cluster_card: &B70Card,
        ckpt: &Checkpoint,
        layers: Range<usize>,
        with_head: bool,
        slot_seqs: &[usize],
        max_block: usize,
        stored: &HashMap<String, i32>,
    ) -> Result<Self, String> {
        if max_block == 0 || max_block > MAX_PLAN_TOKENS {
            return Err(format!("max_block is {max_block}; it must be 1 to {MAX_PLAN_TOKENS}"));
        }
        let weights = CardWeights::upload(cluster_card, ckpt, layers, with_head, stored)?;
        Self::with_weights(cluster_card, &weights, slot_seqs, max_block)
    }

    /// A shard on weights that are on the card already, with one KV slot of `slot_seqs[i]` positions per entry.
    pub fn with_weights(
        cluster_card: &B70Card,
        w: &CardWeights,
        slot_seqs: &[usize],
        max_block: usize,
    ) -> Result<Self, String> {
        let card = cluster_card;
        if max_block == 0 || max_block > MAX_PLAN_TOKENS {
            return Err(format!("max_block is {max_block}; it must be 1 to {MAX_PLAN_TOKENS}"));
        }
        if card.index != w.card {
            return Err(format!(
                "the weights are on card {}, not on card {}",
                w.card, card.index
            ));
        }
        let layers = w.layers.clone();
        let mut slots: Vec<Vec<u64>> = vec![Vec::new(); slot_seqs.len()];
        let mut vram = w.vram_bytes;
        for l in layers.clone() {
            let g = layer_geom(l);
            for (si, &seq) in slot_seqs.iter().enumerate() {
                let kv_elems = 2 * kv_plane_rows(seq, l) * g.kv_heads * g.head_dim;
                let kv = card.alloc_device_usm::<f32>(kv_elems);
                if kv_ring(seq, l) > 0 {
                    // Past its end a ring's key tiles read older rows, masked but read: never
                    // uninitialized memory.
                    card.fill_zero_sync(kv as *mut c_void, kv_elems * 4);
                }
                slots[si].push(kv as u64);
                vram += kv_elems * 4;
            }
        }
        let d = HIDDEN;
        let b = max_block;
        let dev = |n: usize| card.alloc_device_usm::<f32>(n);
        let head = w.head.map(|(final_norm, embed)| Head {
            final_norm,
            embed,
            normed: dev(MAX_LOGIT_ROWS * d),
            logits: card.alloc_host_usm_cached::<f32>(MAX_LOGIT_ROWS * VOCAB),
        });
        let n_layers = layers.len();
        // OP_ATTN_OUT writes both pre-FFN norms: pre2 sits b * d floats after norm.
        let norm = dev(2 * b * d);
        Ok(Self {
            card: card.index,
            weights: w.weights.clone(),
            slots,
            slot_seq: slot_seqs.to_vec(),
            kv_high: slot_seqs
                .iter()
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect(),
            slot: std::sync::atomic::AtomicUsize::new(0),
            max_block,
            vram_bytes: vram,
            x: dev(b * d),
            norm,
            q: dev(b * HEADS * 512),
            attn: dev(b * HEADS * 512),
            tmp: dev(b * d),
            mlp: dev(b * SHARED_I),
            pre2: unsafe { norm.add(b * d) },
            moe_inter: dev(b * TOP_K * MOE_I),
            moe_x: dev(b * TOP_K * d),
            moe_out: dev(b * TOP_K * d),
            router_logits: dev(b * NUM_EXPERTS),
            plan: card.alloc_device_usm::<u8>(MOE_PLAN_BYTES),
            routes: card.alloc_host_usm_cached::<MoERouteDecision>(n_layers * b),
            head,
            meta: card.alloc_host_usm::<[u32; 4]>(b),
            meta_on: std::sync::atomic::AtomicBool::new(false),
            ring: card.alloc_host_usm::<MkTaskDescriptor>(MAX_TASKS_PER_CARD),
            scoreboard: card.alloc_device_usm::<u32>(NUM_SCOREBOARD_EVENTS),
            scratch: dev(attn_scratch_floats(slot_seqs.iter().copied().max().unwrap_or(0), b)),
            scores: dev(HEADS * b * attn_row_floats(slot_seqs.iter().copied().max().unwrap_or(0))),
            stage: card.alloc_host_usm::<f32>(STAGE_FLOATS),
            stats: card.alloc_host_usm::<XeCoreStats>(32),
            layers,
        })
    }

    /// Free the KV slots and the work buffers of the shard. The weights stay, because other shards can share them.
    ///
    /// # Safety
    ///
    /// No kernel may run on the buffers, and the shard must not be used after this call.
    pub unsafe fn free(&self, card: &B70Card) {
        for slot in &self.slots {
            for &kv in slot {
                card.free(kv as *mut c_void);
            }
        }
        if let Some(h) = &self.head {
            card.free(h.normed as *mut c_void);
            card.free(h.logits as *mut c_void);
        }
        for ptr in [
            self.x as *mut c_void,
            self.norm as *mut c_void,
            self.q as *mut c_void,
            self.attn as *mut c_void,
            self.tmp as *mut c_void,
            self.mlp as *mut c_void,
            self.moe_inter as *mut c_void,
            self.moe_x as *mut c_void,
            self.moe_out as *mut c_void,
            self.router_logits as *mut c_void,
            self.plan as *mut c_void,
            self.routes as *mut c_void,
            self.meta as *mut c_void,
            self.ring as *mut c_void,
            self.scoreboard as *mut c_void,
            self.scratch as *mut c_void,
            self.scores as *mut c_void,
            self.stage as *mut c_void,
            self.stats as *mut c_void,
        ] {
            card.free(ptr);
        }
    }

    pub fn slot(&self) -> usize {
        self.slot.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn set_slot(&self, slot: usize) {
        assert!(slot < self.slots.len(), "KV slot {} of {}", slot, self.slots.len());
        self.slot.store(slot, std::sync::atomic::Ordering::SeqCst);
    }

    /// Positions the current KV slot holds.
    pub fn max_seq(&self) -> usize {
        self.slot_seq[self.slot()]
    }

    /// The position after the last row that the rings of `slot` hold (`kv_high`).
    pub fn kv_high(&self, slot: usize) -> usize {
        self.kv_high[slot].load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Set the position after the last row that the rings of `slot` hold, after a restore or a clear.
    pub fn set_kv_high(&self, slot: usize, high: usize) {
        self.kv_high[slot].store(high, std::sync::atomic::Ordering::SeqCst);
    }

    /// Copy the K or V rows of positions `first..len` of layer `l` in the current slot into `out`.
    pub fn read_rows(&self, card: &B70Card, l: usize, value: bool, first: usize, len: usize, out: &mut [u8]) {
        let li = l - self.layers.start;
        let row = layer_geom(l).kv_heads * layer_geom(l).head_dim * 4;
        assert_eq!(out.len(), (len - first) * row, "the buffer holds the rows");
        let mut at = 0;
        for (src, n) in self.kv_runs(li, first, len, value) {
            card.memcpy_sync(out[at..].as_mut_ptr() as *mut c_void, src as *const c_void, n * row);
            at += n * row;
        }
    }

    /// Copy `bytes`, the K or V rows of positions `first..len` of layer `l`, into the current slot.
    pub fn write_rows(&self, card: &B70Card, l: usize, value: bool, first: usize, len: usize, bytes: &[u8]) {
        let li = l - self.layers.start;
        let row = layer_geom(l).kv_heads * layer_geom(l).head_dim * 4;
        assert_eq!(bytes.len(), (len - first) * row, "the bytes hold the rows");
        let mut at = 0;
        for (dst, n) in self.kv_runs(li, first, len, value) {
            self.upload_bytes(card, dst, &bytes[at..at + n * row]);
            at += n * row;
        }
    }

    fn kv_base(&self, li: usize) -> u64 {
        self.slots[self.slot()][li]
    }

    /// The address of position `pos0`'s K or V row of layer `li` of the shard, in the current
    /// slot.
    fn kv_rows(&self, li: usize, pos0: usize, value: bool) -> u64 {
        let l = self.layers.start + li;
        let g = layer_geom(l);
        let row = g.kv_heads * g.head_dim;
        let plane = if value {
            kv_plane_rows(self.max_seq(), l) * row
        } else {
            0
        };
        let at = match kv_ring(self.max_seq(), l) {
            0 => pos0,
            r => pos0 % r,
        };
        self.kv_base(li) + ((plane + at * row) * 4) as u64
    }

    /// The (row address, rows) runs holding positions `first..len` of layer `li`'s K or V:
    /// one, or two where a ring wraps.
    fn kv_runs(&self, li: usize, first: usize, len: usize, value: bool) -> Vec<(u64, usize)> {
        let l = self.layers.start + li;
        let n = len - first;
        match kv_ring(self.max_seq(), l) {
            r if r > 0 && first % r + n > r => {
                let head = r - first % r;
                vec![
                    (self.kv_rows(li, first, value), head),
                    (self.kv_rows(li, first + head, value), n - head),
                ]
            }
            _ => vec![(self.kv_rows(li, first, value), n)],
        }
    }

    pub fn buf_ptr(&self, b: Buf, pos0: usize) -> u64 {
        match b {
            Buf::X => self.x as u64,
            Buf::Norm => self.norm as u64,
            Buf::Q => self.q as u64,
            Buf::Attn => self.attn as u64,
            Buf::Tmp => self.tmp as u64,
            Buf::Mlp => self.mlp as u64,
            Buf::Pre2 => self.pre2 as u64,
            Buf::MoeInter => self.moe_inter as u64,
            Buf::MoeOut => self.moe_out as u64,
            Buf::K(l) => self.kv_rows(l - self.layers.start, pos0, false),
            Buf::V(l) => self.kv_rows(l - self.layers.start, pos0, true),
            Buf::Routes(l) => self.routes as u64 + ((l - self.layers.start) * self.max_block * 128) as u64,
            Buf::Normed => self.head.as_ref().expect("no head on this card").normed as u64,
            Buf::Logits => self.head.as_ref().expect("no head on this card").logits as u64,
        }
    }

    /// Writes this card's task ring for a block of `n` tokens in cache rows `pos0..pos0 + n`
    /// and returns one label per task, such as `L5.q_proj`. The head computes logits for each
    /// block row in `logit_rows`. `meta` is the kernel's token table (position, gap start, gap
    /// end) per token; without it token t sits at position pos0 + t.
    pub unsafe fn build_ring(
        &self,
        n: usize,
        pos0: usize,
        logit_rows: &[usize],
        meta: Option<&[[u32; 4]]>,
    ) -> Vec<String> {
        assert!(
            n >= 1 && n <= self.max_block,
            "block of {} tokens; max_block is {}",
            n,
            self.max_block
        );
        assert!(
            pos0 + n <= self.max_seq(),
            "positions {}..{} exceed max_seq {}",
            pos0,
            pos0 + n,
            self.max_seq()
        );
        self.kv_high[self.slot()].fetch_max(pos0 + n, std::sync::atomic::Ordering::SeqCst);
        assert!(logit_rows.len() <= MAX_LOGIT_ROWS && logit_rows.iter().all(|&r| r < n));
        match meta {
            Some(m) => {
                assert_eq!(m.len(), n);
                unsafe { std::ptr::copy_nonoverlapping(m.as_ptr(), self.meta, n) };
                self.meta_on.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            None => self.meta_on.store(false, std::sync::atomic::Ordering::SeqCst),
        }
        let d = HIDDEN as u32;
        let nn = n as u32;
        let mut labels: Vec<String> = Vec::new();
        // A task waits only for the latest earlier task that writes a buffer it reads, or reads
        // or writes a buffer it writes. Every work-group runs the ring in order, so once all of
        // them have finished that task they have finished every task before it.
        let mut last_write: HashMap<u64, usize> = HashMap::new();
        let mut last_access: HashMap<u64, usize> = HashMap::new();
        let mut push = |label: String, mut t: MkTaskDescriptor, reads: &[u64], writes: &[u64]| {
            let i = labels.len();
            assert!(i + 1 < MAX_TASKS_PER_CARD, "task ring capacity exceeded");
            if matches!(t.opcode, OP_MATMUL | OP_GEGLU | OP_MOE_GATE_UP | OP_MOE_DOWN) {
                assert!(
                    t.m % 16 == 0 && t.k % 16 == 0,
                    "{}: XMX tiles need m and k in multiples of 16",
                    label
                );
            }
            let wait = reads
                .iter()
                .filter_map(|b| last_write.get(b))
                .chain(writes.iter().filter_map(|b| last_access.get(b)))
                .max();
            t.wait_event = wait.map_or(NO_EVENT, |&w| w as u32);
            t.signal_event = i as u32;
            for b in reads {
                last_access.insert(*b, i);
            }
            for b in writes {
                last_write.insert(*b, i);
                last_access.insert(*b, i);
            }
            unsafe { *self.ring.add(i) = t };
            labels.push(label);
        };
        let norm = |a: u64, b: u64, c: u64, rows: u32, k: u32, flags: u16, p0: u32| MkTaskDescriptor {
            opcode: OP_RMSNORM,
            flags,
            a,
            b,
            c,
            n: rows,
            k,
            p0,
            ..Default::default()
        };
        let matmul = |a: u64, b: u64, c: u64, m: u32, k: u32, flags: u16| MkTaskDescriptor {
            opcode: OP_MATMUL,
            flags,
            a,
            b,
            c,
            m,
            n: nn,
            k,
            ..Default::default()
        };

        for (li, w) in self.weights.iter().enumerate() {
            let l = self.layers.start + li;
            let g = layer_geom(l);
            let (hd, kvh) = (g.head_dim as u32, g.kv_heads as u32);
            let q_dim = HEADS as u32 * hd;
            let kr = self.kv_rows(li, pos0, false);
            let vr = self.kv_rows(li, pos0, true);
            let (x, nb, q, at, tmp) = (
                self.x as u64,
                self.norm as u64,
                self.q as u64,
                self.attn as u64,
                self.tmp as u64,
            );
            let tag = |s: &str| format!("L{}.{}", l, s);

            // Buffer keys for dependencies: one per allocation, and one for each of the layer's
            // K and V cache planes.
            let (mlp, pre2, plan, inter, mout) = (
                self.mlp as u64,
                self.pre2 as u64,
                self.plan as u64,
                self.moe_inter as u64,
                self.moe_out as u64,
            );
            let (kv, kp, vp) = (self.kv_base(li), self.kv_rows(li, 0, false), self.kv_rows(li, 0, true));
            let scratch = self.scratch as u64;
            let routes = self.buf_ptr(Buf::Routes(l), pos0);

            push(tag("attn_norm"), norm(x, w.input_ln, nb, nn, d, 0, 0), &[x], &[nb]);
            push(tag("q_proj"), matmul(nb, w.q, q, q_dim, d, 0), &[nb], &[q]);
            push(tag("k_proj"), matmul(nb, w.k, kr, kvh * hd, d, 0), &[nb], &[kp]);
            if g.global {
                // attention_k_eq_v: V is the unnormalized K projection, then v_norm
                push(tag("v_norm"), norm(kr, 0, vr, nn * kvh, hd, 0, 0), &[kp], &[vp]);
            } else {
                push(tag("v_proj"), matmul(nb, w.v, vr, kvh * hd, d, 0), &[nb], &[vp]);
                push(tag("v_norm"), norm(vr, 0, vr, nn * kvh, hd, 0, 0), &[vp], &[vp]);
            }
            let rope = FLAG_ROPE | if g.global { FLAG_ROPE_GLOBAL } else { 0 };
            push(
                tag("q_norm_rope"),
                MkTaskDescriptor {
                    opcode: OP_RMSNORM,
                    flags: rope,
                    a: q,
                    b: w.q_norm,
                    c: q,
                    m: HEADS as u32,
                    n: nn * HEADS as u32,
                    k: hd,
                    p0: pos0 as u32,
                    ..Default::default()
                },
                &[q],
                &[q],
            );
            push(
                tag("k_norm_rope"),
                MkTaskDescriptor {
                    opcode: OP_RMSNORM,
                    flags: rope,
                    a: kr,
                    b: w.k_norm,
                    c: kr,
                    m: kvh,
                    n: nn * kvh,
                    k: hd,
                    p0: pos0 as u32,
                    ..Default::default()
                },
                &[kp],
                &[kp],
            );
            let ring = kv_ring(self.max_seq(), l);
            if ring > 0 && pos0 % ring + n > ring {
                // The block's rows past the ring's end move to its start, bit for bit, before
                // attention reads them there.
                let spill = (pos0 % ring + n - ring) * kvh as usize * hd as usize;
                for (start, key) in [(kp, kp), (vp, vp)] {
                    let past = start + (ring * kvh as usize * hd as usize * 4) as u64;
                    push(
                        tag("kv_wrap"),
                        MkTaskDescriptor {
                            opcode: OP_COPY,
                            a: past,
                            c: start,
                            n: spill as u32,
                            ..Default::default()
                        },
                        &[key],
                        &[key],
                    );
                }
            }
            let window = if g.global { 0u64 } else { SLIDING_WINDOW as u64 };
            let attn = MkTaskDescriptor {
                opcode: OP_ATTN,
                a: q,
                b: kv,
                c: at,
                m: HEADS as u32 | (kvh << 16),
                n: nn,
                k: hd,
                p0: pos0 as u32,
                x: ((self.max_seq() as u64) << 32) | window,
                layer: ring as u32,
                ..Default::default()
            };
            if n > 1 || meta.is_some() {
                let sc = self.scores as u64;
                push(
                    tag("attn_scores"),
                    MkTaskDescriptor {
                        opcode: OP_ATTN_SCORES,
                        c: sc,
                        ..attn
                    },
                    &[q, kp],
                    &[sc],
                );
                push(
                    tag("attn_softmax"),
                    MkTaskDescriptor {
                        opcode: OP_ATTN_SOFTMAX,
                        c: sc,
                        ..attn
                    },
                    &[sc],
                    &[sc],
                );
                // With a small block OP_ATTN_PV splits the keys and leaves partial sums in
                // scratch for OP_ATTN_PV_SUM.
                push(
                    tag("attn_pv"),
                    MkTaskDescriptor {
                        opcode: OP_ATTN_PV,
                        a: sc,
                        ..attn
                    },
                    &[sc, vp],
                    &[at, scratch],
                );
                push(
                    tag("attn_pv_sum"),
                    MkTaskDescriptor {
                        opcode: OP_ATTN_PV_SUM,
                        ..attn
                    },
                    &[scratch],
                    &[at],
                );
            } else {
                push(tag("attn"), attn, &[q, kp, vp], &[at, scratch]);
            }
            if n == 1 && meta.is_none() {
                // The kernel's one-token attention leaves per-chunk partials in scratch.
                push(
                    tag("attn_combine"),
                    MkTaskDescriptor {
                        opcode: OP_ATTN_COMBINE,
                        c: at,
                        m: HEADS as u32,
                        k: hd,
                        p0: pos0 as u32,
                        x: window,
                        ..Default::default()
                    },
                    &[scratch],
                    &[at],
                );
            }
            push(tag("o_proj"), matmul(at, w.o, tmp, d, q_dim, 0), &[at], &[tmp]);
            push(
                tag("attn_out"),
                MkTaskDescriptor {
                    opcode: OP_ATTN_OUT,
                    a: tmp,
                    b: w.attn_out_lns,
                    c: x,
                    x: nb,
                    m: (self.max_block * HIDDEN) as u32,
                    n: nn,
                    k: d,
                    ..Default::default()
                },
                &[tmp, x],
                &[x, nb, pre2],
            );
            // The routed experts and the shared MLP both read x; their tasks alternate so a
            // work-group done with its share of one goes on to the other.
            let rl = self.router_logits as u64;
            push(
                tag("router_logits"),
                MkTaskDescriptor {
                    opcode: OP_ROUTER_LOGITS,
                    a: x,
                    b: w.router,
                    c: rl,
                    n: nn,
                    k: d,
                    ..Default::default()
                },
                &[x],
                &[rl],
            );
            push(
                tag("router"),
                MkTaskDescriptor {
                    opcode: OP_ROUTER,
                    a: x,
                    b: w.router,
                    c: routes,
                    n: nn,
                    k: d,
                    layer: l as u32,
                    x: rl,
                    ..Default::default()
                },
                &[x, rl],
                &[routes],
            );
            push(
                tag("moe_plan"),
                MkTaskDescriptor {
                    opcode: OP_MOE_PLAN,
                    a: routes,
                    c: plan,
                    n: nn,
                    ..Default::default()
                },
                &[routes],
                &[plan],
            );
            // Several tokens: each expert's input rows, gathered in plan order for 2D loads.
            let mx = self.moe_x as u64;
            let moe_in = if n > 1 {
                push(
                    tag("moe_gather"),
                    MkTaskDescriptor {
                        opcode: OP_MOE_GATHER,
                        a: pre2,
                        c: mx,
                        n: nn,
                        k: d,
                        x: plan,
                        ..Default::default()
                    },
                    &[pre2, plan],
                    &[mx],
                );
                mx
            } else {
                pre2
            };
            push(
                tag("moe_gate_up"),
                MkTaskDescriptor {
                    opcode: OP_MOE_GATE_UP,
                    a: moe_in,
                    b: w.experts_gate_up,
                    c: inter,
                    m: MOE_I as u32,
                    n: nn,
                    k: d,
                    x: plan,
                    ..Default::default()
                },
                &[moe_in, plan],
                &[inter],
            );
            push(
                tag("mlp_geglu"),
                MkTaskDescriptor {
                    opcode: OP_GEGLU,
                    a: nb,
                    b: w.mlp_gate_up,
                    c: mlp,
                    m: SHARED_I as u32,
                    n: nn,
                    k: d,
                    ..Default::default()
                },
                &[nb],
                &[mlp],
            );
            push(
                tag("moe_down"),
                MkTaskDescriptor {
                    opcode: OP_MOE_DOWN,
                    a: inter,
                    b: w.experts_down,
                    c: mout,
                    m: d,
                    n: nn,
                    k: MOE_I as u32,
                    x: plan,
                    ..Default::default()
                },
                &[inter, plan],
                &[mout],
            );
            push(
                tag("mlp"),
                matmul(mlp, w.mlp_down, tmp, d, SHARED_I as u32, 0),
                &[mlp],
                &[tmp],
            );
            push(
                tag("out"),
                MkTaskDescriptor {
                    opcode: OP_FFN_OUT,
                    a: tmp,
                    b: w.ffn_out_lns,
                    c: x,
                    x: mout,
                    n: nn,
                    k: d,
                    p0: w.layer_scalar.to_bits(),
                    ..Default::default()
                },
                &[tmp, mout, x],
                &[x],
            );
        }

        if !logit_rows.is_empty() {
            let h = self
                .head
                .as_ref()
                .expect("logits requested from a card without the head");
            for (i, &row) in logit_rows.iter().enumerate() {
                let src = self.x as u64 + (row * HIDDEN * 4) as u64;
                let dst = h.normed as u64 + (i * HIDDEN * 4) as u64;
                push(
                    format!("final_norm.{}", i),
                    norm(src, h.final_norm, dst, 1, d, 0, 0),
                    &[self.x as u64],
                    &[h.normed as u64],
                );
            }
            push(
                "logits".into(),
                MkTaskDescriptor {
                    opcode: OP_MATMUL,
                    flags: FLAG_SOFTCAP_30,
                    a: h.normed as u64,
                    b: h.embed,
                    c: h.logits as u64,
                    m: VOCAB as u32,
                    n: logit_rows.len() as u32,
                    k: d,
                    ..Default::default()
                },
                &[h.normed as u64],
                &[h.logits as u64],
            );
        }
        let end = labels.len();
        unsafe {
            *self.ring.add(end) = MkTaskDescriptor {
                opcode: OP_HALT,
                ..Default::default()
            }
        };
        labels
    }

    /// Runs the first `num_tasks` tasks of the ring last written by `build_ring`.
    pub unsafe fn run(&self, card: &B70Card, num_tasks: usize) -> u128 {
        unsafe { self.run_range(card, 0, num_tasks) }
    }

    /// Runs tasks `start..end` of the ring last written by `build_ring`.
    pub unsafe fn run_range(&self, card: &B70Card, start: usize, end: usize) -> u128 {
        unsafe {
            std::ptr::write_bytes(self.stats, 0, 32);
            let saved_end = *self.ring.add(end);
            *self.ring.add(end) = MkTaskDescriptor {
                opcode: OP_HALT,
                ..Default::default()
            };
            let t0 = Instant::now();
            // Tasks before `start` count as done: their events read as complete.
            card.launch_megakernel(
                self.ring.add(start) as *const c_void,
                (end - start + 1) as u32,
                self.scoreboard as *mut c_void,
                start,
                end + 1,
                if self.meta_on.load(std::sync::atomic::Ordering::SeqCst) {
                    self.meta as *const c_void
                } else {
                    std::ptr::null()
                },
                self.stats as *const c_void,
                self.scratch as *mut c_void,
            );
            let us = t0.elapsed().as_micros();
            *self.ring.add(end) = saved_end;
            us
        }
    }

    /// Writes `tasks` to the ring in place of the last `build_ring` and runs them.
    pub unsafe fn run_tasks(&self, card: &B70Card, tasks: &[MkTaskDescriptor]) -> u128 {
        assert!(tasks.len() < MAX_TASKS_PER_CARD);
        unsafe {
            std::ptr::copy_nonoverlapping(tasks.as_ptr(), self.ring, tasks.len());
            self.run(card, tasks.len())
        }
    }

    /// Copies `count` f32 values from a buffer (device or host USM) into a Vec.
    pub fn read(&self, card: &B70Card, b: Buf, pos0: usize, count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count];
        card.memcpy_sync(
            out.as_mut_ptr() as *mut c_void,
            self.buf_ptr(b, pos0) as *const c_void,
            count * 4,
        );
        out
    }

    /// Copies host values into a buffer (device or host USM).
    pub fn write(&self, card: &B70Card, b: Buf, pos0: usize, values: &[f32]) {
        self.upload(card, self.buf_ptr(b, pos0), values);
    }

    /// Copies host values to device address `dst`, through the pinned staging buffer when they
    /// fit: staging takes 1 MB in 62 us instead of 305, but 64 MB in 3.8 ms instead of 3.1.
    fn upload(&self, card: &B70Card, dst: u64, values: &[f32]) {
        self.upload_bytes(card, dst, f32_bytes(values));
    }

    /// `upload` for the bytes of f32 values.
    pub(crate) fn upload_bytes(&self, card: &B70Card, dst: u64, bytes: &[u8]) {
        if bytes.len() > STAGE_FLOATS * 4 {
            card.memcpy_sync(dst as *mut c_void, bytes.as_ptr() as *const c_void, bytes.len());
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.stage as *mut u8, bytes.len());
            card.memcpy_sync(dst as *mut c_void, self.stage as *const c_void, bytes.len());
        }
    }

    /// The logits of the `i`-th logit row of the last block, in the head's host buffer.
    pub fn logits_row(&self, i: usize) -> &[f32] {
        assert!(i < MAX_LOGIT_ROWS, "logit row {i} of {MAX_LOGIT_ROWS}");
        let logits = self.head.as_ref().expect("logits from a card without the head").logits;
        unsafe { std::slice::from_raw_parts(logits.add(i * VOCAB), VOCAB) }
    }
}

/// Floats per score row of multi-token attention when keys reach `seq` rows: the kernel's
/// attn_row_floats at its largest.
fn attn_row_floats(seq: usize) -> usize {
    (seq + 15) / 16 * 16 + 16
}

/// Floats of scratch that attention needs. One-token attention keeps every head's partials for 32-key chunks over
/// `max_seq` positions (the kernel's ATTN_CHUNK), each a max, a sum, and a 512-wide output. The split attention of a
/// block of up to `max_block` tokens keeps its partial sums (`tasks::attn_pv_scratch_floats`).
fn attn_scratch_floats(max_seq: usize, max_block: usize) -> usize {
    let one_token = HEADS * (max_seq / 32 + 1) * (512 + 2);
    one_token.max(nagaya_mk_bmg::tasks::attn_pv_scratch_floats(max_block, HEADS, 512))
}

/// Host copy of every layer's K and V rows up to position `len`: all of them, or for a
/// sliding layer saved from a ring, the last ring's worth.
pub struct KvSnapshot {
    pub len: usize,
    /// One Vec per layer (both cards, in layer order): K rows then V rows for the layer's last
    /// positions, as many as it holds.
    layers: Vec<Vec<f32>>,
}

impl KvSnapshot {
    /// A snapshot of `len` positions from each layer's K then V rows, in layer order.
    pub fn from_layers(len: usize, layers: Vec<Vec<f32>>) -> Self {
        assert_eq!(layers.len(), NUM_LAYERS);
        Self { len, layers }
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.len() * 4).sum()
    }

    const MAGIC: &'static [u8; 8] = b"MKVSNAP1";

    /// Writes `MKVSNAP1`, the length, then each layer's float count and little-endian floats. A
    /// layer holding fewer rows than the length holds the last positions.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(Self::MAGIC)?;
        w.write_all(&(self.len as u64).to_le_bytes())?;
        w.write_all(&(self.layers.len() as u64).to_le_bytes())?;
        for l in &self.layers {
            w.write_all(&(l.len() as u64).to_le_bytes())?;
            let bytes = unsafe { std::slice::from_raw_parts(l.as_ptr() as *const u8, l.len() * 4) };
            w.write_all(bytes)?;
        }
        Ok(())
    }

    /// Reads what `write_to` wrote, checking every size against the model's layer shapes.
    pub fn read_from(r: &mut impl std::io::Read) -> Result<Self, String> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != Self::MAGIC {
            return Err("not an mkviolet KV snapshot".into());
        }
        let mut u = [0u8; 8];
        let mut next = |r: &mut dyn std::io::Read| -> Result<u64, String> {
            r.read_exact(&mut u).map_err(|e| e.to_string())?;
            Ok(u64::from_le_bytes(u))
        };
        let len = next(r)? as usize;
        let n_layers = next(r)? as usize;
        if n_layers != NUM_LAYERS {
            return Err(format!("{} layers, expected {}", n_layers, NUM_LAYERS));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let n = next(r)? as usize;
            let g = layer_geom(l);
            let row = 2 * g.kv_heads * g.head_dim;
            if n % row != 0 || n / row > len {
                return Err(format!(
                    "layer {} holds {} floats, not whole rows of {} positions or fewer",
                    l, n, len
                ));
            }
            let mut v = vec![0.0f32; n];
            let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            r.read_exact(bytes).map_err(|e| e.to_string())?;
            layers.push(v);
        }
        Ok(Self { len, layers })
    }

    /// Joins a chain of snapshots, newest first, into one of `chain[0].len` positions holding
    /// what a slot of `seq` positions keeps to go on from there: each layer's positions
    /// `kv_keep_from(seq, l, len)..len`, or for a ring at least the sliding window. Each
    /// snapshot holds its layers' last positions, and each takes over where the one before it
    /// in the chain ends. Fails when the chain lacks a position a layer needs.
    pub fn join(chain: &[&KvSnapshot], seq: usize) -> Result<KvSnapshot, String> {
        let len = chain.first().ok_or("an empty snapshot chain")?.len;
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for l in 0..NUM_LAYERS {
            let g = layer_geom(l);
            let row = g.kv_heads * g.head_dim;
            let need = kv_keep_from(seq, l, len);
            let required = if kv_ring(seq, l) > 0 {
                len.saturating_sub(SLIDING_WINDOW as usize)
            } else {
                0
            };
            // (snapshot, first, end) runs of positions, newest first.
            let mut runs: Vec<(&KvSnapshot, usize, usize)> = Vec::new();
            let mut covered = len;
            for s in chain {
                if covered <= need || s.len < covered {
                    break;
                }
                let have = s.len - s.layers[l].len() / 2 / row;
                let first = have.max(need);
                if first < covered {
                    runs.push((s, first, covered));
                    covered = first;
                }
            }
            if covered > required {
                return Err(format!(
                    "layer {}: the chain lacks positions {}..{}",
                    l, required, covered
                ));
            }
            let rows: usize = runs.iter().map(|r| r.2 - r.1).sum();
            let mut v = Vec::with_capacity(2 * rows * row);
            for value in [false, true] {
                for &(s, first, end) in runs.iter().rev() {
                    let held = s.layers[l].len() / 2 / row;
                    let have = s.len - held;
                    let base = if value { held * row } else { 0 };
                    v.extend_from_slice(&s.layers[l][base + (first - have) * row..base + (end - have) * row]);
                }
            }
            layers.push(v);
        }
        Ok(KvSnapshot { len, layers })
    }
}

/// The row of token `token_id` in a BF16 embedding table of `hidden_size` columns, as F32 times `scale`.
/// mkviolet keeps it in `tokenizer.rs`, which stays behind.
pub fn embedding_row_f32(embed_bf16_bytes: &[u8], hidden_size: usize, token_id: u32, scale: f32) -> Vec<f32> {
    let row_bytes = hidden_size * 2;
    let start = (token_id as usize) * row_bytes;
    let slice = &embed_bf16_bytes[start..start + row_bytes];
    slice
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) * scale)
        .collect()
}

/// Bytes of host memory one held token costs: K and V for every layer.
pub fn kv_bytes_per_token() -> usize {
    (0..NUM_LAYERS)
        .map(|l| 2 * layer_geom(l).kv_heads * layer_geom(l).head_dim * 4)
        .sum()
}

/// One block of a shared prefix and its branches, laid out by `branch_block`.
pub struct BranchBlock {
    pub tokens: Vec<u32>,
    pub meta: Vec<[u32; 4]>,
    pub logit_rows: Vec<usize>,
    pub pos0: usize,
}

/// One block for `Gemma4Engine::pipeline_stage`: `tokens` at positions `pos0..` (or as
/// `meta` places them), with logits for each `logit_rows` row. With `kv`, each card restores
/// its layers' cache rows from it before running the block.
pub struct PipeJob<'a> {
    pub tokens: &'a [u32],
    pub pos0: usize,
    pub meta: Option<&'a [[u32; 4]]>,
    pub logit_rows: Vec<usize>,
    pub kv: Option<&'a KvSnapshot>,
}

impl<'a> PipeJob<'a> {
    pub fn plain(tokens: &'a [u32], pos0: usize, logit_rows: Vec<usize>) -> Self {
        PipeJob {
            tokens,
            pos0,
            meta: None,
            logit_rows,
            kv: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PassTiming {
    pub blocks: usize,
    pub card_us: [u128; 2],
    pub wall_us: u128,
}

#[cfg(feature = "xe")]
pub struct Gemma4Engine {
    pub ckpt: Checkpoint,
    pub shards: Vec<CardShard>,
    pub load_ms: u128,
    /// How many times each layer's router has picked each expert since load, over every token
    /// of every forward pass.
    pub expert_hits: std::sync::Mutex<Vec<[u64; NUM_EXPERTS]>>,
    /// When `Some`, every block's routing is appended as (layer, position, top-8 experts).
    pub route_log: std::sync::Mutex<Option<Vec<(usize, usize, [u32; TOP_K])>>>,
}

#[cfg(feature = "xe")]
impl Gemma4Engine {
    pub fn load(cluster: &L0Cluster, model_dir: &Path, max_seq: usize, max_block: usize) -> Result<Self, String> {
        Self::load_slots(cluster, model_dir, &[max_seq], max_block)
    }

    /// Loads the model with one KV slot per entry of `slot_seqs`; slot 0 is current.
    pub fn load_slots(
        cluster: &L0Cluster,
        model_dir: &Path,
        slot_seqs: &[usize],
        max_block: usize,
    ) -> Result<Self, String> {
        Self::load_slots_stored(cluster, model_dir, slot_seqs, max_block, &HashMap::new())
    }

    /// `load_slots`, importing each slab whose fd `stored` holds (by its part name, for both cards)
    /// instead of uploading it. The caller checks them first with `verify_stored_slab`.
    /// Returns how many layers it imported in `imported`.
    pub fn load_slots_stored(
        cluster: &L0Cluster,
        model_dir: &Path,
        slot_seqs: &[usize],
        max_block: usize,
        stored: &HashMap<String, i32>,
    ) -> Result<Self, String> {
        check_config(model_dir)?;
        let ckpt = Checkpoint::open_dir(model_dir)?;
        let t0 = Instant::now();
        let mut shards = Vec::with_capacity(2);
        for (ci, layers) in CARD_LAYERS.iter().enumerate() {
            shards.push(CardShard::new_slots(
                &cluster.cards[ci],
                &ckpt,
                layers.clone(),
                ci == 1,
                slot_seqs,
                max_block,
                stored,
            )?);
        }
        Ok(Self {
            ckpt,
            shards,
            load_ms: t0.elapsed().as_millis(),
            expert_hits: std::sync::Mutex::new(vec![[0u64; NUM_EXPERTS]; NUM_LAYERS]),
            route_log: std::sync::Mutex::new(None),
        })
    }

    /// Adds the last block's routing on `shard` to `expert_hits`.
    fn count_routes(&self, shard: &CardShard, n: usize, pos0: usize) {
        let mut hits = self.expert_hits.lock().unwrap();
        let mut log = self.route_log.lock().unwrap();
        for l in shard.layers.clone() {
            let routes =
                unsafe { std::slice::from_raw_parts(shard.buf_ptr(Buf::Routes(l), 0) as *const MoERouteDecision, n) };
            for (t, r) in routes.iter().enumerate() {
                for &e in &r.top_k_indices {
                    hits[l][e as usize] += 1;
                }
                if let Some(log) = log.as_mut() {
                    log.push((l, pos0 + t, r.top_k_indices));
                }
            }
        }
    }

    /// Per layer: how many distinct experts the router has picked since load, and the fewest
    /// and most picks any expert got.
    pub fn expert_usage(&self) -> Vec<(usize, u64, u64)> {
        let hits = self.expert_hits.lock().unwrap();
        hits.iter()
            .map(|h| {
                (
                    h.iter().filter(|&&c| c > 0).count(),
                    *h.iter().min().unwrap(),
                    *h.iter().max().unwrap(),
                )
            })
            .collect()
    }

    /// Copies the KV rows of positions `0..len` from both cards to host memory.
    /// Copies every layer's K and V rows for the positions a slot needs to go on from `len`
    /// (see `save_kv_layer`) to host memory.
    pub fn save_kv(&self, cluster: &L0Cluster, len: usize) -> KvSnapshot {
        KvSnapshot {
            len,
            layers: (0..NUM_LAYERS).map(|l| self.save_kv_layer(cluster, l, len)).collect(),
        }
    }

    /// The shard (and card) holding layer `l`.
    pub fn card_of(&self, l: usize) -> usize {
        if self.shards[0].layers.contains(&l) { 0 } else { 1 }
    }

    /// The K then V rows of layer `l` for positions `0..len` in the current KV slot.
    /// Layer `l`'s K rows then V rows for positions `kv_keep_from(..)..len` of the current
    /// slot: all of them, or with a ring the last ring's worth.
    pub fn save_kv_layer(&self, cluster: &L0Cluster, l: usize, len: usize) -> Vec<f32> {
        self.save_kv_layer_from(cluster, l, 0, len)
    }

    /// `save_kv_layer`, less the positions before `from`.
    pub fn save_kv_layer_from(&self, cluster: &L0Cluster, l: usize, from: usize, len: usize) -> Vec<f32> {
        let shard = &self.shards[self.card_of(l)];
        let card = &cluster.cards[shard.card];
        let li = l - shard.layers.start;
        let g = layer_geom(l);
        let row = g.kv_heads * g.head_dim;
        let high = shard.kv_high[shard.slot()].load(std::sync::atomic::Ordering::SeqCst);
        // A ring holds the last ring's worth of positions, less any the slot wrote past `len`
        // over the oldest of them.
        let first = match kv_ring(shard.max_seq(), l) {
            0 => 0,
            r => len.saturating_sub(r).max(high.saturating_sub(r)),
        }
        .max(from)
        .min(len);
        let rows = (len - first) * row;
        let mut v = vec![0.0f32; 2 * rows];
        for (value, out) in [(false, 0), (true, rows)] {
            let mut at = out;
            for (src, n) in shard.kv_runs(li, first, len, value) {
                card.memcpy_sync(v[at..].as_mut_ptr() as *mut c_void, src as *const c_void, n * row * 4);
                at += n * row;
            }
        }
        v
    }

    /// Writes layer `l` of a snapshot back to the current KV slot: the positions the slot needs
    /// to go on from `snap.len`. A snapshot that does not hold them, a ring's snapshot for a slot
    /// without a ring, is refused before any write.
    pub fn load_kv_layer(&self, cluster: &L0Cluster, l: usize, snap: &KvSnapshot) -> Result<(), String> {
        let shard = &self.shards[self.card_of(l)];
        let dev = &cluster.cards[shard.card];
        let li = l - shard.layers.start;
        let row = layer_geom(l).kv_heads * layer_geom(l).head_dim;
        let v = &snap.layers[l];
        let held = v.len() / 2 / row;
        let have = snap.len - held;
        let ring = kv_ring(shard.max_seq(), l);
        let need = kv_keep_from(shard.max_seq(), l, snap.len).max(have);
        if ring == 0 && have != 0 {
            return Err(format!(
                "layer {}: the snapshot holds positions {}.., the slot needs all of them",
                l, have
            ));
        }
        if snap.len > shard.max_seq() {
            return Err(format!(
                "layer {}: the snapshot holds {} positions, the slot {}",
                l,
                snap.len,
                shard.max_seq()
            ));
        }
        if ring > 0 {
            shard.kv_high[shard.slot()].store(
                kv_high_after_load(need, snap.len, ring),
                std::sync::atomic::Ordering::SeqCst,
            );
        }
        for (value, base) in [(false, 0), (true, held * row)] {
            let mut at = base + (need - have) * row;
            for (dst, n) in shard.kv_runs(li, need, snap.len, value) {
                shard.upload(dev, dst, &v[at..at + n * row]);
                at += n * row;
            }
        }
        Ok(())
    }

    /// Writes a snapshot's KV rows back to the current slot on both cards.
    pub fn load_kv(&self, cluster: &L0Cluster, snap: &KvSnapshot) -> Result<(), String> {
        self.load_kv_card(cluster, 0, snap)?;
        self.load_kv_card(cluster, 1, snap)
    }

    /// Writes a snapshot's KV rows for one card's layers.
    pub fn load_kv_card(&self, cluster: &L0Cluster, card: usize, snap: &KvSnapshot) -> Result<(), String> {
        for l in self.shards[card].layers.clone() {
            self.load_kv_layer(cluster, l, snap)?;
        }
        Ok(())
    }

    pub fn max_seq(&self) -> usize {
        self.shards[0].max_seq()
    }

    /// Whether KV `slot` can go on from position `keep` without feeding earlier positions
    /// again (see `kv_can_rewind`).
    pub fn kv_can_resume(&self, slot: usize, keep: usize) -> bool {
        self.shards.iter().all(|s| {
            kv_can_rewind(
                s.slot_seq[slot],
                s.kv_high[slot].load(std::sync::atomic::Ordering::SeqCst),
                keep,
            )
        })
    }

    /// Marks KV `slot` as starting over from position 0: its ring then holds what is written
    /// next.
    pub fn kv_restart(&self, slot: usize) {
        for s in &self.shards {
            s.kv_high[slot].store(0, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Selects the KV slot every following pass reads and writes, on both cards.
    pub fn set_slot(&self, slot: usize) {
        for s in &self.shards {
            s.set_slot(slot);
        }
    }

    pub fn embed_bytes(&self) -> &[u8] {
        self.ckpt
            .bf16(&format!("{}embed_tokens.weight", PREFIX), &[VOCAB, HIDDEN])
            .expect("checked at load")
    }

    /// Runs `tokens` at positions `pos0..` through both cards in blocks of `max_block`, writing
    /// every layer's KV cache. Returns the softcapped logits at the last token.
    pub fn forward(&self, cluster: &L0Cluster, tokens: &[u32], pos0: usize) -> Result<(Vec<f32>, PassTiming), String> {
        self.forward_blocks(cluster, tokens, pos0, self.shards[0].max_block)
    }

    /// Runs one block through card 0's layers. Returns the residual leaving layer 14 and the
    /// card time in microseconds.
    pub fn run_card0(
        &self,
        cluster: &L0Cluster,
        tokens: &[u32],
        pos0: usize,
        meta: Option<&[[u32; 4]]>,
    ) -> (Vec<f32>, u128) {
        let n = tokens.len();
        let s0 = &self.shards[0];
        let mut residual = vec![0.0f32; n * HIDDEN];
        let out = unsafe { std::slice::from_raw_parts_mut(residual.as_mut_ptr() as *mut u8, n * HIDDEN * 4) };
        let us = unsafe { run_first_stage(s0, &cluster.cards[0], self.embed_bytes(), tokens, pos0, meta, out) };
        self.count_routes(s0, n, pos0);
        (residual, us)
    }

    /// Runs one block's residual through card 1's layers and the head. Returns the logits of
    /// each `logit_rows` row and the card time in microseconds.
    pub fn run_card1(
        &self,
        cluster: &L0Cluster,
        residual: &[f32],
        pos0: usize,
        meta: Option<&[[u32; 4]]>,
        logit_rows: &[usize],
    ) -> (Vec<Vec<f32>>, u128) {
        let n = residual.len() / HIDDEN;
        let s1 = &self.shards[1];
        let us = unsafe { run_last_stage(s1, &cluster.cards[1], f32_bytes(residual), pos0, meta, logit_rows) };
        self.count_routes(s1, n, pos0);
        let out = (0..logit_rows.len()).map(|i| s1.logits_row(i).to_vec()).collect();
        (out, us)
    }

    /// Runs `jobs` through both cards, one block after the other: card 0, then card 1. A job with `kv` first
    /// restores each card's layers from it. Returns each job's logits.
    ///
    /// The two cards do not overlap here. The nagaya's stage threads run the two stages of the `Engine` at the same
    /// time (engine-abi D1), and they give the same bits, because each card runs the same task ring.
    pub fn run_jobs(&self, cluster: &L0Cluster, jobs: &[PipeJob], timing: &mut PassTiming) -> Vec<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            if let Some(kv) = job.kv {
                self.load_kv(cluster, kv)
                    .expect("the caller gives a snapshot that fits the slot");
            }
            let (residual, us0) = self.run_card0(cluster, job.tokens, job.pos0, job.meta);
            let (logits, us1) = self.run_card1(cluster, &residual, job.pos0, job.meta, &job.logit_rows);
            timing.card_us[0] += us0;
            timing.card_us[1] += us1;
            timing.blocks += 1;
            out.push(logits);
        }
        out
    }

    /// Lays out `shared` at positions `pos0..` and each branch as its own continuation of it.
    /// Returns `None` when they do not fit one block.
    pub fn branch_block(
        &self,
        shared: &[u32],
        branches: &[Vec<u32>],
        pos0: usize,
    ) -> Result<Option<BranchBlock>, String> {
        if branches.is_empty() || branches.len() > MAX_LOGIT_ROWS || branches.iter().any(|b| b.is_empty()) {
            return Err(format!("1 to {} non-empty branches", MAX_LOGIT_ROWS));
        }
        let branch_tokens: usize = branches.iter().map(|b| b.len()).sum();
        if pos0 + shared.len() + branch_tokens > self.max_seq() {
            return Err("branches exceed max_seq".into());
        }
        if let Some(&bad) = shared
            .iter()
            .chain(branches.iter().flatten())
            .find(|&&t| t as usize >= VOCAB)
        {
            return Err(format!("token id {} is outside the vocabulary", bad));
        }
        if shared.len() + branch_tokens > self.shards[0].max_block {
            return Ok(None);
        }
        let shared_end = (pos0 + shared.len()) as u32;
        let mut tokens: Vec<u32> = shared.to_vec();
        let mut meta: Vec<[u32; 4]> = (0..shared.len()).map(|i| [(pos0 + i) as u32, 0, 0, 0]).collect();
        let mut logit_rows = Vec::new();
        for b in branches {
            let first_row = (pos0 + tokens.len()) as u32;
            for (j, &t) in b.iter().enumerate() {
                tokens.push(t);
                meta.push([shared_end + j as u32, shared_end, first_row, 0]);
            }
            logit_rows.push(tokens.len() - 1);
        }
        Ok(Some(BranchBlock {
            tokens,
            meta,
            logit_rows,
            pos0,
        }))
    }

    /// Runs `shared` at positions `pos0..`, then each branch as its own continuation of
    /// `shared` in one block: a branch never sees another. When everything does not fit one
    /// block, the front of `shared` runs first on its own. Returns the logits at each branch's
    /// last token. KV rows past `pos0 + shared.len()` hold the branches afterwards, so a later
    /// forward from that position overwrites them.
    pub fn forward_branches(
        &self,
        cluster: &L0Cluster,
        shared: &[u32],
        branches: &[Vec<u32>],
        pos0: usize,
    ) -> Result<(Vec<Vec<f32>>, PassTiming), String> {
        let branch_tokens: usize = branches.iter().map(|b| b.len()).sum();
        let block = self.shards[0].max_block;
        if branch_tokens > block {
            return Err(format!(
                "{} branch tokens do not fit one {}-token block",
                branch_tokens, block
            ));
        }
        let t_wall = Instant::now();
        let mut timing = PassTiming::default();
        let keep = (shared.len() + branch_tokens).saturating_sub(block);
        let plan = self
            .branch_block(&shared[keep..], branches, pos0 + keep)?
            .expect("the tail and branches fit one block");
        let mut jobs: Vec<PipeJob> = (0..keep)
            .step_by(block)
            .map(|at| PipeJob::plain(&shared[at..keep.min(at + block)], pos0 + at, Vec::new()))
            .collect();
        jobs.push(PipeJob {
            tokens: &plan.tokens,
            pos0: plan.pos0,
            meta: Some(&plan.meta),
            logit_rows: plan.logit_rows.clone(),
            kv: None,
        });
        let logits = self.run_jobs(cluster, &jobs, &mut timing).pop().unwrap();
        timing.wall_us = t_wall.elapsed().as_micros();
        Ok((logits, timing))
    }

    pub fn forward_blocks(
        &self,
        cluster: &L0Cluster,
        tokens: &[u32],
        pos0: usize,
        block: usize,
    ) -> Result<(Vec<f32>, PassTiming), String> {
        if block == 0 || block > self.shards[0].max_block {
            return Err(format!("block size {} outside 1..={}", block, self.shards[0].max_block));
        }
        self.check_tokens(tokens, pos0)?;
        let t_wall = Instant::now();
        let mut timing = PassTiming::default();
        let jobs = Self::block_jobs(tokens, pos0, block);
        let logits = self.run_jobs(cluster, &jobs, &mut timing).pop().unwrap().pop().unwrap();
        timing.wall_us = t_wall.elapsed().as_micros();
        Ok((logits, timing))
    }

    /// Checks that `tokens` at positions `pos0..` can run: at least one, inside max_seq and the
    /// vocabulary.
    pub fn check_tokens(&self, tokens: &[u32], pos0: usize) -> Result<(), String> {
        self.check_tokens_in(tokens, pos0, self.shards[0].max_seq())
    }

    /// `check_tokens` against a slot of `max_seq` positions.
    pub fn check_tokens_in(&self, tokens: &[u32], pos0: usize, max_seq: usize) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("forward needs at least one token".into());
        }
        if pos0 + tokens.len() > max_seq {
            return Err(format!(
                "{} tokens from position {} exceed max_seq {}",
                tokens.len(),
                pos0,
                max_seq
            ));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= VOCAB) {
            return Err(format!("token id {} is outside the vocabulary", bad));
        }
        Ok(())
    }

    /// `tokens` from `pos0` as plain jobs of up to `block` tokens; the last takes logits at its
    /// last token.
    pub fn block_jobs(tokens: &[u32], pos0: usize, block: usize) -> Vec<PipeJob<'_>> {
        let mut jobs = Vec::new();
        let mut at = 0;
        while at < tokens.len() {
            let n = block.min(tokens.len() - at);
            let rows = if at + n == tokens.len() {
                vec![n - 1]
            } else {
                Vec::new()
            };
            jobs.push(PipeJob::plain(&tokens[at..at + n], pos0 + at, rows));
            at += n;
        }
        jobs
    }
}

/// The bytes of f32 values, little-endian.
#[cfg(feature = "xe")]
pub(crate) fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

/// Runs one block through the layers of the first card: writes `embed_tokens[id] * sqrt(hidden_size)` for each token
/// into the card's residual, runs the card's ring, and copies the residual that leaves its last layer into `out`.
/// Returns the card time in microseconds.
///
/// The `Engine` stages and `Gemma4Engine` both run a block through this function, so they give the same bits.
///
/// # Safety
///
/// The caller checked the block: `tokens` fit `max_block`, the positions fit the current slot, and `out` holds one
/// row of `HIDDEN` values for each token.
#[cfg(feature = "xe")]
pub(crate) unsafe fn run_first_stage(
    shard: &CardShard,
    card: &B70Card,
    embed: &[u8],
    tokens: &[u32],
    pos0: usize,
    meta: Option<&[[u32; 4]]>,
    out: &mut [u8],
) -> u128 {
    let n = tokens.len();
    assert_eq!(out.len(), n * HIDDEN * 4, "the residual holds one row for each token");
    let scale = (HIDDEN as f32).sqrt();
    let mut rows = Vec::with_capacity(n * HIDDEN);
    for &t in tokens {
        rows.extend(embedding_row_f32(embed, HIDDEN, t, scale));
    }
    shard.write(card, Buf::X, 0, &rows);
    let us = unsafe {
        let tasks = shard.build_ring(n, pos0, &[], meta).len();
        shard.run(card, tasks)
    };
    card.memcpy_sync(
        out.as_mut_ptr() as *mut c_void,
        shard.buf_ptr(Buf::X, 0) as *const c_void,
        n * HIDDEN * 4,
    );
    us
}

/// Runs one block through the layers of the last card and the head: copies `residual` (little-endian f32 rows of
/// `HIDDEN` values) into the card's residual and runs the card's ring. The logits of each `logit_rows` row are then in
/// `CardShard::logits_row`. Returns the card time in microseconds.
///
/// # Safety
///
/// As for `run_first_stage`, and the shard holds the head.
#[cfg(feature = "xe")]
pub(crate) unsafe fn run_last_stage(
    shard: &CardShard,
    card: &B70Card,
    residual: &[u8],
    pos0: usize,
    meta: Option<&[[u32; 4]]>,
    logit_rows: &[usize],
) -> u128 {
    let n = residual.len() / (HIDDEN * 4);
    assert_eq!(residual.len(), n * HIDDEN * 4, "the residual holds whole rows");
    shard.upload_bytes(card, shard.buf_ptr(Buf::X, 0), residual);
    unsafe {
        let tasks = shard.build_ring(n, pos0, logit_rows, meta).len();
        shard.run(card, tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positions `first..len` of every layer: K rows hold the position, V rows the position
    /// plus a half.
    fn rows(first: usize, len: usize) -> KvSnapshot {
        let layers = (0..NUM_LAYERS)
            .map(|l| {
                let row = layer_geom(l).kv_heads * layer_geom(l).head_dim;
                let mut v: Vec<f32> = (first..len).flat_map(|p| std::iter::repeat_n(p as f32, row)).collect();
                v.extend((first..len).flat_map(|p| std::iter::repeat_n(p as f32 + 0.5, row)));
                v
            })
            .collect();
        KvSnapshot::from_layers(len, layers)
    }

    /// The positions each layer of `s` holds, from its K rows, checked against its V rows.
    fn positions(s: &KvSnapshot, l: usize) -> Vec<usize> {
        let row = layer_geom(l).kv_heads * layer_geom(l).head_dim;
        let held = s.layers[l].len() / 2 / row;
        (0..held)
            .map(|i| {
                let (k, v) = (s.layers[l][i * row], s.layers[l][(held + i) * row + row - 1]);
                assert_eq!(v, k + 0.5, "layer {} row {}", l, i);
                k as usize
            })
            .collect()
    }

    /// A chain of snapshots, each the rows after its parent, joins into every position a slot
    /// without a ring needs, and into the window a ring slot needs; a chain with a gap a slot
    /// needs fails.
    ///
    /// Falsified by a join that reorders runs, repeats a position, or accepts a gap.
    #[test]
    fn a_snapshot_chain_joins_into_the_rows_a_slot_needs() {
        let (a, b, c) = (rows(0, 100), rows(100, 250), rows(250, 300));
        let joined = KvSnapshot::join(&[&c, &b, &a], 4096).unwrap();
        for l in 0..NUM_LAYERS {
            assert_eq!(positions(&joined, l), (0..300).collect::<Vec<_>>(), "layer {}", l);
        }
        // A long chain in a ring slot: global layers keep everything, sliding layers a ring.
        let (d, e) = (rows(0, 9000), rows(9000, 9500));
        let joined = KvSnapshot::join(&[&e, &d], 65536).unwrap();
        for l in 0..NUM_LAYERS {
            let want: Vec<usize> = (kv_keep_from(65536, l, 9500)..9500).collect();
            assert_eq!(positions(&joined, l), want, "layer {}", l);
        }
        // A child saved after its slot lost rows before it: fine for a ring slot that needs
        // only the window, a gap for a slot that needs every row.
        let late = rows(5000, 9500);
        let gapped = [&late, &a];
        assert!(KvSnapshot::join(&gapped, 4096 * 2).is_err());
        let ring = KvSnapshot::join(&[&rows(8000, 9500)], 65536);
        assert!(ring.is_err(), "a global layer needs every row");
    }

    /// A loaded snapshot that holds every row from 0 can be rewound into, as a slot that fed
    /// those rows can; one that holds only a window cannot go back before it.
    #[test]
    fn a_loaded_snapshot_rewinds_as_far_as_its_rows_reach() {
        let (seq, ring) = (65536, KV_RING_ROWS);
        let high = kv_high_after_load(0, 67, ring);
        assert!(
            kv_can_rewind(seq, high.max(232), 217),
            "a short snapshot, fed on to 232, goes back to 217"
        );
        assert!(kv_can_rewind(seq, high, 0));
        let need = 9000 - SLIDING_WINDOW as usize;
        let high = kv_high_after_load(need, 9000, ring);
        assert!(kv_can_rewind(seq, high.max(9100), 9000 + 16 + 8));
        assert!(
            !kv_can_rewind(seq, high.max(9100), 8990),
            "the window before 8990 starts before the snapshot's rows"
        );
    }

    fn snapshot(len: usize) -> KvSnapshot {
        let layers = (0..NUM_LAYERS)
            .map(|l| {
                let g = layer_geom(l);
                (0..2 * len * g.kv_heads * g.head_dim)
                    .map(|i| (i as f32) * 0.5 - l as f32)
                    .collect()
            })
            .collect();
        KvSnapshot { len, layers }
    }

    #[test]
    fn snapshot_files_round_trip_and_reject_truncation() {
        let snap = snapshot(3);
        let mut bytes = Vec::new();
        snap.write_to(&mut bytes).unwrap();
        let back = KvSnapshot::read_from(&mut bytes.as_slice()).unwrap();
        assert_eq!(back.len, 3);
        assert_eq!(back.layers, snap.layers);
        assert!(KvSnapshot::read_from(&mut bytes[..bytes.len() - 4].as_ref()).is_err());
        let mut longer = bytes.clone();
        longer[8] = 4; // each layer then holds the last 3 of 4 positions, as a ring saves them
        assert_eq!(KvSnapshot::read_from(&mut longer.as_slice()).unwrap().len, 4);
        let mut wrong = bytes.clone();
        wrong[8] = 2; // 3 rows cannot be positions of a length of 2
        assert!(KvSnapshot::read_from(&mut wrong.as_slice()).is_err());
    }
}
