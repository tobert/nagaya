//! The vision tower and the projector of `LFM3-Omni-2B`: their numbers, their weights, their buffers, and the ring of
//! the tower for one image. The text model has none of them.
//!
//! The ring follows `Lfm3VlVisionModel.forward` of `modeling_lfm3_vl.py` and `Lfm3OmniMultiModalProjector` of
//! `modeling_lfm3_omni.py` (`LFM3-Omni-2B` at `21d75123`):
//!
//! - the patch embedding (`patchifier`, with no bias) and `embedding_norm`;
//! - for each layer: `operator_norm`, then the gated depthwise conv or the attention in both directions, added to the
//!   residual; then `ffn_norm` and the SwiGLU MLP, added to the residual. The layers have no post norms;
//! - the merge of each block of 2 by 2 patches into one row, `layer_norm`, `linear_1`, GELU and `linear_2`.
//!
//! The rows of an image are in the order of the merge blocks, so the 4 rows of a block are next to each other and the
//! merge is a view of the rows. The CPU forms of this path are `tests/vision_cpu.rs`.

use std::collections::BTreeSet;
use std::ops::Range;

use nagaya_mk_bmg::protocol::{FLAG_ACCUMULATE, MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView, bf16_to_f32};
use nagaya_mk_bmg::tasks;

use crate::media::{MAX_IMAGE_PATCHES, PATCH_ROW};
use crate::weights::{Upload, rounded};

/// The prefix of the tensors of the tower and of the projector.
const TOWER: &str = "model.vision_tower.";
const PROJECTOR: &str = "model.multi_modal_projector.";

/// The operator of one layer of the tower (`vision_config.layer_types`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionLayer {
    /// The gated depthwise 3 by 3 convolution over the patch grid.
    Conv,
    /// Attention in both directions over the patches of a frame.
    Attention,
}

/// The numbers of the tower and of the projector, from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Vision {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub layers: Vec<VisionLayer>,
    pub eps: f32,
    /// The theta of the axial rotation.
    pub theta: f32,
    /// The rows of `linear_1`.
    pub projector_hidden: usize,
    pub projector_eps: f32,
    /// The width of a row of the backbone: the rows of `linear_2`.
    pub out: usize,
}

impl Vision {
    /// Read the numbers from `config.json` of Omni, with the width of the backbone. A value that the ops cannot run is
    /// an error that names the key.
    pub fn parse(text: &str, out: usize) -> Result<Vision, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let c = &root["vision_config"];
        let get = |at: &serde_json::Value, key: &str| -> Result<u64, String> {
            at[key]
                .as_u64()
                .ok_or_else(|| format!("config.json has {key} {}, not a count", at[key]))
        };
        let count = |key: &str| {
            get(c, key)
                .map(|v| v as usize)
                .map_err(|e| format!("vision_config.{e}"))
        };
        let want = |key: &str, ok: bool, need: &str| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(format!(
                    "config.json has {key}, and the vision tower of this engine needs {need}"
                ))
            }
        };
        if c["model_type"] != "lfm3_vl_vision" {
            return Err(format!(
                "config.json has vision_config.model_type {}, not \"lfm3_vl_vision\"",
                c["model_type"]
            ));
        }
        let hidden = count("hidden_size")?;
        let heads = count("num_attention_heads")?;
        let kv_heads = count("num_key_value_heads")?;
        let intermediate = count("intermediate_size")?;
        let n_layers = count("num_hidden_layers")?;
        want(
            "vision_config.hidden_size",
            hidden >= 16 && hidden.is_multiple_of(16),
            "a multiple of 16",
        )?;
        want(
            "vision_config.num_attention_heads",
            heads > 0 && hidden.is_multiple_of(heads),
            "a divisor of hidden_size",
        )?;
        let head_dim = hidden / heads;
        // OP_BIATTN takes heads of 16 to 512 values in 16s.
        want(
            "vision_config.hidden_size",
            (16..=512).contains(&head_dim) && head_dim.is_multiple_of(16),
            "heads of 16 to 512 values in 16s",
        )?;
        want(
            "vision_config.num_key_value_heads",
            kv_heads > 0 && heads.is_multiple_of(kv_heads),
            "a divisor of the head count",
        )?;
        want(
            "vision_config.intermediate_size",
            intermediate > 0 && intermediate.is_multiple_of(16),
            "a multiple of 16",
        )?;
        for (key, need) in [
            ("patch_size", 16),
            ("temporal_patch_size", 2),
            ("in_channels", 3),
            ("spatial_merge_size", 2),
            ("conv_kernel_size", 3),
        ] {
            want(
                &format!("vision_config.{key} {}", c[key]),
                c[key].as_u64() == Some(need),
                &need.to_string(),
            )?;
        }
        want(
            &format!("downsample_factor {}", root["downsample_factor"]),
            root["downsample_factor"].as_u64() == Some(2),
            "2",
        )?;
        want(
            &format!("projector_hidden_act {}", root["projector_hidden_act"]),
            root["projector_hidden_act"] == "gelu",
            "\"gelu\", the GELU with erf",
        )?;
        want(
            &format!("projector_bias {}", root["projector_bias"]),
            root["projector_bias"] == true,
            "true",
        )?;
        let rope = &c["rope_parameters"];
        want(
            &format!("vision_config.rope_parameters {rope}"),
            rope["rope_type"] == "axial" && rope["rope_theta"].as_f64().is_some(),
            "the axial rope with a theta",
        )?;
        let types = c["layer_types"].as_array().ok_or_else(|| {
            format!(
                "config.json has vision_config.layer_types {}, not a list",
                c["layer_types"]
            )
        })?;
        let layers = types
            .iter()
            .map(|t| match t.as_str() {
                Some("conv") => Ok(VisionLayer::Conv),
                Some("full_attention") => Ok(VisionLayer::Attention),
                _ => Err(format!(
                    "config.json has the layer type {t} in vision_config.layer_types; the tower runs conv and \
                     full_attention"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.len() != n_layers {
            return Err(format!(
                "config.json has {} vision layer types and vision_config.num_hidden_layers {n_layers}",
                layers.len()
            ));
        }
        let projector_hidden = get(&root, "projector_hidden_size")? as usize;
        want(
            "projector_hidden_size",
            projector_hidden > 0 && projector_hidden.is_multiple_of(16),
            "a multiple of 16",
        )?;
        let float = |at: &serde_json::Value, key: &str| -> Result<f32, String> {
            at[key]
                .as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| format!("config.json has {key} {}, not a number", at[key]))
        };
        let v = Vision {
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            layers,
            eps: float(c, "norm_eps").map_err(|e| format!("vision_config.{e}"))?,
            theta: rope["rope_theta"].as_f64().expect("checked above") as f32,
            projector_hidden,
            projector_eps: float(&root, "projector_layer_norm_eps")?,
            out,
        };
        // The buffers below hold q, k and v in the rows of in_proj, and the projector in the MLP rows.
        assert!(v.heads * v.head_dim + 2 * v.kv_row() <= 3 * v.hidden);
        if v.projector_hidden > 4 * v.intermediate || v.out > 2 * v.hidden {
            return Err(format!(
                "config.json has projector_hidden_size {} and a backbone of {}; the buffers of the tower hold at most \
                 {} and {}",
                v.projector_hidden,
                v.out,
                4 * v.intermediate,
                2 * v.hidden
            ));
        }
        Ok(v)
    }

    /// The values of one key row or one value row: the key and value heads.
    pub fn kv_row(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The values of a merged row: 2 by 2 rows of the tower.
    pub fn merged(&self) -> usize {
        4 * self.hidden
    }
}

/// The placeholders of the media of Omni, from `image_token_id` and `audio_token_id` of `config.json`. The audio
/// encoder is not built yet, so the rules refuse a clip.
pub fn media_rules(text: &str) -> Result<crate::media::MediaRules, String> {
    let root: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
    let id = |key: &str| -> Result<u32, String> {
        root[key]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("config.json has {key} {}, not a token id", root[key]))
    };
    Ok(crate::media::MediaRules {
        image_token: Some(id("image_token_id")?),
        audio_token: Some(id("audio_token_id")?),
        video_token: Some(id("video_token_id")?),
        // The engine sets the placeholder of speech when it loads the codebook table.
        audio_out_token: None,
        audio_encoder: false,
        // The engine sets the rule of latent rows for each model.
        latent: None,
    })
}

/// The tensors of the tower and of the projector with their shapes, by their whole name.
pub fn tensors(v: &Vision) -> Vec<(String, Vec<usize>)> {
    let (h, i) = (v.hidden, v.intermediate);
    let mut all = vec![
        (format!("{TOWER}patchifier.weight"), vec![h, PATCH_ROW]),
        (format!("{TOWER}embedding_norm.weight"), vec![h]),
    ];
    for (l, kind) in v.layers.iter().enumerate() {
        let mut layer = vec![
            ("operator_norm.weight", vec![h]),
            ("ffn_norm.weight", vec![h]),
            ("feed_forward.w1.weight", vec![i, h]),
            ("feed_forward.w3.weight", vec![i, h]),
            ("feed_forward.w2.weight", vec![h, i]),
        ];
        match kind {
            VisionLayer::Conv => layer.extend([
                ("conv.in_proj.weight", vec![3 * h, h]),
                ("conv.spatial_conv.weight", vec![h, 1, 3, 3]),
                ("conv.out_proj.weight", vec![h, h]),
            ]),
            VisionLayer::Attention => layer.extend([
                ("self_attn.q_proj.weight", vec![v.heads * v.head_dim, h]),
                ("self_attn.k_proj.weight", vec![v.kv_row(), h]),
                ("self_attn.v_proj.weight", vec![v.kv_row(), h]),
                ("self_attn.out_proj.weight", vec![h, v.heads * v.head_dim]),
                ("self_attn.q_layernorm.weight", vec![v.head_dim]),
                ("self_attn.k_layernorm.weight", vec![v.head_dim]),
            ]),
        }
        all.extend(layer.into_iter().map(|(n, s)| (format!("{TOWER}layers.{l}.{n}"), s)));
    }
    let p = v.projector_hidden;
    all.extend([
        (format!("{PROJECTOR}layer_norm.weight"), vec![h]),
        (format!("{PROJECTOR}layer_norm.bias"), vec![h]),
        (format!("{PROJECTOR}linear_1.weight"), vec![p, v.merged()]),
        (format!("{PROJECTOR}linear_1.bias"), vec![p]),
        (format!("{PROJECTOR}linear_2.weight"), vec![v.out, p]),
        (format!("{PROJECTOR}linear_2.bias"), vec![v.out]),
    ]);
    all
}

/// Check the tensors of the tower and of the projector in `view`: each one is there as BF16 with its shape, and no
/// other tensor has their prefixes. The message names the tensor.
pub fn check(v: &Vision, view: &SafeTensorsView<'_>) -> Result<(), String> {
    let want = tensors(v);
    for (name, shape) in &want {
        let meta = view
            .tensors
            .get(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        if meta.dtype != Dtype::BF16 || meta.shape != *shape {
            return Err(format!(
                "tensor {name} is {:?} {:?}, and the vision tower needs BF16 {shape:?}",
                meta.dtype, meta.shape
            ));
        }
    }
    let known: BTreeSet<&String> = want.iter().map(|(n, _)| n).collect();
    let mut other: Vec<&String> = view
        .tensors
        .keys()
        .filter(|name| !known.contains(name) && (name.starts_with(TOWER) || name.starts_with(PROJECTOR)))
        .collect();
    other.sort();
    if let Some(name) = other.first() {
        return Err(format!(
            "the checkpoint has the tensor {name}, which the vision tower of LFM3-Omni-2B does not use"
        ));
    }
    Ok(())
}

/// The weights of one layer of the tower on the card. A field of the other operator is 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct VisionLayerWeights {
    pub operator_norm: u64,
    pub ffn_norm: u64,
    /// The gate rows (`w1`), then the up rows (`w3`): `[2 * intermediate][hidden]`.
    pub w13: u64,
    pub w2: u64,
    /// The rows B, C and x: `[3 * hidden][hidden]`.
    pub in_proj: u64,
    /// The taps of each channel: `[hidden][9]`, down then across.
    pub conv: u64,
    pub out_proj: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub q_norm: u64,
    pub k_norm: u64,
}

/// The weights of the tower and of the projector on the card. Every tensor stays BF16, except the weights and the
/// bias of `layer_norm`, which `OP_LAYERNORM` reads as F32. BF16 values are exact in F32.
#[derive(Clone, Debug)]
pub struct VisionWeights {
    pub patchifier: u64,
    pub embedding_norm: u64,
    pub layers: Vec<VisionLayerWeights>,
    pub ln_w: u64,
    pub ln_b: u64,
    pub linear_1: u64,
    pub linear_1_bias: u64,
    pub linear_2: u64,
    pub linear_2_bias: u64,
}

/// Load the tower and the projector of `view` into `sink`. A checkpoint that fails `check` uploads nothing.
pub fn load(sink: &mut dyn Upload, v: &Vision, view: &SafeTensorsView<'_>) -> Result<VisionWeights, String> {
    check(v, view)?;
    let bytes = |name: &str| -> &[u8] { view.tensor_bytes(name).expect("check found each tensor") };
    let mut put = |names: &[String], f32_values: bool| -> u64 {
        let parts: Vec<&[u8]> = names.iter().map(|n| bytes(n)).collect();
        let len: usize = parts.iter().map(|p| p.len()).sum();
        let at = sink.alloc(if f32_values { 2 * len } else { len });
        if sink.writes() {
            let mut offset = 0u64;
            for part in parts {
                let data: Vec<u8> = if f32_values {
                    part.as_chunks::<2>()
                        .0
                        .iter()
                        .flat_map(|b| bf16_to_f32(u16::from_le_bytes(*b)).to_le_bytes())
                        .collect()
                } else {
                    part.to_vec()
                };
                sink.write(at + offset, &data);
                offset += data.len() as u64;
            }
        }
        at
    };
    let t = |name: &str| vec![format!("{TOWER}{name}")];
    let patchifier = put(&t("patchifier.weight"), false);
    let embedding_norm = put(&t("embedding_norm.weight"), false);
    let mut layers = Vec::with_capacity(v.layers.len());
    for (l, kind) in v.layers.iter().enumerate() {
        let n = |name: &str| vec![format!("{TOWER}layers.{l}.{name}")];
        let mut layer = VisionLayerWeights {
            operator_norm: put(&n("operator_norm.weight"), false),
            ffn_norm: put(&n("ffn_norm.weight"), false),
            w13: put(
                &[
                    format!("{TOWER}layers.{l}.feed_forward.w1.weight"),
                    format!("{TOWER}layers.{l}.feed_forward.w3.weight"),
                ],
                false,
            ),
            w2: put(&n("feed_forward.w2.weight"), false),
            ..VisionLayerWeights::default()
        };
        match kind {
            VisionLayer::Conv => {
                layer.in_proj = put(&n("conv.in_proj.weight"), false);
                layer.conv = put(&n("conv.spatial_conv.weight"), false);
                layer.out_proj = put(&n("conv.out_proj.weight"), false);
            }
            VisionLayer::Attention => {
                layer.q = put(&n("self_attn.q_proj.weight"), false);
                layer.k = put(&n("self_attn.k_proj.weight"), false);
                layer.v = put(&n("self_attn.v_proj.weight"), false);
                layer.o = put(&n("self_attn.out_proj.weight"), false);
                layer.q_norm = put(&n("self_attn.q_layernorm.weight"), false);
                layer.k_norm = put(&n("self_attn.k_layernorm.weight"), false);
            }
        }
        layers.push(layer);
    }
    let p = |name: &str| vec![format!("{PROJECTOR}{name}")];
    Ok(VisionWeights {
        patchifier,
        embedding_norm,
        layers,
        ln_w: put(&p("layer_norm.weight"), true),
        ln_b: put(&p("layer_norm.bias"), true),
        linear_1: put(&p("linear_1.weight"), false),
        linear_1_bias: put(&p("linear_1.bias"), false),
        linear_2: put(&p("linear_2.weight"), false),
        linear_2_bias: put(&p("linear_2.bias"), false),
    })
}

/// The buffers of the tower for an image of at most [`MAX_IMAGE_PATCHES`] patches, and the media rows of a block.
#[derive(Clone, Copy, Debug)]
pub struct VisionBufs {
    /// The patch rows of the image: `[patches][PATCH_ROW]`.
    pub pixels: u64,
    /// The residual stream: `[patches][hidden]`.
    pub x: u64,
    /// The normed rows: `[patches][hidden]`. The projector's LayerNorm writes its rows here too, as merged rows.
    pub h: u64,
    /// The rows of `in_proj`, `[patches][3 * hidden]`. An attention layer keeps q, k and v here instead.
    pub bcx: u64,
    /// The output of an operator: `[patches][hidden]`. The projector writes its rows here.
    pub y: u64,
    /// The SwiGLU rows, `[patches][intermediate]`. The projector keeps the rows of `linear_1` here.
    pub g: u64,
    /// The rows of the encoders for the positions of a block: `[max_block][out]`. The backbone copies them over the
    /// embedding rows of the placeholders.
    pub media: u64,
}

impl VisionBufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, v: &Vision, max_block: usize) -> Self {
        let f = |values: usize| values * MAX_IMAGE_PATCHES * 4;
        Self {
            pixels: z(f(PATCH_ROW)),
            x: z(f(v.hidden)),
            h: z(f(v.hidden)),
            bcx: z(f(3 * v.hidden)),
            y: z(f(v.hidden)),
            g: z(f(v.intermediate)),
            media: z(max_block * v.out * 4),
        }
    }

    /// The query rows, `[patches][heads * head_dim]`, at the start of `bcx`.
    pub fn q(&self) -> u64 {
        self.bcx
    }

    /// The key rows after the queries of the largest image.
    pub fn k(&self, v: &Vision) -> u64 {
        self.bcx + (MAX_IMAGE_PATCHES * v.heads * v.head_dim * 4) as u64
    }

    /// The value rows after the keys of the largest image.
    pub fn v(&self, v: &Vision) -> u64 {
        self.k(v) + (MAX_IMAGE_PATCHES * v.kv_row() * 4) as u64
    }
}

/// The device bytes of the buffers, with no card: the plan (spec 2.4).
pub fn bufs_bytes(v: &Vision, max_block: usize) -> usize {
    let mut total = 0usize;
    VisionBufs::new(
        &mut |bytes| {
            total += rounded(bytes);
            0
        },
        v,
        max_block,
    );
    total
}

/// How far a run of the tower goes. The probe of the oracle (8.3) stops early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Until {
    /// After the patch embedding and its norm, in `x`.
    Patches,
    /// After every layer, in `x`.
    Tower,
    /// After the projector, in `y`: one row for each position of the image.
    Projector,
    /// After the projector, with the rows `take` of `y` copied into `media` from row `block_row`.
    Rows { take: (usize, usize), block_row: usize },
}

fn accumulate(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_ACCUMULATE;
    task
}

/// The tasks of the tower and of the projector for one image of `grid_h` by `grid_w` patches in `bufs.pixels`.
pub fn tower(
    v: &Vision,
    w: &VisionWeights,
    bufs: &VisionBufs,
    grid_h: usize,
    grid_w: usize,
    until: Until,
) -> Vec<MkTaskDescriptor> {
    let n = grid_h * grid_w;
    assert!(
        (4..=MAX_IMAGE_PATCHES).contains(&n) && grid_h.is_multiple_of(2) && grid_w.is_multiple_of(2),
        "an image of {grid_h} by {grid_w} patches does not fit the tower; the engine checks the span first"
    );
    let (n32, h32, i32_) = (n as u32, v.hidden as u32, v.intermediate as u32);
    let (heads, kv, d) = (v.heads as u32, v.kv_heads as u32, v.head_dim as u32);
    let (gh, gw) = (grid_h as u32, grid_w as u32);
    let mut t = vec![
        tasks::matmul(bufs.pixels, w.patchifier, bufs.h, n32, h32, PATCH_ROW as u32, false),
        tasks::rmsnorm(bufs.h, w.embedding_norm, bufs.x, n32, h32, v.eps, false),
    ];
    let layers = if until == Until::Patches { 0 } else { v.layers.len() };
    for (l, lw) in w.layers.iter().enumerate().take(layers) {
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n32, h32, v.eps, false));
        match v.layers[l] {
            VisionLayer::Conv => {
                t.push(tasks::matmul(bufs.h, lw.in_proj, bufs.bcx, n32, 3 * h32, h32, false));
                t.push(tasks::vision_conv(bufs.bcx, lw.conv, bufs.y, n32, h32, gh, gw));
                t.push(tasks::matmul(bufs.y, lw.out_proj, bufs.x, n32, h32, h32, true));
            }
            VisionLayer::Attention => {
                let (q, k, vv) = (bufs.q(), bufs.k(v), bufs.v(v));
                t.push(tasks::matmul(bufs.h, lw.q, q, n32, heads * d, h32, false));
                t.push(tasks::matmul(bufs.h, lw.k, k, n32, kv * d, h32, false));
                t.push(tasks::matmul(bufs.h, lw.v, vv, n32, kv * d, h32, false));
                t.push(tasks::vision_rope(
                    q,
                    lw.q_norm,
                    n32 * heads,
                    heads,
                    d,
                    gh,
                    gw,
                    v.eps,
                    v.theta,
                ));
                t.push(tasks::vision_rope(
                    k,
                    lw.k_norm,
                    n32 * kv,
                    kv,
                    d,
                    gh,
                    gw,
                    v.eps,
                    v.theta,
                ));
                let scale = (v.head_dim as f32).powf(-0.5);
                t.push(tasks::biattn(q, k, vv, bufs.y, n32, heads, kv, d, 0, scale));
                t.push(tasks::matmul(bufs.y, lw.o, bufs.x, n32, h32, heads * d, true));
            }
        }
        t.push(tasks::rmsnorm(bufs.x, lw.ffn_norm, bufs.h, n32, h32, v.eps, false));
        t.push(tasks::swiglu(bufs.h, lw.w13, bufs.g, n32, i32_, h32));
        t.push(accumulate(tasks::matmul(bufs.g, lw.w2, bufs.x, n32, h32, i32_, false)));
    }
    if matches!(until, Until::Projector | Until::Rows { .. }) {
        let (m, p, out) = ((n / 4) as u32, v.projector_hidden as u32, v.out as u32);
        t.push(tasks::layernorm(
            bufs.x,
            w.ln_w,
            w.ln_b,
            bufs.h,
            n32,
            h32,
            v.projector_eps,
        ));
        t.push(tasks::matmul(
            bufs.h,
            w.linear_1,
            bufs.g,
            m,
            p,
            v.merged() as u32,
            false,
        ));
        t.push(tasks::bias(w.linear_1_bias, bufs.g, m, p, false));
        t.push(tasks::gelu(bufs.g, bufs.g, m, p));
        t.push(tasks::matmul(bufs.g, w.linear_2, bufs.y, m, out, p, false));
        t.push(tasks::bias(w.linear_2_bias, bufs.y, m, out, false));
    }
    if let Until::Rows {
        take: (lo, hi),
        block_row,
    } = until
    {
        assert!(
            lo < hi && hi <= n / 4,
            "the rows {lo} to {hi} are not rows of an image of {} positions",
            n / 4
        );
        t.push(tasks::copy(
            bufs.y + (lo * v.out * 4) as u64,
            bufs.media + (block_row * v.out * 4) as u64,
            ((hi - lo) * v.out) as u32,
        ));
    }
    assert!(t.len() < MAX_TASKS_PER_CARD, "the tower has {} tasks", t.len());
    t
}

/// The copy of the media rows of `rows` block rows from `block_row` over the embedding rows of the backbone in `x`.
pub fn place(v: &Vision, bufs: &VisionBufs, x: u64, rows: Range<usize>) -> MkTaskDescriptor {
    tasks::copy(
        bufs.media + (rows.start * v.out * 4) as u64,
        x + (rows.start * v.out * 4) as u64,
        (rows.len() * v.out) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nagaya_mk_bmg::protocol::{OP_BIATTN, OP_COPY, OP_MATMUL, OP_VISION_CONV, OP_VISION_ROPE};

    pub(crate) fn vision() -> Vision {
        Vision::parse(include_str!("../tests/fixtures/lfm3-omni.config.json"), 2048).unwrap()
    }

    fn bufs(v: &Vision) -> VisionBufs {
        let mut next = 1u64 << 36;
        VisionBufs::new(
            &mut |bytes| {
                let at = next;
                next += rounded(bytes) as u64;
                at
            },
            v,
            512,
        )
    }

    fn weights(v: &Vision) -> VisionWeights {
        VisionWeights {
            patchifier: 0x1000,
            embedding_norm: 0x2000,
            layers: vec![VisionLayerWeights::default(); v.layers.len()],
            ln_w: 0x3000,
            ln_b: 0x4000,
            linear_1: 0x5000,
            linear_1_bias: 0x6000,
            linear_2: 0x7000,
            linear_2_bias: 0x8000,
        }
    }

    fn count(t: &[MkTaskDescriptor], op: u16) -> usize {
        t.iter().filter(|task| task.opcode == op).count()
    }

    #[test]
    fn the_config_of_omni_gives_the_numbers_of_the_tower() {
        let v = vision();
        assert_eq!(
            (v.hidden, v.heads, v.kv_heads, v.head_dim, v.intermediate),
            (1024, 16, 8, 64, 2560)
        );
        assert_eq!(
            (v.eps, v.theta, v.projector_hidden, v.projector_eps),
            (1e-5, 1e6, 4096, 1e-6)
        );
        let kinds: String = v
            .layers
            .iter()
            .map(|k| if *k == VisionLayer::Conv { 'c' } else { 'a' })
            .collect();
        assert_eq!(kinds, "ccacacacacacac");
        // The tower and the projector: 14 layers of 9 or 11 tensors, 2 before, 6 after.
        assert_eq!(tensors(&v).len(), 2 + 8 * 8 + 6 * 11 + 6);
    }

    #[test]
    fn a_config_that_the_ops_cannot_run_is_refused_with_its_key() {
        let base: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/lfm3-omni.config.json")).unwrap();
        let refused = |edit: &dyn Fn(&mut serde_json::Value), want: &str| {
            let mut c = base.clone();
            edit(&mut c);
            let e = Vision::parse(&c.to_string(), 2048).unwrap_err();
            assert!(e.contains(want), "{e}");
        };
        refused(
            &|c| c["vision_config"]["spatial_merge_size"] = 4.into(),
            "spatial_merge_size 4",
        );
        refused(&|c| c["vision_config"]["patch_size"] = 14.into(), "patch_size 14");
        refused(
            &|c| c["vision_config"]["rope_parameters"]["rope_type"] = "default".into(),
            "the axial rope",
        );
        refused(
            &|c| c["vision_config"]["layer_types"][0] = "sliding_attention".into(),
            "the layer type",
        );
        refused(
            &|c| c["vision_config"]["num_hidden_layers"] = 13.into(),
            "num_hidden_layers 13",
        );
        refused(
            &|c| c["vision_config"]["num_attention_heads"] = 128.into(),
            "heads of 16 to 512",
        );
        refused(
            &|c| c["projector_hidden_act"] = "gelu_pytorch_tanh".into(),
            "GELU with erf",
        );
        refused(&|c| c["projector_bias"] = false.into(), "projector_bias false");
        refused(&|c| c["downsample_factor"] = 3.into(), "downsample_factor 3");
    }

    #[test]
    fn the_tower_runs_each_layer_and_the_projector_copies_the_rows_of_the_block() {
        let v = vision();
        let (w, b) = (weights(&v), bufs(&v));
        let t = tower(
            &v,
            &w,
            &b,
            18,
            26,
            Until::Rows {
                take: (10, 117),
                block_row: 3,
            },
        );
        assert_eq!(count(&t, OP_VISION_CONV), 8);
        assert_eq!(count(&t, OP_VISION_ROPE), 12);
        assert_eq!(count(&t, OP_BIATTN), 6);
        // patchifier; each conv layer in_proj, out_proj, w2; each attention layer q, k, v, out, w2; two linears.
        assert_eq!(count(&t, OP_MATMUL), 1 + 8 * 3 + 6 * 5 + 2);
        let attn = t.iter().find(|task| task.opcode == OP_BIATTN).unwrap();
        assert_eq!((attn.n, attn.m, attn.p0), (468, 16 | 8 << 16, 64));
        assert_eq!(f32::from_bits(attn.layer), 0.125);
        let rope = t.iter().find(|task| task.opcode == OP_VISION_ROPE).unwrap();
        assert_eq!((rope.n, rope.m, rope.k, rope.p0), (468 * 16, 16, 64, 26 | 18 << 16));
        let last = t.last().unwrap();
        assert_eq!(last.opcode, OP_COPY);
        assert_eq!(
            (last.a, last.c, last.n),
            (b.y + 10 * 2048 * 4, b.media + 3 * 2048 * 4, 107 * 2048)
        );
        // The residual takes each operator and each MLP as a sum, as the layers have no post norms.
        let adds = t
            .iter()
            .filter(|task| task.opcode == OP_MATMUL && task.c == b.x && task.flags & FLAG_ACCUMULATE != 0)
            .count();
        assert_eq!(adds, 2 * 14);
        let patches = tower(&v, &w, &b, 18, 26, Until::Patches);
        assert_eq!(patches.len(), 2);
    }

    #[test]
    fn q_k_and_v_fit_the_rows_of_in_proj_for_the_largest_image() {
        let v = vision();
        let b = bufs(&v);
        let end = b.v(&v) + (MAX_IMAGE_PATCHES * v.kv_row() * 4) as u64;
        assert!(end <= b.bcx + (MAX_IMAGE_PATCHES * 3 * v.hidden * 4) as u64);
        // The merged rows of the LayerNorm, linear_1 and linear_2 of the largest image fit h, g and y.
        let m = MAX_IMAGE_PATCHES / 4;
        assert!(m * v.merged() <= MAX_IMAGE_PATCHES * v.hidden);
        assert!(m * v.projector_hidden <= MAX_IMAGE_PATCHES * v.intermediate);
        assert!(m * v.out <= MAX_IMAGE_PATCHES * v.hidden);
    }
}
