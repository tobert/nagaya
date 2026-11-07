//! The audio encoder of `LFM3-Omni-2B` and its projector on the card (milestone 24 step 7, engine-abi "The audio
//! encoder of LFM3").
//!
//! The tasks follow `Lfm3OmniModel.get_audio_features` of `modeling_lfm3_omni.py` (`21d75123`) for one clip:
//!
//! - the downsampler (`Lfm3OmniDownsampler`, :374-409): three 3 x 3 convolutions with a stride of 2 over the plane of
//!   time and mel, each with silu after it or after its 1 x 1 convolution, then `lin_out` to the width of the encoder;
//! - the layers of `Lfm2Model` (transformers `modeling_lfm2.py`): RMSNorm, a gated short convolution or attention
//!   with a causal window of `sliding_window` keys (:420-440), the residual, then RMSNorm, the SwiGLU MLP and the
//!   residual; then `embedding_norm`;
//! - the projector (`Lfm3OmniProjector`, :498-510): LayerNorm, `linear_1`, GELU with erf, `linear_2`.
//!
//! The plane stays channel-last (`OP_SUBSAMPLE`), so the 1 x 1 convolutions are matmuls. The load puts the columns
//! of `lin_out` in that order, so its rows are the flatten of the reference, `[channel][mel]`.

use std::collections::BTreeSet;
use std::path::Path;

use nagaya_mk_bmg::protocol::{MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView};
use nagaya_mk_bmg::tasks;

use crate::weights::{Upload, rounded};

/// The most positions of one clip, and so the most rows of a span (engine-abi "The audio encoder of LFM3"). It is the
/// largest feed block.
pub const MAX_CLIP_POSITIONS: usize = 512;

/// The mel frames of one position: the three strides of 2 of the downsampler.
pub const FRAMES_PER_POSITION: usize = 8;

/// The most mel frames of one clip.
pub const MAX_CLIP_FRAMES: usize = MAX_CLIP_POSITIONS * FRAMES_PER_POSITION;

/// The prefix of the tensors of the encoder and of the projector.
const ENCODER: &str = "model.audio_encoder.";
const PROJECTOR: &str = "model.audio_projector.";

/// The operator of one layer of the encoder (`audio_encoder_config.text_config.layer_types`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioLayer {
    Conv,
    Attention,
}

/// The numbers of the audio encoder and its projector, from `config.json` of Omni.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoder {
    /// The mel bins of a frame (`num_mel_bins`).
    pub mel: usize,
    /// The channels of the plane of the downsampler (`conv_channels`).
    pub channels: usize,
    /// The width of the encoder (`hidden_size`).
    pub hidden: usize,
    pub layers: Vec<AudioLayer>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The rows of the gate and of the up part of the MLP, after `block_auto_adjust_ff_dim` (modeling_lfm2.py).
    pub intermediate: usize,
    /// The taps of each short convolution (`conv_L_cache`).
    pub taps: usize,
    /// An attention layer reads the keys `q - window + 1` to `q` (`sliding_window`).
    pub window: usize,
    /// The eps of the RMSNorms (`norm_eps`).
    pub eps: f32,
    /// The width of a row of the projector: the width of the backbone (`text_config.hidden_size`).
    pub out: usize,
    /// The eps of the LayerNorm of the projector: the default of `nn.LayerNorm`, because the projector names none.
    pub projector_eps: f32,
}

/// The theta of the rotation of the encoder, which `FLAG_ROPE_FULL` of the LFM3 binary turns with.
const THETA: f64 = 1e6;

impl Encoder {
    /// The numbers of the encoder from the text of `config.json`, or `None` for a model with no audio encoder. A value
    /// that the tasks cannot run is an error that names the key.
    pub fn parse(text: &str) -> Result<Option<Encoder>, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let a = &root["audio_encoder_config"];
        if a.is_null() {
            return Ok(None);
        }
        let c = &a["text_config"];
        let at = |key: &str| format!("audio_encoder_config.{key}");
        let value = |key: &str| -> &serde_json::Value {
            match key.strip_prefix("text_config.") {
                Some(inner) => &c[inner],
                None => &a[key],
            }
        };
        let count = |key: &str| -> Result<usize, String> {
            value(key)
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has {} {}, not a count", at(key), value(key)))
        };
        let want = |key: &str, ok: bool, need: &str| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(format!(
                    "config.json has {} {}, and the audio encoder needs {need}",
                    at(key),
                    value(key)
                ))
            }
        };
        want(
            "model_type",
            a["model_type"] == "lfm3_omni_encoder",
            "\"lfm3_omni_encoder\"",
        )?;
        want("text_config.model_type", c["model_type"] == "lfm2", "\"lfm2\"")?;
        let mel = count("num_mel_bins")?;
        let channels = count("conv_channels")?;
        let hidden = count("hidden_size")?;
        let window = count("sliding_window")?;
        want(
            "num_mel_bins",
            mel >= FRAMES_PER_POSITION && mel.is_multiple_of(FRAMES_PER_POSITION),
            "a multiple of 8, for three strides of 2",
        )?;
        want(
            "conv_channels",
            channels > 0 && channels.is_multiple_of(16),
            "a multiple of 16",
        )?;
        want(
            "hidden_size",
            hidden > 0 && hidden.is_multiple_of(16),
            "a multiple of 16",
        )?;
        want(
            "text_config.hidden_size",
            count("text_config.hidden_size")? == hidden,
            "the hidden_size of the encoder",
        )?;
        want("sliding_window", window > 0, "a window of at least 1 key")?;
        let heads = count("text_config.num_attention_heads")?;
        let kv_heads = count("text_config.num_key_value_heads")?;
        want(
            "text_config.num_attention_heads",
            heads > 0 && hidden.is_multiple_of(heads),
            "a divisor of hidden_size",
        )?;
        let head_dim = hidden / heads;
        // OP_ATTN_PV takes the dims of a head in pairs of 16, and the rotation takes them in halves.
        want(
            "text_config.num_attention_heads",
            head_dim.is_multiple_of(32),
            "heads of a multiple of 32 values",
        )?;
        want(
            "text_config.num_key_value_heads",
            kv_heads > 0 && heads.is_multiple_of(kv_heads),
            "a divisor of the head count",
        )?;
        let taps = count("text_config.conv_L_cache")?;
        want("text_config.conv_L_cache", (2..=8).contains(&taps), "2 to 8 taps")?;
        want("text_config.conv_bias", c["conv_bias"] == false, "false")?;
        let rope = &c["rope_parameters"];
        want(
            "text_config.rope_parameters",
            rope["rope_type"] == "default" && rope["rope_theta"].as_f64() == Some(THETA),
            "the default rope with theta 1e6",
        )?;
        let eps = c["norm_eps"].as_f64().ok_or_else(|| {
            format!(
                "config.json has {} {}, not a number",
                at("text_config.norm_eps"),
                c["norm_eps"]
            )
        })? as f32;
        // Lfm2MLP: int(2 * intermediate / 3), times the multiplier, up to a multiple of block_multiple_of.
        let mut intermediate = count("text_config.intermediate_size")?;
        want(
            "text_config.block_auto_adjust_ff_dim",
            c["block_auto_adjust_ff_dim"] == true,
            "true",
        )?;
        intermediate = 2 * intermediate / 3;
        let multiplier = c["block_ffn_dim_multiplier"].as_f64().ok_or_else(|| {
            format!(
                "config.json has {} {}, not a number",
                at("text_config.block_ffn_dim_multiplier"),
                c["block_ffn_dim_multiplier"]
            )
        })?;
        let multiple = count("text_config.block_multiple_of")?;
        want("text_config.block_multiple_of", multiple > 0, "at least 1")?;
        intermediate = ((multiplier * intermediate as f64) as usize).next_multiple_of(multiple);
        want(
            "text_config.intermediate_size",
            intermediate.is_multiple_of(16),
            "an MLP of a multiple of 16 rows",
        )?;
        let types = c["layer_types"].as_array().ok_or_else(|| {
            format!(
                "config.json has {} {}, not a list",
                at("text_config.layer_types"),
                c["layer_types"]
            )
        })?;
        let layers = types
            .iter()
            .map(|t| match t.as_str() {
                Some("conv") => Ok(AudioLayer::Conv),
                Some("full_attention") => Ok(AudioLayer::Attention),
                _ => Err(format!(
                    "config.json has the layer type {t} in {}; the audio encoder runs conv and full_attention",
                    at("text_config.layer_types")
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        want(
            "text_config.num_hidden_layers",
            count("text_config.num_hidden_layers")? == layers.len(),
            "one layer type for each layer",
        )?;
        let out = root["text_config"]["hidden_size"].as_u64().ok_or_else(|| {
            format!(
                "config.json has text_config.hidden_size {}, not a count",
                root["text_config"]["hidden_size"]
            )
        })? as usize;
        if out == 0 || !out.is_multiple_of(16) {
            return Err(format!(
                "config.json has text_config.hidden_size {out}, and the audio projector needs a multiple of 16"
            ));
        }
        Ok(Some(Encoder {
            mel,
            channels,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            taps,
            window,
            eps,
            out,
            projector_eps: 1e-5,
        }))
    }

    /// The numbers of the encoder from `config.json` of the folder `dir`.
    pub fn read(dir: &Path) -> Result<Option<Encoder>, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Encoder::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The values of one key row or one value row: the key and value heads.
    pub fn kv_row(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The mel rows of the plane after the three strides: the flatten of `lin_out` is these rows of each channel.
    pub fn mel_out(&self) -> usize {
        self.mel / FRAMES_PER_POSITION
    }
}

/// The tensors of the encoder and of the projector with their shapes, by their full names.
pub fn tensors(e: &Encoder) -> Vec<(String, Vec<usize>)> {
    let (c, h, i) = (e.channels, e.hidden, e.intermediate);
    let d = format!("{ENCODER}downsample.");
    let mut all: Vec<(String, Vec<usize>)> = vec![
        (format!("{d}conv_in.weight"), vec![c, 1, 3, 3]),
        (format!("{d}conv_in.bias"), vec![c]),
        (format!("{d}conv_mid.weight"), vec![c, 1, 3, 3]),
        (format!("{d}conv_mid.bias"), vec![c]),
        (format!("{d}lin_mid.weight"), vec![c, c, 1, 1]),
        (format!("{d}lin_mid.bias"), vec![c]),
        (format!("{d}conv_end.weight"), vec![c, 1, 3, 3]),
        (format!("{d}conv_end.bias"), vec![c]),
        (format!("{d}lin_end.weight"), vec![c, c, 1, 1]),
        (format!("{d}lin_end.bias"), vec![c]),
        (format!("{d}lin_out.weight"), vec![h, c * e.mel_out(), 1]),
        (format!("{d}lin_out.bias"), vec![h]),
        (format!("{ENCODER}lfm.embedding_norm.weight"), vec![h]),
    ];
    for (l, kind) in e.layers.iter().enumerate() {
        let mut layer = vec![
            ("operator_norm.weight", vec![h]),
            ("ffn_norm.weight", vec![h]),
            ("feed_forward.w1.weight", vec![i, h]),
            ("feed_forward.w3.weight", vec![i, h]),
            ("feed_forward.w2.weight", vec![h, i]),
        ];
        match kind {
            AudioLayer::Conv => layer.extend([
                ("conv.in_proj.weight", vec![3 * h, h]),
                ("conv.conv.weight", vec![h, 1, e.taps]),
                ("conv.out_proj.weight", vec![h, h]),
            ]),
            AudioLayer::Attention => layer.extend([
                ("self_attn.q_proj.weight", vec![e.heads * e.head_dim, h]),
                ("self_attn.k_proj.weight", vec![e.kv_row(), h]),
                ("self_attn.v_proj.weight", vec![e.kv_row(), h]),
                ("self_attn.out_proj.weight", vec![h, e.heads * e.head_dim]),
                ("self_attn.q_layernorm.weight", vec![e.head_dim]),
                ("self_attn.k_layernorm.weight", vec![e.head_dim]),
            ]),
        }
        all.extend(
            layer
                .into_iter()
                .map(|(n, s)| (format!("{ENCODER}lfm.layers.{l}.{n}"), s)),
        );
    }
    all.extend([
        (format!("{PROJECTOR}layer_norm.weight"), vec![h]),
        (format!("{PROJECTOR}layer_norm.bias"), vec![h]),
        (format!("{PROJECTOR}linear_1.weight"), vec![e.out, h]),
        (format!("{PROJECTOR}linear_1.bias"), vec![e.out]),
        (format!("{PROJECTOR}linear_2.weight"), vec![e.out, e.out]),
        (format!("{PROJECTOR}linear_2.bias"), vec![e.out]),
    ]);
    all
}

/// Check the tensors of the encoder and of the projector in `view`: each one is there as BF16 with its shape, and no
/// other tensor has their prefixes. The message names the tensor.
pub fn check(e: &Encoder, view: &SafeTensorsView<'_>) -> Result<(), String> {
    let want = tensors(e);
    for (name, shape) in &want {
        let meta = view
            .tensors
            .get(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        if meta.dtype != Dtype::BF16 || meta.shape != *shape {
            return Err(format!(
                "tensor {name} is {:?} {:?}, and the audio encoder needs BF16 {shape:?}",
                meta.dtype, meta.shape
            ));
        }
    }
    let known: BTreeSet<&String> = want.iter().map(|(n, _)| n).collect();
    let mut other: Vec<&String> = view
        .tensors
        .keys()
        .filter(|name| !known.contains(name) && (name.starts_with(ENCODER) || name.starts_with(PROJECTOR)))
        .collect();
    other.sort();
    if let Some(name) = other.first() {
        return Err(format!(
            "the checkpoint has the tensor {name}, which the audio encoder does not use; use the checkpoint of \
             LFM3-Omni-2B"
        ));
    }
    Ok(())
}

/// The weights of one layer of the encoder on the card. A field of the other operator is 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerWeights {
    pub operator_norm: u64,
    pub ffn_norm: u64,
    /// The gate rows (`w1`), then the up rows (`w3`): `[2 * intermediate][hidden]`.
    pub w13: u64,
    pub w2: u64,
    pub in_proj: u64,
    pub conv: u64,
    pub out_proj: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub q_norm: u64,
    pub k_norm: u64,
}

/// The weights of the encoder and of the projector on the card. Each is BF16, except the F32 LayerNorm of the
/// projector, which `OP_LAYERNORM` reads.
#[derive(Clone, Debug)]
pub struct AudioWeights {
    /// The taps `[channels][9]` and the bias of the three convolutions: `conv_in`, `conv_mid`, `conv_end`.
    pub conv: [(u64, u64); 3],
    /// The weight `[channels][channels]` and the bias of `lin_mid` and of `lin_end`.
    pub lin: [(u64, u64); 2],
    /// `lin_out`, `[hidden][mel_out * channels]` with the columns channel-last, and its bias.
    pub lin_out: (u64, u64),
    pub layers: Vec<LayerWeights>,
    /// `embedding_norm`.
    pub norm: u64,
    /// The F32 weight and bias of the LayerNorm of the projector.
    pub projector_norm: (u64, u64),
    pub linear_1: (u64, u64),
    pub linear_2: (u64, u64),
}

/// The columns of `lin_out` in the order of the channel-last plane: the column `c * mel_out + h` of the reference
/// becomes the column `h * channels + c`.
pub fn plane_columns(bytes: &[u8], rows: usize, channels: usize, mel_out: usize) -> Vec<u8> {
    let cols = channels * mel_out;
    assert_eq!(
        bytes.len(),
        rows * cols * 2,
        "lin_out holds {rows} rows of {cols} BF16 values"
    );
    let mut out = vec![0u8; bytes.len()];
    for r in 0..rows {
        for c in 0..channels {
            for h in 0..mel_out {
                let from = (r * cols + c * mel_out + h) * 2;
                let to = (r * cols + h * channels + c) * 2;
                out[to..to + 2].copy_from_slice(&bytes[from..from + 2]);
            }
        }
    }
    out
}

/// Allocate `len` bytes in `sink` and write the bytes that `data` makes, when the sink copies.
fn put(sink: &mut dyn Upload, len: usize, data: impl FnOnce() -> Vec<u8>) -> u64 {
    let at = sink.alloc(len);
    if sink.writes() {
        let data = data();
        assert_eq!(
            data.len(),
            len,
            "a weight of the audio encoder has {} bytes, not {len}",
            data.len()
        );
        sink.write(at, &data);
    }
    at
}

/// Load the encoder and the projector of `view` into `sink`. A checkpoint that fails `check` uploads nothing.
pub fn load(sink: &mut dyn Upload, e: &Encoder, view: &SafeTensorsView<'_>) -> Result<AudioWeights, String> {
    check(e, view)?;
    let bytes = |name: &str| -> &[u8] { view.tensor_bytes(name).expect("check found each tensor") };
    let plain = |sink: &mut dyn Upload, name: &str| -> u64 { put(sink, bytes(name).len(), || bytes(name).to_vec()) };
    let pair = |sink: &mut dyn Upload, base: &str| -> (u64, u64) {
        (
            plain(sink, &format!("{base}.weight")),
            plain(sink, &format!("{base}.bias")),
        )
    };
    // OP_LAYERNORM reads F32 weights, and each BF16 value is an f32 value exactly.
    let widened = |sink: &mut dyn Upload, name: &str| -> u64 {
        put(sink, bytes(name).len() * 2, || {
            bytes(name)
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16).to_le_bytes())
                .collect()
        })
    };
    let d = format!("{ENCODER}downsample.");
    let conv = [
        pair(sink, &format!("{d}conv_in")),
        pair(sink, &format!("{d}conv_mid")),
        pair(sink, &format!("{d}conv_end")),
    ];
    let lin = [pair(sink, &format!("{d}lin_mid")), pair(sink, &format!("{d}lin_end"))];
    let out_name = format!("{d}lin_out.weight");
    let lin_out_w = put(sink, bytes(&out_name).len(), || {
        plane_columns(bytes(&out_name), e.hidden, e.channels, e.mel_out())
    });
    let lin_out = (lin_out_w, plain(sink, &format!("{d}lin_out.bias")));
    let mut layers = Vec::with_capacity(e.layers.len());
    for (l, kind) in e.layers.iter().enumerate() {
        let n = |name: &str| format!("{ENCODER}lfm.layers.{l}.{name}");
        let (w1, w3) = (n("feed_forward.w1.weight"), n("feed_forward.w3.weight"));
        let mut layer = LayerWeights {
            operator_norm: plain(sink, &n("operator_norm.weight")),
            ffn_norm: plain(sink, &n("ffn_norm.weight")),
            w13: put(sink, bytes(&w1).len() + bytes(&w3).len(), || {
                [bytes(&w1), bytes(&w3)].concat()
            }),
            w2: plain(sink, &n("feed_forward.w2.weight")),
            ..LayerWeights::default()
        };
        match kind {
            AudioLayer::Conv => {
                layer.in_proj = plain(sink, &n("conv.in_proj.weight"));
                layer.conv = plain(sink, &n("conv.conv.weight"));
                layer.out_proj = plain(sink, &n("conv.out_proj.weight"));
            }
            AudioLayer::Attention => {
                layer.q = plain(sink, &n("self_attn.q_proj.weight"));
                layer.k = plain(sink, &n("self_attn.k_proj.weight"));
                layer.v = plain(sink, &n("self_attn.v_proj.weight"));
                layer.o = plain(sink, &n("self_attn.out_proj.weight"));
                layer.q_norm = plain(sink, &n("self_attn.q_layernorm.weight"));
                layer.k_norm = plain(sink, &n("self_attn.k_layernorm.weight"));
            }
        }
        layers.push(layer);
    }
    let norm = plain(sink, &format!("{ENCODER}lfm.embedding_norm.weight"));
    let projector_norm = (
        widened(sink, &format!("{PROJECTOR}layer_norm.weight")),
        widened(sink, &format!("{PROJECTOR}layer_norm.bias")),
    );
    let linear_1 = pair(sink, &format!("{PROJECTOR}linear_1"));
    let linear_2 = pair(sink, &format!("{PROJECTOR}linear_2"));
    Ok(AudioWeights {
        conv,
        lin,
        lin_out,
        layers,
        norm,
        projector_norm,
        linear_1,
        linear_2,
    })
}

/// The shape of the plane after each stride of 2 for a clip of `frames` frames: (time rows, mel rows) of the input,
/// then of each of the three outputs. The last time rows are the positions.
pub fn plane_shapes(e: &Encoder, frames: usize) -> [(usize, usize); 4] {
    let mut shapes = [(frames, e.mel); 4];
    for s in 1..4 {
        let (t, h) = tasks::subsample_out(shapes[s - 1].0 as u32, shapes[s - 1].1 as u32);
        shapes[s] = (t as usize, h as usize);
    }
    shapes
}

/// The positions of a clip of `frames` frames: `ceil(frames / 8)` (modeling_lfm3_omni.py:407-409).
pub fn positions(frames: usize) -> usize {
    frames.div_ceil(FRAMES_PER_POSITION)
}

/// The device buffers of the encoder for a clip of at most `positions` positions. Each row is one time step.
#[derive(Clone, Copy, Debug)]
pub struct AudioBufs {
    /// The most positions of a clip.
    pub positions: usize,
    /// The mel rows of a clip, `[frames][mel]`, for a caller that has no buffer of its own.
    pub mel: u64,
    /// The plane after `conv_in`, and after `lin_mid` and `lin_end`: `[frames / 2][mel / 2][channels]`.
    pub plane_a: u64,
    /// The plane after `conv_mid` and `conv_end`: `[frames / 4][mel / 4][channels]`.
    pub plane_b: u64,
    /// The residual stream: `[positions][hidden]`. It holds the rows of `lin_out` before the layers.
    pub x: u64,
    /// The normed rows: `[positions][hidden]`. It holds `embedding_norm` after the layers.
    pub h: u64,
    /// The rows B, C and x of a conv layer: `[positions][3 * hidden]`.
    pub bcx: u64,
    /// The output of an operator before its projection: `[positions][hidden]`.
    pub y: u64,
    /// The queries: `[positions][hidden]`.
    pub q: u64,
    /// The keys, then the values after `positions` rows: `[2 * positions][kv_row]`.
    pub kv: u64,
    /// The scores of the attention: `heads` rows for each position.
    pub scores: u64,
    /// The rows of the MLP, and then of `linear_1`: `[positions][max(intermediate, out)]`.
    pub g: u64,
    /// The state of a short convolution: `[taps - 1][hidden]`.
    pub state: u64,
    /// Zeros of the size of `state`, which each conv layer copies into it first. Nothing writes them.
    pub zeros: u64,
    /// The rows of a clip for the backbone, `[positions][out]`. A pass encodes a clip into it, and then copies the
    /// rows of its block into the media rows of the tower.
    pub rows: u64,
}

impl AudioBufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, e: &Encoder, positions: usize) -> Self {
        assert!(positions >= 1, "the encoder needs room for at least one position");
        let frames = positions * FRAMES_PER_POSITION;
        let [_, a, b, _] = plane_shapes(e, frames);
        let f = |values: usize| values * positions * 4;
        Self {
            positions,
            mel: z(frames * e.mel * 4),
            plane_a: z(a.0 * a.1 * e.channels * 4),
            plane_b: z(b.0 * b.1 * e.channels * 4),
            x: z(f(e.hidden)),
            h: z(f(e.hidden)),
            bcx: z(f(3 * e.hidden)),
            y: z(f(e.hidden)),
            q: z(f(e.heads * e.head_dim)),
            kv: z(2 * f(e.kv_row())),
            scores: z(tasks::score_floats(e.heads as u32, positions as u32, 0) * 4),
            g: z(f(e.intermediate.max(e.out))),
            state: z((e.taps - 1) * e.hidden * 4),
            zeros: z((e.taps - 1) * e.hidden * 4),
            rows: z(f(e.out)),
        }
    }
}

/// The device bytes of the buffers, with no card: the plan (spec 2.4).
pub fn bufs_bytes(e: &Encoder, positions: usize) -> usize {
    let mut total = 0usize;
    AudioBufs::new(
        &mut |bytes| {
            total += rounded(bytes);
            0
        },
        e,
        positions,
    );
    total
}

/// Where the tasks of one clip stop. A probe of the oracle stops early.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    /// After `lin_out`: the rows are in `AudioBufs::x`, `[positions][hidden]`.
    Downsample,
    /// After the layers and `embedding_norm`: the rows are in `AudioBufs::h`, `[positions][hidden]`.
    Encoder,
    /// After the projector: the rows are at `Clip::out`, `[positions][out]`.
    Rows,
}

/// One clip on the card: the address of its mel rows `[frames][mel]`, its count of frames, and the address that gets
/// its rows `[ceil(frames / 8)][out]`.
#[derive(Clone, Copy, Debug)]
pub struct Clip {
    pub mel: u64,
    pub frames: usize,
    pub out: u64,
}

/// The tasks of one clip, up to `upto`. They read the mel rows of `clip` and the weights, they use `bufs`, and they
/// write the rows of the clip to `clip.out`. The tasks need no state from an earlier clip.
pub fn encode(e: &Encoder, w: &AudioWeights, bufs: &AudioBufs, clip: Clip, upto: Stage) -> Vec<MkTaskDescriptor> {
    let Clip { mel, frames, out } = clip;
    let p = positions(frames);
    assert!(
        frames >= 1 && p <= bufs.positions,
        "a clip of {frames} frames takes {p} positions, and the buffers hold 1 to {} positions",
        bufs.positions
    );
    let shapes = plane_shapes(e, frames);
    assert_eq!(shapes[3], (p, e.mel_out()), "the plane after three strides");
    let (c, hd) = (e.channels as u32, e.hidden as u32);
    let u = |v: usize| v as u32;
    let mut t = Vec::new();
    // The downsampler: conv_in and silu; conv_mid, lin_mid and silu; conv_end, lin_end and silu; lin_out.
    let (t0, h0) = shapes[0];
    t.push(tasks::subsample(
        mel,
        w.conv[0].0,
        w.conv[0].1,
        bufs.plane_a,
        u(t0),
        u(h0),
        1,
        c,
        true,
    ));
    for s in 1..3 {
        let (ti, hi) = shapes[s];
        let (to, ho) = shapes[s + 1];
        let rows = u(to * ho);
        t.push(tasks::subsample(
            bufs.plane_a,
            w.conv[s].0,
            w.conv[s].1,
            bufs.plane_b,
            u(ti),
            u(hi),
            c,
            c,
            false,
        ));
        t.push(tasks::matmul(
            bufs.plane_b,
            w.lin[s - 1].0,
            bufs.plane_a,
            rows,
            c,
            c,
            false,
        ));
        t.push(tasks::bias(w.lin[s - 1].1, bufs.plane_a, rows, c, false));
        t.push(tasks::silu_mul(bufs.plane_a, 0, bufs.plane_a, rows, c));
    }
    let flat = c * u(e.mel_out());
    t.push(tasks::matmul(bufs.plane_a, w.lin_out.0, bufs.x, u(p), hd, flat, false));
    t.push(tasks::bias(w.lin_out.1, bufs.x, u(p), hd, false));
    if upto == Stage::Downsample {
        return t;
    }
    let (heads, kv, d) = (u(e.heads), u(e.kv_heads), u(e.head_dim));
    let n = u(p);
    let values = bufs.kv + (bufs.positions * e.kv_row() * 4) as u64;
    for (l, lw) in w.layers.iter().enumerate() {
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n, hd, e.eps, false));
        match e.layers[l] {
            AudioLayer::Conv => {
                t.push(tasks::matmul(bufs.h, lw.in_proj, bufs.bcx, n, 3 * hd, hd, false));
                // Each clip starts from a state of zeros, as the reference pads the start.
                t.push(tasks::copy(bufs.zeros, bufs.state, u((e.taps - 1) * e.hidden)));
                t.push(tasks::short_conv(
                    bufs.bcx,
                    lw.conv,
                    bufs.state,
                    bufs.y,
                    n,
                    hd,
                    u(e.taps),
                    false,
                ));
                t.push(tasks::matmul(bufs.y, lw.out_proj, bufs.x, n, hd, hd, true));
            }
            AudioLayer::Attention => {
                t.push(tasks::matmul(bufs.h, lw.q, bufs.q, n, heads * d, hd, false));
                t.push(tasks::matmul(bufs.h, lw.k, bufs.kv, n, kv * d, hd, false));
                t.push(tasks::matmul(bufs.h, lw.v, values, n, kv * d, hd, false));
                // The queries carry the scale head_dim^-0.5, because the attention ops do not scale.
                let scale = (e.head_dim as f32).powf(-0.5);
                t.push(tasks::qk_rope(
                    bufs.q,
                    lw.q_norm,
                    n * heads,
                    heads,
                    d,
                    0,
                    true,
                    true,
                    e.eps,
                    scale,
                ));
                t.push(tasks::qk_rope(
                    bufs.kv,
                    lw.k_norm,
                    n * kv,
                    kv,
                    d,
                    0,
                    true,
                    true,
                    e.eps,
                    1.0,
                ));
                t.extend(tasks::attend_block(
                    bufs.q,
                    bufs.kv,
                    bufs.scores,
                    bufs.y,
                    n,
                    heads,
                    kv,
                    d,
                    0,
                    u(bufs.positions),
                    u(e.window),
                ));
                t.push(tasks::matmul(bufs.y, lw.o, bufs.x, n, hd, heads * d, true));
            }
        }
        t.push(tasks::rmsnorm(bufs.x, lw.ffn_norm, bufs.h, n, hd, e.eps, false));
        t.push(tasks::swiglu(bufs.h, lw.w13, bufs.g, n, u(e.intermediate), hd));
        t.push(tasks::matmul(bufs.g, lw.w2, bufs.x, n, hd, u(e.intermediate), true));
    }
    t.push(tasks::rmsnorm(bufs.x, w.norm, bufs.h, n, hd, e.eps, false));
    if upto == Stage::Encoder {
        return t;
    }
    let o = u(e.out);
    t.push(tasks::layernorm(
        bufs.h,
        w.projector_norm.0,
        w.projector_norm.1,
        bufs.y,
        n,
        hd,
        e.projector_eps,
    ));
    t.push(tasks::matmul(bufs.y, w.linear_1.0, bufs.g, n, o, hd, false));
    t.push(tasks::bias(w.linear_1.1, bufs.g, n, o, false));
    t.push(tasks::gelu(bufs.g, bufs.g, n, o));
    t.push(tasks::matmul(bufs.g, w.linear_2.0, out, n, o, o, false));
    t.push(tasks::bias(w.linear_2.1, out, n, o, false));
    assert!(t.len() < MAX_TASKS_PER_CARD, "the encoder has {} tasks", t.len());
    t
}

#[cfg(test)]
mod tests;
