//! The numbers of the LFM3 backbone, the config of the engine, its limits, and the rules of its slots. None of this
//! needs a card, so the host tests cover it (`docs/new-engine.md` section 2).
//!
//! The backbone numbers come from `config.json` (its `text_config` for Omni). `Backbone::parse` reads them and refuses a
//! value that the kernels of the LFM3 binary cannot run. The plan reads `config.json` and the tensor headers, and no
//! weight bytes (spec 2.4).

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use nagaya_abi::{
    CONTROL_TERM, Capability, ChainLimits, ControlLimit, CoverageSpan, Determinism, Dtype, FeedRule, Inputs, Kind,
    Lifetime, Limits, MandatoryTerm, MemoryLimits, PartId, PartKind, PartLimits, Segment, SegmentInputs, SlotLimit,
    State, TapLimit, Tier,
};
use nagaya_mk_bmg::protocol::{MAX_TASKS_PER_CARD, NUM_SCOREBOARD_EVENTS};
use nagaya_mk_bmg::safetensors::MappedSafeTensors;

use crate::audio::{self, Encoder, MAX_CLIP_POSITIONS};
use crate::media::{self, MediaRules, SpanUse};
use crate::model::{bufs_bytes, scratch_floats};
use crate::vision::{self, Vision};
use crate::weights::{Count, STAGING_BYTES, load};

/// The engine kind in a placement profile (`AGENTS.md` "Layout": kind, model, target).
pub const KIND: &str = "mk-lfm3-bmg";

/// The family library that the engine needs (spec 11.3).
pub const FAMILY: &str = "lfm3";

/// The file of the weights in the model folder.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The feed block when the config names none, as for Gemma 4 (engine-abi D11).
pub const DEFAULT_MAX_BLOCK: usize = 512;

/// The largest feed block. The spill of a ring holds one block (`RING_SPILL`).
pub const MAX_BLOCK: usize = 512;

/// The positions of each slot: the context of Omni's model card (spec 11.11, "The served LFM3 models"). Every slot has
/// this one size, the simplest plan. The attention ops read no slot size, so a second size would give the same bits.
pub const SLOT_TOKENS: usize = 32_768;

/// The longest quantum in milliseconds (D8). It is a budget, not a measurement: the card test
/// `the_longest_quantum_fits_its_bound` measures the last block of a full slot.
pub const MAX_QUANTUM_MS: u32 = 2_000;

/// The most segments in a pass. Several segments are a batch of decode rows, each on its own slot, and the tiled
/// path keeps the bits of a row for 1 to 8 rows (engine-abi "LFM3", "Batched passes").
pub const MAX_SEGMENTS: u32 = 8;

/// The most logit rows of one pass.
pub const MAX_LOGIT_ROWS: usize = 16;

/// The vectors of the control table (engine-abi "Control vectors"). A vector of 24 layers of 2,048 values is 196,608
/// bytes, so the table is 1.5 MiB.
pub const CONTROL_TABLES: u32 = 8;

/// The tap that gives the residual after each layer at each logit row (spec 13.1).
pub const RESIDUAL_TAP: &str = "residual";

/// The tap that gives the logit lens of each layer at the one logit row of a pass (spec 13.1): the final norm and the
/// head of the row of each layer.
pub const LENS_TAP: &str = "lens";

/// A sliding layer of a slot longer than this keeps its keys and values in a ring of this many rows. The row of a key
/// is its position modulo the ring. The ring holds a block and the window before it, in rows of 32.
pub const RING_ROWS: usize = 2048;

/// The rows past a ring that a block may write before `OP_COPY` moves them to the start: the kernel's `KV_RING_SPILL`.
pub const RING_SPILL: usize = 1024;

const _: () = assert!(
    RING_ROWS.is_multiple_of(32) && RING_SPILL >= MAX_BLOCK && MAX_BLOCK <= SLOT_TOKENS && SLOT_TOKENS > RING_ROWS
);

/// The two LFM3 models. An engine serves under its engine name, and the identity names the model (spec 11.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// `LFM3-1.7B`, with the model name `lfm3`.
    Text,
    /// The language model of `LFM3-Omni-2B`, with the model name `lfm3-omni`.
    Omni,
}

impl Model {
    /// The model name of the identity. The family checks its folder against it.
    pub fn name(self) -> &'static str {
        match self {
            Model::Text => "lfm3",
            Model::Omni => "lfm3-omni",
        }
    }

    /// The prefix of the tensor names of the backbone.
    pub fn prefix(self) -> &'static str {
        match self {
            Model::Text => "model.",
            Model::Omni => "model.language_model.",
        }
    }
}

/// The operator of one layer of the backbone (`layer_types` of `config.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// A gated short convolution.
    Conv,
    /// Attention to every key.
    Full,
    /// Attention to the keys of the window.
    Sliding,
}

/// The rotation of one kind of attention layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// The theta of `rope_parameters`.
    pub theta: f64,
    /// The pairs that turn: `int(partial_rotary_factor * head_dim) // 2` (modeling_lfm3.py:108).
    pub pairs: usize,
}

/// The numbers of the backbone, from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Backbone {
    pub model: Model,
    pub hidden: usize,
    pub layers: Vec<LayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    /// The rows of the embedding table, which is also the head (`tie_word_embeddings`).
    pub vocab_rows: usize,
    /// A sliding layer reads the keys `q - window + 1` to `q`.
    pub window: usize,
    /// The taps of each short convolution (`conv_L_cache`). The state holds `taps - 1` inputs.
    pub taps: usize,
    pub eps: f32,
    pub full_rope: Rope,
    pub sliding_rope: Rope,
    /// The audio encoder of Omni, which the engine loads beside the backbone. The text model has none.
    pub audio: Option<Encoder>,
}

/// The theta that the LFM3 binary uses for the full layers (`FLAG_ROPE_FULL`) and for the sliding layers.
const FULL_THETA: f64 = 1e6;
const SLIDING_THETA: f64 = 1e4;

/// The head size and the head count that the attention ops of the binary take. The one-token attention keeps the
/// chunks of 16 heads in its scratch (`attn_pv_segments`, megakernel.cl).
pub(crate) const HEAD_DIM: usize = 128;
const HEADS: usize = 16;

impl Backbone {
    /// Read the numbers from the text of `config.json`. A value that the kernels cannot run is an error that names the
    /// key.
    pub fn parse(text: &str) -> Result<Backbone, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let model = match root["architectures"][0].as_str() {
            Some("Lfm3ForCausalLM") => Model::Text,
            Some("Lfm3OmniForConditionalGeneration") => Model::Omni,
            other => {
                return Err(format!(
                    "config.json has architectures {other:?}; this engine runs Lfm3ForCausalLM and \
                     Lfm3OmniForConditionalGeneration"
                ));
            }
        };
        if model == Model::Text
            && let Some(key) = ["vision_config", "audio_encoder_config", "depth_config"]
                .into_iter()
                .find(|key| !root[*key].is_null())
        {
            return Err(format!(
                "config.json of Lfm3ForCausalLM has {key}; the text model has no vision tower, audio encoder or depth \
                 decoder (engine-abi \"LFM3\")"
            ));
        }
        let c = match model {
            Model::Text => &root,
            Model::Omni => &root["text_config"],
        };
        let at = match model {
            Model::Text => "",
            Model::Omni => "text_config.",
        };
        let count = |key: &str| -> Result<usize, String> {
            c[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has {at}{key} {}, not a count", c[key]))
        };
        let flag = |key: &str| -> Result<bool, String> {
            c[key]
                .as_bool()
                .ok_or_else(|| format!("config.json has {at}{key} {}, not true or false", c[key]))
        };
        let want = |key: &str, ok: bool, need: &str| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(format!(
                    "config.json has {at}{key} {}, and this engine needs {need}",
                    c[key]
                ))
            }
        };
        if c["model_type"] != "lfm3" {
            return Err(format!(
                "config.json has {at}model_type {}, not \"lfm3\"",
                c["model_type"]
            ));
        }
        let hidden = count("hidden_size")?;
        let heads = count("num_attention_heads")?;
        let kv_heads = count("num_key_value_heads")?;
        let head_dim = count("head_dim")?;
        let intermediate = count("intermediate_size")?;
        let vocab_rows = count("vocab_size")?;
        let window = count("sliding_window")?;
        let taps = count("conv_L_cache")?;
        let n_layers = count("num_hidden_layers")?;
        want(
            "hidden_size",
            hidden > 0 && hidden.is_multiple_of(16),
            "a multiple of 16",
        )?;
        want("num_attention_heads", heads == HEADS, "16 heads")?;
        want("head_dim", head_dim == HEAD_DIM, "128")?;
        want(
            "num_key_value_heads",
            kv_heads > 0 && heads.is_multiple_of(kv_heads),
            "a divisor of the head count",
        )?;
        want("hidden_size", heads * head_dim == hidden, "heads × head_dim")?;
        want(
            "intermediate_size",
            intermediate > 0 && intermediate.is_multiple_of(16),
            "a multiple of 16",
        )?;
        want(
            "vocab_size",
            vocab_rows > 0 && vocab_rows.is_multiple_of(16),
            "a multiple of 16",
        )?;
        want("conv_L_cache", (2..=8).contains(&taps), "2 to 8 taps")?;
        want(
            "sliding_window",
            window > 0 && window + MAX_BLOCK + 32 <= RING_ROWS,
            &format!("a window of at most {} positions", RING_ROWS - MAX_BLOCK - 32),
        )?;
        want("conv_bias", !flag("conv_bias")?, "false")?;
        want("use_sandwich_norm", flag("use_sandwich_norm")?, "true")?;
        let tied = flag("tie_word_embeddings")? && (model == Model::Text || root["tie_word_embeddings"] != false);
        want("tie_word_embeddings", tied, "true: the head is the embedding table")?;
        let eps = c["norm_eps"]
            .as_f64()
            .ok_or_else(|| format!("config.json has {at}norm_eps {}, not a number", c["norm_eps"]))?;
        let types = c["layer_types"]
            .as_array()
            .ok_or_else(|| format!("config.json has {at}layer_types {}, not a list", c["layer_types"]))?;
        let layers = types
            .iter()
            .map(|t| match t.as_str() {
                Some("conv") => Ok(LayerKind::Conv),
                Some("full_attention") => Ok(LayerKind::Full),
                Some("sliding_attention") => Ok(LayerKind::Sliding),
                _ => Err(format!(
                    "config.json has the layer type {t} in {at}layer_types; the engine runs conv, full_attention and \
                     sliding_attention"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.len() != n_layers {
            return Err(format!(
                "config.json has {} layer types and {at}num_hidden_layers {n_layers}",
                layers.len()
            ));
        }
        let rope = |kind: &str, theta: f64| -> Result<Rope, String> {
            let r = &c["rope_parameters"][kind];
            let factor = match &r["partial_rotary_factor"] {
                serde_json::Value::Null => 1.0,
                v => v
                    .as_f64()
                    .ok_or_else(|| format!("config.json has {at}rope_parameters.{kind}.partial_rotary_factor {v}"))?,
            };
            if r["rope_type"] != "default" || r["rope_theta"].as_f64() != Some(theta) {
                return Err(format!(
                    "config.json has {at}rope_parameters.{kind} {r}; the LFM3 binary turns {kind} with the default \
                     rope and theta {theta}"
                ));
            }
            let pairs = (factor * head_dim as f64) as usize / 2;
            if !(1..=head_dim / 2).contains(&pairs) {
                return Err(format!(
                    "config.json has {at}rope_parameters.{kind} {r}, which turns {pairs} pairs of a head of {head_dim}"
                ));
            }
            Ok(Rope { theta, pairs })
        };
        Ok(Backbone {
            model,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            vocab_rows,
            window,
            taps,
            eps: eps as f32,
            full_rope: rope("full_attention", FULL_THETA)?,
            sliding_rope: rope("sliding_attention", SLIDING_THETA)?,
            audio: Encoder::parse(text)?,
        })
    }

    /// Read the numbers from `config.json` of the folder `dir`.
    pub fn read(dir: &Path) -> Result<Backbone, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Backbone::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The f32 values of one key row or one value row of an attention layer: the key and value heads.
    pub fn kv_row(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The f32 values of the conv state of one layer: `taps - 1` inputs of each channel.
    pub fn conv_state(&self) -> usize {
        (self.taps - 1) * self.hidden
    }

    /// The layers of one kind.
    pub fn layers_of(&self, kind: LayerKind) -> impl Iterator<Item = usize> + '_ {
        self.layers
            .iter()
            .enumerate()
            .filter(move |(_, k)| **k == kind)
            .map(|(l, _)| l)
    }
}

/// The ring rows of a sliding layer in a slot of `seq` positions, or 0 when its rows are positions.
pub fn ring_of(seq: usize) -> usize {
    if seq > RING_ROWS { RING_ROWS } else { 0 }
}

/// The rows of each key plane and each value plane of a layer of `kind` in a slot of `seq` positions.
pub fn plane_rows(seq: usize, kind: LayerKind) -> usize {
    match (kind, ring_of(seq)) {
        (LayerKind::Sliding, ring) if ring > 0 => ring + RING_SPILL,
        _ => seq,
    }
}

/// True when a layer of `kind` keeps a ring in a slot of `seq` positions.
pub fn ringed(seq: usize, kind: LayerKind) -> bool {
    kind == LayerKind::Sliding && ring_of(seq) > 0
}

/// One part of the held state (spec 11.11, "The held state of LFM3").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The conv state of a conv layer: a fixed part.
    Conv(usize),
    /// The keys of an attention layer.
    K(usize),
    /// The values of an attention layer.
    V(usize),
}

impl Part {
    /// The layer of the part.
    pub fn layer(self) -> usize {
        match self {
            Part::Conv(l) | Part::K(l) | Part::V(l) => l,
        }
    }
}

/// The parts in layer order: `conv.<ll>` for a conv layer, and `k.<ll>` then `v.<ll>` for an attention layer. The part
/// id is the index in this list.
pub fn part_list(b: &Backbone) -> Vec<Part> {
    b.layers
        .iter()
        .enumerate()
        .flat_map(|(l, kind)| match kind {
            LayerKind::Conv => vec![Part::Conv(l)],
            LayerKind::Full | LayerKind::Sliding => vec![Part::K(l), Part::V(l)],
        })
        .collect()
}

/// The parts of a state, as the ABI declares them. A full layer holds cells; a sliding layer holds a ring; a conv layer
/// holds a fixed part of `taps - 1` rows of `hidden` values, oldest first.
pub fn parts(b: &Backbone) -> Vec<PartLimits> {
    part_list(b)
        .into_iter()
        .map(|part| {
            let l = part.layer();
            let (name, kind, shape) = match part {
                Part::Conv(_) => (
                    format!("conv.{l:02}"),
                    PartKind::Fixed,
                    vec![(b.taps - 1) as u64, b.hidden as u64],
                ),
                Part::K(_) | Part::V(_) => (
                    format!("{}.{l:02}", if matches!(part, Part::K(_)) { "k" } else { "v" }),
                    if b.layers[l] == LayerKind::Full {
                        PartKind::Cells
                    } else {
                        PartKind::Ring
                    },
                    vec![b.kv_heads as u64, b.head_dim as u64],
                ),
            };
            PartLimits {
                name,
                kind,
                dtype: Dtype::F32,
                shape,
                block_format: None,
                stage: 0,
            }
        })
        .collect()
}

/// The device bytes of one slot of `seq` positions: the key and value planes of each attention layer, and the conv
/// state of each conv layer.
pub fn slot_bytes(b: &Backbone, seq: usize) -> u64 {
    b.layers
        .iter()
        .map(|kind| match kind {
            LayerKind::Conv => crate::weights::rounded(b.conv_state() * 4),
            _ => crate::weights::rounded(2 * plane_rows(seq, *kind) * b.kv_row() * 4),
        } as u64)
        .sum()
}

/// The config of one engine. The factory of the process reads it from the profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The model folder: `config.json`, `model.safetensors`, `tokenizer.json` and `chat_template.jinja`.
    pub model_dir: PathBuf,
    /// The count of slots. Each slot holds `SLOT_TOKENS` positions.
    pub slots: usize,
    /// The tokens of one block, and the feed block (engine-abi D11).
    pub max_block: usize,
}

impl Config {
    /// Check the sizes. The message names the value and the rule.
    pub fn check(&self) -> Result<(), String> {
        if self.max_block == 0 || self.max_block > MAX_BLOCK {
            return Err(format!(
                "max_block is {}; it must be 1 to {MAX_BLOCK} tokens",
                self.max_block
            ));
        }
        if self.slots == 0 {
            return Err("the config names no slot; give `slots`, the count of slots".to_owned());
        }
        Ok(())
    }

    /// The positions of each slot, in slot order.
    pub fn seqs(&self) -> Vec<usize> {
        vec![SLOT_TOKENS; self.slots]
    }
}

/// The slot classes of the config: each run of equal sizes is one class.
pub fn slot_limits(b: &Backbone, slots: &[usize]) -> Vec<SlotLimit> {
    let mut classes: Vec<SlotLimit> = Vec::new();
    for &seq in slots {
        match classes.last_mut() {
            Some(class) if class.tokens == seq as u64 => class.count += 1,
            _ => classes.push(SlotLimit {
                bytes: slot_bytes(b, seq),
                count: 1,
                tokens: seq as u64,
            }),
        }
    }
    classes
}

fn term(name: &str, tier: Tier, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime: Lifetime::Engine,
        kinds: vec![Kind::Prefill, Kind::Decode, Kind::Read, Kind::Copy],
    }
}

/// The device bytes of the ring of one card: the tasks are host memory, and the scoreboard and the scratch are device
/// memory. `L0Cluster::init_at` also takes the wake buffer and the device copy of the tasks.
fn ring_device_bytes(max_seq: usize) -> u64 {
    (NUM_SCOREBOARD_EVENTS * 4 + scratch_floats(max_seq) * 4 + 128 + MAX_TASKS_PER_CARD * 64) as u64
}

/// The host bytes of the ring and of the copies: the tasks, the statistics and the staging buffer.
pub fn pinned_bytes() -> u64 {
    (MAX_TASKS_PER_CARD * 64 + 32 * 64 + STAGING_BYTES) as u64
}

/// The device bytes of the vision tower of Omni: its weights and the buffers of its largest image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisionBytes {
    pub weights: u64,
    pub work: u64,
}

/// The limits for a config, with the device bytes of the weights of the backbone, and of the vision tower for Omni.
/// Omni declares tokens and rows: a segment of tokens alone is a text block, and the rows are its media (engine-abi
/// "The LFM3 vision binding").
pub fn limits(b: &Backbone, config: &Config, weights: u64, vision: Option<VisionBytes>) -> Limits {
    let kv = config.slots as u64 * slot_bytes(b, SLOT_TOKENS);
    let work = bufs_bytes(b, config.max_block, SLOT_TOKENS) as u64
        + ring_device_bytes(SLOT_TOKENS)
        + audio_bufs_bytes(b) as u64
        + vision.map_or(0, |v| v.work);
    let mut mandatory = vec![
        term("weights", Tier::Device(0), weights),
        term("kv", Tier::Device(0), kv),
        term("work", Tier::Device(0), work),
        term("pinned", Tier::Host, pinned_bytes()),
    ];
    if let Some(v) = vision {
        mandatory.insert(1, term("vision", Tier::Device(0), v.weights));
    }
    // A drop frees the weights, the slots, the buffers and the card's context.
    let mut capabilities = vec![
        Capability::Parts,
        Capability::Slots,
        Capability::Coverage,
        Capability::Close,
    ];
    let control = control_limit(b);
    if let Some(control) = control {
        capabilities.push(Capability::Control);
        mandatory.push(term(CONTROL_TERM, Tier::Device(0), control.table_bytes()));
    }
    Limits {
        capabilities,
        parts: parts(b),
        inputs: if vision.is_some() {
            Inputs::TokensAndRows
        } else {
            Inputs::Tokens
        },
        devices: vec![0],
        stages: Vec::new(),
        // A batch has one row for each segment, and the buffers hold `max_block` rows.
        max_segments: MAX_SEGMENTS.min(config.max_block as u32),
        max_pass_inputs: config.max_block as u32,
        max_logit_rows: MAX_LOGIT_ROWS as u32,
        slots: slot_limits(b, &config.seqs()),
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory,
            elastic: Vec::new(),
        },
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        // The matmuls of the interpreter choose their reduction by the token count of a pass (engine-abi D11).
        segments_exact: false,
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        taps: vec![
            TapLimit {
                name: RESIDUAL_TAP.to_owned(),
                conditions: Vec::new(),
            },
            TapLimit {
                name: LENS_TAP.to_owned(),
                conditions: vec!["one logit row".to_owned()],
            },
        ],
        checkpoint_grain: None,
        residual_row_bytes: 0,
        control,
        feed: Some(FeedRule {
            block: config.max_block as u64,
        }),
    }
}

/// The control table of a backbone. Both models have one. Omni's vector acts on its text passes only: a frame of its
/// speech is a `frame` call, which takes no control list (spec 11.12, AGENTS.md milestone 25 step 9).
pub fn control_limit(b: &Backbone) -> Option<ControlLimit> {
    Some(ControlLimit {
        tables: CONTROL_TABLES,
        layers: b.layers.len() as u32,
        width: b.hidden as u32,
    })
}

/// The device bytes of the buffers of the audio encoder for the longest clip, or 0 for a model with no encoder.
pub fn audio_bufs_bytes(b: &Backbone) -> usize {
    b.audio.as_ref().map_or(0, |e| audio::bufs_bytes(e, MAX_CLIP_POSITIONS))
}

/// The device bytes of the weights that the engine loads from the folder `dir`: the backbone, and the audio encoder
/// of Omni. They come from `config.json` and the tensor headers.
pub fn weights_bytes(b: &Backbone, dir: &Path) -> Result<u64, String> {
    let map = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
    let view = map.view()?;
    let mut count = Count::default();
    load(&mut count, b, &view)?;
    if let Some(e) = &b.audio {
        audio::load(&mut count, e, &view)?;
    }
    Ok(count.bytes as u64)
}

/// The vision tower of the model in the folder `dir`, from `config.json`: `None` for `LFM3-1.7B`.
pub fn read_vision(b: &Backbone, dir: &Path) -> Result<Option<Vision>, String> {
    if b.model == Model::Text {
        return Ok(None);
    }
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Vision::parse(&text, b.hidden)
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// The device bytes of the vision tower in the folder `dir`, from the tensor headers, with its buffers for blocks of
/// `max_block`.
pub fn vision_bytes(v: &Vision, dir: &Path, max_block: usize) -> Result<VisionBytes, String> {
    let map = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
    let mut count = Count::default();
    vision::load(&mut count, v, &map.view()?)?;
    Ok(VisionBytes {
        weights: count.bytes as u64,
        work: vision::bufs_bytes(v, max_block) as u64,
    })
}

/// The limits of a config, from `config.json` and the tensor headers. It reads no weight bytes (spec 2.4).
pub fn plan(config: &Config) -> Result<Limits, String> {
    config.check()?;
    let b = Backbone::read(&config.model_dir)?;
    let weights = weights_bytes(&b, &config.model_dir)?;
    let vision = match read_vision(&b, &config.model_dir)? {
        Some(v) => Some(vision_bytes(&v, &config.model_dir, config.max_block)?),
        None => None,
    };
    let mut limits = limits(&b, config, weights, vision);
    // Omni adds the term `speech` (engine-abi "The speech calls of LFM3").
    if b.model == Model::Omni {
        let bytes = crate::frames::plan_bytes(&config.model_dir, config.slots)?;
        crate::frames::declare(&mut limits, bytes);
    }
    Ok(limits)
}

/// One block that a pass runs, after its checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub slot: usize,
    /// The position of the first token. It is the token count of the slot before the pass.
    pub pos0: usize,
    pub tokens: Vec<u32>,
    /// The rows of the block whose logits the pass gives, in request order.
    pub logit_rows: Vec<usize>,
    /// The media spans that the block reads, in the order of their rows.
    pub spans: Vec<SpanUse>,
}

/// The token count and the writes of each slot, and the rules of the part calls. The engine applies the device side.
///
/// A slot never goes back: the conv state holds only the last inputs, so a pass starts at the token count. A write
/// puts the token count at 0, and `set_n` takes a count only when every part got rows that end at it (engine-abi "The
/// first Gemma 4 binding", "A write empties the slot until `set_n`").
#[derive(Clone, Debug)]
pub struct Held {
    pub kinds: Vec<LayerKind>,
    pub list: Vec<Part>,
    /// The positions of each slot.
    pub seqs: Vec<usize>,
    /// The token count of each slot.
    pub n: Vec<u64>,
    /// The parts that got a write since the last `set_n`, with the positions of a row part.
    writes: Vec<BTreeMap<PartId, Option<Range<u64>>>>,
    /// The first position that the rings of each slot hold since its last restore: 0, or the start of the restored
    /// rows.
    ring_from: Vec<u64>,
    /// The bytes of a row of a row part, and of a whole conv part.
    row_bytes: u64,
    conv_bytes: u64,
    window: u64,
    /// The placeholders of the media of the model, and the encoders of the engine. `LFM3-1.7B` has none.
    pub media: MediaRules,
}

impl Held {
    pub fn new(b: &Backbone, seqs: &[usize]) -> Self {
        Self {
            kinds: b.layers.clone(),
            list: part_list(b),
            seqs: seqs.to_vec(),
            n: vec![0; seqs.len()],
            writes: vec![BTreeMap::new(); seqs.len()],
            ring_from: vec![0; seqs.len()],
            row_bytes: (b.kv_row() * 4) as u64,
            conv_bytes: (b.conv_state() * 4) as u64,
            window: b.window as u64,
            media: MediaRules::default(),
        }
    }

    /// The slot of a state, or `None` past the slots.
    pub fn slot(&self, state: State) -> Option<usize> {
        let State::Slot(slot) = state;
        ((slot as usize) < self.n.len()).then_some(slot as usize)
    }

    /// The part of an id, or `None` past the parts.
    pub fn part(&self, id: PartId) -> Option<Part> {
        self.list.get(id).copied()
    }

    /// The bytes of one row of a row part.
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes
    }

    /// The bytes of a conv part.
    pub fn conv_bytes(&self) -> u64 {
        self.conv_bytes
    }

    /// The first position that a row part of layer `l` holds in `slot`. A cells part, or a sliding layer of a slot
    /// with no ring, holds every position from 0. A ring holds the last `RING_ROWS` positions, and after a restore no
    /// position before the restored rows.
    pub fn base(&self, slot: usize, l: usize) -> u64 {
        let seq = self.seqs[slot];
        if ringed(seq, self.kinds[l]) {
            self.n[slot]
                .saturating_sub(ring_of(seq) as u64)
                .max(self.ring_from[slot])
                .min(self.n[slot])
        } else {
            0
        }
    }

    /// Check a pass before it changes anything, and give its block. The message says which rule breaks.
    ///
    /// - A pass has one segment of tokens below `vocab`, with 1 to `max_block` of them. A segment of tokens alone has
    ///   no `meta`. A segment of tokens and rows has the `meta` of its media spans (`media::check_spans`).
    /// - The positions run on from the token count of the slot. A slot does not go back.
    /// - `logit_rows` names at most `MAX_LOGIT_ROWS` positions of the block.
    pub fn check_pass(&self, segments: &[Segment], max_block: usize, vocab: usize) -> Result<Block, String> {
        let [segment] = segments else {
            return Err(format!(
                "the pass has {} segments; the engine takes {MAX_SEGMENTS}",
                segments.len()
            ));
        };
        let slot = self
            .slot(segment.state)
            .ok_or_else(|| format!("the engine has {} slots, not {:?}", self.n.len(), segment.state))?;
        let (tokens, rows) = match &segment.inputs {
            SegmentInputs::Tokens(tokens) => (tokens, None),
            SegmentInputs::TokensAndRows { tokens, rows } => (tokens, Some(rows)),
            SegmentInputs::Rows(_) => {
                return Err(
                    "the segment has rows and no tokens; the engine takes tokens, with the rows of their \
                            media (D2)"
                        .to_owned(),
                );
            }
        };
        match (rows, segment.meta.is_empty()) {
            (None, false) => {
                return Err(
                    "the segment has meta and no rows; the meta names media spans, and their rows come with the \
                     tokens"
                        .to_owned(),
                );
            }
            (Some(_), true) => {
                return Err("the segment has rows and no meta; the meta names the media span of each row".to_owned());
            }
            _ => {}
        }
        let n = tokens.len();
        if n == 0 || n > max_block {
            return Err(format!("the segment has {n} tokens; a block holds 1 to {max_block}"));
        }
        if let Some(bad) = tokens.iter().find(|token| **token as usize >= vocab) {
            return Err(format!("token id {bad} is outside the vocabulary of {vocab} ids"));
        }
        if segment.positions.len() != n {
            return Err(format!(
                "the segment has {n} tokens and {} positions",
                segment.positions.len()
            ));
        }
        let held = self.n[slot];
        if segment.positions.iter().enumerate().any(|(t, p)| *p != held + t as u64) {
            return Err(format!(
                "slot {slot} holds {held} tokens, so the positions run from {held}; an LFM3 slot does not go back, \
                 because its conv state holds only the last inputs"
            ));
        }
        let pos0 = held as usize;
        let seq = self.seqs[slot];
        if pos0 + n > seq {
            return Err(format!(
                "positions {pos0} to {} do not fit slot {slot} of {seq} positions",
                pos0 + n
            ));
        }
        let wanted = &segment.outputs.logit_rows;
        if wanted.len() > MAX_LOGIT_ROWS {
            return Err(format!(
                "the segment wants {} logit rows; a block gives at most {MAX_LOGIT_ROWS}",
                wanted.len()
            ));
        }
        let mut logit_rows = Vec::with_capacity(wanted.len());
        for row in wanted {
            if *row < pos0 as u64 || *row >= (pos0 + n) as u64 {
                return Err(format!(
                    "a logit row at {row} is not a position of the block, {pos0} to {}",
                    pos0 + n
                ));
            }
            logit_rows.push(*row as usize - pos0);
        }
        let spans = media::decode(&segment.meta)?;
        let spans = media::check_spans(&self.media, &spans, tokens, pos0 as u64, rows.map_or(0, |r| r.len()))?;
        Ok(Block {
            slot,
            pos0,
            tokens: tokens.clone(),
            logit_rows,
            spans,
        })
    }

    /// Take a pass that ran: the slot holds the block's positions, and the writes before it are gone.
    pub fn ran(&mut self, block: &Block) {
        self.n[block.slot] = (block.pos0 + block.tokens.len()) as u64;
        self.writes[block.slot].clear();
    }

    /// Check a read of `n` rows from `from` of a row part, and give its slot and layer.
    pub fn check_read(&self, state: State, id: PartId, from: u64, n: u64, len: usize) -> Result<(usize, Part), String> {
        let (slot, part) = self.row_part(state, id)?;
        let base = self.base(slot, part.layer());
        if len as u64 != n * self.row_bytes || from < base || from + n > self.n[slot] {
            return Err(format!(
                "a read of rows {from} to {} of part {id} in {len} bytes does not fit: slot {slot} holds rows {base} to \
                 {} of {} bytes",
                from + n,
                self.n[slot],
                self.row_bytes
            ));
        }
        Ok((slot, part))
    }

    /// Check a whole read of a part into `len` bytes, and give the first row for a row part.
    pub fn check_part_read(&self, state: State, id: PartId, len: usize) -> Result<(usize, Part, u64), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        let part = self.part(id).ok_or_else(|| format!("the engine has no part {id}"))?;
        match part {
            Part::Conv(_) if len as u64 == self.conv_bytes => Ok((slot, part, 0)),
            Part::Conv(_) => Err(format!(
                "part {id} is a conv state of {} bytes, not {len}",
                self.conv_bytes
            )),
            _ => {
                if !(len as u64).is_multiple_of(self.row_bytes) {
                    return Err(format!("{len} bytes are not whole rows of {} bytes", self.row_bytes));
                }
                let from = self.base(slot, part.layer());
                self.check_read(state, id, from, len as u64 / self.row_bytes, len)?;
                Ok((slot, part, from))
            }
        }
    }

    fn row_part(&self, state: State, id: PartId) -> Result<(usize, Part), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        match self.part(id) {
            Some(part @ (Part::K(_) | Part::V(_))) => Ok((slot, part)),
            Some(Part::Conv(_)) => Err(format!(
                "part {id} is a conv state, a fixed part; use part_read and part_write"
            )),
            None => Err(format!("the engine has no part {id}")),
        }
    }

    /// Check a write of the rows of positions `from` to `from + n`, and give the slot, the part and the positions that
    /// the device keeps: all of them, or the last `RING_ROWS` of a ring.
    pub fn check_cells_write(
        &self,
        state: State,
        id: PartId,
        from: u64,
        n: u64,
        len: usize,
    ) -> Result<(usize, Part, Range<u64>), String> {
        let (slot, part) = self.row_part(state, id)?;
        let seq = self.seqs[slot];
        if len as u64 != n * self.row_bytes || from + n > seq as u64 {
            return Err(format!(
                "rows {from} to {} of part {id} in {len} bytes do not fit slot {slot} of {seq} positions",
                from + n
            ));
        }
        let kept = if ringed(seq, self.kinds[part.layer()]) {
            (from + n).saturating_sub(ring_of(seq) as u64).max(from)..from + n
        } else {
            if from != 0 {
                return Err(format!(
                    "part {id} in slot {slot} holds every position from 0; the write starts at {from}"
                ));
            }
            from..from + n
        };
        Ok((slot, part, kept))
    }

    /// Check a whole write of a conv part, and give its slot and part.
    pub fn check_part_write(&self, state: State, id: PartId, len: usize) -> Result<(usize, Part), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        match self.part(id) {
            Some(part @ Part::Conv(_)) if len as u64 == self.conv_bytes => Ok((slot, part)),
            Some(Part::Conv(_)) => Err(format!(
                "part {id} is a conv state of {} bytes, not {len}",
                self.conv_bytes
            )),
            Some(_) => Err(format!("part {id} is a row part; write its rows with cells_write")),
            None => Err(format!("the engine has no part {id}")),
        }
    }

    /// Take a write that the device made: the slot holds no token until `set_n`.
    pub fn wrote(&mut self, slot: usize, id: PartId, rows: Option<Range<u64>>) {
        self.n[slot] = 0;
        self.writes[slot].insert(id, rows);
    }

    /// Check a `set_n`. With no write since the last count, the count must stay or go to 0. After writes, every part
    /// must have got its bytes, and the rows of each row part must end at `n` and fit the slot.
    pub fn check_set_n(&self, state: State, n: u64) -> Result<usize, String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        let seq = self.seqs[slot];
        if n > seq as u64 {
            return Err(format!("slot {slot} holds {seq} positions, not {n}"));
        }
        let writes = &self.writes[slot];
        if writes.is_empty() {
            if n == self.n[slot] || n == 0 {
                return Ok(slot);
            }
            return Err(format!(
                "slot {slot} holds {} tokens and got no write; an LFM3 slot does not go back to {n}, because its conv \
                 state holds only the last inputs. Restore a snapshot instead",
                self.n[slot]
            ));
        }
        for (id, part) in self.list.iter().enumerate() {
            match (part, writes.get(&id)) {
                (_, None) => return Err(format!("part {id} got no bytes since the last set_n")),
                (Part::Conv(_), Some(_)) => {}
                (_, Some(None)) => return Err(format!("part {id} got no rows")),
                (_, Some(Some(rows))) => {
                    if rows.end != n {
                        return Err(format!("part {id} got rows up to {}, not up to {n}", rows.end));
                    }
                    self.fits(
                        slot,
                        &CoverageSpan {
                            part: id,
                            start: rows.start,
                            end: rows.end,
                        },
                    )?;
                }
            }
        }
        Ok(slot)
    }

    /// Take a count that `check_set_n` passed. After writes, the rings hold the positions from the latest start of the
    /// rows of a ring part.
    pub fn set_n(&mut self, slot: usize, n: u64) {
        let seq = self.seqs[slot];
        let from = self.writes[slot]
            .iter()
            .filter_map(|(id, rows)| match (self.list[*id], rows) {
                (Part::K(l) | Part::V(l), Some(rows)) if ringed(seq, self.kinds[l]) => Some(rows.start),
                _ => None,
            })
            .max();
        if n == 0 {
            self.ring_from[slot] = 0;
        } else if let Some(from) = from {
            self.ring_from[slot] = from;
        }
        self.n[slot] = n;
        self.writes[slot].clear();
    }

    /// Empty a slot.
    pub fn clear(&mut self, slot: usize) {
        self.n[slot] = 0;
        self.ring_from[slot] = 0;
        self.writes[slot].clear();
    }

    /// The positions that `bytes` of a part hold in a state, from the base of the part.
    pub fn coverage(&self, state: State, id: PartId, bytes: &[u8]) -> CoverageSpan {
        match (self.slot(state), self.part(id)) {
            (Some(slot), Some(part @ (Part::K(_) | Part::V(_)))) => {
                let start = self.base(slot, part.layer());
                CoverageSpan {
                    part: id,
                    start,
                    end: start + bytes.len() as u64 / self.row_bytes,
                }
            }
            _ => CoverageSpan {
                part: id,
                start: 0,
                end: 0,
            },
        }
    }

    /// Check that the positions of `span` fit `slot`. A cells part, and any layer of a slot with no ring, needs every
    /// position from 0. A ring needs the window before the end.
    pub fn fits(&self, slot: usize, span: &CoverageSpan) -> Result<(), String> {
        let seq = *self
            .seqs
            .get(slot)
            .ok_or_else(|| format!("the engine has no slot {slot}"))?;
        let part = self
            .part(span.part)
            .ok_or_else(|| format!("the engine has no part {}", span.part))?;
        if matches!(part, Part::Conv(_)) {
            return Err(format!("part {} is a conv state; it has no positions", span.part));
        }
        if span.start > span.end || span.end > seq as u64 {
            return Err(format!(
                "positions {} to {} do not fit a slot of {seq} positions",
                span.start, span.end
            ));
        }
        if !ringed(seq, self.kinds[part.layer()]) {
            if span.start != 0 {
                return Err(format!(
                    "part {} in a slot of {seq} positions needs every position from 0; it starts at {}",
                    span.part, span.start
                ));
            }
            return Ok(());
        }
        let need = span.end.saturating_sub(self.window);
        if span.start > need {
            return Err(format!(
                "part {} needs the window of {} positions before {}; it starts at {}",
                span.part, self.window, span.end, span.start
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;
