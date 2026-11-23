//! Host configuration, model validation, slot sizing, and `Limits` planning for `mk-inkling2-sm121`.
//!
//! `plan` reads only `config.json`, `generation_config.json`, and the safetensors header (zero weight bytes read),
//! so host tests and admission checks need no GPU (spec 2.4).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nagaya_abi::{
    CONTROL_TERM, Capability, ChainLimits, ControlLimit, CyclesLimit, Determinism, Dtype, FeedRule, Inputs, Kind,
    Lifetime, Limits, MandatoryTerm, MemoryLimits, PartKind, PartLimits, SlotLimit, TapLimit, Tier,
};
use nagaya_mk_cuda::attn::{AttnShape, CacheFormat, plane_bytes, sliding_ring_rows};
use nagaya_mk_cuda::protocol::{ATTN_CHUNK_KEYS, ATTN_SPLIT_ROWS, NUM_EXPERTS, NUM_SCOREBOARD_EVENTS, RING_DEV_BYTES};
use nagaya_mk_cuda::safetensors::{SafeTensorsHeader, TensorMeta};

/// Engine kind name in a placement profile (`AGENTS.md`, "Layout").
pub const KIND: &str = "mk-inkling2-sm121";

/// Family library name (spec 11.3).
pub const FAMILY: &str = "inkling2";

/// Quantization mode string recorded in engine identity `numbers` (spec 12.1).
pub const QUANT_MODE: &str = "nvfp4_fouroversix_mae+mxfp8_ue8m0";

/// State layout version in engine identity `numbers` (version 2 includes `cycles`; version 3 has the head-major cache
/// planes and a ring with no spill).
pub const STATE_LAYOUT: u32 = 3;

/// Weight file inside the model folder.
pub const WEIGHTS_FILE: &str = "model-00001-of-00001.safetensors";

// Model architecture constants for Inkling 2 Mini / varloop.
pub const HIDDEN: usize = 2304;
pub const LAYERS: usize = 24;
pub const NVFP4_LAYERS: usize = 20;
pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const KV_ROW: usize = KV_HEADS * HEAD_DIM;
pub const VOCAB: usize = 201_024;
pub const LOGITS: usize = 200_058;
pub const CONV_TAPS: usize = 4;
pub const CONV_ROWS: usize = CONV_TAPS - 1;
pub const CONV_ROW: usize = 2 * KV_HEADS * HEAD_DIM + 2 * HIDDEN;
pub const CYCLE_START: usize = 6;
pub const CYCLE_END: usize = 18;
pub const CYCLE_LAYERS: usize = CYCLE_END - CYCLE_START;
pub const DEFAULT_CYCLES: usize = 3;
pub const DEFAULT_EXTRA_RUNS: usize = (DEFAULT_CYCLES - 1) * CYCLE_LAYERS;
pub const MAX_EXTRA_RUNS: usize = 4 * CYCLE_LAYERS;
pub const DEFAULT_MAX_BLOCK: usize = 512;
pub const MAX_BLOCK: usize = 512;
pub const SLOT_TOKENS: usize = 32_768;
pub const MAX_SLOT_TOKENS: usize = 1_048_576;
pub const WINDOW: usize = 1024;
/// Routed experts in each layer, and the width of one expert (`n_routed_experts`, `intermediate_size`).
pub const EXPERTS: usize = NUM_EXPERTS;
pub const INTERMEDIATE: usize = 1152;
/// The rows of the ring of a sliding state: the window and the longest block, rounded up to a chunk of the attention.
/// The projection writes a new row over the row that is one ring older, so the ring has no spill.
pub const RING_ROWS: usize = sliding_ring_rows(WINDOW, MAX_BLOCK);
/// The format of the cache that the plan counts. An f32 cache is the oracle form, and e4m3 is the form of the long
/// slots (`nagaya_mk_cuda::attn::CacheFormat`).
pub const CACHE_FORMAT: CacheFormat = CacheFormat::F32;
pub const MAX_QUANTUM_MS: u32 = 5_000;
pub const MAX_SEGMENTS: u32 = 1;
pub const MAX_LOGIT_ROWS: usize = 16;
pub const LATENT_TOKEN: u32 = 199_999;

/// Staging host buffer size (64 MiB).
const STAGING_BYTES: usize = 64 << 20;

/// Control table budget: 8 vectors of 24 layers of 2,304 `f32` values.
const CONTROL_TABLES: u32 = 8;

/// Vision tower buffer reserve for up to 4,096 x 4,096 px images (matches `docs/engine-abi.md`).
const VISION_WORK_BYTES: u64 = 1_870_588_096;

/// Fixed block activation buffers for `max_block` tokens (conservative upper bound matching Inkling 2).
const BLOCK_WORK_FLOATS_PER_TOKEN: usize = 64 * 1024;

/// Full-attention layers (layers 5, 11, 17, 23).
pub fn is_full(layer: usize) -> bool {
    layer % 6 == 5
}

/// Layer index for state `s` (layers 0..24, then cycle reserve entries).
pub fn state_layer(s: usize) -> usize {
    match s.checked_sub(LAYERS) {
        None => s,
        Some(i) => CYCLE_START + i % CYCLE_LAYERS,
    }
}

fn state_name(s: usize) -> String {
    match s.checked_sub(LAYERS) {
        None => format!("{s:02}"),
        Some(i) => format!("x{i:02}"),
    }
}

/// Rows allocated per key or value plane of state `s` in a slot of `seq` positions: a full state holds each position,
/// rounded up to a chunk of the attention, and a sliding state holds its ring.
pub fn plane_rows(seq: usize, s: usize) -> usize {
    if is_full(state_layer(s)) {
        seq.next_multiple_of(ATTN_CHUNK_KEYS)
    } else {
        RING_ROWS
    }
}

/// Round an allocation size up to a multiple of 64 bytes (minimum 64).
pub fn rounded(bytes: usize) -> usize {
    bytes.max(64).next_multiple_of(64)
}

/// Device bytes of one slot of `seq` positions with a reserve of `extra` runs, for a cache in `format`. A conv state
/// is always f32.
pub fn slot_bytes(seq: usize, extra: usize, format: CacheFormat) -> u64 {
    (0..LAYERS + extra)
        .map(|s| (rounded(plane_bytes(plane_rows(seq, s), format) * 2) + rounded(CONV_ROWS * CONV_ROW * 4)) as u64)
        .sum()
}

/// The bytes of the scratch of the attention for slots of `max_seq` positions and blocks of `max_block` rows.
///
/// A block of at most `ATTN_SPLIT_ROWS` rows runs in the split mode: each chunk of keys writes a partial of 16 rows of
/// `ATTN_PARTIAL_ROW_FLOATS` floats for each (KV head, tile), and one counter for each group. The bound is the block of
/// the most rows in the split mode, at the end of a full run of `max_seq` positions. A larger block runs in the whole
/// mode, and it needs no scratch (`AttnShape::scratch_floats`).
pub fn attn_scratch_bytes(max_seq: usize, max_block: usize) -> u64 {
    let rows = max_block.min(ATTN_SPLIT_ROWS).min(max_seq);
    if rows == 0 {
        return 0;
    }
    let plane = max_seq.next_multiple_of(ATTN_CHUNK_KEYS);
    let shape = AttnShape::new(rows, max_seq - rows, 0, plane, CacheFormat::F32)
        .expect("a block of the split mode fits a full run of the longest slot");
    (shape.scratch_floats() as u64).saturating_mul(4)
}

/// Supported Inkling 2 model variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Mini,
    Research,
}

impl Model {
    /// Validate `config.json` and return the model variant.
    pub fn parse(text: &str) -> Result<Self, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not valid JSON: {e}"))?;
        if root["architectures"][0] != "Inkling2LoopForConditionalGeneration" {
            return Err(format!(
                "config.json has architectures {}; this engine runs Inkling2LoopForConditionalGeneration",
                root["architectures"]
            ));
        }
        let tc = &root["text_config"];
        let check_u64 = |name: &str, got: &serde_json::Value, want: usize| -> Result<(), String> {
            if got.as_u64() == Some(want as u64) {
                Ok(())
            } else {
                Err(format!(
                    "config.json has {name} {got}, expected {want}. The folder holds another model: use Inkling-2-Mini \
                     or the variable-loop research model"
                ))
            }
        };
        check_u64("text_config.hidden_size", &tc["hidden_size"], HIDDEN)?;
        check_u64("text_config.num_hidden_layers", &tc["num_hidden_layers"], LAYERS)?;
        check_u64("text_config.num_attention_heads", &tc["num_attention_heads"], HEADS)?;
        check_u64("text_config.num_key_value_heads", &tc["num_key_value_heads"], KV_HEADS)?;
        check_u64("text_config.head_dim", &tc["head_dim"], HEAD_DIM)?;
        check_u64("text_config.vocab_size", &tc["vocab_size"], VOCAB)?;
        check_u64("text_config.unpadded_vocab_size", &tc["unpadded_vocab_size"], LOGITS)?;
        check_u64("text_config.model_max_length", &tc["model_max_length"], MAX_SLOT_TOKENS)?;
        // The slot plan reads the window and the taps of the convolution. The ops read the experts.
        check_u64("text_config.sliding_window_size", &tc["sliding_window_size"], WINDOW)?;
        check_u64("text_config.sconv_kernel_size", &tc["sconv_kernel_size"], CONV_TAPS)?;
        check_u64("text_config.n_routed_experts", &tc["n_routed_experts"], EXPERTS)?;
        check_u64("text_config.intermediate_size", &tc["intermediate_size"], INTERMEDIATE)?;
        // The slot plan holds a full state for each layer that is not local.
        let local: Vec<u64> = (0..LAYERS).filter(|&l| !is_full(l)).map(|l| l as u64).collect();
        let got_local: Option<Vec<u64>> = tc["local_layer_ids"]
            .as_array()
            .and_then(|ids| ids.iter().map(serde_json::Value::as_u64).collect());
        if got_local.as_ref() != Some(&local) {
            return Err(format!(
                "config.json has text_config.local_layer_ids {}, expected {local:?}. The folder holds another model: \
                 use Inkling-2-Mini or the variable-loop research model",
                tc["local_layer_ids"]
            ));
        }
        check_u64(
            "text_config.loop_config.core_start",
            &tc["loop_config"]["core_start"],
            CYCLE_START,
        )?;
        check_u64(
            "text_config.loop_config.core_end",
            &tc["loop_config"]["core_end"],
            CYCLE_END,
        )?;
        check_u64(
            "text_config.loop_config.num_passes",
            &tc["loop_config"]["num_passes"],
            DEFAULT_CYCLES,
        )?;

        match tc["loop_config"]["is_varloop"].as_bool() {
            Some(false) => Ok(Self::Mini),
            Some(true) => Ok(Self::Research),
            None => Err(format!(
                "config.json has text_config.loop_config.is_varloop {}, expected true or false",
                tc["loop_config"]["is_varloop"]
            )),
        }
    }

    /// Read and validate `config.json` from `dir`.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Model name in engine identity.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mini => "inkling2-mini",
            Self::Research => "inkling2-varloop",
        }
    }
}

/// Configuration for one `mk-inkling2-sm121` engine instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub model_dir: PathBuf,
    pub slots: Vec<usize>,
    pub max_block: usize,
    pub extra_runs: usize,
}

impl Config {
    /// Validate all config values.
    pub fn check(&self) -> Result<(), String> {
        if self.max_block == 0 || self.max_block > MAX_BLOCK {
            return Err(format!(
                "max_block is {}; it must be 1 to {MAX_BLOCK} tokens",
                self.max_block
            ));
        }
        if self.slots.is_empty() {
            return Err("the config names no slot; give `slots`, a count of slots or a list of positions".to_owned());
        }
        if let Some(seq) = self.slots.iter().find(|&&seq| seq == 0 || seq > MAX_SLOT_TOKENS) {
            return Err(format!(
                "a slot of {seq} positions is outside 1 to {MAX_SLOT_TOKENS}, the positions of the model"
            ));
        }
        if !matches!(self.extra_runs, 24 | 36 | 48) {
            return Err(format!(
                "extra_runs is {}; it must be 24, 36 or 48 (for 3, 4 or 5 cycles)",
                self.extra_runs
            ));
        }
        Ok(())
    }

    /// Sorted slot lengths, shortest first.
    pub fn seqs(&self) -> Vec<usize> {
        let mut seqs = self.slots.clone();
        seqs.sort_unstable();
        seqs
    }

    /// Longest slot in the config.
    pub fn max_seq(&self) -> usize {
        self.slots.iter().copied().max().unwrap_or(0)
    }

    /// Total KV and conv bytes across all configured slots, for a cache in `format`.
    pub fn slots_bytes(&self, format: CacheFormat) -> u64 {
        self.slots
            .iter()
            .map(|&seq| slot_bytes(seq, self.extra_runs, format))
            .sum()
    }
}

/// State parts declared in `Limits::parts` (matches `docs/engine-abi.md` "Inkling 2").
pub fn parts(extra: usize) -> Vec<PartLimits> {
    let mut out = Vec::with_capacity((LAYERS + extra) * 3 + 1);
    for s in 0..LAYERS + extra {
        let name = state_name(s);
        let row_kind = if is_full(state_layer(s)) {
            PartKind::Cells
        } else {
            PartKind::Ring
        };
        out.push(PartLimits {
            name: format!("k.{name}"),
            kind: row_kind,
            dtype: Dtype::F32,
            shape: vec![KV_HEADS as u64, HEAD_DIM as u64],
            block_format: None,
            stage: 0,
        });
        out.push(PartLimits {
            name: format!("v.{name}"),
            kind: row_kind,
            dtype: Dtype::F32,
            shape: vec![KV_HEADS as u64, HEAD_DIM as u64],
            block_format: None,
            stage: 0,
        });
        out.push(PartLimits {
            name: format!("conv.{name}"),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![CONV_ROWS as u64, CONV_ROW as u64],
            block_format: None,
            stage: 0,
        });
    }
    out.push(PartLimits {
        name: "cycles".to_owned(),
        kind: PartKind::Fixed,
        dtype: Dtype::U32,
        shape: vec![1],
        block_format: None,
        stage: 0,
    });
    out
}

fn slot_limits(seqs: &[usize], extra: usize, format: CacheFormat) -> Vec<SlotLimit> {
    let mut by_seq: BTreeMap<usize, u32> = BTreeMap::new();
    for &seq in seqs {
        *by_seq.entry(seq).or_insert(0) += 1;
    }
    by_seq
        .into_iter()
        .map(|(seq, count)| SlotLimit {
            count,
            tokens: seq as u64,
            bytes: slot_bytes(seq, extra, format),
        })
        .collect()
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

/// Sum weight bytes for `model.llm.*` and `model.visual.*` from the safetensors header,
/// verifying expected prefixes and NVFP4/MXFP8 expert tensor presence.
fn sum_checkpoint_weights(tensors: &BTreeMap<String, TensorMeta>) -> Result<(u64, u64), String> {
    let mut llm_bytes: u64 = 0;
    let mut vision_bytes: u64 = 0;
    for (name, meta) in tensors {
        if name.starts_with("model.llm.") {
            llm_bytes = llm_bytes.saturating_add(rounded(meta.nbytes as usize) as u64);
        } else if name.starts_with("model.visual.") {
            vision_bytes = vision_bytes.saturating_add(rounded(meta.nbytes as usize) as u64);
        }
    }
    // Verify NVFP4 expert tensors in layers 0..20 and MXFP8 in layers 20..24 exist with expected dtypes.
    for layer in 0..LAYERS {
        let want_dtype = if layer < NVFP4_LAYERS { "U8" } else { "F8_E4M3" };
        for suffix in ["w13_weight", "w2_weight"] {
            let key = format!("model.llm.layers.{layer}.mlp.experts.{suffix}");
            let meta = tensors
                .get(&key)
                .ok_or_else(|| format!("checkpoint is missing required expert tensor `{key}`"))?;
            if meta.dtype != want_dtype {
                return Err(format!(
                    "tensor `{key}` has dtype `{}`, expected `{want_dtype}`",
                    meta.dtype
                ));
            }
        }
    }
    if llm_bytes == 0 {
        return Err("checkpoint has no `model.llm.` tensors".to_owned());
    }
    Ok((llm_bytes, vision_bytes))
}

/// Build `Limits` from `config` and measured weight byte sizes.
pub fn limits(config: &Config, llm_weights: u64, vision_weights: u64) -> Limits {
    let ring_bytes = (NUM_SCOREBOARD_EVENTS * 4 + RING_DEV_BYTES) as u64;
    let block_work = (config.max_block * BLOCK_WORK_FLOATS_PER_TOKEN * 4) as u64;
    let work = block_work + attn_scratch_bytes(config.max_seq(), config.max_block) + ring_bytes + VISION_WORK_BYTES;
    let pinned = (RING_DEV_BYTES + 256 * 24 + STAGING_BYTES) as u64;
    let control = ControlLimit {
        tables: CONTROL_TABLES,
        layers: LAYERS as u32,
        width: HIDDEN as u32,
    };
    let max_cycles = (1 + config.extra_runs / CYCLE_LAYERS) as u8;
    Limits {
        capabilities: vec![
            Capability::Parts,
            Capability::Slots,
            Capability::Coverage,
            Capability::Close,
            Capability::Cycles,
            Capability::Control,
        ],
        parts: parts(config.extra_runs),
        inputs: Inputs::TokensAndRows,
        devices: vec![0],
        stages: Vec::new(),
        max_segments: MAX_SEGMENTS,
        max_pass_inputs: config.max_block as u32,
        max_logit_rows: MAX_LOGIT_ROWS as u32,
        slots: slot_limits(&config.seqs(), config.extra_runs, CACHE_FORMAT),
        max_quantum_ms: MAX_QUANTUM_MS,
        // No long pass until the vision tower runs here (milestone 37 step 13).
        max_long_quantum_ms: None,
        memory: MemoryLimits {
            mandatory: vec![
                term("weights", Tier::Device(0), llm_weights),
                term("vision", Tier::Device(0), vision_weights),
                term("kv", Tier::Device(0), config.slots_bytes(CACHE_FORMAT)),
                term("work", Tier::Device(0), work),
                term("pinned", Tier::Host, pinned),
                term(CONTROL_TERM, Tier::Device(0), control.table_bytes()),
            ],
            elastic: Vec::new(),
        },
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        segments_exact: false,
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        taps: vec![
            TapLimit {
                name: "residual".to_owned(),
                conditions: vec![nagaya_abi::TAP_EACH_RUN.to_owned()],
            },
            TapLimit {
                name: "fold".to_owned(),
                conditions: Vec::new(),
            },
        ],
        checkpoint_grain: None,
        residual_row_bytes: 0,
        control: Some(control),
        feed: Some(FeedRule {
            block: config.max_block as u64,
        }),
        layer_loops: None,
        cycles: Some(CyclesLimit {
            min: 1,
            max: max_cycles,
            default: DEFAULT_CYCLES as u8,
            start: CYCLE_START as u32,
            end: CYCLE_END as u32,
        }),
    }
}

/// Plan `Limits` from `config.model_dir` without reading weight bytes (spec 2.4).
pub fn plan(config: &Config) -> Result<Limits, String> {
    config.check()?;
    let _model = Model::read(&config.model_dir)?;
    let gen_path = config.model_dir.join("generation_config.json");
    let gen_text =
        std::fs::read_to_string(&gen_path).map_err(|e| format!("cannot read `{}`: {e}", gen_path.display()))?;
    let gen_val: serde_json::Value =
        serde_json::from_str(&gen_text).map_err(|e| format!("`{}` is not valid JSON: {e}", gen_path.display()))?;
    if gen_val["pad_token_id"].as_u64() != Some(u64::from(LATENT_TOKEN)) {
        return Err(format!(
            "`{}` has pad_token_id {}, expected {LATENT_TOKEN}",
            gen_path.display(),
            gen_val["pad_token_id"]
        ));
    }
    let header = SafeTensorsHeader::read(&config.model_dir.join(WEIGHTS_FILE))?;
    let (llm_weights, vision_weights) = sum_checkpoint_weights(&header.tensors)?;
    let lim = limits(config, llm_weights, vision_weights);
    lim.check().map_err(|e| format!("planned limits are invalid: {e}"))?;
    Ok(lim)
}

/// Identity `numbers` map for `mk-inkling2-sm121` (spec 12.1).
pub fn identity_numbers(config: &Config) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    map.insert("quant".to_owned(), QUANT_MODE.to_owned());
    map.insert("max_block".to_owned(), config.max_block.to_string());
    map.insert("ring_rows".to_owned(), RING_ROWS.to_string());
    map.insert(
        "cache".to_owned(),
        match CACHE_FORMAT {
            CacheFormat::F32 => "f32",
            CacheFormat::E4m3 => "e4m3",
        }
        .to_owned(),
    );
    map.insert("cycles".to_owned(), DEFAULT_CYCLES.to_string());
    map.insert("extra_runs".to_owned(), config.extra_runs.to_string());
    map.insert("state".to_owned(), STATE_LAYOUT.to_string());
    #[cfg(feature = "cuda")]
    {
        map.insert("nvcc".to_owned(), nagaya_mk_cuda::NVCC_VERSION.to_owned());
        map.insert(
            "cubin_blake3".to_owned(),
            blake3::hash(nagaya_mk_cuda::INKLING2_CUBIN).to_hex().to_string(),
        );
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_check_refuses_bad_block_slots_and_extra_runs() {
        let base = Config {
            model_dir: PathBuf::from("unused"),
            slots: vec![SLOT_TOKENS],
            max_block: DEFAULT_MAX_BLOCK,
            extra_runs: DEFAULT_EXTRA_RUNS,
        };
        assert!(base.check().is_ok());
        assert!(
            Config {
                max_block: 0,
                ..base.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                max_block: 513,
                ..base.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                slots: vec![],
                ..base.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                slots: vec![0],
                ..base.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                slots: vec![MAX_SLOT_TOKENS + 1],
                ..base.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                extra_runs: 12,
                ..base.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                extra_runs: 36,
                ..base.clone()
            }
            .check()
            .is_ok()
        );
        assert!(Config { extra_runs: 48, ..base }.check().is_ok());
    }

    /// The bytes of the conv part of one state: 3 rows of 6,656 `f32` values.
    const CONV_BYTES: u64 = 3 * 6_656 * 4;
    /// The bytes of the two planes of a sliding state: 1,536 rows of 1,024 values, for the keys and the values.
    const SLIDING_F32: u64 = 2 * 1_536 * 1_024 * 4;
    const SLIDING_E4M3: u64 = 2 * 1_536 * 1_024;

    #[test]
    fn the_state_plan_holds_eight_full_and_forty_sliding_states() {
        // 24 layers and 24 extra runs: 48 states. Layers 5 and 23 are full once. Layers 11 and 17 are full in each
        // of 3 cycles: once in the layers, and twice in the extra runs.
        let full: Vec<usize> = (0..LAYERS + 24).filter(|&s| is_full(state_layer(s))).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        let layers: Vec<usize> = full.iter().map(|&s| state_layer(s)).collect();
        assert_eq!(layers, vec![5, 11, 17, 23, 11, 17, 11, 17]);
        assert_eq!((LAYERS + 24) - full.len(), 40);
        // The local layers are the 20 that the config lists.
        let local: Vec<usize> = (0..LAYERS).filter(|&l| !is_full(l)).collect();
        assert_eq!(
            local,
            vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
        );
        // One row of a key plane holds 8 heads of 128 `f32` values.
        assert_eq!(KV_ROW, 1_024);
        assert_eq!(plane_rows(409_600, 5), 409_600);
        assert_eq!(
            plane_rows(409_600, 0),
            1_536,
            "a ring of the window and a block of 512, with no spill"
        );
        assert_eq!(plane_rows(409_600, 24), 1_536, "state 24 is layer 6, a sliding layer");
        assert_eq!(plane_rows(1_000, 5), 1_024, "a full state rounds up to a chunk of 256");
        assert_eq!(plane_rows(409_600, 29), 409_600, "state 29 is layer 11, a full layer");
    }

    #[test]
    fn slot_bytes_are_exact_for_each_length_and_format() {
        // Each full state holds two planes of `seq` rows of 1,024 values, and each sliding state two planes of 1,536
        // rows. A value is 4 bytes in an f32 cache and 1 byte in an e4m3 cache. Rows are 64-byte multiples.
        let formula =
            |seq: u64, per_value: u64, sliding: u64| 8 * (2 * seq * 1_024 * per_value) + 40 * sliding + 48 * CONV_BYTES;
        assert_eq!(SLIDING_F32, 12_582_912);
        assert_eq!(SLIDING_E4M3, 3_145_728);
        assert_eq!(CONV_BYTES, 79_872);
        let f32_ = CacheFormat::F32;
        let e4m3 = CacheFormat::E4m3;
        assert_eq!(slot_bytes(32_768, 24, f32_), 2_654_633_984);
        assert_eq!(slot_bytes(409_600, 24, f32_), 27_350_695_936);
        assert_eq!(slot_bytes(524_288, 24, f32_), 34_866_888_704);
        assert_eq!(slot_bytes(32_768, 24, e4m3), 666_533_888);
        assert_eq!(slot_bytes(409_600, 24, e4m3), 6_840_549_376);
        assert_eq!(slot_bytes(524_288, 24, e4m3), 8_719_597_568);
        for seq in [32_768u64, 409_600, 524_288] {
            assert_eq!(slot_bytes(seq as usize, 24, f32_), formula(seq, 4, SLIDING_F32));
            assert_eq!(slot_bytes(seq as usize, 24, e4m3), formula(seq, 1, SLIDING_E4M3));
        }
        // More cycles add states: 36 extra runs are 12 more states, 2 of them full, and 10 sliding.
        assert_eq!(
            slot_bytes(409_600, 36, f32_) - slot_bytes(409_600, 24, f32_),
            2 * (2 * 409_600 * 1_024 * 4) + 10 * SLIDING_F32 + 12 * CONV_BYTES
        );
        assert_eq!(
            slot_bytes(409_600, 36, e4m3) - slot_bytes(409_600, 24, e4m3),
            2 * (2 * 409_600 * 1_024) + 10 * SLIDING_E4M3 + 12 * CONV_BYTES
        );
    }

    #[test]
    fn attention_scratch_is_the_partials_of_the_split_mode() {
        // A block of 16 rows is 48 rows of the MMA: 3 tiles, so 24 groups. Each group holds a partial of
        // 16 x 132 floats for each chunk of 256 keys of a full run, after the 24 counters.
        assert_eq!(attn_scratch_bytes(0, 512), 0);
        assert_eq!(attn_scratch_bytes(409_600, 512), (24 + 24 * 1_600 * 16 * 132) * 4);
        assert_eq!(attn_scratch_bytes(524_288, 512), 415_236_192);
        assert_eq!(attn_scratch_bytes(524_288, 512), (24 + 24 * 2_048 * 16 * 132) * 4);
        // A decode row is 3 rows of the MMA: 1 tile, 8 groups.
        assert_eq!(attn_scratch_bytes(524_288, 1), (8 + 8 * 2_048 * 16 * 132) * 4);
        // A block of 1 row in a slot of 1 position: one chunk.
        assert_eq!(attn_scratch_bytes(1, 512), (8 + 8 * 16 * 132) * 4);
    }

    #[test]
    fn parts_follow_the_state_plan() {
        let list = parts(24);
        assert_eq!(list.len(), 48 * 3 + 1);
        let count = |kind: PartKind| list.iter().filter(|p| p.kind == kind).count();
        assert_eq!(
            count(PartKind::Cells),
            16,
            "8 full states, a key part and a value part each"
        );
        assert_eq!(count(PartKind::Ring), 80, "40 sliding states");
        assert_eq!(
            count(PartKind::Fixed),
            48 + 1,
            "a conv part for each state, and `cycles`"
        );
        let find = |name: &str| {
            list.iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("no part {name}"))
        };
        assert_eq!(find("k.04").kind, PartKind::Ring);
        assert_eq!(find("k.05").kind, PartKind::Cells);
        assert_eq!(find("v.x05").kind, PartKind::Cells, "extra entry 5 is layer 11");
        assert_eq!(find("k.x06").kind, PartKind::Ring, "extra entry 6 is layer 12");
        assert_eq!(find("k.05").shape, vec![8, 128]);
        assert_eq!(find("conv.x23").shape, vec![3, 6_656]);
        let last = list.last().unwrap();
        assert_eq!(
            (last.name.as_str(), last.dtype, last.kind),
            ("cycles", Dtype::U32, PartKind::Fixed)
        );
        assert_eq!(parts(0).len(), 24 * 3 + 1);
    }

    #[test]
    fn limits_pass_abi_check_for_512k_slot() {
        let cfg = Config {
            model_dir: PathBuf::from("unused"),
            slots: vec![524_288],
            max_block: 512,
            extra_runs: 24,
        };
        // The sums of the Inkling 2 Mini header, with each tensor rounded up to 64 bytes.
        let lim = limits(&cfg, 18_889_360_896, 636_512_256);
        lim.check().unwrap();
        let bytes = |name: &str| lim.memory.mandatory.iter().find(|t| t.name == name).unwrap().bytes;
        assert_eq!(bytes("weights"), 18_889_360_896);
        assert_eq!(bytes("vision"), 636_512_256);
        assert_eq!(bytes("kv"), 34_866_888_704);
        // The work: 512 rows of 65,536 `f32` values, the scratch, the ring, and the vision tower.
        assert_eq!(
            bytes("work"),
            134_217_728 + 415_236_192 + (2_049 * 4 + 262_144) + 1_870_588_096
        );
        assert_eq!(bytes("pinned"), 262_144 + 6_144 + 67_108_864);
        assert_eq!(lim.slots.len(), 1);
        assert_eq!(
            (lim.slots[0].count, lim.slots[0].tokens, lim.slots[0].bytes),
            (1, 524_288, 34_866_888_704)
        );
        let cycles = lim.cycles.unwrap();
        assert_eq!((cycles.min, cycles.max, cycles.default), (1, 3, 3));
    }

    #[test]
    fn the_identity_holds_each_number_that_moves_the_bits() {
        let cfg = Config {
            model_dir: PathBuf::from("unused"),
            slots: vec![SLOT_TOKENS],
            max_block: 256,
            extra_runs: 36,
        };
        let nums = identity_numbers(&cfg);
        let host: Vec<(&str, &str)> = nums
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "nvcc" | "cubin_blake3"))
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        // The B70 identity holds the same keys, except `quant`, which only this engine has.
        assert_eq!(
            host,
            vec![
                ("cache", "f32"),
                ("cycles", "3"),
                ("extra_runs", "36"),
                ("max_block", "256"),
                ("quant", QUANT_MODE),
                ("ring_rows", "1536"),
                ("state", "3"),
            ]
        );
        assert_eq!(nums.contains_key("nvcc"), cfg!(feature = "cuda"));
        assert_eq!(nums.contains_key("cubin_blake3"), cfg!(feature = "cuda"));
    }
}
