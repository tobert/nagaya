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
use nagaya_mk_cuda::protocol::{MAX_TASKS_PER_CARD, NUM_SCOREBOARD_EVENTS, RING_DEV_BYTES};
use nagaya_mk_cuda::safetensors::{SafeTensorsHeader, TensorMeta};

/// Engine kind name in a placement profile (`AGENTS.md`, "Layout").
pub const KIND: &str = "mk-inkling2-sm121";

/// Family library name (spec 11.3).
pub const FAMILY: &str = "inkling2";

/// Quantization mode string recorded in engine identity `numbers` (spec 12.1).
pub const QUANT_MODE: &str = "nvfp4_fouroversix_mae+mxfp8_ue8m0";

/// State layout version in engine identity `numbers` (version 2 includes `cycles`).
pub const STATE_LAYOUT: u32 = 2;

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
pub const RING_ROWS: usize = 2048;
pub const RING_SPILL: usize = 1024;
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

/// Rows allocated per key or value plane of state `s` in a slot of `seq` positions.
pub fn plane_rows(seq: usize, s: usize) -> usize {
    if is_full(state_layer(s)) {
        seq
    } else {
        RING_ROWS + RING_SPILL
    }
}

/// Round an allocation size up to a multiple of 64 bytes (minimum 64).
pub fn rounded(bytes: usize) -> usize {
    bytes.max(64).next_multiple_of(64)
}

/// Device bytes of one slot of `seq` positions with a reserve of `extra` runs.
pub fn slot_bytes(seq: usize, extra: usize) -> u64 {
    (0..LAYERS + extra)
        .map(|s| (rounded(2 * plane_rows(seq, s) * KV_ROW * 4) + rounded(CONV_ROWS * CONV_ROW * 4)) as u64)
        .sum()
}

/// Attention scratch floats for a block of `max_block` tokens over `max_seq` keys (3,120 bytes/position at 512).
pub fn attn_scratch_bytes(max_seq: usize) -> u64 {
    (max_seq as u64).saturating_mul(3_120)
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
                Err(format!("config.json has {name} {got}, expected {want}"))
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

    /// Total KV and conv bytes across all configured slots.
    pub fn slots_bytes(&self) -> u64 {
        self.slots.iter().map(|&seq| slot_bytes(seq, self.extra_runs)).sum()
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

fn slot_limits(seqs: &[usize], extra: usize) -> Vec<SlotLimit> {
    let mut by_seq: BTreeMap<usize, u32> = BTreeMap::new();
    for &seq in seqs {
        *by_seq.entry(seq).or_insert(0) += 1;
    }
    by_seq
        .into_iter()
        .map(|(seq, count)| SlotLimit {
            count,
            tokens: seq as u64,
            bytes: slot_bytes(seq, extra),
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
    let work = block_work + attn_scratch_bytes(config.max_seq()) + ring_bytes + VISION_WORK_BYTES;
    let pinned = (MAX_TASKS_PER_CARD * 64 + 256 * 24 + STAGING_BYTES) as u64;
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
        slots: slot_limits(&config.seqs(), config.extra_runs),
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory: vec![
                term("weights", Tier::Device(0), llm_weights),
                term("vision", Tier::Device(0), vision_weights),
                term("kv", Tier::Device(0), config.slots_bytes()),
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

    #[test]
    fn long_context_slot_sizing_matches_abi_estimates() {
        // 32K slot at 3 cycles (extra_runs = 24): ~3.16 GB.
        let s32k = slot_bytes(32_768, 24);
        assert_eq!(s32k, 3_157_950_464);

        // 400K slot (409,600) + attn scratch: ~28.1 GB.
        let total_400k = slot_bytes(409_600, 24) + attn_scratch_bytes(409_600);
        assert!(
            (27_500_000_000..=30_000_000_000).contains(&total_400k),
            "400K={total_400k}"
        );

        // 512K slot (524,288) + attn scratch: ~36.0 GB.
        let total_512k = slot_bytes(524_288, 24) + attn_scratch_bytes(524_288);
        assert!(
            (35_000_000_000..=38_000_000_000).contains(&total_512k),
            "512K={total_512k}"
        );
    }

    #[test]
    fn limits_pass_abi_check_for_512k_slot() {
        let cfg = Config {
            model_dir: PathBuf::from("unused"),
            slots: vec![524_288],
            max_block: 512,
            extra_runs: 24,
        };
        let lim = limits(&cfg, 18_881_368_064, 636_825_472);
        lim.check().unwrap();
        let nums = identity_numbers(&cfg);
        assert_eq!(nums["quant"], QUANT_MODE);
        assert_eq!(nums["state"], "2");
    }
}
