//! The numbers of Inkling 2, the config of the engine, its limits, and the rules of its slots. None of this needs a
//! card, so the host tests cover it (`docs/new-engine.md` section 2).
//!
//! `Model::parse` reads `config.json` and refuses each value that the ops of the engine do not run (engine-abi "The
//! first Inkling 2 binding", "The host spec"). The plan reads `config.json` and the tensor headers, and no weight
//! bytes (spec 2.4).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

use nagaya_abi::{
    Capability, ChainLimits, CoverageSpan, Determinism, Dtype, FeedRule, Inputs, Kind, Lifetime, Limits, MandatoryTerm,
    MemoryLimits, PartId, PartKind, PartLimits, Segment, SegmentInputs, SlotLimit, State, Tier,
};
use nagaya_mk_bmg::protocol::{MAX_TASKS_PER_CARD, NUM_SCOREBOARD_EVENTS};
use nagaya_mk_bmg::safetensors::{self, MappedSafeTensors, TensorMeta};
use nagaya_mk_bmg::tasks;

use crate::host::ops::{
    CONV_TAPS, D_REL, EPS, EXPERTS, GATE_RANK, GATE_SOFTCAP, HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS,
    LOG_ALPHA, LOG_FLOOR, LOGIT_SOFTCAP, QK_CLIP, REL_CLIP, REL_EXTENT, ROUTE_SCALE, TOP_K, UP_SOFTCAP,
};
use crate::host::weights::{BLOCK, CYCLE_END, CYCLE_START, LAYERS, LOGITS, NVFP4_LAYERS, VOCAB, is_full};

/// The engine kind in a placement profile (`AGENTS.md` "Layout": kind, model, target).
pub const KIND: &str = "mk-inkling2-bmg";

/// The family library that the engine needs (spec 11.3).
pub const FAMILY: &str = "inkling2";

/// The file of the weights in the model folder.
pub const WEIGHTS_FILE: &str = "model-00001-of-00001.safetensors";

/// The feed block when the config names none, and the largest feed block. The spill of a ring holds one block.
pub const DEFAULT_MAX_BLOCK: usize = 512;
pub const MAX_BLOCK: usize = 512;

/// The positions of a slot when the config gives a count of slots.
pub const SLOT_TOKENS: usize = 32_768;

/// The most positions of one slot: `model_max_length` of the config.
pub const MAX_SLOT_TOKENS: usize = 1_048_576;

/// The longest quantum in milliseconds (D8). It is a budget, not a measurement: a card test measures it later.
pub const MAX_QUANTUM_MS: u32 = 2_000;

/// The most segments in a pass. A batch of decode rows comes later.
pub const MAX_SEGMENTS: u32 = 1;

/// The most logit rows of one pass.
pub const MAX_LOGIT_ROWS: usize = 16;

/// A sliding layer reads the keys of the last `WINDOW` positions.
pub const WINDOW: usize = 1024;

/// A sliding layer of a slot longer than this keeps its keys and values in a ring of this many rows. The row of a key
/// is its position modulo the ring. A block of 512 and the window before it span at most 1,535 positions.
pub const RING_ROWS: usize = 2048;

/// The rows past a ring that a block may write before a copy moves them to the start, as for LFM3.
pub const RING_SPILL: usize = 1024;

/// The layers of one cycle.
pub const CYCLE_LAYERS: usize = CYCLE_END - CYCLE_START;

/// The cycles of a pass. Mini is post-trained at 3 cycles. `limits().cycles` and `segment.cycles` come in milestone
/// 35 step 7; until then each pass runs this count.
pub const DEFAULT_CYCLES: usize = 3;

/// The reserve of each slot when the config names none: the runs of cycles 2 and 3.
pub const DEFAULT_EXTRA_RUNS: usize = (DEFAULT_CYCLES - 1) * CYCLE_LAYERS;

/// The largest reserve: the runs of cycles 2 to 5. The research model is trained at 1 to 5 cycles.
pub const MAX_EXTRA_RUNS: usize = 4 * CYCLE_LAYERS;

/// The values of one row of a conv part: the inputs of the key, the value, the attention output and the expert
/// output convolutions, in that order.
pub const CONV_ROW: usize = 2 * KV_HEADS * HEAD_DIM + 2 * HIDDEN;

/// The rows of a conv part: the 3 last inputs, oldest first.
pub const CONV_ROWS: usize = CONV_TAPS - 1;

/// The rows of the head on the card: the 200,058 logits and 6 rows of zeros. `OP_MATMUL` reads whole tiles of 16 rows
/// (engine-abi "The ring builder", "The head and the sinks").
pub const HEAD_ROWS: usize = LOGITS.next_multiple_of(16);

/// The rows of the sink projection on the card: its 24 rows and 8 rows of zeros.
pub const SINK_ROWS: usize = HEADS.next_multiple_of(16);

/// The bytes of a staging buffer of the host, as for LFM3.
pub const STAGING_BYTES: usize = 64 << 20;

const _: () = assert!(
    RING_ROWS.is_multiple_of(32)
        && RING_SPILL >= MAX_BLOCK
        && WINDOW + MAX_BLOCK + 32 <= RING_ROWS
        && SLOT_TOKENS > RING_ROWS
        && CONV_ROW == 6656
        && DEFAULT_EXTRA_RUNS == 24
);

/// The two models of the kind. They share every file but `config.json` (milestone 35 step 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// `Inkling-2-Mini`: `loop_config.is_varloop` is false.
    Mini,
    /// `TML-d24-varloop-research`: `loop_config.is_varloop` is true.
    Research,
}

/// The value that the engine needs at one key of `config.json`.
enum Want {
    Text(&'static str),
    Count(usize),
    /// A number, compared as the `f32` that the ops use.
    Number(f32),
    Flag(bool),
    Counts(Vec<usize>),
}

impl Want {
    fn holds(&self, v: &serde_json::Value) -> bool {
        match self {
            Want::Text(t) => v.as_str() == Some(t),
            Want::Count(n) => v.as_u64() == Some(*n as u64),
            Want::Number(x) => v.as_f64().is_some_and(|f| f as f32 == *x),
            Want::Flag(b) => v.as_bool() == Some(*b),
            Want::Counts(list) => v.as_array().is_some_and(|a| {
                a.len() == list.len() && a.iter().zip(list).all(|(v, n)| v.as_u64() == Some(*n as u64))
            }),
        }
    }

    fn show(&self) -> String {
        match self {
            Want::Text(t) => format!("\"{t}\""),
            Want::Count(n) => n.to_string(),
            Want::Number(x) => x.to_string(),
            Want::Flag(b) => b.to_string(),
            Want::Counts(list) => format!("{list:?}"),
        }
    }
}

/// The value at a key path such as `text_config.loop_config.core_start`.
fn at<'a>(root: &'a serde_json::Value, path: &str) -> &'a serde_json::Value {
    path.split('.').fold(root, |v, key| &v[key])
}

/// The keys of `config.json` that the ops read, with the values that they run. The research model differs only in
/// `is_varloop`, which `Model::parse` reads.
fn wanted() -> Vec<(&'static str, Want)> {
    use Want::*;
    let t = |key: &'static str, want: Want| (key, want);
    vec![
        t("model_type", Text("inkling2_loop")),
        t("text_config.model_type", Text("inkling2_loop_text")),
        t("text_config.model_max_length", Count(MAX_SLOT_TOKENS)),
        t("text_config.hidden_size", Count(HIDDEN)),
        t("text_config.num_hidden_layers", Count(LAYERS)),
        t("text_config.vocab_size", Count(VOCAB)),
        t("text_config.unpadded_vocab_size", Count(LOGITS)),
        t("text_config.num_attention_heads", Count(HEADS)),
        t("text_config.num_key_value_heads", Count(KV_HEADS)),
        t("text_config.head_dim", Count(HEAD_DIM)),
        t("text_config.swa_num_attention_heads", Count(HEADS)),
        t("text_config.swa_num_key_value_heads", Count(KV_HEADS)),
        t("text_config.swa_head_dim", Count(HEAD_DIM)),
        t("text_config.d_rel", Count(D_REL)),
        t("text_config.rel_extent", Count(REL_EXTENT)),
        t("text_config.q_bias", Flag(false)),
        t("text_config.o_bias", Flag(false)),
        t("text_config.log_scaling_n_floor", Number(LOG_FLOOR)),
        t("text_config.log_scaling_alpha", Number(LOG_ALPHA)),
        t("text_config.rms_norm_eps", Number(EPS)),
        t("text_config.use_value_norm_no_gamma", Flag(true)),
        t("text_config.qk_l2_clip_threshold", Number(QK_CLIP)),
        t("text_config.rel_l2_clip_threshold", Number(REL_CLIP)),
        t(
            "text_config.local_layer_ids",
            Counts((0..LAYERS).filter(|l| !is_full(*l)).collect()),
        ),
        t("text_config.sliding_window_size", Count(WINDOW)),
        t("text_config.sconv_kernel_size", Count(CONV_TAPS)),
        t("text_config.final_logit_softcapping", Number(LOGIT_SOFTCAP)),
        t("text_config.n_layers_per_attn_res_block", Count(BLOCK)),
        t("text_config.attn_res_normalize_values", Flag(true)),
        t("text_config.n_routed_experts", Count(EXPERTS)),
        t("text_config.num_experts_per_tok", Count(TOP_K)),
        t("text_config.n_shared_experts", Count(0)),
        t("text_config.intermediate_size", Count(INTERMEDIATE)),
        t("text_config.mlp_gate_softcap", Number(GATE_SOFTCAP)),
        t("text_config.mlp_up_softcap", Number(UP_SOFTCAP)),
        t("text_config.route_scale", Number(ROUTE_SCALE)),
        t("text_config.use_gate_bias", Flag(true)),
        t("text_config.loop_config.core_start", Count(CYCLE_START)),
        t("text_config.loop_config.core_end", Count(CYCLE_END)),
        t("text_config.loop_config.num_passes", Count(DEFAULT_CYCLES)),
        t("text_config.loop_config.gate_rank", Count(GATE_RANK)),
    ]
}

impl Model {
    /// Read the model from the text of `config.json`. A value that the ops do not run is an error that names the key
    /// and the value.
    pub fn parse(text: &str) -> Result<Model, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        if root["architectures"][0] != "Inkling2LoopForConditionalGeneration" {
            return Err(format!(
                "config.json has architectures {}; this engine runs Inkling2LoopForConditionalGeneration",
                root["architectures"]
            ));
        }
        for (key, want) in wanted() {
            let value = at(&root, key);
            if !want.holds(value) {
                return Err(format!(
                    "config.json has {key} {value}, and the Inkling 2 engine runs {}. The folder holds another model",
                    want.show()
                ));
            }
        }
        let key = "text_config.loop_config.is_varloop";
        match at(&root, key).as_bool() {
            Some(false) => Ok(Model::Mini),
            Some(true) => Ok(Model::Research),
            None => Err(format!("config.json has {key} {}, not true or false", at(&root, key))),
        }
    }

    /// The name of the model in the identity, as the oracle files name it.
    pub fn name(self) -> &'static str {
        match self {
            Model::Mini => "inkling2-mini",
            Model::Research => "inkling2-varloop",
        }
    }

    /// Read the model from `config.json` of the folder `dir`.
    pub fn read(dir: &Path) -> Result<Model, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Model::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// One run of a pass: the layer whose weights it takes, its cycle, and the state of the slot that it reads and
/// writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub layer: usize,
    /// The cycle, from 1, of a run of layers 6 to 17. `None` for the layers outside the cycles.
    pub cycle: Option<usize>,
    /// The state of the slot: the layer for a run outside the cycles or in cycle 1, else `LAYERS + i` for entry `i`
    /// of the reserve.
    pub state: usize,
}

/// The runs of a pass of `cycles` cycles, in order: layers 0 to 5, each cycle of layers 6 to 17, then layers 18 to
/// 23.
pub fn runs(cycles: usize) -> Vec<Run> {
    let outside = |layer| Run {
        layer,
        cycle: None,
        state: layer,
    };
    let inner = (1..=cycles).flat_map(|c| {
        (CYCLE_START..CYCLE_END).map(move |layer| Run {
            layer,
            cycle: Some(c),
            state: if c == 1 {
                layer
            } else {
                LAYERS + (c - 2) * CYCLE_LAYERS + layer - CYCLE_START
            },
        })
    });
    (0..CYCLE_START)
        .map(outside)
        .chain(inner)
        .chain((CYCLE_END..LAYERS).map(outside))
        .collect()
}

/// The layer of state `s` of a slot: the layer itself, or the layer of entry `s - LAYERS` of the reserve.
pub fn state_layer(s: usize) -> usize {
    match s.checked_sub(LAYERS) {
        None => s,
        Some(i) => CYCLE_START + i % CYCLE_LAYERS,
    }
}

/// The name of state `s` in the name of a part: `ll` for a layer, `xii` for entry `ii` of the reserve.
fn state_name(s: usize) -> String {
    match s.checked_sub(LAYERS) {
        None => format!("{s:02}"),
        Some(i) => format!("x{i:02}"),
    }
}

/// The ring rows of a sliding run in a slot of `seq` positions, or 0 when its rows are positions.
pub fn ring_of(seq: usize) -> usize {
    if seq > RING_ROWS { RING_ROWS } else { 0 }
}

/// True when a state keeps a ring in a slot of `seq` positions: a sliding layer in a slot longer than the ring.
pub fn ringed(seq: usize, s: usize) -> bool {
    !is_full(state_layer(s)) && ring_of(seq) > 0
}

/// The rows of each key plane and each value plane of state `s` in a slot of `seq` positions. A sliding state holds a
/// ring on the card in a slot of any size, because the attention applies the window only to a ring. In a slot of
/// `RING_ROWS` positions or fewer the ring does not wrap, so the state still holds every position (`ringed`).
pub fn plane_rows(seq: usize, s: usize) -> usize {
    if is_full(state_layer(s)) {
        seq
    } else {
        RING_ROWS + RING_SPILL
    }
}

/// The `f32` values of one key row or one value row: 8 heads of 128.
pub const KV_ROW: usize = KV_HEADS * HEAD_DIM;

/// One part of the held state (engine-abi "Inkling 2", "The parts").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The keys of a state.
    K(usize),
    /// The values of a state.
    V(usize),
    /// The conv state of a state: a fixed part.
    Conv(usize),
}

impl Part {
    /// The state of the slot that holds the part.
    pub fn state(self) -> usize {
        match self {
            Part::K(s) | Part::V(s) | Part::Conv(s) => s,
        }
    }
}

/// The parts in state order: `k.<ll>`, `v.<ll>` and `conv.<ll>` for each layer, then the same for each entry of a
/// reserve of `extra` runs, as `k.x<ii>`. The part id is the index in this list.
pub fn part_list(extra: usize) -> Vec<Part> {
    (0..LAYERS + extra)
        .flat_map(|s| [Part::K(s), Part::V(s), Part::Conv(s)])
        .collect()
}

/// The parts of a state, as the ABI declares them. A full layer holds cells, a sliding layer holds a ring, and each
/// state holds a fixed conv part of 3 rows of 6,656 values, oldest first.
pub fn parts(extra: usize) -> Vec<PartLimits> {
    part_list(extra)
        .into_iter()
        .map(|part| {
            let s = part.state();
            let row = if is_full(state_layer(s)) {
                PartKind::Cells
            } else {
                PartKind::Ring
            };
            let (prefix, kind, shape) = match part {
                Part::K(_) => ("k", row, vec![KV_HEADS as u64, HEAD_DIM as u64]),
                Part::V(_) => ("v", row, vec![KV_HEADS as u64, HEAD_DIM as u64]),
                Part::Conv(_) => ("conv", PartKind::Fixed, vec![CONV_ROWS as u64, CONV_ROW as u64]),
            };
            PartLimits {
                name: format!("{prefix}.{}", state_name(s)),
                kind,
                dtype: Dtype::F32,
                shape,
                block_format: None,
                stage: 0,
            }
        })
        .collect()
}

/// The bytes of a device allocation of `bytes`: at least 64, in steps of 64.
pub fn rounded(bytes: usize) -> usize {
    bytes.max(64).next_multiple_of(64)
}

/// The device bytes of one slot of `seq` positions with a reserve of `extra` runs: the key and value planes and the
/// conv part of each state.
pub fn slot_bytes(seq: usize, extra: usize) -> u64 {
    (0..LAYERS + extra)
        .map(|s| (rounded(2 * plane_rows(seq, s) * KV_ROW * 4) + rounded(CONV_ROWS * CONV_ROW * 4)) as u64)
        .sum()
}

/// The config of one engine. The factory of the process reads it from the profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The model folder: `config.json`, the weights, and the files of the family.
    pub model_dir: PathBuf,
    /// The positions of each slot, from short to long. Thus the lowest slot that holds a job is the shortest.
    pub slots: Vec<usize>,
    /// The tokens of one block, and the feed block (engine-abi D11).
    pub max_block: usize,
    /// The entries of the reserve of each slot: the runs of the cycles after the first.
    pub extra_runs: usize,
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
        if self.slots.is_empty() {
            return Err("the config names no slot; give `slots`, a count of slots or a list of positions".to_owned());
        }
        if let Some(seq) = self.slots.iter().find(|seq| **seq == 0 || **seq > MAX_SLOT_TOKENS) {
            return Err(format!(
                "a slot of {seq} positions is outside 1 to {MAX_SLOT_TOKENS}, the positions of the model"
            ));
        }
        if !self.slots.is_sorted() {
            return Err(format!(
                "the slots {:?} are not in order; list them from short to long, so that a job takes the shortest \
                 slot that holds it",
                self.slots
            ));
        }
        if self.extra_runs < DEFAULT_EXTRA_RUNS
            || self.extra_runs > MAX_EXTRA_RUNS
            || !self.extra_runs.is_multiple_of(CYCLE_LAYERS)
        {
            return Err(format!(
                "extra_runs is {}; give {DEFAULT_EXTRA_RUNS}, 36 or {MAX_EXTRA_RUNS}. Each cycle after the first \
                 takes {CYCLE_LAYERS} runs, and a pass runs {DEFAULT_CYCLES} cycles",
                self.extra_runs
            ));
        }
        Ok(())
    }

    /// The positions of each slot, in slot order.
    pub fn seqs(&self) -> Vec<usize> {
        self.slots.clone()
    }

    /// The positions of the longest slot. The scratch of the attention holds its keys.
    pub fn max_seq(&self) -> usize {
        self.slots.iter().copied().max().unwrap_or(0)
    }

    /// The device bytes of all the slots.
    pub fn slots_bytes(&self) -> u64 {
        self.slots.iter().map(|seq| slot_bytes(*seq, self.extra_runs)).sum()
    }
}

/// The slot classes of the config: each run of equal sizes is one class.
pub fn slot_limits(slots: &[usize], extra: usize) -> Vec<SlotLimit> {
    let mut classes: Vec<SlotLimit> = Vec::new();
    for &seq in slots {
        match classes.last_mut() {
            Some(class) if class.tokens == seq as u64 => class.count += 1,
            _ => classes.push(SlotLimit {
                bytes: slot_bytes(seq, extra),
                count: 1,
                tokens: seq as u64,
            }),
        }
    }
    classes
}

/// The buffers of a block of `n` tokens, with their counts of 4-byte values, in the order of `model::Bufs`. Each
/// follows a name of "The ring of one run" in engine-abi, and `model::Bufs::new` checks the names.
pub fn block_buffers(n: usize) -> Vec<(&'static str, usize)> {
    let q = HEADS * HEAD_DIM;
    let decision = std::mem::size_of::<nagaya_mk_bmg::protocol::MoERouteDecision>() / 4;
    vec![
        ("ids", n),
        ("sink ids", 3 * n),
        ("x", n * HIDDEN),
        ("B", 4 * n * HIDDEN),
        ("P", n * HIDDEN),
        ("y", n * HIDDEN),
        ("x after layer 5", n * HIDDEN),
        ("a_q, a_k, a_v", 3 * n * HIDDEN),
        ("m", 4 * n * HIDDEN),
        ("h_q, h_k, h_v", 3 * n * HIDDEN),
        ("query", n * q),
        ("bias states", n * HEADS * D_REL),
        ("sinks of the matmul", n * SINK_ROWS),
        ("sinks", n * HEADS),
        ("output gate", n * q),
        ("key input", n * (KV_ROW + KV_HEADS * CONV_TAPS)),
        ("value input", n * (KV_ROW + KV_HEADS * CONV_TAPS)),
        ("key", n * KV_ROW),
        ("value", n * KV_ROW),
        ("attention", n * q),
        ("o", n * HIDDEN),
        ("e", n * HIDDEN),
        ("norm of e", n * HIDDEN),
        ("router", n * EXPERTS),
        ("decisions", n * decision),
        ("plan", nagaya_mk_bmg::protocol::MOE_PLAN_BYTES.div_ceil(4)),
        ("gathered rows", n * TOP_K * HIDDEN),
        ("activations", n * TOP_K * INTERMEDIATE),
        ("down rows", n * TOP_K * HIDDEN),
        ("f", n * HIDDEN),
        ("logits", MAX_LOGIT_ROWS * HEAD_ROWS),
    ]
}

/// The device bytes of the buffers of a block of `max_block` tokens.
pub fn bufs_bytes(max_block: usize) -> u64 {
    block_buffers(max_block)
        .iter()
        .map(|(_, floats)| rounded(floats * 4) as u64)
        .sum()
}

/// The floats of the scratch of the ring for slots of up to `max_seq` positions: the scratch of `OP_INKLING2_ATTN` for
/// a block of `MAX_BLOCK` positions.
pub fn scratch_floats(max_seq: usize) -> usize {
    tasks::inkling2_attn_scratch_floats(MAX_BLOCK, max_seq)
}

/// The device bytes of the ring of one card: the scoreboard, the scratch, the wake buffer and the device copy of the
/// tasks.
fn ring_device_bytes(max_seq: usize) -> u64 {
    (NUM_SCOREBOARD_EVENTS * 4 + scratch_floats(max_seq) * 4 + 128 + MAX_TASKS_PER_CARD * 64) as u64
}

/// The host bytes of the ring and of the copies: the tasks, the statistics and the staging buffer.
pub fn pinned_bytes() -> u64 {
    (MAX_TASKS_PER_CARD * 64 + 32 * 64 + STAGING_BYTES) as u64
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

/// The limits for a config, with the device bytes of the weights.
pub fn limits(config: &Config, weights: u64) -> Limits {
    let work = bufs_bytes(config.max_block) + ring_device_bytes(config.max_seq());
    Limits {
        // A drop frees the weights, the slots, the buffers and the context of the card.
        capabilities: vec![
            Capability::Parts,
            Capability::Slots,
            Capability::Coverage,
            Capability::Close,
        ],
        parts: parts(config.extra_runs),
        inputs: Inputs::Tokens,
        devices: vec![0],
        stages: Vec::new(),
        max_segments: MAX_SEGMENTS,
        max_pass_inputs: config.max_block as u32,
        max_logit_rows: MAX_LOGIT_ROWS as u32,
        slots: slot_limits(&config.seqs(), config.extra_runs),
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory: vec![
                term("weights", Tier::Device(0), weights),
                term("kv", Tier::Device(0), config.slots_bytes()),
                term("work", Tier::Device(0), work),
                term("pinned", Tier::Host, pinned_bytes()),
            ],
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
        // The tap `residual` waits for the plan of its buffer (milestone 35 step 8).
        taps: Vec::new(),
        checkpoint_grain: None,
        residual_row_bytes: 0,
        control: None,
        feed: Some(FeedRule {
            block: config.max_block as u64,
        }),
        // The engine runs cycles, not layer loops (engine-abi "Inkling 2"). The count of a call comes in step 7.
        layer_loops: None,
        cycles: None,
    }
}

/// The bytes of the slab of one expert matrix on the card (`repack::expert_slab`): the codes, the scales, and one
/// `f32` factor of each expert for NVFP4.
pub fn expert_slab_bytes(experts: usize, rows: usize, cols: usize, mx: bool) -> usize {
    let (row_bytes, row_scales, factors) = if mx {
        (cols, cols / 32, 0)
    } else {
        (cols / 2, cols / 16, 4 * experts)
    };
    experts * rows * (row_bytes + row_scales) + factors
}

/// The tensor `name` of the header, which must have `dtype` and `shape`.
fn tensor<'a>(
    tensors: &'a HashMap<String, TensorMeta>,
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
) -> Result<&'a TensorMeta, String> {
    let meta = tensors
        .get(name)
        .ok_or_else(|| format!("the checkpoint holds no tensor {name}. The file is not Inkling 2, or it is damaged"))?;
    if meta.dtype != dtype || meta.shape != shape {
        return Err(format!(
            "the tensor {name} is {:?} {:?}, and the engine needs {dtype:?} {shape:?}. The file is not Inkling 2",
            meta.dtype, meta.shape
        ));
    }
    Ok(meta)
}

/// The device bytes of the weights from the table of tensors of the checkpoint. Each expert matrix is one slab, the
/// embedding keeps its first `LOGITS` rows, the head holds `HEAD_ROWS` rows, each sink projection `SINK_ROWS` rows, and
/// each other tensor of `model.llm.` keeps its bytes. The
/// tensors of `model.visual.` wait for images (milestone 35 step 9). Any other tensor is refused.
pub fn weights_bytes_of(tensors: &HashMap<String, TensorMeta>) -> Result<u64, String> {
    use safetensors::Dtype::{BF16, F8E4M3, F32, U8};
    let mut total = 0usize;
    let mut read: HashSet<String> = HashSet::new();
    let mut need = |name: String, dtype, shape: &[usize]| -> Result<(), String> {
        tensor(tensors, &name, dtype, shape)?;
        read.insert(name);
        Ok(())
    };
    for l in 0..LAYERS {
        let p = |name: &str| format!("model.llm.layers.{l}.mlp.experts.{name}");
        let mx = l >= NVFP4_LAYERS;
        for (w, rows, cols) in [("w13", 2 * INTERMEDIATE, HIDDEN), ("w2", HIDDEN, INTERMEDIATE)] {
            if mx {
                // E4M3 values, and one UE8M0 scale in a `u8` for each 32 values.
                need(p(&format!("{w}_weight")), F8E4M3, &[EXPERTS, rows, cols])?;
                need(p(&format!("{w}_weight_scale")), U8, &[EXPERTS, rows, cols / 32])?;
            } else {
                // Two E2M1 codes in a `u8`, one E4M3 scale for each 16 values, and the two numbers of the factor.
                need(p(&format!("{w}_weight")), U8, &[EXPERTS, rows, cols / 2])?;
                need(p(&format!("{w}_weight_scale")), F8E4M3, &[EXPERTS, rows, cols / 16])?;
                need(p(&format!("{w}_weight_amax")), F32, &[EXPERTS])?;
                need(p(&format!("{w}_weight_outer_scale_denominator")), F32, &[1])?;
            }
            total += rounded(expert_slab_bytes(EXPERTS, rows, cols, mx));
        }
    }
    // The embedding holds its first LOGITS rows, and the head holds HEAD_ROWS rows.
    for (name, rows) in [
        ("model.llm.embed.weight", LOGITS),
        ("model.llm.unembed.weight", HEAD_ROWS),
    ] {
        need(name.to_owned(), BF16, &[VOCAB, HIDDEN])?;
        total += rounded(rows * HIDDEN * 2);
    }
    // Each sink projection holds SINK_ROWS rows.
    for l in 0..LAYERS {
        need(
            format!("model.llm.layers.{l}.attn.attn_sink_proj.weight"),
            BF16,
            &[HEADS, HIDDEN],
        )?;
        total += rounded(SINK_ROWS * HIDDEN * 2);
    }
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    for name in names {
        if read.contains(name) || name.starts_with("model.visual.") {
            continue;
        }
        if !name.starts_with("model.llm.") || name.contains(".mlp.experts.") {
            return Err(format!(
                "the checkpoint holds the tensor {name}, which the Inkling 2 engine does not know. The file is not \
                 Inkling 2"
            ));
        }
        let meta = &tensors[name];
        total += rounded(meta.data_end - meta.data_start);
    }
    Ok(total as u64)
}

/// The device bytes of the weights in the folder `dir`, from the tensor headers.
pub fn weights_bytes(dir: &Path) -> Result<u64, String> {
    let map = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
    weights_bytes_of(&map.view()?.tensors)
}

/// The limits of a config, from `config.json` and the tensor headers. It reads no weight bytes (spec 2.4).
pub fn plan(config: &Config) -> Result<Limits, String> {
    config.check()?;
    Model::read(&config.model_dir)?;
    let weights = weights_bytes(&config.model_dir)?;
    Ok(limits(config, weights))
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
}

impl Block {
    /// True when the pass starts at position 0. Then each run starts from conv states of zeros, as the reference
    /// does (engine-abi "Inkling 2", "Position 0 and no way back").
    pub fn zero_conv(&self) -> bool {
        self.pos0 == 0
    }
}

/// The token count and the writes of each slot, and the rules of the part calls. The engine applies the device side.
///
/// A slot never goes back: the conv state holds only the last inputs, so a pass starts at the token count. A write
/// puts the token count at 0, and `set_n` takes a count only when every part got rows that end at it (engine-abi "The
/// first LFM3 binding", "No way back").
#[derive(Clone, Debug)]
pub struct Held {
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
}

/// The bytes of one row of a row part.
pub const ROW_BYTES: u64 = (KV_ROW * 4) as u64;

/// The bytes of a conv part.
pub const CONV_BYTES: u64 = (CONV_ROWS * CONV_ROW * 4) as u64;

impl Held {
    /// The rules of slots of `seqs` positions, each with a reserve of `extra` runs.
    pub fn new(seqs: &[usize], extra: usize) -> Self {
        Self {
            list: part_list(extra),
            seqs: seqs.to_vec(),
            n: vec![0; seqs.len()],
            writes: vec![BTreeMap::new(); seqs.len()],
            ring_from: vec![0; seqs.len()],
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

    /// The first position that a row part of state `s` holds in `slot`. A cells part, or a sliding state of a slot
    /// with no ring, holds every position from 0. A ring holds the last `RING_ROWS` positions, and after a restore no
    /// position before the restored rows.
    pub fn base(&self, slot: usize, s: usize) -> u64 {
        if ringed(self.seqs[slot], s) {
            self.n[slot]
                .saturating_sub(RING_ROWS as u64)
                .max(self.ring_from[slot])
                .min(self.n[slot])
        } else {
            0
        }
    }

    /// Check a pass before it changes anything, and give its block. The message says which rule breaks.
    ///
    /// - A pass has one segment of 1 to `max_block` tokens below `LOGITS`, with no rows and no `meta`.
    /// - The segment names no control item and no layer loop: the engine declares neither.
    /// - The positions run on from the token count of the slot. A slot does not go back.
    /// - `logit_rows` names at most `MAX_LOGIT_ROWS` positions of the block.
    pub fn check_pass(&self, segments: &[Segment], max_block: usize) -> Result<Block, String> {
        let [segment] = segments else {
            return Err(format!(
                "the pass has {} segments; the engine takes {MAX_SEGMENTS}",
                segments.len()
            ));
        };
        let slot = self
            .slot(segment.state)
            .ok_or_else(|| format!("the engine has {} slots, not {:?}", self.n.len(), segment.state))?;
        let SegmentInputs::Tokens(tokens) = &segment.inputs else {
            return Err("the segment has rows; the Inkling 2 engine takes tokens only, until images come".to_owned());
        };
        if !segment.meta.is_empty() {
            return Err("the segment has meta; the Inkling 2 engine takes tokens only, until images come".to_owned());
        }
        if !segment.control.is_empty() {
            return Err(format!(
                "the segment has {} control items; the Inkling 2 engine does not declare `control` yet",
                segment.control.len()
            ));
        }
        if segment.layer_loop.is_some() {
            return Err(
                "the segment names a layer loop; the Inkling 2 engine runs cycles, and it does not declare \
                 `layer_loops`"
                    .to_owned(),
            );
        }
        if let Some(count) = segment.cycles {
            return Err(format!(
                "the segment names a count of cycles, {count}; the engine runs its default of 3 cycles, and it does \
                 not declare `cycles` yet"
            ));
        }
        let n = tokens.len();
        if n == 0 || n > max_block {
            return Err(format!("the segment has {n} tokens; a block holds 1 to {max_block}"));
        }
        if let Some(bad) = tokens.iter().find(|token| **token as usize >= LOGITS) {
            return Err(format!(
                "token id {bad} is outside the vocabulary of {LOGITS} ids; the rows past it are padding"
            ));
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
                "slot {slot} holds {held} tokens, so the positions run from {held}; an Inkling 2 slot does not go \
                 back, because its conv state holds only the last inputs"
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
        Ok(Block {
            slot,
            pos0,
            tokens: tokens.clone(),
            logit_rows,
        })
    }

    /// Take a pass that ran: the slot holds the block's positions, and the writes before it are gone.
    pub fn ran(&mut self, block: &Block) {
        self.n[block.slot] = (block.pos0 + block.tokens.len()) as u64;
        self.writes[block.slot].clear();
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

    /// Check a read of `n` rows from `from` of a row part, and give its slot and part.
    pub fn check_read(&self, state: State, id: PartId, from: u64, n: u64, len: usize) -> Result<(usize, Part), String> {
        let (slot, part) = self.row_part(state, id)?;
        let base = self.base(slot, part.state());
        if len as u64 != n * ROW_BYTES || from < base || from + n > self.n[slot] {
            return Err(format!(
                "a read of rows {from} to {} of part {id} in {len} bytes does not fit: slot {slot} holds rows {base} to \
                 {} of {ROW_BYTES} bytes",
                from + n,
                self.n[slot],
            ));
        }
        Ok((slot, part))
    }

    /// Check a whole read of a part into `len` bytes, and give the first row for a row part.
    pub fn check_part_read(&self, state: State, id: PartId, len: usize) -> Result<(usize, Part, u64), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        let part = self.part(id).ok_or_else(|| format!("the engine has no part {id}"))?;
        match part {
            Part::Conv(_) if len as u64 == CONV_BYTES => Ok((slot, part, 0)),
            Part::Conv(_) => Err(format!("part {id} is a conv state of {CONV_BYTES} bytes, not {len}")),
            _ => {
                if !(len as u64).is_multiple_of(ROW_BYTES) {
                    return Err(format!("{len} bytes are not whole rows of {ROW_BYTES} bytes"));
                }
                let from = self.base(slot, part.state());
                self.check_read(state, id, from, len as u64 / ROW_BYTES, len)?;
                Ok((slot, part, from))
            }
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
        if len as u64 != n * ROW_BYTES || from + n > seq as u64 {
            return Err(format!(
                "rows {from} to {} of part {id} in {len} bytes do not fit slot {slot} of {seq} positions",
                from + n
            ));
        }
        let kept = if ringed(seq, part.state()) {
            (from + n).saturating_sub(RING_ROWS as u64).max(from)..from + n
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
            Some(part @ Part::Conv(_)) if len as u64 == CONV_BYTES => Ok((slot, part)),
            Some(Part::Conv(_)) => Err(format!("part {id} is a conv state of {CONV_BYTES} bytes, not {len}")),
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
                "slot {slot} holds {} tokens and got no write; an Inkling 2 slot does not go back to {n}, because its \
                 conv state holds only the last inputs. Restore a snapshot instead",
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
                (Part::K(s) | Part::V(s), Some(rows)) if ringed(seq, s) => Some(rows.start),
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

    /// The positions that `bytes` of a part hold in a state, from the base of the part. A conv part holds none.
    pub fn coverage(&self, state: State, id: PartId, bytes: &[u8]) -> CoverageSpan {
        match (self.slot(state), self.part(id)) {
            (Some(slot), Some(part @ (Part::K(_) | Part::V(_)))) => {
                let start = self.base(slot, part.state());
                CoverageSpan {
                    part: id,
                    start,
                    end: start + bytes.len() as u64 / ROW_BYTES,
                }
            }
            _ => CoverageSpan {
                part: id,
                start: 0,
                end: 0,
            },
        }
    }

    /// Check that the positions of `span` fit `slot`. A cells part, and any state of a slot with no ring, needs every
    /// position from 0. A ring needs the window of `WINDOW` positions before the end.
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
        if !ringed(seq, part.state()) {
            if span.start != 0 {
                return Err(format!(
                    "part {} in a slot of {seq} positions needs every position from 0; it starts at {}",
                    span.part, span.start
                ));
            }
            return Ok(());
        }
        let need = span.end.saturating_sub(WINDOW as u64);
        if span.start > need {
            return Err(format!(
                "part {} needs the window of {WINDOW} positions before {}; it starts at {}",
                span.part, span.end, span.start
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
