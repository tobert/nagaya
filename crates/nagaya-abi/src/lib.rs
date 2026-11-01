//! The engine ABI: the `Engine` trait, its types, and a fake engine behind the feature `fake`.

use std::collections::BTreeMap;
use std::ops::Range;

/// One device in a device group.
pub type DeviceId = u32;

/// One slot in an engine that holds slots.
pub type SlotId = u32;

/// One part of a state object.
pub type PartId = usize;

/// What a part call operates on. v0 has slots only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
    /// A slot index.
    Slot(SlotId),
}

/// How a bit setting affects the reference bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitClass {
    /// The setting defines the reference bits.
    ReferenceDefining,
    /// The setting changes the bits by an approved approximation.
    Approx,
    /// The setting is a policy that does not change the bits.
    Policy,
}

/// One setting that changes the bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSetting {
    /// The setting name.
    pub name: String,
    /// The setting value.
    pub value: String,
    /// How the setting affects the bits.
    pub class: BitClass,
}

/// The identity of an engine and its weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// The model name.
    pub model: String,
    /// The digest of the checkpoint.
    pub checkpoint_digest: String,
    /// The name of the family library that the engine needs: `toy`, `qwen38` or `gemma4` (spec 11.3).
    pub family: String,
    /// The hash of the tokenizer.
    pub tokenizer_hash: String,
    /// The chat template.
    pub template: String,
    /// The engine's numbers and their sources.
    pub numbers: BTreeMap<String, String>,
    /// The plan table lineage.
    pub plan_table: String,
    /// The settings that change the bits.
    pub bit_settings: Vec<BitSetting>,
    /// The canary hash.
    pub canary_hash: String,
}

/// What a part is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartKind {
    /// A constant-size block that each pass replaces whole.
    Fixed,
    /// Rows indexed by position, only appended.
    Cells,
    /// Rows in a window that wraps.
    Ring,
    /// A part rebuilt from other parts.
    Derived,
}

/// The element type of a part. The variants are the native dtypes of D10 rule 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// A 32-bit float.
    F32,
    /// A 16-bit float.
    F16,
    /// A bfloat16.
    BF16,
    /// An unsigned byte.
    U8,
    /// A 16-bit unsigned integer.
    U16,
    /// A 32-bit unsigned integer.
    U32,
    /// A 64-bit unsigned integer.
    U64,
    /// A signed byte.
    I8,
    /// A 16-bit signed integer.
    I16,
    /// A 32-bit signed integer.
    I32,
    /// A 64-bit signed integer.
    I64,
    /// An 8-bit float with 4 exponent bits and 3 mantissa bits.
    F8E4M3,
    /// An 8-bit float with 5 exponent bits and 2 mantissa bits.
    F8E5M2,
}

impl Dtype {
    /// Every dtype, so that a test can visit each one.
    pub const ALL: [Dtype; 13] = [
        Dtype::F32,
        Dtype::F16,
        Dtype::BF16,
        Dtype::U8,
        Dtype::U16,
        Dtype::U32,
        Dtype::U64,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F8E4M3,
        Dtype::F8E5M2,
    ];

    /// The size of one element in bytes.
    pub fn element_bytes(self) -> u64 {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::F8E5M2 => 1,
            Dtype::F16 | Dtype::BF16 | Dtype::U16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::U64 | Dtype::I64 => 8,
        }
    }
}

/// One part that an engine declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartLimits {
    /// The part name.
    pub name: String,
    /// The part kind.
    pub kind: PartKind,
    /// The element type.
    pub dtype: Dtype,
    /// The whole shape for a fixed or derived part; the row shape for cells or ring.
    pub shape: Vec<u64>,
    /// The name of the block format, when the dtype is `U8` for a block format. `None` otherwise.
    pub block_format: Option<String>,
    /// The stage that holds the part (engine-abi D1). An engine with no stages gives 0.
    pub stage: u32,
}

impl PartLimits {
    /// True when the store keeps the part.
    pub fn is_stored(&self) -> bool {
        self.kind != PartKind::Derived
    }

    /// True when the shape is one row.
    pub fn is_row(&self) -> bool {
        matches!(self.kind, PartKind::Cells | PartKind::Ring)
    }
}

/// One slot size and how many slots have it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotLimit {
    /// The slot size in bytes.
    pub bytes: u64,
    /// The number of slots with this size.
    pub count: u32,
    /// The most tokens that a slot of this size holds.
    pub tokens: u64,
}

/// What an engine takes as input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inputs {
    /// Token ids only.
    Tokens,
    /// Rows only.
    Rows,
    /// Token ids and rows.
    TokensAndRows,
}

/// One stage of a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    /// The device that runs the stage.
    pub device: DeviceId,
    /// The layers on the device.
    pub layers: Range<u32>,
    /// True when the stage holds the head.
    pub head: bool,
}

/// Where a memory term lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Device memory.
    Device(DeviceId),
    /// Host memory.
    Host,
}

/// How long a memory term lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// The engine holds it.
    Engine,
    /// A context holds it.
    Context,
    /// One job at a time holds it.
    Loan,
}

/// The kind of quantum that uses a memory term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A read.
    Read,
    /// A prefill.
    Prefill,
    /// A decode.
    Decode,
    /// An embed.
    Embed,
    /// A copy.
    Copy,
    /// An image step.
    Step,
    /// A training micro-batch.
    Train,
}

/// One mandatory memory term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MandatoryTerm {
    /// The term name.
    pub name: String,
    /// Where it lives.
    pub tier: Tier,
    /// Its size in bytes.
    pub bytes: u64,
    /// How long it lives.
    pub lifetime: Lifetime,
    /// The quantum kinds that use it.
    pub kinds: Vec<Kind>,
}

/// One elastic memory term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElasticTerm {
    /// The term name.
    pub name: String,
    /// Where it lives.
    pub tier: Tier,
    /// The size of one unit in bytes.
    pub unit_bytes: u64,
    /// The minimum size in units (`unit_bytes` each).
    pub minimum: u64,
    /// The maximum size in units (`unit_bytes` each).
    pub maximum: u64,
    /// The quantum kinds that use it.
    pub kinds: Vec<Kind>,
}

/// The memory terms of an engine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryLimits {
    /// The mandatory terms.
    pub mandatory: Vec<MandatoryTerm>,
    /// The elastic terms.
    pub elastic: Vec<ElasticTerm>,
}

/// How the store may chain this engine's snapshots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainLimits {
    /// The deepest chain.
    pub max_depth: u32,
    /// The largest delta as a fraction of the whole.
    pub delta_fraction: f32,
}

/// The bits that an engine promises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Determinism {
    /// The same identity and inputs give the same bits in any process.
    Always,
    /// The same, because the plan table fixes every free choice.
    Pinned,
    /// The same bits within one process only.
    PerProcess,
}

/// A history that changes the bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDependence {
    /// The bits depend on which state is resident.
    Residency,
}

/// One tap and the conditions that it needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapLimit {
    /// The tap name.
    pub name: String,
    /// The conditions that the tap needs.
    pub conditions: Vec<String>,
}

/// One optional capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// Arena export and check.
    Arena,
    /// Context calls.
    Contexts,
    /// Slot calls.
    Slots,
    /// Part calls.
    Parts,
    /// Seal.
    Seal,
    /// Geometry key.
    Geometry,
    /// Coverage and fit.
    Coverage,
    /// Elastic resize.
    Resize,
    /// Poison check.
    Poison,
    /// The engine's pass check.
    PassCheck,
    /// Stage pass.
    Stages,
    /// IO submit and wait.
    Io,
    /// Telemetry.
    Telemetry,
    /// Training.
    Train,
    /// Close.
    Close,
    /// Speech: `speech_feed`, `speech_frames` and `speech_clear` (engine-abi "Speech").
    Speech,
    /// Images: `image_begin`, `image_step`, `image_decode` and `image_clear` (engine-abi "Image").
    Image,
    /// Embeddings: `embed` (engine-abi "Embedding").
    Embedding,
}

/// The limits that an engine declares.
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// The capabilities that the engine has.
    pub capabilities: Vec<Capability>,
    /// The parts of a state object.
    pub parts: Vec<PartLimits>,
    /// What the engine takes as input.
    pub inputs: Inputs,
    /// The device group.
    pub devices: Vec<DeviceId>,
    /// The stages of a pass.
    pub stages: Vec<Stage>,
    /// The most segments in a pass.
    pub max_segments: u32,
    /// The most inputs in a pass.
    pub max_pass_inputs: u32,
    /// The most logit rows in a pass.
    pub max_logit_rows: u32,
    /// The slot sizes and counts.
    pub slots: Vec<SlotLimit>,
    /// The longest quantum in milliseconds.
    pub max_quantum_ms: u32,
    /// The memory terms.
    pub memory: MemoryLimits,
    /// The chain limits.
    pub chain: ChainLimits,
    /// True when segments share a pass with the bits of a pass alone.
    pub segments_exact: bool,
    /// The bits that the engine promises.
    pub determinism: Determinism,
    /// The history that changes the bits.
    pub bit_dependence: Vec<BitDependence>,
    /// The taps and their conditions.
    pub taps: Vec<TapLimit>,
    /// The token count at which an engine keeps a state that it can copy, or `None` when it keeps none (spec 6.13).
    pub checkpoint_grain: Option<u64>,
    /// The bytes of one residual row at a boundary between two stages (engine-abi D1). An engine with no stages
    /// gives 0.
    pub residual_row_bytes: u64,
    /// The feed rule, when the bits of the engine depend on the token count of a pass (engine-abi D11).
    pub feed: Option<FeedRule>,
}

/// The feed rule of an engine (engine-abi D11, spec 6.14).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedRule {
    /// The tokens of each pass from the start of a span. The last pass of a span holds the remainder.
    pub block: u64,
}

/// The name of the mandatory host term that holds the residual buffers (engine-abi D1).
pub const RESIDUAL_TERM: &str = "residual";

impl Limits {
    /// The rows of one residual buffer: `feed.block`, or `max_pass_inputs` for an engine with no feed rule.
    pub fn residual_rows(&self) -> u64 {
        self.feed.map_or(u64::from(self.max_pass_inputs), |feed| feed.block)
    }

    /// The bytes of the residual term: two buffers at each boundary between stages (engine-abi D1).
    pub fn residual_bytes(&self) -> u64 {
        let boundaries = self.stages.len().saturating_sub(1) as u64;
        2 * boundaries * self.residual_rows() * self.residual_row_bytes
    }

    /// Check the rules of the stages and the feed rule (engine-abi D1, D11). The message says which rule breaks.
    pub fn check(&self) -> Result<(), String> {
        if let Some(feed) = self.feed {
            if feed.block == 0 || feed.block > u64::from(self.max_pass_inputs) {
                return Err(format!(
                    "the feed block is {} tokens; it must be above 0 and at most max_pass_inputs ({})",
                    feed.block, self.max_pass_inputs
                ));
            }
            if self.segments_exact || self.checkpoint_grain.is_some() {
                return Err("an engine with a feed rule declares no segments_exact and no checkpoint grain".to_owned());
            }
        }
        let staged = self.capabilities.contains(&Capability::Stages);
        let stages = if staged { self.stages.len() } else { 1 };
        if let Some(part) = self.parts.iter().find(|part| part.stage as usize >= stages) {
            return Err(format!(
                "the part `{}` names stage {}, but the engine has {stages} stages",
                part.name, part.stage
            ));
        }
        if !staged {
            return Ok(());
        }
        if self.stages.len() < 2 {
            return Err(format!(
                "an engine with stages declares at least two, not {}",
                self.stages.len()
            ));
        }
        for (index, stage) in self.stages.iter().enumerate() {
            if self.stages[..index].iter().any(|other| other.device == stage.device) {
                return Err(format!(
                    "two stages run on device {}; each stage has its own device",
                    stage.device
                ));
            }
        }
        if self.capabilities.contains(&Capability::Seal) || self.checkpoint_grain.is_some() {
            return Err("an engine with stages declares no seal and no checkpoint grain in v0".to_owned());
        }
        if self.residual_row_bytes == 0 {
            return Err("an engine with stages declares residual_row_bytes above 0".to_owned());
        }
        let expected = self.residual_bytes();
        let term = self
            .memory
            .mandatory
            .iter()
            .find(|term| term.name == RESIDUAL_TERM)
            .ok_or_else(|| format!("an engine with stages declares the mandatory term `{RESIDUAL_TERM}`"))?;
        if term.tier != Tier::Host || term.lifetime != Lifetime::Engine || term.bytes != expected {
            return Err(format!(
                "the term `{RESIDUAL_TERM}` is {} bytes in {:?} for {:?}; it must be {expected} host bytes for the engine",
                term.bytes, term.tier, term.lifetime
            ));
        }
        Ok(())
    }
}

/// The result of a pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PassStatus {
    /// The pass ran.
    #[default]
    Ok,
    /// The pass changed a state and poisoned it.
    Poison,
    /// The pass ran past its bound.
    Timeout,
}

/// One segment in a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The state that the segment runs on.
    pub state: State,
    /// The inputs of the segment.
    pub inputs: SegmentInputs,
    /// The positions of the inputs.
    pub positions: Vec<u64>,
    /// Per-token metadata.
    pub meta: Vec<u8>,
    /// The outputs that the caller wants.
    pub outputs: OutputsWanted,
}

/// The inputs of one segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentInputs {
    /// Token ids.
    Tokens(Vec<u32>),
    /// Rows.
    Rows(Vec<u8>),
    /// Token ids and rows.
    TokensAndRows {
        /// The token ids.
        tokens: Vec<u32>,
        /// The rows.
        rows: Vec<u8>,
    },
}

/// The outputs that a segment wants.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OutputsWanted {
    /// The logit rows that the caller wants.
    pub logit_rows: Vec<u64>,
}

/// One tap request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tap {
    /// The tap name.
    pub name: String,
}

/// One output of a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    /// The output name.
    pub name: String,
    /// The element type of the bytes. Conformance compares outputs by integer bit pattern and rejects NaN and Inf
    /// (spec 8.3).
    pub dtype: Dtype,
    /// The output bytes.
    pub bytes: Vec<u8>,
}

/// The result of a pass that ran.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassOut {
    /// The pass status.
    pub status: PassStatus,
    /// The elapsed milliseconds.
    pub ms: u64,
    /// The outputs.
    pub outputs: Vec<Output>,
}

/// Why a pass did not give its outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassErrorKind {
    /// The engine refused the pass before it ran. Nothing changed.
    Refused,
    /// The pass ran and gave no outputs. The status says what it did to its contexts.
    Ran(PassStatus),
}

/// A pass that did not give its outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassError {
    /// Whether the pass ran, and what it did to its contexts.
    pub kind: PassErrorKind,
    /// Why the pass failed.
    pub message: String,
}

/// One position range held in one part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverageSpan {
    /// The part.
    pub part: PartId,
    /// The first position.
    pub start: u64,
    /// The position after the last.
    pub end: u64,
}

/// The result of a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartError {
    /// The engine refused the read and changed nothing.
    Refused,
    /// The read failed after a change, so the state is poisoned.
    Failed,
}

/// The result of a write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteResult {
    /// The write succeeded.
    Ok,
    /// The engine refused the write and changed nothing.
    Refused,
    /// The write failed after a change, so the state is poisoned.
    Failed,
}

/// A seal that did not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealError {
    /// The engine refused the seal and changed nothing.
    Refused,
    /// The seal failed after a change, so the state is poisoned.
    Failed,
}

/// The part capability.
pub trait Parts {
    /// Read a whole part into `out`.
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError>;

    /// Write a whole part from `bytes`.
    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult;

    /// Read `n` rows from `from` into `out`.
    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError>;

    /// Write `n` rows at `from` from `bytes`.
    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult;

    /// Set the number of rows.
    fn set_n(&mut self, state: State, n: u64) -> WriteResult;
}

/// The slot capability.
pub trait Slots {
    /// The highest row that the slot holds.
    fn slot_high(&self, slot: SlotId) -> u64;

    /// Clear the slot.
    fn slot_clear(&mut self, slot: SlotId) -> WriteResult;
}

/// The coverage capability.
pub trait Coverage {
    /// The positions that `bytes` hold for one part, on one state.
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan;

    /// True when the coverage fits the slot.
    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String>;
}

/// The seal capability.
pub trait Seal {
    /// Run the engine's boundary rule on a state.
    fn seal(&mut self, state: State) -> Result<(), SealError>;
}

/// A host buffer of residual rows between two stages (engine-abi D1). The nagaya owns it, so a stage never allocates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residual {
    row_bytes: usize,
    capacity: usize,
    rows: usize,
    bytes: Vec<u8>,
}

impl Residual {
    /// Make a buffer of `capacity` rows of `row_bytes` bytes. It holds no row.
    pub fn new(row_bytes: usize, capacity: usize) -> Self {
        Self {
            row_bytes,
            capacity,
            rows: 0,
            bytes: vec![0; row_bytes * capacity],
        }
    }

    /// The bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The most rows that the buffer holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The rows that the last stage wrote.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The bytes of the rows that the last stage wrote.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.rows * self.row_bytes]
    }

    /// Hold `rows` rows, and give their bytes to write. A count above the capacity is refused and changes nothing.
    pub fn write(&mut self, rows: usize) -> Result<&mut [u8], String> {
        if rows > self.capacity {
            return Err(format!(
                "a stage writes {rows} residual rows; the buffer holds {}",
                self.capacity
            ));
        }
        self.rows = rows;
        Ok(&mut self.bytes[..rows * self.row_bytes])
    }
}

/// What a stage pass gives (engine-abi D1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageOut {
    /// A stage before the last one wrote the residual.
    Residual {
        /// The pass status.
        status: PassStatus,
        /// The elapsed milliseconds.
        ms: u64,
    },
    /// The last stage gave the outputs of the pass.
    Last(PassOut),
}

/// The handle of one stage. It moves to the thread of its device, and it makes every call to that device.
pub trait StageHandle {
    /// Run this stage of one pass. A stage after the first reads `residual`, and a stage before the last writes it.
    fn stage_pass(&mut self, segments: &[Segment], residual: &mut Residual) -> Result<StageOut, PassError>;

    /// The part capability for the parts of this stage only. A part of another stage is refused.
    fn parts(&mut self) -> Option<&mut dyn Parts> {
        None
    }

    /// The slot capability for the parts of this stage only.
    fn slots(&mut self) -> Option<&mut dyn Slots> {
        None
    }

    /// The coverage capability for the parts of this stage only.
    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        None
    }
}

/// The stage capability (engine-abi D1).
pub trait Stages {
    /// Give one handle for each stage, in stage order. The first call gives them all, and a later call gives none.
    /// After it, `Engine::pass` refuses, and the `Engine` gives no parts, slots, coverage or seal.
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>>;
}

/// The strength of a check. Arenas and store entries use one enum (engine-abi "Capabilities").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckStrength {
    /// The check compared every byte.
    Full,
    /// The check compared some bytes.
    Spot,
    /// Nothing was checked.
    None,
}

impl CheckStrength {
    /// The word for the strength in a log line and an attribute.
    pub fn as_str(self) -> &'static str {
        match self {
            CheckStrength::Full => "full",
            CheckStrength::Spot => "spot",
            CheckStrength::None => "none",
        }
    }
}

/// One part of an arena, before the open: its device and its name (engine-abi "Arena calls in v0").
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArenaPart {
    /// The device of the part, as a group index (spec 2.4).
    pub device: usize,
    /// The name that the engine gives the part, for example `w15`.
    pub part: String,
}

/// One fd of an arena. The holder of the value owns the fd.
#[derive(Debug)]
pub struct ArenaFd {
    /// The device of the part, as a group index (spec 2.4).
    pub device: usize,
    /// The name that the engine gives the part.
    pub part: String,
    /// The fd of the memory that holds the part.
    pub fd: std::os::fd::OwnedFd,
}

/// The arena capability (spec 10.2).
pub trait Arena {
    /// Give a new fd for each part of the arena. The engine keeps its own fds, and the caller owns the new ones.
    fn arena_export(&mut self) -> Result<Vec<ArenaFd>, String>;
}

/// One position of a speech prompt (engine-abi "Speech"): an id of the text vocabulary and an id of the codec
/// vocabulary. Either can be absent. The family library makes the pairs; the engine never takes text (D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptPair {
    /// The id of the text vocabulary, or `None`.
    pub text: Option<u32>,
    /// The id of the codec vocabulary, or `None`.
    pub codec: Option<u32>,
}

/// One position of a speech context (engine-abi "Held speech state"): a pair of a text id and a codec id, or a frame
/// of codes that the engine feeds as a frame that it made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeechPosition {
    /// A text id and a codec id, either of which can be absent.
    Pair(PromptPair),
    /// The codes of one frame, one for each codebook.
    Frame(Vec<u32>),
}

/// The sampler of a speech request (engine-abi "Speech", spec 11.2). The same identity, prompt, sampler and seed give
/// the same codes and the same audio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeechSampler {
    /// The temperature.
    pub temperature: f32,
    /// The ids with the largest scores that stay; 0 keeps every id.
    pub top_k: u32,
    /// The repetition penalty of the first codebook.
    pub repetition_penalty: f32,
    /// The seed of the random stream.
    pub seed: u64,
}

/// What one `speech_frames` call made.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpeechOut {
    /// The frames that the call made.
    pub frames: u32,
    /// The codes of each frame, frame after frame, `codebooks` codes for each frame.
    pub codes: Vec<u32>,
    /// The audio of the frames: signed 16-bit samples, mono, at the rate of the engine.
    pub audio: Vec<i16>,
    /// True when the engine sampled the end of speech. The slot makes no more frames.
    pub end: bool,
}

/// The speech capability (engine-abi "Speech").
pub trait Speech {
    /// Feed positions after the state of the slot. With a sampler, the feed starts a new line: the frame counter of the
    /// random stream starts at 0 again, and the feed samples the line's first code. With none, it holds the rows and
    /// starts no line. A slot whose line has not ended refuses a feed.
    fn speech_feed(
        &mut self,
        slot: SlotId,
        positions: &[SpeechPosition],
        sampler: Option<&SpeechSampler>,
    ) -> Result<(), PassError>;

    /// Make at most `max` frames in the slot and decode their audio. It stops at the end of speech.
    fn speech_frames(&mut self, slot: SlotId, max: u32) -> Result<SpeechOut, PassError>;

    /// Free the state of the slot. A slot that holds no request is not a failure.
    fn speech_clear(&mut self, slot: SlotId);
}

/// The ids of one chunk of a prompt for an image engine (spec 11.9a): the start id, 75 tokens and the end id, padded.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageChunk {
    /// The 77 ids of the first text encoder.
    pub ids1: Vec<u32>,
    /// The 77 ids of the second text encoder.
    pub ids2: Vec<u32>,
    /// The weight of each id. The start, the end and the pad ids have weight 1.
    pub weights: Vec<f32>,
}

/// A sampler of an image job (spec 11.9a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSampler {
    /// `euler`.
    Euler,
    /// `euler a`.
    EulerAncestral,
    /// `dpm++ 2m karras`.
    DpmPp2mKarras,
    /// `dpm++ 2m sde karras`.
    DpmPp2mSdeKarras,
    /// `euler` on a flow model: FlowMatch Euler with its shift (spec 11.9c). SDXL refuses it.
    FlowEuler,
}

/// The timesteps of an Euler sampler (spec 11.9a). `Leading` is the spacing of SDXL base; `Trailing` ends at the first
/// step, as few-step checkpoints need.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSpacing {
    /// `leading`: the steps start at 0, with an offset of 1.
    Leading,
    /// `trailing`: the steps start at the last training step.
    Trailing,
}

/// The prompt of an image job, in the form that its engine reads (engine-abi "Image", "Qwen-Image 2.1"). An engine
/// refuses the form that it does not read.
#[derive(Clone, Debug, PartialEq)]
pub enum ImagePrompt {
    /// The form of SDXL (spec 11.9a).
    Chunks {
        /// The chunks of the prompt.
        prompt: Vec<ImageChunk>,
        /// The chunks of the negative prompt.
        negative: Vec<ImageChunk>,
        /// 2 or 3, in A1111's count: the encoders give the hidden state of layer `-clip_skip`. 2 is the
        /// penultimate layer. The `clip_skip` of diffusers is 1 less.
        clip_skip: u32,
    },
    /// The form of Qwen-Image (spec 11.9c): the ids of the whole template, and the count of system ids that the
    /// engine cuts from the encoder output. There is no negative prompt.
    Tokens { ids: Vec<u32>, cut: u32 },
}

/// One image job (engine-abi "Image"). The engine never takes text: the family makes the prompt (D2).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageJob {
    /// The prompt, in the form of the engine.
    pub prompt: ImagePrompt,
    /// The size of each image, in pixels.
    pub width: u32,
    pub height: u32,
    /// The steps of the sampler.
    pub steps: u32,
    /// The scale of classifier-free guidance.
    pub cfg: f32,
    pub sampler: ImageSampler,
    /// The spacing of the Euler timesteps. The DPM++ samplers refuse `Trailing`.
    pub spacing: ImageSpacing,
    /// The images of the job, 1 to 4. Image `i` uses the seed `seed + i`.
    pub count: u32,
    pub seed: u64,
    /// True when each step gives a preview of each image.
    pub previews: bool,
    /// The LoRAs of the job, in the order of the request (spec 11.9a). The engine adds each one's delta to its module.
    pub loras: Vec<LoraInput>,
    /// The rows of a conditioning hit in the store (spec 6.16), or none. With rows, the engine skips its encoder. An
    /// engine with no `image_conditioning` refuses rows.
    pub conditioning: Option<std::sync::Arc<Vec<u8>>>,
}

/// What an engine with cached conditionings declares (spec 6.16, engine-abi "Image").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageConditioning {
    /// The bytes of one row.
    pub row_bytes: u32,
    /// The most rows of one conditioning.
    pub max_rows: u32,
    /// The encoder digest: `blake3:` and 64 hex digits. It names the encoder, the tokenizer, the template, the cut
    /// and the build, and not the rest of the engine. Thus variants with the same encoder share their conditionings.
    pub encoder_digest: String,
}

/// One LoRA of a job: the bytes of its safetensors file, and its weight from -2.0 to 2.0.
#[derive(Clone, PartialEq)]
pub struct LoraInput {
    /// The `blake3:` id of the object that holds the file.
    pub id: String,
    pub weight: f32,
    pub file: std::sync::Arc<Vec<u8>>,
}

impl std::fmt::Debug for LoraInput {
    /// The bytes are left out of a debug print: a LoRA file is megabytes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraInput")
            .field("id", &self.id)
            .field("weight", &self.weight)
            .field("bytes", &self.file.len())
            .finish()
    }
}

/// An image of 8-bit channels, row after row: 3 channels for RGB, or 4 for RGBA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// `width * height * channels` bytes.
    pub pixels: Vec<u8>,
}

/// What one `image_step` call did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageStepOut {
    /// The steps that ran, after this one.
    pub step: u32,
    /// The steps of the job.
    pub steps: u32,
    /// With previews on, one preview of each image at one eighth of its size; else empty.
    pub previews: Vec<DecodedImage>,
}

/// The image capability (engine-abi "Image").
pub trait Image {
    /// The most pixels of all the images of a slot, which keeps a quantum within `max_quantum_ms`.
    fn image_max_pixels(&self) -> u64;

    /// What the engine declares for cached conditionings, or none when it has no such cache (spec 6.16).
    fn image_conditioning(&self) -> Option<ImageConditioning> {
        None
    }

    /// Encode the prompts, make the noise, and keep the job in the slot. It is one quantum of kind `embed`. It gives
    /// the conditioning rows that its encoder made, or none on a hit or with no `image_conditioning`.
    fn image_begin(&mut self, slot: SlotId, job: &ImageJob) -> Result<Option<Vec<u8>>, PassError>;

    /// Run one step of each image of the slot. It is one quantum of kind `step`.
    fn image_step(&mut self, slot: SlotId) -> Result<ImageStepOut, PassError>;

    /// Decode each latent of the slot with the VAE, after the last step. It is one quantum of kind `step`.
    fn image_decode(&mut self, slot: SlotId) -> Result<Vec<DecodedImage>, PassError>;

    /// Free the state of the slot. A slot that holds no job is not a failure.
    fn image_clear(&mut self, slot: SlotId);
}

/// The embedding capability (engine-abi "Embedding"). The engine never takes text: the family makes the ids (D2).
pub trait Embedding {
    /// The most ids of one call. It keeps a call within `max_quantum_ms`.
    fn embedding_max_tokens(&self) -> u32;

    /// The length of the vector of a call.
    fn embedding_dimension(&self) -> u32;

    /// Run one input and give its vector: the mean over each position of the model's last output, in f32, before any
    /// normalization. It is one quantum of kind `embed`, and it holds nothing after it ends.
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, PassError>;
}

/// The core trait that every engine implements.
pub trait Engine {
    /// The identity of the engine and its weights.
    fn identity(&self) -> Identity;

    /// The limits of the engine.
    fn limits(&self) -> Limits;

    /// Run one pass.
    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError>;

    /// Why the last call failed.
    fn last_error(&self) -> Option<String>;

    /// The part capability, when the engine has it.
    fn parts(&mut self) -> Option<&mut dyn Parts> {
        None
    }

    /// The coverage capability, when the engine has it.
    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        None
    }

    /// The seal capability, when the engine has it.
    fn seal(&mut self) -> Option<&mut dyn Seal> {
        None
    }

    /// The slot capability, when the engine has it.
    fn slots(&mut self) -> Option<&mut dyn Slots> {
        None
    }

    /// The stage capability, when the engine has it.
    fn stages(&mut self) -> Option<&mut dyn Stages> {
        None
    }

    /// The arena capability, when the engine has it. Call it before `Stages::take_handles`.
    fn arena(&mut self) -> Option<&mut dyn Arena> {
        None
    }

    /// The speech capability, when the engine has it (engine-abi "Speech").
    fn speech(&mut self) -> Option<&mut dyn Speech> {
        None
    }

    /// The image capability, when the engine has it (engine-abi "Image").
    fn image(&mut self) -> Option<&mut dyn Image> {
        None
    }

    /// The embedding capability, when the engine has it (engine-abi "Embedding").
    fn embedding(&mut self) -> Option<&mut dyn Embedding> {
        None
    }
}

#[cfg(feature = "fake")]
pub mod fake;
