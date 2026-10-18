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

/// The element type of a part.
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
    /// A 32-bit unsigned integer.
    U32,
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
    /// The minimum size in bytes.
    pub minimum: u64,
    /// The maximum size in bytes.
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

    /// Write `n` rows from `bytes`.
    fn cells_write(&mut self, state: State, part: PartId, n: u64, bytes: &[u8]) -> WriteResult;

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
}

#[cfg(feature = "fake")]
pub mod fake;
