//! The fake engines behind the feature `fake`.

use std::collections::{BTreeMap, VecDeque};

mod embedder;
mod feed;

pub use embedder::FakeEmbedder;
pub use feed::FakeFeed;

use std::fs::File;
use std::io::Write;
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::fs::FileExt;

use crate::{
    Arena, ArenaFd, ArenaPart, CONTROL_TERM, Capability, ChainLimits, CheckStrength, Control, ControlItem,
    ControlLimit, ControlOp, Coverage, CoverageSpan, CyclesLimit, Determinism, Dtype, ElasticTerm, Engine, Identity,
    Inputs, Kind, LayerLoop, LayerLoopLimit, Lifetime, Limits, MandatoryTerm, MemoryLimits, Output, PartError,
    PartKind, PartLimits, Parts, PassError, PassErrorKind, PassOut, PassStatus, Seal, SealError, Segment,
    SegmentInputs, SlotId, SlotLimit, Slots, Stage, State, Tap, TapLimit, Tier, WriteResult,
};

/// The layers of a vector in the control table of the fake (engine-abi "Control vectors").
pub const FAKE_CONTROL_LAYERS: u32 = 2;

/// The width of the residual of the fake: the values of one layer of a vector.
pub const FAKE_CONTROL_WIDTH: u32 = 4;

/// The tap that gives the residual after each layer at each logit row (spec 13.1).
pub const RESIDUAL_TAP: &str = "residual";

/// The tap that gives the logit row of each layer's residual at the one logit row of a pass (spec 13.1).
pub const LENS_TAP: &str = "lens";

/// The name of the one part of the fake arena, on device 0 (spec 10.2).
pub const FAKE_ARENA_PART: &str = "w";

/// The ring window in rows.
const RING_WINDOW: usize = 4;

/// The fixed part index.
const STATE_PART: usize = 0;

/// The cells part index.
const CELLS_PART: usize = 1;

/// The ring part index.
const RING_PART: usize = 2;

/// The deterministic fake engine.
pub struct FakeAlways(pub(crate) Fake);

/// The fake engine with one defect, for the conformance gates.
pub struct FakeBroken(pub(crate) Fake);

/// The fake engine whose empty ring forgets the position of its first row. The park and restore gate must fail it.
pub struct FakeRingBase(pub(crate) Fake);

/// The fake engine whose bits depend on the segment count of a pass. The segments gate (spec 8.3) must fail it.
pub struct FakeSegmentCount(pub(crate) Fake);

/// The fake engine whose bits depend on the row count of a pass. The segments gate must fail it on the split.
pub struct FakeRowCount(pub(crate) Fake);

impl FakeAlways {
    /// Make the engine.
    pub fn new() -> Self {
        Self::with_slots(default_slots())
    }

    /// Make the engine with the given slot size classes.
    pub fn with_slots(slots: Vec<SlotLimit>) -> Self {
        FakeAlways(Fake::new("fake-always", false, slots, false))
    }
}

impl FakeAlways {
    /// Make the engine with an arena that holds `bytes`, in a memfd (spec 10.2).
    pub fn with_arena(bytes: &[u8]) -> Self {
        let mut file = memfd("nagaya-fake-arena");
        file.write_all(bytes).expect("cannot write the fake arena");
        Self::new().arena_from(file)
    }

    /// The parts of the fake arena: one part on device 0.
    pub fn arena_parts() -> Vec<ArenaPart> {
        vec![ArenaPart {
            device: 0,
            part: FAKE_ARENA_PART.to_owned(),
        }]
    }

    /// Make the engine on a stored arena. The check compares every byte with `bytes`. A mismatch opens nothing.
    pub fn attach_arena(arenas: Vec<ArenaFd>, bytes: &[u8]) -> Result<(Self, CheckStrength), String> {
        let [arena] = <[ArenaFd; 1]>::try_from(arenas)
            .map_err(|arenas| format!("the fake arena has one fd, and the attach got {}", arenas.len()))?;
        if arena.device != 0 || arena.part != FAKE_ARENA_PART {
            return Err(format!(
                "the fake arena has the part `{FAKE_ARENA_PART}` on device 0, and the attach got `{}` on device {}",
                arena.part, arena.device
            ));
        }
        let file = File::from(arena.fd);
        let size = file
            .metadata()
            .map_err(|error| format!("cannot read the size of the stored arena: {error}"))?
            .len();
        let mut stored = vec![0u8; size as usize];
        file.read_exact_at(&mut stored, 0)
            .map_err(|error| format!("cannot read the stored arena: {error}"))?;
        if stored != bytes {
            return Err("the stored arena differs from the build of the fake".to_owned());
        }
        Ok((Self::new().arena_from(file), CheckStrength::Full))
    }

    fn arena_from(mut self, file: File) -> Self {
        self.0.arena = Some(file);
        self.0.limits.capabilities.push(Capability::Arena);
        self
    }
}

/// A new memfd with no bytes.
fn memfd(name: &str) -> File {
    let name = std::ffi::CString::new(name).expect("a memfd name has no NUL");
    // SAFETY: `name` is a valid C string, and the call gives a new fd or -1.
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create failed: {}", std::io::Error::last_os_error());
    // SAFETY: the fd is new, and nothing else owns it.
    File::from(unsafe { OwnedFd::from_raw_fd(fd) })
}

impl Arena for Fake {
    fn arena_export(&mut self) -> Result<Vec<ArenaFd>, String> {
        let file = self.arena.as_ref().ok_or("the fake has no arena")?;
        let fd = file
            .try_clone()
            .map_err(|error| format!("cannot duplicate the fake arena fd: {error}"))?;
        Ok(vec![ArenaFd {
            device: 0,
            part: FAKE_ARENA_PART.to_owned(),
            fd: fd.into(),
        }])
    }
}

impl FakeAlways {
    /// Name the model of the engine identity. The default is `fake-always`. A test of the cluster gives each node its
    /// own served model name (spec 2.3, "A served model name in the cluster").
    pub fn with_model(mut self, model: &str) -> Self {
        self.0.model = model.to_owned();
        self
    }

    /// Name the family library that the engine needs. The default is `fake`, which has no library (spec 11.3).
    pub fn with_family(mut self, family: &str) -> Self {
        self.0.family = family.to_owned();
        self
    }

    /// Give the engine logits over a vocabulary of `vocab` tokens (engine-abi, "Logit outputs").
    ///
    /// Each requested logit row is a function of the fixed state after the tokens up to its position.
    pub fn with_vocab(mut self, vocab: usize) -> Self {
        assert!(vocab > 0, "a vocabulary has at least one token");
        self.0.vocab = Some(vocab);
        self
    }

    /// Add `bias` to each logit row: the value of token `t` gets `bias[t]`. Call it after [`FakeAlways::with_vocab`].
    ///
    /// A family's test can make the fake favor the tokens that its template renders back.
    pub fn with_logit_bias(mut self, bias: Vec<f32>) -> Self {
        let vocab = self
            .0
            .vocab
            .expect("a logit bias needs a vocabulary; call with_vocab first");
        assert_eq!(
            bias.len(),
            vocab,
            "a logit bias has one bias for each token of the vocabulary"
        );
        self.0.logit_bias = Some(bias);
        self
    }

    /// Give the engine a control table of `tables` entries of [`FAKE_CONTROL_LAYERS`] layers of
    /// [`FAKE_CONTROL_WIDTH`] values, and the tap `residual` (engine-abi "Control vectors", spec 13.1).
    ///
    /// An item changes the state after each token by a function of the vector, the operation and the strength. An add
    /// or an ablate of strength 0 changes no bit. The tap gives values that are a function of the state at the row.
    pub fn with_control(mut self, tables: u32) -> Self {
        let limit = ControlLimit {
            tables,
            layers: FAKE_CONTROL_LAYERS,
            width: FAKE_CONTROL_WIDTH,
        };
        let limits = &mut self.0.limits;
        limits.capabilities.push(Capability::Control);
        limits.control = Some(limit);
        limits.memory.mandatory.push(MandatoryTerm {
            name: CONTROL_TERM.to_owned(),
            tier: Tier::Device(0),
            bytes: limit.table_bytes(),
            lifetime: Lifetime::Engine,
            kinds: vec![Kind::Prefill, Kind::Decode],
        });
        limits.taps.push(TapLimit {
            name: RESIDUAL_TAP.to_owned(),
            conditions: Vec::new(),
        });
        self.0.table = vec![vec![0.0; limit.entry_bytes() as usize / 4]; tables as usize];
        self
    }

    /// Give the engine layer loops over [`FAKE_CONTROL_LAYERS`] layers, with a reserve of `extra_runs` runs in each
    /// slot (engine-abi "Layer loops"). A span starts and ends at any layer and runs up to 4 times. A loop of `count`
    /// 2 or more changes the state after each token by a function of the loop, as a control list does. A loop of
    /// `count` 1 changes no bit.
    pub fn with_layer_loops(mut self, extra_runs: u32) -> Self {
        let limits = &mut self.0.limits;
        limits.capabilities.push(Capability::LayerLoops);
        limits.layer_loops = Some(LayerLoopLimit {
            layers: FAKE_CONTROL_LAYERS,
            bounds: (0..=FAKE_CONTROL_LAYERS).collect(),
            count: 4,
            extra_runs,
        });
        self
    }

    /// Give the engine the cycles of `limit` (engine-abi "Inkling 2"). A count other than `default` changes the state
    /// after each token by a function of the count and the span, as a layer loop does. The default count, and a segment
    /// with no count, change no bit, so they give the bits of a fake without cycles.
    pub fn with_cycles(mut self, limit: CyclesLimit) -> Self {
        let limits = &mut self.0.limits;
        limits.capabilities.push(Capability::Cycles);
        limits.cycles = Some(limit);
        self
    }

    /// Give the engine the tap `lens` (spec 13.1). Call it after [`FakeAlways::with_vocab`] and
    /// [`FakeAlways::with_control`]. The row of each layer but the last is a function of the state and the layer. The
    /// row of the last layer is the logit row, as the lens of the last layer is the head of the model.
    pub fn with_lens(mut self) -> Self {
        assert!(
            self.0.vocab.is_some() && self.0.limits.control.is_some(),
            "a lens needs a vocabulary and the layers of a control table; call with_vocab and with_control first"
        );
        self.0.limits.taps.push(TapLimit {
            name: LENS_TAP.to_owned(),
            conditions: vec!["one logit row".to_owned()],
        });
        self
    }
}

impl FakeBroken {
    /// Make the broken engine.
    pub fn new() -> Self {
        FakeBroken(Fake::new("fake-broken", true, default_slots(), false))
    }
}

impl FakeRingBase {
    /// Make the engine that loses an empty ring's base.
    pub fn new() -> Self {
        FakeRingBase(Fake::new("fake-ring-base", false, default_slots(), true))
    }
}

impl FakeSegmentCount {
    /// Make the engine whose bits depend on the segment count.
    pub fn new() -> Self {
        let mut fake = Fake::new("fake-segment-count", false, default_slots(), false);
        fake.count_defect = CountDefect::Segments;
        FakeSegmentCount(fake)
    }
}

impl FakeRowCount {
    /// Make the engine whose bits depend on the row count of a pass.
    pub fn new() -> Self {
        let mut fake = Fake::new("fake-row-count", false, default_slots(), false);
        fake.count_defect = CountDefect::Rows;
        FakeRowCount(fake)
    }
}

/// Slot 0 is small. The gate for several segments needs one slot for each segment of the largest pass.
fn default_slots() -> Vec<SlotLimit> {
    vec![
        SlotLimit {
            bytes: 64,
            count: 1,
            tokens: 4096,
        },
        SlotLimit {
            bytes: 256,
            count: 7,
            tokens: 4096,
        },
    ]
}

/// A defect in which the bits of a segment depend on the shape of the pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CountDefect {
    None,
    Segments,
    Rows,
}

impl Default for FakeAlways {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FakeBroken {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FakeSegmentCount {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FakeRowCount {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FakeRingBase {
    fn default() -> Self {
        Self::new()
    }
}

/// The state of a fake engine.
pub(crate) struct Fake {
    model: String,
    broken: bool,
    ignore_ring_from: bool,
    count_defect: CountDefect,
    /// The vocabulary size, when the engine gives logits.
    vocab: Option<usize>,
    /// The family library that the engine needs (spec 11.3).
    family: String,
    /// The bias that each logit row adds, one value for each token.
    logit_bias: Option<Vec<f32>>,
    /// The memory of the arena, when the engine has one (spec 10.2).
    arena: Option<File>,
    limits: Limits,
    slots: Vec<SlotData>,
    last_error: Option<String>,
    /// The entries of the control table, each `layers` times `width` values. It is empty with no table.
    table: Vec<Vec<f32>>,
}

/// The data of one slot.
struct SlotData {
    parts: Vec<PartData>,
}

/// The data of one part in one slot.
enum PartData {
    Fixed(Vec<u8>),
    Cells { rows: Vec<Vec<u8>>, n: u64 },
    Ring { rows: VecDeque<Vec<u8>>, base: u64 },
    Derived,
}

impl Fake {
    fn new(model: &str, broken: bool, slot_limits: Vec<SlotLimit>, ignore_ring_from: bool) -> Self {
        let parts = vec![
            PartLimits {
                name: "state".to_owned(),
                kind: PartKind::Fixed,
                dtype: Dtype::U8,
                shape: vec![16],
                block_format: None,
                stage: 0,
            },
            PartLimits {
                name: "cells".to_owned(),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![8],
                block_format: None,
                stage: 0,
            },
            PartLimits {
                name: "ring".to_owned(),
                kind: PartKind::Ring,
                dtype: Dtype::U8,
                shape: vec![8],
                block_format: None,
                stage: 0,
            },
            PartLimits {
                name: "pooled".to_owned(),
                kind: PartKind::Derived,
                dtype: Dtype::U8,
                shape: vec![8],
                block_format: None,
                stage: 0,
            },
        ];
        let limits = Limits {
            capabilities: vec![
                Capability::Parts,
                Capability::Coverage,
                Capability::Seal,
                Capability::Slots,
            ],
            parts: parts.clone(),
            inputs: Inputs::Tokens,
            devices: vec![0],
            stages: vec![Stage {
                device: 0,
                layers: 0..1,
                head: true,
            }],
            max_segments: 8,
            max_pass_inputs: 64,
            max_logit_rows: 8,
            slots: slot_limits,
            max_quantum_ms: 10,
            memory: MemoryLimits {
                mandatory: vec![MandatoryTerm {
                    name: "weights".to_owned(),
                    tier: Tier::Device(0),
                    bytes: 4096,
                    lifetime: Lifetime::Engine,
                    kinds: vec![Kind::Prefill, Kind::Decode],
                }],
                elastic: vec![ElasticTerm {
                    name: "workspace".to_owned(),
                    tier: Tier::Host,
                    unit_bytes: 256,
                    minimum: 1,
                    maximum: 8,
                    kinds: vec![Kind::Prefill, Kind::Decode],
                }],
            },
            chain: ChainLimits {
                max_depth: 0,
                delta_fraction: 0.0,
            },
            segments_exact: true,
            determinism: Determinism::Always,
            bit_dependence: Vec::new(),
            taps: Vec::new(),
            checkpoint_grain: None,
            feed: None,
            residual_row_bytes: 0,
            control: None,
            layer_loops: None,
            cycles: None,
        };
        let slots = (0..limits.slots.iter().map(|slot| slot.count).sum())
            .map(|_| SlotData::new(&parts))
            .collect();
        Fake {
            model: model.to_owned(),
            broken,
            ignore_ring_from,
            count_defect: CountDefect::None,
            vocab: None,
            family: "fake".to_owned(),
            logit_bias: None,
            arena: None,
            limits,
            slots,
            last_error: None,
            table: Vec::new(),
        }
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn identity(&self) -> Identity {
        Identity {
            model: self.model.clone(),
            family: self.family.clone(),
            checkpoint_digest: "fake:0".to_owned(),
            tokenizer_hash: "fake:0".to_owned(),
            template: "fake".to_owned(),
            numbers: BTreeMap::new(),
            plan_table: "fake".to_owned(),
            bit_settings: Vec::new(),
            canary_hash: "fake:0".to_owned(),
            build: std::collections::BTreeMap::new(),
        }
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        if let Some(tap) = taps
            .iter()
            .find(|tap| self.limits.taps.iter().all(|known| known.name != tap.name))
        {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: format!("nothing ran: the fake has no tap `{}`", tap.name),
            });
        }
        let tapped = taps.iter().any(|tap| tap.name == RESIDUAL_TAP);
        let lensed = taps.iter().any(|tap| tap.name == LENS_TAP);
        // A pass that the fake refuses runs no segment and changes no state (engine-abi, "A pass that did not run").
        if segments.len() > self.limits.max_segments as usize {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: format!(
                    "the pass has {} segments; the fake takes {}",
                    segments.len(),
                    self.limits.max_segments
                ),
            });
        }
        // A fake with no control table refuses a control item, which would change no bit that it gives.
        for segment in segments.iter().filter(|segment| !segment.control.is_empty()) {
            let Some(limit) = self.limits.control else {
                return Err(PassError {
                    kind: PassErrorKind::Refused,
                    message: "a segment names control items; the fake does not declare `control`".to_owned(),
                });
            };
            limit.check(&segment.control).map_err(|message| PassError {
                kind: PassErrorKind::Refused,
                message: format!("nothing ran: {message}"),
            })?;
        }
        self.limits.check_segment_loops(segments).map_err(|message| PassError {
            kind: PassErrorKind::Refused,
            message: format!("nothing ran: {message}"),
        })?;
        self.limits
            .check_segment_cycles(segments)
            .map_err(|message| PassError {
                kind: PassErrorKind::Refused,
                message: format!("nothing ran: {message}"),
            })?;
        let rows: usize = segments
            .iter()
            .map(|segment| match &segment.inputs {
                SegmentInputs::Tokens(tokens) => tokens.len(),
                _ => 0,
            })
            .sum();
        if rows > self.limits.max_pass_inputs as usize {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: format!(
                    "the pass has {rows} tokens; the fake takes {}",
                    self.limits.max_pass_inputs
                ),
            });
        }
        let logit_rows: usize = segments.iter().map(|segment| segment.outputs.logit_rows.len()).sum();
        if logit_rows > self.limits.max_logit_rows as usize {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: format!(
                    "the pass asks for {logit_rows} logit rows; the fake gives {}",
                    self.limits.max_logit_rows
                ),
            });
        }
        if lensed && logit_rows != 1 {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: format!(
                    "nothing ran: the tap `{LENS_TAP}` reads one logit row, and the pass asks for {logit_rows}"
                ),
            });
        }
        if segments
            .iter()
            .any(|segment| !matches!(segment.inputs, SegmentInputs::Tokens(_)))
        {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: "the fake takes tokens".to_owned(),
            });
        }
        // A logit row names a position that the segment feeds. A pass with another row did not run.
        for segment in segments {
            if let Some(row) = segment
                .outputs
                .logit_rows
                .iter()
                .find(|row| !segment.positions.contains(row))
            {
                return Err(PassError {
                    kind: PassErrorKind::Refused,
                    message: format!("the segment wants a logit row at position {row}, which it does not feed"),
                });
            }
        }
        let mut outputs = Vec::new();
        // A write that fails before any write of the pass succeeded changed nothing. That pass did not run.
        let mut changed = false;
        for segment in segments {
            let SegmentInputs::Tokens(tokens) = &segment.inputs else {
                unreachable!("the check above admits only tokens");
            };
            let rows: Vec<u8> = tokens
                .iter()
                .flat_map(|token| {
                    let bytes = token.to_le_bytes();
                    [
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[0], bytes[1], bytes[2], bytes[3],
                    ]
                })
                .collect();
            let count = tokens.len() as u64;
            let cells_from = self.cells_high(segment.state);
            if self.cells_write(segment.state, CELLS_PART, cells_from, count, &rows) != WriteResult::Ok {
                return Err(PassError {
                    kind: if changed {
                        PassErrorKind::Ran(PassStatus::Poison)
                    } else {
                        PassErrorKind::Refused
                    },
                    message: "the fake could not append the cells".to_owned(),
                });
            }
            changed = true;
            let ring_from = self.ring_high(segment.state);
            if self.cells_write(segment.state, RING_PART, ring_from, count, &rows) != WriteResult::Ok {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the fake could not advance the ring".to_owned(),
                });
            }
            let mut state_bytes = self.state_bytes(segment.state);
            let steer = self.steer(&segment.control, segment.layer_loop, segment.cycles);
            let mut logit_outputs = Vec::new();
            let mut residual = Vec::new();
            let mut lens = Vec::new();
            {
                // Fold one token at a time, so that a row sees the state at its own position.
                let mut folded = state_bytes;
                let mut rows: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
                let mut tapped_rows: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
                let mut lens_rows: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
                for (index, token) in tokens.iter().enumerate() {
                    let position = segment.positions.get(index).copied().unwrap_or(index as u64);
                    fold_steered(&mut folded, *token, position, steer.as_ref());
                    if !segment.outputs.logit_rows.contains(&position) {
                        continue;
                    }
                    if let Some(vocab) = self.vocab {
                        rows.insert(position, logit_row(&folded, vocab, self.logit_bias.as_deref()));
                    }
                    if tapped {
                        tapped_rows.insert(position, residual_row(&folded));
                    }
                    if lensed {
                        let vocab = self.vocab.expect("a fake with a lens has a vocabulary");
                        let rows = (0..FAKE_CONTROL_LAYERS)
                            .flat_map(|layer| lens_row(&folded, layer, vocab, self.logit_bias.as_deref()))
                            .collect();
                        lens_rows.insert(position, rows);
                    }
                }
                // The output order is the request order.
                for row in &segment.outputs.logit_rows {
                    if let Some(bytes) = rows.get(row) {
                        logit_outputs.push(Output {
                            name: "logits".to_owned(),
                            dtype: Dtype::F32,
                            bytes: bytes.clone(),
                        });
                    }
                    if let Some(bytes) = tapped_rows.get(row) {
                        residual.extend_from_slice(bytes);
                    }
                    if let Some(bytes) = lens_rows.get(row) {
                        lens.extend_from_slice(bytes);
                    }
                }
            }
            for (index, token) in tokens.iter().enumerate() {
                let position = segment.positions.get(index).copied().unwrap_or(index as u64);
                fold_steered(&mut state_bytes, *token, position, steer.as_ref());
            }
            match self.count_defect {
                CountDefect::None => {}
                CountDefect::Segments => {
                    if segments.len() > 1 {
                        state_bytes[0] ^= segments.len() as u8;
                    }
                }
                CountDefect::Rows => {
                    if tokens.len() > 1 {
                        state_bytes[1] ^= tokens.len() as u8;
                    }
                }
            }
            if self.part_write(segment.state, STATE_PART, &state_bytes) != WriteResult::Ok {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the fake could not write the state".to_owned(),
                });
            }
            // The driver splits the outputs of a joined pass by one `logits` for each row (engine-abi, "Logit
            // outputs"), so a fake with a vocabulary gives no `state` in a pass of several segments. A fake with
            // no vocabulary gives `state` in each pass, and the gates of the segments count it.
            if self.vocab.is_none() || segments.len() == 1 {
                outputs.push(Output {
                    name: "state".to_owned(),
                    dtype: Dtype::U8,
                    bytes: state_bytes.to_vec(),
                });
            }
            outputs.extend(logit_outputs);
            if tapped {
                outputs.push(Output {
                    name: RESIDUAL_TAP.to_owned(),
                    dtype: Dtype::F32,
                    bytes: residual,
                });
            }
            if lensed {
                outputs.push(Output {
                    name: LENS_TAP.to_owned(),
                    dtype: Dtype::F32,
                    bytes: lens,
                });
            }
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 0,
            outputs,
        })
    }

    /// The change of the state after each token for a control list, a layer loop and a count of cycles: a mix of the
    /// vector, the operation and the strength of each item that changes bits, in order, then of the loop, then of the
    /// count. A list of adds and ablates of strength 0 with no loop, a loop of `count` 1, and the default count of
    /// cycles give `None`, so they change no bit (engine-abi "Control vectors", "Layer loops", "Inkling 2").
    fn steer(&self, items: &[ControlItem], layer_loop: Option<LayerLoop>, cycles: Option<u8>) -> Option<[u8; 16]> {
        let mut changed = false;
        let mut hash = 0x7374_6565_7200_u64;
        let width = self.limits.control.map_or(1, |limit| limit.width as usize).max(1);
        for item in items.iter().filter(|item| !item.changes_nothing()) {
            changed = true;
            let op = match item.op {
                ControlOp::Add => 1,
                ControlOp::Ablate => 2,
                ControlOp::Clamp => 3,
            };
            let entry = &self.table[item.index as usize];
            // A layer whose rows are all zero has no tensor, so no strength touches it. An item whose layers give each
            // layer with a tensor the item's strength mixes as the item with no layers (spec 11.12).
            let held: Vec<usize> = (0..entry.len() / width)
                .filter(|l| entry[l * width..(l + 1) * width].iter().any(|v| *v != 0.0))
                .collect();
            let uniform = item.layers.is_none()
                || held
                    .iter()
                    .all(|l| item.strength_at(*l as u32).map(f32::to_bits) == Some(item.strength.to_bits()));
            hash = mix(hash ^ op);
            if uniform {
                hash = mix(hash ^ u64::from(item.strength.to_bits()));
                for value in entry {
                    hash = mix(hash ^ u64::from(value.to_bits()));
                }
                continue;
            }
            for l in &held {
                let Some(s) = item.strength_at(*l as u32) else { continue };
                if s == 0.0 && item.op != ControlOp::Clamp {
                    continue;
                }
                hash = mix(hash ^ (*l as u64) << 32);
                hash = mix(hash ^ u64::from(s.to_bits()));
                for value in &entry[l * width..(l + 1) * width] {
                    hash = mix(hash ^ u64::from(value.to_bits()));
                }
            }
        }
        if let Some(lp) = layer_loop.filter(|lp| lp.count > 1) {
            changed = true;
            hash = mix(hash
                ^ 0x6c6f_6f70_0000_0000
                ^ u64::from(lp.start) << 32
                ^ u64::from(lp.end) << 16
                ^ u64::from(lp.count));
        }
        // The pass checked the count against the limit, so a count here has a limit.
        if let Some(limit) = self.limits.cycles
            && limit.resolve(cycles) != limit.default
        {
            changed = true;
            hash = mix(hash
                ^ 0x6379_636c_0000_0000
                ^ u64::from(limit.start) << 32
                ^ u64::from(limit.end) << 16
                ^ u64::from(limit.resolve(cycles)));
        }
        if !changed {
            return None;
        }
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&mix(hash).to_le_bytes());
        // A change of 0 would leave a byte as it is. Thus each byte gets a bit.
        for byte in &mut bytes {
            *byte |= 1;
        }
        Some(bytes)
    }

    /// The bytes of the fixed state part, or zeros for an unknown slot.
    fn state_bytes(&self, state: State) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        if let Some(PartData::Fixed(stored)) = self
            .slots
            .get(Self::slot_id(state))
            .and_then(|slot| slot.parts.get(STATE_PART))
        {
            bytes.copy_from_slice(stored);
        }
        bytes
    }

    fn slot_id(state: State) -> usize {
        let State::Slot(id) = state;
        id as usize
    }

    fn row_bytes(&self, part: usize) -> Option<usize> {
        let part = self.limits.parts.get(part)?;
        Some(part.shape.iter().product::<u64>() as usize)
    }

    fn slot_capacity(&self, slot: usize) -> Option<u64> {
        let mut remaining = slot;
        for class in &self.limits.slots {
            if remaining < class.count as usize {
                return Some(class.bytes);
            }
            remaining -= class.count as usize;
        }
        None
    }

    fn slot_bytes(&self, state: State) -> Option<u64> {
        self.slot_capacity(Self::slot_id(state))
    }

    /// The number of rows that the cells part holds.
    fn cells_high(&self, state: State) -> u64 {
        match self
            .slots
            .get(Self::slot_id(state))
            .and_then(|slot| slot.parts.get(CELLS_PART))
        {
            Some(PartData::Cells { rows, .. }) => rows.len() as u64,
            _ => 0,
        }
    }

    /// The position after the ring's last row.
    fn ring_high(&self, state: State) -> u64 {
        match self
            .slots
            .get(Self::slot_id(state))
            .and_then(|slot| slot.parts.get(RING_PART))
        {
            Some(PartData::Ring { rows, base }) => *base + rows.len() as u64,
            _ => 0,
        }
    }

    fn refuse(&mut self, message: &str) -> WriteResult {
        self.last_error = Some(message.to_owned());
        WriteResult::Refused
    }

    fn part_read_inner(&self, state: State, part: usize, out: &mut [u8]) -> Result<(), PartError> {
        let slot = self.slots.get(Self::slot_id(state)).ok_or(PartError::Refused)?;
        let data = slot.parts.get(part).ok_or(PartError::Refused)?;
        let row_bytes = self.row_bytes(part).ok_or(PartError::Refused)?;
        match data {
            PartData::Fixed(bytes) => {
                if out.len() != bytes.len() {
                    return Err(PartError::Refused);
                }
                out.copy_from_slice(bytes);
                Ok(())
            }
            PartData::Cells { rows, .. } => {
                if out.len() != rows.len() * row_bytes {
                    return Err(PartError::Refused);
                }
                for (index, row) in rows.iter().enumerate() {
                    out[index * row_bytes..(index + 1) * row_bytes].copy_from_slice(row);
                }
                Ok(())
            }
            PartData::Ring { rows, .. } => {
                if out.len() != rows.len() * row_bytes {
                    return Err(PartError::Refused);
                }
                for (index, row) in rows.iter().enumerate() {
                    out[index * row_bytes..(index + 1) * row_bytes].copy_from_slice(row);
                }
                Ok(())
            }
            PartData::Derived => Err(PartError::Refused),
        }
    }

    fn cells_read_inner(
        &self,
        state: State,
        part: usize,
        from: u64,
        count: u64,
        out: &mut [u8],
    ) -> Result<(), PartError> {
        let slot = self.slots.get(Self::slot_id(state)).ok_or(PartError::Refused)?;
        let data = slot.parts.get(part).ok_or(PartError::Refused)?;
        let row_bytes = self.row_bytes(part).ok_or(PartError::Refused)?;
        if out.len() != count as usize * row_bytes {
            return Err(PartError::Refused);
        }
        match data {
            PartData::Cells { rows, n } => {
                if from + count > *n {
                    return Err(PartError::Refused);
                }
                for index in 0..count {
                    let row = &rows[(from + index) as usize];
                    out[index as usize * row_bytes..(index as usize + 1) * row_bytes].copy_from_slice(row);
                }
                Ok(())
            }
            PartData::Ring { rows, base } => {
                let end = *base + rows.len() as u64;
                if from < *base || from + count > end {
                    return Err(PartError::Refused);
                }
                for index in 0..count {
                    let row = &rows[(from + index - *base) as usize];
                    out[index as usize * row_bytes..(index as usize + 1) * row_bytes].copy_from_slice(row);
                }
                Ok(())
            }
            _ => Err(PartError::Refused),
        }
    }

    fn fits_inner(&self, coverage: &CoverageSpan, slot: u32) -> Result<(), String> {
        let slot_bytes = self
            .slot_capacity(slot as usize)
            .ok_or_else(|| "unknown slot".to_owned())?;
        let row_bytes = self.row_bytes(coverage.part).ok_or_else(|| "unknown part".to_owned())? as u64;
        let bytes = (coverage.end - coverage.start) * row_bytes;
        if bytes <= slot_bytes {
            Ok(())
        } else {
            Err(format!("{bytes} bytes do not fit {slot_bytes}"))
        }
    }
}

/// Fold one token into the state. The result does not depend on how a pass splits the tokens.
fn fold_token(state: &mut [u8; 16], token: u32, position: u64) {
    for (offset, byte) in token.to_le_bytes().iter().enumerate() {
        let slot = (position as usize + offset) % 16;
        state[slot] = state[slot].wrapping_mul(31).wrapping_add(*byte);
    }
}

/// Fold one token, then change the state by the mix of the control items, when a segment has one.
fn fold_steered(state: &mut [u8; 16], token: u32, position: u64, steer: Option<&[u8; 16]>) {
    fold_token(state, token, position);
    if let Some(steer) = steer {
        for (byte, change) in state.iter_mut().zip(steer) {
            *byte ^= *change;
        }
    }
}

/// The residual tap of one row: [`FAKE_CONTROL_LAYERS`] layers of [`FAKE_CONTROL_WIDTH`] little-endian F32 values, a
/// function of the state bytes. Each value is a small integer over 32, so it is exact.
fn residual_row(state: &[u8; 16]) -> Vec<u8> {
    let count = (FAKE_CONTROL_LAYERS * FAKE_CONTROL_WIDTH) as usize;
    (0..count)
        .flat_map(|index| {
            let byte = state[(index * 5 + 3) % 16];
            ((f32::from(byte) - 127.5) / 32.0 + (index / FAKE_CONTROL_WIDTH as usize) as f32).to_le_bytes()
        })
        .collect()
}

/// The lens of one layer at a row: the logit row for the last layer, and the logit row of a state that the layer
/// changes for each other layer.
fn lens_row(state: &[u8; 16], layer: u32, vocab: usize, bias: Option<&[f32]>) -> Vec<u8> {
    if layer + 1 == FAKE_CONTROL_LAYERS {
        return logit_row(state, vocab, bias);
    }
    let mut changed = *state;
    changed[0] ^= 0x55 ^ layer as u8;
    logit_row(&changed, vocab, bias)
}

/// One logit row: `vocab` little-endian F32 values between -6 and 6, a function of the state bytes only. A bias adds
/// its value for each token.
pub(crate) fn logit_row(state: &[u8; 16], vocab: usize, bias: Option<&[f32]>) -> Vec<u8> {
    let seed = state
        .iter()
        .fold(0x6C6F_6769_7473_u64, |hash, byte| mix(hash ^ u64::from(*byte)));
    let mut bytes = Vec::with_capacity(vocab * 4);
    for token in 0..vocab as u64 {
        // 24 bits fill an F32 mantissa, so the value is exact.
        let unit = (mix(seed ^ mix(token + 1)) >> 40) as f32 / (1u32 << 24) as f32;
        let offset = bias.map_or(0.0, |bias| bias[token as usize]);
        bytes.extend_from_slice(&(unit * 12.0 - 6.0 + offset).to_le_bytes());
    }
    bytes
}

/// A 64-bit mixing function (splitmix64).
pub(crate) fn mix(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl SlotData {
    fn new(parts: &[PartLimits]) -> Self {
        let parts = parts
            .iter()
            .map(|part| match part.kind {
                PartKind::Fixed => PartData::Fixed(vec![0; part.shape.iter().product::<u64>() as usize]),
                PartKind::Cells => PartData::Cells { rows: Vec::new(), n: 0 },
                PartKind::Ring => PartData::Ring {
                    rows: VecDeque::new(),
                    base: 0,
                },
                PartKind::Derived => PartData::Derived,
            })
            .collect();
        SlotData { parts }
    }
}

macro_rules! engine_impl {
    ($type:ty) => {
        impl Engine for $type {
            fn identity(&self) -> Identity {
                self.0.identity()
            }

            fn limits(&self) -> Limits {
                self.0.limits()
            }

            fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
                self.0.pass(segments, taps)
            }

            fn last_error(&self) -> Option<String> {
                self.0.last_error.clone()
            }

            fn parts(&mut self) -> Option<&mut dyn Parts> {
                Some(&mut self.0)
            }

            fn coverage(&mut self) -> Option<&mut dyn Coverage> {
                Some(&mut self.0)
            }

            fn seal(&mut self) -> Option<&mut dyn Seal> {
                Some(&mut self.0)
            }

            fn slots(&mut self) -> Option<&mut dyn Slots> {
                Some(&mut self.0)
            }

            fn arena(&mut self) -> Option<&mut dyn Arena> {
                if self.0.arena.is_some() {
                    Some(&mut self.0)
                } else {
                    None
                }
            }

            fn control(&mut self) -> Option<&mut dyn Control> {
                if self.0.limits.control.is_some() {
                    Some(&mut self.0)
                } else {
                    None
                }
            }
        }
    };
}

impl Control for Fake {
    /// Write one vector into its entry. A wrong index, a wrong size or a value that is not finite is refused, and
    /// nothing changes.
    fn control_write(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        let limit = self.limits.control.ok_or("the fake has no control table")?;
        let refused = if index >= limit.tables {
            Some(format!(
                "the control table has {} entries, not entry {index}",
                limit.tables
            ))
        } else if bytes.len() as u64 != limit.entry_bytes() {
            Some(format!(
                "a vector is {} layers of {} f32 values, {} bytes, not {} bytes",
                limit.layers,
                limit.width,
                limit.entry_bytes(),
                bytes.len()
            ))
        } else {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .position(|b| !f32::from_le_bytes(*b).is_finite())
                .map(|at| format!("value {at} of the vector is not finite; the op needs finite values"))
        };
        if let Some(message) = refused {
            self.last_error = Some(message.clone());
            return Err(message);
        }
        self.table[index as usize] = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(())
    }
}

engine_impl!(FakeAlways);
engine_impl!(FakeBroken);
engine_impl!(FakeRingBase);
engine_impl!(FakeSegmentCount);
engine_impl!(FakeRowCount);

impl Parts for Fake {
    fn part_read(&mut self, state: State, part: usize, out: &mut [u8]) -> Result<(), PartError> {
        let result = self.part_read_inner(state, part, out);
        match result {
            Ok(()) => {
                if self.broken
                    && let Some(byte) = out.first_mut()
                {
                    *byte ^= 1;
                }
                Ok(())
            }
            Err(error) => {
                self.last_error = Some("part read refused".to_owned());
                Err(error)
            }
        }
    }

    fn part_write(&mut self, state: State, part: usize, bytes: &[u8]) -> WriteResult {
        let slot_id = Self::slot_id(state);
        let expected = match self.limits.parts.get(part) {
            Some(part) if part.kind == PartKind::Fixed => part.shape.iter().product::<u64>() as usize,
            _ => return self.refuse("part is not writable as a whole"),
        };
        if bytes.len() != expected {
            return self.refuse("wrong byte count for the part");
        }
        let Some(slot) = self.slots.get_mut(slot_id) else {
            return self.refuse("unknown slot");
        };
        match slot.parts.get_mut(part) {
            Some(PartData::Fixed(stored)) => {
                stored.copy_from_slice(bytes);
                WriteResult::Ok
            }
            _ => unreachable!("the part kind was checked as fixed"),
        }
    }

    fn cells_read(&mut self, state: State, part: usize, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let result = self.cells_read_inner(state, part, from, n, out);
        if result.is_err() {
            self.last_error = Some("cells read refused".to_owned());
        }
        result
    }

    fn cells_write(&mut self, state: State, part: usize, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let slot_id = Self::slot_id(state);
        let row_bytes = match self.row_bytes(part) {
            Some(row_bytes) => row_bytes,
            None => return self.refuse("unknown part"),
        };
        let count = n as usize;
        if bytes.len() != count * row_bytes {
            return self.refuse("wrong byte count for the rows");
        }
        let kind = match self.limits.parts.get(part) {
            Some(part) => part.kind,
            None => return self.refuse("unknown part"),
        };
        match kind {
            PartKind::Cells => {
                let current = match self.slots.get(slot_id) {
                    Some(slot) => match slot.parts.get(part) {
                        Some(PartData::Cells { rows, .. }) => rows.len(),
                        _ => return self.refuse("part is not cells"),
                    },
                    None => return self.refuse("unknown slot"),
                };
                if from != current as u64 {
                    return self.refuse("a cells part appends at its last row");
                }
                let slot_bytes = match self.slot_bytes(state) {
                    Some(bytes) => bytes,
                    None => return self.refuse("unknown slot"),
                };
                if ((current + count) as u64) * row_bytes as u64 > slot_bytes {
                    return self.refuse("rows do not fit the slot");
                }
                let Some(slot) = self.slots.get_mut(slot_id) else {
                    return self.refuse("unknown slot");
                };
                match slot.parts.get_mut(part) {
                    Some(PartData::Cells { rows, n: logical }) => {
                        for chunk in bytes.chunks(row_bytes) {
                            rows.push(chunk.to_vec());
                        }
                        *logical = rows.len() as u64;
                        WriteResult::Ok
                    }
                    _ => unreachable!("the part kind was checked as cells"),
                }
            }
            PartKind::Ring => {
                let ring = match self.slots.get(slot_id).and_then(|slot| slot.parts.get(part)) {
                    Some(PartData::Ring { rows, base }) => Some((*base, rows.len())),
                    _ => return self.refuse("part is not a ring"),
                };
                let (base, length) = ring.expect("the ring was checked");
                if length > 0 && from != base + length as u64 {
                    return self.refuse("the ring cannot place rows at this position");
                }
                let Some(slot) = self.slots.get_mut(slot_id) else {
                    return self.refuse("unknown slot");
                };
                match slot.parts.get_mut(part) {
                    Some(PartData::Ring { rows, base }) => {
                        if rows.is_empty() && !self.ignore_ring_from {
                            *base = from;
                        }
                        for chunk in bytes.chunks(row_bytes) {
                            rows.push_back(chunk.to_vec());
                        }
                        while rows.len() > RING_WINDOW {
                            rows.pop_front();
                            *base += 1;
                        }
                        WriteResult::Ok
                    }
                    _ => unreachable!("the part kind was checked as a ring"),
                }
            }
            _ => self.refuse("part is not cells or ring"),
        }
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let slot_id = Self::slot_id(state);
        let rows = match self.slots.get(slot_id).and_then(|slot| slot.parts.get(CELLS_PART)) {
            Some(PartData::Cells { rows, .. }) => rows.len() as u64,
            _ => return self.refuse("part is not cells"),
        };
        if n > rows {
            return self.refuse("n is larger than the stored rows");
        }
        let Some(slot) = self.slots.get_mut(slot_id) else {
            return self.refuse("unknown slot");
        };
        match slot.parts.get_mut(CELLS_PART) {
            Some(PartData::Cells { n: logical, .. }) => {
                *logical = n;
                WriteResult::Ok
            }
            _ => unreachable!("the part kind was checked as cells"),
        }
    }
}

impl Slots for Fake {
    /// The highest row that any row part of the slot holds.
    fn slot_high(&self, slot: SlotId) -> u64 {
        let Some(slot) = self.slots.get(slot as usize) else {
            return 0;
        };
        slot.parts.iter().fold(0, |high, part| match part {
            PartData::Cells { n, .. } => high.max(*n),
            PartData::Ring { rows, base } => high.max(*base + rows.len() as u64),
            PartData::Fixed(_) | PartData::Derived => high,
        })
    }

    /// Clear every part of the slot.
    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        let Some(slot) = self.slots.get_mut(slot as usize) else {
            return self.refuse("unknown slot");
        };
        for part in &mut slot.parts {
            match part {
                PartData::Fixed(bytes) => bytes.fill(0),
                PartData::Cells { rows, n } => {
                    rows.clear();
                    *n = 0;
                }
                PartData::Ring { rows, base } => {
                    rows.clear();
                    *base = 0;
                }
                PartData::Derived => {}
            }
        }
        WriteResult::Ok
    }
}

impl Coverage for Fake {
    fn coverage(&self, state: State, part: usize, bytes: &[u8]) -> CoverageSpan {
        let row_bytes = self.row_bytes(part).map(|bytes| bytes.max(1)).unwrap_or(1);
        let rows = match self.limits.parts.get(part) {
            Some(part) if part.is_row() => bytes.len() as u64 / row_bytes as u64,
            _ => 0,
        };
        let start = match self
            .slots
            .get(Self::slot_id(state))
            .and_then(|slot| slot.parts.get(part))
        {
            Some(PartData::Ring { base, .. }) => *base,
            _ => 0,
        };
        CoverageSpan {
            part,
            start,
            end: start + rows,
        }
    }

    fn fits(&self, coverage: &CoverageSpan, slot: u32) -> Result<(), String> {
        self.fits_inner(coverage, slot)
    }
}

impl Seal for Fake {
    fn seal(&mut self, state: State) -> Result<(), SealError> {
        let slot_id = Self::slot_id(state);
        let Some(slot) = self.slots.get_mut(slot_id) else {
            return Err(SealError::Refused);
        };
        match slot.parts.get_mut(STATE_PART) {
            Some(PartData::Fixed(bytes)) if !bytes.is_empty() => {
                bytes[0] ^= 1;
                Ok(())
            }
            _ => Err(SealError::Refused),
        }
    }
}
