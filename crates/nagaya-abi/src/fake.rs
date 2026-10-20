//! The fake engines behind the feature `fake`.

use std::collections::{BTreeMap, VecDeque};

use crate::{
    Capability, ChainLimits, Coverage, CoverageSpan, Determinism, Dtype, ElasticTerm, Engine, Identity, Inputs, Kind,
    Lifetime, Limits, MandatoryTerm, MemoryLimits, Output, PartError, PartKind, PartLimits, Parts, PassError,
    PassErrorKind, PassOut, PassStatus, Seal, SealError, Segment, SegmentInputs, SlotId, SlotLimit, Slots, Stage,
    State, Tap, Tier, WriteResult,
};

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
    /// Give the engine logits over a vocabulary of `vocab` tokens (engine-abi, "Logit outputs").
    ///
    /// Each requested logit row is a function of the fixed state after the tokens up to its position.
    pub fn with_vocab(mut self, vocab: usize) -> Self {
        assert!(vocab > 0, "a vocabulary has at least one token");
        self.0.vocab = Some(vocab);
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
    vec![SlotLimit { bytes: 64, count: 1 }, SlotLimit { bytes: 256, count: 7 }]
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
    limits: Limits,
    slots: Vec<SlotData>,
    last_error: Option<String>,
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
            },
            PartLimits {
                name: "cells".to_owned(),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![8],
                block_format: None,
            },
            PartLimits {
                name: "ring".to_owned(),
                kind: PartKind::Ring,
                dtype: Dtype::U8,
                shape: vec![8],
                block_format: None,
            },
            PartLimits {
                name: "pooled".to_owned(),
                kind: PartKind::Derived,
                dtype: Dtype::U8,
                shape: vec![8],
                block_format: None,
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
            limits,
            slots,
            last_error: None,
        }
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn identity(&self) -> Identity {
        Identity {
            model: self.model.clone(),
            checkpoint_digest: "fake:0".to_owned(),
            tokenizer_hash: "fake:0".to_owned(),
            template: "fake".to_owned(),
            numbers: BTreeMap::new(),
            plan_table: "fake".to_owned(),
            bit_settings: Vec::new(),
            canary_hash: "fake:0".to_owned(),
        }
    }

    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
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
            let mut logit_outputs = Vec::new();
            if let Some(vocab) = self.vocab {
                // Fold one token at a time, so that a row sees the state at its own position.
                let mut folded = state_bytes;
                let mut rows: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
                for (index, token) in tokens.iter().enumerate() {
                    let position = segment.positions.get(index).copied().unwrap_or(index as u64);
                    fold_token(&mut folded, *token, position);
                    if segment.outputs.logit_rows.contains(&position) {
                        rows.insert(position, logit_row(&folded, vocab));
                    }
                }
                // The output order is the request order.
                for row in &segment.outputs.logit_rows {
                    logit_outputs.push(Output {
                        name: "logits".to_owned(),
                        dtype: Dtype::F32,
                        bytes: rows[row].clone(),
                    });
                }
            }
            fold_tokens(&mut state_bytes, tokens, &segment.positions);
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
            outputs.push(Output {
                name: "state".to_owned(),
                dtype: Dtype::U8,
                bytes: state_bytes.to_vec(),
            });
            outputs.extend(logit_outputs);
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 0,
            outputs,
        })
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

/// Fold the tokens into the state, one after another. The result does not depend on how a pass splits the tokens.
fn fold_tokens(state: &mut [u8; 16], tokens: &[u32], positions: &[u64]) {
    for (index, token) in tokens.iter().enumerate() {
        let position = positions.get(index).copied().unwrap_or(index as u64);
        fold_token(state, *token, position);
    }
}

fn fold_token(state: &mut [u8; 16], token: u32, position: u64) {
    for (offset, byte) in token.to_le_bytes().iter().enumerate() {
        let slot = (position as usize + offset) % 16;
        state[slot] = state[slot].wrapping_mul(31).wrapping_add(*byte);
    }
}

/// One logit row: `vocab` little-endian F32 values between -6 and 6, a function of the state bytes only.
fn logit_row(state: &[u8; 16], vocab: usize) -> Vec<u8> {
    let seed = state
        .iter()
        .fold(0x6C6F_6769_7473_u64, |hash, byte| mix(hash ^ u64::from(*byte)));
    let mut bytes = Vec::with_capacity(vocab * 4);
    for token in 0..vocab as u64 {
        // 24 bits fill an F32 mantissa, so the value is exact.
        let unit = (mix(seed ^ mix(token + 1)) >> 40) as f32 / (1u32 << 24) as f32;
        bytes.extend_from_slice(&(unit * 12.0 - 6.0).to_le_bytes());
    }
    bytes
}

/// A 64-bit mixing function (splitmix64).
fn mix(value: u64) -> u64 {
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
        }
    };
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
