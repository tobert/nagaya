//! The fake engines behind the feature `fake`.

use std::collections::{BTreeMap, VecDeque};

use crate::{
    Capability, ChainLimits, Coverage, CoverageSpan, Determinism, Dtype, ElasticTerm, Engine, Identity, Inputs, Kind,
    Lifetime, Limits, MandatoryTerm, MemoryLimits, Output, PartError, PartKind, PartLimits, Parts, PassError,
    PassErrorKind, PassOut, PassStatus, Seal, SealError, Segment, SegmentInputs, SlotLimit, Stage, State, Tap, Tier,
    WriteResult,
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

impl FakeAlways {
    /// Make the engine.
    pub fn new() -> Self {
        Self::with_slots(default_slots())
    }

    /// Make the engine with the given slot size classes.
    pub fn with_slots(slots: Vec<SlotLimit>) -> Self {
        FakeAlways(Fake::new("fake-always", false, slots))
    }
}

impl FakeBroken {
    /// Make the broken engine.
    pub fn new() -> Self {
        FakeBroken(Fake::new("fake-broken", true, default_slots()))
    }
}

fn default_slots() -> Vec<SlotLimit> {
    vec![SlotLimit { bytes: 64, count: 1 }, SlotLimit { bytes: 256, count: 1 }]
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

/// The state of a fake engine.
pub(crate) struct Fake {
    model: String,
    broken: bool,
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
    fn new(model: &str, broken: bool, slot_limits: Vec<SlotLimit>) -> Self {
        let parts = vec![
            PartLimits {
                name: "state".to_owned(),
                kind: PartKind::Fixed,
                dtype: Dtype::U8,
                shape: vec![16],
            },
            PartLimits {
                name: "cells".to_owned(),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![8],
            },
            PartLimits {
                name: "ring".to_owned(),
                kind: PartKind::Ring,
                dtype: Dtype::U8,
                shape: vec![8],
            },
            PartLimits {
                name: "pooled".to_owned(),
                kind: PartKind::Derived,
                dtype: Dtype::U8,
                shape: vec![8],
            },
        ];
        let limits = Limits {
            capabilities: vec![Capability::Parts, Capability::Coverage, Capability::Seal],
            parts: parts.clone(),
            inputs: Inputs::Tokens,
            devices: vec![0],
            stages: vec![Stage {
                device: 0,
                layers: 0..1,
                head: true,
            }],
            max_segments: 1,
            max_pass_inputs: 1,
            max_logit_rows: 1,
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
        };
        let slots = (0..limits.slots.iter().map(|slot| slot.count).sum())
            .map(|_| SlotData::new(&parts))
            .collect();
        Fake {
            model: model.to_owned(),
            broken,
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
        let mut outputs = Vec::new();
        for segment in segments {
            let tokens = match &segment.inputs {
                SegmentInputs::Tokens(tokens) => tokens,
                _ => {
                    return Err(PassError {
                        kind: PassErrorKind::Ran(PassStatus::Poison),
                        message: "the fake takes tokens".to_owned(),
                    });
                }
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
            if self.cells_write(segment.state, CELLS_PART, count, &rows) != WriteResult::Ok {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the fake could not append the cells".to_owned(),
                });
            }
            if self.cells_write(segment.state, RING_PART, count, &rows) != WriteResult::Ok {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the fake could not advance the ring".to_owned(),
                });
            }
            let state_bytes = state_from_tokens(tokens);
            if self.part_write(segment.state, STATE_PART, &state_bytes) != WriteResult::Ok {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the fake could not write the state".to_owned(),
                });
            }
            outputs.push(Output {
                name: "state".to_owned(),
                bytes: state_bytes.to_vec(),
            });
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 0,
            outputs,
        })
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

fn state_from_tokens(tokens: &[u32]) -> [u8; 16] {
    let mut state = [0u8; 16];
    for (index, token) in tokens.iter().enumerate() {
        for (offset, byte) in token.to_le_bytes().iter().enumerate() {
            let slot = (index + offset) % 16;
            state[slot] = state[slot].wrapping_mul(31).wrapping_add(*byte);
        }
    }
    state
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
        }
    };
}

engine_impl!(FakeAlways);
engine_impl!(FakeBroken);

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

    fn cells_write(&mut self, state: State, part: usize, n: u64, bytes: &[u8]) -> WriteResult {
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
                let Some(slot) = self.slots.get_mut(slot_id) else {
                    return self.refuse("unknown slot");
                };
                match slot.parts.get_mut(part) {
                    Some(PartData::Ring { rows, base }) => {
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
