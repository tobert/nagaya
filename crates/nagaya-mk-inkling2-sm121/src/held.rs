//! The rules of the held state of a slot: the token count, the writes of a restore, the count of cycles, and the checks
//! of a pass and of the part calls. They need no GPU. The engine applies the device side.
//!
//! This module mirrors the `Held` and the `Block` of the B70 engine (`nagaya_mk_inkling2_bmg::spec`), with these
//! differences. A sliding state keeps a ring of `RING_ROWS` rows with no spill. The engine has no media, no control
//! vectors and no taps, so a segment names tokens only. The cache has a format, so a row has the bytes of that format.
//! Proposed: the two engines may share one crate later.

use std::collections::BTreeMap;
use std::ops::Range;

use nagaya_abi::{CoverageSpan, CyclesLimit, PartId, Segment, SegmentInputs, State};
use nagaya_mk_cuda::attn::CacheFormat;

use crate::spec::{
    CONV_ROW, CONV_ROWS, CYCLE_LAYERS, KV_HEADS, LAYERS, LOGITS, MAX_LOGIT_ROWS, MAX_SEGMENTS, RING_ROWS, WINDOW,
    cycles_limit, is_full, state_layer,
};

/// One part of the held state (engine-abi "Inkling 2", "The parts").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The keys of a state.
    K(usize),
    /// The values of a state.
    V(usize),
    /// The conv state of a state: a fixed part.
    Conv(usize),
    /// The count of cycles of the slot: a fixed part, `u32 [1]`, the last part.
    Cycles,
}

impl Part {
    /// The state of the slot that holds the part. The part `cycles` has no state, so a call for it is a defect.
    pub fn state(self) -> usize {
        match self {
            Part::K(s) | Part::V(s) | Part::Conv(s) => s,
            Part::Cycles => panic!("the part cycles has no state of a run; the caller must match it first"),
        }
    }
}

/// Where the bytes of a part read come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The memory of the card.
    Device,
    /// Zeros: a run that the count of the slot does not reach.
    Zeros,
    /// The count of cycles of the slot, or 0 for an empty slot.
    Count(u32),
}

/// True when a pass of `cycles` cycles runs state `s`: a state outside the reserve, or an entry of the reserve up to
/// the count.
pub fn runs_at(cycles: usize, s: usize) -> bool {
    s < LAYERS || s - LAYERS < CYCLE_LAYERS * cycles.saturating_sub(1)
}

/// The parts in state order: `k`, `v` and `conv` for each layer, then the same for each entry of a reserve of `extra`
/// runs, then `cycles`. The part id is the index in this list, and `spec::parts` declares the same order.
pub fn part_list(extra: usize) -> Vec<Part> {
    (0..LAYERS + extra)
        .flat_map(|s| [Part::K(s), Part::V(s), Part::Conv(s)])
        .chain([Part::Cycles])
        .collect()
}

/// True when a state keeps a ring in a slot of `seq` positions: a sliding layer in a slot longer than the ring. In a
/// slot of `RING_ROWS` positions or fewer the ring does not wrap, so the state still holds every position.
pub fn ringed(seq: usize, s: usize) -> bool {
    !is_full(state_layer(s)) && seq > RING_ROWS
}

/// The bytes of one row of a row part in a cache of `format`: 8 heads of 128 values.
pub fn row_bytes(format: CacheFormat) -> u64 {
    (KV_HEADS * format.row_bytes()) as u64
}

/// The bytes of a conv part: 3 rows of 6,656 `f32` values. The conv state is `f32` in each cache format.
pub const CONV_BYTES: u64 = (CONV_ROWS * CONV_ROW * 4) as u64;

/// Check that `bytes` hold only values that a cache of `format` can hold, and that the bytes make whole values.
///
/// A cache holds finite numbers. The code of a NaN or an infinity in the format is a value that the format cannot hold
/// (the e4m3 format has a NaN and no infinity). The message names the first bad value.
pub fn check_values(format: CacheFormat, bytes: &[u8]) -> Result<(), String> {
    let width = match format {
        CacheFormat::F32 => 4,
        CacheFormat::Bf16 => 2,
        CacheFormat::E4m3 => 1,
    };
    if !bytes.len().is_multiple_of(width) {
        return Err(format!(
            "{} bytes are not whole values of {width} bytes in a cache of this format",
            bytes.len()
        ));
    }
    let bad = bytes.chunks_exact(width).position(|value| match format {
        CacheFormat::F32 => (u32::from_le_bytes([value[0], value[1], value[2], value[3]]) >> 23) & 0xff == 0xff,
        CacheFormat::Bf16 => (u16::from_le_bytes([value[0], value[1]]) >> 7) & 0xff == 0xff,
        CacheFormat::E4m3 => value[0] & 0x7f == 0x7f,
    });
    match bad {
        None => Ok(()),
        Some(index) => Err(format!(
            "value {index} is a NaN or an infinity, and a cache of this format holds only finite values; write the \
             bytes that a read gave"
        )),
    }
}

/// One block that a pass runs, after its checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub slot: usize,
    /// The position of the first token. It is the token count of the slot before the pass.
    pub pos0: usize,
    pub tokens: Vec<u32>,
    /// The rows of the block whose logits the pass gives, in request order. Each is an index into the block.
    pub logit_rows: Vec<usize>,
    /// The count of cycles of the pass: the count of the segment, or the default.
    pub cycles: usize,
}

impl Block {
    /// True when the pass starts at position 0. Then each run starts from conv states of zeros, as the reference
    /// does (engine-abi "Inkling 2", "Position 0 and no way back").
    pub fn zero_conv(&self) -> bool {
        self.pos0 == 0
    }
}

/// One write of a restore: the positions of a row part, and whether its bytes are all zeros.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Write {
    pub rows: Option<Range<u64>>,
    pub zero: bool,
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
    /// The format of the cache, which sets the bytes of a row.
    pub format: CacheFormat,
    /// The parts that got a write since the last `set_n`, with the positions of a row part.
    writes: Vec<BTreeMap<PartId, Write>>,
    /// The count of cycles that a restore wrote since the last `set_n`.
    pending: Vec<Option<u32>>,
    /// The first position that the rings of each slot hold since its last restore: 0, or the start of the restored
    /// rows.
    ring_from: Vec<u64>,
    /// The count of cycles that fed each slot, or `None` when no pass fed its positions: an empty slot, or a slot
    /// after a restore. The next pass then sets it.
    pub cycles: Vec<Option<usize>>,
    /// The counts of cycles that a pass can run.
    limit: CyclesLimit,
}

impl Held {
    /// The rules of slots of `seqs` positions, each with a reserve of `extra` runs and a cache in `format`.
    pub fn new(seqs: &[usize], extra: usize, format: CacheFormat) -> Self {
        Self {
            list: part_list(extra),
            seqs: seqs.to_vec(),
            n: vec![0; seqs.len()],
            format,
            writes: vec![BTreeMap::new(); seqs.len()],
            pending: vec![None; seqs.len()],
            ring_from: vec![0; seqs.len()],
            cycles: vec![None; seqs.len()],
            limit: cycles_limit(extra),
        }
    }

    /// The bytes of one row of a row part.
    pub fn row_bytes(&self) -> u64 {
        row_bytes(self.format)
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
    ///
    /// An inactive ring reads as zeros, so it holds the last `RING_ROWS` positions with no restored start.
    pub fn base(&self, slot: usize, s: usize) -> u64 {
        if !ringed(self.seqs[slot], s) {
            return 0;
        }
        let last = self.n[slot].saturating_sub(RING_ROWS as u64);
        if self.active(slot, s) {
            last.max(self.ring_from[slot]).min(self.n[slot])
        } else {
            last
        }
    }

    /// True when the count of `slot` runs state `s`. A slot with no count, empty or in a restore, runs each state.
    pub fn active(&self, slot: usize, s: usize) -> bool {
        self.cycles[slot].is_none_or(|c| runs_at(c, s))
    }

    /// The source of a read of state `s` of `slot`.
    fn source(&self, slot: usize, s: usize) -> Source {
        if self.active(slot, s) {
            Source::Device
        } else {
            Source::Zeros
        }
    }

    /// Check a pass before it changes anything, and give its block. The message says which rule breaks.
    ///
    /// - A pass has one segment of 1 to `max_block` tokens below `LOGITS`. The segment has tokens only: no rows, no
    ///   meta and no control items.
    /// - The segment names no layer loop, and its count of cycles fits the limit. A slot that holds positions runs the
    ///   count that fed them.
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
        let tokens = match &segment.inputs {
            SegmentInputs::Tokens(tokens) => tokens,
            SegmentInputs::TokensAndRows { .. } | SegmentInputs::Rows(_) => {
                return Err(
                    "the segment has rows; the engine takes tokens only, until it runs the tower of images".to_owned(),
                );
            }
        };
        if !segment.meta.is_empty() {
            return Err("the segment has meta; the engine takes no image spans yet".to_owned());
        }
        if !segment.control.is_empty() {
            return Err("the segment has control items; the engine does not declare `control` yet".to_owned());
        }
        if segment.layer_loop.is_some() {
            return Err(
                "the segment names a layer loop; the Inkling 2 engine runs cycles, and it does not declare \
                 `layer_loops`"
                    .to_owned(),
            );
        }
        let cycles = usize::from(self.limit.resolve(segment.cycles));
        self.limit.check(cycles as u8)?;
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
        if let Some(fed) = self.cycles[slot]
            && held > 0
            && fed != cycles
        {
            return Err(format!(
                "slot {slot} holds {held} positions of {fed} cycles, and the pass runs {cycles}; each run of a cycle \
                 reads the keys of its own run at the earlier positions. The council keys a state by its count, so \
                 feed the slot at {fed} cycles, or clear it first"
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
            cycles,
        })
    }

    /// Check a probe of the oracle (8.3): `ids` from position 0 of `slot` at the default count, as if the slot were
    /// empty. It changes nothing, so the caller empties the slot only after the check passes.
    pub fn check_probe(&self, slot: usize, ids: &[u32], max_block: usize) -> Result<Block, String> {
        // The probe segment names the slot as `u32`, so the check of the slot comes before that cast.
        if slot >= self.n.len() {
            return Err(format!("the engine has {} slots, not slot {slot}", self.n.len()));
        }
        let mut empty = self.clone();
        empty.clear(slot);
        empty.check_pass(&[probe_segment(slot, ids)], max_block)
    }

    /// Take a pass that ran: the slot holds the block's positions, and the writes before it are gone.
    pub fn ran(&mut self, block: &Block) {
        self.n[block.slot] = (block.pos0 + block.tokens.len()) as u64;
        self.writes[block.slot].clear();
        self.pending[block.slot] = None;
        self.cycles[block.slot] = Some(block.cycles);
    }

    fn row_part(&self, state: State, id: PartId) -> Result<(usize, Part), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        match self.part(id) {
            Some(part @ (Part::K(_) | Part::V(_))) => Ok((slot, part)),
            Some(Part::Conv(_)) => Err(format!(
                "part {id} is a conv state, a fixed part; use part_read and part_write"
            )),
            Some(Part::Cycles) => Err(format!(
                "part {id} is the count of cycles, a fixed part; use part_read and part_write"
            )),
            None => Err(format!("the engine has no part {id}")),
        }
    }

    /// Check a read of `n` rows from `from` of a row part, and give its slot, its part and the source of its bytes.
    pub fn check_read(
        &self,
        state: State,
        id: PartId,
        from: u64,
        n: u64,
        len: usize,
    ) -> Result<(usize, Part, Source), String> {
        let (slot, part) = self.row_part(state, id)?;
        let base = self.base(slot, part.state());
        let row = self.row_bytes();
        if len as u64 != n * row || from < base || from + n > self.n[slot] {
            return Err(format!(
                "a read of rows {from} to {} of part {id} in {len} bytes does not fit: slot {slot} holds rows {base} to \
                 {} of {row} bytes",
                from + n,
                self.n[slot],
            ));
        }
        Ok((slot, part, self.source(slot, part.state())))
    }

    /// Check a whole read of a part into `len` bytes, and give the first row for a row part and the source of the
    /// bytes. The part `cycles` reads the count of the slot, or 0 for an empty slot.
    pub fn check_part_read(&self, state: State, id: PartId, len: usize) -> Result<(usize, Part, u64, Source), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        let part = self.part(id).ok_or_else(|| format!("the engine has no part {id}"))?;
        match part {
            Part::Conv(s) if len as u64 == CONV_BYTES => Ok((slot, part, 0, self.source(slot, s))),
            Part::Conv(_) => Err(format!("part {id} is a conv state of {CONV_BYTES} bytes, not {len}")),
            Part::Cycles if len == 4 => Ok((slot, part, 0, Source::Count(self.count(slot)))),
            Part::Cycles => Err(format!("part {id} is the count of cycles, 4 bytes, not {len}")),
            _ => {
                let row = self.row_bytes();
                if !(len as u64).is_multiple_of(row) {
                    return Err(format!("{len} bytes are not whole rows of {row} bytes"));
                }
                let from = self.base(slot, part.state());
                let (_, _, source) = self.check_read(state, id, from, len as u64 / row, len)?;
                Ok((slot, part, from, source))
            }
        }
    }

    /// The count of cycles of `slot`: the count that fed its positions, or 0 for an empty slot.
    pub fn count(&self, slot: usize) -> u32 {
        if self.n[slot] == 0 {
            return 0;
        }
        let cycles = self.cycles[slot].expect("a slot that holds positions has a count of cycles");
        cycles as u32
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
        if len as u64 != n * self.row_bytes() || from + n > seq as u64 {
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

    /// Check a whole write of a fixed part, and give its slot and part. The part `cycles` takes a count of 1 to the
    /// most of the limit, or 0 for an empty slot, and one count in each restore.
    pub fn check_part_write(&self, state: State, id: PartId, bytes: &[u8]) -> Result<(usize, Part), String> {
        let slot = self.slot(state).ok_or_else(|| format!("the engine has no {state:?}"))?;
        let len = bytes.len();
        match self.part(id) {
            Some(part @ Part::Conv(_)) if len as u64 == CONV_BYTES => {
                check_values(CacheFormat::F32, bytes)?;
                Ok((slot, part))
            }
            Some(Part::Conv(_)) => Err(format!("part {id} is a conv state of {CONV_BYTES} bytes, not {len}")),
            Some(Part::Cycles) => {
                let Ok(word) = <[u8; 4]>::try_from(bytes) else {
                    return Err(format!("part {id} is the count of cycles, 4 bytes, not {len}"));
                };
                let count = u32::from_le_bytes(word);
                if count > u32::from(self.limit.max) {
                    return Err(format!(
                        "the part cycles holds the count of cycles {count}; the engine runs 1 to {}, and 0 is an \
                         empty slot",
                        self.limit.max
                    ));
                }
                if let Some(had) = self.pending[slot]
                    && had != count
                {
                    return Err(format!(
                        "slot {slot} got the count {had} in this restore, and this write names {count}; clear the \
                         slot first to restore another count"
                    ));
                }
                Ok((slot, Part::Cycles))
            }
            Some(_) => Err(format!("part {id} is a row part; write its rows with cells_write")),
            None => Err(format!("the engine has no part {id}")),
        }
    }

    /// Take a write that the device made: the slot holds no token until `set_n`. `zero` is true when its bytes are
    /// all zeros, the bytes of an inactive run.
    pub fn wrote(&mut self, slot: usize, id: PartId, rows: Option<Range<u64>>, zero: bool) {
        self.n[slot] = 0;
        self.cycles[slot] = None;
        self.writes[slot].insert(id, Write { rows, zero });
    }

    /// Take a write of the part `cycles` that `check_part_write` passed.
    pub fn wrote_count(&mut self, slot: usize, count: u32) {
        let id = self.list.len() - 1;
        self.wrote(slot, id, None, true);
        self.pending[slot] = Some(count);
    }

    /// True when a pass of `cycles` cycles runs state `s` (`runs_at`).
    pub fn runs_at(&self, cycles: usize, s: usize) -> bool {
        runs_at(cycles, s)
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
        if let Some(id) = (0..self.list.len()).find(|id| !writes.contains_key(id)) {
            return Err(format!("part {id} got no bytes since the last set_n"));
        }
        let cycles = self.pending[slot].unwrap_or(0) as usize;
        if cycles == 0 && n > 0 {
            return Err(format!(
                "the restore wrote the count of cycles 0, and set_n names {n} positions; a count of 0 restores only an \
                 empty slot"
            ));
        }
        for (id, part) in self.list.iter().enumerate() {
            let write = &writes[&id];
            let inactive = match part {
                Part::Cycles => continue,
                Part::K(s) | Part::V(s) | Part::Conv(s) => n > 0 && !runs_at(cycles, *s),
            };
            if inactive && !write.zero {
                return Err(format!(
                    "part {id} is a run that {cycles} cycles do not reach, so it must hold zeros; the restore wrote \
                     other bytes"
                ));
            }
            match (part, &write.rows) {
                (Part::Conv(_), _) => {}
                (_, None) => return Err(format!("part {id} got no rows")),
                (_, Some(rows)) => {
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
        // The base of a slot is one position for all its active rings, so they must start at one position.
        let mut ring_start: Option<(PartId, u64)> = None;
        for (id, part) in self.list.iter().enumerate() {
            let write = &writes[&id];
            if let (Part::K(s) | Part::V(s), Some(rows)) = (part, &write.rows)
                && n > 0
                && ringed(seq, *s)
                && runs_at(cycles, *s)
            {
                match ring_start {
                    None => ring_start = Some((id, rows.start)),
                    Some((first, start)) if start != rows.start => {
                        return Err(format!(
                            "the active rings of a restore must start at one position: part {first} starts at \
                             {start}, and part {id} at {}. Restore a snapshot that the engine read",
                            rows.start
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(slot)
    }

    /// Take a count that `check_set_n` passed. After writes, the slot takes the restored count of cycles, and the
    /// rings hold the positions from the start of the rows of the active ring parts, which `check_set_n` found equal.
    pub fn set_n(&mut self, slot: usize, n: u64) {
        let seq = self.seqs[slot];
        let restored = (!self.writes[slot].is_empty() && n > 0).then(|| self.pending[slot].unwrap_or(0) as usize);
        let from = restored.and_then(|cycles| {
            self.writes[slot]
                .iter()
                .filter_map(|(id, write)| match (self.list[*id], &write.rows) {
                    (Part::K(s) | Part::V(s), Some(rows)) if ringed(seq, s) && runs_at(cycles, s) => Some(rows.start),
                    _ => None,
                })
                .max()
        });
        if n == 0 {
            self.ring_from[slot] = 0;
            self.cycles[slot] = None;
        } else if let Some(cycles) = restored {
            self.cycles[slot] = Some(cycles);
            if let Some(from) = from {
                self.ring_from[slot] = from;
            }
        }
        self.n[slot] = n;
        self.writes[slot].clear();
        self.pending[slot] = None;
    }

    /// Empty a slot. It forgets the count, the restored start of the rings and the writes of a restore. The engine
    /// zeroes the conv state and the count on the card; the cache rows stay.
    pub fn clear(&mut self, slot: usize) {
        self.n[slot] = 0;
        self.ring_from[slot] = 0;
        self.writes[slot].clear();
        self.pending[slot] = None;
        self.cycles[slot] = None;
    }

    /// The first position of the rings of `slot`: 0, or the start of the restored rows. A test and the engine read it.
    pub fn ring_from(&self, slot: usize) -> u64 {
        self.ring_from[slot]
    }

    /// The positions that `bytes` of a part hold in a state, from the base of the part. A conv part holds none.
    pub fn coverage(&self, state: State, id: PartId, bytes: &[u8]) -> CoverageSpan {
        match (self.slot(state), self.part(id)) {
            (Some(slot), Some(part @ (Part::K(_) | Part::V(_)))) => {
                let start = self.base(slot, part.state());
                CoverageSpan {
                    part: id,
                    start,
                    end: start + bytes.len() as u64 / self.row_bytes(),
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
        match part {
            Part::Conv(_) => return Err(format!("part {} is a conv state; it has no positions", span.part)),
            Part::Cycles => {
                return Err(format!(
                    "part {} is the count of cycles; it has no positions",
                    span.part
                ));
            }
            Part::K(_) | Part::V(_) => {}
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

/// The segment of a probe: `ids` from position 0 of `slot`, with no logit row.
pub fn probe_segment(slot: usize, ids: &[u32]) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot as u32),
        inputs: SegmentInputs::Tokens(ids.to_vec()),
        positions: (0..ids.len() as u64).collect(),
        meta: Vec::new(),
        outputs: Default::default(),
        layer_loop: None,
        cycles: None,
    }
}

#[cfg(test)]
mod tests;
