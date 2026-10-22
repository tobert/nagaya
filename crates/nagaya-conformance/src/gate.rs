//! The conformance gates: the properties of spec 8.3 that need no GPU.
//!
//! A fill writes synthetic bytes into every stored part. An engine with a value rule refuses them, as mk refuses a
//! PLE history that is not a token id. The gate then starts the state with a one-token pass and reads it back, so
//! every part holds a value that the engine accepts. Both paths call `set_n` after the parts.

use std::fmt;

use nagaya_abi::{
    CoverageSpan, Dtype, Engine, Inputs, Limits, Output, OutputsWanted, PartId, PartKind, PartLimits, PassOut,
    PassStatus, Segment, SegmentInputs, State, WriteResult,
};
use nagaya_ledger::{Admission, Ledger, LedgerError, Profile};
use nagaya_store::{EngineFacts, Put, Reason, Restored, Snapshot, Store, StoreError, Tensor, Tier, recipe_root};

use crate::{Failure, compare_bytes};

/// The result of one gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The engine passed the gate.
    Pass,
    /// The engine failed the gate.
    Fail(Failure),
    /// The gate does not apply to this engine, with the reason.
    NotApplicable(String),
}

impl Outcome {
    /// True when the gate failed.
    pub fn failed(&self) -> bool {
        matches!(self, Outcome::Fail(_))
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Pass => write!(formatter, "pass"),
            Outcome::Fail(failure) => write!(formatter, "fail: {failure}"),
            Outcome::NotApplicable(reason) => write!(formatter, "not applicable: {reason}"),
        }
    }
}

/// A ledger that can admit a profile. The gate uses it; the broken variant changes state on a refusal.
pub trait Admit {
    /// Admit a profile.
    fn admit(&mut self, profile: &Profile) -> Result<Admission, LedgerError>;
}

impl Admit for Ledger {
    fn admit(&mut self, profile: &Profile) -> Result<Admission, LedgerError> {
        Ledger::admit(self, profile)
    }
}

/// The whole size of a fixed part, or the row size of a cells or ring part.
fn shape_bytes(part: &PartLimits) -> u64 {
    part.shape.iter().product::<u64>() * part.dtype.element_bytes()
}

fn second_slot(limits: &Limits) -> Option<u32> {
    let total: u32 = limits.slots.iter().map(|class| class.count).sum();
    (total >= 2).then_some(1)
}

fn smallest_slot(limits: &Limits) -> Option<(u32, u64)> {
    let mut id = 0u32;
    let mut best: Option<(u32, u64)> = None;
    for class in &limits.slots {
        for _ in 0..class.count {
            if best.is_none_or(|(_, bytes)| class.bytes < bytes) {
                best = Some((id, class.bytes));
            }
            id += 1;
        }
    }
    best
}

/// Synthetic bytes for one part. A float part gets finite values, because the suite refuses a non-finite one.
fn pattern(len: usize, seed: u8, dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F32 => (0..len / 4)
            .flat_map(|index| {
                let value = ((index % 97) as f32) * 0.25 - 7.0 + f32::from(seed) * 0.5;
                value.to_le_bytes()
            })
            .collect(),
        Dtype::F16 => (0..len / 2)
            .flat_map(|index| (0x3c00u16 | ((index as u16) % 0x3ff)).to_le_bytes())
            .collect(),
        Dtype::BF16 => (0..len / 2)
            .flat_map(|index| (0x3f80u16 | ((index as u16) % 0x7f)).to_le_bytes())
            .collect(),
        // The mask clears the top exponent bit of an F8 value, so that the value is finite.
        Dtype::F8E4M3 | Dtype::F8E5M2 => (0..len)
            .map(|index| (((index as u8).wrapping_mul(37).wrapping_add(seed)) | 1) & 0xbf)
            .collect(),
        _ => (0..len)
            .map(|index| ((index as u8).wrapping_mul(37).wrapping_add(seed)) | 1)
            .collect(),
    }
}

/// The first index of a non-finite value in a float part, or `None` for an integer part.
fn non_finite(dtype: Dtype, bytes: &[u8]) -> Option<usize> {
    // A value is non-finite when its bits under the mask equal the pattern. E4M3 has a NaN and no Inf.
    let (width, mask, pattern) = match dtype {
        Dtype::F32 => (4, 0x7f80_0000u32, 0x7f80_0000u32),
        Dtype::F16 => (2, 0x7c00, 0x7c00),
        Dtype::BF16 => (2, 0x7f80, 0x7f80),
        Dtype::F8E5M2 => (1, 0x7c, 0x7c),
        Dtype::F8E4M3 => (1, 0x7f, 0x7f),
        _ => return None,
    };
    bytes.chunks_exact(width).position(|chunk| {
        let bits = match width {
            4 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            2 => u32::from(u16::from_le_bytes([chunk[0], chunk[1]])),
            _ => u32::from(chunk[0]),
        };
        bits & mask == pattern
    })
}

/// The number of rows that `fill` leaves in a cells part: one.
fn filled_rows(limits: &Limits) -> u64 {
    u64::from(limits.parts.iter().any(|part| part.kind == PartKind::Cells))
}

fn fill<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    seed: u8,
    gate: &'static str,
) -> Result<Vec<(PartId, Vec<u8>)>, Failure> {
    let mut written = Vec::new();
    let mut cells = false;
    for (id, part) in limits.parts.iter().enumerate() {
        if !part.is_stored() {
            continue;
        }
        let bytes = pattern(shape_bytes(part) as usize, seed.wrapping_add(id as u8), part.dtype);
        let result = match part.kind {
            PartKind::Fixed => engine.parts().unwrap().part_write(state, id, &bytes),
            PartKind::Cells | PartKind::Ring => engine.parts().unwrap().cells_write(state, id, 0, 1, &bytes),
            PartKind::Derived => unreachable!("a derived part is not stored"),
        };
        match result {
            WriteResult::Ok => written.push((id, bytes)),
            // An engine with a value rule refuses synthetic bytes. Start its state with its own pass instead.
            WriteResult::Refused => return fill_from_a_pass(engine, state, limits, gate),
            WriteResult::Failed => {
                return Err(Failure::new(
                    gate,
                    format!("the engine failed to fill part {} and poisoned the state", part.name),
                ));
            }
        }
        if part.kind == PartKind::Cells {
            cells = true;
        }
    }
    if cells && engine.parts().unwrap().set_n(state, 1) != WriteResult::Ok {
        return Err(Failure::new(gate, "the engine refused to set n after the parts"));
    }
    Ok(written)
}

/// Fill a state through the engine's own pass.
///
/// The engine refused the synthetic bytes. Zero every fixed part, then run a one-token pass. The pass writes the row
/// parts from position 0, and it gives every fixed part a value that the engine accepts, such as a token id in the
/// PLE history. The one-token input makes each cells part hold one row, as the synthetic path writes.
fn fill_from_a_pass<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    gate: &'static str,
) -> Result<Vec<(PartId, Vec<u8>)>, Failure> {
    if limits.inputs != Inputs::Tokens {
        return Err(Failure::new(
            gate,
            "the engine refused the synthetic bytes and it takes no tokens, so the gate cannot fill it",
        ));
    }
    for (id, part) in limits.parts.iter().enumerate() {
        if part.kind != PartKind::Fixed {
            continue;
        }
        let zeros = vec![0u8; shape_bytes(part) as usize];
        match engine.parts().unwrap().part_write(state, id, &zeros) {
            WriteResult::Ok => {}
            // A refusal leaves the part as it was. The pass writes every part, so the gate needs no zero fill.
            WriteResult::Refused => {}
            WriteResult::Failed => {
                return Err(Failure::new(
                    gate,
                    format!("the engine failed to zero part {} and poisoned the state", part.name),
                ));
            }
        }
    }
    // Keep the value before the pass. A read-back after the pass alone cannot tell a fill from a constant read.
    let before = read_fixed(engine, state, limits, gate)?;
    let segment = token_segment(state, 0, &[1]);
    run_pass(engine, &segment, gate)?;
    let after = read_fixed(engine, state, limits, gate)?;
    if before.is_empty() {
        // The engine has no fixed part to change. Compare its two read paths instead: a constant `part_read` cannot
        // agree with the rows that `cells_read` returns.
        compare_part_and_rows(engine, state, limits, gate)?;
    } else if !any_part_differs(&before, &after) {
        return Err(Failure::new(
            gate,
            "the pass changed no fixed part, so the gate cannot tell a fill from a constant read",
        ));
    }
    read_filled(engine, state, limits, gate)
}

/// Check that `part_read` and `cells_read` agree on one row of every stored row part.
///
/// The gate needs this check when the engine has no fixed part. A read that does not agree with the engine's own
/// range read cannot show what the pass wrote.
fn compare_part_and_rows<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    gate: &'static str,
) -> Result<(), Failure> {
    let mut compared = 0;
    for (id, part) in limits.parts.iter().enumerate() {
        if !part.is_row() {
            continue;
        }
        let row = shape_bytes(part) as usize;
        if row == 0 {
            return Err(Failure::new(gate, format!("part {} has a zero row", part.name)));
        }
        let mut whole = vec![0u8; row];
        if let Err(error) = engine.parts().unwrap().part_read(state, id, &mut whole) {
            return Err(Failure::new(
                gate,
                format!("the engine refused to read part {}: {error:?}", part.name),
            ));
        }
        let mut range = vec![0u8; row];
        if let Err(error) = engine.parts().unwrap().cells_read(state, id, 0, 1, &mut range) {
            return Err(Failure::new(
                gate,
                format!(
                    "the engine refused a range read of part {}: {error:?}; the gate cannot tell a fill from a \
                     constant read",
                    part.name
                ),
            ));
        }
        if whole != range {
            return Err(Failure::new(
                gate,
                format!("part {} gives other bytes by part_read than by cells_read", part.name),
            ));
        }
        compared += 1;
    }
    if compared == 0 {
        return Err(Failure::new(
            gate,
            "the engine has no fixed part and no row part, so the gate cannot tell a fill from a constant read",
        ));
    }
    Ok(())
}

/// True when at least one part's bytes differ between two reads.
fn any_part_differs(left: &[(PartId, Vec<u8>)], right: &[(PartId, Vec<u8>)]) -> bool {
    left.len() != right.len()
        || left
            .iter()
            .zip(right)
            .any(|((left_id, left), (right_id, right))| left_id != right_id || left != right)
}

fn read_filled<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    gate: &'static str,
) -> Result<Vec<(PartId, Vec<u8>)>, Failure> {
    let mut read = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        if !part.is_stored() {
            continue;
        }
        // The fill leaves one row in each row part. Only a moved ring holds more, and the gate moves it forward,
        // so the slot's high is the ring's high.
        let length = if part.kind == PartKind::Ring {
            let State::Slot(slot) = state;
            let start = engine
                .coverage()
                .map(|coverage| coverage.coverage(state, id, &[]).start)
                .unwrap_or(0);
            let high = engine.slots().map_or(1, |slots| slots.slot_high(slot));
            ((high.saturating_sub(start)) * shape_bytes(part)) as usize
        } else {
            shape_bytes(part) as usize
        };
        let mut out = vec![0u8; length];
        if let Err(error) = engine.parts().unwrap().part_read(state, id, &mut out) {
            return Err(Failure::new(
                gate,
                format!("the engine refused to read part {}: {error:?}", part.name),
            ));
        }
        read.push((id, out));
    }
    Ok(read)
}

fn compare_filled(
    gate: &'static str,
    limits: &Limits,
    left: &[(PartId, Vec<u8>)],
    right: &[(PartId, Vec<u8>)],
) -> Result<(), Failure> {
    if left.len() != right.len() {
        return Err(Failure::new(
            gate,
            format!("the part counts differ: {} and {}", left.len(), right.len()),
        ));
    }
    for ((left_id, left_bytes), (right_id, right_bytes)) in left.iter().zip(right) {
        if left_id != right_id {
            return Err(Failure::new(
                gate,
                format!("the part ids differ: {left_id} and {right_id}"),
            ));
        }
        let part = &limits.parts[*left_id];
        // Spec 8.3: a NaN or an infinity fails, even when the two bit patterns are equal.
        for (side, bytes) in [("left", left_bytes), ("right", right_bytes)] {
            if let Some(index) = non_finite(part.dtype, bytes) {
                return Err(Failure::new(
                    gate,
                    format!("part {} {side} index {index} is not finite", part.name),
                ));
            }
        }
        if let Err(failure) = compare_bytes(gate, left_bytes, right_bytes) {
            return Err(Failure {
                message: format!("part {}: {}", limits.parts[*left_id].name, failure.message),
                ..failure
            });
        }
    }
    Ok(())
}

/// Restore parked parts into a state through `Store::restore`, the path that production uses (spec 8.3).
///
/// The function puts the parts in a store in a temporary directory, then restores that entry. Thus the gate and the
/// store cannot drift apart. `spans` gives the positions of each row part. Without it, a row part starts at 0.
fn restore_through_store(
    engine: &mut dyn Engine,
    state: State,
    limits: &Limits,
    parts: &[(PartId, Vec<u8>)],
    spans: Option<&[(PartId, CoverageSpan)]>,
    gate: &'static str,
) -> Result<(), Failure> {
    let mut tensors = Vec::new();
    for (id, bytes) in parts {
        let part = &limits.parts[*id];
        let row = shape_bytes(part);
        let (shape, coverage) = if part.is_row() {
            if row == 0 {
                return Err(Failure::new(gate, format!("part {} has a zero row", part.name)));
            }
            if !(bytes.len() as u64).is_multiple_of(row) {
                return Err(Failure::new(
                    gate,
                    format!("part {} ends with a partial row", part.name),
                ));
            }
            let count = bytes.len() as u64 / row;
            let start = spans
                .and_then(|spans| {
                    spans
                        .iter()
                        .find(|(span_part, _)| span_part == id)
                        .map(|(_, span)| span.start)
                })
                .unwrap_or(0);
            let mut shape = vec![count];
            shape.extend(part.shape.iter().copied());
            (shape, Some((start, start + count)))
        } else {
            (part.shape.clone(), None)
        };
        tensors.push(Tensor {
            name: part.name.clone(),
            kind: part.kind,
            dtype: part.dtype,
            shape,
            data: bytes.clone(),
            coverage,
            block_format: part.block_format.clone(),
        });
    }
    let snapshot = Snapshot { tensors, parent: None };
    let store_failure = |what: &str, error: StoreError| Failure::new(gate, format!("{what}: {error}"));
    let dir = tempfile::tempdir().map_err(|error| Failure::new(gate, format!("no temporary directory: {error}")))?;
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, u64::MAX, u64::MAX, 0)
        .map_err(|error| store_failure("the store did not open", error))?;
    let identity = [0u8; 32];
    let id = store
        .put(Put {
            snapshot: &snapshot,
            recipe: recipe_root(&identity, b"", &[], b""),
            previous_recipe: None,
            chains_from: None,
            engine: EngineFacts::of("conformance", limits),
            identity,
            owner: 0,
            tier: Tier::Host,
            reason: Reason::Anchor,
            sealed_recipe: None,
            provenance: None,
            now_ms: 0,
        })
        .map_err(|error| store_failure("the store refused the parked parts", error))?;
    match store
        .restore(&id, engine, state, &identity, 0)
        .map_err(|error| store_failure("the store refused the restore", error))?
    {
        Restored::Done => Ok(()),
        Restored::Refused { part } => Err(Failure::new(gate, format!("the engine refused to restore part {part}"))),
        Restored::Failed { part } => Err(Failure::new(
            gate,
            format!("the engine failed to restore part {part} and poisoned the state"),
        )),
    }
}

fn coverage_spans<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    parts: &[(PartId, Vec<u8>)],
) -> Option<Vec<(PartId, CoverageSpan)>> {
    let coverage = engine.coverage()?;
    let mut spans = Vec::new();
    for (id, bytes) in parts {
        if limits.parts[*id].is_row() {
            spans.push((*id, coverage.coverage(state, *id, bytes)));
        }
    }
    Some(spans)
}

fn compare_spans(
    gate: &'static str,
    left: &[(PartId, CoverageSpan)],
    right: &[(PartId, CoverageSpan)],
) -> Result<(), Failure> {
    if left.len() != right.len() {
        return Err(Failure::new(
            gate,
            format!("the coverage counts differ: {} and {}", left.len(), right.len()),
        ));
    }
    for ((id, left), (right_id, right)) in left.iter().zip(right) {
        if id != right_id || left.part != *id || right.part != *right_id {
            return Err(Failure::new(
                gate,
                format!(
                    "the coverage is for different parts: asked for {id} and {right_id}, got {} and {}",
                    left.part, right.part
                ),
            ));
        }
        if left.start != right.start || left.end != right.end {
            return Err(Failure::new(
                gate,
                format!(
                    "part {id} coverage differs: {}..{} and {}..{}",
                    left.start, left.end, right.start, right.end
                ),
            ));
        }
    }
    Ok(())
}

/// Park every stored part of one state, restore it into a second state, and compare every bit.
///
/// The gate also checks the counts and the coverage spans. It is quantum condition 3.
pub fn park_and_restore<E: Engine>(engine: &mut E) -> Outcome {
    let limits = engine.limits();
    if engine.parts().is_none() {
        return Outcome::NotApplicable("the engine has no parts capability".to_owned());
    }
    let Some(destination) = second_slot(&limits) else {
        return Outcome::NotApplicable(
            "the engine declares one slot; park and restore needs a second state".to_owned(),
        );
    };
    match park_and_restore_inner(engine, &limits, destination) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

fn park_and_restore_inner<E: Engine>(engine: &mut E, limits: &Limits, destination: u32) -> Result<(), Failure> {
    let gate = "park_and_restore";
    let source = State::Slot(0);
    let target = State::Slot(destination);
    let written = fill(engine, source, limits, 0x21, gate)?;
    if written.is_empty() {
        return Err(Failure::new(
            gate,
            "the engine declares no stored part; park has nothing to move",
        ));
    }
    let mut parked = read_filled(engine, source, limits, gate)?;
    compare_filled(gate, limits, &written, &parked)?;
    move_ring_window(engine, source, limits, &mut parked, gate)?;
    let source_spans = coverage_spans(engine, source, limits, &parked);
    restore_through_store(engine, target, limits, &parked, source_spans.as_deref(), gate)?;
    let restored = read_filled(engine, target, limits, gate)?;
    compare_filled(gate, limits, &parked, &restored)?;
    if let (Some(source_spans), Some(restored_spans)) =
        (source_spans, coverage_spans(engine, target, limits, &restored))
    {
        compare_spans(gate, &source_spans, &restored_spans)?;
    }
    Ok(())
}

/// The bytes of one slot.
fn slot_capacity(limits: &Limits, slot: u32) -> Option<u64> {
    let mut rest = slot;
    for class in &limits.slots {
        if rest < class.count {
            return Some(class.bytes);
        }
        rest -= class.count;
    }
    None
}

/// Move a ring part's window, so that a restore that writes at row 0 loses its base.
fn move_ring_window<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    parked: &mut [(PartId, Vec<u8>)],
    gate: &'static str,
) -> Result<(), Failure> {
    let Some((id, part)) = limits
        .parts
        .iter()
        .enumerate()
        .find(|(_, part)| part.kind == PartKind::Ring)
    else {
        return Ok(());
    };
    let row_bytes = shape_bytes(part);
    if row_bytes == 0 {
        return Err(Failure::new(gate, format!("part {} has a zero row", part.name)));
    }
    let Some(entry) = parked.iter_mut().find(|(part_id, _)| *part_id == id) else {
        return Ok(());
    };
    if !(entry.1.len() as u64).is_multiple_of(row_bytes) {
        return Err(Failure::new(gate, format!("part {} is a partial row", part.name)));
    }
    let Some(coverage) = engine.coverage() else {
        return Ok(());
    };
    let State::Slot(slot) = state;
    let Some(slot_bytes) = slot_capacity(limits, slot) else {
        return Ok(());
    };
    let span = coverage.coverage(state, id, &entry.1);
    if span.end < span.start {
        return Err(Failure::new(
            gate,
            format!("part {} has coverage {}..{}", part.name, span.start, span.end),
        ));
    }
    // A ring keeps only its window, so a write larger than the slot forces the window to move.
    let count = slot_bytes / row_bytes + 1;
    let bytes = pattern((count * row_bytes) as usize, 0xC3, part.dtype);
    match engine.parts().unwrap().cells_write(state, id, span.end, count, &bytes) {
        WriteResult::Ok => {}
        WriteResult::Refused => return Ok(()),
        WriteResult::Failed => {
            return Err(Failure::new(
                gate,
                format!("the engine failed to move part {} and poisoned the state", part.name),
            ));
        }
    }
    let Some(coverage) = engine.coverage() else {
        return Ok(());
    };
    let base = coverage.coverage(state, id, &[]).start;
    if base == 0 {
        return Ok(());
    }
    let high = span.end + count;
    let length = ((high - base) * row_bytes) as usize;
    let mut moved = vec![0u8; length];
    if let Err(error) = engine.parts().unwrap().part_read(state, id, &mut moved) {
        return Err(Failure::new(
            gate,
            format!("the engine refused to read the moved part {}: {error:?}", part.name),
        ));
    }
    entry.1 = moved;
    Ok(())
}

/// Check that a read and a park change nothing, and that a seal is a separate call.
///
/// The scheduler chooses the boundaries. The engine-side rule is: only `seal` and `pass` may change bits, and a
/// read or a park may not. The gate checks a read, a second read, a seal, and two reads after the seal.
pub fn seal_only_at_boundaries<E: Engine + ?Sized>(engine: &mut E) -> Outcome {
    if engine.parts().is_none() {
        return Outcome::NotApplicable("the engine has no parts capability".to_owned());
    }
    if engine.seal().is_none() {
        return Outcome::NotApplicable("the engine has no seal capability".to_owned());
    }
    match seal_only_at_boundaries_inner(engine) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

fn seal_only_at_boundaries_inner<E: Engine + ?Sized>(engine: &mut E) -> Result<(), Failure> {
    let gate = "seal_only_at_boundaries";
    let limits = engine.limits();
    let state = State::Slot(0);
    let written = fill(engine, state, &limits, 0x41, gate)?;
    if written.is_empty() {
        return Err(Failure::new(
            gate,
            "the engine declares no stored part; a read has nothing to keep",
        ));
    }
    let first = read_filled(engine, state, &limits, gate)?;
    compare_filled(gate, &limits, &written, &first)?;
    let first_spans = coverage_spans(engine, state, &limits, &first);
    let second = read_filled(engine, state, &limits, gate)?;
    compare_filled(gate, &limits, &first, &second)?;
    if let (Some(first_spans), Some(second_spans)) = (first_spans, coverage_spans(engine, state, &limits, &second)) {
        compare_spans(gate, &first_spans, &second_spans)?;
    }
    if let Err(error) = engine.seal().unwrap().seal(state) {
        return Err(Failure::new(
            gate,
            format!("the engine refused to seal a state that it can read: {error:?}"),
        ));
    }
    let sealed = read_filled(engine, state, &limits, gate)?;
    let again = read_filled(engine, state, &limits, gate)?;
    compare_filled(gate, &limits, &sealed, &again)?;
    Ok(())
}

/// Check that two fresh engines have the same identity and the same limits (quantum condition 1).
pub fn quantum_identity<E: Engine>(make: &mut dyn FnMut() -> E) -> Outcome {
    let first = make();
    let second = make();
    if first.identity() != second.identity() {
        return Outcome::Fail(Failure::new(
            "quantum_identity",
            "the identity changed between two instances",
        ));
    }
    if first.limits() != second.limits() {
        return Outcome::Fail(Failure::new(
            "quantum_identity",
            "the limits changed between two instances",
        ));
    }
    Outcome::Pass
}

/// Check that a pass on one state changes no other state (quantum condition 5).
pub fn quantum_isolation<E: Engine>(make: &mut dyn FnMut() -> E) -> Outcome {
    let mut engine = make();
    let limits = engine.limits();
    if limits.inputs != Inputs::Tokens {
        return Outcome::NotApplicable("the gate makes token segments; this engine takes rows".to_owned());
    }
    if engine.parts().is_none() {
        return Outcome::NotApplicable("the engine has no parts capability".to_owned());
    }
    let Some(other) = second_slot(&limits) else {
        return Outcome::NotApplicable("the engine declares one slot; the gate needs a second state".to_owned());
    };
    match quantum_isolation_inner(&mut engine, &limits, other) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

fn quantum_isolation_inner<E: Engine + ?Sized>(engine: &mut E, limits: &Limits, other: u32) -> Result<(), Failure> {
    let gate = "quantum_isolation";
    let target = State::Slot(0);
    let other = State::Slot(other);
    fill(engine, target, limits, 0x61, gate)?;
    fill(engine, other, limits, 0x81, gate)?;
    let before = read_filled(engine, other, limits, gate)?;
    let before_spans = coverage_spans(engine, other, limits, &before);
    let segment = token_segment(target, filled_rows(limits), &[1, 2, 3]);
    run_pass(engine, &segment, gate)?;
    let after = read_filled(engine, other, limits, gate)?;
    compare_filled(gate, limits, &before, &after)?;
    if let (Some(before_spans), Some(after_spans)) = (before_spans, coverage_spans(engine, other, limits, &after)) {
        compare_spans(gate, &before_spans, &after_spans)?;
    }
    Ok(())
}

/// Check that the same boundaries give the same bits, and that history does not change them.
///
/// The gate runs the same segment on two fresh engines and compares the pass outputs and the fixed state. It
/// then seals both at the same boundary and compares again. Finally, it puts an unrelated quantum between two
/// parts of one block and compares with a run without it (quantum conditions 2 and 7). It skips the last check
/// when the engine declares a bit dependence, as the ABI says.
pub fn quantum_same_boundaries<E: Engine>(make: &mut dyn FnMut() -> E) -> Outcome {
    let mut first = make();
    let limits = first.limits();
    if limits.inputs != Inputs::Tokens {
        return Outcome::NotApplicable("the gate makes token segments; this engine takes rows".to_owned());
    }
    if first.parts().is_none() {
        return Outcome::NotApplicable("the engine has no parts capability".to_owned());
    }
    if first.seal().is_none() {
        return Outcome::NotApplicable("the engine has no seal capability".to_owned());
    }
    match same_boundaries_inner(first, make, &limits) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

fn same_boundaries_inner<E: Engine>(mut first: E, make: &mut dyn FnMut() -> E, limits: &Limits) -> Result<(), Failure> {
    let gate = "quantum_same_boundaries";
    let mut second = make();
    let state = State::Slot(0);
    let segment = with_last_logit(token_segment(state, 0, &[11, 12, 13]), limits);
    let first_out = run_pass(&mut first, &segment, gate)?;
    let second_out = run_pass(&mut second, &segment, gate)?;
    if limits.max_logit_rows > 0 && first_out.outputs.is_empty() {
        return Err(Failure::new(
            gate,
            "the gate asked for a logit row and the engine returned no output",
        ));
    }
    compare_outputs(gate, &first_out.outputs, &second_out.outputs)?;
    compare_fixed(gate, &mut first, &mut second, state, limits)?;
    seal_state(&mut first, state, gate, "the first engine")?;
    seal_state(&mut second, state, gate, "the second engine")?;
    compare_fixed(gate, &mut first, &mut second, state, limits)?;
    // The next check makes two more engines. An engine that holds a large pool (mk: 11 GiB) needs these two freed.
    drop(first);
    drop(second);
    if limits.bit_dependence.is_empty()
        && let Some(other) = second_slot(limits)
    {
        let mut with_gap = make();
        let mut without_gap = make();
        run_pass(
            &mut with_gap,
            &with_last_logit(token_segment(state, 0, &[1, 2]), limits),
            gate,
        )?;
        run_pass(
            &mut with_gap,
            &with_last_logit(token_segment(State::Slot(other), 0, &[9]), limits),
            gate,
        )?;
        let with_gap_out = run_pass(
            &mut with_gap,
            &with_last_logit(token_segment(state, 2, &[3]), limits),
            gate,
        )?;
        run_pass(
            &mut without_gap,
            &with_last_logit(token_segment(state, 0, &[1, 2]), limits),
            gate,
        )?;
        let without_gap_out = run_pass(
            &mut without_gap,
            &with_last_logit(token_segment(state, 2, &[3]), limits),
            gate,
        )?;
        compare_outputs(gate, &with_gap_out.outputs, &without_gap_out.outputs)?;
        compare_fixed(gate, &mut with_gap, &mut without_gap, state, limits)?;
    }
    Ok(())
}

fn seal_state<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    gate: &'static str,
    which: &str,
) -> Result<(), Failure> {
    match engine.seal() {
        Some(seal) => seal
            .seal(state)
            .map_err(|error| Failure::new(gate, format!("{which} refused the seal: {error:?}"))),
        None => Err(Failure::new(gate, "the engine has no seal capability")),
    }
}

/// Check that a parked state's coverage fits its slot, and that a slot that is too small is refused.
///
/// This is quantum condition 4.
pub fn quantum_resume_fit<E: Engine + ?Sized>(engine: &mut E) -> Outcome {
    if engine.parts().is_none() {
        return Outcome::NotApplicable("the engine has no parts capability".to_owned());
    }
    if engine.coverage().is_none() {
        return Outcome::NotApplicable("the engine has no coverage capability".to_owned());
    }
    match quantum_resume_fit_inner(engine) {
        Ok(Some(())) => Outcome::Pass,
        Ok(None) => Outcome::NotApplicable("the engine has no row part; coverage has nothing to hold".to_owned()),
        Err(failure) => Outcome::Fail(failure),
    }
}

fn quantum_resume_fit_inner<E: Engine + ?Sized>(engine: &mut E) -> Result<Option<()>, Failure> {
    let gate = "quantum_resume_fit";
    let limits = engine.limits();
    let Some(slot) = limits.slots.iter().any(|class| class.count > 0).then_some(0) else {
        return Err(Failure::new(gate, "the engine declares no slot"));
    };
    let state = State::Slot(slot);
    fill(engine, state, &limits, 0xA1, gate)?;
    let parked = read_filled(engine, state, &limits, gate)?;
    let mut checked = 0;
    for (id, bytes) in &parked {
        let part = &limits.parts[*id];
        if !part.is_row() {
            continue;
        }
        let row_bytes = shape_bytes(part);
        if row_bytes == 0 {
            return Err(Failure::new(gate, format!("part {} has a zero row", part.name)));
        }
        if !(bytes.len() as u64).is_multiple_of(row_bytes) {
            return Err(Failure::new(gate, format!("part {} is a partial row", part.name)));
        }
        let rows = bytes.len() as u64 / row_bytes;
        let span = engine.coverage().unwrap().coverage(state, *id, bytes);
        if span.part != *id {
            return Err(Failure::new(
                gate,
                format!("coverage names part {} for part {}", span.part, part.name),
            ));
        }
        if span.end - span.start != rows {
            return Err(Failure::new(
                gate,
                format!(
                    "part {} has {rows} rows but coverage holds {}",
                    part.name,
                    span.end - span.start
                ),
            ));
        }
        if let Err(error) = engine.coverage().unwrap().fits(&span, slot) {
            return Err(Failure::new(
                gate,
                format!("part {} does not fit its own slot: {error}", part.name),
            ));
        }
        checked += 1;
    }
    if checked == 0 {
        return Ok(None);
    }
    let Some((small_slot, small_bytes)) = smallest_slot(&limits) else {
        return Err(Failure::new(gate, "the engine declares no slot"));
    };
    let (id, part) = limits
        .parts
        .iter()
        .enumerate()
        .filter(|(_, part)| part.is_row())
        .min_by_key(|(_, part)| shape_bytes(part))
        .ok_or_else(|| Failure::new(gate, "the engine has no row part"))?;
    let row_bytes = shape_bytes(part);
    if row_bytes == 0 {
        return Err(Failure::new(gate, format!("part {} has a zero row", part.name)));
    }
    let too_big = CoverageSpan {
        part: id,
        start: 0,
        end: small_bytes / row_bytes + 1,
    };
    if engine.coverage().unwrap().fits(&too_big, small_slot).is_ok() {
        return Err(Failure::new(
            gate,
            format!(
                "part {} accepts {} bytes in a {} byte slot",
                part.name,
                (too_big.end - too_big.start) * row_bytes,
                small_bytes
            ),
        ));
    }
    Ok(Some(()))
}

/// Check that a refused admission changes nothing (spec 4.3).
///
/// The caller gives one profile that fits and one profile that must be refused. The gate checks the shortfall
/// and the ledger state.
pub fn admission_refusal_changes_nothing<L>(ledger: &mut L, fitting: &Profile, refused: &Profile) -> Outcome
where
    L: Admit + Clone + PartialEq,
{
    let gate = "admission_refusal_changes_nothing";
    if let Err(error) = ledger.admit(fitting) {
        return Outcome::Fail(Failure::new(gate, format!("the fitting profile did not fit: {error}")));
    }
    let before = ledger.clone();
    match ledger.admit(refused) {
        Ok(_) => Outcome::Fail(Failure::new(gate, "the profile fit, so the gate saw no refusal")),
        Err(LedgerError::Invalid(message)) => Outcome::Fail(Failure::new(
            gate,
            format!("the profile is invalid, not refused: {message}"),
        )),
        Err(LedgerError::ContextRefused(_) | LedgerError::StagingRefused(_)) => Outcome::Fail(Failure::new(
            gate,
            "the ledger refused a context or a copy while it admitted a profile: a profile admission must not do that",
        )),
        Err(LedgerError::Refused(refusal)) => {
            if refusal.shortfall.iter().all(|(_, bytes)| *bytes == 0) {
                return Outcome::Fail(Failure::new(gate, "the refusal reports no shortfall"));
            }
            if *ledger != before {
                return Outcome::Fail(Failure::new(gate, "a refusal changed the ledger"));
            }
            Outcome::Pass
        }
    }
}

/// The name of the segments gate.
const SEGMENTS_GATE: &str = "segments_exact_bits";

/// Check that a segment gives the same bits in passes with one, two and the most segments (spec 5.7, 8.3).
///
/// The gate also splits one segment into two passes and compares it with one pass. The row count of a pass must not
/// change bits either. It compares the pass outputs and the fixed parts by bit pattern, and a value that is not
/// finite fails.
///
/// First the gate runs each of the `max_segments` segments alone, on its own state of a fresh engine. Then, for
/// each count, it runs the first segments together in one pass on another fresh engine. Each segment must give the
/// bits that it gave alone. The segments differ in tokens and in length, so a sum that mixes them shows.
/// The gate needs one state for each segment, and a pass that takes `2 * max_segments` tokens.
pub fn segments_exact_bits<E: Engine>(make: &mut dyn FnMut() -> E) -> Outcome {
    let mut reference = make();
    let limits = reference.limits();
    if !limits.segments_exact {
        return Outcome::NotApplicable("the engine does not declare segments_exact".to_owned());
    }
    if limits.inputs != Inputs::Tokens {
        return Outcome::NotApplicable("the gate makes token segments; this engine takes rows".to_owned());
    }
    if reference.parts().is_none() {
        return Outcome::NotApplicable("the engine has no parts capability".to_owned());
    }
    let most = limits.max_segments;
    if most < 2 {
        return Outcome::NotApplicable("max_segments is below 2; the gate needs a pass of two segments".to_owned());
    }
    let slots: u32 = limits.slots.iter().map(|class| class.count).sum();
    if slots < most {
        return Outcome::NotApplicable(format!(
            "the engine declares {slots} slots; the gate needs {most}, one for each segment of the largest pass"
        ));
    }
    if limits.max_pass_inputs < 2 * most {
        return Outcome::NotApplicable(format!(
            "max_pass_inputs is {}; the gate needs {} for the largest pass",
            limits.max_pass_inputs,
            2 * most
        ));
    }
    match segments_exact_bits_inner(&mut reference, make, &limits, most) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

/// The segment for state `index`: its tokens and its length differ from the other segments.
fn counted_segment(index: u32, limits: &Limits) -> Segment {
    let length = 1 + index % 2;
    let tokens: Vec<u32> = (0..length).map(|offset| 1000 + 17 * index + 3 * offset).collect();
    with_last_logit(token_segment(State::Slot(index), 0, &tokens), limits)
}

fn segments_exact_bits_inner<E: Engine>(
    reference: &mut E,
    make: &mut dyn FnMut() -> E,
    limits: &Limits,
    most: u32,
) -> Result<(), Failure> {
    let gate = SEGMENTS_GATE;
    // The reference: each segment alone, in a pass of one segment.
    let mut alone = Vec::new();
    for index in 0..most {
        alone.push(run_pass(reference, &counted_segment(index, limits), gate)?.outputs);
    }
    let mut counts = vec![1, 2, most];
    counts.dedup();
    for count in counts {
        let mut engine = make();
        let segments: Vec<Segment> = (0..count).map(|index| counted_segment(index, limits)).collect();
        let out = run_pass_of(&mut engine, &segments, gate)?;
        let expected: usize = alone[..count as usize].iter().map(Vec::len).sum();
        if out.outputs.len() != expected {
            return Err(Failure::new(
                gate,
                format!(
                    "a pass of {count} segments gave {} outputs; the segments alone gave {expected}",
                    out.outputs.len()
                ),
            ));
        }
        let mut offset = 0;
        for index in 0..count {
            let wanted = &alone[index as usize];
            let found = &out.outputs[offset..offset + wanted.len()];
            offset += wanted.len();
            let context = |failure: Failure| Failure {
                message: format!("in a pass of {count} segments, segment {index}: {}", failure.message),
                ..failure
            };
            compare_outputs(gate, wanted, found).map_err(context)?;
            compare_fixed(gate, reference, &mut engine, State::Slot(index), limits).map_err(context)?;
        }
    }
    split_pass(make, limits)
}

/// Check that one pass of four tokens gives the bits of a pass of one token and then a pass of three.
fn split_pass<E: Engine>(make: &mut dyn FnMut() -> E, limits: &Limits) -> Result<(), Failure> {
    let gate = SEGMENTS_GATE;
    let state = State::Slot(0);
    let tokens = [2001, 2002, 2003, 2004];
    let mut whole = make();
    let wanted = run_pass(
        &mut whole,
        &with_last_logit(token_segment(state, 0, &tokens), limits),
        gate,
    )?;
    let mut split = make();
    run_pass(
        &mut split,
        &with_last_logit(token_segment(state, 0, &tokens[..1]), limits),
        gate,
    )?;
    let found = run_pass(
        &mut split,
        &with_last_logit(token_segment(state, 1, &tokens[1..]), limits),
        gate,
    )?;
    let context = |failure: Failure| Failure {
        message: format!(
            "a pass split in two passes gave other bits than one pass: {}",
            failure.message
        ),
        ..failure
    };
    compare_outputs(gate, &wanted.outputs, &found.outputs).map_err(context)?;
    compare_fixed(gate, &mut whole, &mut split, state, limits).map_err(context)
}

fn token_segment(state: State, start: u64, tokens: &[u32]) -> Segment {
    Segment {
        state,
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (start..start + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    }
}

/// Ask for the logit row at the segment's last position, when the engine gives logit rows.
fn with_last_logit(mut segment: Segment, limits: &Limits) -> Segment {
    if limits.max_logit_rows > 0
        && let Some(last) = segment.positions.last()
    {
        segment.outputs.logit_rows = vec![*last];
    }
    segment
}

fn run_pass<E: Engine + ?Sized>(engine: &mut E, segment: &Segment, gate: &'static str) -> Result<PassOut, Failure> {
    run_pass_of(engine, std::slice::from_ref(segment), gate)
}

/// Run one pass of several segments.
fn run_pass_of<E: Engine + ?Sized>(
    engine: &mut E,
    segments: &[Segment],
    gate: &'static str,
) -> Result<PassOut, Failure> {
    match engine.pass(segments, &[]) {
        Ok(output) if output.status == PassStatus::Ok => Ok(output),
        Ok(output) => Err(Failure::new(gate, format!("the pass returned {:?}", output.status))),
        Err(error) => Err(Failure::new(
            gate,
            format!("the pass failed: {:?}: {}", error.kind, error.message),
        )),
    }
}

pub(crate) fn compare_outputs(gate: &'static str, left: &[Output], right: &[Output]) -> Result<(), Failure> {
    if left.len() != right.len() {
        return Err(Failure::new(
            gate,
            format!("the output counts differ: {} and {}", left.len(), right.len()),
        ));
    }
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if left.name != right.name {
            return Err(Failure::new(
                gate,
                format!("output {index} is named {} and {}", left.name, right.name),
            ));
        }
        // Spec 8.3: a NaN or an infinity fails, even when the two bit patterns are equal.
        for (side, output) in [("left", left), ("right", right)] {
            if let Some(element) = non_finite(output.dtype, &output.bytes) {
                return Err(Failure::new(
                    gate,
                    format!("output {} {side} index {element} is not finite", output.name),
                ));
            }
        }
        if let Err(failure) = compare_bytes(gate, &left.bytes, &right.bytes) {
            return Err(Failure {
                message: format!("output {}: {}", left.name, failure.message),
                ..failure
            });
        }
    }
    Ok(())
}

fn read_fixed<E: Engine + ?Sized>(
    engine: &mut E,
    state: State,
    limits: &Limits,
    gate: &'static str,
) -> Result<Vec<(PartId, Vec<u8>)>, Failure> {
    let mut parts = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        if part.kind != PartKind::Fixed {
            continue;
        }
        let mut out = vec![0u8; shape_bytes(part) as usize];
        if let Err(error) = engine.parts().unwrap().part_read(state, id, &mut out) {
            return Err(Failure::new(
                gate,
                format!("the engine refused to read part {}: {error:?}", part.name),
            ));
        }
        parts.push((id, out));
    }
    Ok(parts)
}

fn compare_fixed<E: Engine + ?Sized>(
    gate: &'static str,
    first: &mut E,
    second: &mut E,
    state: State,
    limits: &Limits,
) -> Result<(), Failure> {
    let left = read_fixed(first, state, limits, gate)?;
    let right = read_fixed(second, state, limits, gate)?;
    compare_filled(gate, limits, &left, &right)
}

#[cfg(test)]
mod tests {
    use nagaya_abi::Dtype;

    use super::non_finite;

    #[test]
    fn every_float_width_finds_a_nan_and_an_infinity() {
        assert_eq!(non_finite(Dtype::F32, &f32::NAN.to_le_bytes()), Some(0));
        assert_eq!(non_finite(Dtype::F32, &f32::INFINITY.to_le_bytes()), Some(0));
        assert_eq!(non_finite(Dtype::F32, &1.0f32.to_le_bytes()), None);
        assert_eq!(non_finite(Dtype::F16, &0x7e00u16.to_le_bytes()), Some(0));
        assert_eq!(non_finite(Dtype::F16, &0x7c00u16.to_le_bytes()), Some(0));
        assert_eq!(non_finite(Dtype::F16, &0x3c00u16.to_le_bytes()), None);
        assert_eq!(non_finite(Dtype::BF16, &0x7fc0u16.to_le_bytes()), Some(0));
        assert_eq!(non_finite(Dtype::BF16, &0x7f80u16.to_le_bytes()), Some(0));
        assert_eq!(non_finite(Dtype::BF16, &0x3f80u16.to_le_bytes()), None);
        // An integer part has no NaN.
        assert_eq!(non_finite(Dtype::U32, &0xffff_ffffu32.to_le_bytes()), None);
        assert_eq!(non_finite(Dtype::F8E5M2, &[0x3c]), None);
        assert_eq!(non_finite(Dtype::F8E5M2, &[0x7c]), Some(0));
        assert_eq!(non_finite(Dtype::F8E5M2, &[0xfd]), Some(0));
        assert_eq!(non_finite(Dtype::F8E4M3, &[0x78]), None);
        assert_eq!(non_finite(Dtype::F8E4M3, &[0x7f]), Some(0));
        assert_eq!(non_finite(Dtype::F8E4M3, &[0xff]), Some(0));
        assert_eq!(non_finite(Dtype::I64, &[0xff; 8]), None);
    }
}
