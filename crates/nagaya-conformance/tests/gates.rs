//! The engine gates of spec 8.3, on `FakeAlways` and on broken variants.

use std::sync::atomic::{AtomicU32, Ordering};

use nagaya_abi::fake::{FakeAlways, FakeBroken, FakeRingBase, FakeRowCount, FakeSegmentCount};
use nagaya_abi::{
    Capability, ChainLimits, Coverage, CoverageSpan, Determinism, Dtype, ElasticTerm, Engine, Identity, Inputs,
    Lifetime, Limits, MandatoryTerm, MemoryLimits, PartError, PartId, PartKind, PartLimits, Parts, PassError,
    PassErrorKind, PassOut, PassStatus, Segment, SegmentInputs, SlotLimit, Slots, Stage, State, Tap, Tier, WriteResult,
};
use nagaya_conformance::{
    Admit, Outcome, admission_refusal_changes_nothing, park_and_restore, quantum_identity, quantum_isolation,
    quantum_resume_fit, quantum_same_boundaries, run_all, seal_only_at_boundaries, segments_exact_bits,
};
use nagaya_ledger::{Admission, Budget, EnginePlan, Ledger, LedgerError, Profile};

fn assert_pass(outcome: Outcome, what: &str) {
    assert!(matches!(outcome, Outcome::Pass), "{what}: expected Pass, got {outcome}");
}

fn assert_fail(outcome: Outcome, what: &str) {
    match outcome {
        Outcome::Fail(failure) => assert!(!failure.message.is_empty(), "{what}: the failure has no message"),
        other => panic!("{what}: expected Fail, got {other}"),
    }
}

fn slot_bytes(limits: &Limits, slot: u32) -> Option<u64> {
    let mut rest = slot as usize;
    for class in &limits.slots {
        if rest < class.count as usize {
            return Some(class.bytes);
        }
        rest -= class.count as usize;
    }
    None
}

#[test]
fn fake_always_passes_every_engine_gate() {
    assert_pass(park_and_restore(&mut FakeAlways::new()), "park_and_restore");
    assert_pass(
        seal_only_at_boundaries(&mut FakeAlways::new()),
        "seal_only_at_boundaries",
    );
    let mut make = || FakeAlways::new();
    assert_pass(quantum_identity(&mut make), "quantum_identity");
    assert_pass(quantum_isolation(&mut make), "quantum_isolation");
    assert_pass(quantum_same_boundaries(&mut make), "quantum_same_boundaries");
    assert_pass(quantum_resume_fit(&mut FakeAlways::new()), "quantum_resume_fit");
    assert_pass(segments_exact_bits(&mut make), "segments_exact_bits");
}

#[test]
fn fake_broken_fails_park_and_restore() {
    assert_fail(park_and_restore(&mut FakeBroken::new()), "fake_broken park_and_restore");
}

#[test]
fn a_ring_that_loses_its_base_fails_park_and_restore() {
    match park_and_restore(&mut FakeRingBase::new()) {
        Outcome::Fail(failure) => {
            assert!(failure.message.contains("coverage differs"), "{failure}");
        }
        other => panic!("expected a coverage failure, got {other}"),
    }
}

#[test]
fn a_one_slot_engine_gets_not_applicable() {
    let mut one_slot = || FakeAlways::with_slots(vec![SlotLimit { bytes: 256, count: 1 }]);
    assert!(matches!(park_and_restore(&mut one_slot()), Outcome::NotApplicable(_)));
    assert!(matches!(quantum_isolation(&mut one_slot), Outcome::NotApplicable(_)));
    assert!(run_all(&mut one_slot).passed());
}

// The broken fake flips a bit in `part_read`. It also fails the seal gate, because a read then returns different
// bits than the gate wrote. This test keeps the park gate's failure on its own.

/// An engine whose `part_read` seals the state after the read.
struct SealsOnRead {
    inner: FakeAlways,
}

impl SealsOnRead {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for SealsOnRead {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Parts for SealsOnRead {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let result = self.inner.parts().unwrap().part_read(state, part, out);
        if result.is_ok() {
            let _ = self.inner.seal().unwrap().seal(state);
        }
        result
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

/// An engine that reports the coverage of one state under the wrong part.
struct WrongPartCoverage {
    inner: std::cell::RefCell<FakeAlways>,
    wrong_state: State,
}

impl WrongPartCoverage {
    /// The engine that reports the spans of `wrong_state` under the wrong part.
    fn new(wrong_state: State) -> Self {
        Self {
            inner: std::cell::RefCell::new(FakeAlways::new()),
            wrong_state,
        }
    }
}

impl Engine for WrongPartCoverage {
    fn identity(&self) -> Identity {
        self.inner.borrow().identity()
    }

    fn limits(&self) -> Limits {
        self.inner.borrow().limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.get_mut().pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.borrow().last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.get_mut().parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.get_mut().seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.get_mut().slots()
    }
}

impl Coverage for WrongPartCoverage {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        let mut engine = self.inner.borrow_mut();
        let mut span = engine.coverage().unwrap().coverage(state, part, bytes);
        if state == self.wrong_state {
            span.part = part + 1;
        }
        span
    }

    fn fits(&self, coverage: &CoverageSpan, slot: u32) -> Result<(), String> {
        self.inner.borrow_mut().coverage().unwrap().fits(coverage, slot)
    }
}

#[test]
fn coverage_under_the_wrong_part_fails_park_and_restore() {
    // Slot 0 is the source of the park. Slot 1 is the target of the restore.
    for wrong in [State::Slot(0), State::Slot(1)] {
        match park_and_restore(&mut WrongPartCoverage::new(wrong)) {
            Outcome::Fail(failure) => assert!(failure.message.contains("part"), "{wrong:?} {failure}"),
            other => panic!("{wrong:?}: expected a failure, got {other}"),
        }
    }
}

#[test]
fn a_seal_on_read_fails_the_seal_gate() {
    assert_pass(
        park_and_restore(&mut SealsOnRead::new()),
        "SealsOnRead park_and_restore",
    );
    assert_fail(
        seal_only_at_boundaries(&mut SealsOnRead::new()),
        "SealsOnRead seal_only_at_boundaries",
    );
}

/// An engine that refuses a fixed part whose bytes are all odd. The synthetic fill writes such bytes; a zero fill
/// and a pass's state do not.
struct RefusesOddBytes {
    inner: FakeAlways,
}

impl RefusesOddBytes {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for RefusesOddBytes {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Parts for RefusesOddBytes {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        if part == 0 && bytes.iter().all(|byte| byte & 1 == 1) {
            return WriteResult::Refused;
        }
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

#[test]
fn a_value_rule_falls_back_to_a_pass() {
    // The synthetic bytes are all odd, so the engine refuses them. The gate fills the state through a pass instead.
    assert_pass(
        park_and_restore(&mut RefusesOddBytes::new()),
        "RefusesOddBytes park_and_restore",
    );
}

/// An engine that refuses the synthetic fill and fails the zero write that the pass path uses.
struct FailsZeroWrite {
    inner: FakeAlways,
}

impl FailsZeroWrite {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for FailsZeroWrite {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Parts for FailsZeroWrite {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        // The synthetic fill is all odd bytes, so the gate takes the pass path. Its zero fill then fails.
        if bytes.iter().all(|byte| *byte == 0) {
            return WriteResult::Failed;
        }
        if part == 0 && bytes.iter().all(|byte| byte & 1 == 1) {
            return WriteResult::Refused;
        }
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

#[test]
fn a_failed_zero_write_fails_the_gate() {
    assert_fail(park_and_restore(&mut FailsZeroWrite::new()), "FailsZeroWrite");
}

/// An engine that refuses the synthetic fill and returns the same bytes for every read.
struct RefusesOddBytesAndReadsConstant {
    inner: FakeAlways,
}

impl RefusesOddBytesAndReadsConstant {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for RefusesOddBytesAndReadsConstant {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Parts for RefusesOddBytesAndReadsConstant {
    fn part_read(&mut self, _state: State, _part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        out.fill(0x5a);
        Ok(())
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        if part == 0 && bytes.iter().all(|byte| byte & 1 == 1) {
            return WriteResult::Refused;
        }
        self.inner.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

#[test]
fn a_constant_read_fails_the_gate() {
    assert_fail(
        park_and_restore(&mut RefusesOddBytesAndReadsConstant::new()),
        "RefusesOddBytesAndReadsConstant",
    );
}

/// An engine with one cells part and no fixed part. Its `part_read` returns the same bytes for every read.
///
/// The gate's first fallback check reads the fixed parts before and after the pass. This engine has none, so the
/// gate must compare `part_read` with `cells_read` instead. The two disagree, so the gate must fail it.
struct CellsOnlyConstantRead {
    rows: [Vec<u8>; 2],
}

impl CellsOnlyConstantRead {
    /// The row width of the one cells part.
    const ROW: usize = 8;

    fn new() -> Self {
        Self {
            rows: [Vec::new(), Vec::new()],
        }
    }

    fn rows(&self, state: State) -> Option<&Vec<u8>> {
        let State::Slot(slot) = state;
        self.rows.get(slot as usize)
    }

    fn rows_mut(&mut self, state: State) -> Option<&mut Vec<u8>> {
        let State::Slot(slot) = state;
        self.rows.get_mut(slot as usize)
    }
}

impl Engine for CellsOnlyConstantRead {
    fn identity(&self) -> Identity {
        Identity {
            model: "cells-only".to_owned(),
            checkpoint_digest: "cells-only:0".to_owned(),
            tokenizer_hash: "cells-only:0".to_owned(),
            template: "cells-only".to_owned(),
            numbers: std::collections::BTreeMap::new(),
            plan_table: "cells-only".to_owned(),
            bit_settings: Vec::new(),
            canary_hash: "cells-only:0".to_owned(),
        }
    }

    fn limits(&self) -> Limits {
        Limits {
            capabilities: vec![Capability::Parts],
            parts: vec![PartLimits {
                name: "cells".to_owned(),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![Self::ROW as u64],
                block_format: None,
            }],
            inputs: Inputs::Tokens,
            devices: vec![0],
            stages: vec![Stage {
                device: 0,
                layers: 0..1,
                head: true,
            }],
            max_segments: 1,
            max_pass_inputs: 1,
            max_logit_rows: 0,
            slots: vec![SlotLimit { bytes: 64, count: 2 }],
            max_quantum_ms: 10,
            memory: MemoryLimits::default(),
            chain: ChainLimits {
                max_depth: 0,
                delta_fraction: 0.0,
            },
            segments_exact: true,
            determinism: Determinism::Always,
            bit_dependence: Vec::new(),
            taps: Vec::new(),
            checkpoint_grain: None,
        }
    }

    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        for segment in segments {
            let SegmentInputs::Tokens(tokens) = &segment.inputs else {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the variant takes tokens".to_owned(),
                });
            };
            let Some(rows) = self.rows_mut(segment.state) else {
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message: "the variant has no such slot".to_owned(),
                });
            };
            for token in tokens {
                let bytes = token.to_ne_bytes();
                rows.extend_from_slice(&[
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[0], bytes[1], bytes[2], bytes[3],
                ]);
            }
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 0,
            outputs: Vec::new(),
        })
    }

    fn last_error(&self) -> Option<String> {
        None
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        None
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        None
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        None
    }
}

impl Parts for CellsOnlyConstantRead {
    /// The defect: every read gives the same bytes, whatever the pass wrote.
    fn part_read(&mut self, _state: State, _part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        out.fill(0x5a);
        Ok(())
    }

    fn part_write(&mut self, _state: State, _part: PartId, _bytes: &[u8]) -> WriteResult {
        WriteResult::Refused
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        if part != 0 || out.len() != n as usize * Self::ROW {
            return Err(PartError::Refused);
        }
        let Some(rows) = self.rows(state) else {
            return Err(PartError::Refused);
        };
        let start = from as usize * Self::ROW;
        let end = start + n as usize * Self::ROW;
        let Some(slice) = rows.get(start..end) else {
            return Err(PartError::Refused);
        };
        out.copy_from_slice(slice);
        Ok(())
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        // Refuse the gate's synthetic fill, which is all odd bytes, so that the gate takes the pass path.
        if from != 0 || part != 0 || bytes.len() != n as usize * Self::ROW || bytes.iter().all(|byte| byte & 1 == 1) {
            return WriteResult::Refused;
        }
        let Some(rows) = self.rows_mut(state) else {
            return WriteResult::Refused;
        };
        rows.extend_from_slice(bytes);
        WriteResult::Ok
    }

    fn set_n(&mut self, _state: State, _n: u64) -> WriteResult {
        WriteResult::Ok
    }
}

#[test]
fn a_cells_only_engine_with_a_constant_read_fails_the_gate() {
    assert_fail(
        park_and_restore(&mut CellsOnlyConstantRead::new()),
        "CellsOnlyConstantRead",
    );
}

/// An engine whose pass returns no output, so a gate that asks for a logit row gets nothing to compare.
struct DropsOutputs {
    inner: FakeAlways,
}

impl DropsOutputs {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for DropsOutputs {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let mut output = self.inner.pass(segments, taps)?;
        output.outputs.clear();
        Ok(output)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn an_engine_that_drops_the_wanted_outputs_fails_same_boundaries() {
    let mut make = || DropsOutputs::new();
    assert_fail(
        quantum_same_boundaries(&mut make),
        "DropsOutputs quantum_same_boundaries",
    );
}

/// An engine that refuses a pass unless the segment asks for a logit row.
struct NeedsTheAsk {
    inner: FakeAlways,
}

impl NeedsTheAsk {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for NeedsTheAsk {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        if segments.iter().any(|segment| segment.outputs.logit_rows.is_empty()) {
            return Err(PassError {
                kind: PassErrorKind::Ran(PassStatus::Ok),
                message: "the segment asks for no logit row".to_owned(),
            });
        }
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn same_boundaries_asks_for_the_last_logit_row() {
    let mut make = || NeedsTheAsk::new();
    assert_pass(
        quantum_same_boundaries(&mut make),
        "NeedsTheAsk quantum_same_boundaries",
    );
}

/// An engine whose pass writes the fixed part of another slot.
struct PassWritesOtherSlots {
    inner: FakeAlways,
    other: State,
    part: PartId,
    bytes: Vec<u8>,
}

impl PassWritesOtherSlots {
    fn new() -> Self {
        let inner = FakeAlways::new();
        let limits = inner.limits();
        let (part, fixed) = limits
            .parts
            .iter()
            .enumerate()
            .find(|(_, part)| part.kind == PartKind::Fixed)
            .expect("the fake has a fixed part");
        Self {
            inner,
            other: State::Slot(1),
            part,
            bytes: vec![0xFF; fixed.shape.iter().product::<u64>() as usize],
        }
    }
}

impl Engine for PassWritesOtherSlots {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let output = self.inner.pass(segments, taps)?;
        let _ = self
            .inner
            .parts()
            .unwrap()
            .part_write(self.other, self.part, &self.bytes);
        Ok(output)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn a_pass_that_writes_another_slot_fails_isolation() {
    let mut make = || PassWritesOtherSlots::new();
    assert_fail(quantum_isolation(&mut make), "PassWritesOtherSlots quantum_isolation");
}

/// An engine whose identity changes with each call.
struct UnstableIdentity {
    inner: FakeAlways,
}

static UNSTABLE_CALLS: AtomicU32 = AtomicU32::new(0);

impl UnstableIdentity {
    fn new() -> Self {
        Self {
            inner: FakeAlways::new(),
        }
    }
}

impl Engine for UnstableIdentity {
    fn identity(&self) -> Identity {
        let call = UNSTABLE_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let mut identity = self.inner.identity();
        identity.model = format!("fake-always-{call}");
        identity
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn an_unstable_identity_fails_the_identity_gate() {
    let mut make = || UnstableIdentity::new();
    assert_fail(quantum_identity(&mut make), "UnstableIdentity quantum_identity");
}

/// An engine whose pass writes a call counter into the state, but declares no bit dependence.
struct HistoryDependent {
    inner: FakeAlways,
    calls: u8,
    part: PartId,
    len: usize,
}

impl HistoryDependent {
    fn new() -> Self {
        let inner = FakeAlways::new();
        let limits = inner.limits();
        let (part, fixed) = limits
            .parts
            .iter()
            .enumerate()
            .find(|(_, part)| part.kind == PartKind::Fixed)
            .expect("the fake has a fixed part");
        Self {
            inner,
            calls: 0,
            part,
            len: fixed.shape.iter().product::<u64>() as usize,
        }
    }
}

impl Engine for HistoryDependent {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let output = self.inner.pass(segments, taps)?;
        self.calls = self.calls.wrapping_add(1);
        if let Some(segment) = segments.first() {
            let bytes = vec![self.calls; self.len];
            let _ = self.inner.parts().unwrap().part_write(segment.state, self.part, &bytes);
        }
        Ok(output)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn a_history_dependent_pass_fails_same_boundaries() {
    let mut make = || HistoryDependent::new();
    assert_fail(
        quantum_same_boundaries(&mut make),
        "HistoryDependent quantum_same_boundaries",
    );
}

/// An engine whose coverage reports no rows.
struct CoverageLosesRows {
    inner: FakeAlways,
    limits: Limits,
}

impl CoverageLosesRows {
    fn new() -> Self {
        let inner = FakeAlways::new();
        let limits = inner.limits();
        Self { inner, limits }
    }
}

impl Engine for CoverageLosesRows {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Coverage for CoverageLosesRows {
    fn coverage(&self, _state: State, part: PartId, _bytes: &[u8]) -> CoverageSpan {
        CoverageSpan { part, start: 0, end: 0 }
    }

    fn fits(&self, coverage: &CoverageSpan, slot: u32) -> Result<(), String> {
        let row_bytes = self
            .limits
            .parts
            .get(coverage.part)
            .map(|part| part.shape.iter().product::<u64>())
            .unwrap_or(1);
        let bytes = (coverage.end - coverage.start) * row_bytes;
        let slot_bytes = slot_bytes(&self.limits, slot).unwrap_or(0);
        if bytes <= slot_bytes {
            Ok(())
        } else {
            Err(format!("{bytes} bytes do not fit {slot_bytes}"))
        }
    }
}

#[test]
fn coverage_that_loses_rows_fails_resume_fit() {
    assert_fail(
        quantum_resume_fit(&mut CoverageLosesRows::new()),
        "CoverageLosesRows quantum_resume_fit",
    );
}

/// An engine whose fit check accepts every coverage.
struct FitsAlways {
    inner: FakeAlways,
    limits: Limits,
}

impl FitsAlways {
    fn new() -> Self {
        let inner = FakeAlways::new();
        let limits = inner.limits();
        Self { inner, limits }
    }
}

impl Engine for FitsAlways {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

impl Coverage for FitsAlways {
    fn coverage(&self, _state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        let row_bytes = self
            .limits
            .parts
            .get(part)
            .map(|part| part.shape.iter().product::<u64>())
            .unwrap_or(1)
            .max(1);
        let rows = bytes.len() as u64 / row_bytes;
        CoverageSpan {
            part,
            start: 0,
            end: rows,
        }
    }

    fn fits(&self, _coverage: &CoverageSpan, _slot: u32) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn fits_that_always_accepts_fails_resume_fit() {
    assert_fail(
        quantum_resume_fit(&mut FitsAlways::new()),
        "FitsAlways quantum_resume_fit",
    );
}

/// An engine with a fixed F32 part that its pass fills with NaN. It refuses the gate's synthetic bytes, so the gate
/// takes the pass path, and both sides of the comparison hold the same NaN bits. Spec 8.3 says that fails.
struct NanAfterAPass {
    limits: Limits,
    fixed: [Vec<u8>; 2],
    cells: [Vec<Vec<u8>>; 2],
    last_error: Option<String>,
}

/// The fixed F32 part of `NanAfterAPass`.
const NAN_FIXED: PartId = 0;
/// The cells part of `NanAfterAPass`.
const NAN_CELLS: PartId = 1;

impl NanAfterAPass {
    fn new() -> Self {
        let limits = Limits {
            capabilities: vec![Capability::Parts],
            parts: vec![
                PartLimits {
                    name: "weights".to_owned(),
                    kind: PartKind::Fixed,
                    dtype: Dtype::F32,
                    shape: vec![4],
                    block_format: None,
                },
                PartLimits {
                    name: "rows".to_owned(),
                    kind: PartKind::Cells,
                    dtype: Dtype::U8,
                    shape: vec![8],
                    block_format: None,
                },
            ],
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
            slots: vec![SlotLimit { bytes: 64, count: 1 }, SlotLimit { bytes: 256, count: 1 }],
            max_quantum_ms: 10,
            memory: MemoryLimits::default(),
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
        Self {
            limits,
            fixed: [vec![0u8; 16], vec![0u8; 16]],
            cells: [Vec::new(), Vec::new()],
            last_error: None,
        }
    }

    fn slot(state: State) -> usize {
        let State::Slot(id) = state;
        id as usize
    }
}

impl Engine for NanAfterAPass {
    fn identity(&self) -> Identity {
        Identity {
            model: "nan-after-a-pass".to_owned(),
            checkpoint_digest: "fake:0".to_owned(),
            tokenizer_hash: "fake:0".to_owned(),
            template: String::new(),
            numbers: std::collections::BTreeMap::new(),
            plan_table: String::new(),
            bit_settings: Vec::new(),
            canary_hash: String::new(),
        }
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        for segment in segments {
            let slot = Self::slot(segment.state);
            self.fixed[slot] = f32::NAN.to_le_bytes().repeat(4);
            self.cells[slot].push(vec![7u8; 8]);
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 0,
            outputs: Vec::new(),
        })
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }
}

impl Parts for NanAfterAPass {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let slot = Self::slot(state);
        match part {
            NAN_FIXED => {
                if out.len() != self.fixed[slot].len() {
                    return Err(PartError::Refused);
                }
                out.copy_from_slice(&self.fixed[slot]);
                Ok(())
            }
            NAN_CELLS => {
                let row = 8;
                if out.len() != self.cells[slot].len() * row {
                    return Err(PartError::Refused);
                }
                for (index, bytes) in self.cells[slot].iter().enumerate() {
                    out[index * row..(index + 1) * row].copy_from_slice(bytes);
                }
                Ok(())
            }
            _ => Err(PartError::Refused),
        }
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        if part != NAN_FIXED || bytes.len() != 16 {
            return WriteResult::Refused;
        }
        let nan = f32::NAN.to_le_bytes().repeat(4);
        // The zero fill and the pass's NaN are welcome; the gate's synthetic pattern is not.
        if bytes.iter().all(|byte| *byte == 0) || bytes == nan {
            self.fixed[Self::slot(state)].copy_from_slice(bytes);
            WriteResult::Ok
        } else {
            WriteResult::Refused
        }
    }

    fn cells_read(
        &mut self,
        _state: State,
        _part: PartId,
        _from: u64,
        _n: u64,
        _out: &mut [u8],
    ) -> Result<(), PartError> {
        Err(PartError::Refused)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        if from != 0 || part != NAN_CELLS || bytes.len() as u64 != n * 8 {
            return WriteResult::Refused;
        }
        let rows = &mut self.cells[Self::slot(state)];
        for chunk in bytes.chunks(8) {
            rows.push(chunk.to_vec());
        }
        WriteResult::Ok
    }

    fn set_n(&mut self, _state: State, _n: u64) -> WriteResult {
        WriteResult::Ok
    }
}

#[test]
fn a_nan_part_fails_park_and_restore() {
    assert_fail(park_and_restore(&mut NanAfterAPass::new()), "NanAfterAPass");
}

fn limits_with(mandatory_bytes: u64) -> Limits {
    Limits {
        capabilities: Vec::new(),
        parts: Vec::new(),
        inputs: Inputs::Tokens,
        devices: Vec::new(),
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        slots: Vec::new(),
        max_quantum_ms: 0,
        memory: MemoryLimits {
            mandatory: vec![MandatoryTerm {
                name: "gate".to_owned(),
                tier: Tier::Host,
                bytes: mandatory_bytes,
                lifetime: Lifetime::Engine,
                kinds: Vec::new(),
            }],
            elastic: Vec::new(),
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
    }
}

fn profile(budget_bytes: u64, mandatory_bytes: u64) -> Profile {
    Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: budget_bytes,
        }],
        engines: vec![EnginePlan {
            name: "gate".to_owned(),
            limits: limits_with(mandatory_bytes),
            share: 1,
        }],
    }
}

fn elastic_limits() -> Limits {
    let mut limits = limits_with(0);
    limits.memory.elastic = vec![ElasticTerm {
        name: "gate-cache".to_owned(),
        tier: Tier::Host,
        unit_bytes: 8,
        minimum: 2,
        maximum: 4,
        kinds: Vec::new(),
    }];
    limits
}

fn elastic_profile(budget_bytes: u64) -> Profile {
    Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: budget_bytes,
        }],
        engines: vec![EnginePlan {
            name: "gate".to_owned(),
            limits: elastic_limits(),
            share: 1,
        }],
    }
}

#[test]
fn an_elastic_shortfall_passes_the_admission_gate() {
    let fitting = elastic_profile(16);
    let refused = elastic_profile(8);
    assert_pass(
        admission_refusal_changes_nothing(&mut Ledger::new(), &fitting, &refused),
        "elastic admission",
    );
}

#[test]
fn a_refusing_ledger_passes_the_admission_gate() {
    let fitting = profile(16, 1);
    let refused = profile(0, 1);
    assert_pass(
        admission_refusal_changes_nothing(&mut Ledger::new(), &fitting, &refused),
        "Ledger admission",
    );
}

#[test]
fn a_profile_that_fits_fails_the_admission_gate() {
    let fitting = profile(16, 1);
    assert_fail(
        admission_refusal_changes_nothing(&mut Ledger::new(), &fitting, &fitting),
        "a fitting profile",
    );
}

/// A ledger that records the profile before the refusal.
#[derive(Clone, PartialEq)]
struct GreedyLedger {
    inner: Ledger,
    seen: Vec<Profile>,
}

impl GreedyLedger {
    fn new() -> Self {
        Self {
            inner: Ledger::new(),
            seen: Vec::new(),
        }
    }
}

impl Admit for GreedyLedger {
    fn admit(&mut self, profile: &Profile) -> Result<Admission, LedgerError> {
        self.seen.push(profile.clone());
        self.inner.admit(profile)
    }
}

#[test]
fn a_ledger_that_writes_on_refusal_fails_the_admission_gate() {
    let fitting = profile(16, 1);
    let refused = profile(0, 1);
    assert_fail(
        admission_refusal_changes_nothing(&mut GreedyLedger::new(), &fitting, &refused),
        "GreedyLedger admission",
    );
}

#[test]
fn the_suite_runs_every_gate_and_fake_always_passes() {
    let mut make = || FakeAlways::new();
    let suite = run_all(&mut make);
    let names: Vec<&str> = suite.gates.iter().map(|gate| gate.name).collect();
    assert_eq!(
        names,
        [
            "park_and_restore",
            "seal_only_at_boundaries",
            "quantum_identity",
            "quantum_isolation",
            "quantum_same_boundaries",
            "quantum_resume_fit",
            "admission_refusal_changes_nothing",
            "segments_exact_bits",
        ]
    );
    assert!(suite.passed(), "{suite}");
    assert_eq!(suite.failures().count(), 0);
    assert_eq!(suite.not_applicable().count(), 0);
}

#[test]
fn the_suite_fails_a_broken_engine() {
    let mut make = || FakeBroken::new();
    let suite = run_all(&mut make);
    assert!(!suite.passed(), "{suite}");
    assert!(suite.failures().any(|gate| gate.name == "park_and_restore"), "{suite}");
}

/// An engine whose pass returns the same NaN F32 row each time. The bits are equal in two runs. Spec 8.3 says that
/// a NaN fails, even when the bit patterns are equal.
struct NanOutput {
    inner: FakeAlways,
}

impl Engine for NanOutput {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let mut out = self.inner.pass(segments, taps)?;
        for output in &mut out.outputs {
            output.dtype = Dtype::F32;
            output.bytes = f32::NAN.to_le_bytes().repeat(2);
        }
        Ok(out)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn a_nan_output_with_equal_bits_fails_same_boundaries() {
    let mut make = || NanOutput {
        inner: FakeAlways::new(),
    };
    match quantum_same_boundaries(&mut make) {
        Outcome::Fail(failure) => assert!(failure.message.contains("not finite"), "{}", failure.message),
        other => panic!("expected Fail, got {other}"),
    }
}

fn assert_fail_with(outcome: Outcome, what: &str, text: &str) {
    match outcome {
        Outcome::Fail(failure) => assert!(failure.message.contains(text), "{what}: {}", failure.message),
        other => panic!("{what}: expected Fail, got {other}"),
    }
}

#[test]
fn bits_that_depend_on_the_segment_count_fail_the_segments_gate() {
    let mut make = || FakeSegmentCount::new();
    assert_fail_with(segments_exact_bits(&mut make), "FakeSegmentCount", "2 segments");
    assert!(!run_all(&mut make).passed());
}

#[test]
fn bits_that_depend_on_the_row_count_of_a_pass_fail_the_segments_gate() {
    let mut make = || FakeRowCount::new();
    assert_fail_with(segments_exact_bits(&mut make), "FakeRowCount", "split");
    assert!(!run_all(&mut make).passed());
}

#[test]
fn an_engine_that_does_not_declare_segments_exact_gets_not_applicable() {
    let mut make = || NotExact {
        inner: FakeSegmentCount::new(),
    };
    assert!(matches!(segments_exact_bits(&mut make), Outcome::NotApplicable(_)));
}

#[test]
fn an_engine_with_fewer_slots_than_segments_gets_not_applicable() {
    let mut make = || FakeAlways::with_slots(vec![SlotLimit { bytes: 256, count: 2 }]);
    match segments_exact_bits(&mut make) {
        Outcome::NotApplicable(reason) => assert!(reason.contains("slots"), "{reason}"),
        other => panic!("expected NotApplicable, got {other}"),
    }
}

/// Wraps an engine and changes its answer after a pass of `from` or more segments, or after a pass that continues.
struct AfterManySegments {
    inner: FakeAlways,
    /// The least segment count that changes the answer.
    from: usize,
    /// Change the answer of a pass that starts after position 0.
    continued: bool,
    /// Flip a bit of the pass outputs.
    outputs: bool,
    /// Flip a bit of the fixed state of the first segment.
    state: bool,
}

impl Engine for AfterManySegments {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let mut out = self.inner.pass(segments, taps)?;
        if segments.len() >= self.from || (self.continued && segments[0].positions[0] > 0) {
            if self.outputs {
                out.outputs[0].bytes[0] ^= 1;
            }
            if self.state {
                let mut bytes = [0u8; 16];
                let state = segments[0].state;
                self.inner.parts().unwrap().part_read(state, 0, &mut bytes).unwrap();
                bytes[0] ^= 1;
                self.inner.parts().unwrap().part_write(state, 0, &bytes);
            }
        }
        Ok(out)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

#[test]
fn an_output_that_changes_with_the_segment_count_fails_the_segments_gate() {
    let mut make = || AfterManySegments {
        inner: FakeAlways::new(),
        from: 2,
        continued: false,
        outputs: true,
        state: false,
    };
    assert_fail_with(segments_exact_bits(&mut make), "outputs", "output");
}

#[test]
fn a_state_that_changes_with_the_segment_count_fails_the_segments_gate() {
    let mut make = || AfterManySegments {
        inner: FakeAlways::new(),
        from: 2,
        continued: false,
        outputs: false,
        state: true,
    };
    assert_fail_with(segments_exact_bits(&mut make), "state", "part state");
}

#[test]
fn a_change_at_the_maximum_segment_count_only_fails_the_segments_gate() {
    let max = FakeAlways::new().limits().max_segments as usize;
    let mut make = || AfterManySegments {
        inner: FakeAlways::new(),
        from: max,
        continued: false,
        outputs: true,
        state: false,
    };
    assert_fail_with(segments_exact_bits(&mut make), "maximum", &format!("{max} segments"));
}

#[test]
fn an_output_that_changes_with_a_split_fails_the_segments_gate() {
    let mut make = || AfterManySegments {
        inner: FakeAlways::new(),
        from: usize::MAX,
        continued: true,
        outputs: true,
        state: false,
    };
    assert_fail_with(segments_exact_bits(&mut make), "split output", "split in two passes");
}

#[test]
fn a_state_that_changes_with_a_split_fails_the_segments_gate() {
    let mut make = || AfterManySegments {
        inner: FakeAlways::new(),
        from: usize::MAX,
        continued: true,
        outputs: false,
        state: true,
    };
    assert_fail_with(segments_exact_bits(&mut make), "split state", "part state");
}

#[test]
fn a_nan_output_fails_the_segments_gate() {
    let mut make = || NanOutput {
        inner: FakeAlways::new(),
    };
    assert_fail_with(segments_exact_bits(&mut make), "NaN", "not finite");
}

/// Wraps an engine and declares that its segments are not exact.
struct NotExact {
    inner: FakeSegmentCount,
}

impl Engine for NotExact {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        let mut limits = self.inner.limits();
        limits.segments_exact = false;
        limits
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn nagaya_abi::Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}
