//! The engine gates of spec 8.3, on `FakeAlways` and on broken variants.

use std::sync::atomic::{AtomicU32, Ordering};

use nagaya_abi::fake::{FakeAlways, FakeBroken};
use nagaya_abi::{
    ChainLimits, Coverage, CoverageSpan, Determinism, Engine, Identity, Inputs, Lifetime, Limits, MandatoryTerm,
    MemoryLimits, PartError, PartId, PartKind, Parts, PassError, PassOut, Segment, Slots, State, Tap, Tier,
    WriteResult,
};
use nagaya_conformance::{
    Admit, Outcome, admission_refusal_changes_nothing, park_and_restore, quantum_identity, quantum_isolation,
    quantum_resume_fit, quantum_same_boundaries, seal_only_at_boundaries,
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
}

#[test]
fn fake_broken_fails_park_and_restore() {
    assert_fail(park_and_restore(&mut FakeBroken::new()), "fake_broken park_and_restore");
}

// The broken fake flips a bit in `part_read`. It also fails the seal gate, because that gate compares two reads.
// This test keeps the park gate's failure on its own.

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

    fn cells_write(&mut self, state: State, part: PartId, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner.parts().unwrap().cells_write(state, part, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
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
