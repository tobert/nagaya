//! The gates of the stage path and the feed rule (engine-abi D1, D11), on FakeFeed and its broken variants.

use nagaya_abi::fake::{FakeAlways, FakeFeed};
use nagaya_abi::{
    Coverage, CoverageSpan, Dtype, Engine, Identity, Limits, PartError, PartId, Parts, PassError, PassOut, Residual,
    Seal, SealError, Segment, SlotLimit, Slots, StageHandle, StageOut, Stages, State, Tap, WriteResult,
};
use nagaya_conformance::{
    Outcome, feed_rule_bits, quantum_same_boundaries, run_all, segments_exact_bits, stage_path_bits,
};

fn slots() -> Vec<SlotLimit> {
    vec![SlotLimit {
        bytes: 8 * 64,
        count: 2,
        tokens: 64,
    }]
}

fn assert_fail(outcome: Outcome, words: &str) {
    match outcome {
        Outcome::Fail(failure) => assert!(failure.message.contains(words), "{failure}"),
        other => panic!("expected a failure that names {words}, got {other}"),
    }
}

#[test]
fn the_fake_with_two_stages_passes_every_gate() {
    let suite = run_all(&mut || FakeFeed::two_stages(4, slots()).with_vocab(32));
    assert!(suite.passed(), "{suite}");
    for name in ["stage_path_bits", "feed_rule_bits"] {
        let gate = suite.gates.iter().find(|gate| gate.name == name).unwrap();
        assert_eq!(gate.outcome, Outcome::Pass, "{name}");
    }
}

#[test]
fn the_fake_with_one_stage_passes_the_feed_gate_and_has_no_stage_gate() {
    let mut make = || FakeFeed::new(4, slots()).with_vocab(32);
    assert_eq!(feed_rule_bits(&mut make), Outcome::Pass);
    assert!(matches!(stage_path_bits(&mut make), Outcome::NotApplicable(_)));
    assert!(matches!(
        feed_rule_bits(&mut FakeAlways::new),
        Outcome::NotApplicable(_)
    ));
}

#[test]
fn a_stage_path_that_reads_the_residual_in_reverse_fails_the_stage_gate() {
    let mut make = || FakeFeed::reverse_residual(4, slots()).with_vocab(32);
    assert_fail(stage_path_bits(&mut make), "output logits");
}

#[test]
fn an_engine_whose_bits_depend_on_its_history_fails_the_feed_gate() {
    let mut make = || FakeFeed::history(4, slots()).with_vocab(32);
    assert_fail(feed_rule_bits(&mut make), "output logits");
}

/// The fake with two stages, whose stage handles flip the low bit of the first row on a read. Its outputs are right.
struct FlipOnStageRead(FakeFeed);

impl Engine for FlipOnStageRead {
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
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.0.parts()
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        Some(self)
    }
}

impl Stages for FlipOnStageRead {
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>> {
        let handles = self.0.stages().unwrap().take_handles();
        handles
            .into_iter()
            .map(|inner| Box::new(Flip(inner)) as Box<dyn StageHandle + Send>)
            .collect()
    }
}

struct Flip(Box<dyn StageHandle + Send>);

impl StageHandle for Flip {
    fn stage_pass(&mut self, segments: &[Segment], residual: &mut Residual) -> Result<StageOut, PassError> {
        self.0.stage_pass(segments, residual)
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }
}

impl Parts for Flip {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.0.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        self.0.parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let result = self.0.parts().unwrap().cells_read(state, part, from, n, out);
        if let Some(byte) = out.first_mut() {
            *byte ^= 1;
        }
        result
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.0.parts().unwrap().cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.0.parts().unwrap().set_n(state, n)
    }
}

#[test]
fn a_stage_handle_whose_parts_differ_fails_the_stage_gate() {
    let mut make = || FlipOnStageRead(FakeFeed::two_stages(4, slots()).with_vocab(32));
    assert_fail(stage_path_bits(&mut make), "part `k.00`");
}

/// A fake with row parts only (no fixed parts) that flips a row cell in `cells_read` on its second instance, or
/// writes NaN into a float row part, or refuses `Coverage::fits` during `Store::restore`.
struct RowOnlyBroken {
    inner: std::cell::RefCell<FakeFeed>,
    flip_cells: bool,
    nan_cells: bool,
    refuse_fits: bool,
}

impl Engine for RowOnlyBroken {
    fn identity(&self) -> Identity {
        self.inner.borrow().identity()
    }

    fn limits(&self) -> Limits {
        let mut limits = self.inner.borrow().limits();
        limits.segments_exact = true;
        if self.nan_cells {
            for part in &mut limits.parts {
                part.dtype = Dtype::F32;
                part.shape = vec![2];
            }
        }
        limits
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.get_mut().pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.borrow().last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.get_mut().slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        Some(self)
    }
}

impl Seal for RowOnlyBroken {
    fn seal(&mut self, _state: State) -> Result<(), SealError> {
        Ok(())
    }
}

impl Coverage for RowOnlyBroken {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        Coverage::coverage(&*self.inner.borrow(), state, part, bytes)
    }

    fn fits(&self, coverage: &CoverageSpan, slot: u32) -> Result<(), String> {
        if self.refuse_fits {
            return Err("test: fits refused in feed_rule_bits".to_owned());
        }
        Coverage::fits(&*self.inner.borrow(), coverage, slot)
    }
}

impl Parts for RowOnlyBroken {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.get_mut().parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        self.inner.get_mut().parts().unwrap().part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let result = self
            .inner
            .get_mut()
            .parts()
            .unwrap()
            .cells_read(state, part, from, n, out);
        if self.flip_cells
            && let Some(byte) = out.first_mut()
        {
            *byte ^= 1;
        }
        if self.nan_cells && out.len() >= 4 {
            out[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        result
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.inner
            .get_mut()
            .parts()
            .unwrap()
            .cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.get_mut().parts().unwrap().set_n(state, n)
    }
}

#[test]
fn quantum_same_boundaries_and_segments_exact_bits_check_row_parts_when_no_fixed_part_exists() {
    let count = std::cell::Cell::new(0usize);
    let mut make = || {
        let index = count.get();
        count.set(index + 1);
        RowOnlyBroken {
            inner: std::cell::RefCell::new(
                FakeFeed::new(
                    4,
                    vec![SlotLimit {
                        bytes: 8 * 64,
                        count: 8,
                        tokens: 64,
                    }],
                )
                .with_max_pass_inputs(32)
                .with_vocab(32),
            ),
            flip_cells: index % 2 == 1,
            nan_cells: false,
            refuse_fits: false,
        }
    };
    assert_fail(quantum_same_boundaries(&mut make), "part k.00");
    assert_fail(segments_exact_bits(&mut make), "part k.00");
}

#[test]
fn non_finite_row_parts_fail_feed_rule_bits() {
    let mut make = || RowOnlyBroken {
        inner: std::cell::RefCell::new(FakeFeed::new(4, slots()).with_vocab(32)),
        flip_cells: false,
        nan_cells: true,
        refuse_fits: false,
    };
    assert_fail(feed_rule_bits(&mut make), "not finite");
}

#[test]
fn feed_rule_bits_restores_through_the_store_and_checks_fits() {
    let mut make = || RowOnlyBroken {
        inner: std::cell::RefCell::new(FakeFeed::new(4, slots()).with_vocab(32)),
        flip_cells: false,
        nan_cells: false,
        refuse_fits: true,
    };
    assert_fail(feed_rule_bits(&mut make), "fits refused");
}
