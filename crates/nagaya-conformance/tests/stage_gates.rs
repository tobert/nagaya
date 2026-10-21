//! The gates of the stage path and the feed rule (engine-abi D1, D11), on FakeFeed and its broken variants.

use nagaya_abi::fake::{FakeAlways, FakeFeed};
use nagaya_abi::{
    Engine, Identity, Limits, PartError, PartId, Parts, PassError, PassOut, Residual, Segment, SlotLimit, StageHandle,
    StageOut, Stages, State, Tap, WriteResult,
};
use nagaya_conformance::{Outcome, feed_rule_bits, run_all, stage_path_bits};

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
