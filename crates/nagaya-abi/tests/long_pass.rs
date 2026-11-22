//! The rules of a long pass: the ceiling in `Limits::check`, the declared bound of a pass, and the default of
//! `pass_bound_ms` (engine-abi D8, "A long pass with a declared bound").
#![cfg(feature = "fake")]

use std::num::NonZeroU32;

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Engine, LONG_QUANTUM_CEILING_MS, Limits, OutputsWanted, Segment, SegmentInputs, Stage, State,
};

fn ms(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("a bound above 0")
}

/// The limits of the fake engine with a ceiling of `ceiling` ms.
fn long_limits(ceiling: u32) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.max_quantum_ms = 5_000;
    limits.max_long_quantum_ms = Some(ms(ceiling));
    limits
}

fn refused(limits: &Limits, words: &str) {
    match limits.check() {
        Ok(()) => panic!("the limits must break a rule about {words}"),
        Err(message) => assert!(message.contains(words), "the message `{message}` must name {words}"),
    }
}

fn segment(slot: u32) -> Segment {
    Segment {
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(vec![5, 6, 7]),
        positions: vec![0, 1, 2],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![2] },
        control: Vec::new(),
        layer_loop: None,
        cycles: None,
    }
}

#[test]
fn an_engine_declares_no_long_pass_by_default() {
    let engine = FakeAlways::new();
    assert_eq!(engine.limits().max_long_quantum_ms, None);
    assert_eq!(engine.pass_bound_ms(&[segment(0)], &[]), Ok(None));
}

#[test]
fn the_ceiling_is_above_the_bound_and_at_most_180_s() {
    assert_eq!(LONG_QUANTUM_CEILING_MS, 180_000);
    long_limits(5_001).check().unwrap();
    long_limits(180_000).check().unwrap();
    refused(&long_limits(5_000), "more than max_quantum_ms");
    refused(&long_limits(180_001), "at most 180000 ms");
}

#[test]
fn an_engine_with_stages_declares_no_ceiling() {
    let mut limits = long_limits(60_000);
    limits.capabilities.push(Capability::Stages);
    limits.stages = vec![
        Stage {
            device: 0,
            layers: 0..2,
            head: false,
        },
        Stage {
            device: 1,
            layers: 2..4,
            head: true,
        },
    ];
    refused(&limits, "no long pass");
}

#[test]
fn a_bound_fits_between_the_quantum_bound_and_the_ceiling() {
    let limits = long_limits(60_000);
    let one = [segment(0)];
    limits.check_pass_bound(&one, ms(5_001)).unwrap();
    limits.check_pass_bound(&one, ms(60_000)).unwrap();
    let message = limits.check_pass_bound(&one, ms(5_000)).unwrap_err();
    assert!(message.contains("more than max_quantum_ms"), "{message}");
    let message = limits.check_pass_bound(&one, ms(60_001)).unwrap_err();
    assert!(message.contains("at most max_long_quantum_ms"), "{message}");
}

#[test]
fn a_bound_with_no_ceiling_is_refused() {
    let limits = FakeAlways::new().limits();
    let message = limits.check_pass_bound(&[segment(0)], ms(60_000)).unwrap_err();
    assert!(message.contains("declares no max_long_quantum_ms"), "{message}");
}

#[test]
fn a_long_pass_holds_one_segment() {
    let limits = long_limits(60_000);
    let message = limits
        .check_pass_bound(&[segment(0), segment(1)], ms(10_000))
        .unwrap_err();
    assert!(message.contains("one segment"), "{message}");
}
