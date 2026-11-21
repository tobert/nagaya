//! The rules of cycles in `Limits::check` and the cycles of the segments of a pass (engine-abi "Inkling 2").
#![cfg(feature = "fake")]

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, CyclesLimit, Engine, LayerLoop, LayerLoopLimit, Limits, OutputsWanted, Segment, SegmentInputs, State,
};

fn limit() -> CyclesLimit {
    CyclesLimit {
        min: 1,
        max: 5,
        default: 3,
        start: 6,
        end: 18,
    }
}

fn cycled_limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Cycles);
    limits.cycles = Some(limit());
    limits
}

fn refused(limits: &Limits, words: &str) {
    match limits.check() {
        Ok(()) => panic!("the limits must break a rule about {words}"),
        Err(message) => assert!(message.contains(words), "the message `{message}` must name {words}"),
    }
}

fn segment(slot: u32, cycles: Option<u8>) -> Segment {
    Segment {
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(vec![5, 6, 7]),
        positions: vec![0, 1, 2],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![2] },
        control: Vec::new(),
        layer_loop: None,
        cycles,
    }
}

#[test]
fn the_capability_and_the_limit_come_together_in_order() {
    cycled_limits().check().unwrap();
    let mut limits = cycled_limits();
    limits.capabilities.retain(|c| *c != Capability::Cycles);
    refused(&limits, "declares the capability `cycles`");
    let mut limits = cycled_limits();
    limits.cycles = None;
    refused(&limits, "gives limits().cycles");
    for bad in [
        CyclesLimit { min: 0, ..limit() },
        CyclesLimit { default: 6, ..limit() },
        CyclesLimit { min: 4, ..limit() },
        CyclesLimit { start: 18, ..limit() },
    ] {
        let mut limits = cycled_limits();
        limits.cycles = Some(bad);
        refused(&limits, "1 <= min <= default <= max, and start < end");
    }
}

#[test]
fn a_count_is_from_min_to_max_and_none_is_the_default() {
    let l = limit();
    assert_eq!(l.resolve(None), 3);
    assert_eq!(l.resolve(Some(5)), 5);
    l.check(1).unwrap();
    l.check(5).unwrap();
    for count in [0, 6] {
        let message = l.check(count).unwrap_err();
        assert!(message.contains("1 to 5"), "{message}");
    }
}

#[test]
fn the_segments_of_one_pass_run_one_count_and_none_is_the_default() {
    let limits = cycled_limits();
    limits
        .check_segment_cycles(&[segment(0, None), segment(1, Some(3))])
        .unwrap();
    limits
        .check_segment_cycles(&[segment(0, Some(4)), segment(1, Some(4))])
        .unwrap();
    let message = limits
        .check_segment_cycles(&[segment(0, None), segment(1, Some(4))])
        .unwrap_err();
    assert!(
        message.contains("segment 1 runs 4 cycles, and segment 0 runs 3"),
        "{message}"
    );
    let message = limits.check_segment_cycles(&[segment(0, Some(6))]).unwrap_err();
    assert!(message.starts_with("segment 0:"), "{message}");
}

#[test]
fn an_engine_without_cycles_refuses_a_count() {
    let limits = FakeAlways::new().limits();
    limits.check_segment_cycles(&[segment(0, None)]).unwrap();
    let message = limits.check_segment_cycles(&[segment(0, Some(3))]).unwrap_err();
    assert!(message.contains("does not declare `cycles`"), "{message}");
}

#[test]
fn a_segment_names_a_loop_or_cycles_and_not_both() {
    let mut limits = cycled_limits();
    limits.capabilities.push(Capability::LayerLoops);
    limits.layer_loops = Some(LayerLoopLimit {
        layers: 24,
        bounds: vec![0, 6, 18, 24],
        count: 2,
        extra_runs: 12,
    });
    let mut both = segment(0, Some(3));
    both.layer_loop = Some(LayerLoop {
        start: 6,
        end: 18,
        count: 2,
    });
    let message = limits.check_segment_cycles(&[both]).unwrap_err();
    assert!(message.contains("a loop and cycles"), "{message}");
}
