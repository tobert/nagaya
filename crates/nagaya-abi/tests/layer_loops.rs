//! The rules of layer loops in `Limits::check`, the loops of the segments of a pass, and the loops of the fake
//! (engine-abi "Layer loops").
#![cfg(feature = "fake")]

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Engine, LayerLoop, LayerLoopLimit, Limits, OutputsWanted, PassErrorKind, Segment, SegmentInputs, State,
};

fn limit() -> LayerLoopLimit {
    LayerLoopLimit {
        layers: 24,
        bounds: vec![2, 6, 10, 14, 18, 22],
        count: 4,
        extra_runs: 8,
    }
}

fn looped_limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::LayerLoops);
    limits.layer_loops = Some(limit());
    limits
}

fn refused(limits: &Limits, words: &str) {
    match limits.check() {
        Ok(()) => panic!("the limits must break a rule about {words}"),
        Err(message) => assert!(message.contains(words), "the message `{message}` must name {words}"),
    }
}

fn span(start: u32, end: u32, count: u32) -> LayerLoop {
    LayerLoop { start, end, count }
}

fn segment(slot: u32, layer_loop: Option<LayerLoop>) -> Segment {
    Segment {
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(vec![5, 6, 7]),
        positions: vec![0, 1, 2],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![2] },
        control: Vec::new(),
        layer_loop,
    }
}

#[test]
fn the_capability_and_the_limit_come_together_with_rising_bounds() {
    looped_limits().check().unwrap();
    let mut limits = looped_limits();
    limits.capabilities.retain(|c| *c != Capability::LayerLoops);
    refused(&limits, "declares the capability `layer_loops`");
    let mut limits = looped_limits();
    limits.layer_loops = None;
    refused(&limits, "gives limits().layer_loops");
    for bounds in [vec![2], vec![6, 2], vec![2, 2, 6], vec![2, 26]] {
        let mut limits = looped_limits();
        limits.layer_loops = Some(LayerLoopLimit {
            bounds: bounds.clone(),
            ..limit()
        });
        refused(&limits, "in rising order, up to 24");
    }
    let mut limits = looped_limits();
    limits.layer_loops = Some(LayerLoopLimit { count: 1, ..limit() });
    refused(&limits, "the limit is 2 or more");
}

#[test]
fn a_loop_fits_the_bounds_the_count_and_the_reserve() {
    let l = limit();
    l.check(&span(10, 14, 2)).unwrap();
    l.check(&span(6, 14, 2)).unwrap();
    l.check(&span(2, 22, 1)).unwrap();
    assert_eq!(span(6, 14, 2).extra_runs(), 8);
    let seven = LayerLoopLimit {
        extra_runs: 7,
        ..limit()
    };
    let message = seven.check(&span(6, 14, 2)).unwrap_err();
    assert!(
        message.contains("adds 8 runs") && message.contains("each slot holds 7"),
        "{message}"
    );
    for (lp, words) in [
        (span(11, 14, 2), "starts and ends at one of"),
        (span(10, 13, 2), "starts and ends at one of"),
        (span(14, 10, 2), "end above its start"),
        (span(10, 10, 2), "end above its start"),
        (span(10, 14, 0), "1 to 4 times"),
        (span(10, 14, 5), "1 to 4 times"),
        (span(6, 14, 3), "each slot holds 8"),
    ] {
        let message = l.check(&lp).unwrap_err();
        assert!(message.contains(words), "{lp:?}: `{message}` must name {words}");
    }
}

#[test]
fn the_segments_of_one_pass_name_one_loop_that_fits() {
    let limits = looped_limits();
    let a = Some(span(10, 14, 2));
    limits.check_segment_loops(&[segment(0, a), segment(1, a)]).unwrap();
    limits
        .check_segment_loops(&[segment(0, None), segment(1, None)])
        .unwrap();
    let message = limits
        .check_segment_loops(&[segment(0, a), segment(1, None)])
        .unwrap_err();
    assert!(
        message.contains("segment 1") && message.contains("name one loop"),
        "{message}"
    );
    let message = limits
        .check_segment_loops(&[segment(0, a), segment(1, Some(span(6, 10, 2)))])
        .unwrap_err();
    assert!(message.contains("segment 1"), "{message}");
    let message = limits
        .check_segment_loops(&[segment(0, Some(span(10, 14, 5)))])
        .unwrap_err();
    assert!(
        message.contains("segment 0: ") && message.contains("1 to 4 times"),
        "{message}"
    );
    let plain = FakeAlways::new().limits();
    let message = plain.check_segment_loops(&[segment(0, a)]).unwrap_err();
    assert!(message.contains("does not declare `layer_loops`"), "{message}");
}

/// The logits of one pass of three tokens on slot 0 of a new fake, under `layer_loop`.
fn logits(fake: FakeAlways, layer_loop: Option<LayerLoop>) -> Vec<u8> {
    let mut engine = fake;
    let out = engine.pass(&[segment(0, layer_loop)], &[]).expect("a pass");
    out.outputs[0].bytes.clone()
}

#[test]
fn the_fake_runs_a_loop_whose_count_1_changes_no_bit() {
    let fake = || FakeAlways::new().with_vocab(64).with_layer_loops(2);
    let limits = fake().limits();
    limits.check().unwrap();
    assert!(limits.capabilities.contains(&Capability::LayerLoops));
    assert_eq!(
        limits.layer_loops.as_ref().map(|l| (l.layers, l.bounds.clone())),
        Some((2, vec![0, 1, 2]))
    );
    let plain = logits(fake(), None);
    assert_eq!(logits(fake(), Some(span(0, 2, 1))), plain, "count 1 is the plain pass");
    let twice = logits(fake(), Some(span(0, 2, 2)));
    assert_ne!(twice, plain);
    assert_ne!(
        logits(fake(), Some(span(1, 2, 2))),
        twice,
        "another span is another model"
    );
    assert_eq!(
        logits(fake(), Some(span(0, 2, 2))),
        twice,
        "the same loop gives the same bits"
    );
    let mut engine = fake();
    let error = engine.pass(&[segment(0, Some(span(0, 2, 3)))], &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{}", error.message);
    assert!(error.message.contains("each slot holds 2"), "{}", error.message);
    let mut engine = FakeAlways::new().with_vocab(64);
    let error = engine.pass(&[segment(0, Some(span(0, 2, 2)))], &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{}", error.message);
}
