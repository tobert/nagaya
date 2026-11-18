//! The fake engine with a feed rule, in one stage and in two (engine-abi D1, D11).

use nagaya_abi::fake::FakeFeed;
use nagaya_abi::{
    Capability, Engine, OutputsWanted, PassErrorKind, Residual, Segment, SegmentInputs, SlotLimit, StageHandle,
    StageOut, State, WriteResult,
};

const VOCAB: usize = 32;

fn slots() -> Vec<SlotLimit> {
    vec![SlotLimit {
        bytes: 8 * 64,
        count: 2,
        tokens: 64,
    }]
}

fn segment(slot: u32, from: u64, tokens: &[u32], logits: bool) -> Segment {
    let to = from + tokens.len() as u64;
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..to).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: if logits { vec![to - 1] } else { Vec::new() },
        },
        layer_loop: None,
    }
}

fn tokens(count: u32) -> Vec<u32> {
    (0..count).map(|token| (token * 7 + 3) % VOCAB as u32).collect()
}

/// Feed `tokens` into slot 0 in passes of the given sizes, and give the logits of the last pass.
fn feed(engine: &mut dyn Engine, tokens: &[u32], split: &[usize]) -> Vec<u8> {
    let mut at = 0;
    let mut logits = Vec::new();
    for (index, size) in split.iter().enumerate() {
        let last = index + 1 == split.len();
        let out = engine
            .pass(&[segment(0, at as u64, &tokens[at..at + size], last)], &[])
            .unwrap();
        if last {
            logits = out.outputs[0].bytes.clone();
        }
        at += size;
    }
    assert_eq!(at, tokens.len(), "the split covers the tokens");
    logits
}

/// The rows of each part of slot 0.
fn rows(engine: &mut dyn Engine) -> Vec<Vec<u8>> {
    let parts = engine.limits().parts.len();
    let high = engine.slots().unwrap().slot_high(0);
    (0..parts)
        .map(|part| {
            let mut out = vec![0u8; high as usize * 8];
            engine
                .parts()
                .unwrap()
                .cells_read(State::Slot(0), part, 0, high, &mut out)
                .unwrap();
            out
        })
        .collect()
}

#[test]
fn the_fake_declares_a_feed_rule_and_its_stages() {
    let one = FakeFeed::new(4, slots()).limits();
    one.check().unwrap();
    assert_eq!(one.feed.unwrap().block, 4);
    assert!(!one.segments_exact);
    assert!(!one.capabilities.contains(&Capability::Stages));
    assert!(one.parts.iter().all(|part| part.stage == 0));
    let two = FakeFeed::two_stages(4, slots()).limits();
    two.check().unwrap();
    assert!(two.capabilities.contains(&Capability::Stages));
    assert_eq!(two.parts.iter().map(|part| part.stage).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(two.devices, vec![0, 1]);
    let identity = FakeFeed::new(4, slots()).identity();
    assert!(
        identity
            .bit_settings
            .iter()
            .any(|setting| setting.name == "feed.block" && setting.value == "4")
    );
}

#[test]
fn the_bits_depend_on_the_split_and_not_on_the_run() {
    let input = tokens(8);
    let mut first = FakeFeed::new(8, slots()).with_vocab(VOCAB);
    let mut second = FakeFeed::new(8, slots()).with_vocab(VOCAB);
    let mut other = FakeFeed::new(8, slots()).with_vocab(VOCAB);
    let logits = feed(&mut first, &input, &[4, 4]);
    assert_eq!(
        feed(&mut second, &input, &[4, 4]),
        logits,
        "one split gives one set of bits"
    );
    assert_eq!(rows(&mut first), rows(&mut second));
    assert_ne!(
        feed(&mut other, &input, &[8]),
        logits,
        "another split gives other logits"
    );
    assert_ne!(rows(&mut other), rows(&mut first), "another split gives other rows");
}

#[test]
fn one_stage_and_two_stages_give_the_same_bits() {
    let input = tokens(10);
    let mut one = FakeFeed::new(4, slots()).with_vocab(VOCAB);
    let mut two = FakeFeed::two_stages(4, slots()).with_vocab(VOCAB);
    assert_eq!(feed(&mut one, &input, &[4, 4, 2]), feed(&mut two, &input, &[4, 4, 2]));
    assert_eq!(rows(&mut one), rows(&mut two));
}

/// Run one pass through the handles, in stage order, with one residual buffer.
fn stage_pass(handles: &mut [Box<dyn StageHandle + Send>], segments: &[Segment], residual: &mut Residual) -> Vec<u8> {
    let mut logits = Vec::new();
    for handle in handles.iter_mut() {
        match handle.stage_pass(segments, residual).unwrap() {
            StageOut::Residual { .. } => {}
            StageOut::Last(out) => {
                logits = out
                    .outputs
                    .first()
                    .map(|output| output.bytes.clone())
                    .unwrap_or_default()
            }
        }
    }
    logits
}

#[test]
fn the_stage_path_gives_the_bits_of_a_pass() {
    let input = tokens(7);
    let mut whole = FakeFeed::two_stages(4, slots()).with_vocab(VOCAB);
    let expected = feed(&mut whole, &input, &[4, 3]);
    let expected_rows = rows(&mut whole);

    let mut staged = FakeFeed::two_stages(4, slots()).with_vocab(VOCAB);
    let mut handles = staged.stages().unwrap().take_handles();
    assert_eq!(handles.len(), 2);
    let mut residual = Residual::new(16, 4);
    stage_pass(&mut handles, &[segment(0, 0, &input[..4], false)], &mut residual);
    let logits = stage_pass(&mut handles, &[segment(0, 4, &input[4..], true)], &mut residual);
    assert_eq!(logits, expected);
    for (part, expected) in expected_rows.iter().enumerate() {
        let mut out = vec![0u8; expected.len()];
        handles[part]
            .parts()
            .unwrap()
            .cells_read(State::Slot(0), part, 0, 7, &mut out)
            .unwrap();
        assert_eq!(&out, expected, "part {part}");
        // A stage refuses the part of the other stage.
        let other = 1 - part;
        assert!(
            handles[part]
                .parts()
                .unwrap()
                .cells_read(State::Slot(0), other, 0, 7, &mut out)
                .is_err()
        );
    }
}

#[test]
fn after_take_handles_the_engine_refuses_its_device_calls() {
    let mut engine = FakeFeed::two_stages(4, slots());
    assert!(
        FakeFeed::new(4, slots()).stages().is_none(),
        "one stage has no stages capability"
    );
    let handles = engine.stages().unwrap().take_handles();
    assert_eq!(handles.len(), 2);
    assert!(
        engine.stages().unwrap().take_handles().is_empty(),
        "a second take gives none"
    );
    let error = engine.pass(&[segment(0, 0, &[1], false)], &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(engine.parts().is_none() && engine.slots().is_none() && engine.coverage().is_none());
    assert!(engine.seal().is_none());
}

#[test]
fn a_refused_pass_changes_nothing() {
    let mut engine = FakeFeed::two_stages(4, slots()).with_vocab(VOCAB);
    feed(&mut engine, &tokens(3), &[3]);
    let before = rows(&mut engine);
    // The first segment is good, and the second one starts at the wrong position.
    let error = engine
        .pass(&[segment(1, 0, &[1, 2], false), segment(0, 0, &[1], false)], &[])
        .unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{}", error.message);
    assert_eq!(rows(&mut engine), before);
    assert_eq!(engine.slots().unwrap().slot_high(1), 0);
}

#[test]
fn a_continued_span_feeds_its_short_last_pass_again() {
    let input = tokens(9);
    let mut once = FakeFeed::new(4, slots()).with_vocab(VOCAB);
    let expected = feed(&mut once, &input, &[4, 4, 1]);
    let expected_rows = rows(&mut once);

    // The first job fed 6 tokens as 4 and 2. The next job drops the short pass and feeds it again with the rest.
    let mut continued = FakeFeed::new(4, slots()).with_vocab(VOCAB);
    feed(&mut continued, &input[..6], &[4, 2]);
    assert_eq!(continued.parts().unwrap().set_n(State::Slot(0), 4), WriteResult::Ok);
    let out = continued.pass(&[segment(0, 4, &input[4..8], false)], &[]).unwrap();
    assert!(out.outputs.is_empty());
    let out = continued.pass(&[segment(0, 8, &input[8..], true)], &[]).unwrap();
    assert_eq!(out.outputs[0].bytes, expected);
    assert_eq!(rows(&mut continued), expected_rows);

    // A job that keeps the short pass and feeds after it breaks the rule, and its bits differ.
    let mut broken = FakeFeed::new(4, slots()).with_vocab(VOCAB);
    feed(&mut broken, &input[..6], &[4, 2]);
    let out = broken.pass(&[segment(0, 6, &input[6..], true)], &[]).unwrap();
    assert_ne!(out.outputs[0].bytes, expected);
}

#[test]
fn the_reverse_residual_fake_differs_only_on_the_stage_path() {
    let input = tokens(4);
    let mut whole = FakeFeed::reverse_residual(4, slots()).with_vocab(VOCAB);
    let mut good = FakeFeed::two_stages(4, slots()).with_vocab(VOCAB);
    let expected = feed(&mut good, &input, &[4]);
    assert_eq!(feed(&mut whole, &input, &[4]), expected, "Engine::pass is right");
    let mut staged = FakeFeed::reverse_residual(4, slots()).with_vocab(VOCAB);
    let mut handles = staged.stages().unwrap().take_handles();
    let mut residual = Residual::new(16, 4);
    assert_ne!(
        stage_pass(&mut handles, &[segment(0, 0, &input, true)], &mut residual),
        expected
    );
}

#[test]
fn the_history_fake_gives_other_bits_after_a_restore() {
    let input = tokens(8);
    let mut once = FakeFeed::history(4, slots()).with_vocab(VOCAB);
    let expected = feed(&mut once, &input, &[4, 4]);
    // Copy the first span into a fresh engine, as a restore does, and feed the second span.
    let mut first = FakeFeed::history(4, slots()).with_vocab(VOCAB);
    feed(&mut first, &input[..4], &[4]);
    let held = rows(&mut first);
    let mut restored = FakeFeed::history(4, slots()).with_vocab(VOCAB);
    for (part, bytes) in held.iter().enumerate() {
        assert_eq!(
            restored.parts().unwrap().cells_write(State::Slot(0), part, 0, 4, bytes),
            WriteResult::Ok
        );
    }
    let out = restored.pass(&[segment(0, 4, &input[4..], true)], &[]).unwrap();
    assert_ne!(out.outputs[0].bytes, expected);
    // The good fake gives the same bits after the same restore.
    let mut good = FakeFeed::new(4, slots()).with_vocab(VOCAB);
    let expected = feed(&mut good, &input, &[4, 4]);
    let mut restored = FakeFeed::new(4, slots()).with_vocab(VOCAB);
    for (part, bytes) in held.iter().enumerate() {
        assert_eq!(
            restored.parts().unwrap().cells_write(State::Slot(0), part, 0, 4, bytes),
            WriteResult::Ok
        );
    }
    let out = restored.pass(&[segment(0, 4, &input[4..], true)], &[]).unwrap();
    assert_eq!(out.outputs[0].bytes, expected);
}
