//! The rules of the control table in `Limits::check`, and the control list of a segment (engine-abi "Control
//! vectors").
#![cfg(feature = "fake")]

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    CONTROL_TERM, Capability, ControlItem, ControlLimit, ControlOp, Engine, Lifetime, Limits, MandatoryTerm,
    OutputsWanted, PassErrorKind, Segment, SegmentInputs, State, Tap, Tier,
};

const TABLE: ControlLimit = ControlLimit {
    tables: 8,
    layers: 2,
    width: 4,
};

/// The limits of the fake with a control table of 8 entries of 2 layers of 4 values.
fn controlled() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Control);
    limits.control = Some(TABLE);
    limits.memory.mandatory.push(MandatoryTerm {
        name: CONTROL_TERM.to_owned(),
        tier: Tier::Device(0),
        bytes: 8 * 2 * 4 * 4,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    });
    limits
}

fn refused(limits: &Limits, words: &str) {
    match limits.check() {
        Ok(()) => panic!("the limits must break a rule about {words}"),
        Err(message) => assert!(message.contains(words), "the message `{message}` must name {words}"),
    }
}

fn item(index: u32, op: ControlOp, strength: f32) -> ControlItem {
    ControlItem {
        index,
        op,
        strength,
        layers: None,
    }
}

#[test]
fn the_capability_the_limit_and_the_term_come_together() {
    controlled().check().unwrap();
    assert_eq!(TABLE.entry_bytes(), 32);
    assert_eq!(TABLE.table_bytes(), 256);
    let mut limits = controlled();
    limits.capabilities.retain(|c| *c != Capability::Control);
    refused(&limits, "declares the capability `control`");
    let mut limits = controlled();
    limits.control = None;
    refused(&limits, "gives limits().control");
    let mut limits = controlled();
    limits.memory.mandatory.retain(|term| term.name != CONTROL_TERM);
    refused(&limits, "the mandatory term `control`");
    let mut limits = controlled();
    limits.memory.mandatory.last_mut().unwrap().bytes = 128;
    refused(&limits, "it must be 256 device bytes");
    let mut limits = controlled();
    limits.memory.mandatory.last_mut().unwrap().tier = Tier::Host;
    refused(&limits, "it must be 256 device bytes");
    let mut limits = controlled();
    limits.memory.mandatory.last_mut().unwrap().lifetime = Lifetime::Context;
    refused(&limits, "it must be 256 device bytes");
    let mut limits = controlled();
    limits.control = Some(ControlLimit { tables: 0, ..TABLE });
    refused(&limits, "each count is above 0");
}

#[test]
fn a_control_list_names_entries_of_the_table_with_finite_strengths() {
    TABLE
        .check(&[item(0, ControlOp::Add, 4.0), item(7, ControlOp::Clamp, 0.0)])
        .unwrap();
    let error = TABLE
        .check(&[item(0, ControlOp::Add, 1.0), item(8, ControlOp::Add, 1.0)])
        .unwrap_err();
    assert!(error.contains("item 1 names table entry 8"), "{error}");
    let error = TABLE.check(&[item(0, ControlOp::Ablate, f32::NAN)]).unwrap_err();
    assert!(error.contains("finite"), "{error}");
    let error = TABLE.check(&[item(0, ControlOp::Ablate, f32::INFINITY)]).unwrap_err();
    assert!(error.contains("finite"), "{error}");
}

#[test]
fn an_add_or_an_ablate_of_strength_0_changes_nothing_and_a_clamp_always_changes() {
    assert!(item(0, ControlOp::Add, 0.0).changes_nothing());
    assert!(item(0, ControlOp::Ablate, -0.0).changes_nothing());
    assert!(!item(0, ControlOp::Clamp, 0.0).changes_nothing());
    assert!(!item(0, ControlOp::Add, 1e-30).changes_nothing());
}

#[test]
fn two_items_are_equal_when_their_bits_are_equal() {
    assert_eq!(item(1, ControlOp::Add, 2.0), item(1, ControlOp::Add, 2.0));
    assert_ne!(item(1, ControlOp::Add, 0.0), item(1, ControlOp::Add, -0.0));
    assert_ne!(item(1, ControlOp::Add, 2.0), item(1, ControlOp::Ablate, 2.0));
    assert_ne!(item(1, ControlOp::Add, 2.0), item(2, ControlOp::Add, 2.0));
}

#[test]
fn the_fake_has_no_control_and_refuses_a_segment_with_items() {
    let mut engine = FakeAlways::new();
    assert!(engine.control().is_none());
    assert!(!engine.limits().capabilities.contains(&Capability::Control));
    let segment = Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(vec![1, 2]),
        positions: vec![0, 1],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![1] },
        control: vec![item(0, ControlOp::Add, 1.0)],
        layer_loop: None,
    };
    let error = engine.pass(&[segment], &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(error.message.contains("control"), "{}", error.message);
}

/// The fake over a vocabulary of 32 tokens, with a table of 8 entries of 2 layers of 4 values.
fn steered_fake() -> FakeAlways {
    FakeAlways::new().with_vocab(32).with_control(8)
}

/// The bytes of a vector of the fake's table: 2 layers of 4 values.
fn vector(values: [f32; 8]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A pass of `tokens` from position 0 of a cleared slot 0, with the last row and `items`. Give the names and the bytes
/// of the outputs.
fn run(engine: &mut FakeAlways, tokens: &[u32], items: &[ControlItem], taps: &[Tap]) -> Vec<(String, Vec<u8>)> {
    engine.slots().unwrap().slot_clear(0);
    let n = tokens.len() as u64;
    let segment = Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (0..n).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: vec![n - 1],
        },
        control: items.to_vec(),
        layer_loop: None,
    };
    let out = engine.pass(&[segment], taps).unwrap();
    out.outputs.into_iter().map(|o| (o.name, o.bytes)).collect()
}

fn residual() -> Tap {
    Tap {
        name: "residual".to_owned(),
    }
}

#[test]
fn the_fake_with_a_table_declares_control_and_the_residual_tap() {
    let mut engine = steered_fake();
    let limits = engine.limits();
    limits.check().unwrap();
    assert_eq!(limits.control, Some(TABLE));
    assert!(limits.capabilities.contains(&Capability::Control));
    assert!(limits.taps.iter().any(|tap| tap.name == "residual"));
    assert!(engine.control().is_some());
    let plain = FakeAlways::new().limits();
    assert!(plain.taps.is_empty() && plain.control.is_none());
}

#[test]
fn the_fake_table_refuses_a_wrong_entry_a_wrong_size_or_a_value_that_is_not_finite() {
    let mut engine = steered_fake();
    let control = engine.control().unwrap();
    control.control_write(7, &vector([1.0; 8])).unwrap();
    assert!(
        control
            .control_write(8, &vector([1.0; 8]))
            .unwrap_err()
            .contains("8 entries")
    );
    assert!(
        control
            .control_write(0, &vector([1.0; 8])[4..])
            .unwrap_err()
            .contains("32 bytes")
    );
    let mut bad = [1.0; 8];
    bad[5] = f32::INFINITY;
    assert!(control.control_write(0, &vector(bad)).unwrap_err().contains("value 5"));
}

#[test]
fn an_add_or_an_ablate_of_0_gives_the_bits_of_plain_and_other_items_change_them() {
    let mut engine = steered_fake();
    let tokens = [3, 1, 4, 1, 5];
    let control = engine.control().unwrap();
    control
        .control_write(2, &vector([0.5, -1.0, 2.0, 0.25, 1.0, 1.0, -3.0, 0.0]))
        .unwrap();
    control
        .control_write(5, &vector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let plain = run(&mut engine, &tokens, &[], &[residual()]);
    let zero = run(
        &mut engine,
        &tokens,
        &[item(2, ControlOp::Add, 0.0), item(5, ControlOp::Ablate, -0.0)],
        &[residual()],
    );
    assert_eq!(zero, plain, "an add and an ablate of strength 0 change no bit");
    for (what, items) in [
        ("a clamp of 0", vec![item(2, ControlOp::Clamp, 0.0)]),
        ("an add of 1", vec![item(2, ControlOp::Add, 1.0)]),
        ("an ablate of 1", vec![item(2, ControlOp::Ablate, 1.0)]),
        ("an add of the other vector", vec![item(5, ControlOp::Add, 1.0)]),
        ("an add of 2", vec![item(2, ControlOp::Add, 2.0)]),
    ] {
        let steered = run(&mut engine, &tokens, &items, &[residual()]);
        let names: Vec<&str> = steered.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["state", "logits", "residual"], "{what}");
        for (index, (name, bytes)) in steered.iter().enumerate() {
            assert_ne!(*bytes, plain[index].1, "{what} changes the output `{name}`");
        }
        assert_eq!(
            run(&mut engine, &tokens, &items, &[residual()]),
            steered,
            "{what} twice"
        );
    }
    let one = run(&mut engine, &tokens, &[item(2, ControlOp::Add, 1.0)], &[]);
    let other = run(&mut engine, &tokens, &[item(5, ControlOp::Add, 1.0)], &[]);
    assert_ne!(one, other, "another vector in the entry gives other bits");
}

#[test]
fn the_fake_residual_tap_gives_each_layer_of_each_logit_row() {
    let mut engine = steered_fake();
    let out = run(&mut engine, &[7, 8, 9], &[], &[residual()]);
    let rows = &out.iter().find(|(name, _)| name == "residual").unwrap().1;
    assert_eq!(rows.len(), 2 * 4 * 4, "one logit row of 2 layers of 4 f32 values");
    let values: Vec<f32> = rows.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert!(
        values.iter().all(|v| v.is_finite()) && values.iter().any(|v| *v != 0.0),
        "{values:?}"
    );
    // A pass with no tap gives no rows, and a tap that the fake does not declare is refused.
    let untapped = run(&mut engine, &[7, 8, 9], &[], &[]);
    assert!(untapped.iter().all(|(name, _)| name != "residual"));
    let error = engine
        .pass(
            &[Segment {
                state: State::Slot(0),
                inputs: SegmentInputs::Tokens(vec![1]),
                positions: vec![0],
                meta: Vec::new(),
                outputs: OutputsWanted::default(),
                control: Vec::new(),
                layer_loop: None,
            }],
            &[Tap {
                name: "routes".to_owned(),
            }],
        )
        .unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(error.message.contains("routes"), "{}", error.message);
    // The fake without a table has no tap either.
    let error = FakeAlways::new().pass(&[], &[residual()]).unwrap_err();
    assert!(error.message.contains("residual"), "{}", error.message);
}

fn lens() -> Tap {
    Tap {
        name: "lens".to_owned(),
    }
}

/// The tap `lens` of the fake (spec 13.1): a logit row for each layer of its one logit row. The row of the last layer
/// is the logit row, as the head of the last layer is the head of the model.
#[test]
fn the_fake_lens_gives_a_logit_row_for_each_layer_and_the_last_is_the_head() {
    let mut engine = steered_fake().with_lens();
    let limits = engine.limits();
    limits.check().unwrap();
    let declared = limits.taps.iter().find(|tap| tap.name == "lens").unwrap();
    assert_eq!(declared.conditions, ["one logit row"]);
    let out = run(&mut engine, &[7, 8, 9], &[], &[lens()]);
    let names: Vec<&str> = out.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        ["state", "logits", "lens"],
        "the lens alone gives no residual rows"
    );
    let logits = &out[1].1;
    let rows = &out[2].1;
    assert_eq!(rows.len(), 2 * 32 * 4, "2 layers of 32 f32 values");
    assert_eq!(
        &rows[32 * 4..],
        &logits[..],
        "the lens of the last layer is the logit row"
    );
    assert_ne!(&rows[..32 * 4], &logits[..], "the lens of layer 0 is another row");
    let both = run(&mut engine, &[7, 8, 9], &[], &[residual(), lens()]);
    let names: Vec<&str> = both.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["state", "logits", "residual", "lens"]);
    assert_eq!(both[0..2], out[0..2], "the taps change no other output");
    // A control item changes the lens of each layer.
    engine.control().unwrap().control_write(1, &vector([1.0; 8])).unwrap();
    let steered = run(&mut engine, &[7, 8, 9], &[item(1, ControlOp::Add, 1.0)], &[lens()]);
    assert_ne!(steered[2].1[..32 * 4], rows[..32 * 4]);
    // The lens reads one logit row: a pass with two is refused.
    engine.slots().unwrap().slot_clear(0);
    let two = Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(vec![1, 2]),
        positions: vec![0, 1],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![0, 1] },
        control: Vec::new(),
        layer_loop: None,
    };
    let error = engine.pass(&[two], &[lens()]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(error.message.contains("one logit row"), "{}", error.message);
    // A fake with no lens refuses the tap.
    let error = steered_fake().pass(&[], &[lens()]).unwrap_err();
    assert!(error.message.contains("lens"), "{}", error.message);
}
