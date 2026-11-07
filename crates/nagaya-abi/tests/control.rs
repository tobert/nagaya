//! The rules of the control table in `Limits::check`, and the control list of a segment (engine-abi "Control
//! vectors").
#![cfg(feature = "fake")]

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    CONTROL_TERM, Capability, ControlItem, ControlLimit, ControlOp, Engine, Lifetime, Limits, MandatoryTerm,
    OutputsWanted, PassErrorKind, Segment, SegmentInputs, State, Tier,
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
    ControlItem { index, op, strength }
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
    };
    let error = engine.pass(&[segment], &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(error.message.contains("control"), "{}", error.message);
}
