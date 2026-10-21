//! Each instrument has the name, the unit and the attributes of spec 9.4.
//!
//! These tests call the instruments directly. The tests of the crates that own the call sites show that those sites
//! record (spec 9.5).

use nagaya_telemetry::testing::{Capture, Point};
use nagaya_telemetry::{
    EngineState, EvictReason, LookupOutcome, QuantumLabels, RefusalGate, RestoreResult, TierLabel, instruments,
};

fn only(points: Vec<Point>) -> Point {
    assert_eq!(points.len(), 1, "one point wanted: {points:?}");
    points.into_iter().next().unwrap()
}

fn attributes(point: &Point) -> Vec<(&str, &str)> {
    point
        .attributes
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

#[test]
fn quantum_durations_carry_engine_group_kind_and_class() {
    let capture = Capture::start(Vec::new());
    let labels = QuantumLabels {
        engine: "qwen",
        device_group: 2,
        kind: "prefill",
        class: "interactive",
    };
    instruments().quantum_wait(&labels, 4.0);
    instruments().quantum_run(&labels, 30.0);
    instruments().quantum_run(&labels, 10.0);

    let wait = only(capture.metric("nagaya.quantum.wait.duration"));
    assert_eq!(wait.unit, "ms");
    assert_eq!((wait.count, wait.value), (1, 4.0));
    assert_eq!(
        attributes(&wait),
        [
            ("class", "interactive"),
            ("device_group", "2"),
            ("engine", "qwen"),
            ("kind", "prefill")
        ]
    );
    let run = only(capture.metric("nagaya.quantum.run.duration"));
    assert_eq!((run.count, run.value), (2, 40.0));
}

#[test]
fn ledger_instruments_carry_tier_engine_and_term_or_gate() {
    let capture = Capture::start(Vec::new());
    instruments().ledger_bytes(TierLabel::Device(0), "qwen", "weights", 4096);
    instruments().ledger_refusal(&[TierLabel::Host, TierLabel::Device(0)], "all", RefusalGate::Budget);

    let bytes = only(capture.metric("nagaya.ledger.bytes"));
    assert_eq!(bytes.unit, "By");
    assert_eq!(bytes.value, 4096.0);
    assert_eq!(
        attributes(&bytes),
        [("engine", "qwen"), ("term", "weights"), ("tier", "device0")]
    );
    let refusals = only(capture.metric("nagaya.ledger.refusals"));
    assert_eq!(refusals.unit, "{refusal}");
    assert_eq!(
        attributes(&refusals),
        [("engine", "all"), ("gate", "budget"), ("tier", "device0+host")]
    );
}

#[test]
fn store_instruments_carry_the_labels_of_the_spec() {
    let capture = Capture::start(Vec::new());
    instruments().store_held(TierLabel::Nvme, "qwen", 900, 3);
    instruments().store_lookup("qwen", LookupOutcome::Approx);
    instruments().store_eviction(TierLabel::Host, EvictReason::PutRoom, 2);
    instruments().store_restore("qwen", RestoreResult::Failed);

    let bytes = only(capture.metric("nagaya.store.bytes"));
    assert_eq!((bytes.unit.as_str(), bytes.value), ("By", 900.0));
    assert_eq!(attributes(&bytes), [("engine", "qwen"), ("tier", "nvme")]);
    let entries = only(capture.metric("nagaya.store.entries"));
    assert_eq!((entries.unit.as_str(), entries.value), ("{entry}", 3.0));
    let lookups = only(capture.metric("nagaya.store.lookups"));
    assert_eq!(lookups.unit, "{lookup}");
    assert_eq!(attributes(&lookups), [("engine", "qwen"), ("outcome", "approx")]);
    let evictions = only(capture.metric("nagaya.store.evictions"));
    assert_eq!((evictions.unit.as_str(), evictions.value), ("{entry}", 2.0));
    assert_eq!(attributes(&evictions), [("reason", "put_room"), ("tier", "host")]);
    let restores = only(capture.metric("nagaya.store.restores"));
    assert_eq!(attributes(&restores), [("engine", "qwen"), ("result", "failed")]);
}

#[test]
fn engine_state_is_one_for_the_current_state_and_zero_for_the_rest() {
    let capture = Capture::start(Vec::new());
    instruments().engine_state("qwen", EngineState::Loading);
    instruments().engine_state("qwen", EngineState::Ready);

    let points = capture.metric("nagaya.engine.state");
    assert_eq!(points.len(), 5, "one series for each state: {points:?}");
    for point in &points {
        let state = point.attributes["state"].as_str();
        assert_eq!(point.unit, "1");
        assert_eq!(point.value, if state == "ready" { 1.0 } else { 0.0 }, "state {state}");
    }
}

#[test]
fn a_thread_with_no_scope_and_no_install_records_nothing() {
    // No `Capture` here: the instruments are the no-op ones. The call must not panic.
    instruments().store_lookup("qwen", LookupOutcome::Miss);
}
