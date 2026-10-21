//! A thread that the process starts records on the telemetry of the thread that started it (spec 9.5).

use nagaya_telemetry::testing::Capture;
use nagaya_telemetry::{Inherit, QuantumLabels, instruments};

fn record_on_a_new_thread(inherit: Option<Inherit>) {
    std::thread::spawn(move || {
        let _entered = inherit.map(Inherit::enter);
        let labels = QuantumLabels {
            engine: "qwen",
            device_group: 0,
            kind: "copy",
            class: "batch",
        };
        instruments().quantum_run(&labels, 3.0);
        tracing::info_span!("nagaya.quantum").in_scope(|| {});
    })
    .join()
    .unwrap();
}

#[test]
fn a_thread_that_enters_the_inherited_telemetry_records_into_the_capture() {
    let capture = Capture::start(Vec::new());
    record_on_a_new_thread(Some(Inherit::capture()));

    let run = capture.metric("nagaya.quantum.run.duration");
    assert_eq!(run.len(), 1, "{run:?}");
    assert_eq!((run[0].count, run[0].value), (1, 3.0));
    assert_eq!(capture.spans().len(), 1);
}

#[test]
fn a_thread_that_does_not_enter_records_nowhere() {
    let capture = Capture::start(Vec::new());
    record_on_a_new_thread(None);

    assert!(capture.metric("nagaya.quantum.run.duration").is_empty());
    assert!(capture.spans().is_empty());
}

#[test]
fn the_guard_restores_the_telemetry_of_the_thread() {
    let outer = Capture::start(Vec::new());
    {
        let inner = Capture::start(Vec::new());
        let inherit = Inherit::capture();
        drop(inner);
        let _entered = inherit.enter();
    }
    // The guards drop in reverse order, so this thread records on `outer` again.
    let labels = QuantumLabels {
        engine: "qwen",
        device_group: 0,
        kind: "copy",
        class: "batch",
    };
    instruments().quantum_run(&labels, 1.0);
    assert_eq!(outer.metric("nagaya.quantum.run.duration").len(), 1);
}
