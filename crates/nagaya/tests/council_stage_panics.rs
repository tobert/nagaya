//! A stage thread of a group with two stages that panics on purpose (spec 8.5). These tests live in their own binary.
//! A planned panic makes the other threads of its process panic too. Then the panic hook of `feed_support` stops for
//! the binary, and in a shared binary it would miss a panic with no plan in another test.

mod feed_support;

use std::sync::atomic::Ordering;
use std::time::Duration;

use feed_support::{FeedFactory, FeedFixture};
use gogi_proto::context::Context;
use nagaya::exit::ExitCode;
use nagaya::signal::SignalFlag;
use serde_json::json;

fn context() -> Context {
    serde_json::from_value(json!({"thinking": false, "tools": [], "messages": [
        {"role": "system", "content": "You are a careful agent. Read the files first."},
        {"role": "user", "content": "List the crates of the workspace, with one line each.", "snap": true},
    ]}))
    .unwrap()
}

/// The exit code that the lifecycle wait gives for `process`. A timer raises a signal at `bound`, so a wait that
/// reads no event ends with `Clean` and the test fails. It does not hang.
fn code_within(process: &nagaya::process::Process, bound: Duration) -> ExitCode {
    let signals = std::sync::Arc::new(SignalFlag::new());
    let timer = std::sync::Arc::clone(&signals);
    std::thread::spawn(move || {
        std::thread::sleep(bound);
        timer.raise();
    });
    nagaya::lifecycle::wait(process, &signals, None, Duration::from_millis(20))
}

#[test]
fn a_stage_thread_that_panics_stops_the_process_with_exit_one() {
    let factory = FeedFactory::new(2);
    let fixture = FeedFixture::with(factory.clone(), Duration::from_secs(10));
    factory.panic_stage.store(1, Ordering::SeqCst);
    let service = std::sync::Arc::new(fixture.service);
    std::thread::spawn(move || {
        let _ = service.build(&context(), false, Default::default());
    });
    assert_eq!(code_within(&fixture.process, Duration::from_secs(3)), ExitCode::Crash);
}

#[test]
fn a_stop_after_a_stage_panic_ends_with_exit_one_within_the_budget() {
    let factory = FeedFactory::new(2);
    let mut fixture = FeedFixture::with(factory.clone(), Duration::from_secs(10));
    factory.panic_stage.store(0, Ordering::SeqCst);
    let service = std::sync::Arc::new(fixture.service);
    let builder = std::sync::Arc::clone(&service);
    std::thread::spawn(move || {
        let _ = builder.build(&context(), false, Default::default());
    });
    assert_eq!(code_within(&fixture.process, Duration::from_secs(3)), ExitCode::Crash);
    let started = std::time::Instant::now();
    let report = fixture.process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert!(started.elapsed() < Duration::from_secs(6), "the stop returned late");
    assert_eq!(report.exit_code(), ExitCode::Crash, "{report:?}");
}
