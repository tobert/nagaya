use nagaya::admission::OpenErrorKind;
use nagaya::exit::ExitCode;
use nagaya::process::Event;

#[test]
fn the_four_codes_have_the_spec_numbers() {
    assert_eq!(ExitCode::Clean.code(), 0);
    assert_eq!(ExitCode::Crash.code(), 1);
    assert_eq!(ExitCode::RefusedStart.code(), 3);
    assert_eq!(ExitCode::PlannedRestart.code(), 4);
}

#[test]
fn an_open_report_stops_nothing() {
    let event = Event::Opened {
        group: 0,
        engines: Vec::new(),
    };
    assert_eq!(event.exit_code(), None);
}

#[test]
fn a_stall_is_a_crash() {
    let event = Event::Stalled {
        group: 0,
        devices: vec![0],
    };
    assert_eq!(event.exit_code(), Some(ExitCode::Crash));
}

#[test]
fn an_open_mismatch_is_a_refusal_to_start() {
    let event = Event::OpenFailed {
        group: 0,
        engine: "a".to_owned(),
        kind: OpenErrorKind::Identity,
        message: "no weights".to_owned(),
    };
    assert_eq!(event.exit_code(), Some(ExitCode::RefusedStart));
}

#[test]
fn another_open_failure_is_a_crash() {
    let event = Event::OpenFailed {
        group: 0,
        engine: "a".to_owned(),
        kind: OpenErrorKind::Other,
        message: "no weights".to_owned(),
    };
    assert_eq!(event.exit_code(), Some(ExitCode::Crash));
}

#[test]
fn a_thread_panic_is_a_crash() {
    let event = Event::Panicked {
        thread: "group",
        message: "boom".to_owned(),
    };
    assert_eq!(event.exit_code(), Some(ExitCode::Crash));
}

#[test]
fn an_internal_error_after_the_last_tick_is_a_crash_at_every_clean_stop() {
    use nagaya::exit::after_stop;
    assert_eq!(after_stop(ExitCode::Clean, ExitCode::Clean, true), ExitCode::Crash);
    assert_eq!(
        after_stop(ExitCode::PlannedRestart, ExitCode::Clean, true),
        ExitCode::Crash
    );
    assert_eq!(after_stop(ExitCode::Clean, ExitCode::Clean, false), ExitCode::Clean);
    assert_eq!(
        after_stop(ExitCode::PlannedRestart, ExitCode::Clean, false),
        ExitCode::PlannedRestart
    );
    // A failed stop is a crash, and a failure does not hide it.
    assert_eq!(after_stop(ExitCode::Clean, ExitCode::Crash, false), ExitCode::Crash);
    assert_eq!(
        after_stop(ExitCode::PlannedRestart, ExitCode::Crash, true),
        ExitCode::Crash
    );
}

/// Each event that stops the process has a reason for the log; an event that stops nothing has none (spec 8.5, 9.1).
/// On 2026-11-01 a stall on gfx1151host stopped the process with only systemd's status line, and the log had no cause.
#[test]
fn a_stopping_event_has_a_reason_for_the_log() {
    let stall = Event::Stalled {
        group: 0,
        devices: vec![0],
    };
    let reason = nagaya::lifecycle::stop_reason(&stall).expect("a stall stops the process");
    assert!(
        reason.contains("group 0 stalled") && !reason.starts_with("STATUS="),
        "{reason}"
    );
    let panic = Event::Panicked {
        thread: "group",
        message: "boom".to_owned(),
    };
    assert!(nagaya::lifecycle::stop_reason(&panic).unwrap().contains("boom"));
    let opened = Event::Opened {
        group: 0,
        engines: Vec::new(),
    };
    assert_eq!(nagaya::lifecycle::stop_reason(&opened), None);
}
