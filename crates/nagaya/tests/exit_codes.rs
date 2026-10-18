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
fn an_open_failure_is_a_refusal_to_start() {
    let event = Event::OpenFailed {
        group: 0,
        engine: "a".to_owned(),
        message: "no weights".to_owned(),
    };
    assert_eq!(event.exit_code(), Some(ExitCode::RefusedStart));
}

#[test]
fn a_thread_panic_is_a_crash() {
    let event = Event::Panicked {
        thread: "group",
        message: "boom".to_owned(),
    };
    assert_eq!(event.exit_code(), Some(ExitCode::Crash));
}
