//! The one exit path of the process (spec 10.4, 3.3).
//!
//! The HIP runtime of the Qwen3.8 engine stops the program in its static destructors ("pure virtual method called"). `exit(3)` runs those
//! destructors and the `atexit` handlers, so the process must leave without them. The test runs itself again as a
//! child. The child registers an `atexit` handler that aborts, as that runtime does, and then leaves.

use std::process::Command;

use nagaya::exit::{ExitCode, leave};

/// The variable that makes the child test leave.
const CHILD: &str = "NAGAYA_EXIT_PATH_CHILD";

extern "C" fn abort_at_exit() {
    std::process::abort();
}

/// The child: it registers the handler, writes a line with no newline, and leaves with the code that `CHILD` names.
#[test]
fn child_leaves_with_the_code() {
    let Ok(code) = std::env::var(CHILD) else {
        return;
    };
    let code = match code.as_str() {
        "0" => ExitCode::Clean,
        "3" => ExitCode::RefusedStart,
        other => panic!("the child takes 0 or 3, not {other}"),
    };
    // SAFETY: the handler is a plain function that lives for the process.
    assert_eq!(unsafe { libc::atexit(abort_at_exit) }, 0, "cannot register the handler");
    print!("the last words of the child");
    leave(code);
}

fn run_child(code: &str) -> std::process::Output {
    Command::new(std::env::current_exe().unwrap())
        .args([
            "child_leaves_with_the_code",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, code)
        .output()
        .unwrap()
}

#[test]
fn the_process_leaves_with_its_code_and_runs_no_atexit_handler() {
    for (code, expected) in [("0", 0), ("3", 3)] {
        let output = run_child(code);
        assert_eq!(
            output.status.code(),
            Some(expected),
            "the child did not exit {expected}: {:?}; an atexit handler ran or the exit failed. stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("the last words of the child"),
            "the exit lost the buffered output: {stdout}"
        );
    }
}
