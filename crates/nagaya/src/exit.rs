//! The process exit codes (spec 10.4).

use std::any::Any;

/// The text of a panic payload.
pub fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a worker thread panicked".to_owned()
    }
}

/// Why the process stopped. The number is the exit code of spec 10.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitCode {
    /// A clean stop (spec 10.4 row 0). systemd does not restart.
    Clean,
    /// A crash (spec 8.5, 10.4 row 1). systemd restarts.
    Crash,
    /// A refusal to start (spec 10.4 row 3). systemd does not restart at once.
    RefusedStart,
    /// A planned restart for a profile change (spec 3.3, 10.4 row 4).
    ///
    /// A desired profile from a peer that changes the engines gives this code (spec 2.4 note).
    PlannedRestart,
}

impl ExitCode {
    /// The exit code that this process gives to the operating system.
    pub fn code(self) -> i32 {
        match self {
            ExitCode::Clean => 0,
            ExitCode::Crash => 1,
            ExitCode::RefusedStart => 3,
            ExitCode::PlannedRestart => 4,
        }
    }
}

/// The exit code after the stop path ran (spec 10.3, 10.4).
///
/// `wait_code` is why the wait ended, and `stop_code` is the result of the stop. A planned restart stays a planned
/// restart when the stop is clean. `failed` is true when a route recorded an internal error. That error can come
/// after the last wait tick, so the stop path checks again. An internal error is a crash (spec 8.5, 11.7).
pub fn after_stop(wait_code: ExitCode, stop_code: ExitCode, failed: bool) -> ExitCode {
    if stop_code == ExitCode::Clean && failed {
        ExitCode::Crash
    } else if wait_code == ExitCode::PlannedRestart && stop_code == ExitCode::Clean {
        ExitCode::PlannedRestart
    } else {
        stop_code
    }
}

/// End the process with `code`. Every exit of the binary goes through here, for every code and every build.
///
/// mk's HIP runtime stops the program in its static destructors ("pure virtual method called", spec 3.3). Then a
/// clean stop exits 134 and writes a core dump. `std::process::exit` runs those destructors and the `atexit`
/// handlers, so this function calls `_exit(2)`, which runs neither. The fake build takes the same path, so the tests
/// run the path of the GPU build. Nothing in the process needs a handler: the caller drops the telemetry first, and
/// this function flushes the Rust and C output streams.
pub fn leave(code: ExitCode) -> ! {
    use std::io::Write;

    // A failed flush loses only output. The exit code is the result that matters, so the errors are ignored.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `fflush(NULL)` flushes every C output stream, and `_exit` ends the process. Neither takes Rust state.
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::_exit(code.code())
    }
}
