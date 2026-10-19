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
