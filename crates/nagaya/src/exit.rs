//! The process exit codes (spec 10.4).

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
    /// No path reaches this code in milestone 10. The profile change arrives with the HTTP coordinator.
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
