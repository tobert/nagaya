//! The start and wait phases of the process (spec 10.1, 10.3).

use std::io;
use std::path::Path;
use std::time::Duration;

use crate::admission::Registry;
use crate::exit::ExitCode;
use crate::inventory::Inventory;
use crate::notify::{EngineState, Notifier};
use crate::process::{Event, OpenedEngine, Process, Start, StartError, StopReport, WaitError};
use crate::profile::{PlacementProfile, ProfileError};
use crate::signal::SignalFlag;

/// The backstop wait for every engine to open. It is two watchdog timeouts (spec 7.4).
pub const DEFAULT_OPEN_TIMEOUT: Duration = Duration::from_secs(60);

/// How often the wait loop reads the signal counter (spec 10.3 note).
pub const WAIT_POLL: Duration = Duration::from_millis(50);

/// Why the process did not become ready.
#[derive(Debug)]
pub enum LifecycleError {
    /// The profile cannot be admitted or a thread cannot start.
    Start(StartError),
    /// A group reported a failure before every engine opened.
    Event(Event),
    /// The engines did not open in time.
    Timeout,
    /// The group channels closed.
    Disconnected,
}

impl LifecycleError {
    /// The exit code of a failure to start. A worker panic is a crash (spec 8.5); the rest is exit 3.
    pub fn exit_code(&self) -> ExitCode {
        match self {
            LifecycleError::Event(event) => event.exit_code().unwrap_or(ExitCode::RefusedStart),
            LifecycleError::Start(_) | LifecycleError::Timeout | LifecycleError::Disconnected => ExitCode::RefusedStart,
        }
    }
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LifecycleError::Start(error) => write!(formatter, "the process cannot start: {error}"),
            LifecycleError::Event(Event::OpenFailed { engine, message, .. }) => {
                write!(formatter, "engine `{engine}` did not open: {message}")
            }
            LifecycleError::Event(Event::Panicked { thread, message }) => {
                write!(formatter, "the {thread} thread panicked: {message}")
            }
            LifecycleError::Event(Event::Stalled { group, .. }) => {
                write!(formatter, "group {group} stalled before the process was ready")
            }
            LifecycleError::Event(Event::Opened { group, .. }) => {
                write!(formatter, "group {group} sent an open report during the start")
            }
            LifecycleError::Timeout => write!(formatter, "the engines did not open in time"),
            LifecycleError::Disconnected => write!(formatter, "the group threads stopped"),
        }
    }
}

impl std::error::Error for LifecycleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LifecycleError::Start(error) => Some(error),
            LifecycleError::Event(_) | LifecycleError::Timeout | LifecycleError::Disconnected => None,
        }
    }
}

/// The process after every engine opened.
#[derive(Debug)]
pub struct Ready {
    /// The started threads.
    pub process: Process,
    /// The engines that opened, in arrival order. Two group threads race.
    pub opened: Vec<OpenedEngine>,
}

/// The profile to start. A missing file gives an empty profile (spec 10.1 note).
pub fn active_profile(store_dir: &Path) -> Result<PlacementProfile, ProfileError> {
    match PlacementProfile::read_active(store_dir) {
        Ok(profile) => Ok(profile),
        Err(ProfileError::Read { source, .. }) if source.kind() == io::ErrorKind::NotFound => Ok(PlacementProfile {
            engines: Vec::new(),
            budgets: Vec::new(),
            store: None,
        }),
        Err(error) => Err(error),
    }
}

/// Start the threads, wait for every engine to open, then send `READY=1` (spec 10.1).
pub fn start(
    profile: PlacementProfile,
    registry: Registry,
    inventory: Inventory,
    store_dir: &Path,
    notifier: Option<&Notifier>,
    timeout: Duration,
) -> Result<Ready, LifecycleError> {
    let mut process = {
        let names: Vec<String> = profile.engines.iter().map(|engine| engine.name.clone()).collect();
        if let Some(notifier) = notifier
            && let Err(error) = notifier.states(&names, EngineState::Loading)
        {
            eprintln!("cannot send STATUS= to systemd: {error}");
        }
        Start::new(profile, registry, inventory)
            .store(store_dir)
            .start()
            .map_err(LifecycleError::Start)?
    };
    match process.wait_opens(timeout) {
        Ok(opened) => {
            let mut names: Vec<String> = opened.iter().map(|engine| engine.name.clone()).collect();
            names.sort();
            if let Some(notifier) = notifier
                && let Err(error) = notifier.ready(&names)
            {
                eprintln!("cannot send READY=1 to systemd: {error}");
            }
            Ok(Ready { process, opened })
        }
        Err(WaitError::Event(event)) => {
            if let Some(notifier) = notifier
                && let Err(error) = notifier.notify(&status(&event))
            {
                eprintln!("cannot send STATUS= to systemd: {error}");
            }
            Err(LifecycleError::Event(event))
        }
        Err(WaitError::Timeout) => Err(LifecycleError::Timeout),
        Err(WaitError::Disconnected) => Err(LifecycleError::Disconnected),
    }
}

/// Wait for a stop signal or a failure event. Gives the exit code of the event, or `Clean` for a signal.
pub fn wait(process: &Process, signals: &SignalFlag, notifier: Option<&Notifier>, poll: Duration) -> ExitCode {
    wait_with(process, signals, notifier, poll, &mut || None)
}

/// Like `wait`, and call `tick` in each turn of the loop. A `tick` that gives a code ends the wait with that code.
pub fn wait_with(
    process: &Process,
    signals: &SignalFlag,
    notifier: Option<&Notifier>,
    poll: Duration,
    tick: &mut dyn FnMut() -> Option<ExitCode>,
) -> ExitCode {
    loop {
        if signals.count() > 0 {
            return ExitCode::Clean;
        }
        if let Some(code) = tick() {
            return code;
        }
        let Some(event) = process.recv(poll) else {
            if process.finished() {
                // Every worker ended without an event. That is a crash, not an idle process (spec 8.5).
                return ExitCode::Crash;
            }
            continue;
        };
        let Some(code) = event.exit_code() else {
            continue;
        };
        if let Some(notifier) = notifier {
            let _ = notifier.notify(&status(&event));
        }
        return code;
    }
}

/// Stop the process and report each engine's state: draining, then unloaded or fatal (spec 10.1, 10.3).
///
/// An engine that parked every state is `unloaded`. An engine that did not park is `fatal`.
pub fn stop(ready: &mut Ready, notifier: Option<&Notifier>, budget: Duration, signals: &SignalFlag) -> StopReport {
    let names: Vec<String> = ready.opened.iter().map(|engine| engine.name.clone()).collect();
    if let Some(notifier) = notifier
        && let Err(error) = notifier.states(&names, EngineState::Draining)
    {
        eprintln!("cannot send STATUS= to systemd: {error}");
    }
    let report = ready.process.stop(budget, signals);
    if let Some(notifier) = notifier {
        let (parked, unparked) = report.split_parked(&names);
        for (engines, state) in [(parked, EngineState::Unloaded), (unparked, EngineState::Fatal)] {
            if let Err(error) = notifier.states(&engines, state) {
                eprintln!("cannot send STATUS= to systemd: {error}");
            }
        }
    }
    report
}

/// The `STATUS=` text of an event that stops the process.
fn status(event: &Event) -> String {
    match event {
        Event::Stalled { group, devices } => {
            format!("STATUS=group {group} stalled on devices {devices:?}")
        }
        Event::Panicked { thread, message } => format!("STATUS=the {thread} thread panicked: {message}"),
        Event::OpenFailed { engine, message, .. } => format!("STATUS=engine {engine} fatal: {message}"),
        Event::Opened { group, .. } => format!("STATUS=group {group} opened"),
    }
}
