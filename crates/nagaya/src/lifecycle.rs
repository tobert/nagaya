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
use nagaya_telemetry::instruments;

/// The backstop wait for every engine to open. It equals `TimeoutStartSec` in the unit (spec 7.4, 10.1).
///
/// A group beats while its engine opens, so the watchdog cannot end a hung open. This wait ends it.
pub const DEFAULT_OPEN_TIMEOUT: Duration = Duration::from_secs(300);

/// How often the start tells systemd that the engines still open. It equals the heartbeat (spec 7.4, 10.1).
pub const DEFAULT_OPEN_TICK: Duration = crate::process::DEFAULT_HEARTBEAT;

/// How far each tick moves the start deadline of systemd. It equals the watchdog timeout (spec 10.1).
pub const DEFAULT_START_EXTEND: Duration = crate::process::DEFAULT_WATCHDOG_TIMEOUT;

/// The timing of the start. The tests use short values.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    timeout: Duration,
    tick: Duration,
    extend: Duration,
}

impl Options {
    /// Options with the default tick and extension, and a wait of `timeout` for the opens.
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            tick: DEFAULT_OPEN_TICK,
            extend: DEFAULT_START_EXTEND,
        }
    }

    /// Set how often the start tells systemd that the engines still open.
    pub fn tick(mut self, tick: Duration) -> Self {
        self.tick = tick;
        self
    }

    /// Set how far each tick moves the start deadline of systemd.
    pub fn extend(mut self, extend: Duration) -> Self {
        self.extend = extend;
        self
    }
}

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
    Timeout {
        /// The engines that had not opened.
        waiting: Vec<String>,
    },
    /// The group channels closed.
    Disconnected,
}

impl LifecycleError {
    /// The exit code of a failure to start (spec 10.4). A profile that does not fit, or an engine that refuses to
    /// open, is exit 3. A timeout, a closed channel and a worker panic are crashes: exit 1.
    pub fn exit_code(&self) -> ExitCode {
        match self {
            LifecycleError::Event(event) => event.exit_code().unwrap_or(ExitCode::RefusedStart),
            LifecycleError::Start(_) => ExitCode::RefusedStart,
            LifecycleError::Timeout { .. } | LifecycleError::Disconnected => ExitCode::Crash,
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
            LifecycleError::Timeout { waiting } => {
                write!(formatter, "engine {} did not open in time", waiting.join(", "))
            }
            LifecycleError::Disconnected => write!(formatter, "the group threads stopped"),
        }
    }
}

impl std::error::Error for LifecycleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LifecycleError::Start(error) => Some(error),
            LifecycleError::Event(_) | LifecycleError::Timeout { .. } | LifecycleError::Disconnected => None,
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
///
/// The start makes the span `nagaya.start`. It records the state of each engine in `nagaya.engine.state` (spec 9.4).
pub fn start(
    profile: PlacementProfile,
    registry: Registry,
    inventory: Inventory,
    store_dir: &Path,
    notifier: Option<&Notifier>,
    timeout: Duration,
) -> Result<Ready, LifecycleError> {
    start_with(profile, registry, inventory, store_dir, notifier, Options::new(timeout))
}

/// Start as `start` does, with the timing in `options`.
///
/// While the engines open, each tick sends `EXTEND_TIMEOUT_USEC=` and the `loading` status to systemd (spec 10.1).
pub fn start_with(
    profile: PlacementProfile,
    registry: Registry,
    inventory: Inventory,
    store_dir: &Path,
    notifier: Option<&Notifier>,
    options: Options,
) -> Result<Ready, LifecycleError> {
    let span = tracing::info_span!(
        "nagaya.start",
        engines = profile.engines.len(),
        outcome = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    let _entered = span.enter();
    let result = start_engines(profile, registry, inventory, store_dir, notifier, options);
    match &result {
        Ok(_) => span.record("outcome", "ready"),
        Err(error) => {
            tracing::error!("{error}");
            span.record("outcome", "failed");
            span.record("otel.status_code", "ERROR")
        }
    };
    result
}

/// Record that each engine in `names` entered `state`.
fn record_states<'a>(names: impl IntoIterator<Item = &'a String>, state: EngineState) {
    let instruments = instruments();
    for name in names {
        instruments.engine_state(name, state);
    }
}

fn start_engines(
    profile: PlacementProfile,
    registry: Registry,
    inventory: Inventory,
    store_dir: &Path,
    notifier: Option<&Notifier>,
    options: Options,
) -> Result<Ready, LifecycleError> {
    let names: Vec<String> = profile.engines.iter().map(|engine| engine.name.clone()).collect();
    let mut process = {
        record_states(&names, EngineState::Loading);
        if !names.is_empty() {
            tracing::info!(engines = names.join(", "), "opening engines");
        }
        if let Some(notifier) = notifier
            && let Err(error) = notifier.states(&names, EngineState::Loading)
        {
            tracing::warn!("cannot send STATUS= to systemd: {error}");
        }
        Start::new(profile, registry, inventory)
            .store(store_dir)
            .start()
            .map_err(LifecycleError::Start)?
    };
    let extend = |notifier: &Notifier| {
        if let Err(error) = notifier.extend_start(&names, options.extend) {
            tracing::warn!("cannot send EXTEND_TIMEOUT_USEC to systemd: {error}");
        }
    };
    match process.wait_opens_with(options.timeout, options.tick, || {
        if let Some(notifier) = notifier {
            extend(notifier);
        }
    }) {
        Ok(opened) => {
            let mut names: Vec<String> = opened.iter().map(|engine| engine.name.clone()).collect();
            names.sort();
            record_states(&names, EngineState::Ready);
            if let Some(notifier) = notifier
                && let Err(error) = notifier.ready(&names)
            {
                tracing::warn!("cannot send READY=1 to systemd: {error}");
            }
            Ok(Ready { process, opened })
        }
        Err(WaitError::Event(event)) => {
            if let Event::OpenFailed { engine, .. } = &event {
                instruments().engine_state(engine, EngineState::Fatal);
            }
            if let Some(notifier) = notifier
                && let Err(error) = notifier.notify(&status(&event))
            {
                tracing::warn!("cannot send STATUS= to systemd: {error}");
            }
            Err(LifecycleError::Event(event))
        }
        Err(WaitError::Timeout { waiting }) => {
            record_states(&waiting, EngineState::Fatal);
            if let Some(notifier) = notifier
                && let Err(error) =
                    notifier.notify(&format!("STATUS=engine {} did not open in time", waiting.join(", ")))
            {
                tracing::warn!("cannot send STATUS= to systemd: {error}");
            }
            Err(LifecycleError::Timeout { waiting })
        }
        Err(WaitError::Disconnected) => Err(LifecycleError::Disconnected),
    }
}

/// Wait for a stop signal or a failure event. Gives the exit code of the event, or `Clean` for a signal.
pub fn wait(process: &Process, signals: &SignalFlag, notifier: Option<&Notifier>, poll: Duration) -> ExitCode {
    wait_with(process, signals, notifier, poll, &mut || None)
}

/// Like `wait`, and call `tick` in each turn of the loop. A `tick` that gives a code ends the wait with that code.
/// A signal that arrived during the tick turns a planned restart into a clean stop.
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
            // A signal that arrived during the tick wins: an operator stop is not a planned restart. A profile that
            // the tick persisted stays persisted. A crash stays a crash (spec 8.5).
            if code == ExitCode::PlannedRestart && signals.count() > 0 {
                return ExitCode::Clean;
            }
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
/// The stop makes the span `nagaya.stop` and records each state in `nagaya.engine.state` (spec 9.4).
pub fn stop(ready: &mut Ready, notifier: Option<&Notifier>, budget: Duration, signals: &SignalFlag) -> StopReport {
    let names: Vec<String> = ready.opened.iter().map(|engine| engine.name.clone()).collect();
    let span = tracing::info_span!(
        "nagaya.stop",
        engines = names.len(),
        outcome = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    let _entered = span.enter();
    record_states(&names, EngineState::Draining);
    if let Some(notifier) = notifier
        && let Err(error) = notifier.states(&names, EngineState::Draining)
    {
        tracing::warn!("cannot send STATUS= to systemd: {error}");
    }
    let report = ready.process.stop(budget, signals);
    let (parked, unparked) = report.split_parked(&names);
    record_states(&parked, EngineState::Unloaded);
    record_states(&unparked, EngineState::Fatal);
    if let Some(notifier) = notifier {
        for (engines, state) in [(parked, EngineState::Unloaded), (unparked, EngineState::Fatal)] {
            if let Err(error) = notifier.states(&engines, state) {
                tracing::warn!("cannot send STATUS= to systemd: {error}");
            }
        }
    }
    if report.exit_code() == ExitCode::Clean {
        span.record("outcome", "clean");
    } else {
        span.record("outcome", "failed");
        span.record("otel.status_code", "ERROR");
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
