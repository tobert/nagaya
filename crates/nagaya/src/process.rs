//! The process threads: the ledger, the watchdog, and one thread per device group (spec 3.2).

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use nagaya_abi::{DeviceId, Engine, Limits};
use nagaya_ledger::Ledger;

use crate::admission::{AdmitError, Admitted, Factory, Registry, admit};
use crate::exit::{ExitCode, panic_text};
use crate::inventory::Inventory;
use crate::park::{self, StoreBudget, StoreMessage};
use crate::placement::{self, DeviceGroup};
use crate::profile::{EnginePlacement, PlacementProfile};
use crate::signal::SignalFlag;

/// The default watchdog timeout. Three times the longest declared quantum bound (mk: 10 s).
pub const DEFAULT_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(30);

/// The default heartbeat. A group beats six times in the default timeout.
pub const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(5);

/// How often the stop path reads the signal counter (spec 10.3).
pub const STOP_POLL: Duration = Duration::from_millis(10);

/// What one group parked: the counts, and each part that it could not read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParkReport {
    /// The engines that the group parked.
    pub engines: usize,
    /// The states that the group parked.
    pub states: usize,
    /// The parts that the group read.
    pub parts: usize,
    /// The bytes that the group read.
    pub bytes: u64,
    /// The store entries that the group wrote.
    pub entries: usize,
    /// The row parts that the group read with no row count, because the engine has no `Slots` capability.
    pub unmeasured_row_parts: usize,
    /// The parts that the group could not read, one message for each.
    pub failures: Vec<String>,
}

impl ParkReport {
    /// Add the counts of another report.
    pub fn merge(&mut self, other: ParkReport) {
        self.engines += other.engines;
        self.states += other.states;
        self.parts += other.parts;
        self.bytes += other.bytes;
        self.entries += other.entries;
        self.unmeasured_row_parts += other.unmeasured_row_parts;
        self.failures.extend(other.failures);
    }
}

/// What the stop did (spec 10.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StopReport {
    /// The park of every group.
    pub parked: ParkReport,
    /// True when every group answered inside the budget.
    pub complete: bool,
    /// True when a second signal stopped the park at once.
    pub interrupted: bool,
}

impl StopReport {
    /// The exit code of the stop. A failed write or an unfinished park is a crash (spec 8.5).
    pub fn exit_code(&self) -> ExitCode {
        if !self.parked.failures.is_empty() {
            return ExitCode::Crash;
        }
        if self.interrupted || self.complete {
            ExitCode::Clean
        } else {
            ExitCode::Crash
        }
    }

    /// The line that the process writes to stderr when the stop ends (spec 10.3).
    pub fn log_line(&self) -> String {
        if self.interrupted {
            return "the stop was interrupted before the park finished".to_owned();
        }
        let failed = self.parked.failures.len();
        if self.parked.states == 0 && self.parked.entries == 0 && failed == 0 {
            return "the stop parked nothing: no state needed a park".to_owned();
        }
        if !self.complete {
            return format!(
                "the park did not finish; it wrote {} states and {} parts",
                self.parked.states, self.parked.parts
            );
        }
        if failed == 0 {
            return format!(
                "the stop wrote {} states and {} parts ({} bytes) to the store",
                self.parked.states, self.parked.parts, self.parked.bytes
            );
        }
        format!(
            "the stop wrote {} states and {} parts, then failed {failed} part(s)",
            self.parked.states, self.parked.parts
        )
    }
}

/// Run one worker body. A panic becomes an event for the main thread (spec 8.5).
fn guard(thread: &'static str, events: &Sender<Event>, body: impl FnOnce()) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(body)) {
        let message = panic_text(payload.as_ref());
        let _ = events.send(Event::Panicked { thread, message });
    }
}

/// A ledger that one thread owns. A write from another thread panics (spec 3.2, 8.5).
pub struct OwnedLedger {
    ledger: Ledger,
    owner: ThreadId,
}

impl OwnedLedger {
    /// Make a ledger for the current thread.
    pub fn new() -> Self {
        Self {
            ledger: Ledger::new(),
            owner: thread::current().id(),
        }
    }

    /// Admit a profile. Only the owner thread may call this.
    pub fn admit(
        &mut self,
        profile: &PlacementProfile,
        registry: &Registry,
        inventory: &Inventory,
    ) -> Result<Admitted, AdmitError> {
        assert_eq!(
            thread::current().id(),
            self.owner,
            "the ledger is written off its thread"
        );
        admit(profile, registry, inventory, &mut self.ledger)
    }

    /// The ledger's state.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }
}

impl Default for OwnedLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// An engine that opened on its group thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenedEngine {
    /// The engine name.
    pub name: String,
    /// The thread that opened it.
    pub thread: ThreadId,
}

/// A report from a group thread or the watchdog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A group opened every engine.
    Opened {
        /// The group index.
        group: usize,
        /// The engines, in profile order.
        engines: Vec<OpenedEngine>,
    },
    /// A group could not open an engine.
    OpenFailed {
        /// The group index.
        group: usize,
        /// The engine name.
        engine: String,
        /// Why the open failed.
        message: String,
    },
    /// A group stopped its heartbeat.
    Stalled {
        /// The group index.
        group: usize,
        /// The devices of the group.
        devices: Vec<DeviceId>,
    },
    /// A worker thread panicked. The process stops (spec 8.5).
    Panicked {
        /// The worker: `group`, `ledger` or `watchdog`.
        thread: &'static str,
        /// The panic message.
        message: String,
    },
}

impl Event {
    /// The process exit code, when this event stops the process.
    pub fn exit_code(&self) -> Option<ExitCode> {
        match self {
            Event::Stalled { .. } => Some(ExitCode::Crash),
            Event::OpenFailed { .. } => Some(ExitCode::RefusedStart),
            Event::Panicked { .. } => Some(ExitCode::Crash),
            Event::Opened { .. } => None,
        }
    }
}

/// Why the opens did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitError {
    /// A group reported a failure.
    Event(Event),
    /// The groups did not open in time.
    Timeout,
    /// The group channels closed.
    Disconnected,
}

/// Why the process did not start.
#[derive(Debug)]
pub enum StartError {
    /// The profile cannot be admitted. The ledger keeps its state.
    Admit(AdmitError),
    /// The machine's store did not open.
    Store {
        /// The store directory.
        path: PathBuf,
        /// Why the store did not open.
        message: String,
    },
}

impl std::fmt::Display for StartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::Admit(error) => write!(formatter, "the process cannot start: {error}"),
            StartError::Store { path, message } => {
                write!(formatter, "cannot open the store `{}`: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for StartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StartError::Admit(error) => Some(error),
            StartError::Store { .. } => None,
        }
    }
}

/// A request to start the process threads.
pub struct Start {
    profile: PlacementProfile,
    registry: Registry,
    inventory: Inventory,
    heartbeat: Duration,
    timeout: Duration,
    store_dir: Option<PathBuf>,
    store_budget: Option<StoreBudget>,
}

impl Start {
    /// Make a start request with the default timeouts.
    pub fn new(profile: PlacementProfile, registry: Registry, inventory: Inventory) -> Self {
        Self {
            profile,
            registry,
            inventory,
            heartbeat: DEFAULT_HEARTBEAT,
            timeout: DEFAULT_WATCHDOG_TIMEOUT,
            store_dir: None,
            store_budget: None,
        }
    }

    /// Set the store directory. The process opens the machine's store there (spec 6.1).
    pub fn store(mut self, dir: impl Into<PathBuf>) -> Self {
        self.store_dir = Some(dir.into());
        self
    }

    /// Set the store budgets. Without this, the profile's unassigned host bytes set them (spec 4.1 note).
    pub fn store_budget(mut self, budget: StoreBudget) -> Self {
        self.store_budget = Some(budget);
        self
    }

    /// Set the heartbeat interval. The tests use a short value.
    pub fn heartbeat(mut self, heartbeat: Duration) -> Self {
        self.heartbeat = heartbeat;
        self
    }

    /// Set the watchdog timeout. The tests use a short value.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Start the ledger thread, the watchdog thread, and one thread per device group, in that order.
    pub fn start(self) -> Result<Process, StartError> {
        let Start {
            profile,
            registry,
            inventory,
            heartbeat,
            timeout,
            store_dir,
            store_budget,
        } = self;
        assert!(
            heartbeat < timeout,
            "the heartbeat must be shorter than the watchdog timeout"
        );
        let registry = Arc::new(registry);
        let inventory = Arc::new(inventory);
        let (events_tx, events) = mpsc::channel();
        let (watch_tx, watch_rx) = mpsc::channel();
        let (ledger_tx, ledger_rx) = mpsc::channel();

        // The ledger thread. It is the only writer of the ledger (spec 3.2).
        let ledger_registry = Arc::clone(&registry);
        let ledger_inventory = Arc::clone(&inventory);
        let ledger_events = events_tx.clone();
        let ledger_thread = thread::spawn(move || {
            guard("ledger", &ledger_events, || {
                let mut ledger = OwnedLedger::new();
                while let Ok(message) = ledger_rx.recv() {
                    match message {
                        LedgerMessage::Admit(profile, reply) => {
                            let result = ledger.admit(&profile, &ledger_registry, &ledger_inventory);
                            let _ = reply.send(result);
                        }
                        LedgerMessage::Stop => break,
                    }
                }
            });
        });

        // The watchdog thread. It watches the group threads from outside (spec 3.2, 7.4).
        let watchdog_events = events_tx.clone();
        let watchdog_body = watchdog_events.clone();
        let watchdog_thread = thread::spawn(move || {
            guard("watchdog", &watchdog_events, || {
                watchdog(watch_rx, watchdog_body, timeout);
            });
        });

        // Admission goes through the ledger thread, before any engine opens (spec 2.4).
        let (reply_tx, reply_rx) = mpsc::channel();
        ledger_tx
            .send(LedgerMessage::Admit(profile.clone(), reply_tx))
            .expect("the ledger thread stopped");
        let admitted = match reply_rx.recv() {
            Ok(Ok(admitted)) => admitted,
            Ok(Err(error)) => {
                let _ = watch_tx.send(WatchMessage::Stop);
                let _ = ledger_tx.send(LedgerMessage::Stop);
                let _ = ledger_thread.join();
                let _ = watchdog_thread.join();
                return Err(StartError::Admit(error));
            }
            Err(_) => panic!("the ledger thread stopped"),
        };

        // The store thread. It is the one owner of the store (spec 6.1).
        let store = match store_dir {
            Some(dir) => {
                let budget = store_budget.unwrap_or_else(|| StoreBudget::from_admission(&admitted.admission));
                match park::open_store(&dir, budget) {
                    Ok(store) => Some(store),
                    Err(error) => {
                        let _ = watch_tx.send(WatchMessage::Stop);
                        let _ = ledger_tx.send(LedgerMessage::Stop);
                        let _ = ledger_thread.join();
                        let _ = watchdog_thread.join();
                        return Err(StartError::Store {
                            path: dir,
                            message: error.to_string(),
                        });
                    }
                }
            }
            None => None,
        };
        let (store_tx, store_thread) = match store {
            Some(store) => {
                let (messages, replies) = mpsc::channel();
                let store_events = events_tx.clone();
                let handle = thread::spawn(move || {
                    guard("store", &store_events, || {
                        park::store_thread(store, replies);
                    });
                });
                (Some(messages), Some(handle))
            }
            None => (None, None),
        };

        let groups = placement::device_groups(&profile);
        let mut group_threads = Vec::with_capacity(groups.len());
        let mut group_stops = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let jobs = group_jobs(&profile, &admitted, &registry, group);
            let _ = watch_tx.send(WatchMessage::Start {
                group: index,
                devices: group.devices.clone(),
            });
            let (stop_tx, stop_rx) = mpsc::channel();
            let group_events = events_tx.clone();
            let group_guard = group_events.clone();
            let group_watch = watch_tx.clone();
            let group_store = store_tx.clone();
            let handle = thread::spawn(move || {
                guard("group", &group_guard, || {
                    group_thread(index, jobs, group_events, group_watch, stop_rx, heartbeat, group_store);
                });
            });
            group_threads.push(handle);
            group_stops.push(stop_tx);
        }

        Ok(Process {
            ledger: ledger_tx,
            ledger_thread: Some(ledger_thread),
            watchdog_stop: Some(watch_tx),
            watchdog_thread: Some(watchdog_thread),
            store: store_tx,
            store_thread,
            groups: group_threads,
            group_stops,
            events,
            group_count: groups.len(),
            accepting: AtomicBool::new(true),
            stopped: false,
        })
    }
}

/// An engine that one group thread opens.
struct GroupJob {
    placement: EnginePlacement,
    limits: Limits,
    factory: Arc<dyn Factory + Send + Sync>,
}

fn group_jobs(
    profile: &PlacementProfile,
    admitted: &Admitted,
    registry: &Registry,
    group: &DeviceGroup,
) -> Vec<GroupJob> {
    group
        .engines
        .iter()
        .map(|&index| {
            let placement = &profile.engines[index];
            let limits = admitted.engines[index].limits.clone();
            let factory = registry
                .factory(&placement.kind)
                .expect("the admitted kind has no factory");
            GroupJob {
                placement: placement.clone(),
                limits,
                factory,
            }
        })
        .collect()
}

fn group_thread(
    group: usize,
    jobs: Vec<GroupJob>,
    events: Sender<Event>,
    heartbeats: Sender<WatchMessage>,
    stop: Receiver<GroupMessage>,
    interval: Duration,
    store: Option<Sender<StoreMessage>>,
) {
    let _ = heartbeats.send(WatchMessage::Beat {
        group,
        at: Instant::now(),
    });
    let mut engines: Vec<Box<dyn Engine>> = Vec::with_capacity(jobs.len());
    let mut opened = Vec::with_capacity(jobs.len());
    for job in jobs {
        let engine = match job.factory.open(&job.placement.config) {
            Ok(engine) => engine,
            Err(error) => {
                let _ = events.send(Event::OpenFailed {
                    group,
                    engine: job.placement.name.clone(),
                    message: error.to_string(),
                });
                return;
            }
        };
        if engine.limits() != job.limits {
            let _ = events.send(Event::OpenFailed {
                group,
                engine: job.placement.name.clone(),
                message: "limits() after open do not equal the admitted limits".to_owned(),
            });
            return;
        }
        opened.push(OpenedEngine {
            name: job.placement.name.clone(),
            thread: thread::current().id(),
        });
        engines.push(engine);
    }
    let names: Vec<String> = opened.iter().map(|engine| engine.name.clone()).collect();
    let _ = events.send(Event::Opened { group, engines: opened });
    loop {
        match stop.recv_timeout(interval) {
            Ok(GroupMessage::Park(reply)) => {
                let report = park_group(&names, &mut engines, store.as_ref());
                let _ = reply.send(report);
                break;
            }
            Ok(GroupMessage::Stop) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {
                let beat = WatchMessage::Beat {
                    group,
                    at: Instant::now(),
                };
                if heartbeats.send(beat).is_err() {
                    break;
                }
            }
        }
    }
    drop(engines);
}

/// Park every engine of one group. A part that the engine cannot read or write is a failure.
fn park_group(names: &[String], engines: &mut [Box<dyn Engine>], store: Option<&Sender<StoreMessage>>) -> ParkReport {
    let mut report = ParkReport::default();
    for (index, engine) in engines.iter_mut().enumerate() {
        report.engines += 1;
        let name = names.get(index).map(String::as_str).unwrap_or("engine");
        park::park_engine(name, engine.as_mut(), store, &mut report);
    }
    report
}

fn watchdog(messages: Receiver<WatchMessage>, events: Sender<Event>, timeout: Duration) {
    let mut watches: BTreeMap<usize, Watch> = BTreeMap::new();
    let poll = (timeout / 4).clamp(Duration::from_millis(5), Duration::from_millis(250));
    loop {
        match messages.recv_timeout(poll) {
            Ok(WatchMessage::Start { group, devices }) => {
                watches.insert(
                    group,
                    Watch {
                        devices,
                        last: Instant::now(),
                    },
                );
            }
            Ok(WatchMessage::Beat { group, at }) => {
                if let Some(watch) = watches.get_mut(&group) {
                    watch.last = at;
                }
            }
            Ok(WatchMessage::Stop) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        let now = Instant::now();
        let stalled: Vec<usize> = watches
            .iter()
            .filter(|(_, watch)| now.duration_since(watch.last) > timeout)
            .map(|(group, _)| *group)
            .collect();
        for group in stalled {
            if let Some(watch) = watches.remove(&group) {
                let _ = events.send(Event::Stalled {
                    group,
                    devices: watch.devices,
                });
            }
        }
    }
}

struct Watch {
    devices: Vec<DeviceId>,
    last: Instant,
}

enum WatchMessage {
    Start { group: usize, devices: Vec<DeviceId> },
    Beat { group: usize, at: Instant },
    Stop,
}

enum LedgerMessage {
    Admit(PlacementProfile, Sender<Result<Admitted, AdmitError>>),
    Stop,
}

/// A request to one group thread.
enum GroupMessage {
    /// Park every engine, then stop the thread (spec 10.3).
    Park(Sender<ParkReport>),
    /// Stop the thread without a park. `Process::drop` uses it.
    Stop,
}

/// The started process threads.
#[derive(Debug)]
pub struct Process {
    ledger: Sender<LedgerMessage>,
    ledger_thread: Option<JoinHandle<()>>,
    watchdog_stop: Option<Sender<WatchMessage>>,
    watchdog_thread: Option<JoinHandle<()>>,
    store: Option<Sender<StoreMessage>>,
    store_thread: Option<JoinHandle<()>>,
    groups: Vec<JoinHandle<()>>,
    group_stops: Vec<Sender<GroupMessage>>,
    events: Receiver<Event>,
    group_count: usize,
    accepting: AtomicBool,
    stopped: bool,
}

impl Process {
    /// Send an admission request to the ledger thread and wait for its answer.
    ///
    /// A stop refuses every later request (spec 10.3 step 1).
    pub fn admit(&self, profile: PlacementProfile) -> Result<Admitted, AdmitError> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(AdmitError::Stopping);
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::Admit(profile, reply_tx))
            .expect("the ledger thread stopped");
        reply_rx.recv().expect("the ledger thread stopped")
    }

    /// Wait for every group to open its engines, or for a failure.
    pub fn wait_opens(&mut self, timeout: Duration) -> Result<Vec<OpenedEngine>, WaitError> {
        let deadline = Instant::now() + timeout;
        let mut opened = Vec::new();
        let mut done = 0;
        while done < self.group_count {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(WaitError::Timeout);
            }
            match self.events.recv_timeout(left) {
                Ok(Event::Opened { engines, .. }) => {
                    done += 1;
                    opened.extend(engines);
                }
                Ok(event) => return Err(WaitError::Event(event)),
                Err(RecvTimeoutError::Timeout) => return Err(WaitError::Timeout),
                Err(RecvTimeoutError::Disconnected) => return Err(WaitError::Disconnected),
            }
        }
        Ok(opened)
    }

    /// Read the next event, or give `None` when the timeout passes.
    pub fn recv(&self, timeout: Duration) -> Option<Event> {
        self.events.recv_timeout(timeout).ok()
    }

    /// Stop the process (spec 10.3). Refuse new work, park each group in the budget, then stop the threads.
    ///
    /// A second signal stops the park at once. A group that does not answer inside the budget stops nothing else.
    pub fn stop(&mut self, budget: Duration, signals: &SignalFlag) -> StopReport {
        if self.stopped {
            return StopReport::default();
        }
        self.accepting.store(false, Ordering::SeqCst);
        let started = Instant::now();
        let entered = signals.count();
        if entered > 1 {
            self.finish_stop();
            return StopReport {
                parked: ParkReport::default(),
                complete: false,
                interrupted: true,
            };
        }
        let mut replies = Vec::with_capacity(self.group_stops.len());
        let mut complete = true;
        for stop in &self.group_stops {
            let (reply_tx, reply_rx) = mpsc::channel();
            if stop.send(GroupMessage::Park(reply_tx)).is_ok() {
                replies.push(reply_rx);
            } else {
                complete = false;
            }
        }
        let mut parked = ParkReport::default();
        for reply in replies {
            loop {
                if signals.count() != entered {
                    self.finish_stop();
                    return StopReport {
                        parked,
                        complete: false,
                        interrupted: true,
                    };
                }
                let left = budget.saturating_sub(started.elapsed());
                if left.is_zero() {
                    complete = false;
                    break;
                }
                match reply.recv_timeout(left.min(STOP_POLL)) {
                    Ok(report) => {
                        parked.merge(report);
                        break;
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        complete = false;
                        break;
                    }
                }
            }
        }
        self.finish_stop();
        StopReport {
            parked,
            complete,
            interrupted: false,
        }
    }

    /// Ask every worker to stop, without a park and without a join. A crash exits at once (spec 8.5).
    pub fn abort(&mut self) {
        self.accepting.store(false, Ordering::SeqCst);
        for stop in &self.group_stops {
            let _ = stop.send(GroupMessage::Stop);
        }
        self.finish_stop();
    }

    /// True when every worker thread has ended.
    pub fn finished(&self) -> bool {
        self.groups.iter().all(JoinHandle::is_finished)
            && self.ledger_thread.as_ref().is_none_or(JoinHandle::is_finished)
            && self.watchdog_thread.as_ref().is_none_or(JoinHandle::is_finished)
            && self.store_thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Stop the watchdog, the store and the ledger. The drop joins them.
    fn finish_stop(&mut self) {
        if let Some(stop) = self.watchdog_stop.take() {
            let _ = stop.send(WatchMessage::Stop);
        }
        if let Some(store) = &self.store {
            let _ = store.send(StoreMessage::Stop);
        }
        let _ = self.ledger.send(LedgerMessage::Stop);
        self.stopped = true;
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        for stop in self.group_stops.drain(..) {
            let _ = stop.send(GroupMessage::Stop);
        }
        if let Some(stop) = self.watchdog_stop.take() {
            let _ = stop.send(WatchMessage::Stop);
        }
        let _ = self.ledger.send(LedgerMessage::Stop);
        if let Some(handle) = self.ledger_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.watchdog_thread.take() {
            let _ = handle.join();
        }
        // A group that is inside a stalled engine call cannot stop. The process exits (8.5).
        for handle in self.groups.drain(..) {
            let _ = handle.join();
        }
        if let Some(store) = &self.store {
            let _ = store.send(StoreMessage::Stop);
        }
        if let Some(handle) = self.store_thread.take() {
            let _ = handle.join();
        }
    }
}
