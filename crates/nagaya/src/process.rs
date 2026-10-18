//! The process threads: the ledger, the watchdog, and one thread per device group (spec 3.2).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use nagaya_abi::{DeviceId, Engine, Limits};
use nagaya_ledger::Ledger;

use crate::admission::{AdmitError, Admitted, Factory, Registry, admit};
use crate::exit::ExitCode;
use crate::inventory::Inventory;
use crate::placement::{self, DeviceGroup};
use crate::profile::{EnginePlacement, PlacementProfile};

/// The default watchdog timeout. Three times the longest declared quantum bound (mk: 10 s).
pub const DEFAULT_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(30);

/// The default heartbeat. A group beats six times in the default timeout.
pub const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(5);

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
}

impl std::fmt::Display for StartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::Admit(error) => write!(formatter, "the process cannot start: {error}"),
        }
    }
}

impl std::error::Error for StartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StartError::Admit(error) => Some(error),
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
        }
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
        let ledger_thread = thread::spawn(move || {
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

        // The watchdog thread. It watches the group threads from outside (spec 3.2, 7.4).
        let watchdog_events = events_tx.clone();
        let watchdog_thread = thread::spawn(move || watchdog(watch_rx, watchdog_events, timeout));

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
            let group_watch = watch_tx.clone();
            let handle = thread::spawn(move || {
                group_thread(index, jobs, group_events, group_watch, stop_rx, heartbeat);
            });
            group_threads.push(handle);
            group_stops.push(stop_tx);
        }

        Ok(Process {
            ledger: ledger_tx,
            ledger_thread: Some(ledger_thread),
            watchdog_stop: Some(watch_tx),
            watchdog_thread: Some(watchdog_thread),
            groups: group_threads,
            group_stops,
            events,
            group_count: groups.len(),
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
    stop: Receiver<()>,
    interval: Duration,
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
    let _ = events.send(Event::Opened { group, engines: opened });
    loop {
        match stop.recv_timeout(interval) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
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

/// The started process threads.
pub struct Process {
    ledger: Sender<LedgerMessage>,
    ledger_thread: Option<JoinHandle<()>>,
    watchdog_stop: Option<Sender<WatchMessage>>,
    watchdog_thread: Option<JoinHandle<()>>,
    groups: Vec<JoinHandle<()>>,
    group_stops: Vec<Sender<()>>,
    events: Receiver<Event>,
    group_count: usize,
}

impl Process {
    /// Send an admission request to the ledger thread and wait for its answer.
    pub fn admit(&self, profile: PlacementProfile) -> Result<Admitted, AdmitError> {
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
}

impl Drop for Process {
    fn drop(&mut self) {
        for stop in self.group_stops.drain(..) {
            let _ = stop.send(());
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
    }
}
