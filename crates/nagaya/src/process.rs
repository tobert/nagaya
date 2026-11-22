//! The process threads: the ledger, the watchdog, and one thread per device group (spec 3.2).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use nagaya_abi::{ArenaPart, CheckStrength, DeviceId, Engine, Identity, Limits, Tier};
use nagaya_ledger::{Ledger, LedgerError};
use nagaya_store::Store;
use nagaya_telemetry::{EngineState, Inherit};
use tracing::Span;

use crate::admission::{
    AdmitError, Admitted, Factory, GroupDevice, OpenErrorKind, Planned, Registry, admit_loaded_planned, admit_planned,
    group_devices, plan_all,
};
use crate::affinity::{pin_current_thread, set_batch_policy};
use crate::arena::{ArenaSet, FdName, FdStore};
use crate::cpu_sets::CpuSets;
use crate::evidence::Machine;
use crate::exit::{ExitCode, panic_text};
use crate::inventory::Inventory;
use crate::jobs::{GroupJobs, JobSpec};
use crate::park::{self, StoreBudget, StoreMessage};
use crate::placement::{self, DeviceGroup};
use crate::profile::{EnginePlacement, PlacementProfile, Residency};
use crate::signal::SignalFlag;
use in_place::{Exit, Member, Unloaded};
use loads::Ask;
pub use loads::{DeviceReply, EngineStates, LoadError, LoadReply};
use long_pass::LongLinks;
pub use long_pass::{LongPass, LongPasses};

mod handoff;
mod in_place;
mod loads;
mod long_pass;
mod staged;

/// The default watchdog timeout. Three times the longest declared quantum bound (the Qwen3.8 engine: 10 s).
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
    /// The engines that parked every state without a failure.
    pub parked_engines: Vec<String>,
    /// The queued quanta that the stop cancelled before they started (spec 10.3, 5.9).
    pub cancelled: usize,
    /// The pinned entries of the host tier that the stop parked to NVMe (spec 6.11).
    pub pinned: usize,
    /// The bytes of those entries.
    pub pinned_bytes: u64,
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
        self.cancelled += other.cancelled;
        self.pinned += other.pinned;
        self.pinned_bytes += other.pinned_bytes;
        self.parked_engines.extend(other.parked_engines);
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
    /// The exit code of the stop. Only a stop that parked every state is clean.
    ///
    /// A failed write, an unfinished park and an interrupted stop are a crash (spec 8.5). systemd does not restart
    /// the process after `systemctl stop`, so exit 1 for an interrupted stop is honest and harmless.
    pub fn exit_code(&self) -> ExitCode {
        if !self.parked.failures.is_empty() {
            return ExitCode::Crash;
        }
        if self.complete {
            ExitCode::Clean
        } else {
            ExitCode::Crash
        }
    }

    /// Split engine names into the engines that parked every state and the engines that did not (spec 10.1, 10.3).
    ///
    /// An engine of a group that did not answer, or of a stop that was interrupted, did not park.
    pub fn split_parked(&self, names: &[String]) -> (Vec<String>, Vec<String>) {
        names
            .iter()
            .cloned()
            .partition(|name| self.parked.parked_engines.contains(name))
    }

    /// The line that the process writes to stderr when the stop ends (spec 10.3).
    pub fn log_line(&self) -> String {
        if self.interrupted {
            return "the stop was interrupted before the park finished".to_owned();
        }
        let cancelled = match self.parked.cancelled {
            0 => String::new(),
            count => format!("; cancelled {count} queued quanta"),
        };
        let failed = self.parked.failures.len();
        if !self.complete {
            return format!(
                "the park did not finish; it wrote {} states and {} parts{cancelled}",
                self.parked.states, self.parked.parts
            );
        }
        if self.parked.states == 0 && self.parked.entries == 0 && self.parked.pinned == 0 && failed == 0 {
            return format!("the stop parked nothing: no state needed a park{cancelled}");
        }
        let pinned = match self.parked.pinned {
            0 => String::new(),
            count => format!(
                " and parked {count} pinned entries ({} bytes)",
                self.parked.pinned_bytes
            ),
        };
        if failed == 0 {
            return format!(
                "the stop wrote {} states and {} parts ({} bytes) to the store{pinned}{cancelled}",
                self.parked.states, self.parked.parts, self.parked.bytes
            );
        }
        format!(
            "the stop wrote {} states and {} parts, then failed {failed} part(s){cancelled}",
            self.parked.states, self.parked.parts
        )
    }
}

/// Start a worker thread with a name. The name shows in `ps`, in `top` and in the CPU mask test (spec 3.2a).
fn spawn_named(name: String, body: impl FnOnce() + Send + 'static) -> JoinHandle<()> {
    // The new thread records on the telemetry of this thread. In the process this is the same everywhere.
    let inherit = Inherit::capture();
    thread::Builder::new()
        .name(name)
        .spawn(move || {
            let _telemetry = inherit.enter();
            body()
        })
        .expect("cannot start a worker thread")
}

/// Pin the calling thread to `cpus` (spec 3.2a). With no CPUs, the thread keeps the mask that it inherited.
///
/// A failure panics. The `guard` of the thread turns the panic into an event, and the process exits 1 (spec 8.5).
fn pin(thread: &str, cpus: Option<&[u32]>) {
    if let Some(cpus) = cpus {
        pin_current_thread(cpus)
            .unwrap_or_else(|error| panic!("cannot pin the {thread} thread to CPUs {cpus:?}: {error}"));
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
    fd_ceiling: Option<usize>,
    /// The names of the loaded engines of the active profile.
    loaded: BTreeSet<String>,
    /// The plans of the engines of the profile that the ledger admitted last. A swap admits the loaded engines from
    /// them, so it runs no plan (spec 3.3, 3.3a).
    planned: Option<Planned>,
    /// The swaps that groups planned and did not charge yet, as the loads and the unloads of each. Their engines that
    /// unload keep their bytes until the drop, and a charge must also fit the ledger after them (spec 3.3a).
    pending: Vec<(Vec<String>, Vec<String>)>,
    /// True from the accept of a profile change until the ledger takes the new profile. No load is planned then
    /// (spec 3.3a).
    changing: bool,
    /// The plans and the engine names of the profile of an accepted change, until the ledger takes it. Each charge
    /// must also fit it (spec 3.3a).
    incoming: Option<(Planned, BTreeSet<String>)>,
}

/// The loaded engines after a swap that loads `load` and unloads `unload`.
fn swapped_set(loaded: &BTreeSet<String>, load: &[String], unload: &[String]) -> BTreeSet<String> {
    let mut loaded = loaded.clone();
    for name in unload {
        assert!(
            loaded.remove(name),
            "the swap unloads engine `{name}`, which is not loaded"
        );
    }
    for name in load {
        assert!(
            loaded.insert(name.clone()),
            "the swap loads engine `{name}`, which is loaded"
        );
    }
    loaded
}

impl OwnedLedger {
    /// Make a ledger for the current thread.
    pub fn new() -> Self {
        Self::with_fd_ceiling(None)
    }

    /// Make a ledger for the current thread that refuses a profile with more arena fds than `fd_ceiling`
    /// (spec 10.2).
    pub fn with_fd_ceiling(fd_ceiling: Option<usize>) -> Self {
        Self {
            ledger: Ledger::new(),
            owner: thread::current().id(),
            fd_ceiling,
            loaded: BTreeSet::new(),
            planned: None,
            pending: Vec::new(),
            changing: false,
            incoming: None,
        }
    }

    /// Admit a profile. Its resident engines are the loaded engines (spec 3.3a). Only the owner thread may call this.
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
        let plans = plan_all(profile, registry, inventory, self.fd_ceiling)?;
        let admitted = admit_planned(profile, &plans, &mut self.ledger, true)?;
        self.planned = Some(plans);
        self.loaded = profile
            .engines
            .iter()
            .filter(|engine| engine.residency == Residency::Resident)
            .map(|engine| engine.name.clone())
            .collect();
        Ok(admitted)
    }

    /// The loaded engines after a swap that loads `load` and unloads `unload`.
    fn swapped(&self, load: &[String], unload: &[String]) -> BTreeSet<String> {
        swapped_set(&self.loaded, load, unload)
    }

    /// Check whether a swap fits, and change nothing (spec 3.3a). Only the owner thread may call this.
    pub fn check_swap(&mut self, load: &[String], unload: &[String]) -> Result<(), AdmitError> {
        self.check_owner();
        let plans = self
            .planned
            .as_ref()
            .expect("a swap comes after the admission of a profile");
        let loaded = self.swapped(load, unload);
        admit_loaded_planned(plans, &mut self.ledger, &loaded, false).map(|_| ())
    }

    /// Take a swap that a group plans, before any engine unloads (spec 3.3a). It must fit after every pending swap, with
    /// the live charges. Nothing is charged until [`OwnedLedger::swap`], so the engines that unload keep their bytes
    /// while they drain. A refusal changes nothing. Only the owner thread may call this.
    pub fn plan_swap(&mut self, load: &[String], unload: &[String]) -> Result<(), AdmitError> {
        self.check_owner();
        if self.changing && !load.is_empty() {
            return Err(AdmitError::Busy(
                "a profile change waits for the ledger, and no load runs until the ledger takes it".to_owned(),
            ));
        }
        self.pending.push((load.to_vec(), unload.to_vec()));
        // A swap that only unloads leaves fewer engines, so the ledger never refuses it. It is pending so that a
        // profile change waits for it.
        if load.is_empty() {
            return Ok(());
        }
        let fits = self.check_pending().map_err(AdmitError::Ledger);
        if fits.is_err() {
            self.pending.pop();
        }
        fits
    }

    /// Check that the live charges fit when each pending swap is charged, in any order (spec 3.3a). A swap is charged
    /// after its own unloads, beside any set of the other pending swaps that ended before it. A swap that ended is
    /// whole: its load is charged only after its unload. There are few pending swaps, one for each group at most. It
    /// changes nothing.
    fn check_pending(&self) -> Result<(), LedgerError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let plans = self
            .planned
            .as_ref()
            .expect("a swap comes after the admission of a profile");
        let apply = |loaded: &mut BTreeSet<String>, (load, unload): &(Vec<String>, Vec<String>)| {
            for name in unload {
                loaded.remove(name);
            }
            loaded.extend(load.iter().cloned());
        };
        let count = self.pending.len();
        for (index, swap) in self.pending.iter().enumerate() {
            for ended in 0..1u32 << (count - 1) {
                let mut charged = self.loaded.clone();
                let others = (0..count).filter(|other| *other != index);
                for (bit, other) in others.enumerate() {
                    if ended & 1 << bit != 0 {
                        apply(&mut charged, &self.pending[other]);
                    }
                }
                apply(&mut charged, swap);
                self.check_charged(plans, &charged)?;
            }
        }
        Ok(())
    }

    /// Check that the live charges fit beside the engines of `charged`. It changes nothing.
    fn check_charged(&self, plans: &Planned, charged: &BTreeSet<String>) -> Result<(), LedgerError> {
        {
            let mut after = self.ledger.clone();
            match admit_loaded_planned(plans, &mut after, charged, true) {
                Ok(_) => {}
                Err(AdmitError::Ledger(error)) => return Err(error),
                Err(error) => panic!("the ledger cannot check its pending swaps: {error}"),
            }
        }
        Ok(())
    }

    /// Check that the live charges fit the profile of an accepted change, as the ledger will take it: without the
    /// contexts of the engines that it removes, and with the loaded engines that it keeps (spec 3.3a). It changes
    /// nothing.
    fn check_incoming(&self) -> Result<(), LedgerError> {
        let Some((plans, names)) = &self.incoming else {
            return Ok(());
        };
        let mut live = self.ledger.clone();
        if let Some(old) = &self.planned {
            for name in old.names() {
                if !names.contains(name) {
                    while live.release_context(name).is_ok() {}
                }
            }
        }
        let staying: BTreeSet<String> = self.loaded.intersection(names).cloned().collect();
        match admit_loaded_planned(plans, &mut live, &staying, true) {
            Ok(_) => Ok(()),
            Err(AdmitError::Ledger(error)) => Err(error),
            Err(error) => panic!("the ledger cannot check the profile of the change: {error}"),
        }
    }

    /// True when an accepted change removes the engine, so it takes no new charge (spec 3.3a).
    pub fn leaving(&self, engine: &str) -> bool {
        self.incoming.as_ref().is_some_and(|(_, names)| !names.contains(engine))
    }

    /// Check a charge that the ledger just took, and take it back when it does not fit after the pending swaps or in
    /// the profile of an accepted change.
    fn check_charge(&self) -> Result<(), LedgerError> {
        self.check_pending()?;
        self.check_incoming()
    }

    /// Admit the loaded engines after a swap, when its engines that unload dropped (spec 3.3a). A swap that loads an
    /// engine was taken by [`OwnedLedger::plan_swap`]. A refusal changes nothing. Only the owner thread may call this.
    pub fn swap(&mut self, load: &[String], unload: &[String]) -> Result<Admitted, AdmitError> {
        self.check_owner();
        let plans = self
            .planned
            .as_ref()
            .expect("a swap comes after the admission of a profile");
        let loaded = self.swapped(load, unload);
        let admitted = admit_loaded_planned(plans, &mut self.ledger, &loaded, true)?;
        self.loaded = loaded;
        if let Some(at) = self
            .pending
            .iter()
            .position(|(planned_load, planned_unload)| planned_load == load && planned_unload == unload)
        {
            self.pending.remove(at);
        } else {
            assert!(load.is_empty(), "a swap loads {load:?}, and no group planned it");
        }
        Ok(admitted)
    }

    /// Check a whole profile, with no change (spec 3.3). Only the owner thread may call this.
    ///
    /// The profile must fit an empty machine, and its loaded engines must fit the live ledger with its contexts and
    /// staging. The change keeps the profile before the ledger takes it, so a refusal after the keep would stop the
    /// process.
    pub fn check_profile(
        &mut self,
        profile: &PlacementProfile,
        registry: &Registry,
        inventory: &Inventory,
    ) -> Result<Admitted, AdmitError> {
        self.check_profile_plans(profile, registry, inventory)
            .map(|(admitted, _)| admitted)
    }

    /// Check a whole profile as [`OwnedLedger::check_profile`] does, and give its plans too.
    fn check_profile_plans(
        &mut self,
        profile: &PlacementProfile,
        registry: &Registry,
        inventory: &Inventory,
    ) -> Result<(Admitted, Planned), AdmitError> {
        self.check_owner();
        let plans = plan_all(profile, registry, inventory, self.fd_ceiling)?;
        let admitted = admit_planned(profile, &plans, &mut Ledger::new(), false)?;
        // The change retires each loaded engine that the profile lacks before the ledger takes it.
        let staying: BTreeSet<String> = self
            .loaded
            .iter()
            .filter(|name| profile.engines.iter().any(|engine| &engine.name == *name))
            .cloned()
            .collect();
        // An engine that the profile removes retires, and its contexts end, before the ledger takes the profile.
        let mut live = self.ledger.clone();
        if let Some(old) = &self.planned {
            for name in old.names() {
                if !profile.engines.iter().any(|entry| entry.name == name) {
                    while live.release_context(name).is_ok() {}
                }
            }
        }
        admit_loaded_planned(&plans, &mut live, &staying, false)?;
        Ok((admitted, plans))
    }

    /// Check a profile change and begin it (spec 3.3a). A swap that waits or runs makes it busy, and so does another
    /// change. No load is planned after it until [`OwnedLedger::try_reprofile`] takes the profile. Only the owner
    /// thread may call this.
    pub fn begin_change(
        &mut self,
        profile: &PlacementProfile,
        registry: &Registry,
        inventory: &Inventory,
    ) -> Result<Admitted, AdmitError> {
        self.check_owner();
        if !self.pending.is_empty() {
            return Err(AdmitError::Busy("a swap waits or runs now".to_owned()));
        }
        if self.changing {
            return Err(AdmitError::Busy("a profile change runs now".to_owned()));
        }
        let (admitted, plans) = self.check_profile_plans(profile, registry, inventory)?;
        let names = profile.engines.iter().map(|engine| engine.name.clone()).collect();
        self.incoming = Some((plans, names));
        self.changing = true;
        Ok(admitted)
    }

    /// True from the start of a profile change until the ledger takes its profile or the change ends (spec 3.3a).
    pub fn changing(&self) -> bool {
        self.changing
    }

    /// End a profile change that was begun and not kept. Only the owner thread may call this.
    pub fn end_change(&mut self) {
        self.check_owner();
        self.changing = false;
        self.incoming = None;
    }

    /// Take a new profile when no swap waits or runs, and the engines of `gone` dropped and hold no context (spec
    /// 3.3a). `Err` in the inner result names what it waits for. Only the owner thread may call this.
    pub fn try_reprofile(
        &mut self,
        profile: &PlacementProfile,
        registry: &Registry,
        inventory: &Inventory,
        gone: &[String],
    ) -> Result<Result<Admitted, String>, AdmitError> {
        self.check_owner();
        if !self.pending.is_empty() {
            return Ok(Err(format!("{} swaps that wait or run", self.pending.len())));
        }
        for name in gone {
            if self.loaded.contains(name) {
                return Ok(Err(format!("the drop of engine `{name}`")));
            }
            let live = self.ledger.live_contexts(name);
            if live > 0 {
                return Ok(Err(format!("{live} contexts of engine `{name}`")));
            }
        }
        let admitted = self.reprofile(profile, registry, inventory)?;
        self.changing = false;
        self.incoming = None;
        Ok(Ok(admitted))
    }

    /// Admit the loaded engines under a new profile, as a swap does (spec 3.3). A refusal changes nothing. Only the
    /// owner thread may call this.
    ///
    /// Each loaded engine must be in the new profile: the change retires an engine before it changes the ledger. A
    /// loaded engine that the profile lacks holds memory that the ledger would not count, so it is a defect (8.5).
    pub fn reprofile(
        &mut self,
        profile: &PlacementProfile,
        registry: &Registry,
        inventory: &Inventory,
    ) -> Result<Admitted, AdmitError> {
        self.check_owner();
        for name in &self.loaded {
            assert!(
                profile.engines.iter().any(|engine| &engine.name == name),
                "engine `{name}` is loaded, and the new profile lacks it; retire it before the ledger takes the profile"
            );
        }
        let plans = plan_all(profile, registry, inventory, None)?;
        let admitted = admit_loaded_planned(&plans, &mut self.ledger, &self.loaded, true)?;
        self.planned = Some(plans);
        Ok(admitted)
    }

    /// Charge a new context of an engine (spec 4.2). Only the owner thread may call this.
    pub fn admit_context(&mut self, engine: &str) -> Result<(), LedgerError> {
        self.check_owner();
        self.ledger.admit_context(engine)?;
        // The charge must also fit after the pending swaps, so that no charge refuses a swap after its unload.
        if let Err(error) = self.check_charge() {
            self.ledger
                .release_context(engine)
                .expect("a context that was just charged releases");
            return Err(match error {
                LedgerError::Refused(refusal) => LedgerError::ContextRefused(refusal),
                other => other,
            });
        }
        Ok(())
    }

    /// Release the charge of an ended context. Only the owner thread may call this.
    pub fn release_context(&mut self, engine: &str) -> Result<(), LedgerError> {
        self.check_owner();
        self.ledger.release_context(engine)
    }

    /// Charge staging bytes of an engine (spec 4.2). Only the owner thread may call this.
    pub fn admit_staging(&mut self, engine: &str, tier: Tier, bytes: u64) -> Result<(), LedgerError> {
        self.check_owner();
        self.ledger.admit_staging(engine, tier, bytes)?;
        // The charge must also fit after the pending swaps, as a context does.
        if let Err(error) = self.check_charge() {
            self.ledger
                .release_staging(tier, bytes)
                .expect("staging that was just charged releases");
            return Err(match error {
                LedgerError::Refused(refusal) => LedgerError::StagingRefused(refusal),
                other => other,
            });
        }
        Ok(())
    }

    /// Release staging bytes. Only the owner thread may call this.
    pub fn release_staging(&mut self, tier: Tier, bytes: u64) -> Result<(), LedgerError> {
        self.check_owner();
        self.ledger.release_staging(tier, bytes)
    }

    fn check_owner(&self) {
        assert_eq!(
            thread::current().id(),
            self.owner,
            "the ledger is written off its thread"
        );
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
#[derive(Clone, Debug, PartialEq)]
pub struct OpenedEngine {
    /// The engine name.
    pub name: String,
    /// The thread that opened it.
    pub thread: ThreadId,
    /// The identity that the engine reported after it opened (spec 12.1).
    pub identity: Identity,
    /// The limits that the engine reported after it opened (spec 11.3: the slot sizes give the context limit).
    pub limits: Limits,
    /// How the engine got its weights, when its kind keeps an arena (spec 10.2).
    pub handoff: Option<Handoff>,
}

/// How an engine with an arena got its weights at start (spec 10.2, 9.4 `nagaya.arena.handoff`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handoff {
    /// The engine attached a stored arena, and the check had this strength.
    Hot(CheckStrength),
    /// The engine loaded its weights from the checkpoint.
    Cold,
}

impl Handoff {
    /// The value of the attribute `status` of `nagaya.arena.handoff`.
    pub fn status(self) -> &'static str {
        match self {
            Handoff::Hot(_) => "hot",
            Handoff::Cold => "cold",
        }
    }
}

/// The work of one quantum: the group's engines, in profile order.
type QuantumWork = Box<dyn FnOnce(&mut [Box<dyn Engine>]) + Send>;

/// What became of a quantum that reached its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumOutcome {
    /// The group ran the quantum to its end.
    Ran,
    /// The stop cancelled the quantum before it started (spec 10.3, 5.9). The work did not run.
    Cancelled,
}

/// One quantum of work for a group thread.
///
/// The group runs a started quantum to its end before the park reads a state. A stop cancels each quantum that has
/// not started (spec 10.3).
pub struct Quantum {
    work: QuantumWork,
    outcome: Option<Sender<QuantumOutcome>>,
}

impl Quantum {
    /// Make a quantum from its work. The work gets the group's engines, in profile order.
    pub fn new(work: impl FnOnce(&mut [Box<dyn Engine>]) + Send + 'static) -> Self {
        Self {
            work: Box::new(work),
            outcome: None,
        }
    }

    /// Make a quantum and the receiver of its outcome.
    ///
    /// The receiver gets `Ran` or `Cancelled`. It gets no outcome and reports a closed channel when the quantum never
    /// reached its group, or when its work panicked.
    pub fn with_outcome(
        work: impl FnOnce(&mut [Box<dyn Engine>]) + Send + 'static,
    ) -> (Self, Receiver<QuantumOutcome>) {
        let (sender, receiver) = mpsc::channel();
        let quantum = Self {
            work: Box::new(work),
            outcome: Some(sender),
        };
        (quantum, receiver)
    }

    fn finish(self, engines: &mut [Box<dyn Engine>]) {
        (self.work)(engines);
        if let Some(outcome) = self.outcome {
            let _ = outcome.send(QuantumOutcome::Ran);
        }
    }

    fn cancel(self) {
        if let Some(outcome) = self.outcome {
            let _ = outcome.send(QuantumOutcome::Cancelled);
        }
    }
}

/// Why a quantum did not reach its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// The process is stopping and takes no new jobs.
    Stopping,
    /// There is no group at this index.
    UnknownGroup,
    /// The group thread stopped.
    Stopped,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Stopping => write!(formatter, "the process is stopping"),
            SubmitError::UnknownGroup => write!(formatter, "there is no group at this index"),
            SubmitError::Stopped => write!(formatter, "the group stopped"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// A report from a group thread or the watchdog.
#[derive(Clone, Debug, PartialEq)]
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
        /// The kind of failure, for the exit code (spec 10.4).
        kind: OpenErrorKind,
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
    /// A group swapped engines in place (spec 3.3a).
    Swapped {
        /// The group index.
        group: usize,
        /// The engines that unloaded, in profile order.
        unloaded: Vec<String>,
        /// The engines that loaded.
        loaded: Vec<OpenedEngine>,
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
            Event::OpenFailed { kind, .. } => Some(if kind.refuses_start() {
                ExitCode::RefusedStart
            } else {
                ExitCode::Crash
            }),
            Event::Panicked { .. } => Some(ExitCode::Crash),
            Event::Opened { .. } | Event::Swapped { .. } => None,
        }
    }
}

/// Why the opens did not finish.
#[derive(Clone, Debug, PartialEq)]
pub enum WaitError {
    /// A group reported a failure.
    Event(Event),
    /// The groups did not open in time.
    Timeout {
        /// The engines that had not opened, in profile order.
        waiting: Vec<String>,
    },
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
    /// The profile names no store NVMe budget (spec 4.1 rule).
    NoStoreBudget,
}

impl std::fmt::Display for StartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::Admit(error) => write!(formatter, "the process cannot start: {error}"),
            StartError::Store { path, message } => {
                write!(formatter, "cannot open the store `{}`: {message}", path.display())
            }
            StartError::NoStoreBudget => {
                write!(
                    formatter,
                    "the placement profile has no `store.nvme` budget; name it in the profile"
                )
            }
        }
    }
}

impl std::error::Error for StartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StartError::Admit(error) => Some(error),
            StartError::Store { .. } | StartError::NoStoreBudget => None,
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
    cpu_sets: Option<CpuSets>,
    machine: Machine,
    stored: ArenaSet,
    fd_store: Option<FdStore>,
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
            cpu_sets: None,
            machine: Machine::real(),
            stored: ArenaSet::empty(),
            fd_store: None,
        }
    }

    /// Set the fds that systemd passed back, and the fd store (spec 10.2). With no store, no engine exports its
    /// arena, and admission counts no arena fd.
    pub fn arenas(mut self, stored: ArenaSet, fd_store: Option<FdStore>) -> Self {
        self.stored = stored;
        self.fd_store = fd_store;
        self
    }

    /// Set the machine whose counters a group reads when a quantum fails (spec 9.6). The default is the real machine.
    /// A test gives a fixed tree of files.
    pub fn machine(mut self, machine: Machine) -> Self {
        self.machine = machine;
        self
    }

    /// Set the CPU sets. Each thread pins itself to its set when it starts (spec 3.2a). With no sets, no thread pins.
    pub fn cpu_sets(mut self, sets: CpuSets) -> Self {
        self.cpu_sets = Some(sets);
        self
    }

    /// Set the store directory. The process opens the machine's store there (spec 6.1).
    pub fn store(mut self, dir: impl Into<PathBuf>) -> Self {
        self.store_dir = Some(dir.into());
        self
    }

    /// Set the store budgets for a test. Only the `fake` feature builds this method.
    ///
    /// A normal build has no way to override the profile's `[store]` table (spec 4.1a).
    #[cfg(feature = "fake")]
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
            cpu_sets,
            machine,
            stored,
            fd_store,
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
        let housekeeping = cpu_sets.as_ref().map(|sets| sets.housekeeping.clone());
        let pool = cpu_sets.as_ref().map(|sets| sets.pool.clone());
        let ledger_cpus = housekeeping.clone();
        let fd_ceiling = fd_store.as_ref().map(FdStore::ceiling);
        let ledger_thread = spawn_named("nagaya-ledger".to_owned(), move || {
            guard("ledger", &ledger_events, || {
                pin("ledger", ledger_cpus.as_deref());
                let mut ledger = OwnedLedger::with_fd_ceiling(fd_ceiling);
                while let Ok(message) = ledger_rx.recv() {
                    match message {
                        LedgerMessage::Admit(profile, reply) => {
                            let result = ledger.admit(&profile, &ledger_registry, &ledger_inventory);
                            let _ = reply.send(result);
                        }
                        LedgerMessage::AdmitContext(engine, reply) => {
                            // An engine that an accepted change removes takes no new context (spec 3.3a).
                            let answer = if ledger.leaving(&engine) {
                                Err(ContextChargeError::Leaving(engine))
                            } else {
                                ledger.admit_context(&engine).map_err(ContextChargeError::Ledger)
                            };
                            let _ = reply.send(answer);
                        }
                        LedgerMessage::ReleaseContext(engine, reply) => {
                            let _ = reply.send(ledger.release_context(&engine));
                        }
                        LedgerMessage::AdmitStaging(engine, tier, bytes, reply) => {
                            let answer = if ledger.leaving(&engine) {
                                Err(ContextChargeError::Leaving(engine))
                            } else {
                                ledger
                                    .admit_staging(&engine, tier, bytes)
                                    .map_err(ContextChargeError::Ledger)
                            };
                            let _ = reply.send(answer);
                        }
                        LedgerMessage::ReleaseStaging(tier, bytes, reply) => {
                            let _ = reply.send(ledger.release_staging(tier, bytes));
                        }
                        LedgerMessage::Unassigned(reply) => {
                            let _ = reply.send(ledger.ledger().unassigned());
                        }
                        LedgerMessage::PlanSwap { load, unload, reply } => {
                            let _ = reply.send(ledger.plan_swap(&load, &unload));
                        }
                        LedgerMessage::Swap { load, unload, reply } => {
                            let _ = reply.send(ledger.swap(&load, &unload));
                        }
                        LedgerMessage::Loaded(reply) => {
                            let _ = reply.send(ledger.loaded.clone());
                        }
                        LedgerMessage::Changing(reply) => {
                            let _ = reply.send(ledger.changing());
                        }
                        LedgerMessage::CheckProfile(profile, reply) => {
                            let _ = reply.send(ledger.check_profile(&profile, &ledger_registry, &ledger_inventory));
                        }
                        LedgerMessage::BeginChange(profile, reply) => {
                            let _ = reply.send(ledger.begin_change(&profile, &ledger_registry, &ledger_inventory));
                        }
                        LedgerMessage::EndChange => ledger.end_change(),
                        LedgerMessage::TryReprofile(profile, gone, reply) => {
                            let _ =
                                reply.send(ledger.try_reprofile(&profile, &ledger_registry, &ledger_inventory, &gone));
                        }
                        LedgerMessage::Reprofile(profile, reply) => {
                            let _ = reply.send(ledger.reprofile(&profile, &ledger_registry, &ledger_inventory));
                        }
                        LedgerMessage::Stop => break,
                    }
                }
            });
        });

        // The watchdog thread. It watches the group threads from outside (spec 3.2, 7.4).
        let long_passes = LongPasses::default();
        let watchdog_passes = long_passes.clone();
        let watchdog_events = events_tx.clone();
        let watchdog_body = watchdog_events.clone();
        let watchdog_thread = spawn_named("nagaya-watchdog".to_owned(), move || {
            guard("watchdog", &watchdog_events, || {
                pin("watchdog", housekeeping.as_deref());
                watchdog(watch_rx, watchdog_body, timeout, watchdog_passes);
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
                let budget = match store_budget {
                    Some(budget) => Some(budget),
                    None => match &profile.store {
                        Some(spec) => Some(StoreBudget::from_spec(spec, &admitted.admission)),
                        // A profile with no engine parks nothing, so it can keep an empty store (spec 10.1 note).
                        None if profile.engines.is_empty() => Some(StoreBudget {
                            host: StoreBudget::host_from_admission(&admitted.admission),
                            nvme: 0,
                            interval_copies: 0,
                        }),
                        None => None,
                    },
                };
                let Some(budget) = budget else {
                    let _ = watch_tx.send(WatchMessage::Stop);
                    let _ = ledger_tx.send(LedgerMessage::Stop);
                    let _ = ledger_thread.join();
                    let _ = watchdog_thread.join();
                    return Err(StartError::NoStoreBudget);
                };
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
        let store = store.map(|store| Arc::new(Mutex::new(store)));
        let store_handle = store.clone();
        let (store_tx, store_thread) = match store {
            Some(store_for_thread) => {
                let (messages, replies) = mpsc::channel();
                let store_events = events_tx.clone();
                let store_cpus = pool.clone();
                let handle = spawn_named("nagaya-store".to_owned(), move || {
                    guard("store", &store_events, || {
                        // The store hashes and converts bytes. It is pool work (spec 3.2).
                        if store_cpus.is_some() {
                            pin("store", store_cpus.as_deref());
                            set_batch_policy()
                                .unwrap_or_else(|error| panic!("cannot set SCHED_BATCH on the store thread: {error}"));
                        }
                        park::store_thread(&store_for_thread, replies);
                    });
                });
                (Some(messages), Some(handle))
            }
            None => (None, None),
        };

        // The profile fits, so a cold load fits too. Only now may the process drop a stored fd (spec 10.2 step 2).
        let names: Vec<String> = profile.engines.iter().map(|engine| engine.name.clone()).collect();
        let mut split = stored.split(&names);
        handoff::drop_orphans(split.orphans, fd_store.as_ref());
        // An on-demand engine stays unloaded at start, and its stored fds hold device memory (spec 3.3a).
        for placement in profile
            .engines
            .iter()
            .filter(|engine| engine.residency == Residency::OnDemand)
        {
            if let Some(stored) = split.engines.remove(&placement.name) {
                handoff::drop_stored(&placement.name, stored, fd_store.as_ref());
            }
        }

        let accepting = Arc::new(AtomicBool::new(true));
        // A resident engine loads at start, and an on-demand engine waits for a load (spec 3.3a).
        let states = EngineStates::default();
        for placement in &profile.engines {
            let state = match placement.residency {
                Residency::Resident => EngineState::Loading,
                Residency::OnDemand => EngineState::Unloaded,
            };
            states.set(&placement.name, state);
        }
        let groups = placement::device_groups(&profile);
        let mut group_threads = Vec::with_capacity(groups.len());
        let mut group_stops = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let jobs = group_jobs(&profile, &admitted, &registry, &inventory, group, &mut split.engines);
            let _ = watch_tx.send(WatchMessage::Start {
                group: index,
                devices: group.devices.clone(),
            });
            let (stop_tx, stop_rx) = mpsc::channel();
            let group_events = events_tx.clone();
            let group_guard = group_events.clone();
            let group_watch = watch_tx.clone();
            let group_store = store_tx.clone();
            let group_accepting = Arc::clone(&accepting);
            let group_cpus = pool.clone();
            let group_sets = cpu_sets.clone();
            let group_machine = machine.clone();
            let group_own = stop_tx.clone();
            let group_fd_store = fd_store.clone();
            let group_ledger = ledger_tx.clone();
            let group_states = states.clone();
            let group_long = LongLinks {
                passes: long_passes.clone(),
                timeout,
                interval: heartbeat,
            };
            let handle = spawn_named(format!("nagaya-group-{index}"), move || {
                guard("group", &group_guard, || {
                    pin("group", group_cpus.as_deref());
                    let links = GroupLinks {
                        events: group_events,
                        heartbeats: group_watch,
                        messages: stop_rx,
                        own: group_own,
                        store: group_store,
                        accepting: group_accepting,
                        machine: group_machine,
                        sets: group_sets,
                        fd_store: group_fd_store,
                        ledger: group_ledger,
                        states: group_states,
                        long: group_long,
                    };
                    group_thread(index, jobs, links, heartbeat);
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
            store_handle,
            store_thread,
            groups: group_threads,
            group_stops,
            events,
            group_count: groups.len(),
            group_engines: groups
                .iter()
                .map(|group| {
                    group
                        .engines
                        .iter()
                        .map(|&index| profile.engines[index].name.clone())
                        .collect()
                })
                .collect(),
            accepting,
            failure: Arc::new(Mutex::new(None)),
            states,
            released: Released::default(),
            long_passes,
            stopped: false,
        })
    }
}

/// An engine that one group thread opens.
struct GroupJob {
    placement: EnginePlacement,
    limits: Limits,
    devices: Vec<GroupDevice>,
    factory: Arc<dyn Factory + Send + Sync>,
    /// The admitted parts of the arena (spec 10.2).
    arena_parts: Vec<ArenaPart>,
    /// The stored fds of the engine that systemd passed back.
    stored: Vec<(FdName, std::os::fd::OwnedFd)>,
}

impl GroupJob {
    /// A copy with no stored fd, to hold the place of the job while a stage thread opens its engine.
    fn shell(&self) -> Self {
        Self {
            placement: self.placement.clone(),
            limits: self.limits.clone(),
            devices: self.devices.clone(),
            factory: Arc::clone(&self.factory),
            arena_parts: self.arena_parts.clone(),
            stored: Vec::new(),
        }
    }
}

fn group_jobs(
    profile: &PlacementProfile,
    admitted: &Admitted,
    registry: &Registry,
    inventory: &Inventory,
    group: &DeviceGroup,
    stored: &mut BTreeMap<String, Vec<(FdName, std::os::fd::OwnedFd)>>,
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
            let devices = group_devices(placement, inventory)
                .unwrap_or_else(|error| panic!("the admitted profile has a device that the inventory lacks: {error}"));
            GroupJob {
                placement: placement.clone(),
                limits,
                devices,
                factory,
                arena_parts: admitted.engines[index].arena_parts.clone(),
                stored: stored.remove(&placement.name).unwrap_or_default(),
            }
        })
        .collect()
}

/// The channels and flags that connect a group thread to the rest of the process.
struct GroupLinks {
    events: Sender<Event>,
    heartbeats: Sender<WatchMessage>,
    messages: Receiver<GroupMessage>,
    /// A sender to the group's own channel. The stage threads send the end of each stage quantum on it.
    own: Sender<GroupMessage>,
    store: Option<Sender<StoreMessage>>,
    accepting: Arc<AtomicBool>,
    machine: Machine,
    /// The CPU sets, from which each stage thread takes the set of its device (spec 3.2b).
    sets: Option<CpuSets>,
    /// The fd store of systemd, when the process keeps arenas (spec 10.2).
    fd_store: Option<FdStore>,
    /// The ledger thread, which checks and admits a swap (spec 3.3a).
    ledger: Sender<LedgerMessage>,
    /// The state of each engine. The group writes the states of its own engines (spec 3.3b).
    states: EngineStates,
    /// The links of the long passes of the engines that the group opens (engine-abi D8).
    long: LongLinks,
}

fn group_thread(group: usize, jobs: Vec<GroupJob>, links: GroupLinks, interval: Duration) {
    let _ = links.heartbeats.send(WatchMessage::Beat {
        group,
        at: Instant::now(),
    });
    // The group holds each engine of its profile entries, and an engine that is not loaded has its place (3.3a).
    let mut members: Vec<Member> = jobs
        .into_iter()
        .map(|job| Member {
            job,
            loaded: false,
            handles: Vec::new(),
            retired: false,
        })
        .collect();
    let mut engines: Vec<Box<dyn Engine>> = members.iter().map(Unloaded::of).collect();
    let mut opened = Vec::with_capacity(members.len());
    // A group of more than one device has a device thread for each device, for the life of the group (spec 3.2).
    // Each resident engine with one device opens on its device thread, in the serve loop. With a resident engine with
    // stages, that thread is also the stage thread of the device, and the engine opens after the engine with stages.
    let devices = staged::group_devices(&members);
    let threads = devices.len() > 1;
    let mut device_threads = threads.then(|| staged::StageThreads::start(group, &devices, &links, interval));
    let on_thread = |member: &Member| threads && !member.staged() && member.job.devices.len() == 1;
    let mut later = Vec::new();
    // An open can take minutes, and it blocks this thread. The guard beats for the group until the open ends (7.4).
    let open_beat = OpenBeat::start(group, links.heartbeats.clone(), interval);
    for index in 0..members.len() {
        // An on-demand engine stays unloaded until a load (spec 3.3a, 3.3b).
        if members[index].job.placement.residency != Residency::Resident {
            continue;
        }
        if on_thread(&members[index]) {
            later.push(index);
            continue;
        }
        match in_place::open(
            &mut members[index].job,
            links.fd_store.as_ref(),
            (group, None),
            &links.long,
        ) {
            Ok((engine, handles, info)) => {
                engines[index] = engine;
                members[index].handles = handles;
                members[index].loaded = true;
                links.states.opened(members[index].name(), &info.identity);
                links.states.set(members[index].name(), EngineState::Ready);
                opened.push(info);
            }
            Err((kind, message)) => {
                let _ = links.events.send(Event::OpenFailed {
                    group,
                    engine: members[index].name().to_owned(),
                    kind,
                    message,
                });
                return;
            }
        }
    }
    drop(open_beat);
    let mut entry = if later.is_empty() {
        let _ = links.events.send(Event::Opened { group, engines: opened });
        staged::Entry::Serve
    } else {
        staged::Entry::Start { open: later, opened }
    };
    let names: Vec<String> = members.iter().map(|member| member.name().to_owned()).collect();
    let mut plain = Plain {
        jobs: GroupJobs::new(group as u32, names).machine(links.machine.clone()),
        cancelled: 0,
    };
    // The loads and the unloads that wait for the group to plan them, in order of arrival (spec 3.3b).
    let mut asks: VecDeque<Ask> = VecDeque::new();
    loop {
        let staged = (0..members.len()).find(|index| members[*index].loaded && members[*index].staged());
        let begin = std::mem::replace(&mut entry, staged::Entry::Serve);
        let exit = match device_threads.as_mut() {
            Some(device_threads) => staged::serve(
                group,
                staged,
                begin,
                &mut members,
                &mut engines,
                &mut asks,
                &links,
                interval,
                device_threads,
            ),
            None => serve_plain(
                group,
                &mut members,
                &mut engines,
                &mut plain,
                &mut asks,
                &links,
                interval,
            ),
        };
        match exit {
            Exit::Stop => break,
            // A swap that loads only engines with one device in a group of device threads unloads here, and the next
            // serve loop opens the engines on their device threads (spec 3.2).
            Exit::Swap(plan)
                if threads
                    && !plan.load.is_empty()
                    && plan
                        .load
                        .iter()
                        .all(|index| !members[*index].staged() && members[*index].job.devices.len() == 1) =>
            {
                for &index in &plan.unload {
                    if members[index].loaded {
                        in_place::unload(
                            index,
                            &mut members,
                            &mut engines,
                            links.store.as_ref(),
                            links.fd_store.as_ref(),
                            &links.states,
                        );
                    }
                }
                entry = staged::Entry::Swap(plan);
            }
            Exit::Swap(plan) => {
                let swap_links = in_place::SwapLinks {
                    events: &links.events,
                    heartbeats: &links.heartbeats,
                    store: links.store.as_ref(),
                    ledger: &links.ledger,
                    fd_store: links.fd_store.as_ref(),
                    states: &links.states,
                    long: &links.long,
                };
                in_place::swap(group, &mut members, &mut engines, plan, swap_links, interval);
                if links
                    .heartbeats
                    .send(WatchMessage::Beat {
                        group,
                        at: Instant::now(),
                    })
                    .is_err()
                {
                    break;
                }
            }
        }
    }
    // The engine with stages holds clones of the senders of the device threads, so it drops before the threads end.
    drop(engines);
    if let Some(device_threads) = device_threads {
        device_threads.end(group, &links);
    }
}

/// The scheduler state of a group with no stages. It lives across swaps: the jobs of an engine that stays loaded go
/// on (spec 3.3a).
struct Plain {
    jobs: GroupJobs,
    cancelled: usize,
}

/// Serve the jobs of a group with no loaded engine of stages, until the group stops or swaps (spec 3.3a).
fn serve_plain(
    group: usize,
    members: &mut [Member],
    engines: &mut [Box<dyn Engine>],
    plain: &mut Plain,
    asks: &mut VecDeque<Ask>,
    links: &GroupLinks,
    interval: Duration,
) -> Exit {
    let Plain { jobs, cancelled } = plain;
    let mut pending: Option<in_place::Plan> = None;
    let beat = || WatchMessage::Beat {
        group,
        at: Instant::now(),
    };
    loop {
        // A group with a job to run takes the messages that wait and does not sleep. An idle group waits (spec 7.4).
        let mut message = if jobs.idle() {
            links.messages.recv_timeout(interval)
        } else {
            links.messages.try_recv().map_err(|error| match error {
                TryRecvError::Empty => RecvTimeoutError::Timeout,
                TryRecvError::Disconnected => RecvTimeoutError::Disconnected,
            })
        };
        loop {
            match message {
                Ok(GroupMessage::Quantum(quantum)) => {
                    // A stop cancels each queued quantum that has not started. The running quantum went to its end
                    // before this receive (spec 10.3 step 2).
                    if links.accepting.load(Ordering::SeqCst) {
                        quantum.finish(engines);
                    } else {
                        quantum.cancel();
                        *cancelled += 1;
                    }
                    // A group with steady work never waits out the interval, so it beats after each quantum (spec 7.4).
                    if links.heartbeats.send(beat()).is_err() {
                        return Exit::Stop;
                    }
                }
                // A job that arrives after the stop began is cancelled below, before it runs.
                Ok(GroupMessage::Job(spec, parent)) => {
                    let member = members.get(spec.engine).unwrap_or_else(|| {
                        panic!(
                            "a job names engine {} of group {group}, which has {} engines",
                            spec.engine,
                            members.len()
                        )
                    });
                    // A planned swap set the states of its engines, so a job for an engine that unloads ends here.
                    match links.states.get(member.name()) {
                        Some(EngineState::Ready) => {
                            jobs.add_under(spec, parent);
                        }
                        state => refuse(spec, member.name(), state),
                    }
                }
                Ok(GroupMessage::Ask(ask)) => loads::take_ask(ask, asks, &links.ledger),
                Ok(GroupMessage::Park(reply)) => {
                    *cancelled += jobs.cancel_all();
                    // An ask that waits gets no reply. Its caller hears that the group stopped.
                    asks.clear();
                    // The drain ends here. The park budget of the group starts now.
                    let _ = reply.send(ParkProgress::Draining { ended: Instant::now() });
                    let mut report = park_loaded(members, engines, links.store.as_ref());
                    report.cancelled = *cancelled;
                    let _ = reply.send(ParkProgress::Parked(report));
                    return Exit::Stop;
                }
                Ok(GroupMessage::Stop) | Err(RecvTimeoutError::Disconnected) => return Exit::Stop,
                Ok(GroupMessage::StageDone(_)) => unreachable!("a group with no stages gets no end of a stage quantum"),
                Ok(GroupMessage::HostedDone(_))
                | Ok(GroupMessage::HostedJoinedDone(_))
                | Ok(GroupMessage::SwapStep(_)) => {
                    unreachable!("a group with no stages holds no engine on a stage thread")
                }
                Err(RecvTimeoutError::Timeout) => {
                    if jobs.idle() && links.heartbeats.send(beat()).is_err() {
                        return Exit::Stop;
                    }
                }
            }
            // Take every message that waits, so that the scheduler sees each arrival before it chooses.
            message = links.messages.try_recv().map_err(|error| match error {
                TryRecvError::Empty => RecvTimeoutError::Timeout,
                TryRecvError::Disconnected => RecvTimeoutError::Disconnected,
            });
            if matches!(message, Err(RecvTimeoutError::Timeout)) {
                break;
            }
        }
        // A stop plans no swap. The asks that waited through a swap plan the next one here (spec 3.3b).
        let serving = links.accepting.load(Ordering::SeqCst);
        if serving && pending.is_none() {
            pending = loads::next_plan(asks, members, &links.ledger, &links.states);
        }
        // The swap waits until the engines that unload hold no job. Jobs of the other engines go on (spec 3.3a).
        if serving
            && pending
                .as_ref()
                .is_some_and(|plan| plan.unload.iter().all(|index| !jobs.holds(*index)))
        {
            return Exit::Swap(pending.take().expect("a pending swap"));
        }
        if !jobs.idle() {
            if serving {
                jobs.run_one(engines);
                if links.heartbeats.send(beat()).is_err() {
                    return Exit::Stop;
                }
            } else {
                *cancelled += jobs.cancel_all();
            }
        }
    }
}

/// End a job whose engine is not `ready`. Nothing ran, and the body tells its caller the state (spec 3.3a).
fn refuse(mut spec: JobSpec, engine: &str, state: Option<EngineState>) {
    let state = state.unwrap_or_else(|| panic!("engine `{engine}` of a group has no state"));
    spec.body.unavailable(engine, state);
}

/// Park each loaded engine of a group. An engine that is not loaded holds no state (spec 3.3a).
fn park_loaded(
    members: &[Member],
    engines: &mut [Box<dyn Engine>],
    store: Option<&Sender<StoreMessage>>,
) -> ParkReport {
    let mut report = ParkReport::default();
    for (index, engine) in engines.iter_mut().enumerate() {
        if !members[index].loaded {
            continue;
        }
        report.engines += 1;
        let name = members[index].name();
        let failed = report.failures.len();
        park::park_engine(name, engine.as_mut(), store, &mut report);
        if report.failures.len() == failed {
            report.parked_engines.push(name.to_owned());
        }
    }
    report
}

/// The heartbeat of a group while its engine opens (spec 7.4).
///
/// The open blocks the group thread, so a helper thread sends the beats. The helper stops when this guard drops,
/// also when the group thread panics or returns early. A hung open is not a stall: the wait for the opens ends it.
struct OpenBeat {
    stop: Option<Sender<()>>,
    helper: Option<JoinHandle<()>>,
}

impl OpenBeat {
    fn start(group: usize, heartbeats: Sender<WatchMessage>, interval: Duration) -> Self {
        Self::beat(format!("nagaya-open-{group}"), heartbeats, interval, move |at| {
            WatchMessage::Beat { group, at }
        })
    }

    /// Beat for the stage thread of `stage` while an open blocks it (engine-abi "How a group loads both").
    fn stage(group: usize, stage: usize, heartbeats: Sender<WatchMessage>, interval: Duration) -> Self {
        Self::beat(
            format!("nagaya-open-{group}-{stage}"),
            heartbeats,
            interval,
            move |at| WatchMessage::BeatStage { group, stage, at },
        )
    }

    fn beat(
        name: String,
        heartbeats: Sender<WatchMessage>,
        interval: Duration,
        message: impl Fn(Instant) -> WatchMessage + Send + 'static,
    ) -> Self {
        let (stop, stopped) = mpsc::channel();
        let helper = thread::Builder::new()
            .name(name)
            .spawn(move || {
                // A timeout is a beat due. A message or a closed channel means the open ended.
                while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(interval) {
                    let beat = message(Instant::now());
                    if heartbeats.send(beat).is_err() {
                        return;
                    }
                }
            })
            .expect("cannot start the open heartbeat thread");
        Self {
            stop: Some(stop),
            helper: Some(helper),
        }
    }
}

impl Drop for OpenBeat {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(helper) = self.helper.take() {
            let _ = helper.join();
        }
    }
}

/// Park every engine of one group. A part that the engine cannot read or write is a failure.
fn park_group(names: &[String], engines: &mut [Box<dyn Engine>], store: Option<&Sender<StoreMessage>>) -> ParkReport {
    let mut report = ParkReport::default();
    for (index, engine) in engines.iter_mut().enumerate() {
        report.engines += 1;
        let name = names.get(index).map(String::as_str).unwrap_or("engine");
        let failed = report.failures.len();
        park::park_engine(name, engine.as_mut(), store, &mut report);
        if report.failures.len() == failed {
            report.parked_engines.push(name.to_owned());
        }
    }
    report
}

fn watchdog(messages: Receiver<WatchMessage>, events: Sender<Event>, timeout: Duration, passes: LongPasses) {
    // The key is the group, and the stage for a stage thread.
    let mut watches: BTreeMap<(usize, Option<usize>), Watch> = BTreeMap::new();
    let poll = (timeout / 4).clamp(Duration::from_millis(5), Duration::from_millis(250));
    loop {
        match messages.recv_timeout(poll) {
            Ok(WatchMessage::Start { group, devices }) => {
                watches.insert(
                    (group, None),
                    Watch {
                        devices,
                        last: Instant::now(),
                    },
                );
            }
            Ok(WatchMessage::StartStage { group, stage, devices }) => {
                watches.insert(
                    (group, Some(stage)),
                    Watch {
                        devices,
                        last: Instant::now(),
                    },
                );
            }
            Ok(WatchMessage::Beat { group, at }) => {
                if let Some(watch) = watches.get_mut(&(group, None)) {
                    watch.last = at;
                }
            }
            Ok(WatchMessage::BeatStage { group, stage, at }) => {
                if let Some(watch) = watches.get_mut(&(group, Some(stage))) {
                    watch.last = at;
                }
            }
            Ok(WatchMessage::StopStage { group, stage }) => {
                watches.remove(&(group, Some(stage)));
            }
            Ok(WatchMessage::Stop) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        let now = Instant::now();
        // The table of long passes decides for each watch with a long pass. A pass past its bound gives one
        // report (engine-abi D8).
        let judged = passes.judge(now);
        for ((group, _), bound_ms) in &judged.over_bound {
            tracing::warn!(
                group,
                bound_ms,
                "a long pass runs past the bound that its engine declared; at its deadline the process stops \
                 (spec 7.4)"
            );
        }
        // A watch with a long pass in time does not read its beats. The end of a long pass in time is a beat. Any
        // other watch reads the timeout (spec 7.4).
        let mut stalled: Vec<(usize, Option<usize>)> = watches
            .iter()
            .filter(|(key, watch)| {
                if judged.running.contains(key) {
                    return false;
                }
                let last = judged.ended.get(key).map_or(watch.last, |ended| watch.last.max(*ended));
                now.duration_since(last) > timeout
            })
            .map(|(key, _)| *key)
            .collect();
        stalled.extend(judged.late.iter().copied());
        for key in stalled {
            if let Some(watch) = watches.remove(&key) {
                let _ = events.send(Event::Stalled {
                    group: key.0,
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
    Start {
        group: usize,
        devices: Vec<DeviceId>,
    },
    Beat {
        group: usize,
        at: Instant,
    },
    /// Watch the stage thread of one stage of a group (spec 7.4). A stall of it is a stall of the group.
    StartStage {
        group: usize,
        stage: usize,
        devices: Vec<DeviceId>,
    },
    BeatStage {
        group: usize,
        stage: usize,
        at: Instant,
    },
    /// The stage thread of one stage of a group ended, so the watch of it ends too.
    StopStage {
        group: usize,
        stage: usize,
    },
    Stop,
}

enum LedgerMessage {
    Admit(PlacementProfile, Sender<Result<Admitted, AdmitError>>),
    /// Charge a new context of the named engine (spec 4.2).
    AdmitContext(String, Sender<Result<(), ContextChargeError>>),
    /// Release the charge of an ended context of the named engine.
    ReleaseContext(String, Sender<Result<(), LedgerError>>),
    /// Charge staging bytes of the named engine in one tier (spec 4.2).
    AdmitStaging(String, Tier, u64, Sender<Result<(), ContextChargeError>>),
    /// Release staging bytes in one tier.
    ReleaseStaging(Tier, u64, Sender<Result<(), LedgerError>>),
    /// Read the unassigned bytes of each budget.
    Unassigned(Sender<Vec<(Vec<Tier>, u64)>>),
    /// Take a swap that a group plans, before any unload (spec 3.3a). A refusal changes nothing.
    PlanSwap {
        load: Vec<String>,
        unload: Vec<String>,
        reply: Sender<Result<(), AdmitError>>,
    },
    /// Admit the loaded engines after a swap, when its engines that unload dropped (spec 3.3a).
    Swap {
        load: Vec<String>,
        unload: Vec<String>,
        reply: Sender<Result<Admitted, AdmitError>>,
    },
    /// Read the names of the loaded engines.
    Loaded(Sender<BTreeSet<String>>),
    /// Check a whole profile, with no change (spec 3.3).
    CheckProfile(PlacementProfile, Sender<Result<Admitted, AdmitError>>),
    /// Admit the loaded engines under a new profile (spec 3.3).
    Reprofile(PlacementProfile, Sender<Result<Admitted, AdmitError>>),
    /// Check a profile change and begin it, or give busy (spec 3.3a).
    BeginChange(PlacementProfile, Sender<Result<Admitted, AdmitError>>),
    /// End a profile change that was not kept.
    EndChange,
    /// Tell if a profile change runs, so that a group refuses a load at once (spec 3.3a).
    Changing(Sender<bool>),
    /// Take a new profile when no swap runs and the gone engines are free, or give `None` (spec 3.3a).
    TryReprofile(
        PlacementProfile,
        Vec<String>,
        Sender<Result<Result<Admitted, String>, AdmitError>>,
    ),
    Stop,
}

/// What a group tells the stop while it parks.
enum ParkProgress {
    /// The queued quanta are cancelled and the running quantum ended. The park starts at `ended`.
    Draining { ended: Instant },
    /// The park ended.
    Parked(ParkReport),
}

/// How the wait for one drain ended.
enum Drain {
    /// The drain ended at this time.
    Ended(Instant),
    /// The group answered no further, or the group thread ended.
    Gone,
    /// The stop ends now. The flag is true for a second signal.
    Stopped(bool),
}

/// A request to one group thread.
enum GroupMessage {
    /// Run one quantum to its end.
    Quantum(Quantum),
    /// Add a job. The scheduler of the group chooses when its quanta run (spec 5).
    ///
    /// The span is the one that was current when the request submitted the job. It is the parent of the job's span.
    Job(JobSpec, Span),
    /// Park every engine, then stop the thread (spec 10.3).
    Park(Sender<ParkProgress>),
    /// Stop the thread without a park. `Process::drop` uses it.
    Stop,
    /// A stage thread ended a stage quantum (engine-abi D1).
    StageDone(staged::StageDone),
    /// A stage thread ended a quantum of an engine that it holds (engine-abi "How a group loads both").
    HostedDone(staged::HostedDone),
    /// A device thread ended a joined decode quantum of the jobs of engines that it holds (engine-abi "LFM3",
    /// "Batched passes").
    HostedJoinedDone(staged::HostedJoinedDone),
    /// A stage or device thread ended one unload or one open of a swap (spec 3.3a, "Single-card swaps").
    SwapStep(staged::SwapStep),
    /// Load or unload one engine of the group by a swap (spec 3.3b).
    Ask(Ask),
}

/// Why a context charge did not happen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextChargeError {
    /// The process is stopping and takes no new work (spec 10.3).
    Stopping,
    /// The ledger refused, or the call was wrong.
    Ledger(LedgerError),
    /// An accepted profile change removes the engine, so it takes no new charge (spec 3.3a).
    Leaving(String),
}

impl std::fmt::Display for ContextChargeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextChargeError::Stopping => write!(formatter, "the process is stopping"),
            ContextChargeError::Ledger(error) => write!(formatter, "{error}"),
            ContextChargeError::Leaving(engine) => {
                write!(
                    formatter,
                    "engine `{engine}` leaves the profile, so it takes no new charge"
                )
            }
        }
    }
}

impl std::error::Error for ContextChargeError {}

/// The ways into the process that a service needs: the ledger, the groups and the store.
///
/// A copy holds only channels, so it can outlive a borrow of the [`Process`]. A stop makes each call refuse.
#[derive(Clone)]
pub struct Ports {
    ledger: Sender<LedgerMessage>,
    groups: Vec<Sender<GroupMessage>>,
    /// The engine names of each group, in profile order.
    group_engines: Arc<Vec<Vec<String>>>,
    store: Option<Arc<Mutex<Store>>>,
    accepting: Arc<AtomicBool>,
    /// The first failure that a service saw outside a call. The main loop reads it (spec 8.5).
    failure: Arc<Mutex<Option<String>>>,
    /// The state of each engine (spec 3.3b).
    states: EngineStates,
    /// The released devices, each with the engines that it holds (spec 3.3b, "The release of a device").
    released: Released,
    /// The long passes that run now (engine-abi D8, spec 7.4).
    long_passes: LongPasses,
}

impl Ports {
    /// Record a failed release of a ledger charge. The ledger is wrong now, so the process must stop (spec 4.2, 8.5).
    ///
    /// A stopped ledger has no charge left to release, so a stop is not a failure. The first failure stays.
    pub fn release_failed(&self, what: &str, error: &ContextChargeError) {
        if matches!(error, ContextChargeError::Stopping) {
            return;
        }
        let mut failure = self.failure.lock().expect("the failure lock is poisoned");
        if failure.is_none() {
            let message = format!("the release of {what} failed: {error}");
            tracing::error!("{message}; the process stops (spec 8.5)");
            *failure = Some(message);
        }
    }

    /// The long passes that run now, which the route `busy` of a device shows (spec 7.4).
    pub fn long_passes(&self) -> LongPasses {
        self.long_passes.clone()
    }

    /// The first failure that a service saw outside a call, or `None`.
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().expect("the failure lock is poisoned").clone()
    }

    /// Charge a new context of an engine with the ledger, and wait for the answer (spec 4.2, 4.3).
    ///
    /// A refusal changes nothing. The engine name is the name in the profile.
    pub fn admit_context(&self, engine: &str) -> Result<(), ContextChargeError> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(ContextChargeError::Stopping);
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::AdmitContext(engine.to_owned(), reply_tx))
            .map_err(|_| ContextChargeError::Stopping)?;
        reply_rx.recv().map_err(|_| ContextChargeError::Stopping)?
    }

    /// Release the charge of an ended context. A release must follow each charge, also after a failure.
    pub fn release_context(&self, engine: &str) -> Result<(), ContextChargeError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::ReleaseContext(engine.to_owned(), reply_tx))
            .map_err(|_| ContextChargeError::Stopping)?;
        reply_rx
            .recv()
            .map_err(|_| ContextChargeError::Stopping)?
            .map_err(ContextChargeError::Ledger)
    }

    /// Charge staging bytes of an engine with the ledger, and wait for the answer (spec 4.2, 4.3).
    ///
    /// Staging is a copy of engine state in host memory. A refusal changes nothing. A release must follow each
    /// charge, also after a failure.
    pub fn admit_staging(&self, engine: &str, tier: Tier, bytes: u64) -> Result<(), ContextChargeError> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(ContextChargeError::Stopping);
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::AdmitStaging(engine.to_owned(), tier, bytes, reply_tx))
            .map_err(|_| ContextChargeError::Stopping)?;
        reply_rx.recv().map_err(|_| ContextChargeError::Stopping)?
    }

    /// Release staging bytes that `admit_staging` charged.
    pub fn release_staging(&self, tier: Tier, bytes: u64) -> Result<(), ContextChargeError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::ReleaseStaging(tier, bytes, reply_tx))
            .map_err(|_| ContextChargeError::Stopping)?;
        reply_rx
            .recv()
            .map_err(|_| ContextChargeError::Stopping)?
            .map_err(ContextChargeError::Ledger)
    }

    /// The bytes that no term holds, for each budget. New contexts take from them (spec 4.4 step 4).
    pub fn unassigned(&self) -> Vec<(Vec<Tier>, u64)> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::Unassigned(reply_tx))
            .expect("the ledger thread stopped");
        reply_rx.recv().expect("the ledger thread stopped")
    }

    /// Send one job to a group thread, as [`Process::submit_job`] does.
    pub fn submit_job(&self, group: usize, job: JobSpec) -> Result<(), SubmitError> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(SubmitError::Stopping);
        }
        let Some(stop) = self.groups.get(group) else {
            return Err(SubmitError::UnknownGroup);
        };
        stop.send(GroupMessage::Job(job, Span::current()))
            .map_err(|_| SubmitError::Stopped)
    }

    /// The group and the place in the group of the engine with this name in the profile.
    pub fn locate(&self, engine: &str) -> Option<(usize, usize)> {
        self.group_engines
            .iter()
            .enumerate()
            .find_map(|(group, names)| names.iter().position(|name| name == engine).map(|index| (group, index)))
    }

    /// The machine's store, when the process has one. Run `lookup` and `restore` on a group thread (spec 11.3c).
    pub fn store(&self) -> Option<&Arc<Mutex<Store>>> {
        self.store.as_ref()
    }

    /// The state of the engine with this name, or `None` for a name that is not in the profile (spec 3.3b).
    pub fn engine_state(&self, engine: &str) -> Option<EngineState> {
        self.states.get(engine)
    }

    /// The identity of an engine at its last open since the start, or `None` before its first open.
    pub fn engine_identity(&self, engine: &str) -> Option<Identity> {
        self.states.identity(engine)
    }

    /// The count of the opens of an engine since the start. The group thread counts an open before the engine runs a
    /// job, so a job on the engine reads the count of its own open.
    pub fn engine_opens(&self, engine: &str) -> u64 {
        self.states.opens(engine)
    }

    /// Load an engine, as [`Process::load`] does.
    pub fn load(&self, engine: &str) -> Result<LoadReply, LoadError> {
        self.load_with(engine, &[])
    }

    /// Load an engine after the engines of `unload` unload, as [`Process::load_with`] does.
    pub fn load_with(&self, engine: &str, unload: &[String]) -> Result<LoadReply, LoadError> {
        let released = self.released.lock().expect("the released devices are poisoned");
        let held = released
            .iter()
            .find(|(_, names)| names.iter().any(|name| name == engine));
        if let Some((&device, _)) = held {
            return Err(LoadError::Released {
                engine: engine.to_owned(),
                device,
            });
        }
        drop(released);
        self.ask(engine, loads::Asking::Load(unload.to_vec()))
    }

    /// Release a device (spec 3.3b, "The release of a device"): unload each loaded engine of `profile` that names it,
    /// and refuse each load of such an engine until [`Ports::reclaim_device`]. `profile` is the active profile.
    ///
    /// A device with a loaded pinned engine is refused, and nothing changes. An unload that a group refuses ends the
    /// release: the device is not released, and the engines before it stay unloaded.
    pub fn release_device(
        &self,
        device: nagaya_abi::DeviceId,
        profile: &PlacementProfile,
    ) -> Result<DeviceReply, LoadError> {
        let on_device: Vec<&EnginePlacement> = profile
            .engines
            .iter()
            .filter(|entry| entry.devices.contains(&device))
            .collect();
        if on_device.is_empty() {
            return Err(LoadError::NoDevice(device));
        }
        let loaded: Vec<&EnginePlacement> = on_device
            .iter()
            .copied()
            .filter(|entry| !matches!(self.states.get(&entry.name), None | Some(EngineState::Unloaded)))
            .collect();
        if let Some(pinned) = loaded.iter().find(|entry| entry.pinned) {
            return Err(LoadError::Pinned(pinned.name.clone()));
        }
        let names: Vec<String> = on_device.iter().map(|entry| entry.name.clone()).collect();
        self.released
            .lock()
            .expect("the released devices are poisoned")
            .insert(device, names);
        let mut unloaded = Vec::new();
        for entry in &loaded {
            match self.unload(&entry.name) {
                Ok(LoadReply::Started { .. }) => unloaded.push(entry.name.clone()),
                Ok(LoadReply::Already(_)) => {}
                Err(error) => {
                    self.released
                        .lock()
                        .expect("the released devices are poisoned")
                        .remove(&device);
                    return Err(error);
                }
            }
        }
        Ok(if unloaded.is_empty() {
            DeviceReply::Already
        } else {
            DeviceReply::Started(unloaded)
        })
    }

    /// End the release of a device, and load each resident engine of `profile` that names it and that is not loaded
    /// (spec 3.3b). An engine on demand stays unloaded. A device that is not released changes nothing.
    ///
    /// A load that a group refuses ends the reclaim: the device stays released, and the engines before it stay loaded.
    pub fn reclaim_device(
        &self,
        device: nagaya_abi::DeviceId,
        profile: &PlacementProfile,
    ) -> Result<DeviceReply, LoadError> {
        if !profile.engines.iter().any(|entry| entry.devices.contains(&device)) {
            return Err(LoadError::NoDevice(device));
        }
        let held = self
            .released
            .lock()
            .expect("the released devices are poisoned")
            .remove(&device);
        let Some(held) = held else {
            return Ok(DeviceReply::Already);
        };
        let mut loaded = Vec::new();
        for entry in &profile.engines {
            let resident = entry.residency == crate::profile::Residency::Resident;
            if !entry.devices.contains(&device) || !resident {
                continue;
            }
            match self.load(&entry.name) {
                Ok(LoadReply::Started { .. }) => loaded.push(entry.name.clone()),
                Ok(LoadReply::Already(_)) => {}
                Err(error) => {
                    // The device stays released, so the next reclaim tries the load again.
                    self.released
                        .lock()
                        .expect("the released devices are poisoned")
                        .insert(device, held);
                    return Err(error);
                }
            }
        }
        Ok(if loaded.is_empty() {
            DeviceReply::Already
        } else {
            DeviceReply::Started(loaded)
        })
    }

    /// Unload an engine, as [`Process::unload`] does.
    pub fn unload(&self, engine: &str) -> Result<LoadReply, LoadError> {
        self.ask(engine, loads::Asking::Unload)
    }

    /// Unload an engine that a new profile lacks, and keep it unloaded until the next start (spec 3.3). A resident
    /// engine retires too.
    pub fn retire(&self, engine: &str) -> Result<LoadReply, LoadError> {
        self.ask(engine, loads::Asking::Retire)
    }

    /// Give an engine the residency, the pin and the share of its new profile entry (spec 3.3). An engine that becomes
    /// resident loads when it fits beside the loaded engines.
    pub fn place(&self, entry: &EnginePlacement) -> Result<LoadReply, LoadError> {
        self.ask(&entry.name, loads::Asking::Place(Box::new(entry.clone())))
    }

    fn ask(&self, engine: &str, asking: loads::Asking) -> Result<LoadReply, LoadError> {
        let accepting = self.accepting.load(Ordering::SeqCst);
        loads::ask(&self.groups, &self.group_engines, accepting, engine, asking)
    }

    /// Check a whole profile, with no change, as [`Process::check_profile`] does.
    pub fn check_profile(&self, profile: &PlacementProfile) -> Result<Admitted, AdmitError> {
        let (reply, answer) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::CheckProfile(profile.clone(), reply))
            .expect("the ledger thread stopped");
        answer.recv().expect("the ledger thread stopped")
    }

    /// The names of the engines that the ledger counts as loaded, or `None` when the ledger thread stopped.
    pub fn loaded_engines(&self) -> Option<BTreeSet<String>> {
        let (reply, answer) = mpsc::channel();
        self.ledger.send(LedgerMessage::Loaded(reply)).ok()?;
        answer.recv().ok()
    }

    /// Run `body`, and stop the process when it panics (spec 8.5). A thread outside the group threads uses it.
    pub fn guard(&self, thread: &str, body: impl FnOnce()) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(body)) {
            self.fail(format!(
                "the thread {thread} panicked: {}",
                panic_text(payload.as_ref())
            ));
        }
    }

    /// Record a failure outside a call. The process must stop (spec 8.5). The first failure stays.
    pub fn fail(&self, message: String) {
        let mut failure = self.failure.lock().expect("the failure lock is poisoned");
        if failure.is_none() {
            tracing::error!("{message}; the process stops (spec 8.5)");
            *failure = Some(message);
        }
    }

    /// Admit the loaded engines under a new profile (spec 3.3). The engines that it lacks must be unloaded first.
    pub fn reprofile(&self, profile: &PlacementProfile) -> Result<Admitted, AdmitError> {
        let (reply, answer) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::Reprofile(profile.clone(), reply))
            .expect("the ledger thread stopped");
        answer.recv().expect("the ledger thread stopped")
    }

    /// Check a profile change and begin it, or give busy when a swap or another change runs (spec 3.3a).
    pub fn begin_change(&self, profile: &PlacementProfile) -> Result<Admitted, AdmitError> {
        let (reply, answer) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::BeginChange(profile.clone(), reply))
            .expect("the ledger thread stopped");
        answer.recv().expect("the ledger thread stopped")
    }

    /// End a profile change that was begun and not kept.
    pub fn end_change(&self) {
        let _ = self.ledger.send(LedgerMessage::EndChange);
    }

    /// Take a new profile when no swap runs and the engines of `gone` dropped and hold no context. `Ok(Err(why))`
    /// names what it waits for, and `None` means the ledger thread stopped.
    pub fn try_reprofile(
        &self,
        profile: &PlacementProfile,
        gone: &[String],
    ) -> Option<Result<Result<Admitted, String>, AdmitError>> {
        let (reply, answer) = mpsc::channel();
        self.ledger
            .send(LedgerMessage::TryReprofile(profile.clone(), gone.to_vec(), reply))
            .ok()?;
        answer.recv().ok()
    }
}

/// The started process threads.
#[derive(Debug)]
pub struct Process {
    ledger: Sender<LedgerMessage>,
    ledger_thread: Option<JoinHandle<()>>,
    watchdog_stop: Option<Sender<WatchMessage>>,
    watchdog_thread: Option<JoinHandle<()>>,
    store: Option<Sender<StoreMessage>>,
    /// The store, shared with the group threads. A lookup and a restore need an engine, which lives on its group
    /// thread (spec 11.3c).
    store_handle: Option<Arc<Mutex<Store>>>,
    store_thread: Option<JoinHandle<()>>,
    groups: Vec<JoinHandle<()>>,
    group_stops: Vec<Sender<GroupMessage>>,
    events: Receiver<Event>,
    group_count: usize,
    /// The engine names of each group, in profile order.
    group_engines: Vec<Vec<String>>,
    accepting: Arc<AtomicBool>,
    /// The first failure that a service saw outside a call. [`Ports::release_failed`] sets it.
    failure: Arc<Mutex<Option<String>>>,
    /// The state of each engine (spec 3.3b).
    states: EngineStates,
    /// The released devices. The process keeps them in memory only, so a start releases none (spec 3.3b).
    released: Released,
    /// The long passes that run now (engine-abi D8).
    long_passes: LongPasses,
    stopped: bool,
}

/// The released devices, each with the names of the engines that the profile named on it at the release.
type Released = Arc<Mutex<BTreeMap<nagaya_abi::DeviceId, Vec<String>>>>;

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

    /// True when the process takes no new admission, because it stops (spec 10.3 step 1).
    pub fn stopping(&self) -> bool {
        !self.accepting.load(Ordering::SeqCst)
    }

    /// Send one quantum to a group thread. A stop refuses every later quantum (spec 10.3 step 1).
    ///
    /// The group runs the quantum to its end before it parks. A stop cancels a quantum that has not started.
    pub fn submit(&self, group: usize, quantum: Quantum) -> Result<(), SubmitError> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(SubmitError::Stopping);
        }
        let Some(stop) = self.group_stops.get(group) else {
            return Err(SubmitError::UnknownGroup);
        };
        stop.send(GroupMessage::Quantum(quantum))
            .map_err(|_| SubmitError::Stopped)
    }

    /// The ways into the process for a service. A copy can outlive this borrow.
    pub fn ports(&self) -> Ports {
        Ports {
            ledger: self.ledger.clone(),
            groups: self.group_stops.clone(),
            group_engines: Arc::new(self.group_engines.clone()),
            store: self.store_handle.clone(),
            accepting: Arc::clone(&self.accepting),
            failure: Arc::clone(&self.failure),
            states: self.states.clone(),
            released: Arc::clone(&self.released),
            long_passes: self.long_passes.clone(),
        }
    }

    /// Release a device, as [`Ports::release_device`] does.
    pub fn release_device(
        &self,
        device: nagaya_abi::DeviceId,
        profile: &PlacementProfile,
    ) -> Result<DeviceReply, LoadError> {
        self.ports().release_device(device, profile)
    }

    /// Reclaim a device, as [`Ports::reclaim_device`] does.
    pub fn reclaim_device(
        &self,
        device: nagaya_abi::DeviceId,
        profile: &PlacementProfile,
    ) -> Result<DeviceReply, LoadError> {
        self.ports().reclaim_device(device, profile)
    }

    /// Load the engine with this name by a swap on its group thread (spec 3.3b). It waits until the group plans the
    /// swap, not for the swap to end.
    pub fn load(&self, engine: &str) -> Result<LoadReply, LoadError> {
        self.ports().load(engine)
    }

    /// Load the engine with this name after the engines of `unload` unload, in one swap (spec 3.3a, "Explicit
    /// layout"). A swap that does not fit is a refusal, and nothing changes.
    pub fn load_with(&self, engine: &str, unload: &[String]) -> Result<LoadReply, LoadError> {
        self.ports().load_with(engine, unload)
    }

    /// Unload the engine with this name (spec 3.3b). No other engine loads.
    pub fn unload(&self, engine: &str) -> Result<LoadReply, LoadError> {
        self.ports().unload(engine)
    }

    /// Check a whole profile with the registry and the inventory of the process, with no change (spec 3.3).
    pub fn check_profile(&self, profile: &PlacementProfile) -> Result<Admitted, AdmitError> {
        self.ports().check_profile(profile)
    }

    /// Check a profile change and begin it, or give busy (spec 3.3a).
    pub fn begin_change(&self, profile: &PlacementProfile) -> Result<Admitted, AdmitError> {
        self.ports().begin_change(profile)
    }

    /// End a profile change that was begun and not kept.
    pub fn end_change(&self) {
        self.ports().end_change();
    }

    /// Send one job to a group thread. A stop refuses every later job (spec 10.3 step 1).
    ///
    /// The scheduler of the group chooses when the quanta of the job run, by class and priority (spec 5.3, 5.4).
    pub fn submit_job(&self, group: usize, job: JobSpec) -> Result<(), SubmitError> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(SubmitError::Stopping);
        }
        let Some(stop) = self.group_stops.get(group) else {
            return Err(SubmitError::UnknownGroup);
        };
        stop.send(GroupMessage::Job(job, Span::current()))
            .map_err(|_| SubmitError::Stopped)
    }

    /// Wait for every group to open its engines, or for a failure.
    pub fn wait_opens(&mut self, timeout: Duration) -> Result<Vec<OpenedEngine>, WaitError> {
        self.wait_opens_with(timeout, timeout, || {})
    }

    /// Wait as `wait_opens` does, and call `tick` each time `every` passes with no event (spec 10.1, 7.4).
    ///
    /// The caller uses the tick to tell systemd that a long open goes on. The tick does not run after the timeout.
    pub fn wait_opens_with(
        &mut self,
        timeout: Duration,
        every: Duration,
        mut tick: impl FnMut(),
    ) -> Result<Vec<OpenedEngine>, WaitError> {
        let deadline = Instant::now() + timeout;
        let mut opened = Vec::new();
        let mut done = vec![false; self.group_count];
        while done.iter().any(|&finished| !finished) {
            let left = deadline.saturating_duration_since(Instant::now());
            let event = if left.is_zero() {
                Err(RecvTimeoutError::Timeout)
            } else {
                self.events.recv_timeout(left.min(every))
            };
            match event {
                Ok(Event::Opened { group, engines }) => {
                    done[group] = true;
                    opened.extend(engines);
                }
                Ok(event) => return Err(WaitError::Event(event)),
                Err(RecvTimeoutError::Timeout) if Instant::now() < deadline => tick(),
                Err(RecvTimeoutError::Timeout) => {
                    let waiting = (0..self.group_count)
                        .filter(|&group| !done[group])
                        .flat_map(|group| self.group_engines[group].iter().cloned())
                        .collect();
                    return Err(WaitError::Timeout { waiting });
                }
                Err(RecvTimeoutError::Disconnected) => return Err(WaitError::Disconnected),
            }
        }
        Ok(opened)
    }

    /// Read the next event, or give `None` when the timeout passes.
    pub fn recv(&self, timeout: Duration) -> Option<Event> {
        self.events.recv_timeout(timeout).ok()
    }

    /// Stop the process (spec 10.3). Refuse new work, drain each group, park it in the budget, then stop the threads.
    ///
    /// A group cancels its queued quanta and lets the running quantum end. The park budget of a group starts when
    /// its drain ends, so a long quantum does not use it. A drain that never ends is a stall, which the watchdog
    /// reports (spec 7.4); the stop then ends with a crash. A second signal stops the park at once.
    pub fn stop(&mut self, budget: Duration, signals: &SignalFlag) -> StopReport {
        if self.stopped {
            return StopReport::default();
        }
        self.accepting.store(false, Ordering::SeqCst);
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
            let ended = match self.wait_drain(&reply, signals, entered, &mut parked) {
                Drain::Ended(ended) => ended,
                Drain::Gone => {
                    complete = false;
                    continue;
                }
                Drain::Stopped(interrupted) => {
                    self.finish_stop();
                    return StopReport {
                        parked,
                        complete: false,
                        interrupted,
                    };
                }
            };
            let deadline = ended + budget;
            loop {
                if signals.count() != entered {
                    self.finish_stop();
                    return StopReport {
                        parked,
                        complete: false,
                        interrupted: true,
                    };
                }
                let left = deadline.saturating_duration_since(Instant::now());
                // With no budget left, read only the replies that wait in the channel. A group that finished its
                // park while an earlier group used the budget has its report there.
                let reply_now = if left.is_zero() {
                    reply.try_recv().map_err(|error| match error {
                        TryRecvError::Empty => RecvTimeoutError::Timeout,
                        TryRecvError::Disconnected => RecvTimeoutError::Disconnected,
                    })
                } else {
                    reply.recv_timeout(left.min(STOP_POLL))
                };
                match reply_now {
                    Ok(ParkProgress::Parked(report)) => {
                        parked.merge(report);
                        break;
                    }
                    Ok(ParkProgress::Draining { .. }) => {}
                    Err(RecvTimeoutError::Timeout) if left.is_zero() => {
                        complete = false;
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
        match self.park_pinned(budget, signals, entered, &mut parked) {
            Some(parked_all) => complete &= parked_all,
            None => {
                self.finish_stop();
                return StopReport {
                    parked,
                    complete: false,
                    interrupted: true,
                };
            }
        }
        self.finish_stop();
        StopReport {
            parked,
            complete,
            interrupted: false,
        }
    }

    /// Park the pinned entries of the host tier after the groups park, so a pin stays through a restart (spec 6.11).
    ///
    /// The wait is at most one park budget. It gives false when the store did not answer in it, and `None` when a
    /// second signal came. A park that failed is in `parked.failures`.
    fn park_pinned(
        &self,
        budget: Duration,
        signals: &SignalFlag,
        entered: usize,
        parked: &mut ParkReport,
    ) -> Option<bool> {
        let Some(store) = &self.store else {
            return Some(true);
        };
        let (reply, answer) = mpsc::channel();
        let message = StoreMessage::ParkPinned {
            now_ms: park::now_ms(),
            reply,
        };
        if store.send(message).is_err() {
            parked
                .failures
                .push("the pinned entries were not parked: the store thread stopped".to_owned());
            return Some(true);
        }
        let deadline = Instant::now() + budget;
        loop {
            if signals.count() != entered {
                return None;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match answer.recv_timeout(left.min(STOP_POLL)) {
                Ok(report) => {
                    parked.pinned += report.parked.len();
                    parked.pinned_bytes += report.bytes;
                    parked.failures.extend(report.failures);
                    return Some(true);
                }
                Err(RecvTimeoutError::Timeout) if left.is_zero() => return Some(false),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    parked
                        .failures
                        .push("the pinned entries were not parked: the store thread stopped".to_owned());
                    return Some(true);
                }
            }
        }
    }

    /// Wait for a group to end its drain. A stall or a panic ends the wait: the drain cannot end (spec 7.4, 8.5).
    fn wait_drain(
        &self,
        reply: &Receiver<ParkProgress>,
        signals: &SignalFlag,
        entered: usize,
        parked: &mut ParkReport,
    ) -> Drain {
        loop {
            if signals.count() != entered {
                return Drain::Stopped(true);
            }
            while let Ok(event) = self.events.try_recv() {
                match event {
                    Event::Stalled { group, .. } => {
                        parked
                            .failures
                            .push(format!("group {group} stalled while it drained its quanta"));
                        return Drain::Stopped(false);
                    }
                    Event::Panicked { thread, message } => {
                        parked.failures.push(format!("the {thread} thread panicked: {message}"));
                        return Drain::Stopped(false);
                    }
                    Event::Opened { .. } | Event::OpenFailed { .. } | Event::Swapped { .. } => {}
                }
            }
            match reply.recv_timeout(STOP_POLL) {
                Ok(ParkProgress::Draining { ended }) => return Drain::Ended(ended),
                Ok(ParkProgress::Parked(_)) => unreachable!("a group parks only after it reports the end of its drain"),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Drain::Gone,
            }
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
