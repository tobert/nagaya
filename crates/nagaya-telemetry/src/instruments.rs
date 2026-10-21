//! The instruments of the nagaya (spec 9.4), all in one place.
//!
//! Each instrument has one method here. The method takes bounded labels, never an id of a context, a trace or a job.
//! Spec 9.5: define an instrument only with its call site. A method here that no crate calls is a defect.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter};

/// The name of the meter. It is the `service.name` of the nagaya (spec 9.1).
pub const METER_NAME: &str = "nagaya";

/// A place where the ledger or the store keeps bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierLabel {
    /// Host memory.
    Host,
    /// NVMe.
    Nvme,
    /// Device memory of the device with this id.
    Device(u32),
}

impl TierLabel {
    /// The attribute value: `host`, `nvme` or `device<id>`. A machine has few devices, so the set is bounded.
    pub fn value(self) -> String {
        match self {
            TierLabel::Host => "host".to_owned(),
            TierLabel::Nvme => "nvme".to_owned(),
            TierLabel::Device(id) => format!("device{id}"),
        }
    }
}

/// The attribute value for the tiers of one budget: the values in sorted order, joined with `+`.
///
/// A budget can cover more than one tier (spec 4.1). The set is fixed by the profile, so it is bounded.
pub fn tier_set(tiers: &[TierLabel]) -> String {
    let mut values: Vec<String> = tiers.iter().map(|tier| tier.value()).collect();
    values.sort();
    values.join("+")
}

/// Why the ledger refused (spec 9.4). The ledger has the gate `budget` only. The gates `floor` (4.7) and `resize` (4.5)
/// come with the code that refuses at them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalGate {
    /// A term does not fit its budget.
    Budget,
}

impl RefusalGate {
    fn value(self) -> &'static str {
        match self {
            RefusalGate::Budget => "budget",
        }
    }
}

/// How a store lookup ended (spec 9.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupOutcome {
    /// The store served a chain with no approximate entry.
    Exact,
    /// The store served a chain with an approximate entry.
    Approx,
    /// The store held no chain for the recipe.
    Miss,
    /// The recipe is refused after a tripwire (spec 6.7).
    Refused,
}

impl LookupOutcome {
    fn value(self) -> &'static str {
        match self {
            LookupOutcome::Exact => "exact",
            LookupOutcome::Approx => "approx",
            LookupOutcome::Miss => "miss",
            LookupOutcome::Refused => "refused",
        }
    }
}

/// Why the store removed an entry (spec 9.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictReason {
    /// A call to `evict` freed room for an owner (spec 6.9).
    Evict,
    /// A `put` freed room for its new entry (spec 6.9).
    PutRoom,
    /// A `put` of an interval copy went over the cap (spec 6.13).
    IntervalCap,
    /// A newer interval copy replaced this one (spec 6.13).
    IntervalReplaced,
}

impl EvictReason {
    fn value(self) -> &'static str {
        match self {
            EvictReason::Evict => "evict",
            EvictReason::PutRoom => "put_room",
            EvictReason::IntervalCap => "interval_cap",
            EvictReason::IntervalReplaced => "interval_replaced",
        }
    }
}

/// How a store restore ended (spec 6.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreResult {
    /// Every part was written.
    Done,
    /// The engine refused the first write, so the state is as it was.
    Refused,
    /// A write failed, so the state is poisoned.
    Failed,
    /// The store refused before any write: a quarantine, a missing parent or a bad entry.
    Error,
}

impl RestoreResult {
    fn value(self) -> &'static str {
        match self {
            RestoreResult::Done => "done",
            RestoreResult::Refused => "refused",
            RestoreResult::Failed => "failed",
            RestoreResult::Error => "error",
        }
    }
}

/// The state of an engine (spec 10.1, 2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineState {
    /// The engine is opening.
    Loading,
    /// The engine opened and serves work.
    Ready,
    /// The engine takes no new work, and its running quantum ends.
    Draining,
    /// The engine closed its state.
    Unloaded,
    /// The engine failed.
    Fatal,
}

impl EngineState {
    /// Every state, in the order of a life.
    pub const ALL: [EngineState; 5] = [
        EngineState::Loading,
        EngineState::Ready,
        EngineState::Draining,
        EngineState::Unloaded,
        EngineState::Fatal,
    ];

    /// The word that `STATUS=` and the attribute `state` use.
    pub fn as_str(self) -> &'static str {
        match self {
            EngineState::Loading => "loading",
            EngineState::Ready => "ready",
            EngineState::Draining => "draining",
            EngineState::Unloaded => "unloaded",
            EngineState::Fatal => "fatal",
        }
    }
}

/// The labels of a quantum metric (spec 9.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantumLabels<'a> {
    /// The engine name.
    pub engine: &'a str,
    /// The device group.
    pub device_group: u32,
    /// The quantum kind, for example `prefill`.
    pub kind: &'static str,
    /// The quantum class, for example `interactive`.
    pub class: &'static str,
}

impl QuantumLabels<'_> {
    fn attributes(&self) -> [KeyValue; 4] {
        [
            KeyValue::new("engine", self.engine.to_owned()),
            KeyValue::new("device_group", i64::from(self.device_group)),
            KeyValue::new("kind", self.kind),
            KeyValue::new("class", self.class),
        ]
    }
}

/// The instruments of the nagaya. Build them once from a meter.
#[derive(Clone, Debug)]
pub struct Instruments {
    quantum_wait: Histogram<f64>,
    quantum_run: Histogram<f64>,
    ledger_bytes: Gauge<u64>,
    ledger_refusals: Counter<u64>,
    store_bytes: Gauge<u64>,
    store_entries: Gauge<u64>,
    store_lookups: Counter<u64>,
    store_evictions: Counter<u64>,
    store_restores: Counter<u64>,
    engine_state: Gauge<u64>,
    http_duration: Histogram<f64>,
}

impl Instruments {
    /// Build the instruments on `meter`.
    pub fn new(meter: &Meter) -> Self {
        Self {
            quantum_wait: meter
                .f64_histogram("nagaya.quantum.wait.duration")
                .with_unit("ms")
                .with_description("The time from ready to start of a quantum.")
                .build(),
            quantum_run: meter
                .f64_histogram("nagaya.quantum.run.duration")
                .with_unit("ms")
                .with_description("The time that a quantum runs on its engine.")
                .build(),
            ledger_bytes: meter
                .u64_gauge("nagaya.ledger.bytes")
                .with_unit("By")
                .with_description("The bytes that the ledger gave to a term of an engine.")
                .build(),
            ledger_refusals: meter
                .u64_counter("nagaya.ledger.refusals")
                .with_unit("{refusal}")
                .with_description("The admissions that the ledger refused.")
                .build(),
            store_bytes: meter
                .u64_gauge("nagaya.store.bytes")
                .with_unit("By")
                .with_description("The bytes of the entries that the store holds.")
                .build(),
            store_entries: meter
                .u64_gauge("nagaya.store.entries")
                .with_unit("{entry}")
                .with_description("The entries that the store holds.")
                .build(),
            store_lookups: meter
                .u64_counter("nagaya.store.lookups")
                .with_unit("{lookup}")
                .with_description("The store lookups by outcome.")
                .build(),
            store_evictions: meter
                .u64_counter("nagaya.store.evictions")
                .with_unit("{entry}")
                .with_description("The entries that the store removed.")
                .build(),
            store_restores: meter
                .u64_counter("nagaya.store.restores")
                .with_unit("{restore}")
                .with_description("The store restores by result.")
                .build(),
            engine_state: meter
                .u64_gauge("nagaya.engine.state")
                .with_unit("1")
                .with_description("One for the state that an engine is in, and zero for the other states.")
                .build(),
            http_duration: meter
                .f64_histogram("http.server.request.duration")
                .with_unit("s")
                .with_description("The time from the start of an HTTP request to its response head.")
                .build(),
        }
    }

    /// Instruments that record nothing.
    pub fn noop() -> Self {
        // The nagaya never sets the global meter provider, so the global meter records nothing.
        Self::new(&opentelemetry::global::meter(METER_NAME))
    }

    /// Record the wait of a quantum, in milliseconds (spec 5, 9.4).
    pub fn quantum_wait(&self, labels: &QuantumLabels<'_>, milliseconds: f64) {
        self.quantum_wait.record(milliseconds, &labels.attributes());
    }

    /// Record the run time of a quantum, in milliseconds (spec 9.4).
    pub fn quantum_run(&self, labels: &QuantumLabels<'_>, milliseconds: f64) {
        self.quantum_run.record(milliseconds, &labels.attributes());
    }

    /// Record the bytes that the ledger gave to one term of one engine (spec 4.4).
    pub fn ledger_bytes(&self, tier: TierLabel, engine: &str, term: &str, bytes: u64) {
        self.ledger_bytes.record(
            bytes,
            &[
                KeyValue::new("tier", tier.value()),
                KeyValue::new("engine", engine.to_owned()),
                KeyValue::new("term", term.to_owned()),
            ],
        );
    }

    /// Count one refusal of the ledger (spec 4.3). `tiers` are the tiers of the budget that was short.
    pub fn ledger_refusal(&self, tiers: &[TierLabel], engine: &str, gate: RefusalGate) {
        self.ledger_refusals.add(
            1,
            &[
                KeyValue::new("tier", tier_set(tiers)),
                KeyValue::new("engine", engine.to_owned()),
                KeyValue::new("gate", gate.value()),
            ],
        );
    }

    /// Record what the store holds in one tier for one engine (spec 6).
    pub fn store_held(&self, tier: TierLabel, engine: &str, bytes: u64, entries: u64) {
        let attributes = [
            KeyValue::new("tier", tier.value()),
            KeyValue::new("engine", engine.to_owned()),
        ];
        self.store_bytes.record(bytes, &attributes);
        self.store_entries.record(entries, &attributes);
    }

    /// Count one store lookup (spec 6.6).
    pub fn store_lookup(&self, engine: &str, outcome: LookupOutcome) {
        self.store_lookups.add(
            1,
            &[
                KeyValue::new("engine", engine.to_owned()),
                KeyValue::new("outcome", outcome.value()),
            ],
        );
    }

    /// Count entries that the store removed (spec 6.9).
    pub fn store_eviction(&self, tier: TierLabel, reason: EvictReason, entries: u64) {
        self.store_evictions.add(
            entries,
            &[
                KeyValue::new("tier", tier.value()),
                KeyValue::new("reason", reason.value()),
            ],
        );
    }

    /// Count one store restore (spec 6.8).
    pub fn store_restore(&self, engine: &str, result: RestoreResult) {
        self.store_restores.add(
            1,
            &[
                KeyValue::new("engine", engine.to_owned()),
                KeyValue::new("result", result.value()),
            ],
        );
    }

    /// Record the duration of one HTTP request, in seconds (spec 9.4). `route` is the route template, a bounded set.
    pub fn http_request(&self, route: &str, status: u16, seconds: f64) {
        self.http_duration.record(
            seconds,
            &[
                KeyValue::new("http.route", route.to_owned()),
                KeyValue::new("status", i64::from(status)),
            ],
        );
    }

    /// Record that an engine entered a state. The gauge is one for that state and zero for the other four.
    pub fn engine_state(&self, engine: &str, state: EngineState) {
        for candidate in EngineState::ALL {
            self.engine_state.record(
                u64::from(candidate == state),
                &[
                    KeyValue::new("engine", engine.to_owned()),
                    KeyValue::new("state", candidate.as_str()),
                ],
            );
        }
    }
}

static GLOBAL: OnceLock<Arc<Instruments>> = OnceLock::new();
static NOOP: OnceLock<Arc<Instruments>> = OnceLock::new();

thread_local! {
    static SCOPED: RefCell<Option<Arc<Instruments>>> = const { RefCell::new(None) };
}

/// The instruments that this thread records on.
///
/// A scope on this thread wins (see [`scope`]). Then the instruments that [`install`] set. With neither, the instruments
/// record nothing.
pub fn instruments() -> Arc<Instruments> {
    if let Some(scoped) = SCOPED.with(|cell| cell.borrow().clone()) {
        return scoped;
    }
    if let Some(global) = GLOBAL.get() {
        return Arc::clone(global);
    }
    Arc::clone(NOOP.get_or_init(|| Arc::new(Instruments::noop())))
}

/// The process installed its instruments twice.
#[derive(Debug, PartialEq, Eq)]
pub struct AlreadyInstalled;

impl std::fmt::Display for AlreadyInstalled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the instruments are installed already: install them once per process"
        )
    }
}

impl std::error::Error for AlreadyInstalled {}

/// Install the instruments of the process. Every thread records on them, unless a scope covers the thread.
pub fn install(instruments: Arc<Instruments>) -> Result<(), AlreadyInstalled> {
    GLOBAL.set(instruments).map_err(|_| AlreadyInstalled)
}

/// The guard of [`scope`]. The thread records on its earlier instruments again when the guard drops.
pub struct ScopeGuard {
    previous: Option<Arc<Instruments>>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPED.with(|cell| *cell.borrow_mut() = self.previous.take());
    }
}

/// Make this thread record on `instruments` until the guard drops. A test uses this to read what a call site records,
/// and tests on other threads are not affected.
pub fn scope(instruments: Arc<Instruments>) -> ScopeGuard {
    let previous = SCOPED.with(|cell| cell.borrow_mut().replace(instruments));
    ScopeGuard { previous }
}
