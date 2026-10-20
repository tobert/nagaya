//! The council service for one served model (spec 11): builds, snapshots, pins and specs, with no HTTP.
//!
//! A [`Service`] holds an engine, the family library of the engine, the store and the ledger, through the [`Ports`]
//! of the process. A route calls its methods, and each call blocks until the job ends. The routes of step 6 run the
//! calls on a blocking thread.

pub mod build;
pub mod error;
pub mod pins;
pub mod slots;
pub mod specs;

use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use gogi_proto::SnapshotId;
use gogi_proto::context::{Context, SnapshotInfo, Tier as SnapshotTier, Usage};
use gogi_proto::pins::{PinHeld, PinOwner, PinsPut, PinsPutResult, SnapshotDetail, validate_owner};
use gogi_proto::spec::{HeldSpec, SpecRules};
use nagaya_abi::{Capability, Engine, Limits};
use nagaya_council::build::{boundary_recipes, root_recipe};
use nagaya_council::family::Family;
use nagaya_sched::Class;
use nagaya_store::{ContentId, EntryState, Store, Tier};
use serde::Serialize;
use serde_json::Value;

use self::build::{BuildBody, LookupBody, Outcome, Shared};
use self::error::ServiceError;
use self::pins::{OwnerPins, PinBook, owner_key};
use self::slots::SlotPool;
use self::specs::Specs;
use crate::jobs::{JobBody, JobSpec, Ran};
use crate::park::identity_digest;
use crate::process::{ContextChargeError, Ports, SubmitError};

/// The time in milliseconds. The service reads its clock for the store and for pins. Tests set their own.
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// The clock of the process: milliseconds since the Unix epoch.
pub fn system_clock() -> Clock {
    Arc::new(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the clock is before the epoch")
            .as_millis() as u64
    })
}

/// How a request is scheduled (spec 5.3, 5.4). It never changes a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// The class of the job.
    pub class: Class,
    /// The priority of the job. A higher priority goes first in its class.
    pub priority: i32,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            class: Class::Interactive,
            priority: 0,
        }
    }
}

impl Schedule {
    /// The schedule of a request, from its optional `class` and `priority` (gogi "Served models").
    pub fn of(class: Option<gogi_proto::identity::Class>, priority: Option<i32>) -> Self {
        Self {
            class: match class.unwrap_or_default() {
                gogi_proto::identity::Class::Interactive => Class::Interactive,
                gogi_proto::identity::Class::Batch => Class::Batch,
            },
            priority: priority.unwrap_or(0),
        }
    }
}

/// What the service needs to know about its served model.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// The served model: the `model` of the council identity.
    pub model: String,
    /// The name of the engine in the placement profile.
    pub engine: String,
    /// The most tokens of a context, or of a prompt and its reply (gogi "Identity, limits, capabilities").
    pub context_tokens: usize,
    /// The most snapshot bytes that all pins keep together.
    pub pin_bytes: u64,
    /// What a spec may hold.
    pub spec_rules: SpecRules,
}

impl ServiceConfig {
    /// A config with the default limits.
    pub fn new(model: &str, engine: &str) -> Self {
        Self {
            model: model.to_owned(),
            engine: engine.to_owned(),
            context_tokens: 4096,
            pin_bytes: 1 << 30,
            spec_rules: SpecRules::default(),
        }
    }
}

/// The result of a build (gogi "Contexts").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Built {
    /// The snapshot at the end of the context. A dry run that finds none held gives `None` (spec 11.3c).
    pub head: Option<SnapshotId>,
    /// The snapshot that the build started from.
    pub from: Option<SnapshotId>,
    /// The tokens of the context, the tokens kept and the tokens fed.
    pub usage: Usage,
    /// The snapshots that the store holds at the boundaries when the build ends. A dry run lists none.
    pub snapshots: Vec<SnapshotInfo>,
}

/// The pins of one owner (gogi "Snapshots and pins").
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PinListing {
    /// The owner.
    pub owner: String,
    /// The snapshots that the owner pinned.
    pub snapshots: Vec<SnapshotId>,
    /// The seconds until the pins end.
    pub expires_in_s: u64,
}

/// The work of a one-quantum job: it gets the engines of the group.
type Work<T> = Box<dyn FnOnce(&mut [Box<dyn Engine>]) -> T + Send>;

/// A job body that runs a closure once.
struct Once<T: Send + 'static> {
    work: Option<Work<T>>,
    tx: mpsc::Sender<T>,
}

impl<T: Send + 'static> JobBody for Once<T> {
    fn run(&mut self, _kind: nagaya_abi::Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let work = self.work.take().expect("a one-quantum job runs once");
        let _ = self.tx.send(work(engines));
        Ran::Done
    }
}

/// The charge of a context on the ledger. The charge ends with `release`, and also when the guard drops.
struct Charge<'a> {
    ports: &'a Ports,
    engine: &'a str,
    held: bool,
}

impl<'a> Charge<'a> {
    fn take(ports: &'a Ports, engine: &'a str) -> Result<Self, ServiceError> {
        ports.admit_context(engine).map_err(|error| match error {
            ContextChargeError::Stopping => ServiceError::Stopping,
            ContextChargeError::Ledger(error @ nagaya_ledger::LedgerError::ContextRefused(_)) => {
                ServiceError::ContextRefused(error)
            }
            ContextChargeError::Ledger(error) => ServiceError::internal(error.to_string()),
        })?;
        Ok(Self {
            ports,
            engine,
            held: true,
        })
    }

    fn release(mut self) -> Result<(), ServiceError> {
        self.held = false;
        match self.ports.release_context(self.engine) {
            Ok(()) => Ok(()),
            // A stopped ledger has no charge left to release.
            Err(ContextChargeError::Stopping) => Ok(()),
            Err(ContextChargeError::Ledger(error)) => Err(ServiceError::internal(error.to_string())),
        }
    }
}

impl Drop for Charge<'_> {
    fn drop(&mut self) {
        if self.held {
            // The caller went away without `release`, for example on a panic. Release the bytes, and say so.
            if let Err(error) = self.ports.release_context(self.engine) {
                eprintln!(
                    "the release of a context charge for engine `{}` failed: {error}",
                    self.engine
                );
            }
        }
    }
}

/// The council service of one served model.
pub struct Service {
    ports: Ports,
    family: Arc<dyn Family>,
    config: ServiceConfig,
    group: usize,
    shared: Arc<Shared>,
    root: nagaya_store::RecipeId,
    specs: Mutex<Specs>,
    pins: Mutex<PinBook>,
}

impl Service {
    /// Attach a service to an engine of the process.
    ///
    /// The engine must have the parts and slots capabilities, and its limits must name slots and pass inputs. The
    /// call reads the identity and the limits of the engine on its group thread. It opens the specs and the pins in
    /// `store_dir`.
    pub fn attach(
        ports: Ports,
        family: Arc<dyn Family>,
        store_dir: &Path,
        config: ServiceConfig,
    ) -> Result<Self, ServiceError> {
        Self::attach_with_clock(ports, family, store_dir, config, system_clock())
    }

    /// Like [`Service::attach`], with a clock that the caller chooses.
    pub fn attach_with_clock(
        ports: Ports,
        family: Arc<dyn Family>,
        store_dir: &Path,
        config: ServiceConfig,
        clock: Clock,
    ) -> Result<Self, ServiceError> {
        let (group, index) = ports.locate(&config.engine).ok_or_else(|| {
            ServiceError::internal(format!(
                "the profile has no engine named `{}`; serve a model that the profile holds",
                config.engine
            ))
        })?;
        let store = ports
            .store()
            .cloned()
            .ok_or_else(|| ServiceError::internal("the process has no store; start it with a store directory"))?;
        let (tx, rx) = mpsc::channel();
        let work = move |engines: &mut [Box<dyn Engine>]| {
            let engine = &mut engines[index];
            let usable = engine.parts().is_some() && engine.slots().is_some();
            (engine.identity(), engine.limits(), usable)
        };
        submit(
            &ports,
            group,
            Schedule::default(),
            nagaya_abi::Kind::Read,
            Box::new(Once {
                work: Some(Box::new(work)),
                tx,
            }),
        )?;
        let (identity, limits, usable): (nagaya_abi::Identity, Limits, bool) =
            rx.recv().map_err(|_| ServiceError::Cancelled)?;
        if !usable
            || !limits.capabilities.contains(&Capability::Parts)
            || !limits.capabilities.contains(&Capability::Slots)
        {
            return Err(ServiceError::internal(format!(
                "the engine `{}` lacks the parts or slots capability, which a build needs",
                config.engine
            )));
        }
        if limits.max_pass_inputs == 0 || limits.slots.iter().map(|slot| slot.count).sum::<u32>() == 0 {
            return Err(ServiceError::internal(format!(
                "the engine `{}` declares no pass input or no slot",
                config.engine
            )));
        }
        let digest = identity_digest(&identity);
        let pool = Arc::new(Mutex::new(SlotPool::new(&limits)));
        let shared = Arc::new(Shared {
            store,
            engine_index: index,
            engine_name: config.engine.clone(),
            identity: digest,
            chunk: limits.max_pass_inputs as usize,
            limits: Arc::new(limits),
            pool,
            clock,
        });
        let specs = Specs::open(store_dir, config.spec_rules)?;
        let book = PinBook::open(store_dir)?;
        let service = Self {
            ports,
            family,
            group,
            shared,
            root: root_recipe(&digest, b""),
            specs: Mutex::new(specs),
            pins: Mutex::new(book),
            config,
        };
        service.settle_pins()?;
        Ok(service)
    }

    /// The number of slots that no build uses.
    pub fn free_slots(&self) -> usize {
        self.shared.pool.lock().expect("the slot pool lock is poisoned").free()
    }

    fn now(&self) -> u64 {
        (self.shared.clock)()
    }

    fn store(&self) -> std::sync::MutexGuard<'_, Store> {
        self.shared.store.lock().expect("the store lock is poisoned")
    }

    // ---- builds

    /// Build a context (gogi "Contexts"): render it, start from the longest held prefix, feed the rest, and hold a
    /// snapshot at each boundary. A dry run renders and looks up, and feeds and holds nothing.
    pub fn build(&self, context: &Context, dry_run: bool, schedule: Schedule) -> Result<Built, ServiceError> {
        let rendering = self.family.render(context, false)?;
        if rendering.end >= self.config.context_tokens {
            return Err(ServiceError::ContextFull {
                prompt_tokens: rendering.end as u64,
                context_tokens: self.config.context_tokens as u64,
            });
        }
        let recipes = boundary_recipes(self.root, &rendering)?;
        let tokens: Arc<[u32]> = rendering.tokens[..rendering.end].into();
        let (tx, rx) = mpsc::channel();
        let outcome = if dry_run {
            let body = LookupBody::new(Arc::clone(&self.shared), recipes.clone(), tokens.len(), tx);
            submit(
                &self.ports,
                self.group,
                schedule,
                nagaya_abi::Kind::Read,
                Box::new(body),
            )?;
            wait(&rx)?
        } else {
            let charge = Charge::take(&self.ports, &self.config.engine)?;
            let body = BuildBody::new(Arc::clone(&self.shared), Arc::clone(&tokens), recipes.clone(), tx);
            let outcome = submit(
                &self.ports,
                self.group,
                schedule,
                nagaya_abi::Kind::Copy,
                Box::new(body),
            )
            .and_then(|()| wait(&rx));
            charge.release()?;
            outcome?
        };
        let prompt = rendering.end as u64;
        let usage = Usage {
            prompt,
            kept: outcome.kept as u64,
            fed: prompt - outcome.kept as u64,
            completion: 0,
            reasoning: 0,
        };
        let snapshots = if dry_run {
            Vec::new()
        } else {
            let store = self.store();
            recipes
                .iter()
                .zip(&outcome.held)
                .filter_map(|(recipe, id)| {
                    let id = (*id)?;
                    let entry = store.entries().get(&id)?;
                    Some(SnapshotInfo {
                        id: snapshot_id(&id),
                        message: recipe.boundary.message as u64,
                        tokens: recipe.boundary.tokens as u64,
                        tier: Some(snapshot_tier(entry.tier)),
                        bytes: Some(entry.size),
                    })
                })
                .collect()
        };
        Ok(Built {
            head: outcome.held.last().copied().flatten().map(|id| snapshot_id(&id)),
            from: outcome.from.map(|id| snapshot_id(&id)),
            usage,
            snapshots,
        })
    }

    // ---- snapshots and pins

    /// A held snapshot: its served model, tokens, tier, size and the owners that pin it (gogi "Snapshots and pins").
    pub fn snapshot(&self, id: &SnapshotId) -> Result<SnapshotDetail, ServiceError> {
        let content = self
            .held_content_id(id.as_str())
            .ok_or_else(|| ServiceError::NotFound(format!("no snapshot {id} is held: build the context to hold it")))?;
        let now = self.now();
        let store = self.store();
        let entry = store.entries().get(&content).expect("a held content id has an entry");
        let tokens = entry.coverage.iter().map(|part| part.end).max().unwrap_or(0);
        let book = self.pins.lock().expect("the pin book lock is poisoned");
        let pins = store
            .pins()
            .get(&content)
            .map(|pins| {
                pins.iter()
                    .filter(|pin| pin.expires_ms > now)
                    .filter_map(|pin| {
                        book.name_of(pin.owner).map(|owner| PinOwner {
                            owner: owner.to_owned(),
                            expires_in_s: (pin.expires_ms - now).div_ceil(1000),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(SnapshotDetail {
            id: id.clone(),
            model: self.config.model.clone(),
            tokens,
            tier: snapshot_tier(entry.tier),
            bytes: Some(entry.size),
            pins,
        })
    }

    /// The content id of a snapshot id that this service holds: a `blake3:` id, a live entry of this engine.
    fn held_content_id(&self, id: &str) -> Option<ContentId> {
        let hex = id.strip_prefix("blake3:")?;
        let content = ContentId::from_hex(hex).ok()?;
        self.is_ours(&self.store(), &content).then_some(content)
    }

    /// True when the store holds a live entry that this engine made.
    fn is_ours(&self, store: &Store, content: &ContentId) -> bool {
        store
            .entries()
            .get(content)
            .is_some_and(|entry| entry.state == EntryState::Live && entry.engine == self.config.engine)
    }

    /// The bytes of the entries that pins keep: the entries, and the parents that they are built on.
    fn pinned_bytes(store: &Store, ids: impl IntoIterator<Item = ContentId>) -> u64 {
        let mut seen = std::collections::BTreeSet::new();
        for id in ids {
            let mut at = Some(id);
            while let Some(current) = at {
                if !seen.insert(current) {
                    break;
                }
                at = store.entries().get(&current).and_then(|entry| entry.parent);
            }
        }
        seen.iter()
            .filter_map(|id| store.entries().get(id))
            .map(|entry| entry.size)
            .sum()
    }

    /// Replace the pins of an owner (gogi "Snapshots and pins"). A snapshot that the store does not hold reports
    /// `held: false` and pins nothing. A set past `pin_bytes` pins nothing and is `too_large`.
    pub fn put_pins(&self, owner: &str, put: &PinsPut) -> Result<PinsPutResult, ServiceError> {
        validate_owner(owner)?;
        put.validate()?;
        let now = self.now();
        let expires_ms = now + put.ttl_s * 1000;
        let key = owner_key(owner);
        let mut book = self.pins.lock().expect("the pin book lock is poisoned");
        let mut store = self.store();
        let mut report = Vec::with_capacity(put.snapshots.len());
        let mut wanted: Vec<ContentId> = Vec::new();
        for text in &put.snapshots {
            let content = {
                let hex = text.strip_prefix("blake3:");
                hex.and_then(|hex| ContentId::from_hex(hex).ok())
                    .filter(|content| self.is_ours(&store, content))
            };
            report.push(PinHeld {
                id: text.clone(),
                held: content.is_some(),
            });
            if let Some(content) = content
                && !wanted.contains(&content)
            {
                wanted.push(content);
            }
        }
        // The limit counts every pin together: this set, and the live pins of the other owners.
        let others = store
            .pins()
            .iter()
            .filter(|(_, pins)| pins.iter().any(|pin| pin.owner != key && pin.expires_ms > now))
            .map(|(id, _)| *id);
        let total = Self::pinned_bytes(&store, wanted.iter().copied().chain(others));
        if total > self.config.pin_bytes {
            return Err(ServiceError::TooLarge {
                param: "pin_bytes".to_owned(),
                message: format!(
                    "the pins would keep {total} bytes, past the limit of {}; pin fewer snapshots, or drop other pins",
                    self.config.pin_bytes
                ),
            });
        }
        let old: Vec<ContentId> = book.get(owner).map(|pins| pins.snapshots.clone()).unwrap_or_default();
        for id in old.iter().filter(|id| !wanted.contains(id)) {
            store
                .unpin(id, key)
                .map_err(|error| ServiceError::internal(format!("cannot drop a pin: {error}")))?;
        }
        for id in &wanted {
            store
                .pin(*id, key, expires_ms, now)
                .map_err(|error| ServiceError::internal(format!("cannot pin a snapshot: {error}")))?;
        }
        let expires_in_s = if wanted.is_empty() {
            book.set(owner, None);
            None
        } else {
            book.set(
                owner,
                Some(OwnerPins {
                    snapshots: wanted,
                    expires_ms,
                }),
            );
            Some(put.ttl_s)
        };
        book.save()?;
        Ok(PinsPutResult {
            owner: owner.to_owned(),
            snapshots: report,
            expires_in_s,
        })
    }

    /// The pins of an owner. An owner with no pins, or with pins that ended, is `not_found`.
    pub fn get_pins(&self, owner: &str) -> Result<PinListing, ServiceError> {
        validate_owner(owner)?;
        let now = self.now();
        let book = self.pins.lock().expect("the pin book lock is poisoned");
        match book.get(owner) {
            Some(pins) if pins.expires_ms > now => Ok(PinListing {
                owner: owner.to_owned(),
                snapshots: pins.snapshots.iter().map(snapshot_id).collect(),
                expires_in_s: (pins.expires_ms - now).div_ceil(1000),
            }),
            _ => Err(ServiceError::NotFound(format!("{owner} pins nothing"))),
        }
    }

    /// Drop the pins of an owner.
    pub fn delete_pins(&self, owner: &str) -> Result<(), ServiceError> {
        validate_owner(owner)?;
        let mut book = self.pins.lock().expect("the pin book lock is poisoned");
        let Some(pins) = book.get(owner).cloned() else {
            return Err(ServiceError::NotFound(format!("{owner} pins nothing")));
        };
        let key = owner_key(owner);
        let mut store = self.store();
        for id in &pins.snapshots {
            store
                .unpin(id, key)
                .map_err(|error| ServiceError::internal(format!("cannot drop a pin: {error}")))?;
        }
        book.set(owner, None);
        book.save()
    }

    /// At attach: drop the owners whose pins ended, and the snapshots that a restart took (spec 6.11).
    fn settle_pins(&self) -> Result<(), ServiceError> {
        let now = self.now();
        let mut book = self.pins.lock().expect("the pin book lock is poisoned");
        let mut store = self.store();
        let owners: Vec<(String, OwnerPins)> = book.owners().iter().map(|(o, p)| (o.clone(), p.clone())).collect();
        let mut changed = false;
        for (owner, pins) in owners {
            let key = owner_key(&owner);
            let live = pins.expires_ms > now;
            let kept: Vec<ContentId> = pins
                .snapshots
                .iter()
                .copied()
                .filter(|id| live && store.entries().contains_key(id))
                .collect();
            if kept.len() == pins.snapshots.len() {
                continue;
            }
            for id in pins.snapshots.iter().filter(|id| !kept.contains(id)) {
                if store.entries().contains_key(id) {
                    store
                        .unpin(id, key)
                        .map_err(|error| ServiceError::internal(format!("cannot drop a pin: {error}")))?;
                }
            }
            book.set(
                &owner,
                (!kept.is_empty()).then_some(OwnerPins {
                    snapshots: kept,
                    expires_ms: pins.expires_ms,
                }),
            );
            changed = true;
        }
        if changed {
            book.save()?;
        }
        Ok(())
    }

    // ---- specs

    /// Hold a spec. The same spec gets the same id.
    pub fn hold_spec(&self, spec: Value) -> Result<HeldSpec, ServiceError> {
        self.specs.lock().expect("the spec lock is poisoned").hold(spec)
    }

    /// One held spec.
    pub fn spec(&self, id: &str) -> Result<HeldSpec, ServiceError> {
        self.specs.lock().expect("the spec lock is poisoned").get(id)
    }

    /// The held specs.
    pub fn specs(&self) -> Vec<HeldSpec> {
        self.specs.lock().expect("the spec lock is poisoned").list()
    }

    /// Drop a spec.
    pub fn drop_spec(&self, id: &str) -> Result<(), ServiceError> {
        self.specs.lock().expect("the spec lock is poisoned").drop_spec(id)
    }
}

/// The snapshot id of a content id (spec 11.1): `blake3:` and the 64 hex digits.
pub fn snapshot_id(id: &ContentId) -> SnapshotId {
    SnapshotId::parse(format!("blake3:{}", id.hex())).expect("a content id makes a valid snapshot id")
}

fn snapshot_tier(tier: Tier) -> SnapshotTier {
    match tier {
        Tier::Host => SnapshotTier::Host,
        Tier::Nvme => SnapshotTier::Disk,
    }
}

fn submit(
    ports: &Ports,
    group: usize,
    schedule: Schedule,
    first: nagaya_abi::Kind,
    body: Box<dyn JobBody>,
) -> Result<(), ServiceError> {
    ports
        .submit_job(
            group,
            JobSpec {
                class: schedule.class,
                priority: i64::from(schedule.priority),
                first,
                body,
            },
        )
        .map_err(|error| match error {
            SubmitError::Stopping | SubmitError::Stopped => ServiceError::Stopping,
            SubmitError::UnknownGroup => ServiceError::internal("the service names a group that does not exist"),
        })
}

fn wait(rx: &Receiver<Result<Outcome, ServiceError>>) -> Result<Outcome, ServiceError> {
    rx.recv().map_err(|_| ServiceError::Cancelled)?
}
