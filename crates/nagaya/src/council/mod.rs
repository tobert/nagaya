//! The council service for one served model (spec 11): builds, snapshots, pins and specs, with no HTTP.
//!
//! A [`Service`] holds an engine, the family library of the engine, the store and the ledger, through the [`Ports`]
//! of the process. A route calls its methods, and each call blocks until the job ends. The routes of step 6 run the
//! calls on a blocking thread.

pub mod build;
pub mod decisions;
pub mod error;
pub mod generations;
pub mod pins;
pub mod reads;
pub mod slots;
pub mod specs;
pub mod staging;

use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gogi_proto::Invalid;
use gogi_proto::SnapshotId;
use gogi_proto::context::{Context, SnapshotInfo, Tier as SnapshotTier, Usage};
use gogi_proto::decision::DecisionIdentity;
use gogi_proto::pins::{PinHeld, PinOwner, PinsPut, PinsPutResult, SnapshotDetail, validate_owner};
use gogi_proto::spec::{HeldSpec, SpecRules};
use nagaya_abi::{Capability, Engine, Limits};
use nagaya_council::build::{BoundaryRecipe, boundary_recipes, recipes_with_cuts, root_recipe};
use nagaya_council::family::{Boundary, Family, Rendering};
use nagaya_sched::{Class, IntervalCopies, IntervalPolicy};
use nagaya_store::{ContentId, EntryState, RecipeId, Store, Tier, recipe_boundary};
use serde::Serialize;
use serde_json::Value;

use self::build::{BuildBody, ContextBody, Cut, Finished, LookupBody, Outcome, Shared, Then};
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
    /// The most tokens of a context that a decision reads, not counting the spec and the case after it.
    pub decision_context_tokens: usize,
    /// The most contexts of one decision, at most 8.
    pub contexts_per_decision: usize,
    /// The most bytes of a case.
    pub case_bytes: usize,
    /// The `timeout_ms` of a decision that sends none.
    pub default_timeout_ms: u64,
    /// The sample settings of a generation that does not send them (spec 11.2).
    pub sample: SampleDefaults,
    /// The most snapshot bytes that all pins keep together.
    pub pin_bytes: u64,
    /// What a spec may hold.
    pub spec_rules: SpecRules,
    /// The other engines of the server, whose snapshots this service reads and pins too: snapshots and pins belong to
    /// the server (gogi "Served models"), such as those of a speech engine (spec 11.8a). Each is the name of the
    /// engine in the profile and its served model.
    pub peers: Vec<(String, String)>,
}

/// The sample settings that a generation takes when its request does not name them. Each engine has its own
/// (spec 11.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleDefaults {
    /// The temperature. 0 is greedy.
    pub temperature: f64,
    /// The top_p filter. 1 is off.
    pub top_p: f64,
    /// The top_k filter. 0 is off.
    pub top_k: u64,
    /// The min_p filter. 0 is off.
    pub min_p: f64,
}

impl Default for SampleDefaults {
    /// No filter, at temperature 1.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
        }
    }
}

impl ServiceConfig {
    /// A config with the default limits: mkviolet's decision limits (gogi "Identity, limits, capabilities").
    pub fn new(model: &str, engine: &str) -> Self {
        Self {
            model: model.to_owned(),
            engine: engine.to_owned(),
            context_tokens: 4096,
            decision_context_tokens: 3072,
            contexts_per_decision: gogi_proto::decision::MAX_CONTEXTS,
            case_bytes: 65536,
            default_timeout_ms: 30_000,
            sample: SampleDefaults::default(),
            pin_bytes: 1 << 30,
            spec_rules: SpecRules::default(),
            peers: Vec::new(),
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
    /// The recipe of the head. A dry run that finds no held head names the head with it (spec 11.3c).
    pub head_recipe: nagaya_store::RecipeId,
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

    fn request(&self) -> &'static str {
        // The only one-quantum job asks the engine for its identity and limits when the service attaches.
        "attach"
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
            // The caller went away without `release`, for example on a panic. Release the bytes. A release that
            // fails leaves the ledger wrong, and the process stops (spec 4.2, 8.5).
            if let Err(error) = self.ports.release_context(self.engine) {
                self.ports
                    .release_failed(&format!("a context charge for engine `{}`", self.engine), &error);
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
    /// The place of the engine in its group, which each job names (spec 3.3a).
    index: usize,
    shared: Arc<Shared>,
    root: nagaya_store::RecipeId,
    identity: nagaya_abi::Identity,
    specs: Mutex<Specs>,
    /// The pin book. Lock order: take `pins` before the store (`shared.store`), never the store before `pins`. A
    /// method that needs both locks `pins` first. Two methods that lock in two orders can wait for each other for ever.
    pins: Mutex<PinBook>,
}

impl Service {
    /// The first failure that the service saw outside a call, or `None`: a release of a ledger charge that failed.
    pub fn failure(&self) -> Option<String> {
        self.ports.failure()
    }

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
            (group, index),
            Schedule::default(),
            None,
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
        let cap = store.lock().expect("the store lock is poisoned").interval_copies();
        let intervals = IntervalPolicy::from_limits(&limits, cap)
            .map_err(|error| {
                ServiceError::internal(format!(
                    "the engine `{}` declares a checkpoint grain that the scheduler cannot use: {error:?}; declare a \
                     grain above 0, and `segments_exact` with it (spec 6.13)",
                    config.engine
                ))
            })?
            .map(|policy| Mutex::new(IntervalCopies::new(policy)));
        let root = root_recipe(&digest, b"");
        let shared = Arc::new(Shared {
            root,
            intervals,
            ports: ports.clone(),
            store,
            engine_index: index,
            engine_name: config.engine.clone(),
            identity: digest,
            // An engine with a feed rule gets passes of its block from the start of each span (spec 6.14).
            chunk: limits
                .feed
                .map_or(limits.max_pass_inputs as usize, |feed| feed.block as usize),
            vocab: family.vocab_size(),
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
            index,
            shared,
            root,
            identity,
            specs: Mutex::new(specs),
            pins: Mutex::new(book),
            config,
        };
        service.settle_pins()?;
        Ok(service)
    }

    /// The identity fields that a number depends on, with no spec (gogi "Identity, limits, capabilities").
    ///
    /// The weight hash is the checkpoint digest (spec 12.2). The template is the template id of the family (spec
    /// 11.3a). The engine is `nagaya:` and the first 16 hex digits of the identity digest. The digest covers each
    /// field of the engine identity, its `numbers` too (spec 6.2, 12.1).
    pub fn identity(&self) -> DecisionIdentity {
        let digest: String = self.shared.identity[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        DecisionIdentity {
            model: self.config.model.clone(),
            weight_hash: self.identity.checkpoint_digest.clone(),
            tokenizer_hash: self.identity.tokenizer_hash.clone(),
            template: self.family.template_id().to_owned(),
            engine: format!("nagaya:{digest}"),
            spec_id: None,
        }
    }

    /// The served model: the `model` of the council identity.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// The family library of the served model. The text routes under `/nagaya/v1` use it (spec 11.4b).
    pub fn family(&self) -> &dyn Family {
        &*self.family
    }

    /// The config of the service: its limits and its defaults.
    pub fn config(&self) -> &ServiceConfig {
        &self.config
    }

    /// What one token of a snapshot costs, in bytes: one row of each cells part (gogi "Identity, limits,
    /// capabilities").
    pub fn snapshot_bytes(&self) -> u64 {
        self.shared
            .limits
            .parts
            .iter()
            .filter(|part| part.kind == nagaya_abi::PartKind::Cells)
            .map(|part| part.dtype.element_bytes() * part.shape.iter().product::<u64>())
            .sum()
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
    ///
    /// An open message builds only as a dry run (gogi "Continuations"). The dry run looks up the build plan of a
    /// continuation: the context before the open message. Its `head` names the recipe of the open rendering, which
    /// the store never holds.
    pub fn build(&self, context: &Context, dry_run: bool, schedule: Schedule) -> Result<Built, ServiceError> {
        self.reads_text_only(&gogi_proto::objects::objects_of(&context.messages))?;
        if context.open().is_some() && !dry_run {
            return Err(Invalid::param(
                "continue",
                "nothing was built: a build reads an open message only with dry_run: true; send dry_run: true to test \
                 a continuation, or send it to a generation",
            )
            .into());
        }
        let rendering = self.family.render(context, false)?;
        if rendering.end > self.config.context_tokens {
            return Err(ServiceError::past_context(rendering.end, self.config.context_tokens));
        }
        let rendering_end = rendering.end;
        let head_recipe = boundary_recipes(self.root, &rendering)?
            .last()
            .expect("a rendering has a boundary")
            .recipe;
        let open = context.open().is_some();
        let open_rendering = rendering.clone();
        let rendering = match open {
            true => self.open_plan(context, &rendering)?,
            false => rendering,
        };
        let recipes = plan_recipes(self.root, &rendering)?;
        let tokens: Arc<[u32]> = rendering.tokens[..rendering.end].into();
        let (cuts, extension) = match open {
            true => self.starts(context, &open_rendering.tokens, &rendering)?,
            false => (self.cuts(context, &rendering)?, None),
        };
        let outcome = if dry_run {
            let (tx, rx) = mpsc::channel();
            // A continuation's slot holds all its open rendering, and a dry run reports what the generation would.
            let room = if open {
                open_rendering.tokens.len()
            } else {
                tokens.len()
            };
            let body = LookupBody::new(Arc::clone(&self.shared), recipes.clone(), Arc::clone(&tokens), room, tx)
                .cuts(cuts)
                .extension(extension);
            submit(
                &self.ports,
                (self.group, self.index),
                schedule,
                None,
                nagaya_abi::Kind::Read,
                Box::new(body),
            )?;
            wait(&rx)?
        } else {
            let shared = Arc::clone(&self.shared);
            let job = |tx| BuildBody::new(shared, tokens, recipes.clone(), tx).cuts(cuts);
            self.run_context(job, schedule, None)?.outcome
        };
        let prompt = rendering_end as u64;
        let usage = Usage {
            prompt,
            kept: outcome.kept as u64,
            fed: prompt - outcome.kept as u64,
            completion: 0,
            reasoning: 0,
            media: None,
        };
        let snapshots = if dry_run {
            Vec::new()
        } else {
            self.infos(&held_boundaries(&outcome))
        };
        Ok(Built {
            head_recipe,
            // The store can evict the head before the list is read. A build names only a head that it lists.
            head: match open {
                true => None,
                false => outcome
                    .held
                    .last()
                    .copied()
                    .flatten()
                    .map(|id| snapshot_id(&id))
                    .filter(|head| dry_run || snapshots.iter().any(|snapshot| snapshot.id == *head)),
            },
            from: outcome.from.map(|id| snapshot_id(&id)),
            usage,
            snapshots,
        })
    }

    /// The build plan of an open context (gogi "Continuations", spec 11.3c): the rendering of the context before its
    /// open message. `rendering` is the rendering of the open context.
    pub(crate) fn open_plan(&self, context: &Context, rendering: &Rendering) -> Result<Rendering, ServiceError> {
        let mut before = context.clone();
        before.messages.pop();
        let before = match before.messages.is_empty() {
            true => Rendering {
                tokens: Vec::new(),
                end: 0,
                boundaries: Vec::new(),
                control_text: Vec::new(),
            },
            false => self.family.render(&before, false)?,
        };
        Ok(nagaya_council::continuation::open_plan(rendering, &before)?)
    }

    /// The cuts of a context that has no open message (see [`Service::starts`]).
    fn cuts(&self, context: &Context, rendering: &Rendering) -> Result<Vec<Cut>, ServiceError> {
        Ok(self.starts(context, &rendering.tokens[..rendering.end], rendering)?.0)
    }

    /// The held snapshots where a job may start, other than the plain boundaries of its plan (spec 11.3c).
    ///
    /// The store may hold a snapshot whose chain of seals differs from the plan's: at the unmarked end of a message,
    /// at the head of a cut reply, or after such a cut. The search follows the store's index by `chains_from` from
    /// the root. From each seal, it goes to the next boundary of the plan, and to each held entry that chains from
    /// the seal and whose recipe the context's `tokens` give. Thus it reads only the entries that chain from this
    /// context, and it renders nothing.
    ///
    /// `tokens` are the context's tokens: the plan's, or for an open context all the open rendering. A held entry at
    /// or before the end of the plan is a cut, with the seals of its chain inserted among the plan's boundaries. For
    /// an open context, the longest held entry after the plan, at most one token before the end, is the extension.
    fn starts(
        &self,
        context: &Context,
        tokens: &[u32],
        plan: &Rendering,
    ) -> Result<(Vec<Cut>, Option<build::Extension>), ServiceError> {
        let open = context.open().is_some();
        let limit = if open { tokens.len().saturating_sub(1) } else { plan.end };
        // The message of a seal that is not a boundary: the message of the next boundary, or the open message.
        let message_at = |position: usize| {
            plan.boundaries
                .iter()
                .find(|boundary| boundary.tokens >= position)
                .map_or(context.messages.len() - 1, |boundary| boundary.message)
        };
        struct Node {
            recipe: RecipeId,
            at: usize,
            path: Vec<BoundaryRecipe>,
        }
        let mut stack = vec![Node {
            recipe: self.root,
            at: 0,
            path: Vec::new(),
        }];
        let mut seen: std::collections::BTreeSet<RecipeId> = std::collections::BTreeSet::new();
        let mut found: Vec<Node> = Vec::new();
        let store = self.store();
        while let Some(node) = stack.pop() {
            if let Some(next) = plan.boundaries.iter().find(|boundary| boundary.tokens > node.at) {
                let recipe = recipe_boundary(&node.recipe, &tokens[node.at..next.tokens], next.tokens as u64);
                if seen.insert(recipe) {
                    stack.push(Node {
                        recipe,
                        at: next.tokens,
                        path: node.path.clone(),
                    });
                }
            }
            for entry in store.chained_from(&node.recipe) {
                let usable = entry.state == EntryState::Live
                    && entry.reason != nagaya_store::Reason::Interval
                    && entry.engine == self.shared.engine_name
                    && entry.identity == self.shared.identity;
                let Some(at) = entry.coverage.iter().map(|span| span.end).max().map(|end| end as usize) else {
                    continue;
                };
                if !usable || at <= node.at || at > limit {
                    continue;
                }
                if recipe_boundary(&node.recipe, &tokens[node.at..at], at as u64) != entry.recipe {
                    continue;
                }
                let mut path = node.path.clone();
                if plan.boundaries.iter().all(|boundary| boundary.tokens != at) {
                    path.push(BoundaryRecipe {
                        boundary: Boundary {
                            message: message_at(at),
                            tokens: at,
                        },
                        recipe: entry.recipe,
                    });
                }
                if !path.is_empty() {
                    found.push(Node {
                        recipe: entry.recipe,
                        at,
                        path: path.clone(),
                    });
                }
                if seen.insert(entry.recipe) {
                    stack.push(Node {
                        recipe: entry.recipe,
                        at,
                        path,
                    });
                }
            }
        }
        drop(store);
        let chain = |seals: &[BoundaryRecipe]| -> Result<Vec<BoundaryRecipe>, ServiceError> {
            if plan.boundaries.is_empty() {
                return Ok(Vec::new());
            }
            let inserted: Vec<Boundary> = seals.iter().map(|seal| seal.boundary).collect();
            Ok(recipes_with_cuts(self.root, plan, &inserted)?)
        };
        let mut cuts: Vec<Cut> = Vec::new();
        let mut extension: Option<build::Extension> = None;
        for node in found {
            if node.at > plan.end {
                if extension
                    .as_ref()
                    .is_some_and(|e| e.seals.last().is_some_and(|s| s.boundary.tokens >= node.at))
                {
                    continue;
                }
                let (inside, after): (Vec<BoundaryRecipe>, Vec<BoundaryRecipe>) =
                    node.path.iter().partition(|seal| seal.boundary.tokens <= plan.end);
                extension = Some(build::Extension {
                    tokens: tokens[plan.end..node.at].to_vec(),
                    chain: chain(&inside)?,
                    seals: after,
                });
                continue;
            }
            if cuts.iter().any(|(cut, _)| cut.tokens == node.at) {
                continue;
            }
            let recipes = chain(&node.path)?;
            let Some(cut) = recipes.iter().find(|recipe| recipe.boundary.tokens == node.at) else {
                continue;
            };
            if cut.recipe != node.recipe {
                return Err(ServiceError::internal(format!(
                    "the chain of a held snapshot at token {} does not give its recipe",
                    node.at
                )));
            }
            cuts.push((cut.boundary, recipes));
        }
        cuts.sort_by_key(|(cut, _)| std::cmp::Reverse(cut.tokens));
        Ok((cuts, extension))
    }

    /// Run a context job with the ledger charge of its context, and wait for its result (spec 4.2).
    fn run_context<T: Then>(
        &self,
        job: impl FnOnce(Sender<Result<Finished<T>, ServiceError>>) -> ContextBody<T>,
        schedule: Schedule,
        deadline: Option<Instant>,
    ) -> Result<Finished<T>, ServiceError> {
        let charge = Charge::take(&self.ports, &self.config.engine)?;
        let (tx, rx) = mpsc::channel();
        let result = submit(
            &self.ports,
            (self.group, self.index),
            schedule,
            deadline,
            nagaya_abi::Kind::Copy,
            Box::new(job(tx)),
        )
        .and_then(|()| rx.recv().map_err(|_| ServiceError::Cancelled)?);
        charge.release()?;
        result
    }

    /// The snapshots that the store holds at boundaries, as a response lists them. A boundary whose entry eviction
    /// removed is not in the list (spec 11.3c).
    fn infos(&self, held: &[(nagaya_council::family::Boundary, ContentId)]) -> Vec<SnapshotInfo> {
        let store = self.store();
        held.iter()
            .filter_map(|(boundary, id)| {
                let entry = store.entries().get(id)?;
                Some(SnapshotInfo {
                    id: snapshot_id(id),
                    message: boundary.message as u64,
                    tokens: boundary.tokens as u64,
                    tier: Some(snapshot_tier(entry.tier)),
                    bytes: Some(entry.size),
                })
            })
            .collect()
    }

    // ---- snapshots and pins

    /// A held snapshot: its served model, tokens, tier, size and the owners that pin it (gogi "Snapshots and pins").
    pub fn snapshot(&self, id: &SnapshotId) -> Result<SnapshotDetail, ServiceError> {
        let now = self.now();
        // Lock order: the pins, then the store (see the field `pins`). One store guard covers the check and the read,
        // so an eviction cannot fall between them.
        let book = self.pins.lock().expect("the pin book lock is poisoned");
        let store = self.store();
        let not_held = || ServiceError::NotFound(format!("no snapshot {id} is held: build the context to hold it"));
        let content = id
            .as_str()
            .strip_prefix("blake3:")
            .and_then(|hex| ContentId::from_hex(hex).ok())
            .filter(|content| self.is_ours(&store, content))
            .ok_or_else(not_held)?;
        let entry = store.entries().get(&content).ok_or_else(not_held)?;
        let tokens = entry.coverage.iter().map(|part| part.end).max().unwrap_or(0);
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
        // A peer's snapshot names the peer's served model.
        let model = self
            .config
            .peers
            .iter()
            .find(|(peer, _)| *peer == entry.engine)
            .map_or(&self.config.model, |(_, model)| model)
            .clone();
        Ok(SnapshotDetail {
            id: id.clone(),
            model,
            tokens,
            tier: snapshot_tier(entry.tier),
            bytes: Some(entry.size),
            pins,
        })
    }

    /// True when the store holds a live entry that this engine or a peer made.
    fn is_ours(&self, store: &Store, content: &ContentId) -> bool {
        store.entries().get(content).is_some_and(|entry| {
            entry.state == EntryState::Live
                && (entry.engine == self.config.engine
                    || self.config.peers.iter().any(|(peer, _)| *peer == entry.engine))
        })
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

    /// Refuses the first object a request names, with a `400` that names its member (gogi "Contexts"). Each served model
    /// of the nagaya reads text only: no family renders an object yet, so its identity declares no `media`.
    pub(crate) fn reads_text_only(&self, objects: &[gogi_proto::objects::Named]) -> Result<(), ServiceError> {
        match objects.first() {
            None => Ok(()),
            Some(named) => Err(Invalid::param(
                named.at.clone(),
                format!(
                    "nothing ran: {} is {} of media type {}, and {} reads text only",
                    named.at, named.object.id, named.object.mime, self.config.model
                ),
            )
            .into()),
        }
    }

    /// The objects of each context of a request with several, named `contexts[k].messages[i].content[j]`.
    pub(crate) fn objects_in(contexts: &[Context]) -> Vec<gogi_proto::objects::Named> {
        contexts
            .iter()
            .enumerate()
            .flat_map(|(k, c)| {
                gogi_proto::objects::objects_of(&c.messages)
                    .into_iter()
                    .map(move |named| gogi_proto::objects::Named {
                        at: format!("contexts[{k}].{}", named.at),
                        ..named
                    })
            })
            .collect()
    }

    /// Replace the pins of an owner (gogi "Snapshots and pins"). A snapshot that the store does not hold reports
    /// `held: false` and pins nothing. So does an interval copy. A set past `pin_bytes` pins nothing and is `too_large`.
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
                // A client cannot pin an interval copy (spec 6.13). A pin would stop the park and the next copy
                // from removing it.
                hex.and_then(|hex| ContentId::from_hex(hex).ok())
                    .filter(|content| self.is_ours(&store, content))
                    .filter(|content| store.entries()[content].reason != nagaya_store::Reason::Interval)
            };
            report.push(PinHeld {
                id: gogi_proto::SnapshotId::parse(text.clone()).expect("put.validate checked each id"),
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

/// The boundaries of a job's context that the store holds, with their entries. A cut is not a boundary of the context,
/// so it is not in the list.
pub(crate) fn held_boundaries(outcome: &Outcome) -> Vec<(nagaya_council::family::Boundary, ContentId)> {
    outcome
        .recipes
        .iter()
        .zip(&outcome.held)
        .filter(|(recipe, _)| !outcome.inserted.contains(&recipe.boundary))
        .filter_map(|(recipe, id)| Some((recipe.boundary, (*id)?)))
        .collect()
}

/// The recipes of a build plan. A plan with no boundary builds nothing: the job starts on an empty slot.
pub(crate) fn plan_recipes(
    root: nagaya_store::RecipeId,
    plan: &Rendering,
) -> Result<Vec<nagaya_council::build::BoundaryRecipe>, ServiceError> {
    match plan.boundaries.is_empty() {
        true => Ok(Vec::new()),
        false => Ok(boundary_recipes(root, plan)?),
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
    (group, engine): (usize, usize),
    schedule: Schedule,
    deadline: Option<Instant>,
    first: nagaya_abi::Kind,
    body: Box<dyn JobBody>,
) -> Result<(), ServiceError> {
    ports
        .submit_job(
            group,
            JobSpec {
                class: schedule.class,
                priority: i64::from(schedule.priority),
                deadline,
                first,
                engine,
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
