//! The jobs of a context: the lookup and the restore, the feed, and the snapshot at each boundary (spec 11.3c). A
//! read or a generation goes on after the build, on the same slot.
//!
//! The bodies run on the group thread, because `lookup` and `restore` take the engine (spec 3.2).

use std::collections::VecDeque;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use nagaya_abi::{
    Engine, Kind, Limits, OutputsWanted, PassError, PassOut, PassStatus, Segment, SegmentInputs, State, WriteResult,
};
use nagaya_council::build::{BoundaryRecipe, Step, plan_chunks, plan_steps};
use nagaya_council::family::Boundary;
use nagaya_council::read::logit_row;
use nagaya_sched::{IntervalCopies, Item, JobState};
use nagaya_store::{
    ContentId, EngineFacts, EntryState, Put, Reason, RecipeId, Restored, Snapshot, Store, StoreError, Tier, Tolerance,
    recipe_boundary, recipe_not_sealed, write_snapshots,
};
use nagaya_telemetry::EngineState;

use super::Clock;
use super::error::ServiceError;
use super::slots::{self, NoSlot, SlotLease, SlotPool};
use super::staging::{self, Staging};
use crate::jobs::{JobBody, Ran};
use crate::park;
use crate::process::Ports;

/// The owner of the entries that builds put. A client pin has its own owner key (spec 11.3c).
pub const COUNCIL_OWNER: u64 = 1;

/// What every body of one build or read shares.
pub struct Shared {
    /// The ways into the process. A copy of engine state charges the ledger through them (spec 4.2).
    pub ports: Ports,
    /// The machine's store.
    pub store: Arc<Mutex<Store>>,
    /// The index of the engine in its group.
    pub engine_index: usize,
    /// The name of the engine in the profile.
    pub engine_name: String,
    /// The digest of the engine identity (spec 6.2).
    pub identity: [u8; 32],
    /// The limits of the engine.
    pub limits: Arc<Limits>,
    /// The most tokens of one pass of a feed. With a feed rule it is the block, and each span starts its passes at
    /// its own start (spec 6.14).
    pub chunk: usize,
    /// The tokens of the vocabulary of the family. A logit row has this many values.
    pub vocab: usize,
    /// The slots of the engine.
    pub pool: Arc<Mutex<SlotPool>>,
    /// The time in milliseconds.
    pub clock: Clock,
    /// The recipe of an empty context (spec 6.2).
    pub root: RecipeId,
    /// The interval copies of the contexts of the engine, by slot. `None` when the engine declares no grain or the
    /// profile caps the copies at 0 (spec 6.13).
    pub intervals: Option<Mutex<IntervalCopies<ContentId>>>,
}

/// What a lookup or a build found and made.
#[derive(Debug)]
pub struct Outcome {
    /// The snapshot that the build started from.
    pub from: Option<ContentId>,
    /// The tokens of that snapshot.
    pub kept: usize,
    /// For each boundary, the entry that the store holds at the end of the job.
    pub held: Vec<Option<ContentId>>,
    /// The recipes of the boundaries, in the order of `held`. With a cut, they have the cut and the seals that it
    /// chains through too.
    pub recipes: Vec<BoundaryRecipe>,
    /// The boundaries of `recipes` that are not boundaries of the context: the cut that the build started from and the
    /// seals that it chains through. A response lists no snapshot there.
    pub inserted: Vec<Boundary>,
}

/// A cut: a held snapshot that is not at a boundary of the context, and the recipes of the context with the seals of
/// its chain inserted (spec 11.3c).
pub type Cut = (Boundary, Vec<BoundaryRecipe>);

/// Where a job starts: the recipes of its boundaries, the held entry of each, and the boundaries that a cut inserted.
type Start = (Vec<BoundaryRecipe>, Vec<Option<ContentId>>, Vec<Boundary>);

/// The boundaries of `chain` that `plan` does not have.
fn inserted(plan: &[BoundaryRecipe], chain: &[BoundaryRecipe]) -> Vec<Boundary> {
    chain
        .iter()
        .map(|recipe| recipe.boundary)
        .filter(|boundary| plan.iter().all(|recipe| recipe.boundary.tokens != boundary.tokens))
        .collect()
}

/// Choose where a job starts. Give the recipes, the held entry of each, and the cut that the job starts from.
///
/// The job starts from the highest held boundary, or from a held cut above it (gogi "Terms": build, the longest
/// held snapshot whose tokens begin the context). A cut holds a snapshot only when an earlier context had a
/// boundary there, such as the head of a generation after its reply. `cuts` run from the highest down.
fn choose(
    shared: &Shared,
    engine: &mut dyn Engine,
    slot: u32,
    recipes: &[BoundaryRecipe],
    cuts: &[Cut],
) -> Result<Start, ServiceError> {
    let held = lookup_all(shared, engine, slot, recipes)?;
    let reached = highest(&held).map_or(0, |index| recipes[index].boundary.tokens);
    for (cut, with_cut) in cuts.iter().filter(|(cut, _)| cut.tokens > reached) {
        let at = with_cut
            .iter()
            .position(|recipe| recipe.boundary == *cut)
            .expect("the recipes of a cut have the cut");
        if lookup_all(shared, engine, slot, &with_cut[at..=at])?[0].is_some() {
            let held = lookup_all(shared, engine, slot, with_cut)?;
            return Ok((with_cut.clone(), held, inserted(recipes, with_cut)));
        }
    }
    Ok((recipes.to_vec(), held, Vec::new()))
}

/// Look up the entry of each boundary. A miss and an ambiguous lookup give `None` (spec 6.6).
fn lookup_all(
    shared: &Shared,
    engine: &mut dyn Engine,
    slot: u32,
    recipes: &[BoundaryRecipe],
) -> Result<Vec<Option<ContentId>>, ServiceError> {
    let now = (shared.clock)();
    let mut store = shared.store.lock().expect("the store lock is poisoned");
    let mut held = Vec::with_capacity(recipes.len());
    for recipe in recipes {
        match store.lookup(recipe.recipe, &Tolerance::Exact, engine, slot, now) {
            Ok(chain) => held.push(chain.last().copied()),
            // The process treats two incomparable chains as a miss and computes the state again (spec 6.6).
            Err(StoreError::Ambiguous { .. }) => held.push(None),
            Err(error) => {
                return Err(ServiceError::internal(format!(
                    "the lookup at the boundary of message {} failed: {error}",
                    recipe.boundary.message
                )));
            }
        }
    }
    Ok(held)
}

/// The open text of a continuation after its build plan, up to a held seal inside it (spec 11.3c). The head of a cut
/// reply is a snapshot at the seam of its open rendering. A continuation of that reply starts from it when the store
/// holds it, and feeds the rest of the open text.
#[derive(Clone, Debug)]
pub struct Extension {
    /// The tokens after the end of the plan, up to the last seal.
    pub tokens: Vec<u32>,
    /// The recipes of the plan's boundaries, with the seals of the chain that the plan part holds.
    pub chain: Vec<BoundaryRecipe>,
    /// The seals after the end of the plan, in order. The job looks up the last one.
    pub seals: Vec<BoundaryRecipe>,
}

/// Look up one recipe. A miss and an ambiguous lookup give `None` (spec 6.6).
fn lookup_one(
    shared: &Shared,
    engine: &mut dyn Engine,
    slot: u32,
    recipe: &BoundaryRecipe,
) -> Result<Option<ContentId>, ServiceError> {
    Ok(lookup_all(shared, engine, slot, std::slice::from_ref(recipe))?[0])
}

/// Restore an entry into `state`. An entry that left the store since its lookup, or that the store quarantines at
/// the restore, is a miss: the result is `Refused`, and the state is as it was. The store checks the entry before it
/// writes a part.
fn restore_entry(
    shared: &Shared,
    id: &ContentId,
    engine: &mut dyn Engine,
    state: State,
) -> Result<Restored, StoreError> {
    let mut store = shared.store.lock().expect("the store lock is poisoned");
    let live = store
        .entries()
        .get(id)
        .is_some_and(|entry| entry.state == EntryState::Live);
    if !live {
        return Ok(Restored::Refused {
            part: "the entry left the store or is quarantined".to_owned(),
        });
    }
    match store.restore(id, engine, state, &shared.identity, (shared.clock)()) {
        Err(StoreError::Quarantined(_)) => Ok(Restored::Refused {
            part: "the store quarantined the entry".to_owned(),
        }),
        other => other,
    }
}

/// The recipe of an interval copy at a token of a context with `tokens` and boundaries `recipes`, with the last seal
/// before it. A token on a boundary has none: the sealed snapshot there serves a build (spec 6.13).
fn copy_recipe_at(
    root: RecipeId,
    recipes: &[BoundaryRecipe],
    tokens: &[u32],
    position: usize,
) -> Option<(RecipeId, RecipeId, usize)> {
    if recipes.iter().any(|recipe| recipe.boundary.tokens == position) || position > tokens.len() {
        return None;
    }
    let (base, at) = recipes
        .iter()
        .rev()
        .find(|recipe| recipe.boundary.tokens < position)
        .map_or((root, 0), |recipe| (recipe.recipe, recipe.boundary.tokens));
    let sealed = recipe_boundary(&base, &tokens[at..position], position as u64);
    Some((recipe_not_sealed(&sealed), base, at))
}

/// The longest interval copy of a context above the token `above`, when the store holds one: its position, its
/// content id, and the last seal before it with its position (spec 6.13). A build and a dry run use it.
fn best_copy(
    shared: &Shared,
    recipes: &[BoundaryRecipe],
    tokens: &[u32],
    above: usize,
) -> Option<(usize, ContentId, RecipeId, usize)> {
    shared.intervals.as_ref()?;
    let store = shared.store.lock().expect("the store lock is poisoned");
    store
        .entries()
        .values()
        .filter(|entry| {
            entry.reason == Reason::Interval
                && entry.state == EntryState::Live
                && entry.engine == shared.engine_name
                && entry.identity == shared.identity
        })
        .filter_map(|entry| {
            let position = entry.coverage.iter().map(|span| span.end).max()? as usize;
            if position <= above {
                return None;
            }
            let (recipe, base, at) = copy_recipe_at(shared.root, recipes, tokens, position)?;
            (recipe == entry.recipe).then_some((position, entry.content_id, base, at))
        })
        .max_by_key(|(position, ..)| *position)
}

/// The highest boundary that the store holds.
fn highest(held: &[Option<ContentId>]) -> Option<usize> {
    held.iter().rposition(Option::is_some)
}

/// The job of a dry run: look up the prefixes and report. It restores and feeds nothing.
pub struct LookupBody {
    shared: Arc<Shared>,
    recipes: Vec<BoundaryRecipe>,
    cuts: Vec<Cut>,
    extension: Option<Extension>,
    /// The tokens of the plan.
    tokens: Arc<[u32]>,
    /// The tokens that the slot of the job would hold: the context, or all the open rendering of a continuation.
    room: usize,
    tx: Option<Sender<Result<Outcome, ServiceError>>>,
}

impl LookupBody {
    /// Make the body. The result goes to `tx`.
    /// Make the body. `room` is the tokens that the job's slot would hold. The result goes to `tx`.
    pub fn new(
        shared: Arc<Shared>,
        recipes: Vec<BoundaryRecipe>,
        tokens: Arc<[u32]>,
        room: usize,
        tx: Sender<Result<Outcome, ServiceError>>,
    ) -> Self {
        Self {
            shared,
            recipes,
            cuts: Vec::new(),
            extension: None,
            tokens,
            room,
            tx: Some(tx),
        }
    }

    /// Let the lookup start from a held cut (see [`ContextBody::cuts`]).
    pub fn cuts(mut self, cuts: Vec<Cut>) -> Self {
        self.cuts = cuts;
        self
    }

    /// Let the lookup start from a held seam (see [`ContextBody::extension`]).
    pub fn extension(mut self, extension: Option<Extension>) -> Self {
        self.extension = extension;
        self
    }

    fn look(&self, engine: &mut dyn Engine) -> Result<Outcome, ServiceError> {
        let pool = self.shared.pool.lock().expect("the slot pool lock is poisoned");
        let Some(slot) = slots::lowest_fitting(engine, &self.shared.limits, &pool, self.room) else {
            return Err(too_large(self.room));
        };
        drop(pool);
        if let Some(extension) = &self.extension
            && let Some(seam) = extension.seals.last()
            && let Some(id) = lookup_one(&self.shared, engine, slot, seam)?
        {
            let held = lookup_all(&self.shared, engine, slot, &extension.chain)?;
            let mut inserted = inserted(&self.recipes, &extension.chain);
            inserted.extend(extension.seals.iter().map(|seal| seal.boundary));
            return Ok(Outcome {
                from: Some(id),
                kept: seam.boundary.tokens,
                held,
                recipes: extension.chain.clone(),
                inserted,
            });
        }
        let (recipes, held, inserted) = choose(&self.shared, engine, slot, &self.recipes, &self.cuts)?;
        let start = highest(&held);
        let above = start.map_or(0, |index| recipes[index].boundary.tokens);
        // A build restores an interval copy above the highest held boundary, so a dry run reports it (spec 6.13).
        if let Some((position, id, ..)) = best_copy(&self.shared, &recipes, &self.tokens, above) {
            return Ok(Outcome {
                from: Some(id),
                kept: position,
                held,
                recipes,
                inserted,
            });
        }
        Ok(Outcome {
            from: start.and_then(|index| held[index]),
            kept: above,
            held,
            recipes,
            inserted,
        })
    }
}

impl JobBody for LookupBody {
    fn request(&self) -> &'static str {
        "dry_run"
    }

    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        assert_eq!(kind, Kind::Read, "a dry run has one quantum, of the kind read");
        let result = self.look(engines[self.shared.engine_index].as_mut());
        let failed = result.is_err();
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
        if failed { Ran::Failed } else { Ran::Done }
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Err(ServiceError::Unavailable {
                engine: engine.to_owned(),
                state,
            }));
        }
    }
}

/// The error of a context that no slot holds. The limit that breaks is the capacity of the slot, which identity gives
/// as `context_tokens` (spec 4.3), whatever kind of job asks.
fn too_large(tokens: usize) -> ServiceError {
    ServiceError::TooLarge {
        param: "context_tokens".to_owned(),
        message: format!(
            "no slot of the engine holds {tokens} tokens; shorten the context, or serve a model with larger slots"
        ),
    }
}

/// What a job asks of its slot after the build. A [`Then`] gives one want at a time.
#[derive(Clone, Debug, PartialEq)]
pub enum Want {
    /// Feed tokens after the rows of the slot. With `logits`, give the logit row of the last token. The feed takes
    /// one quantum of `kind` for each pass.
    Feed {
        /// The tokens.
        tokens: Vec<u32>,
        /// True when the caller wants the logit row of the last token.
        logits: bool,
        /// The kind of the quanta (spec 5.2).
        kind: Kind,
    },
    /// Keep a copy of the state of the slot in memory.
    Mark,
    /// Put the copy that `Mark` kept back into the slot.
    Rewind,
    /// Seal the state and hold a snapshot under a recipe (spec 11.3c).
    Snapshot {
        /// The recipe of the snapshot.
        recipe: RecipeId,
        /// The reason of the store (spec 6.4).
        reason: Reason,
    },
    /// The work is complete.
    Done,
    /// The client went away. The job ends as cancelled and holds nothing more (spec 5.9).
    Cancel,
}

/// What the slot gives back for the last want.
#[derive(Clone, Debug, PartialEq)]
pub enum Got {
    /// The build is complete. The slot holds the context when the work needs its state. The recipes are the seals of
    /// the state: the boundaries of the context, and the cut and the seals of its chain when the build started from a
    /// cut.
    Built(Vec<BoundaryRecipe>),
    /// The build restored a held seal inside the open text of a continuation (see [`Extension`]), such as the head of
    /// a cut reply. The slot holds the tokens to the last of the recipes, which are the seals of the state.
    Resumed(Vec<BoundaryRecipe>),
    /// The feed is complete. The row is there when the feed asked for it.
    Fed(Option<Vec<f32>>),
    /// The mark or the rewind is complete.
    Copied,
    /// The store holds the snapshot, or holds its bytes under another recipe (`None`).
    Held(Option<ContentId>),
}

/// The work after a build, as a state machine with no I/O. The body drives it on the group thread.
pub trait Then: Send + 'static {
    /// True when the work needs the state of the context in the slot. Then a build whose head the store holds
    /// restores the head.
    const NEEDS_STATE: bool;

    /// The kind of request that the work serves, for the span `nagaya.job` (spec 9.2).
    const REQUEST: &'static str;

    /// Give the next want, from what the last want gave.
    fn next(&mut self, got: Got) -> Result<Want, ServiceError>;
}

/// No work after the build.
pub struct NoThen;

impl Then for NoThen {
    const NEEDS_STATE: bool = false;
    const REQUEST: &'static str = "build";

    fn next(&mut self, _got: Got) -> Result<Want, ServiceError> {
        Ok(Want::Done)
    }
}

/// What a context job gives back: what the build found and made, and the work after it.
pub struct Finished<T> {
    /// The build.
    pub outcome: Outcome,
    /// The work after the build, with its results.
    pub then: T,
}

/// The feed that runs: its tokens, the kind of its quanta, and the rows of the slot when it started.
struct Feeding {
    tokens: Vec<u32>,
    kind: Kind,
    base: usize,
}

/// What the body does in one of its later quanta.
enum Unit {
    /// A step of the build.
    Build(Step),
    /// A pass over the tokens `start..end` of the feed that runs. With `row`, it gives the row of its last token.
    Feed {
        start: usize,
        end: usize,
        row: bool,
    },
    Mark,
    Rewind,
    Snapshot(RecipeId, Reason),
    /// An interval copy of the state after the last pass (spec 6.13).
    Interval,
}

/// A pass that began: its tokens, the position of its first token, and what its end gives.
struct Pending {
    tokens: Vec<u32>,
    from: usize,
    /// True when the pass gives the row of its last token.
    row: bool,
    /// True for a pass of the build, false for a pass of a feed of the work.
    build: bool,
}

/// A copy of the state of a slot that `Mark` kept in host memory. The charge ends when the copy drops.
struct Marked {
    snapshot: Snapshot,
    /// The last seal of the state, and the tokens that the slot held.
    sealed: (RecipeId, usize),
    fed: usize,
    /// The charge. It is the last field, so the copy is gone before the ledger gets the bytes back.
    _staging: Staging,
}

/// The job of a context: a build, then the work of a [`Then`].
pub struct ContextBody<T: Then> {
    shared: Arc<Shared>,
    tokens: Arc<[u32]>,
    recipes: Vec<BoundaryRecipe>,
    /// The cuts where the job may start, from the highest down.
    cuts: Vec<Cut>,
    /// The open text of a continuation after the plan. The job may start from a held seal in it.
    extension: Option<Extension>,
    /// True when the job started from a seal of the extension.
    resumed: bool,
    /// The boundaries of `recipes` that are not boundaries of the context.
    inserted: Vec<Boundary>,
    /// The tokens that the slot must hold when the job takes it: the context, and what the work feeds after it when
    /// the job knows that.
    room: usize,
    lease: Option<SlotLease>,
    started: bool,
    /// True while the build runs, before its work starts.
    building: bool,
    units: VecDeque<Unit>,
    feeding: Option<Feeding>,
    /// The row that the last pass of the feed gave.
    row: Option<Vec<f32>>,
    /// The state that `Mark` kept, with the ledger charge of its bytes.
    mark: Option<Marked>,
    held: Vec<Option<ContentId>>,
    from: Option<ContentId>,
    kept: usize,
    /// The tokens that the slot holds, in order.
    fed: Vec<u32>,
    /// The recipe of the last seal of the state and its position. An interval copy names the recipe of a seal at its
    /// own position, which chains from this one (spec 6.2, 6.13).
    sealed: Option<(RecipeId, usize)>,
    /// True while the interval tracker holds the slot of the job.
    tracked: bool,
    /// What the last pass gave, while the interval copy after it runs.
    after_copy: Option<Got>,
    /// The passes that began and have not ended, in order. On an engine with stages, a block can begin before the
    /// block before it ends (engine-abi D1).
    in_flight: VecDeque<Pending>,
    then: Option<T>,
    tx: Option<Sender<Result<Finished<T>, ServiceError>>>,
}

/// The job of a build alone.
pub type BuildBody = ContextBody<NoThen>;

impl BuildBody {
    /// Make the body of a build. `tokens` are the tokens of the context, without a generation prompt.
    pub fn new(
        shared: Arc<Shared>,
        tokens: Arc<[u32]>,
        recipes: Vec<BoundaryRecipe>,
        tx: Sender<Result<Finished<NoThen>, ServiceError>>,
    ) -> Self {
        let room = tokens.len();
        ContextBody::with(shared, tokens, recipes, room, NoThen, tx)
    }
}

impl<T: Then> ContextBody<T> {
    /// Make the body of a context and its work. The job takes the lowest free slot that holds `room` tokens. A job
    /// that no slot holds, or a feed past what its slot holds, is `too_large` and names `context_tokens`. With no recipe,
    /// the job builds nothing: the work starts on an empty slot.
    pub fn with(
        shared: Arc<Shared>,
        tokens: Arc<[u32]>,
        recipes: Vec<BoundaryRecipe>,
        room: usize,
        then: T,
        tx: Sender<Result<Finished<T>, ServiceError>>,
    ) -> Self {
        assert!(room >= tokens.len(), "the room of a job holds its context");
        assert_eq!(
            recipes.last().map_or(0, |recipe| recipe.boundary.tokens),
            tokens.len(),
            "the last boundary of a context is its end"
        );
        Self {
            shared,
            tokens,
            recipes,
            cuts: Vec::new(),
            extension: None,
            resumed: false,
            inserted: Vec::new(),
            room,
            lease: None,
            started: false,
            building: true,
            units: VecDeque::new(),
            feeding: None,
            row: None,
            mark: None,
            held: Vec::new(),
            from: None,
            kept: 0,
            fed: Vec::new(),
            sealed: None,
            tracked: false,
            after_copy: None,
            in_flight: VecDeque::new(),
            then: Some(then),
            tx: Some(tx),
        }
    }

    /// Let the job start from a held cut above its highest held boundary. A cut is the end of a message that the
    /// client did not mark. `cuts` run from the highest down.
    pub fn cuts(mut self, cuts: Vec<Cut>) -> Self {
        self.cuts = cuts;
        self
    }

    /// Let the job start from a held seam after its plan: the head of a cut reply that a continuation sends back
    /// open (spec 11.3c). The job never seals at the seam itself.
    pub fn extension(mut self, extension: Option<Extension>) -> Self {
        self.extension = extension;
        self
    }

    /// What the build gives the work when it is complete.
    fn built(&self) -> Got {
        let mut recipes = self.recipes.clone();
        match &self.extension {
            Some(extension) if self.resumed => {
                recipes.extend(&extension.seals);
                Got::Resumed(recipes)
            }
            _ => Got::Built(recipes),
        }
    }

    /// Restore the last seal of the extension, when the store holds it: the head of a cut reply, for example. It gives
    /// true on a restore. A refused restore changed nothing (spec 6.8), and the build goes on from the held
    /// boundaries.
    fn restore_seam(&mut self, engine: &mut dyn Engine) -> Result<bool, ServiceError> {
        let Some(extension) = self.extension.clone() else {
            return Ok(false);
        };
        let Some(seam) = extension.seals.last().copied() else {
            return Ok(false);
        };
        let slot = self.slot();
        let Some(id) = lookup_one(&self.shared, engine, slot, &seam)? else {
            return Ok(false);
        };
        let restored = restore_entry(&self.shared, &id, engine, self.state());
        match restored {
            Ok(Restored::Done) => {
                let tokens = seam.boundary.tokens;
                let rows = self.rows(engine)?;
                if rows != tokens {
                    return Err(ServiceError::internal(format!(
                        "the restored seal of slot {slot} holds {rows} rows, but the seal is at token {tokens}"
                    )));
                }
                self.held = lookup_all(&self.shared, engine, slot, &extension.chain)?;
                self.inserted = inserted(&self.recipes, &extension.chain);
                self.inserted.extend(extension.seals.iter().map(|seal| seal.boundary));
                self.recipes = extension.chain;
                self.from = Some(id);
                self.kept = tokens;
                self.fed = self.tokens.to_vec();
                self.fed.extend(&extension.tokens);
                self.sealed = Some((seam.recipe, tokens));
                self.resumed = true;
                Ok(true)
            }
            Ok(Restored::Refused { .. }) => Ok(false),
            Ok(Restored::Failed { part }) => Err(ServiceError::internal(format!(
                "the restore of the seal {} stopped at part `{part}` and poisoned slot {slot}: {}",
                id.hex(),
                engine.last_error().unwrap_or_default()
            ))),
            Err(error) => Err(ServiceError::internal(format!(
                "the restore of the seal {} failed: {error}",
                id.hex()
            ))),
        }
    }

    fn slot(&self) -> u32 {
        self.lease
            .as_ref()
            .expect("a job holds a slot after its first quantum")
            .slot()
    }

    fn state(&self) -> State {
        State::Slot(self.slot())
    }

    fn rows(&self, engine: &mut dyn Engine) -> Result<usize, ServiceError> {
        Ok(engine
            .slots()
            .ok_or_else(|| ServiceError::internal("the engine lost its slot capability"))?
            .slot_high(self.slot()) as usize)
    }

    /// Send the result to the caller. The slot is free before the caller hears, so the caller sees it free.
    fn send(&mut self, result: Result<Finished<T>, ServiceError>) {
        // The store keeps the last copy after the context ends. The cap and eviction remove it (spec 6.13).
        if self.tracked {
            self.tracked = false;
            let slot = u64::from(self.slot());
            if let Some(intervals) = &self.shared.intervals {
                intervals
                    .lock()
                    .expect("the interval lock is poisoned")
                    .untrack(slot)
                    .expect("a tracked slot is in the tracker");
            }
        }
        self.lease = None;
        // The mark is host memory. Free it and its charge before the caller hears the result.
        self.mark = None;
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
    }

    fn finish(&mut self) -> Ran {
        let finished = Finished {
            outcome: Outcome {
                from: self.from,
                kept: self.kept,
                held: std::mem::take(&mut self.held),
                recipes: std::mem::take(&mut self.recipes),
                inserted: std::mem::take(&mut self.inserted),
            },
            then: self.then.take().expect("a job finishes once"),
        };
        self.send(Ok(finished));
        Ran::Done
    }

    /// The first quantum: take a slot, look up the prefixes, restore the longest, and plan the rest.
    fn start(&mut self, engine: &mut dyn Engine) -> Result<Ran, ServiceError> {
        let lease =
            slots::take(&self.shared.pool, engine, &self.shared.limits, self.room).map_err(|reason| match reason {
                NoSlot::TooLarge => too_large(self.room),
                NoSlot::Busy => ServiceError::Busy(format!(
                    "every slot that holds {} tokens is in use; try again, or send the request as batch",
                    self.room
                )),
            })?;
        let slot = lease.slot();
        self.lease = Some(lease);
        let state = self.state();
        // A slot keeps what its last context left. A job starts from an empty one.
        let slots = engine
            .slots()
            .ok_or_else(|| ServiceError::internal("the engine lost its slot capability"))?;
        if slots.slot_clear(slot) != WriteResult::Ok {
            return Err(ServiceError::internal(format!(
                "the engine could not clear slot {slot}: {}",
                engine.last_error().unwrap_or_default()
            )));
        }
        self.sealed = Some((self.shared.root, 0));
        // The longest held snapshot wins: a seal inside the open text of a continuation is after every boundary.
        if self.restore_seam(engine)? {
            self.track(engine)?;
            let got = self.built();
            return self.next_want(engine, got);
        }
        if self.recipes.is_empty() {
            self.track(engine)?;
            let got = self.built();
            return self.next_want(engine, got);
        }
        let (recipes, held, inserted) = choose(&self.shared, engine, slot, &self.recipes, &self.cuts)?;
        (self.recipes, self.held, self.inserted) = (recipes, held, inserted);
        let last = self.recipes.len() - 1;
        let mut candidate = highest(&self.held);
        let above = candidate.map_or(0, |index| self.recipes[index].boundary.tokens);
        if self.restore_copy(engine, above)? {
            candidate = None;
        }
        while let Some(index) = candidate {
            let id = self.held[index].expect("the highest held boundary has an entry");
            let tokens = self.recipes[index].boundary.tokens;
            // The head is held. A build alone needs no state in the engine.
            if index == last && !T::NEEDS_STATE {
                self.from = Some(id);
                self.kept = tokens;
                break;
            }
            let restored = restore_entry(&self.shared, &id, engine, state);
            match restored {
                Ok(Restored::Done) => {
                    let rows = self.rows(engine)?;
                    if rows != tokens {
                        return Err(ServiceError::internal(format!(
                            "the restored state of slot {slot} holds {rows} rows, but its boundary is at token {tokens}"
                        )));
                    }
                    self.from = Some(id);
                    self.kept = tokens;
                    self.fed = self.tokens[..tokens].to_vec();
                    self.sealed = Some((self.recipes[index].recipe, tokens));
                    break;
                }
                // A refused restore changed nothing (spec 6.8). The build tries the next shorter prefix.
                Ok(Restored::Refused { .. }) => {
                    candidate = self.held[..index].iter().rposition(Option::is_some);
                }
                Ok(Restored::Failed { part }) => {
                    return Err(ServiceError::internal(format!(
                        "the restore of the entry {} stopped at part `{part}` and poisoned slot {slot}: {}",
                        id.hex(),
                        engine.last_error().unwrap_or_default()
                    )));
                }
                Err(error) => {
                    return Err(ServiceError::internal(format!(
                        "the restore of the entry {} failed: {error}",
                        id.hex()
                    )));
                }
            }
        }
        self.track(engine)?;
        let boundaries: Vec<_> = self.recipes.iter().map(|recipe| recipe.boundary).collect();
        let steps = plan_steps(self.kept, &boundaries, self.shared.chunk);
        if steps.is_empty() {
            let got = self.built();
            return self.next_want(engine, got);
        }
        let kinds = steps.iter().map(Step::kind).collect();
        self.units = steps.into_iter().map(Unit::Build).collect();
        Ok(Ran::More(kinds))
    }

    /// Ask the work for its next want, and plan the quanta of the want.
    fn next_want(&mut self, engine: &mut dyn Engine, got: Got) -> Result<Ran, ServiceError> {
        self.building = false;
        let want = self
            .then
            .as_mut()
            .expect("the work runs before the job finishes")
            .next(got)?;
        let kind = match want {
            Want::Done => return Ok(self.finish()),
            Want::Cancel => return Ok(Ran::Cancelled),
            Want::Mark => {
                self.units.push_back(Unit::Mark);
                Kind::Copy
            }
            Want::Rewind => {
                if self.mark.is_none() {
                    return Err(ServiceError::internal("a job asked for a rewind before a mark"));
                }
                self.units.push_back(Unit::Rewind);
                Kind::Copy
            }
            Want::Snapshot { recipe, reason } => {
                self.units.push_back(Unit::Snapshot(recipe, reason));
                Kind::Copy
            }
            Want::Feed { tokens, logits, kind } => {
                if tokens.is_empty() {
                    return Err(ServiceError::internal("a job asked to feed no token"));
                }
                let base = self.rows(engine)?;
                let end = base + tokens.len();
                if !slots::fits(engine, &self.shared.limits, self.slot(), end) {
                    return Err(too_large(end));
                }
                let wanted: Vec<usize> = if logits { vec![tokens.len() - 1] } else { Vec::new() };
                let max_rows = self.shared.limits.max_logit_rows as usize;
                if logits && max_rows == 0 {
                    return Err(ServiceError::internal("the engine gives no logit row in a pass"));
                }
                for range in plan_chunks(tokens.len(), &wanted, self.shared.chunk, max_rows.max(1)) {
                    self.units.push_back(Unit::Feed {
                        start: range.start,
                        end: range.end,
                        row: logits && range.end == tokens.len(),
                    });
                }
                self.feeding = Some(Feeding { tokens, kind, base });
                kind
            }
        };
        Ok(Ran::More(vec![kind; self.units.len()]))
    }

    /// The kind of the quantum of a unit.
    fn kind_of(&self, unit: &Unit) -> Kind {
        match unit {
            Unit::Build(step) => step.kind(),
            Unit::Feed { .. } => self.feeding.as_ref().expect("a pass belongs to a feed").kind,
            Unit::Mark | Unit::Rewind | Unit::Snapshot(..) | Unit::Interval => Kind::Copy,
        }
    }

    /// One later quantum: a step of the build, or a unit of the work.
    fn advance(&mut self, kind: Kind, engine: &mut dyn Engine) -> Result<Ran, ServiceError> {
        let unit = self.units.pop_front().expect("a quantum has a unit");
        assert_eq!(
            self.kind_of(&unit),
            kind,
            "the scheduler gave a quantum of another kind than the unit"
        );
        if matches!(unit, Unit::Build(Step::Feed { .. }) | Unit::Feed { .. }) {
            let segment = self.begin(unit)?;
            let result = engine.pass(&[segment], &[]);
            let last_error = engine.last_error();
            return self.passed(result, last_error, engine);
        }
        let got = match unit {
            Unit::Build(Step::Feed { .. }) | Unit::Feed { .. } => unreachable!("a pass begins above"),
            Unit::Build(Step::Snapshot { boundary }) => {
                let recipe = self.recipes[boundary].recipe;
                let reason = if boundary + 1 == self.recipes.len() {
                    Reason::PrefillDone
                } else {
                    Reason::Anchor
                };
                let message = self.recipes[boundary].boundary.message;
                let id = self.snapshot(engine, recipe, reason, &format!("the boundary of message {message}"))?;
                self.held[boundary] = id;
                None
            }
            Unit::Mark => {
                // A second mark replaces the first. Release the first before the charge of the second.
                self.mark = None;
                let state = self.state();
                let staging = self.stage(engine, state, "keep the state of the slot")?;
                let snapshot = park::park_snapshot(engine, state).map_err(|message| {
                    ServiceError::internal(format!("cannot keep the state of the slot: {message}"))
                })?;
                self.mark = Some(Marked {
                    snapshot,
                    sealed: self.sealed.expect("a job knows its last seal after its start"),
                    fed: self.fed.len(),
                    _staging: staging,
                });
                Some(Got::Copied)
            }
            Unit::Rewind => {
                self.rewind(engine)?;
                Some(Got::Copied)
            }
            Unit::Snapshot(recipe, reason) => {
                let id = self.snapshot(engine, recipe, reason, "the end of the reply")?;
                Some(Got::Held(id))
            }
            Unit::Interval => {
                self.interval_copy(engine)?;
                self.after_copy.take()
            }
        };
        self.after(got, false, engine)
    }

    /// Begin the pass of a feed unit: check its first position and make its segment.
    fn begin(&mut self, unit: Unit) -> Result<Segment, ServiceError> {
        let pending = match unit {
            Unit::Build(Step::Feed { from, to }) => Pending {
                tokens: self.tokens[from..to].to_vec(),
                from,
                row: false,
                build: true,
            },
            Unit::Feed { start, end, row } => {
                let feeding = self.feeding.as_ref().expect("a pass belongs to a feed");
                Pending {
                    tokens: feeding.tokens[start..end].to_vec(),
                    from: feeding.base + start,
                    row,
                    build: false,
                }
            }
            _ => unreachable!("only a feed unit begins a pass"),
        };
        let expected = self.fed.len() + self.in_flight.iter().map(|pending| pending.tokens.len()).sum::<usize>();
        if pending.from != expected {
            return Err(ServiceError::internal(format!(
                "a pass starts at token {}, but the job knows {expected} tokens in the slot",
                pending.from
            )));
        }
        let to = pending.from + pending.tokens.len();
        let segment = Segment {
            control: Vec::new(),
            state: self.state(),
            inputs: SegmentInputs::Tokens(pending.tokens.clone()),
            positions: (pending.from as u64..to as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if pending.row { vec![to as u64 - 1] } else { Vec::new() },
            },
        };
        self.in_flight.push_back(pending);
        Ok(segment)
    }

    /// The end of the oldest pass that began. `last_error` is the engine's message when the pass failed.
    fn passed(
        &mut self,
        result: Result<PassOut, PassError>,
        last_error: Option<String>,
        engine: &mut dyn Engine,
    ) -> Result<Ran, ServiceError> {
        let pending = self.in_flight.pop_front().expect("a pass that ends began");
        let (from, to) = (pending.from, pending.from + pending.tokens.len());
        let out = match result {
            Ok(out) if out.status == PassStatus::Ok => {
                self.fed.extend_from_slice(&pending.tokens);
                out
            }
            Ok(out) => {
                return Err(ServiceError::internal(format!(
                    "a pass over tokens {from} to {to} ended with {:?}: {}",
                    out.status,
                    last_error.unwrap_or_default()
                )));
            }
            Err(error) => {
                return Err(ServiceError::internal(format!(
                    "a pass over tokens {from} to {to} failed: {} ({:?})",
                    error.message, error.kind
                )));
            }
        };
        if pending.row {
            let mut rows = out.outputs.iter().filter(|output| output.name == "logits");
            let (Some(output), None) = (rows.next(), rows.next()) else {
                return Err(ServiceError::internal(format!(
                    "a pass over tokens {from} to {to} asked for one logit row and did not give exactly one"
                )));
            };
            let row = logit_row(&output.bytes, self.shared.vocab)
                .map_err(|error| ServiceError::internal(format!("the pass over tokens {from} to {to}: {error}")))?;
            self.row = Some(row);
        }
        let got = if !pending.build && self.units.is_empty() && self.in_flight.is_empty() {
            self.feeding = None;
            Some(Got::Fed(self.row.take()))
        } else {
            None
        };
        self.after(got, true, engine)
    }

    /// What follows a unit: an interval copy after a pass that crossed a multiple of the grain, the next unit, or the
    /// next want of the work.
    fn after(&mut self, got: Option<Got>, fed: bool, engine: &mut dyn Engine) -> Result<Ran, ServiceError> {
        // A copy is due after a pass that crossed a multiple of the grain, as the next quantum of the job. When the
        // pass ends on a boundary, the snapshot there holds the state, so the copy is not made.
        if fed && self.copy_due(engine)? && !matches!(self.units.front(), Some(Unit::Build(Step::Snapshot { .. }))) {
            self.units.push_front(Unit::Interval);
            self.after_copy = got;
            return Ok(Ran::First(vec![Kind::Copy]));
        }
        if !self.units.is_empty() || !self.in_flight.is_empty() {
            return Ok(Ran::Done);
        }
        match got {
            Some(got) => self.next_want(engine, got),
            None if self.building => {
                let got = self.built();
                self.next_want(engine, got)
            }
            None => unreachable!("the last unit of a want gives back what it did"),
        }
    }

    /// Start to track the slot for interval copies at its row count (spec 6.13).
    fn track(&mut self, engine: &mut dyn Engine) -> Result<(), ServiceError> {
        let Some(intervals) = &self.shared.intervals else {
            return Ok(());
        };
        let rows = self.rows(engine)?;
        if rows != self.fed.len() {
            return Err(ServiceError::internal(format!(
                "slot {} holds {rows} rows, but the job knows {} tokens",
                self.slot(),
                self.fed.len()
            )));
        }
        intervals
            .lock()
            .expect("the interval lock is poisoned")
            .track(u64::from(self.slot()), rows as u64)
            .map_err(|error| ServiceError::internal(format!("the interval tracker refused the slot: {error:?}")))?;
        self.tracked = true;
        Ok(())
    }

    /// Track the slot again after a rewind moved its rows back. The context keeps its copy.
    fn retrack(&mut self, engine: &mut dyn Engine) -> Result<(), ServiceError> {
        let shared = Arc::clone(&self.shared);
        let Some(intervals) = &shared.intervals else {
            return Ok(());
        };
        let copy = intervals
            .lock()
            .expect("the interval lock is poisoned")
            .untrack(u64::from(self.slot()))
            .map_err(|error| ServiceError::internal(format!("the interval tracker lost the slot: {error:?}")))?;
        self.tracked = false;
        self.track(engine)?;
        if let Some(copy) = copy {
            intervals
                .lock()
                .expect("the interval lock is poisoned")
                .stored(u64::from(self.slot()), copy)
                .expect("the slot is tracked");
        }
        Ok(())
    }

    /// Tell the tracker the rows after a pass. True when an interval copy is due (spec 6.13).
    fn copy_due(&mut self, engine: &mut dyn Engine) -> Result<bool, ServiceError> {
        let Some(intervals) = &self.shared.intervals else {
            return Ok(false);
        };
        let rows = self.rows(engine)?;
        if rows != self.fed.len() {
            return Err(ServiceError::internal(format!(
                "slot {} holds {rows} rows after a pass, but the job fed {} tokens",
                self.slot(),
                self.fed.len()
            )));
        }
        let due = intervals
            .lock()
            .expect("the interval lock is poisoned")
            .advance(u64::from(self.slot()), rows as u64)
            .map_err(|error| ServiceError::internal(format!("the interval tracker refused the rows: {error:?}")))?;
        Ok(due.is_some())
    }

    /// Put an interval copy of the state in the store, with no seal (spec 6.13). The store then removes the previous
    /// copy of the context. A refused charge or a full store skips the copy: the next multiple tries again.
    fn interval_copy(&mut self, engine: &mut dyn Engine) -> Result<(), ServiceError> {
        let state = self.state();
        let position = self.fed.len();
        let (base, at) = self.sealed.expect("a job knows its last seal after its start");
        let sealed = recipe_boundary(&base, &self.fed[at..position], position as u64);
        let _staging = match self.stage(engine, state, "make an interval copy") {
            Ok(staging) => staging,
            Err(ServiceError::ContextRefused(error)) => {
                tracing::warn!(
                    "no interval copy at token {position} of slot {}: the ledger refused its host bytes: {error}",
                    self.slot()
                );
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let snapshot = park::park_snapshot(engine, state).map_err(|message| {
            ServiceError::internal(format!("cannot read the state for an interval copy: {message}"))
        })?;
        let facts = EngineFacts::of(&self.shared.engine_name, &self.shared.limits);
        let now_ms = (self.shared.clock)();
        let mut store = self.shared.store.lock().expect("the store lock is poisoned");
        let put = store.put(Put {
            snapshot: &snapshot,
            recipe: recipe_not_sealed(&sealed),
            previous_recipe: None,
            chains_from: None,
            engine: facts,
            identity: self.shared.identity,
            owner: COUNCIL_OWNER,
            tier: Tier::Host,
            reason: Reason::Interval,
            sealed_recipe: Some(sealed),
            provenance: None,
            now_ms,
        });
        let id = match put {
            Ok(id) => id,
            Err(StoreError::NoSpace) => {
                tracing::warn!(
                    "no interval copy at token {position} of slot {}: the store has no room, or pins hold the cap",
                    self.slot()
                );
                return Ok(());
            }
            Err(error) => {
                return Err(ServiceError::internal(format!(
                    "the store refused the interval copy at token {position}: {error}"
                )));
            }
        };
        let intervals = self.shared.intervals.as_ref().expect("a due copy has a tracker");
        let previous = intervals
            .lock()
            .expect("the interval lock is poisoned")
            .stored(u64::from(self.slot()), id)
            .expect("the slot is tracked");
        if let Some(previous) = previous.filter(|previous| *previous != id) {
            store
                .remove(&previous, now_ms)
                .map_err(|error| ServiceError::internal(format!("cannot remove the previous copy: {error}")))?;
        }
        Ok(())
    }

    /// Restore the longest interval copy of the context above the token `above`, when the store holds one. A restore
    /// writes the copy's parts, and the build replays the tokens after it (spec 6.13). It gives true on a restore.
    fn restore_copy(&mut self, engine: &mut dyn Engine, above: usize) -> Result<bool, ServiceError> {
        let Some((position, id, base, at)) = best_copy(&self.shared, &self.recipes, &self.tokens, above) else {
            return Ok(false);
        };
        let slot = self.slot();
        let restored = restore_entry(&self.shared, &id, engine, self.state());
        match restored {
            Ok(Restored::Done) => {
                let rows = self.rows(engine)?;
                if rows != position {
                    return Err(ServiceError::internal(format!(
                        "the interval copy {} in slot {slot} holds {rows} rows, but its position is {position}",
                        id.hex()
                    )));
                }
                self.from = Some(id);
                self.kept = position;
                self.fed = self.tokens[..position].to_vec();
                self.sealed = Some((base, at));
                Ok(true)
            }
            // A refused restore changed nothing (spec 6.8). The build uses the held boundaries.
            Ok(Restored::Refused { .. }) => Ok(false),
            Ok(Restored::Failed { part }) => Err(ServiceError::internal(format!(
                "the restore of the interval copy {} stopped at part `{part}` and poisoned slot {slot}: {}",
                id.hex(),
                engine.last_error().unwrap_or_default()
            ))),
            Err(error) => Err(ServiceError::internal(format!(
                "the restore of the interval copy {} failed: {error}",
                id.hex()
            ))),
        }
    }

    /// Charge the host bytes of a copy of the state, before the copy runs (spec 4.2). A refusal is an error that the
    /// client sees, and the engine is as it was.
    fn stage(&self, engine: &mut dyn Engine, state: State, what: &str) -> Result<Staging, ServiceError> {
        let bytes = staging::state_bytes(engine, &self.shared.limits, state)
            .map_err(|message| ServiceError::internal(format!("cannot size the copy to {what}: {message}")))?;
        Staging::take(&self.shared.ports, &self.shared.engine_name, bytes)
    }

    /// Put the state that `Mark` kept back into the slot.
    fn rewind(&mut self, engine: &mut dyn Engine) -> Result<(), ServiceError> {
        let slot = self.slot();
        let mark = &self.mark.as_ref().expect("a rewind follows a mark").snapshot;
        let slots = engine
            .slots()
            .ok_or_else(|| ServiceError::internal("the engine lost its slot capability"))?;
        if slots.slot_clear(slot) != WriteResult::Ok {
            return Err(ServiceError::internal(format!(
                "the engine could not clear slot {slot} for a rewind: {}",
                engine.last_error().unwrap_or_default()
            )));
        }
        let (sealed, fed) = {
            let marked = self.mark.as_ref().expect("a rewind follows a mark");
            (marked.sealed, marked.fed)
        };
        match write_snapshots(engine, State::Slot(slot), std::slice::from_ref(mark)) {
            Ok(Restored::Done) => {
                self.sealed = Some(sealed);
                self.fed.truncate(fed);
                self.retrack(engine)
            }
            Ok(stopped) => Err(ServiceError::internal(format!(
                "the rewind of slot {slot} stopped: {stopped:?}: {}",
                engine.last_error().unwrap_or_default()
            ))),
            Err(error) => Err(ServiceError::internal(format!(
                "the rewind of slot {slot} failed: {error}"
            ))),
        }
    }

    /// Seal the context, read its parts, and put the snapshot in the store (spec 11.3c).
    fn snapshot(
        &mut self,
        engine: &mut dyn Engine,
        recipe: RecipeId,
        reason: Reason,
        place: &str,
    ) -> Result<Option<ContentId>, ServiceError> {
        let state = self.state();
        // The buffer of the copy is host memory. The charge stays until the store has taken the snapshot.
        let _staging = self.stage(engine, state, &format!("read the state at {place}"))?;
        if let Some(seal) = engine.seal() {
            seal.seal(state)
                .map_err(|error| ServiceError::internal(format!("the seal at {place} failed: {error:?}")))?;
        }
        // The recipe chains from the last seal before this one. The store indexes the snapshot by it.
        let chains_from = self.sealed.map(|(sealed, _)| sealed);
        // The state is sealed now, so a later copy chains from this recipe.
        self.sealed = Some((recipe, self.fed.len()));
        let snapshot = park::park_snapshot(engine, state)
            .map_err(|message| ServiceError::internal(format!("cannot read the state at {place}: {message}")))?;
        let facts = EngineFacts::of(&self.shared.engine_name, &self.shared.limits);
        let id = snapshot
            .content_id()
            .map_err(|error| ServiceError::internal(format!("cannot hash the state at {place}: {error}")))?;
        let mut store = self.shared.store.lock().expect("the store lock is poisoned");
        // The store can hold the same bytes under another recipe: on an engine with no seal, a build with and a build
        // without a boundary make the same state. The store keeps one recipe for each content id, so this boundary
        // holds no snapshot. The state is the same, so nothing is lost. The store can also hold the bytes as a
        // quarantined entry, which it never replaces (spec 6.7).
        if let Some(entry) = store.entries().get(&id)
            && (entry.state != EntryState::Live || entry.recipe != recipe)
        {
            tracing::debug!(
                "no snapshot at {place}: the store holds its bytes {} under another recipe or in quarantine",
                id.hex()
            );
            return Ok(None);
        }
        store
            .put(Put {
                snapshot: &snapshot,
                recipe,
                previous_recipe: None,
                chains_from,
                engine: facts,
                identity: self.shared.identity,
                owner: COUNCIL_OWNER,
                tier: Tier::Host,
                reason,
                sealed_recipe: None,
                provenance: None,
                now_ms: (self.shared.clock)(),
            })
            .map(Some)
            .or_else(|error| match error {
                // Pins hold the room of the tier. The boundary holds no snapshot, as an interval copy does (6.13).
                StoreError::NoSpace => {
                    tracing::warn!("no snapshot at {place}: the store has no room, and pins hold the rest");
                    Ok(None)
                }
                error => Err(ServiceError::internal(format!(
                    "the store refused the snapshot at {place}: {error}"
                ))),
            })
    }
}

impl<T: Then> JobBody for ContextBody<T> {
    fn request(&self) -> &'static str {
        T::REQUEST
    }

    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let engine = engines[self.shared.engine_index].as_mut();
        let result = if self.started {
            self.advance(kind, engine)
        } else {
            assert_eq!(kind, Kind::Copy, "the first quantum of a context job is a copy");
            self.started = true;
            self.start(engine)
        };
        match result {
            Ok(ran) => ran,
            Err(error) => {
                self.send(Err(error));
                Ran::Failed
            }
        }
    }

    fn item(&self, kind: Kind) -> Item {
        match kind {
            Kind::Copy => Item::Whole(kind),
            _ => Item::Block(kind),
        }
    }

    fn begin_pass(&mut self, kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        let unit = self.units.pop_front().expect("a pass has a unit");
        assert_eq!(
            self.kind_of(&unit),
            kind,
            "the scheduler gave a quantum of another kind than the unit"
        );
        assert!(
            matches!(unit, Unit::Build(Step::Feed { .. }) | Unit::Feed { .. }),
            "a pass of a context job is a feed unit"
        );
        match self.begin(unit) {
            Ok(segment) => Ok(vec![segment]),
            Err(error) => {
                self.send(Err(error));
                Err(Ran::Failed)
            }
        }
    }

    fn end_pass(&mut self, _kind: Kind, result: Result<PassOut, PassError>, engines: &mut [Box<dyn Engine>]) -> Ran {
        let engine = engines[self.shared.engine_index].as_mut();
        let last_error = engine.last_error();
        match self.passed(result, last_error, engine) {
            Ok(ran) => ran,
            Err(error) => {
                self.send(Err(error));
                Ran::Failed
            }
        }
    }

    fn ended(&mut self, state: JobState) {
        // A job that ends with no result tells its caller why.
        let error = match state {
            JobState::Expired => ServiceError::Timeout,
            _ => ServiceError::Cancelled,
        };
        self.send(Err(error));
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        self.send(Err(ServiceError::Unavailable {
            engine: engine.to_owned(),
            state,
        }));
    }
}
