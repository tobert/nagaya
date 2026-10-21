//! The jobs of a context: the lookup and the restore, the feed, and the snapshot at each boundary (spec 11.3c). A
//! read or a generation goes on after the build, on the same slot.
//!
//! The bodies run on the group thread, because `lookup` and `restore` take the engine (spec 3.2).

use std::collections::VecDeque;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use nagaya_abi::{Engine, Kind, Limits, OutputsWanted, PassStatus, Segment, SegmentInputs, State, WriteResult};
use nagaya_council::build::{BoundaryRecipe, Step, plan_chunks, plan_steps};
use nagaya_council::read::logit_row;
use nagaya_sched::JobState;
use nagaya_store::{
    ContentId, EngineFacts, Put, Reason, RecipeId, Restored, Snapshot, Store, StoreError, Tier, Tolerance,
    write_snapshots,
};

use super::Clock;
use super::error::ServiceError;
use super::slots::{self, NoSlot, SlotLease, SlotPool};
use crate::jobs::{JobBody, Ran};
use crate::park;

/// The owner of the entries that builds put. A client pin has its own owner key (spec 11.3c).
pub const COUNCIL_OWNER: u64 = 1;

/// What every body of one build or read shares.
pub struct Shared {
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
    /// The most tokens of one feed.
    pub chunk: usize,
    /// The tokens of the vocabulary of the family. A logit row has this many values.
    pub vocab: usize,
    /// The slots of the engine.
    pub pool: Arc<Mutex<SlotPool>>,
    /// The time in milliseconds.
    pub clock: Clock,
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

/// The highest boundary that the store holds.
fn highest(held: &[Option<ContentId>]) -> Option<usize> {
    held.iter().rposition(Option::is_some)
}

/// The job of a dry run: look up the prefixes and report. It restores and feeds nothing.
pub struct LookupBody {
    shared: Arc<Shared>,
    recipes: Vec<BoundaryRecipe>,
    tokens: usize,
    tx: Option<Sender<Result<Outcome, ServiceError>>>,
}

impl LookupBody {
    /// Make the body. The result goes to `tx`.
    pub fn new(
        shared: Arc<Shared>,
        recipes: Vec<BoundaryRecipe>,
        tokens: usize,
        tx: Sender<Result<Outcome, ServiceError>>,
    ) -> Self {
        Self {
            shared,
            recipes,
            tokens,
            tx: Some(tx),
        }
    }

    fn look(&self, engine: &mut dyn Engine) -> Result<Outcome, ServiceError> {
        let pool = self.shared.pool.lock().expect("the slot pool lock is poisoned");
        let Some(slot) = slots::lowest_fitting(engine, &self.shared.limits, &pool, self.tokens) else {
            return Err(too_large(self.tokens, "context_tokens"));
        };
        drop(pool);
        let held = lookup_all(&self.shared, engine, slot, &self.recipes)?;
        let start = highest(&held);
        Ok(Outcome {
            from: start.and_then(|index| held[index]),
            kept: start.map_or(0, |index| self.recipes[index].boundary.tokens),
            held,
        })
    }
}

impl JobBody for LookupBody {
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        assert_eq!(kind, Kind::Read, "a dry run has one quantum, of the kind read");
        let result = self.look(engines[self.shared.engine_index].as_mut());
        let failed = result.is_err();
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
        if failed { Ran::Failed } else { Ran::Done }
    }
}

fn too_large(tokens: usize, param: &str) -> ServiceError {
    ServiceError::TooLarge {
        param: param.to_owned(),
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
    /// The build is complete. The slot holds the context when the work needs its state.
    Built,
    /// The feed is complete. The row is there when the feed asked for it.
    Fed(Option<Vec<f32>>),
    /// The mark or the rewind is complete.
    Copied,
    /// The store holds the snapshot.
    Held(ContentId),
}

/// The work after a build, as a state machine with no I/O. The body drives it on the group thread.
pub trait Then: Send + 'static {
    /// True when the work needs the state of the context in the slot. Then a build whose head the store holds
    /// restores the head.
    const NEEDS_STATE: bool;

    /// Give the next want, from what the last want gave.
    fn next(&mut self, got: Got) -> Result<Want, ServiceError>;
}

/// No work after the build.
pub struct NoThen;

impl Then for NoThen {
    const NEEDS_STATE: bool = false;

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
}

/// The job of a context: a build, then the work of a [`Then`].
pub struct ContextBody<T: Then> {
    shared: Arc<Shared>,
    tokens: Arc<[u32]>,
    recipes: Vec<BoundaryRecipe>,
    /// The tokens that the slot must hold: the context and what the work feeds after it.
    room: usize,
    /// The limit that a job past `room` breaks, for the error.
    room_param: &'static str,
    lease: Option<SlotLease>,
    started: bool,
    /// True while the build runs, before its work starts.
    building: bool,
    units: VecDeque<Unit>,
    feeding: Option<Feeding>,
    /// The row that the last pass of the feed gave.
    row: Option<Vec<f32>>,
    /// The state that `Mark` kept.
    mark: Option<Snapshot>,
    held: Vec<Option<ContentId>>,
    from: Option<ContentId>,
    kept: usize,
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
        ContextBody::with(shared, tokens, recipes, room, "context_tokens", NoThen, tx)
    }
}

impl<T: Then> ContextBody<T> {
    /// Make the body of a context and its work. The slot must hold `room` tokens, or the job is `too_large` and
    /// names `room_param`. With no recipe, the job builds nothing: the work starts on an empty slot.
    pub fn with(
        shared: Arc<Shared>,
        tokens: Arc<[u32]>,
        recipes: Vec<BoundaryRecipe>,
        room: usize,
        room_param: &'static str,
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
            room,
            room_param,
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
            then: Some(then),
            tx: Some(tx),
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

    fn finish(&mut self) -> Ran {
        let finished = Finished {
            outcome: Outcome {
                from: self.from,
                kept: self.kept,
                held: std::mem::take(&mut self.held),
            },
            then: self.then.take().expect("a job finishes once"),
        };
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Ok(finished));
        }
        Ran::Done
    }

    /// The first quantum: take a slot, look up the prefixes, restore the longest, and plan the rest.
    fn start(&mut self, engine: &mut dyn Engine) -> Result<Ran, ServiceError> {
        let lease =
            slots::take(&self.shared.pool, engine, &self.shared.limits, self.room).map_err(|reason| match reason {
                NoSlot::TooLarge => too_large(self.room, self.room_param),
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
        if self.recipes.is_empty() {
            return self.next_want(engine, Got::Built);
        }
        self.held = lookup_all(&self.shared, engine, slot, &self.recipes)?;
        let last = self.recipes.len() - 1;
        let mut candidate = highest(&self.held);
        while let Some(index) = candidate {
            let id = self.held[index].expect("the highest held boundary has an entry");
            let tokens = self.recipes[index].boundary.tokens;
            // The head is held. A build alone needs no state in the engine.
            if index == last && !T::NEEDS_STATE {
                self.from = Some(id);
                self.kept = tokens;
                break;
            }
            let restored = {
                let mut store = self.shared.store.lock().expect("the store lock is poisoned");
                store.restore(&id, engine, state, &self.shared.identity, (self.shared.clock)())
            };
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
        let boundaries: Vec<_> = self.recipes.iter().map(|recipe| recipe.boundary).collect();
        let steps = plan_steps(self.kept, &boundaries, self.shared.chunk);
        if steps.is_empty() {
            return self.next_want(engine, Got::Built);
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
                if end > self.room {
                    return Err(too_large(end, self.room_param));
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
            Unit::Mark | Unit::Rewind | Unit::Snapshot(..) => Kind::Copy,
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
        let got = match unit {
            Unit::Build(Step::Feed { from, to }) => {
                let tokens = self.tokens[from..to].to_vec();
                self.pass(engine, tokens, from, false)?;
                None
            }
            Unit::Build(Step::Snapshot { boundary }) => {
                let recipe = self.recipes[boundary].recipe;
                let reason = if boundary + 1 == self.recipes.len() {
                    Reason::PrefillDone
                } else {
                    Reason::Anchor
                };
                let message = self.recipes[boundary].boundary.message;
                let id = self.snapshot(engine, recipe, reason, &format!("the boundary of message {message}"))?;
                self.held[boundary] = Some(id);
                None
            }
            Unit::Feed { start, end, row } => {
                let feeding = self.feeding.as_ref().expect("a pass belongs to a feed");
                let tokens = feeding.tokens[start..end].to_vec();
                let from = feeding.base + start;
                let logits = self.pass(engine, tokens, from, row)?;
                if row {
                    self.row = logits;
                }
                if self.units.is_empty() {
                    self.feeding = None;
                    Some(Got::Fed(self.row.take()))
                } else {
                    None
                }
            }
            Unit::Mark => {
                let snapshot = park::park_snapshot(engine, self.state()).map_err(|message| {
                    ServiceError::internal(format!("cannot keep the state of the slot: {message}"))
                })?;
                self.mark = Some(snapshot);
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
        };
        if !self.units.is_empty() {
            return Ok(Ran::Done);
        }
        match got {
            Some(got) => self.next_want(engine, got),
            None if self.building => self.next_want(engine, Got::Built),
            None => unreachable!("the last unit of a want gives back what it did"),
        }
    }

    /// Run one pass over `tokens` at positions from `from`. With `row`, give the logit row of the last token.
    fn pass(
        &mut self,
        engine: &mut dyn Engine,
        tokens: Vec<u32>,
        from: usize,
        row: bool,
    ) -> Result<Option<Vec<f32>>, ServiceError> {
        let to = from + tokens.len();
        let segment = Segment {
            state: self.state(),
            inputs: SegmentInputs::Tokens(tokens),
            positions: (from as u64..to as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if row { vec![to as u64 - 1] } else { Vec::new() },
            },
        };
        let out = match engine.pass(&[segment], &[]) {
            Ok(out) if out.status == PassStatus::Ok => out,
            Ok(out) => {
                return Err(ServiceError::internal(format!(
                    "a pass over tokens {from} to {to} ended with {:?}: {}",
                    out.status,
                    engine.last_error().unwrap_or_default()
                )));
            }
            Err(error) => {
                return Err(ServiceError::internal(format!(
                    "a pass over tokens {from} to {to} failed: {} ({:?})",
                    error.message, error.kind
                )));
            }
        };
        if !row {
            return Ok(None);
        }
        let mut rows = out.outputs.iter().filter(|output| output.name == "logits");
        let (Some(output), None) = (rows.next(), rows.next()) else {
            return Err(ServiceError::internal(format!(
                "a pass over tokens {from} to {to} asked for one logit row and did not give exactly one"
            )));
        };
        logit_row(&output.bytes, self.shared.vocab)
            .map(Some)
            .map_err(|error| ServiceError::internal(format!("the pass over tokens {from} to {to}: {error}")))
    }

    /// Put the state that `Mark` kept back into the slot.
    fn rewind(&mut self, engine: &mut dyn Engine) -> Result<(), ServiceError> {
        let slot = self.slot();
        let mark = self.mark.as_ref().expect("a rewind follows a mark");
        let slots = engine
            .slots()
            .ok_or_else(|| ServiceError::internal("the engine lost its slot capability"))?;
        if slots.slot_clear(slot) != WriteResult::Ok {
            return Err(ServiceError::internal(format!(
                "the engine could not clear slot {slot} for a rewind: {}",
                engine.last_error().unwrap_or_default()
            )));
        }
        match write_snapshots(engine, State::Slot(slot), std::slice::from_ref(mark)) {
            Ok(Restored::Done) => Ok(()),
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
    ) -> Result<ContentId, ServiceError> {
        let state = self.state();
        if let Some(seal) = engine.seal() {
            seal.seal(state)
                .map_err(|error| ServiceError::internal(format!("the seal at {place} failed: {error:?}")))?;
        }
        let snapshot = park::park_snapshot(engine, state)
            .map_err(|message| ServiceError::internal(format!("cannot read the state at {place}: {message}")))?;
        let facts = EngineFacts::of(&self.shared.engine_name, &self.shared.limits);
        self.shared
            .store
            .lock()
            .expect("the store lock is poisoned")
            .put(Put {
                snapshot: &snapshot,
                recipe,
                previous_recipe: None,
                engine: facts,
                identity: self.shared.identity,
                owner: COUNCIL_OWNER,
                tier: Tier::Host,
                reason,
                sealed_recipe: None,
                provenance: None,
                now_ms: (self.shared.clock)(),
            })
            .map_err(|error| ServiceError::internal(format!("the store refused the snapshot at {place}: {error}")))
    }
}

impl<T: Then> JobBody for ContextBody<T> {
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
                if let Some(tx) = self.tx.take() {
                    let _ = tx.send(Err(error));
                }
                Ran::Failed
            }
        }
    }

    fn ended(&mut self, state: JobState) {
        // A job that ends with no result tells its caller why.
        if let Some(tx) = self.tx.take() {
            let error = match state {
                JobState::Expired => ServiceError::Timeout,
                _ => ServiceError::Cancelled,
            };
            let _ = tx.send(Err(error));
        }
    }
}
