//! The jobs of a build: the lookup and the restore, the feed, and the snapshot at each boundary (spec 11.3c).
//!
//! The bodies run on the group thread, because `lookup` and `restore` take the engine (spec 3.2).

use std::collections::VecDeque;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use nagaya_abi::{Engine, Kind, Limits, OutputsWanted, PassStatus, Segment, SegmentInputs, State, WriteResult};
use nagaya_council::build::{BoundaryRecipe, Step, plan_steps};
use nagaya_store::{ContentId, EngineFacts, Put, Reason, Restored, Store, StoreError, Tier, Tolerance};

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
            return Err(too_large(self.tokens));
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

fn too_large(tokens: usize) -> ServiceError {
    ServiceError::TooLarge {
        param: "context_tokens".to_owned(),
        message: format!(
            "no slot of the engine holds {tokens} tokens; shorten the context, or serve a model with larger slots"
        ),
    }
}

/// The job of a build.
pub struct BuildBody {
    shared: Arc<Shared>,
    tokens: Arc<[u32]>,
    recipes: Vec<BoundaryRecipe>,
    lease: Option<SlotLease>,
    started: bool,
    steps: VecDeque<Step>,
    held: Vec<Option<ContentId>>,
    from: Option<ContentId>,
    kept: usize,
    tx: Option<Sender<Result<Outcome, ServiceError>>>,
}

impl BuildBody {
    /// Make the body. `tokens` are the tokens of the context, without a generation prompt.
    pub fn new(
        shared: Arc<Shared>,
        tokens: Arc<[u32]>,
        recipes: Vec<BoundaryRecipe>,
        tx: Sender<Result<Outcome, ServiceError>>,
    ) -> Self {
        Self {
            shared,
            tokens,
            recipes,
            lease: None,
            started: false,
            steps: VecDeque::new(),
            held: Vec::new(),
            from: None,
            kept: 0,
            tx: Some(tx),
        }
    }

    fn state(&self) -> State {
        State::Slot(
            self.lease
                .as_ref()
                .expect("a build holds a slot after its first quantum")
                .slot(),
        )
    }

    fn finish(&mut self) -> Ran {
        let outcome = Outcome {
            from: self.from,
            kept: self.kept,
            held: std::mem::take(&mut self.held),
        };
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Ok(outcome));
        }
        Ran::Done
    }

    /// The first quantum: take a slot, look up the prefixes, restore the longest, and plan the rest.
    fn start(&mut self, engine: &mut dyn Engine) -> Result<Ran, ServiceError> {
        let lease =
            slots::take(&self.shared.pool, engine, &self.shared.limits, self.tokens.len()).map_err(|reason| {
                match reason {
                    NoSlot::TooLarge => too_large(self.tokens.len()),
                    NoSlot::Busy => ServiceError::Busy(format!(
                        "every slot that holds {} tokens is in use; try again, or send the request as batch",
                        self.tokens.len()
                    )),
                }
            })?;
        let slot = lease.slot();
        self.lease = Some(lease);
        let state = self.state();
        // A slot keeps what its last context left. A build starts from an empty one.
        let slots = engine
            .slots()
            .ok_or_else(|| ServiceError::internal("the engine lost its slot capability"))?;
        if slots.slot_clear(slot) != WriteResult::Ok {
            return Err(ServiceError::internal(format!(
                "the engine could not clear slot {slot}: {}",
                engine.last_error().unwrap_or_default()
            )));
        }
        self.held = lookup_all(&self.shared, engine, slot, &self.recipes)?;
        let last = self.recipes.len() - 1;
        let mut candidate = highest(&self.held);
        while let Some(index) = candidate {
            let id = self.held[index].expect("the highest held boundary has an entry");
            let tokens = self.recipes[index].boundary.tokens;
            // The head is held: nothing needs the state of the engine.
            if index == last {
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
                    let State::Slot(slot_id) = state;
                    let rows = engine
                        .slots()
                        .ok_or_else(|| ServiceError::internal("the engine lost its slot capability"))?
                        .slot_high(slot_id);
                    if rows != tokens as u64 {
                        return Err(ServiceError::internal(format!(
                            "the restored state of slot {slot_id} holds {rows} rows, but its boundary is at token {tokens}"
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
        self.steps = plan_steps(self.kept, &boundaries, self.shared.chunk).into();
        if self.steps.is_empty() {
            return Ok(self.finish());
        }
        Ok(Ran::More(self.steps.iter().map(Step::kind).collect()))
    }

    /// One later quantum: a feed, or a seal and a put.
    fn advance(&mut self, kind: Kind, engine: &mut dyn Engine) -> Result<Ran, ServiceError> {
        let step = self.steps.pop_front().expect("a quantum has a step");
        assert_eq!(
            step.kind(),
            kind,
            "the scheduler gave a quantum of another kind than the step"
        );
        match step {
            Step::Feed { from, to } => self.feed(engine, from, to)?,
            Step::Snapshot { boundary } => self.snapshot(engine, boundary)?,
        }
        Ok(if self.steps.is_empty() {
            self.finish()
        } else {
            Ran::Done
        })
    }

    fn feed(&mut self, engine: &mut dyn Engine, from: usize, to: usize) -> Result<(), ServiceError> {
        let segment = Segment {
            state: self.state(),
            inputs: SegmentInputs::Tokens(self.tokens[from..to].to_vec()),
            positions: (from as u64..to as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted::default(),
        };
        match engine.pass(&[segment], &[]) {
            Ok(out) if out.status == PassStatus::Ok => Ok(()),
            Ok(out) => Err(ServiceError::internal(format!(
                "a pass over tokens {from} to {to} ended with {:?}: {}",
                out.status,
                engine.last_error().unwrap_or_default()
            ))),
            Err(error) => Err(ServiceError::internal(format!(
                "a pass over tokens {from} to {to} failed: {} ({:?})",
                error.message, error.kind
            ))),
        }
    }

    /// Seal the context at a boundary, read its parts, and put the snapshot in the store (spec 11.3c).
    fn snapshot(&mut self, engine: &mut dyn Engine, boundary: usize) -> Result<(), ServiceError> {
        let state = self.state();
        if let Some(seal) = engine.seal() {
            seal.seal(state).map_err(|error| {
                ServiceError::internal(format!("the seal at boundary {boundary} failed: {error:?}"))
            })?;
        }
        let snapshot = park::park_snapshot(engine, state).map_err(|message| {
            ServiceError::internal(format!("cannot read the state at boundary {boundary}: {message}"))
        })?;
        let recipe = &self.recipes[boundary];
        let reason = if boundary + 1 == self.recipes.len() {
            Reason::PrefillDone
        } else {
            Reason::Anchor
        };
        let facts = EngineFacts::of(&self.shared.engine_name, &self.shared.limits);
        let id = self
            .shared
            .store
            .lock()
            .expect("the store lock is poisoned")
            .put(Put {
                snapshot: &snapshot,
                recipe: recipe.recipe,
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
            .map_err(|error| {
                ServiceError::internal(format!(
                    "the store refused the snapshot at boundary {boundary} (message {}): {error}",
                    recipe.boundary.message
                ))
            })?;
        self.held[boundary] = Some(id);
        Ok(())
    }
}

impl JobBody for BuildBody {
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let engine = engines[self.shared.engine_index].as_mut();
        let result = if self.started {
            self.advance(kind, engine)
        } else {
            assert_eq!(kind, Kind::Copy, "the first quantum of a build is a copy");
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
}
