//! A generation of a speech context (spec 11.8a): the service renders the context, starts from the longest held end of
//! a message, feeds the rest, speaks the last line, and holds the head after the line's frames.
//!
//! The job runs on the group of the engine. Its first quantum looks up, restores and feeds. Each later quantum makes
//! the frames of one `speech_frames` call and sends their audio. The last quantum holds the head.

use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya_abi::{Engine, Kind, PassErrorKind, SlotId, SpeechPosition, SpeechSampler, State};
use nagaya_council::build::root_recipe;
use nagaya_sched::JobState;
use nagaya_store::{
    ContentId, EngineFacts, EntryState, Put, Reason, RecipeId, Restored, Store, StoreError, Tier, Tolerance,
};

use super::{Frame, SlotPool, SpeechError};
use crate::council::Clock;
use crate::council::build::COUNCIL_OWNER;
use crate::council::staging::{Staging, state_bytes};
use crate::jobs::{JobBody, Ran};
use crate::park::{identity_digest, park_snapshot};
use crate::process::Ports;

/// The geometry key of the root recipe of a speech context. It keeps the position words of speech apart from tokens.
const GEOMETRY: &[u8] = b"speech positions";

/// One message of a speech context after the voice design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextItem {
    /// A line to speak: the text of a `user` message.
    Line(String),
    /// The audio of the line before it: the codes of the output of an `assistant` message.
    Audio(Vec<Frame>),
}

/// A speech context, checked, with the codes of its outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechContext {
    /// The text of the `system` message: the voice design.
    pub voice: Option<String>,
    /// The messages after it. The last one is a line.
    pub items: Vec<ContextItem>,
    /// `snap` of each message: of the voice design when there is one, then of each item.
    pub snaps: Vec<bool>,
    /// The language of each line.
    pub language: String,
    pub seed: u64,
}

/// What the job of a generation sends to its route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Told {
    /// The audio of one quantum: signed 16-bit samples, mono, at the rate of the engine.
    Audio(Vec<i16>),
    /// The end of the line. Nothing follows.
    End(Spoken),
}

/// The result of a generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spoken {
    /// The codes of each frame of the line.
    pub codes: Vec<Frame>,
    /// The positions of the context.
    pub prompt: usize,
    /// The positions that came from a held snapshot.
    pub kept: usize,
    /// The positions that the job fed.
    pub fed: usize,
    /// The snapshot after the line's frames.
    pub head: Option<ContentId>,
    /// The snapshots that the build held: the index of the message that each one ends, its positions, and its id.
    pub held: Vec<(usize, usize, ContentId)>,
}

/// The words of a position in a recipe: `0, text, codec` for a pair, with `u32::MAX` for an absent id, and `1` and
/// the 16 codes for a frame.
pub fn recipe_words(positions: &[SpeechPosition]) -> Vec<u32> {
    let mut words = Vec::with_capacity(positions.len() * 3);
    for position in positions {
        match position {
            SpeechPosition::Pair(pair) => {
                words.extend([0, pair.text.unwrap_or(u32::MAX), pair.codec.unwrap_or(u32::MAX)]);
            }
            SpeechPosition::Frame(codes) => {
                words.push(1);
                words.extend(codes);
            }
        }
    }
    words
}

/// The recipe at the end of each message (spec 11.8a "Each message ends a recipe"). `ends` must increase.
pub fn end_recipes(root: RecipeId, positions: &[SpeechPosition], ends: &[usize]) -> Vec<RecipeId> {
    let mut recipes = Vec::with_capacity(ends.len());
    let (mut previous, mut at) = (root, 0);
    for &end in ends {
        assert!(
            end >= at && end <= positions.len(),
            "the end {end} of a message is not after {at}"
        );
        previous = nagaya_store::recipe_boundary(&previous, &recipe_words(&positions[at..end]), end as u64);
        recipes.push(previous);
        at = end;
    }
    recipes
}

/// The root recipe of the speech contexts of an engine.
pub fn speech_root(identity: &nagaya_abi::Identity) -> RecipeId {
    root_recipe(&identity_digest(identity), GEOMETRY)
}

/// The job of one generation of a speech context.
pub(super) struct ContextJob {
    pub engine: usize,
    pub engine_name: String,
    pub slot: SlotId,
    pub positions: Vec<SpeechPosition>,
    /// The position count at the end of each message. The last one is the end of the context.
    pub ends: Vec<usize>,
    pub snaps: Vec<bool>,
    pub sampler: SpeechSampler,
    pub out: SyncSender<Result<Told, SpeechError>>,
    pub pool: Arc<SlotPool>,
    pub store: Arc<Mutex<Store>>,
    pub ports: Ports,
    pub clock: Clock,
    pub timeout: Duration,
    pub started: bool,
    pub released: bool,
    pub told: bool,
    pub kept: usize,
    pub codes: Vec<Frame>,
    pub held: Vec<(usize, usize, ContentId)>,
    /// The recipe at the end of each message, and the identity digest, from the first quantum on.
    pub recipes: Vec<RecipeId>,
    pub digest: [u8; 32],
    /// The identity of the engine, which the service shows once the engine opened.
    pub identity: Arc<Mutex<Option<nagaya_abi::Identity>>>,
}

/// The failure of a job: what the route sees.
fn failed(message: String) -> SpeechError {
    tracing::error!("a speech generation failed: {message}");
    SpeechError::Failed(message)
}

impl ContextJob {
    fn release(&mut self) {
        if !self.released {
            self.released = true;
            self.pool.give(self.slot);
        }
    }

    fn tell(&mut self, item: Result<Told, SpeechError>) -> bool {
        self.told |= !matches!(item, Ok(Told::Audio(_)));
        match self.out.try_send(item) {
            Ok(()) => true,
            // A full queue waits for the client; the quantum ended, so the wait holds no device work.
            Err(TrySendError::Full(item)) => self.out.send(item).is_ok(),
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    fn now(&self) -> u64 {
        (self.clock)()
    }

    /// Clear the slot, give it back, and tell the failure.
    fn fail(&mut self, engine: &mut dyn Engine, error: SpeechError) -> Ran {
        if let Some(speech) = engine.speech() {
            speech.speech_clear(self.slot);
        }
        self.release();
        self.tell(Err(error));
        Ran::Failed
    }

    /// The longest held end of a message before the last line, restored into the slot. It gives the positions that
    /// the slot then holds: 0 for none.
    fn restore(&mut self, engine: &mut dyn Engine) -> Result<usize, SpeechError> {
        let total = self.positions.len();
        let state = State::Slot(self.slot);
        for index in (0..self.ends.len())
            .rev()
            .filter(|&index| self.ends[index] < total && self.ends[index] > 0)
        {
            let now = self.now();
            let mut store = self.store.lock().expect("the store lock is poisoned");
            let tip = match store.lookup(self.recipes[index], &Tolerance::Exact, engine, self.slot, now) {
                Ok(chain) => chain.last().copied(),
                // The process treats two incomparable chains as a miss and computes the state again (spec 6.6).
                Err(StoreError::Ambiguous { .. }) => None,
                Err(error) => {
                    return Err(failed(format!(
                        "the lookup at the end of message {index} failed: {error}"
                    )));
                }
            };
            let Some(tip) = tip else { continue };
            match store.restore(&tip, engine, state, &self.digest, now) {
                Ok(Restored::Done) => {}
                // An entry that the restore refused or quarantined is a miss. The state is as it was.
                Ok(Restored::Refused { part }) => {
                    tracing::warn!(
                        "a restore of {} refused the part {part}; the job feeds instead",
                        tip.hex()
                    );
                    continue;
                }
                Err(StoreError::Quarantined(_)) => continue,
                Ok(Restored::Failed { part }) => {
                    return Err(failed(format!(
                        "the restore of {} failed at the part {part}; the state of slot {} is poisoned",
                        tip.hex(),
                        self.slot
                    )));
                }
                Err(error) => return Err(failed(format!("the restore of {} failed: {error}", tip.hex()))),
            }
            let rows = engine.slots().map_or(0, |slots| slots.slot_high(self.slot));
            if rows != self.ends[index] as u64 {
                return Err(failed(format!(
                    "slot {} holds {rows} rows after the restore of {}, but the message ends at position {}",
                    self.slot,
                    tip.hex(),
                    self.ends[index]
                )));
            }
            return Ok(self.ends[index]);
        }
        Ok(0)
    }

    /// Hold the state of the slot under `recipe`. It gives the content id, or `None` when the store already holds
    /// these bytes under another recipe or has no room.
    fn hold(
        &mut self,
        engine: &mut dyn Engine,
        recipe: RecipeId,
        reason: Reason,
        place: &str,
    ) -> Result<Option<ContentId>, SpeechError> {
        let state = State::Slot(self.slot);
        let limits = engine.limits();
        let bytes = state_bytes(engine, &limits, state)
            .map_err(|message| failed(format!("cannot size the copy of {place}: {message}")))?;
        // The buffer of the copy is host memory. The charge stays until the store has taken the snapshot.
        let _staging = Staging::take(&self.ports, &self.engine_name, bytes)
            .map_err(|error| failed(format!("cannot charge the copy of {place}: {error}")))?;
        let snapshot = park_snapshot(engine, state)
            .map_err(|message| failed(format!("cannot read the state at {place}: {message}")))?;
        let id = snapshot
            .content_id()
            .map_err(|error| failed(format!("cannot hash the state at {place}: {error}")))?;
        let now_ms = self.now();
        let mut store = self.store.lock().expect("the store lock is poisoned");
        if let Some(entry) = store.entries().get(&id)
            && (entry.state != EntryState::Live || entry.recipe != recipe)
        {
            tracing::debug!(
                "no snapshot at {place}: the store holds its bytes {} under another recipe",
                id.hex()
            );
            return Ok(None);
        }
        match store.put(Put {
            snapshot: &snapshot,
            recipe,
            previous_recipe: None,
            chains_from: None,
            engine: EngineFacts::of(&self.engine_name, &limits),
            identity: self.digest,
            owner: COUNCIL_OWNER,
            tier: Tier::Host,
            reason,
            sealed_recipe: None,
            provenance: None,
            now_ms,
        }) {
            Ok(id) => Ok(Some(id)),
            // Pins hold the room of the tier, so this end holds no snapshot, as a text boundary (6.13).
            Err(StoreError::NoSpace) => {
                tracing::warn!("no snapshot at {place}: the store has no room, and pins hold the rest");
                Ok(None)
            }
            Err(error) => Err(failed(format!("the store refused the snapshot at {place}: {error}"))),
        }
    }

    /// Feed positions `from..to` into the slot, with the sampler or with none.
    fn feed(&mut self, engine: &mut dyn Engine, from: usize, to: usize, sample: bool) -> Result<(), SpeechError> {
        let speech = engine.speech().expect("the job checked the speech capability");
        let sampler = sample.then_some(&self.sampler);
        speech
            .speech_feed(self.slot, &self.positions[from..to], sampler)
            .map_err(|error| match error.kind {
                PassErrorKind::Refused => SpeechError::Invalid {
                    param: "context",
                    message: error.message,
                },
                PassErrorKind::Ran(_) => failed(error.message),
            })
    }

    /// The first quantum: clear the slot, restore the longest held end, feed to each marked end and hold it, and feed
    /// the last line with the sampler.
    fn start(&mut self, engine: &mut dyn Engine) -> Result<(), SpeechError> {
        let identity = engine.identity();
        self.digest = identity_digest(&identity);
        *self.identity.lock().expect("the identity lock is poisoned") = Some(identity.clone());
        self.recipes = end_recipes(speech_root(&identity), &self.positions, &self.ends);
        // A slot can hold rows of a job that passed its deadline. A clear of an empty slot does nothing.
        engine
            .speech()
            .expect("the job checked the speech capability")
            .speech_clear(self.slot);
        let total = self.positions.len();
        let base = self.restore(engine)?;
        self.kept = base;
        let mut at = base;
        for index in 0..self.ends.len() {
            let end = self.ends[index];
            if !self.snaps[index] || end <= at || end >= total {
                continue;
            }
            self.feed(engine, at, end, false)?;
            at = end;
            if let Some(id) = self.hold(
                engine,
                self.recipes[index],
                Reason::Anchor,
                &format!("the end of message {index}"),
            )? {
                self.held.push((index, end, id));
            }
        }
        self.feed(engine, at, total, true)
    }

    /// The end of the line: hold the head after its frames, clear the slot, and tell the end.
    fn finish(&mut self, engine: &mut dyn Engine) -> Result<Spoken, SpeechError> {
        let total = self.positions.len();
        let frames: Vec<SpeechPosition> = self
            .codes
            .iter()
            .map(|codes| SpeechPosition::Frame(codes.to_vec()))
            .collect();
        let last = *self.recipes.last().expect("a context has an end");
        let head_recipe = nagaya_store::recipe_boundary(&last, &recipe_words(&frames), (total + frames.len()) as u64);
        let rows = engine.slots().map_or(0, |slots| slots.slot_high(self.slot));
        if rows != (total + frames.len()) as u64 {
            return Err(failed(format!(
                "slot {} holds {rows} rows after the line, but the context and its frames are {}",
                self.slot,
                total + frames.len()
            )));
        }
        let head = self.hold(engine, head_recipe, Reason::PrefillDone, "the end of the line")?;
        Ok(Spoken {
            codes: std::mem::take(&mut self.codes),
            prompt: total,
            kept: self.kept,
            fed: total - self.kept,
            head,
            held: std::mem::take(&mut self.held),
        })
    }
}

impl JobBody for ContextJob {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let engine = engines[self.engine].as_mut();
        if engine.speech().is_none() {
            let message = "the engine of the speech service has no speech capability; check its kind in the profile";
            return self.fail(engine, failed(message.to_owned()));
        }
        if !self.started {
            self.started = true;
            return match self.start(engine) {
                Ok(()) => Ran::More(vec![Kind::Decode]),
                Err(error) => self.fail(engine, error),
            };
        }
        let speech = engine.speech().expect("checked above");
        let out = match speech.speech_frames(self.slot, u32::MAX) {
            Ok(out) => out,
            Err(error) => return self.fail(engine, failed(format!("a speech quantum failed: {}", error.message))),
        };
        if out.codes.len() != 16 * out.frames as usize {
            let message = format!(
                "the engine gave {} codes for {} frames, not 16 each",
                out.codes.len(),
                out.frames
            );
            return self.fail(engine, failed(message));
        }
        self.codes.extend(out.codes.as_chunks::<16>().0.iter().copied());
        if !out.audio.is_empty() && !self.tell(Ok(Told::Audio(out.audio))) {
            engine.speech().expect("checked above").speech_clear(self.slot);
            self.release();
            return Ran::Cancelled;
        }
        if !out.end {
            return Ran::More(vec![Kind::Decode]);
        }
        match self.finish(engine) {
            Ok(spoken) => {
                engine.speech().expect("checked above").speech_clear(self.slot);
                self.release();
                self.tell(Ok(Told::End(spoken)));
                Ran::Done
            }
            Err(error) => self.fail(engine, error),
        }
    }

    fn ended(&mut self, state: JobState) {
        self.release();
        if !self.told {
            let message = match state {
                JobState::Expired => SpeechError::Timeout(format!(
                    "the request passed its timeout of {} ms; send a shorter context or try again later",
                    self.timeout.as_millis()
                )),
                other => SpeechError::Failed(format!("the speech job ended as {other:?} before its end of speech")),
            };
            let _ = self.out.try_send(Err(message));
        }
    }

    fn request(&self) -> &'static str {
        "speech_generation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nagaya_abi::PromptPair;

    #[test]
    fn each_code_and_each_absent_id_changes_the_words() {
        let frame = |c: u32| SpeechPosition::Frame((0..16).map(|g| if g == 5 { c } else { g }).collect());
        assert_ne!(
            recipe_words(&[frame(1)]),
            recipe_words(&[frame(2)]),
            "code 5 of a frame"
        );
        let pair = |text, codec| SpeechPosition::Pair(PromptPair { text, codec });
        let words = [
            recipe_words(&[pair(Some(3), None)]),
            recipe_words(&[pair(None, Some(3))]),
            recipe_words(&[pair(Some(3), Some(3))]),
        ];
        assert!(
            words[0] != words[1] && words[1] != words[2] && words[0] != words[2],
            "{words:?}"
        );
    }
}
