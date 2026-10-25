//! The speech service for one speech engine (spec 11.8), with no HTTP.
//!
//! A request becomes one job on the group of the engine. Its first quantum feeds the prompt (`speech_feed`), and each
//! later quantum makes the frames of one `speech_frames` call. The job sends the audio of each quantum to the route as
//! it decodes. A request waits for a free slot of the engine, and a job for an engine that is not loaded waits for
//! the swap that loads it (spec 3.3a).

use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nagaya_abi::{Engine, Kind, PassErrorKind, PromptPair, SlotId, SpeechPosition, SpeechSampler};
use nagaya_sched::JobState;

use crate::jobs::{JobBody, JobSpec, Ran};
use crate::process::{Ports, SubmitError};

pub mod codes;
pub mod context;

pub use context::{ContextItem, SpeechContext, Spoken, Told};

/// The codes of one frame: one code of each of the 16 codebooks.
pub type Frame = [u32; 16];

/// The positions of a speech context, and the position count at the end of each message (spec 11.8a).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextRendering {
    pub positions: Vec<SpeechPosition>,
    /// The position count after the voice design, when the context has one, then after each item.
    pub ends: Vec<usize>,
}

/// The chunks that wait for a slow client. A full queue holds the job back at its next quantum; it drops nothing.
const QUEUE: usize = 64;

/// A family library of a speech engine: it makes the prompt of a request (spec 11.3, 11.8).
pub trait SpeechFamily: Send + Sync {
    /// The prompt positions of `text` in the voice of `instructions`, in `language`.
    fn prompt(&self, text: &str, instructions: &str, language: &str) -> Result<Vec<PromptPair>, SpeechError>;

    /// The rendering of a speech context in one language (spec 11.8a). `voice` is the text of the `system` message.
    fn context(
        &self,
        voice: Option<&str>,
        items: &[ContextItem],
        language: &str,
    ) -> Result<ContextRendering, SpeechError>;
}

/// The config of a speech service.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechConfig {
    /// The served speech model: the `model` of the engine identity.
    pub model: String,
    /// The name of the engine in the placement profile.
    pub engine: String,
    /// The slots of the engine: the requests that it holds at once.
    pub slots: u32,
    /// The time of a request from its arrival to its last chunk. The wait for a slot and for a swap count toward it.
    pub timeout: Duration,
    /// The sampler of every request, without its seed (spec 11.8).
    pub temperature: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    /// The positions that a slot holds: a speech context and the frames of its last line.
    pub context_positions: u64,
}

impl SpeechConfig {
    /// The config with the generation config of Qwen3-TTS: temperature 0.9, top_k 50, repetition penalty 1.05.
    pub fn new(model: &str, engine: &str, slots: u32) -> Self {
        Self {
            model: model.to_owned(),
            engine: engine.to_owned(),
            slots,
            timeout: Duration::from_secs(120),
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
            context_positions: 0,
        }
    }
}

/// One request.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechRequest {
    pub input: String,
    pub instructions: String,
    pub language: String,
    pub seed: u64,
}

/// What a speech job sends to its route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Chunk {
    /// The audio of one quantum: signed 16-bit samples, mono, at 24 kHz.
    Audio(Vec<i16>),
    /// The end of speech. Nothing follows.
    End,
}

/// Why a request failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeechError {
    /// The request breaks a rule. `param` names the key.
    Invalid { param: &'static str, message: String },
    /// No slot came free before the timeout, or the ledger refused a copy.
    Busy(String),
    /// The context and a frame do not fit in a slot.
    ContextFull { positions: u64, room: u64 },
    /// The request passed its timeout.
    Timeout(String),
    /// The process is stopping.
    Stopping(String),
    /// The engine failed. The process logs it.
    Failed(String),
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechError::Invalid { param, message } => write!(formatter, "`{param}`: {message}"),
            SpeechError::ContextFull { positions, room } => write!(
                formatter,
                "nothing ran: the context has {positions} positions, and a slot holds {room} with the frames of its \
                 line; start a shorter context"
            ),
            SpeechError::Busy(message)
            | SpeechError::Timeout(message)
            | SpeechError::Stopping(message)
            | SpeechError::Failed(message) => formatter.write_str(message),
        }
    }
}

/// The free slots of the engine. A request takes one before its job starts, and the job gives it back at its end.
struct SlotPool {
    free: Mutex<Vec<SlotId>>,
    freed: Condvar,
}

impl SlotPool {
    fn new(slots: u32) -> Self {
        Self {
            free: Mutex::new((0..slots).rev().collect()),
            freed: Condvar::new(),
        }
    }

    /// A free slot, or `None` when none came free before `deadline`.
    fn take(&self, deadline: Instant) -> Option<SlotId> {
        let mut free = self.free.lock().expect("the slot lock is poisoned");
        loop {
            if let Some(slot) = free.pop() {
                return Some(slot);
            }
            let left = deadline.checked_duration_since(Instant::now())?;
            free = self
                .freed
                .wait_timeout(free, left)
                .expect("the slot lock is poisoned")
                .0;
        }
    }

    fn give(&self, slot: SlotId) {
        let mut free = self.free.lock().expect("the slot lock is poisoned");
        assert!(!free.contains(&slot), "slot {slot} came back twice");
        free.push(slot);
        self.freed.notify_one();
    }

    fn count(&self) -> usize {
        self.free.lock().expect("the slot lock is poisoned").len()
    }
}

/// The work of one speech request on the group thread.
struct SpeechBody {
    engine: usize,
    slot: SlotId,
    prompt: Vec<PromptPair>,
    sampler: SpeechSampler,
    out: SyncSender<Result<Chunk, SpeechError>>,
    pool: Arc<SlotPool>,
    begun: bool,
    /// True when the slot went back to the pool.
    released: bool,
    /// True when the job sent its end or its failure.
    told: bool,
    timeout: Duration,
}

impl SpeechBody {
    /// Give the slot back to the pool, once. The job does it before it tells its end, so a client that reads the end
    /// finds the slot free.
    fn release(&mut self) {
        if !self.released {
            self.released = true;
            self.pool.give(self.slot);
        }
    }

    fn tell(&mut self, item: Result<Chunk, SpeechError>) -> bool {
        self.told |= !matches!(item, Ok(Chunk::Audio(_)));
        match self.out.try_send(item) {
            Ok(()) => true,
            // A full queue waits for the client; the quantum ended, so the wait holds no device work.
            Err(TrySendError::Full(item)) => self.out.send(item).is_ok(),
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

impl JobBody for SpeechBody {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let Some(speech) = engines[self.engine].speech() else {
            let message = "the engine of the speech service has no speech capability; check its kind in the profile";
            tracing::error!("{message}");
            self.tell(Err(SpeechError::Failed(message.to_owned())));
            return Ran::Failed;
        };
        if !self.begun {
            // A job that passed its deadline after its begin left its slot full. A clear of an empty slot does nothing.
            speech.speech_clear(self.slot);
            let positions: Vec<SpeechPosition> = self.prompt.iter().map(|p| SpeechPosition::Pair(*p)).collect();
            if let Err(error) = speech.speech_feed(self.slot, &positions, Some(&self.sampler)) {
                let failure = match error.kind {
                    PassErrorKind::Refused => SpeechError::Invalid {
                        param: "input",
                        message: error.message,
                    },
                    PassErrorKind::Ran(_) => SpeechError::Failed(error.message),
                };
                self.release();
                self.tell(Err(failure));
                return Ran::Failed;
            }
            self.begun = true;
            return Ran::More(vec![Kind::Decode]);
        }
        let out = match speech.speech_frames(self.slot, u32::MAX) {
            Ok(out) => out,
            Err(error) => {
                speech.speech_clear(self.slot);
                tracing::error!("a speech quantum failed: {}", error.message);
                self.release();
                self.tell(Err(SpeechError::Failed(error.message)));
                return Ran::Failed;
            }
        };
        let end = out.end;
        if !out.audio.is_empty() && !self.tell(Ok(Chunk::Audio(out.audio))) {
            speech.speech_clear(self.slot);
            self.release();
            return Ran::Cancelled;
        }
        if end {
            speech.speech_clear(self.slot);
            self.release();
            self.tell(Ok(Chunk::End));
            return Ran::Done;
        }
        Ran::More(vec![Kind::Decode])
    }

    fn ended(&mut self, state: JobState) {
        self.release();
        if !self.told {
            let message = match state {
                JobState::Expired => SpeechError::Timeout(format!(
                    "the request passed its timeout of {} ms; send a shorter input or try again later",
                    self.timeout.as_millis()
                )),
                other => SpeechError::Failed(format!("the speech job ended as {other:?} before its end of speech")),
            };
            let _ = self.out.try_send(Err(message));
        }
    }

    fn request(&self) -> &'static str {
        "speech"
    }
}

/// The speech service of one engine.
pub struct SpeechService {
    config: SpeechConfig,
    family: Arc<dyn SpeechFamily>,
    ports: Ports,
    group: usize,
    engine: usize,
    pool: Arc<SlotPool>,
    clock: crate::council::Clock,
    /// The codes of the outputs that the service wrote, when it serves speech contexts (spec 11.8a).
    codes: Option<codes::CodeBook>,
    /// The identity of the engine after its first open since the start.
    identity: Arc<Mutex<Option<nagaya_abi::Identity>>>,
    /// The code puts since the start.
    puts: std::sync::atomic::AtomicU64,
}

impl SpeechService {
    /// Make the service of the engine `config.engine`. The engine must be in the profile of the process.
    pub fn new(config: SpeechConfig, family: Arc<dyn SpeechFamily>, ports: Ports) -> Result<Self, String> {
        let (group, engine) = ports.locate(&config.engine).ok_or_else(|| {
            format!(
                "the speech engine `{}` is not in the profile of the process",
                config.engine
            )
        })?;
        if config.slots == 0 {
            return Err(format!("the speech engine `{}` has no slot", config.engine));
        }
        Ok(Self {
            pool: Arc::new(SlotPool::new(config.slots)),
            config,
            family,
            ports,
            group,
            engine,
            clock: crate::council::system_clock(),
            codes: None,
            identity: Arc::default(),
            puts: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Serve speech contexts, with the codes of the outputs in this book (spec 11.8a).
    pub fn with_codes(mut self, codes: codes::CodeBook) -> Self {
        self.codes = Some(codes);
        self
    }

    /// The config of the service.
    pub fn config(&self) -> &SpeechConfig {
        &self.config
    }

    /// The identity of the engine, once it opened since the start.
    pub fn identity(&self) -> Option<nagaya_abi::Identity> {
        self.identity.lock().expect("the identity lock is poisoned").clone()
    }

    /// True when the service serves speech contexts: it has a code book.
    pub fn serves_contexts(&self) -> bool {
        self.codes.is_some()
    }

    /// The codes of the output `id`, when this service wrote it and still holds them.
    pub fn codes(&self, id: &nagaya_store::ContentId) -> Result<Option<Vec<Frame>>, String> {
        self.codes.as_ref().map_or(Ok(None), |book| book.get(id))
    }

    /// Keep the codes of the output `id`. At the first put and at each 256th one, drop the code files of the outputs
    /// that `held` says the object store no longer holds, so the book does not grow past the objects.
    pub fn keep_codes(
        &self,
        id: &nagaya_store::ContentId,
        frames: &[Frame],
        held: &dyn Fn(&nagaya_store::ContentId) -> bool,
    ) -> Result<(), String> {
        let book = self
            .codes
            .as_ref()
            .ok_or_else(|| "the speech service has no code book; start it with a store directory".to_owned())?;
        book.put(id, frames)?;
        if self
            .puts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .is_multiple_of(256)
        {
            let dropped = book.prune(held)?;
            tracing::debug!("the code book dropped the codes of {dropped} outputs that the object store left");
        }
        Ok(())
    }

    /// The service with a clock that the caller chooses, for the times of the store.
    pub fn with_clock(mut self, clock: crate::council::Clock) -> Self {
        self.clock = clock;
        self
    }

    /// Start a generation of a speech context (spec 11.8a): render it, wait for a slot, and send its job. The
    /// receiver gives the audio of each quantum in order, then `Told::End` or an error. It blocks while it waits for
    /// a slot.
    pub fn generate(&self, context: &SpeechContext) -> Result<Receiver<Result<Told, SpeechError>>, SpeechError> {
        let deadline = Instant::now() + self.config.timeout;
        if !matches!(context.items.last(), Some(ContextItem::Line(_))) {
            return Err(SpeechError::Invalid {
                param: "messages",
                message: "the last message of a speech context is not a `user` line; end the context with the line \
                          to speak"
                    .to_owned(),
            });
        }
        let messages = context.items.len() + usize::from(context.voice.is_some());
        assert_eq!(
            context.snaps.len(),
            messages,
            "a speech context has one snap for each message"
        );
        let rendering = self
            .family
            .context(context.voice.as_deref(), &context.items, &context.language)?;
        if rendering.ends.len() != messages || rendering.ends.last() != Some(&rendering.positions.len()) {
            return Err(SpeechError::Failed(format!(
                "the family rendered {} message ends for {messages} messages, the last at {:?} of {} positions; \
                 report this defect of the family library",
                rendering.ends.len(),
                rendering.ends.last(),
                rendering.positions.len()
            )));
        }
        let room = self.config.context_positions;
        if room > 0 && rendering.positions.len() as u64 + 1 >= room {
            return Err(SpeechError::ContextFull {
                positions: rendering.positions.len() as u64,
                room,
            });
        }
        let store = self.ports.store().cloned().ok_or_else(|| {
            SpeechError::Failed("the process has no store; start it with a store directory".to_owned())
        })?;
        let slot = self.pool.take(deadline).ok_or_else(|| {
            SpeechError::Busy(format!(
                "nothing ran: the {} slots of the speech engine stayed full for {} ms; try again later",
                self.config.slots,
                self.config.timeout.as_millis()
            ))
        })?;
        let (out, told) = sync_channel(QUEUE);
        let body = context::ContextJob {
            engine: self.engine,
            engine_name: self.config.engine.clone(),
            slot,
            positions: rendering.positions,
            ends: rendering.ends,
            snaps: context.snaps.clone(),
            sampler: SpeechSampler {
                temperature: self.config.temperature,
                top_k: self.config.top_k,
                repetition_penalty: self.config.repetition_penalty,
                seed: context.seed,
            },
            out,
            pool: Arc::clone(&self.pool),
            store,
            ports: self.ports.clone(),
            clock: Arc::clone(&self.clock),
            timeout: self.config.timeout,
            started: false,
            released: false,
            told: false,
            kept: 0,
            codes: Vec::new(),
            held: Vec::new(),
            recipes: Vec::new(),
            digest: [0; 32],
            identity: Arc::clone(&self.identity),
        };
        self.submit(slot, deadline, context.schedule, Box::new(body))?;
        Ok(told)
    }

    /// Send a job that holds `slot`. A refused job drops its body with no call of `ended`, so the slot comes back
    /// here.
    fn submit(
        &self,
        slot: SlotId,
        deadline: Instant,
        schedule: crate::council::Schedule,
        body: Box<dyn JobBody>,
    ) -> Result<(), SpeechError> {
        let job = JobSpec {
            class: schedule.class,
            priority: i64::from(schedule.priority),
            deadline: Some(deadline),
            first: Kind::Prefill,
            engine: self.engine,
            body,
        };
        if let Err(error) = self.ports.submit_job(self.group, job) {
            self.pool.give(slot);
            return Err(match error {
                SubmitError::Stopping | SubmitError::Stopped => {
                    SpeechError::Stopping(format!("nothing ran: {error}; send the request to another node"))
                }
                SubmitError::UnknownGroup => SpeechError::Failed(format!("the speech service has no group: {error}")),
            });
        }
        Ok(())
    }

    /// The served speech model.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// The slots that no request holds.
    pub fn free_slots(&self) -> usize {
        self.pool.count()
    }

    /// Start a request: make its prompt, wait for a slot, and send its job. The receiver gives the chunks in order,
    /// then `Chunk::End` or an error. It blocks while it waits for a slot.
    pub fn start(&self, request: &SpeechRequest) -> Result<Receiver<Result<Chunk, SpeechError>>, SpeechError> {
        let deadline = Instant::now() + self.config.timeout;
        let prompt = self
            .family
            .prompt(&request.input, &request.instructions, &request.language)?;
        let slot = self.pool.take(deadline).ok_or_else(|| {
            SpeechError::Busy(format!(
                "nothing ran: the {} slots of the speech engine stayed full for {} ms; try again later",
                self.config.slots,
                self.config.timeout.as_millis()
            ))
        })?;
        let (out, chunks) = sync_channel(QUEUE);
        let body = SpeechBody {
            engine: self.engine,
            slot,
            prompt,
            sampler: SpeechSampler {
                temperature: self.config.temperature,
                top_k: self.config.top_k,
                repetition_penalty: self.config.repetition_penalty,
                seed: request.seed,
            },
            out,
            pool: Arc::clone(&self.pool),
            begun: false,
            released: false,
            told: false,
            timeout: self.config.timeout,
        };
        self.submit(slot, deadline, crate::council::Schedule::default(), Box::new(body))?;
        Ok(chunks)
    }
}
