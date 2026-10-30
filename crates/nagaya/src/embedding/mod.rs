//! The embedding service of one embedding engine (spec 11.10), with no HTTP.
//!
//! A request becomes one job on the group of the engine, with one quantum of kind `embed` for each input, in input
//! order. Each quantum gives the vector of one input before any normalization. The route cuts and normalizes the
//! vectors. An embedding job charges no context and stores nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use nagaya_abi::{Engine, Kind};
use nagaya_sched::JobState;
use nagaya_telemetry::EngineState;

use crate::jobs::{JobBody, JobSpec, Ran};
use crate::process::{Ports, SubmitError};

/// What the job of a request gives its route: the vector of each input before any normalization, or an error.
pub type Vectors = Receiver<Result<Vec<Vec<f32>>, EmbedError>>;

/// A family library of an embedding engine: the ids of an input and its task (spec 11.3, 11.10).
pub trait EmbedFamily: Send + Sync {
    /// The tasks of a request, in the order of the family.
    fn tasks(&self) -> Vec<String>;
    /// The ids of one input with the prompt of its task and the special tokens. An error says what is wrong.
    fn ids(&self, task: &str, text: &str) -> Result<Vec<u32>, String>;
    /// The template id of the identity: it names the prompts of the tasks.
    fn template(&self) -> String;
}

/// The config of an embedding service.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingConfig {
    /// The served embedding model: the name of the engine in the profile (spec 11.10).
    pub model: String,
    /// The name of the engine in the placement profile.
    pub engine: String,
    /// The time of a request from its arrival to its last vector.
    pub timeout: Duration,
    /// The most tokens of one input, with its task prompt and its special tokens.
    pub context_tokens: u64,
    /// The most inputs of one request.
    pub inputs_per_request: u32,
    /// The vector lengths that a request can ask for, largest first. The first is the length of a vector of the engine.
    pub dimensions: Vec<u32>,
}

impl EmbeddingConfig {
    /// The config of EmbeddingGemma 2 (spec 11.10): 2048 tokens, 64 inputs, and 768, 512, 256 or 128 dimensions.
    pub fn new(model: &str, engine: &str) -> Self {
        Self {
            model: model.to_owned(),
            engine: engine.to_owned(),
            timeout: Duration::from_secs(120),
            context_tokens: 2048,
            inputs_per_request: 64,
            dimensions: vec![768, 512, 256, 128],
        }
    }
}

/// Why a request failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedError {
    /// The process is stopping.
    Stopping(String),
    /// The request passed its timeout.
    Timeout(String),
    /// The engine of the served model is not `ready`, so nothing ran (spec 3.3).
    Unavailable { engine: String, state: EngineState },
    /// The engine failed. The process logs it.
    Failed(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Unavailable { engine, state } => write!(
                formatter,
                "nothing ran: engine `{engine}` is {}, not ready; load it, then retry",
                state.as_str()
            ),
            EmbedError::Stopping(message) | EmbedError::Timeout(message) | EmbedError::Failed(message) => {
                formatter.write_str(message)
            }
        }
    }
}

/// A token that cancels a job between its quanta when it drops, for example with the connection of its client.
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// Make a token that has not cancelled.
    pub fn new() -> Self {
        Cancel(Arc::new(AtomicBool::new(false)))
    }

    /// The flag that the job reads. It is true after the token drops.
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}

impl Default for Cancel {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// The embedding service of one engine.
pub struct EmbeddingService {
    config: EmbeddingConfig,
    family: Arc<dyn EmbedFamily>,
    ports: Ports,
    group: usize,
    engine: usize,
}

impl EmbeddingService {
    /// Make the service of the engine `config.engine`. The engine must be in the profile of the process.
    pub fn new(config: EmbeddingConfig, family: Arc<dyn EmbedFamily>, ports: Ports) -> Result<Self, String> {
        if config.dimensions.is_empty() || config.inputs_per_request == 0 || config.context_tokens == 0 {
            return Err(format!(
                "the embedding engine `{}` needs dimensions, inputs and tokens; give each one",
                config.engine
            ));
        }
        let (group, engine) = ports.locate(&config.engine).ok_or_else(|| {
            format!(
                "the embedding engine `{}` is not in the profile of the process",
                config.engine
            )
        })?;
        Ok(Self {
            config,
            family,
            ports,
            group,
            engine,
        })
    }

    /// The config of the service.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// The served embedding model.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// The family library of the engine.
    pub fn family(&self) -> &dyn EmbedFamily {
        self.family.as_ref()
    }

    /// The state of the engine of the service (spec 3.3a).
    pub fn engine_state(&self) -> EngineState {
        self.ports
            .engine_state(&self.config.engine)
            .unwrap_or_else(|| panic!("engine `{}` of the embedding service has no state", self.config.engine))
    }

    /// The identity of the engine at its last open since the start, and `None` before its first open.
    pub fn identity(&self) -> Option<nagaya_abi::Identity> {
        self.ports.engine_identity(&self.config.engine)
    }

    /// Start a job that embeds each input of `inputs`, in order. The receiver gets the vectors before any
    /// normalization, or an error. The job ends at its next quantum after `cancelled` becomes true ([`Cancel`]).
    pub fn embed(
        &self,
        inputs: Vec<Vec<u32>>,
        schedule: crate::council::Schedule,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Vectors, EmbedError> {
        assert!(!inputs.is_empty(), "the route sends at least one input");
        let deadline = Instant::now() + self.config.timeout;
        let (out, made) = sync_channel(1);
        let body = EmbedJobBody {
            engine: self.engine,
            dimension: self.config.dimensions[0] as usize,
            vectors: Vec::with_capacity(inputs.len()),
            inputs,
            out,
            cancelled,
            timeout: self.config.timeout,
            told: false,
        };
        let spec = JobSpec {
            class: schedule.class,
            priority: i64::from(schedule.priority),
            deadline: Some(deadline),
            first: Kind::Embed,
            engine: self.engine,
            body: Box::new(body),
        };
        if let Err(error) = self.ports.submit_job(self.group, spec) {
            return Err(match error {
                SubmitError::Stopping | SubmitError::Stopped => {
                    EmbedError::Stopping(format!("nothing ran: {error}; send the request to another node"))
                }
                SubmitError::UnknownGroup => EmbedError::Failed(format!("the embedding service has no group: {error}")),
            });
        }
        Ok(made)
    }
}

/// The body of an embedding job: one quantum of kind `embed` for each input (spec 11.10).
struct EmbedJobBody {
    engine: usize,
    /// The length of a vector of the engine.
    dimension: usize,
    inputs: Vec<Vec<u32>>,
    vectors: Vec<Vec<f32>>,
    out: SyncSender<Result<Vec<Vec<f32>>, EmbedError>>,
    cancelled: Arc<AtomicBool>,
    timeout: Duration,
    /// True once the route heard an end: the vectors or an error.
    told: bool,
}

impl EmbedJobBody {
    fn tell(&mut self, item: Result<Vec<Vec<f32>>, EmbedError>) {
        self.told = true;
        // A route that went away drops the receiver, and the send gives an error at once.
        let _ = self.out.send(item);
    }

    fn fail(&mut self, message: String) -> Ran {
        tracing::error!("an embedding job failed: {message}");
        self.tell(Err(EmbedError::Failed(message)));
        Ran::Failed
    }
}

impl JobBody for EmbedJobBody {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        if self.cancelled.load(Ordering::SeqCst) {
            self.told = true;
            return Ran::Cancelled;
        }
        let Some(engine) = engines[self.engine].embedding() else {
            return self.fail("the engine of the embedding service has no embedding capability; check its kind".into());
        };
        let index = self.vectors.len();
        let vector = match engine.embed(&self.inputs[index]) {
            Ok(vector) => vector,
            Err(error) => return self.fail(format!("the embed of input {index} failed: {}", error.message)),
        };
        if vector.len() != self.dimension || vector.iter().any(|value| !value.is_finite()) {
            let message = format!(
                "the engine gave a vector of {} components for input {index}, and the service takes {} finite values",
                vector.len(),
                self.dimension
            );
            return self.fail(message);
        }
        self.vectors.push(vector);
        if self.vectors.len() < self.inputs.len() {
            return Ran::More(vec![Kind::Embed]);
        }
        let vectors = std::mem::take(&mut self.vectors);
        self.tell(Ok(vectors));
        Ran::Done
    }

    fn ended(&mut self, state: JobState) {
        if !self.told {
            let error = match state {
                JobState::Expired => EmbedError::Timeout(format!(
                    "the request passed its timeout of {} ms; send fewer or shorter inputs",
                    self.timeout.as_millis()
                )),
                other => EmbedError::Failed(format!("the embedding job ended as {other:?} before its vectors")),
            };
            self.tell(Err(error));
        }
    }

    fn request(&self) -> &'static str {
        "embeddings"
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        self.tell(Err(EmbedError::Unavailable {
            engine: engine.to_owned(),
            state,
        }));
    }
}
