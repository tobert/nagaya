//! The beats service of one Beat This! engine (spec 11.19), with no HTTP.
//!
//! A request becomes one job on the CPU group, with one quantum of kind `beats` for each chunk of its log-mel, in
//! order. Each quantum gives the head outputs of one chunk. The route joins the chunks and picks the peaks. A beats
//! job charges no context and stores nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use nagaya_abi::{Engine, Kind};
use nagaya_sched::JobState;
use nagaya_telemetry::EngineState;

use crate::jobs::{JobBody, JobSpec, Ran};
use crate::process::{Ports, SubmitError};
use crate::profile::PlacementProfile;

/// The head outputs of each chunk, in order, or why the job gave none.
pub type Outputs = Receiver<Result<Vec<Vec<f32>>, BeatsError>>;

/// The most seconds of audio in one request (spec 11.19).
pub const MAX_SECONDS: u32 = 600;

/// The timeout of a request that names none.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Why a beats job gave no outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BeatsError {
    /// The process stops, so nothing ran.
    Stopping(String),
    /// The request passed its timeout.
    Timeout(String),
    /// The engine is not ready.
    Unavailable { engine: String, state: EngineState },
    /// A defect or a failed engine call.
    Failed(String),
}

impl std::fmt::Display for BeatsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeatsError::Unavailable { engine, state } => write!(
                formatter,
                "nothing ran: engine `{engine}` is {}, not ready; load it, then retry",
                state.as_str()
            ),
            BeatsError::Stopping(m) | BeatsError::Timeout(m) | BeatsError::Failed(m) => formatter.write_str(m),
        }
    }
}

/// The beats service of one engine. Its served model is the engine's name (spec 11.19).
pub struct BeatsService {
    engine: String,
    ports: Ports,
    group: usize,
    index: usize,
}

impl BeatsService {
    /// Make the service of the engine `engine`. The engine must be in the profile of the process.
    pub fn new(engine: &str, ports: Ports) -> Result<Self, String> {
        let (group, index) = ports
            .locate(engine)
            .ok_or_else(|| format!("the beats engine `{engine}` is not in the profile of the process"))?;
        Ok(Self {
            engine: engine.to_owned(),
            ports,
            group,
            index,
        })
    }

    /// The served model.
    pub fn model(&self) -> &str {
        &self.engine
    }

    /// The state of the engine of the service (spec 3.3a).
    pub fn engine_state(&self) -> EngineState {
        self.ports
            .engine_state(&self.engine)
            .unwrap_or_else(|| panic!("engine `{}` of the beats service has no state", self.engine))
    }

    /// Start a job that runs each chunk of `chunks` (its log-mel rows and its frames), in order. The receiver gets
    /// the head outputs of each chunk, or an error. The job ends at its next quantum after `cancelled` is true.
    pub fn run(
        &self,
        chunks: Vec<(Vec<f32>, u32)>,
        schedule: crate::council::Schedule,
        timeout: Duration,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Outputs, BeatsError> {
        assert!(!chunks.is_empty(), "the route sends at least one chunk");
        let (out, made) = sync_channel(1);
        let body = BeatsJobBody {
            engine: self.index,
            outputs: Vec::with_capacity(chunks.len()),
            chunks,
            out,
            cancelled,
            timeout,
            told: false,
        };
        let spec = JobSpec {
            class: schedule.class,
            priority: i64::from(schedule.priority),
            deadline: Some(Instant::now() + timeout),
            first: Kind::Beats,
            engine: self.index,
            body: Box::new(body),
        };
        if let Err(error) = self.ports.submit_job(self.group, spec) {
            return Err(match error {
                SubmitError::Stopping | SubmitError::Stopped => {
                    BeatsError::Stopping(format!("nothing ran: {error}; send the request to another node"))
                }
                SubmitError::UnknownGroup => BeatsError::Failed(format!("the beats service has no group: {error}")),
            });
        }
        Ok(made)
    }
}

/// The beats services of the profile's engines of the kind `mk-beatthis-cpu`.
pub fn services(profile: &PlacementProfile, ports: &Ports) -> Result<Vec<BeatsService>, String> {
    profile
        .engines
        .iter()
        .filter(|e| e.kind == crate::beatthis::KIND)
        .map(|e| BeatsService::new(&e.name, ports.clone()))
        .collect()
}

/// The body of a beats job: one quantum of kind `beats` for each chunk.
struct BeatsJobBody {
    engine: usize,
    chunks: Vec<(Vec<f32>, u32)>,
    outputs: Vec<Vec<f32>>,
    out: SyncSender<Result<Vec<Vec<f32>>, BeatsError>>,
    cancelled: Arc<AtomicBool>,
    timeout: Duration,
    /// True once the route heard an end: the outputs or an error.
    told: bool,
}

impl BeatsJobBody {
    fn tell(&mut self, item: Result<Vec<Vec<f32>>, BeatsError>) {
        self.told = true;
        // A route that went away drops the receiver, and the send gives an error at once.
        let _ = self.out.send(item);
    }

    fn fail(&mut self, message: String) -> Ran {
        tracing::error!("a beats job failed: {message}");
        self.tell(Err(BeatsError::Failed(message)));
        Ran::Failed
    }
}

impl JobBody for BeatsJobBody {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        if self.cancelled.load(Ordering::SeqCst) {
            self.told = true;
            return Ran::Cancelled;
        }
        let Some(engine) = engines[self.engine].beats() else {
            return self.fail("the engine of the beats service has no beats capability; check its kind".into());
        };
        let index = self.outputs.len();
        let (rows, frames) = &self.chunks[index];
        let out = match engine.beats(rows, *frames) {
            Ok(out) => out,
            Err(error) => return self.fail(format!("the beats of chunk {index} failed: {}", error.message)),
        };
        if out.len() != *frames as usize * 2 || out.iter().any(|v| !v.is_finite()) {
            let message = format!(
                "the engine gave {} values for chunk {index} of {frames} frames, and the service takes {} finite values",
                out.len(),
                *frames as usize * 2
            );
            return self.fail(message);
        }
        self.outputs.push(out);
        if self.outputs.len() < self.chunks.len() {
            return Ran::More(vec![Kind::Beats]);
        }
        let outputs = std::mem::take(&mut self.outputs);
        self.tell(Ok(outputs));
        Ran::Done
    }

    fn ended(&mut self, state: JobState) {
        if !self.told {
            let error = match state {
                JobState::Expired => BeatsError::Timeout(format!(
                    "the request passed its timeout of {} ms; send a longer timeout_ms or a shorter clip",
                    self.timeout.as_millis()
                )),
                other => BeatsError::Failed(format!("the beats job ended as {other:?} before its outputs")),
            };
            self.tell(Err(error));
        }
    }

    fn request(&self) -> &'static str {
        "beats"
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        self.tell(Err(BeatsError::Unavailable {
            engine: engine.to_owned(),
            state,
        }));
    }
}
