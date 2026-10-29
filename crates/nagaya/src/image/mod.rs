//! The image service for one image engine (spec 11.9, 11.9a), with no HTTP.
//!
//! A generation becomes one job on the group of the engine. Its first quantum (`embed`) encodes the prompts. Each
//! later quantum (`step`) runs one step of every image, and the last one decodes them. The job sends its previews and
//! its images to the route. An image job holds no state in the store. A job for an engine that is not `ready` ends
//! at once (spec 3.3a).

use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};
use std::time::{Duration, Instant};

use nagaya_abi::{ImageChunk, ImageJob, Kind, RgbImage, SlotId};
use nagaya_telemetry::EngineState;

use crate::jobs::JobSpec;
use crate::process::{Ports, SubmitError};
use crate::speech::SlotPool;

mod job;

/// The items that wait for a slow client. A full queue holds the job back at its next quantum; it drops nothing.
const QUEUE: usize = 8;

/// A family library of an image engine: it makes the chunks of a prompt (spec 11.3, 11.9).
pub trait ImageFamily: Send + Sync {
    /// The chunks of an A1111 prompt, and the count of its tokens. An error says what is wrong with the text.
    fn prompt(&self, text: &str) -> Result<(Vec<ImageChunk>, u64), String>;
}

/// The config of an image service.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageConfig {
    /// The served image model: the name of the engine in the profile ("The served image model").
    pub model: String,
    /// The name of the engine in the placement profile.
    pub engine: String,
    /// The slots of the engine: the requests that it holds at once.
    pub slots: u32,
    /// The time of a request from its arrival to its last image. The wait for a slot counts toward it.
    pub timeout: Duration,
    /// The prompt tokens that the family takes: 3 chunks of 75 tokens.
    pub context_tokens: u64,
    /// The `clip_skip` of a request that gives none (spec 11.9a).
    pub clip_skip: u32,
}

impl ImageConfig {
    /// The config of SDXL: one slot, 225 prompt tokens, and `clip_skip` 2, the penultimate layer.
    pub fn new(model: &str, engine: &str) -> Self {
        Self {
            model: model.to_owned(),
            engine: engine.to_owned(),
            slots: 1,
            timeout: Duration::from_secs(600),
            context_tokens: 225,
            clip_skip: 2,
        }
    }
}

/// Why a request failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// No slot came free before the timeout.
    Busy(String),
    /// The request passed its timeout.
    Timeout(String),
    /// The process is stopping.
    Stopping(String),
    /// The engine of the image model is not `ready`, so nothing ran (spec 3.3).
    Unavailable { engine: String, state: EngineState },
    /// The engine refused the job, so nothing ran. `param` names the part of the request at fault (spec 11.9a).
    Refused { param: String, message: String },
    /// The engine failed. The process logs it.
    Failed(String),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageError::Unavailable { engine, state } => write!(
                formatter,
                "nothing ran: engine `{engine}` is {}, not ready; load it, then retry",
                state.as_str()
            ),
            ImageError::Busy(message)
            | ImageError::Timeout(message)
            | ImageError::Stopping(message)
            | ImageError::Failed(message) => formatter.write_str(message),
            ImageError::Refused { message, .. } => formatter.write_str(message),
        }
    }
}

/// What the job of a generation sends to its route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Made {
    /// The previews after a step: one for each image, at one eighth of its size.
    Previews { step: u32, previews: Vec<RgbImage> },
    /// A step ended, and the job makes no previews. A route that went away hears of it here, so the job ends at its
    /// next step.
    Step { step: u32 },
    /// The steps ended, and the decode starts.
    Decoding,
    /// The images, in index order. Nothing follows.
    Images(Vec<RgbImage>),
}

/// The image service of one engine.
pub struct ImageService {
    config: ImageConfig,
    family: Arc<dyn ImageFamily>,
    ports: Ports,
    group: usize,
    engine: usize,
    pool: Arc<SlotPool>,
}

impl ImageService {
    /// Make the service of the engine `config.engine`. The engine must be in the profile of the process.
    pub fn new(config: ImageConfig, family: Arc<dyn ImageFamily>, ports: Ports) -> Result<Self, String> {
        if config.slots == 0 {
            return Err(format!(
                "the image engine `{}` has no slot; give it one or more",
                config.engine
            ));
        }
        let (group, engine) = ports.locate(&config.engine).ok_or_else(|| {
            format!(
                "the image engine `{}` is not in the profile of the process",
                config.engine
            )
        })?;
        Ok(Self {
            pool: Arc::new(SlotPool::new(config.slots)),
            config,
            family,
            ports,
            group,
            engine,
        })
    }

    /// The config of the service.
    pub fn config(&self) -> &ImageConfig {
        &self.config
    }

    /// The served image model.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// The family library of the engine.
    pub fn family(&self) -> &dyn ImageFamily {
        self.family.as_ref()
    }

    /// The state of the engine of the service (spec 3.3a).
    pub fn engine_state(&self) -> EngineState {
        self.ports
            .engine_state(&self.config.engine)
            .unwrap_or_else(|| panic!("engine `{}` of the image service has no state", self.config.engine))
    }

    /// The identity of the engine at its last open since the start, and `None` before its first open.
    pub fn identity(&self) -> Option<nagaya_abi::Identity> {
        self.ports.engine_identity(&self.config.engine)
    }

    /// The slots that no request holds.
    pub fn free_slots(&self) -> usize {
        self.pool.count()
    }

    /// Start a generation of a checked job. The receiver gets the previews, `Decoding`, and then the images or an
    /// error. A dropped receiver cancels the job at its next quantum.
    pub fn generate(
        &self,
        image: ImageJob,
        schedule: crate::council::Schedule,
    ) -> Result<Receiver<Result<Made, ImageError>>, ImageError> {
        let deadline = Instant::now() + self.config.timeout;
        let slot: SlotId = self.pool.take(deadline).ok_or_else(|| {
            ImageError::Busy(format!(
                "nothing ran: the {} slots of the image engine stayed full for {} ms; try again later",
                self.config.slots,
                self.config.timeout.as_millis()
            ))
        })?;
        let (out, made) = sync_channel(QUEUE);
        let body = job::ImageJobBody::new(job::Parts {
            engine: self.engine,
            slot,
            image,
            out,
            pool: Arc::clone(&self.pool),
            timeout: self.config.timeout,
        });
        let spec = JobSpec {
            class: schedule.class,
            priority: i64::from(schedule.priority),
            deadline: Some(deadline),
            first: Kind::Embed,
            engine: self.engine,
            body: Box::new(body),
        };
        // A refused job drops its body with no call of `ended`; the body gives its slot back when it drops.
        if let Err(error) = self.ports.submit_job(self.group, spec) {
            return Err(match error {
                SubmitError::Stopping | SubmitError::Stopped => {
                    ImageError::Stopping(format!("nothing ran: {error}; send the request to another node"))
                }
                SubmitError::UnknownGroup => ImageError::Failed(format!("the image service has no group: {error}")),
            });
        }
        Ok(made)
    }
}
