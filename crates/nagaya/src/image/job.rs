//! The job of one image generation on the group of its engine (spec 11.9a "The quanta").
//!
//! The first quantum (`embed`) begins the job in its slot. A job with an image then runs the VAE encode as one `step`
//! quantum (spec 11.9b). Each later `step` quantum runs one step of every image, and the `step` after the last one
//! decodes them. The body sends what each quantum made to the route.

use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya_abi::{Engine, ImageConditioning, ImageJob, ImagePrompt, Kind, PassErrorKind, SlotId, steps_that_run};
use nagaya_sched::JobState;
use nagaya_store::{EngineFacts, Put, Reason, Store, Tier, conditioning};
use nagaya_telemetry::EngineState;

use super::{ImageError, Made};
use crate::jobs::{JobBody, Ran};
use crate::speech::SlotPool;

/// What a job body needs from its service.
pub(crate) struct Parts {
    pub engine: usize,
    pub slot: SlotId,
    pub image: ImageJob,
    pub out: SyncSender<Result<Made, ImageError>>,
    pub pool: Arc<SlotPool>,
    pub timeout: Duration,
    /// The store, which keeps the conditionings of an engine that declares them (spec 6.16).
    pub store: Option<Arc<Mutex<Store>>>,
    /// The name of the engine in the profile, for the facts of a put.
    pub engine_name: String,
}

/// Where the job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Begin,
    /// The VAE encode of a job with an image: the first `image_step` gives step 0 and no previews.
    Encode,
    Steps,
    Decode,
}

/// The current time in milliseconds since the epoch, as the store records it.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is after 1970")
        .as_millis() as u64
}

/// The rows of the conditioning of `job` in the store, or none on a miss (spec 6.16). A store that fails gives none,
/// with a log line: the encoder then runs, and the image is the same.
fn cached_rows(store: &Mutex<Store>, declared: &ImageConditioning, job: &ImageJob) -> Option<Vec<u8>> {
    let ImagePrompt::Tokens { ids, cut } = &job.prompt else {
        return None;
    };
    let digest = match conditioning::digest_bytes(&declared.encoder_digest) {
        Ok(digest) => digest,
        Err(error) => {
            tracing::error!("the engine declares a bad encoder digest, so no conditioning is cached: {error}");
            return None;
        }
    };
    let recipe = conditioning::recipe(&digest, ids);
    let rows = ids.len().saturating_sub(*cut as usize);
    let found = store.lock().expect("the store lock is poisoned").find(recipe, now_ms());
    match found {
        Ok(Some((id, snapshot))) => match conditioning::rows(&snapshot, declared.row_bytes as usize, rows) {
            Ok(bytes) => {
                tracing::debug!("a conditioning hit: entry {}", id.hex());
                Some(bytes)
            }
            Err(error) => {
                tracing::error!(
                    "the conditioning entry {} has another form, and the encoder runs: {error}",
                    id.hex()
                );
                None
            }
        },
        Ok(None) => None,
        Err(error) => {
            tracing::error!("the lookup of a conditioning failed, and the encoder runs: {error}");
            None
        }
    }
}

/// Put the conditioning that the encoder made for `job` in the host tier of the store (spec 6.16). A put that fails
/// costs only a later encode, so it is a log line.
fn put_rows(
    store: &Mutex<Store>,
    declared: &ImageConditioning,
    job: &ImageJob,
    rows: &[u8],
    engine: &str,
    limits: &nagaya_abi::Limits,
) {
    let ImagePrompt::Tokens { ids, cut } = &job.prompt else {
        return;
    };
    let result = conditioning::digest_bytes(&declared.encoder_digest).and_then(|digest| {
        let snapshot = conditioning::snapshot(rows, declared.row_bytes as usize, u64::from(*cut), ids.len() as u64)?;
        store.lock().expect("the store lock is poisoned").put(Put {
            snapshot: &snapshot,
            recipe: conditioning::recipe(&digest, ids),
            previous_recipe: None,
            chains_from: None,
            engine: EngineFacts::of(engine, limits),
            identity: digest,
            owner: 0,
            tier: Tier::Host,
            reason: Reason::PrefillDone,
            sealed_recipe: None,
            provenance: None,
            now_ms: now_ms(),
        })
    });
    if let Err(error) = result {
        tracing::warn!("the store did not keep a conditioning of engine `{engine}`: {error}");
    }
}

/// The body of an image job.
pub(crate) struct ImageJobBody {
    parts: Parts,
    phase: Phase,
    released: bool,
    /// True once the route heard an end: the images or an error.
    told: bool,
}

fn failed(message: String) -> ImageError {
    tracing::error!("an image generation failed: {message}");
    ImageError::Failed(message)
}

impl ImageJobBody {
    pub fn new(parts: Parts) -> Self {
        Self {
            parts,
            phase: Phase::Begin,
            released: false,
            told: false,
        }
    }

    /// The steps that the job runs: `n` with an image (spec 11.9b), else the steps of the request.
    fn steps_run(&self) -> u32 {
        let job = &self.parts.image;
        job.init
            .as_ref()
            .map_or(job.steps, |init| steps_that_run(job.steps, init.strength))
    }

    fn release(&mut self) {
        if !self.released {
            self.released = true;
            self.parts.pool.give(self.parts.slot);
        }
    }

    /// Send an item to the route. False when the route went away.
    fn tell(&mut self, item: Result<Made, ImageError>) -> bool {
        self.told |= matches!(item, Ok(Made::Images(_)) | Err(_));
        match self.parts.out.try_send(item) {
            Ok(()) => true,
            // A full queue waits for the worker of the route, which never waits on a client (spec 11.9a). The
            // quantum ended, so the wait holds no device work.
            Err(TrySendError::Full(item)) => self.parts.out.send(item).is_ok(),
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Clear the slot, give it back, and tell the failure.
    fn fail(&mut self, engine: &mut dyn Engine, error: ImageError) -> Ran {
        if let Some(image) = engine.image() {
            image.image_clear(self.parts.slot);
        }
        self.release();
        self.tell(Err(error));
        Ran::Failed
    }

    /// Clear the slot and give it back, because the route went away.
    fn cancel(&mut self, engine: &mut dyn Engine) -> Ran {
        if let Some(image) = engine.image() {
            image.image_clear(self.parts.slot);
        }
        self.release();
        Ran::Cancelled
    }
}

impl Drop for ImageJobBody {
    /// A stop drops the bodies of its jobs with no call of `ended`, so the slot comes back here.
    fn drop(&mut self) {
        self.release();
    }
}

impl JobBody for ImageJobBody {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let engine = engines[self.parts.engine].as_mut();
        if engine.image().is_none() {
            let message = "the engine of the image service has no image capability; check its kind in the profile";
            return self.fail(engine, failed(message.to_owned()));
        }
        let slot = self.parts.slot;
        match self.phase {
            Phase::Begin => {
                let limits = engine.limits();
                let image = engine.image().expect("checked above");
                let declared = image.image_conditioning();
                if let (Some(store), Some(declared)) = (&self.parts.store, &declared) {
                    self.parts.image.conditioning = cached_rows(store, declared, &self.parts.image).map(Arc::new);
                }
                let begun = image.image_begin(slot, &self.parts.image);
                if let (Ok(Some(rows)), Some(store), Some(declared)) = (&begun, &self.parts.store, &declared) {
                    put_rows(
                        store,
                        declared,
                        &self.parts.image,
                        rows,
                        &self.parts.engine_name,
                        &limits,
                    );
                }
                if let Err(error) = begun {
                    // A refusal is the client's fault: the job names a part the engine cannot take (spec 11.9a).
                    let error = match error.kind {
                        PassErrorKind::Refused => {
                            let param = if self.parts.image.loras.is_empty() {
                                "output"
                            } else {
                                "output.options.loras"
                            };
                            ImageError::Refused {
                                param: param.to_owned(),
                                message: error.message,
                            }
                        }
                        _ => failed(format!("an image job did not begin: {}", error.message)),
                    };
                    return self.fail(engine, error);
                }
                self.phase = if self.parts.image.init.is_some() {
                    Phase::Encode
                } else {
                    Phase::Steps
                };
                Ran::More(vec![Kind::Step])
            }
            Phase::Encode => {
                let out = match engine.image().expect("checked above").image_step(slot) {
                    Ok(out) => out,
                    Err(error) => {
                        return self.fail(engine, failed(format!("an image encode failed: {}", error.message)));
                    }
                };
                let steps = self.steps_run();
                if out.step != 0 || out.steps != steps || !out.previews.is_empty() {
                    let message = format!(
                        "the engine gave step {} of {} with {} previews for the encode of a job of {steps} steps; the \
                         encode gives step 0 with no previews",
                        out.step,
                        out.steps,
                        out.previews.len()
                    );
                    return self.fail(engine, failed(message));
                }
                // The encode is not a step of `progress` (spec 11.9b), so the route hears nothing of it.
                self.phase = Phase::Steps;
                Ran::More(vec![Kind::Step])
            }
            Phase::Steps => {
                let out = match engine.image().expect("checked above").image_step(slot) {
                    Ok(out) => out,
                    Err(error) => return self.fail(engine, failed(format!("an image step failed: {}", error.message))),
                };
                let steps = self.steps_run();
                if out.steps != steps || out.step == 0 || out.step > steps {
                    let message = format!(
                        "the engine gave step {} of {}, for a job of {steps} steps",
                        out.step, out.steps
                    );
                    return self.fail(engine, failed(message));
                }
                if self.parts.image.previews {
                    if out.previews.len() != self.parts.image.count as usize {
                        let message = format!(
                            "the engine gave {} previews for {} images",
                            out.previews.len(),
                            self.parts.image.count
                        );
                        return self.fail(engine, failed(message));
                    }
                    let made = Made::Previews {
                        step: out.step,
                        previews: out.previews,
                    };
                    if !self.tell(Ok(made)) {
                        return self.cancel(engine);
                    }
                } else if !self.tell(Ok(Made::Step { step: out.step })) {
                    return self.cancel(engine);
                }
                if out.step < steps {
                    return Ran::More(vec![Kind::Step]);
                }
                if !self.tell(Ok(Made::Decoding)) {
                    return self.cancel(engine);
                }
                self.phase = Phase::Decode;
                Ran::More(vec![Kind::Step])
            }
            Phase::Decode => {
                let images = match engine.image().expect("checked above").image_decode(slot) {
                    Ok(images) => images,
                    Err(error) => {
                        return self.fail(engine, failed(format!("an image decode failed: {}", error.message)));
                    }
                };
                let job = &self.parts.image;
                let (count, width, height) = (job.count as usize, job.width, job.height);
                let wrong = images.iter().find(|i| {
                    i.width != width
                        || i.height != height
                        || !(3..=4).contains(&i.channels)
                        || i.pixels.len() != (width * height * i.channels) as usize
                });
                if images.len() != count || wrong.is_some() {
                    let message = format!(
                        "the engine gave {} images, and the job asked for {count} of {width} by {height}",
                        images.len()
                    );
                    return self.fail(engine, failed(message));
                }
                engine.image().expect("checked above").image_clear(slot);
                self.release();
                self.tell(Ok(Made::Images(images)));
                Ran::Done
            }
        }
    }

    fn ended(&mut self, state: JobState) {
        self.release();
        if !self.told {
            let message = match state {
                JobState::Expired => ImageError::Timeout(format!(
                    "the request passed its timeout of {} ms; ask for fewer steps, fewer images or smaller images",
                    self.parts.timeout.as_millis()
                )),
                other => ImageError::Failed(format!("the image job ended as {other:?} before its images")),
            };
            // A route that went away drops the receiver, and the send gives an error at once.
            let _ = self.parts.out.send(Err(message));
        }
    }

    fn request(&self) -> &'static str {
        "image_generation"
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        self.release();
        self.tell(Err(ImageError::Unavailable {
            engine: engine.to_owned(),
            state,
        }));
    }
}
