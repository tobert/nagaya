//! The job of one image generation on the group of its engine (spec 11.9a "The quanta").
//!
//! The first quantum (`embed`) begins the job in its slot. Each `step` quantum runs one step of every image, and the
//! `step` after the last one decodes them. The body sends what each quantum made to the route.

use std::sync::Arc;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::Duration;

use nagaya_abi::{Engine, ImageJob, Kind, SlotId};
use nagaya_sched::JobState;
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
}

/// Where the job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Begin,
    Steps,
    Decode,
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
            // A full queue waits for the client; the quantum ended, so the wait holds no device work.
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
                let image = engine.image().expect("checked above");
                if let Err(error) = image.image_begin(slot, &self.parts.image) {
                    return self.fail(engine, failed(format!("an image job did not begin: {}", error.message)));
                }
                self.phase = Phase::Steps;
                Ran::More(vec![Kind::Step])
            }
            Phase::Steps => {
                let out = match engine.image().expect("checked above").image_step(slot) {
                    Ok(out) => out,
                    Err(error) => return self.fail(engine, failed(format!("an image step failed: {}", error.message))),
                };
                let steps = self.parts.image.steps;
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
                let wrong = images
                    .iter()
                    .find(|i| i.width != width || i.height != height || i.rgb.len() != (width * height * 3) as usize);
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
