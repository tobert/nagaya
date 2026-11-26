//! `POST /nagaya/v1/analysis/beats` (spec 11.19): the beat and downbeat times of an `audio/wav` object.
//!
//! The family reads the WAV, computes the log-mel and splits it into chunks on a call thread. The engine runs each
//! chunk as one quantum of kind `beats` on the CPU group. The family joins the chunks and picks the peaks.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::response::Response;
use axum::routing::post;
use gogi_proto::Invalid;
use nagaya_family_beatthis::{audio, chunks, mel, peaks, plan};
use serde::{Deserialize, Serialize};

use super::errors::{Failure, parse};
use super::routes::{answer, bytes};
use super::{Council, RequestId};
use crate::beats::{BeatsError, DEFAULT_TIMEOUT, MAX_SECONDS};
use crate::council::Schedule;
use crate::council::error::ServiceError;
use crate::embedding::Cancel;

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The route of the beats services.
pub fn routes() -> Router<Arc<Council>> {
    Router::new().route("/nagaya/v1/analysis/beats", post(beats))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BeatsRequest {
    model: String,
    audio: String,
    class: Option<gogi_proto::identity::Class>,
    priority: Option<i32>,
    timeout_ms: Option<u64>,
    #[serde(default)]
    logits: bool,
}

#[derive(Serialize)]
struct Logits {
    beat: Vec<f32>,
    downbeat: Vec<f32>,
}

#[derive(Serialize)]
struct BeatsUsage {
    chunks: usize,
    ms: u64,
}

#[derive(Serialize)]
struct BeatsReply {
    model: String,
    beats: Vec<f64>,
    downbeats: Vec<f64>,
    frames: usize,
    fps: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    logits: Option<Logits>,
    usage: BeatsUsage,
}

fn invalid(param: &str, message: impl Into<String>) -> ServiceError {
    ServiceError::Invalid(Invalid::param(param, message.into()))
}

/// The bytes and the media type of a held object, or the 404 that names `audio`.
fn object(council: &Council, id: &str) -> Result<(Vec<u8>, String), ServiceError> {
    let store = council
        .objects()
        .ok_or_else(|| ServiceError::NotFound("nothing ran: this node holds no objects; put the audio first".into()))?;
    let content = id
        .strip_prefix("blake3:")
        .and_then(|hex| nagaya_store::ContentId::from_hex(hex).ok())
        .ok_or_else(|| {
            invalid(
                "audio",
                format!("`{id}` is not a content id; write `blake3:` and 64 hex digits"),
            )
        })?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is before the epoch")
        .as_millis() as u64;
    let found = store
        .lock()
        .expect("the object store lock is poisoned")
        .get(&content, now)
        .map_err(|e| ServiceError::internal(format!("the object store cannot read `{id}`: {e}")))?
        .ok_or_else(|| {
            ServiceError::NotFound(format!("nothing ran: this node holds no object `{id}`; put it first"))
        })?;
    let bytes = std::fs::read(&found.path)
        .map_err(|e| ServiceError::internal(format!("cannot read the file of object `{id}`: {e}")))?;
    Ok((bytes, found.mime))
}

/// What the call of a request gave.
struct Made {
    frames: usize,
    chunks: usize,
    beat: Vec<f32>,
    downbeat: Vec<f32>,
}

async fn beats(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let start = Instant::now();
    let checked = async {
        let request: BeatsRequest = parse(&bytes(body).await?, "beats")?;
        let service = council.beats_service(&request.model).ok_or_else(|| {
            Failure::from(ServiceError::NotFound(format!(
                "nothing ran: this node serves no beats model `{}`",
                request.model
            )))
        })?;
        let state = service.engine_state();
        if state != nagaya_telemetry::EngineState::Ready {
            return Err(council.failed(ServiceError::Unavailable {
                engine: request.model.clone(),
                state,
            }));
        }
        let (wav, mime) = object(&council, &request.audio).map_err(Failure::from)?;
        if mime != "audio/wav" {
            return Err(Failure::from(invalid(
                "audio",
                format!(
                    "object `{}` is `{mime}`, and Beat This! reads `audio/wav`",
                    request.audio
                ),
            )));
        }
        Ok((service, request, wav))
    }
    .await;
    let (service, request, wav) = match checked {
        Ok(checked) => checked,
        Err(failure) => return failure.respond(&id),
    };
    let schedule = Schedule::of(request.class, request.priority);
    let timeout = request.timeout_ms.map(Duration::from_millis).unwrap_or(DEFAULT_TIMEOUT);
    // The token lives in this future. A client that goes away drops it, and the job ends at its next quantum.
    let cancel = Cancel::new();
    let cancelled = cancel.flag();
    let result = council
        .call(move || {
            let samples = audio::read_wav(&wav).map_err(|e| invalid("audio", e.to_string()))?;
            let most = MAX_SECONDS as usize * audio::SAMPLE_RATE as usize;
            if samples.len() > most {
                return Err(invalid(
                    "audio",
                    format!(
                        "the audio has {} samples, more than {MAX_SECONDS} s; send a shorter clip",
                        samples.len()
                    ),
                ));
            }
            let spect = mel::log_mel(&samples).map_err(|e| invalid("audio", e.to_string()))?;
            let frames = mel::frames(samples.len());
            let plan_of = chunks::plan(frames);
            let inputs: Vec<(Vec<f32>, u32)> = plan_of
                .iter()
                .map(|c| (chunks::input(&spect, frames, c), c.len() as u32))
                .collect();
            let made = service
                .run(inputs, schedule, timeout, cancelled)
                .map_err(service_error)?;
            let outputs = match made.recv() {
                Ok(Ok(outputs)) => outputs,
                Ok(Err(error)) => return Err(service_error(error)),
                // The job ended with no word: its client went away, and the cancel ended it.
                Err(_) => return Err(ServiceError::Cancelled),
            };
            let logits: Vec<(Vec<f32>, Vec<f32>)> = outputs.iter().map(|o| plan::logits(o)).collect();
            let (beat, downbeat) = chunks::join(frames, &plan_of, &logits);
            Ok(Made {
                frames,
                chunks: plan_of.len(),
                beat,
                downbeat,
            })
        })
        .await;
    drop(cancel);
    let made = match result {
        Ok(made) => made,
        Err(failure) => return failure.respond(&id),
    };
    let found = peaks::beats(&made.beat, &made.downbeat);
    let reply = BeatsReply {
        model: request.model,
        beats: found.beats,
        downbeats: found.downbeats,
        frames: made.frames,
        fps: mel::FPS,
        logits: request.logits.then_some(Logits {
            beat: made.beat,
            downbeat: made.downbeat,
        }),
        usage: BeatsUsage {
            chunks: made.chunks,
            ms: start.elapsed().as_millis() as u64,
        },
    };
    answer(&id, Ok::<_, Failure>(reply))
}

fn service_error(error: BeatsError) -> ServiceError {
    match error {
        BeatsError::Unavailable { engine, state } => ServiceError::Unavailable { engine, state },
        BeatsError::Stopping(_) => ServiceError::Stopping,
        BeatsError::Timeout(_) => ServiceError::Timeout,
        BeatsError::Failed(message) => ServiceError::internal(message),
    }
}
