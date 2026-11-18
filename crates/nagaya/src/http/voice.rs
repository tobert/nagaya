//! Generations of a served speech model (spec 11.8a): a speech context in, the audio of its last line out as a WAV
//! object, with the codes kept beside it and the head held after the line.

use std::collections::BTreeMap;
use std::sync::Arc;

use gogi_proto::context::{MediaCount, MediaUsage, SnapshotInfo, Tier as SnapshotTier, Usage};
use gogi_proto::generation::{Finish, GenerationDone, GenerationRequest, SealedOutput};
use gogi_proto::identity::ServedModel;
use gogi_proto::message::{Content, Part, Role};
use gogi_proto::objects::{Named, ObjectRef, check_reads, objects_of};
use gogi_proto::{Invalid, content_id};
use nagaya_store::ContentId;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use gogi_proto::generation::{OutputEvent, StreamEvent};
use tokio::sync::mpsc;

use super::pending::PendingOutput;
use super::{Council, RequestId, guarded};
use crate::council::error::ServiceError;
use crate::speech::{ContextItem, SpeechContext, SpeechError, SpeechService, Spoken, Told};

/// The media type of every output of a speech model.
pub const WAV: &str = "audio/wav";

/// The sample rate of the engine's audio.
const RATE: u32 = 24_000;

/// A speech generation after its checks.
pub struct Checked {
    pub context: SpeechContext,
    /// The distinct audio objects that the context reads, with their bytes.
    pub read: BTreeMap<String, u64>,
}

fn invalid(param: impl Into<String>, message: String) -> ServiceError {
    ServiceError::Invalid(Invalid::param(param, message))
}

/// The service error of a speech error. A failed engine or store is internal, so the process stops (spec 8.5).
pub fn service_error(error: SpeechError) -> ServiceError {
    match error {
        SpeechError::Invalid { param, message } => invalid(param, message),
        SpeechError::Busy(message) => ServiceError::Busy(message),
        SpeechError::ContextFull { positions, room } => ServiceError::ContextFull {
            prompt_tokens: positions,
            context_tokens: room,
        },
        SpeechError::Timeout(_) => ServiceError::Timeout,
        SpeechError::Stopping(_) => ServiceError::Stopping,
        SpeechError::Unavailable { engine, state } => ServiceError::Unavailable { engine, state },
        SpeechError::Failed(message) => ServiceError::internal(message),
    }
}

/// The text of a message with text only: its string, or its text parts joined.
fn text_of(content: &Option<Content>) -> String {
    content
        .as_ref()
        .map(|c| {
            c.parts()
                .into_iter()
                .filter_map(|part| match part {
                    Part::Text { text } => Some(text),
                    Part::Object { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

/// Check a speech generation and read the codes of the audio that its context holds. It runs on a call thread: it
/// reads the object index and the code files.
pub fn check(
    council: &Council,
    service: &SpeechService,
    model: &ServedModel,
    request: &GenerationRequest,
) -> Result<Checked, ServiceError> {
    request.validate().map_err(ServiceError::Invalid)?;
    let Some(output) = &request.output else {
        return Err(invalid(
            "output",
            format!(
                "nothing ran: {} writes audio only; give an `output` of `{WAV}`",
                model.model
            ),
        ));
    };
    output.check(model).map_err(ServiceError::Invalid)?;
    let mut language = "auto".to_owned();
    for (key, value) in &output.options {
        match (key.as_str(), value.as_str()) {
            ("language", Some(given)) => language = given.to_owned(),
            ("language", None) => {
                return Err(invalid(
                    "output.options.language",
                    format!("nothing ran: output.options.language is {value}, not a string such as \"english\""),
                ));
            }
            _ => {
                return Err(invalid(
                    format!("output.options.{key}"),
                    format!(
                        "nothing ran: {} takes the output option `language` only, not `{key}`",
                        model.model
                    ),
                ));
            }
        }
    }
    // The engine samples with a temperature and top_k. Temperature 0 is greedy: top_k 1 (spec 11.8a).
    let mut temperature = None;
    let mut top_k = None;
    if let Some(sample) = &request.sample {
        if let Some((key, value)) = [("top_p", sample.top_p, 1.0), ("min_p", sample.min_p, 0.0)]
            .into_iter()
            .find_map(|(key, given, off)| given.filter(|v| *v != off).map(|v| (key, v)))
        {
            return Err(invalid(
                format!("sample.{key}"),
                format!(
                    "nothing ran: sample.{key} is {value}, and {} samples with temperature and top_k only",
                    model.model
                ),
            ));
        }
        if sample.top_k == Some(0) {
            return Err(invalid(
                "sample.top_k",
                format!(
                    "nothing ran: {} samples from top_k codes; give a top_k of 1 or more",
                    model.model
                ),
            ));
        }
        top_k = sample.top_k.map(|k| u32::try_from(k).unwrap_or(u32::MAX));
        match sample.temperature {
            Some(0.0) => {
                temperature = Some(1.0);
                top_k = Some(1);
            }
            Some(t) => temperature = Some(t as f32),
            None => {}
        }
    }
    if !request.decisions.is_empty() {
        return Err(invalid(
            "decisions",
            format!("nothing ran: {} takes no riding decision", model.model),
        ));
    }
    let objects: Vec<Named> = objects_of(&request.context.messages);
    for named in &objects {
        named.object.check(&named.at).map_err(ServiceError::Invalid)?;
    }
    check_reads(model, &objects).map_err(ServiceError::Invalid)?;

    let store = council.objects().ok_or_else(|| {
        ServiceError::internal("a speech model is served with no object store; name an `objects` budget in `[store]`")
    })?;
    let mut voice = None;
    let mut items = Vec::new();
    let mut snaps = Vec::new();
    let mut read = BTreeMap::new();
    for (i, message) in request.context.messages.iter().enumerate() {
        if message.open {
            return Err(invalid(
                "continue",
                format!("nothing ran: {} continues no message", model.model),
            ));
        }
        match message.role {
            Role::System if i == 0 => voice = Some(text_of(&message.content)),
            Role::User => {
                let line = text_of(&message.content);
                if line.trim().is_empty() {
                    return Err(invalid(
                        format!("messages[{i}].content"),
                        format!("nothing ran: messages[{i}] is an empty line; give the text to speak"),
                    ));
                }
                items.push(ContextItem::Line(line));
            }
            Role::Assistant => {
                let parts: Vec<(usize, ObjectRef)> = message
                    .content
                    .as_ref()
                    .map(|content| content.parts())
                    .unwrap_or_default()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(j, part)| match part {
                        Part::Object { id, mime } => Some((j, ObjectRef { id, mime })),
                        Part::Text { .. } => None,
                    })
                    .collect();
                let [(j, object)] = parts.as_slice() else {
                    return Err(invalid(
                        format!("messages[{i}]"),
                        format!(
                            "nothing ran: messages[{i}] holds {} objects; an assistant message of a speech context holds \
                             the one audio output of the line before it",
                            parts.len()
                        ),
                    ));
                };
                let at = format!("messages[{i}].content[{j}]");
                let id = ContentId::from_hex(object.id.trim_start_matches("blake3:"))
                    .map_err(|_| invalid(at.clone(), format!("nothing ran: {at}.id is not a content id")))?;
                let held = {
                    let store = store.lock().expect("the object store lock is poisoned");
                    store.mime_of(&id).map(str::to_owned).zip(store.bytes_of(&id))
                };
                let Some((held, bytes)) = held else {
                    return Err(ServiceError::NotHeld {
                        message: format!(
                            "nothing ran: {at} is {}, which the server does not hold; speak its line again",
                            object.id
                        ),
                        param: at,
                    });
                };
                if held != object.mime {
                    return Err(invalid(
                        at.clone(),
                        format!(
                            "nothing ran: {at} is {} of media type {}, but it was put as {held}",
                            object.id, object.mime
                        ),
                    ));
                }
                // Codes that the book does not hold are as an object that the server does not hold (spec 11.8a).
                let frames = service.codes(&id).map_err(ServiceError::internal)?.ok_or_else(|| ServiceError::NotHeld {
                    message: format!(
                        "nothing ran: {at} is {}, whose codes {} does not hold: it did not write it, or it dropped the \
                         codes; the engine has no codec encoder, so speak its line again",
                        object.id, model.model
                    ),
                    param: at.clone(),
                })?;
                read.insert(object.id.clone(), bytes);
                items.push(ContextItem::Audio(frames));
            }
            Role::System | Role::Tool => {
                return Err(invalid(
                    format!("messages[{i}].role"),
                    format!(
                        "nothing ran: messages[{i}] is a `{}` message; a speech context has one `system` message, \
                         first, then `user` lines and `assistant` audio",
                        message.role.as_str()
                    ),
                ));
            }
        }
        snaps.push(message.snap);
    }
    let seed = request
        .sample
        .as_ref()
        .and_then(|sample| sample.seed)
        .unwrap_or_else(chosen_seed);
    Ok(Checked {
        context: SpeechContext {
            voice,
            items,
            snaps,
            language,
            seed,
            temperature,
            top_k,
            schedule: crate::council::Schedule::of(request.class, request.priority),
        },
        read,
    })
}

/// A seed for a request that names none: from the clock, so two requests differ.
fn chosen_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64)
}

/// The header of a WAV of PCM, mono, 16 bits, at 24 kHz, with `data` bytes of samples. `None` gives the sizes of a
/// stream, `0xFFFFFFFF`, because its length is not known (spec 11.8a "The WAV of an output").
pub fn wav_header(data: Option<u32>) -> [u8; 44] {
    let mut h = [0u8; 44];
    let (riff, data) = match data {
        Some(data) => (data + 36, data),
        None => (u32::MAX, u32::MAX),
    };
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&riff.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&1u16.to_le_bytes());
    h[24..28].copy_from_slice(&RATE.to_le_bytes());
    h[28..32].copy_from_slice(&(RATE * 2).to_le_bytes());
    h[32..34].copy_from_slice(&2u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data.to_le_bytes());
    h
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as u64)
}

/// Speak a checked generation, seal its WAV, keep its codes, and give the reply. With `pending`, each quantum's audio
/// goes to the pending id as it decodes. It runs on a call thread, and blocks until the end of the line.
pub fn speak(
    council: &Council,
    service: &SpeechService,
    model: &str,
    checked: Checked,
    pending: Option<&PendingOutput>,
    stopped: &dyn Fn() -> bool,
) -> Result<GenerationDone, ServiceError> {
    let told = service.generate(&checked.context).map_err(service_error)?;
    if let Some(pending) = pending {
        pending.push(&wav_header(None));
    }
    let mut pcm = Vec::new();
    let mut spoken: Option<Spoken> = None;
    for item in told.iter() {
        match item.map_err(service_error)? {
            Told::Audio(samples) => {
                // A closed stream cancels the output. The receiver drops, so the job stops at its next quantum.
                if stopped() {
                    return Err(ServiceError::Cancelled);
                }
                let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                if let Some(pending) = pending {
                    pending.push(&bytes);
                }
                pcm.extend(bytes);
            }
            Told::End(end) => {
                spoken = Some(end);
                break;
            }
        }
    }
    // A stop drops the job with no end, so the line was cancelled (spec 11.7).
    let spoken = spoken.ok_or(ServiceError::Cancelled)?;
    let data = u32::try_from(pcm.len()).map_err(|_| ServiceError::internal("the audio of a line is over 4 GiB"))?;
    let mut wav = wav_header(Some(data)).to_vec();
    wav.extend(&pcm);
    let id = seal(council, &wav, WAV)?;
    let objects = council.objects();
    let held = |id: &ContentId| {
        objects.as_ref().is_some_and(|store| {
            store
                .lock()
                .expect("the object store lock is poisoned")
                .bytes_of(id)
                .is_some()
        })
    };
    service
        .keep_codes(&id, &spoken.codes, &held)
        .map_err(ServiceError::internal)?;
    let text = format!("blake3:{}", id.hex());
    let samples = pcm.len() as u64 / 2;
    // The snapshots of the context with the reply appended: the marked ends that the build held, then the head.
    let reply = (checked.context.snaps.len(), spoken.prompt + spoken.codes.len());
    let snapshots = spoken
        .held
        .iter()
        .map(|(message, positions, content)| (*message, *positions, content))
        .chain(spoken.head.as_ref().map(|head| (reply.0, reply.1, head)))
        .map(|(message, positions, content)| SnapshotInfo {
            id: crate::council::snapshot_id(content),
            message: message as u64,
            tokens: positions as u64,
            tier: Some(SnapshotTier::Host),
            bytes: None,
        })
        .collect();
    Ok(GenerationDone {
        model: model.to_owned(),
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        finish: Finish::EndTurn,
        usage: Usage {
            prompt: spoken.prompt as u64,
            kept: spoken.kept as u64,
            fed: spoken.fed as u64,
            completion: 0,
            reasoning: 0,
            media: Some(MediaUsage {
                read: MediaCount {
                    objects: checked.read.len() as u64,
                    bytes: checked.read.values().sum(),
                    ..MediaCount::default()
                },
                written: MediaCount {
                    objects: 1,
                    bytes: wav.len() as u64,
                    audio_ms: Some(samples * 1000 / u64::from(RATE)),
                    ..MediaCount::default()
                },
            }),
        },
        head: spoken.head.as_ref().map(crate::council::snapshot_id),
        context: None,
        snapshots,
        stats: None,
        decisions: Vec::new(),
        seed: Some(checked.context.seed),
        phase: None,
        outputs: vec![SealedOutput {
            index: 0,
            id: gogi_proto::ids::ContentId::parse(text).expect("a content id parses"),
            mime: WAV.to_owned(),
            bytes: wav.len() as u64,
        }],
    })
}

/// Hold the bytes of an output of media type `mime` in the object store, and give their content id.
pub(super) fn seal(council: &Council, wav: &[u8], mime: &str) -> Result<ContentId, ServiceError> {
    let store = council
        .objects()
        .ok_or_else(|| ServiceError::internal("a model with outputs is served with no object store"))?;
    let text = content_id(wav);
    let id = ContentId::from_hex(text.trim_start_matches("blake3:")).expect("a content id has 64 hex digits");
    let staged = store.lock().expect("the object store lock is poisoned").staging_path();
    // The bytes reach the disk before the rename, so a crash leaves no object whose bytes do not hash to its id.
    std::fs::File::create(&staged)
        .and_then(|mut file| {
            std::io::Write::write_all(&mut file, wav)?;
            file.sync_all()
        })
        .map_err(|error| ServiceError::internal(format!("cannot write `{}`: {error}", staged.display())))?;
    let mut store = store.lock().expect("the object store lock is poisoned");
    store
        .commit(&id, mime, &staged, wav.len() as u64, now_ms())
        .map_err(|error| match error {
            nagaya_store::objects::ObjectError::TooLarge { limit, .. } => ServiceError::TooLarge {
                param: limit.to_owned(),
                message: error.to_string(),
            },
            nagaya_store::objects::ObjectError::Io(message) => ServiceError::internal(message),
        })?;
    Ok(id)
}

/// The served model of a speech service (spec 11.8a). Its hashes are the engine's after its first open since the
/// start, and empty before.
pub fn served_model(service: &SpeechService, status: gogi_proto::identity::ModelStatus) -> ServedModel {
    let identity = service.identity();
    let config = service.config();
    ServedModel {
        model: config.model.clone(),
        aliases: Vec::new(),
        status,
        weight_hash: identity
            .as_ref()
            .map(|i| i.checkpoint_digest.clone())
            .unwrap_or_default(),
        tokenizer_hash: identity.as_ref().map(|i| i.tokenizer_hash.clone()).unwrap_or_default(),
        template: String::new(),
        engine: config.engine.clone(),
        device: None,
        text_stop: None,
        limits: gogi_proto::identity::ModelLimits {
            control_vector_contexts: None,
            context_tokens: config.context_positions,
            decision_context_tokens: 0,
            snapshot_bytes: config.snapshot_bytes,
            contexts_per_decision: 0,
            default_timeout_ms: config.timeout.as_millis() as u64,
            text_tokens: None,
            replies_per_generation: None,
        },
        embeddings: None,
        layer_loops: None,
        capabilities: vec![gogi_proto::identity::Capability::Generate],
        media: Some(gogi_proto::identity::Media {
            inputs: Vec::new(),
            outputs: vec![WAV.to_owned()],
            // A reply holds audio and no text, so the model takes no build, decision or raw read (gogi c3acffe).
            text_replies: false,
        }),
    }
}

/// The served model and the service of a speech model that a request names.
pub fn route(council: &Council, name: &str) -> Option<(Arc<SpeechService>, ServedModel)> {
    let service = council
        .speech_service(name)
        .filter(|service| service.serves_contexts())?;
    let model = served_model(&service, council.model_status(service.engine_state()));
    Some((service, model))
}

/// What the call thread of a stream gives the response.
enum Item {
    Output(OutputEvent),
    End(Box<Result<GenerationDone, ServiceError>>),
}

/// Serve a generation of a speech model: a reply with no stream, or a stream with an `output` event and `done`.
pub async fn generation(
    council: Arc<Council>,
    service: Arc<SpeechService>,
    model: ServedModel,
    request: GenerationRequest,
    id: RequestId,
) -> Response {
    if !request.stream.unwrap_or(false) {
        let inner = Arc::clone(&council);
        let result = council
            .call(move || {
                let checked = check(&inner, &service, &model, &request)?;
                inner.recorded(&request.context, || {
                    speak(&inner, &service, &model.model, checked, None, &|| false)
                })
            })
            .await;
        return super::routes::answer(&id, result);
    }
    let permit = match council.permit() {
        Ok(permit) => permit,
        Err(failure) => return failure.respond(&id),
    };
    let (out, mut items) = mpsc::channel::<Item>(1);
    let inner = Arc::clone(&council);
    let started = council.spawn(move || {
        let mut started: Option<String> = None;
        let result = guarded(|| {
            let checked = check(&inner, &service, &model, &request)?;
            let (pending, output) = inner.pending().start(WAV);
            started = Some(pending.clone());
            let event = OutputEvent {
                index: 0,
                mime: WAV.to_owned(),
                pending,
            };
            if out.blocking_send(Item::Output(event)).is_err() {
                return Err(ServiceError::Cancelled);
            }
            inner.recorded(&request.context, || {
                speak(&inner, &service, &model.model, checked, Some(&output), &|| {
                    out.is_closed()
                })
            })
        });
        // Outside the guard, so a panic in the call also ends its pending id.
        if let Some(pending) = &started {
            match &result {
                Ok(done) => inner.pending().seal(pending, done.outputs[0].id.as_str()),
                Err(_) => inner.pending().cancel(pending),
            }
        }
        let _ = out.blocking_send(Item::End(Box::new(result)));
        drop(permit);
    });
    if let Err(failure) = started {
        return failure.respond(&id);
    }
    // A failure before the output begins gets a status; a failure after the `200` is an `error` event.
    let first = match items.recv().await {
        Some(Item::End(result)) if result.is_err() => {
            let error = result.err().expect("the result is an error");
            return council.failed(error).respond(&id);
        }
        Some(first) => first,
        None => {
            return council
                .failed(ServiceError::internal("a speech generation ended with no result"))
                .respond(&id);
        }
    };
    let state = (Some(first), items, council, id, false);
    let stream = futures_util::stream::unfold(state, |(first, mut items, council, id, ended)| async move {
        if ended {
            return None;
        }
        let item = match first {
            Some(item) => Some(item),
            None => items.recv().await,
        };
        let (event, ended) = match item {
            Some(Item::Output(output)) => (StreamEvent::Output(output), false),
            Some(Item::End(result)) => match *result {
                Ok(done) => (StreamEvent::Done(Box::new(done)), true),
                Err(error) => (super::stream::error_event(&council, error, &id), true),
            },
            None => (
                super::stream::error_event(
                    &council,
                    ServiceError::internal("a speech generation ended with no result"),
                    &id,
                ),
                true,
            ),
        };
        let sse = Event::default().event(event.name()).data(event.data());
        Some((
            Ok::<Event, std::convert::Infallible>(sse),
            (None, items, council, id, ended),
        ))
    });
    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sealed_header_has_the_true_sizes_and_a_stream_header_the_unknown_ones() {
        let sealed = wav_header(Some(4800));
        assert_eq!(u32::from_le_bytes(sealed[4..8].try_into().unwrap()), 4836);
        assert_eq!(u32::from_le_bytes(sealed[40..44].try_into().unwrap()), 4800);
        let stream = wav_header(None);
        assert_eq!(stream[4..8], [0xff; 4]);
        assert_eq!(stream[40..44], [0xff; 4]);
        assert_eq!(
            sealed[8..40],
            stream[8..40],
            "the two headers differ only in the size fields"
        );
        assert_eq!(u32::from_le_bytes(sealed[24..28].try_into().unwrap()), 24_000);
    }
}
