//! Generations with an `output` of `audio/wav` of a served text model that also speaks, such as LFM3 Omni (spec
//! 11.11). The reply holds its text and one WAV output, sealed as an object. A stream sends the `output` event with a
//! pending id, then the token events of the text, then `done`.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use gogi_proto::context::{MediaCount, MediaUsage};
use gogi_proto::generation::{GenerationDone, GenerationRequest, OutputEvent, SealedOutput, StreamEvent, TokenEvent};
use tokio::sync::mpsc;

use super::pending::PendingOutput;
use super::voice::{seal, wav_header};
use super::{Council, RequestId, Served, guarded};
use crate::council::error::ServiceError;
pub use crate::council::speech_out::WAV;
use crate::council::speech_out::{AudioSink, RATE, Spoken, pcm_bytes};

/// The token events that wait for a slow client, as for a text stream.
const QUEUE: usize = 256;

/// True when the request asks a served model that speaks for audio. The route then serves it here.
pub fn wants_audio(served: &Served, request: &GenerationRequest) -> bool {
    request.output.is_some() && served.service.speech_out().is_some()
}

/// The whole WAV of PCM samples, with the true sizes in its header (spec 11.8a, "The WAV of an output").
pub fn wav(pcm: &[i16]) -> Result<Vec<u8>, ServiceError> {
    let data = pcm_bytes(pcm);
    let size = u32::try_from(data.len())
        .ok()
        .filter(|size| *size <= u32::MAX - 36)
        .ok_or_else(|| ServiceError::internal("the audio of a reply is over 4 GiB"))?;
    let mut wav = wav_header(Some(size)).to_vec();
    wav.extend(data);
    Ok(wav)
}

/// Seal the WAV of a reply, keep its codes, and give the reply with its output.
fn sealed(council: &Council, served: &Served, spoken: Spoken) -> Result<GenerationDone, ServiceError> {
    let wav = wav(&spoken.pcm)?;
    let id = seal(council, &wav, WAV)?;
    let objects = council.objects();
    let held = |id: &nagaya_store::ContentId| {
        objects.as_ref().is_some_and(|store| {
            store
                .lock()
                .expect("the object store lock is poisoned")
                .bytes_of(id)
                .is_some()
        })
    };
    served.service.keep_output_codes(id, spoken.codes, &held);
    let mut done = spoken.done;
    let samples = spoken.pcm.len() as u64;
    let read = done.usage.media.map(|media| media.read).unwrap_or_default();
    done.usage.media = Some(MediaUsage {
        read,
        written: MediaCount {
            objects: 1,
            bytes: wav.len() as u64,
            audio_ms: Some(samples * 1000 / u64::from(RATE)),
            ..MediaCount::default()
        },
    });
    done.outputs = vec![SealedOutput {
        index: 0,
        id: gogi_proto::ids::ContentId::parse(format!("blake3:{}", id.hex())).expect("a content id parses"),
        mime: WAV.to_owned(),
        bytes: wav.len() as u64,
    }];
    Ok(done)
}

/// What the call thread of a stream gives the response.
enum Item {
    Output(OutputEvent),
    Token(TokenEvent),
    End(Box<Result<GenerationDone, ServiceError>>),
}

/// The sink of a stream: the PCM goes to the pending id. It gives false when the client closed the stream.
fn sink(pending: Arc<PendingOutput>, out: mpsc::Sender<Item>) -> AudioSink {
    Arc::new(move |bytes: &[u8]| {
        pending.push(bytes);
        !out.is_closed()
    })
}

/// Serve a generation with audio: a reply with no stream, or a stream of `output`, `token` events and `done`.
pub async fn generation(
    council: Arc<Council>,
    served: Arc<Served>,
    request: GenerationRequest,
    id: RequestId,
) -> Response {
    if !request.stream.unwrap_or(false) {
        let inner = Arc::clone(&council);
        let result = council
            .call(move || {
                let plan = served.service.prepare_speech(&request)?;
                let spoken = served.service.speak(plan, None, None)?;
                sealed(&inner, &served, spoken)
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
            // The checks run first, so a request that breaks a rule gets a status (spec 11.11, "The stream").
            let plan = served.service.prepare_speech(&request)?;
            let (pending, output) = inner.pending().start(WAV);
            started = Some(pending.clone());
            output.push(&wav_header(None));
            let event = OutputEvent {
                index: 0,
                mime: WAV.to_owned(),
                pending,
            };
            if out.blocking_send(Item::Output(event)).is_err() {
                return Err(ServiceError::Cancelled);
            }
            let (tokens, events) = sync_channel::<TokenEvent>(QUEUE);
            let relay = out.clone();
            let forwarder = std::thread::Builder::new()
                .name("nagaya-stream".to_owned())
                .spawn(move || {
                    for event in events {
                        if relay.blocking_send(Item::Token(event)).is_err() {
                            return;
                        }
                    }
                })
                .map_err(|error| ServiceError::internal(format!("cannot start a stream thread: {error}")))?;
            let spoken = served
                .service
                .speak(plan, Some(tokens), Some(sink(Arc::clone(&output), out.clone())));
            // The sender of the token events dropped with the job, so the forward thread ends.
            let _ = forwarder.join();
            sealed(&inner, &served, spoken?)
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
                .failed(ServiceError::internal("a generation with audio ended with no result"))
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
            Some(Item::Token(token)) => (StreamEvent::Token(token), false),
            Some(Item::End(result)) => match *result {
                Ok(done) => (StreamEvent::Done(Box::new(done)), true),
                Err(error) => (super::stream::error_event(&council, error, &id), true),
            },
            None => (
                super::stream::error_event(
                    &council,
                    ServiceError::internal("a generation with audio ended with no result"),
                    &id,
                ),
                true,
            ),
        };
        let sse = Event::default().event(event.name()).data(event.data());
        Some((Ok::<Event, Infallible>(sse), (None, items, council, id, ended)))
    });
    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wav_of_a_reply_has_the_true_sizes_and_the_samples_in_order() {
        let pcm = [0i16, 1, -1, 32767, -32767];
        let bytes = wav(&pcm).unwrap();
        assert_eq!(bytes.len(), 44 + 10);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 36 + 10);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 10);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1, "mono");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16, "16 bits");
        let samples: Vec<i16> = bytes[44..]
            .chunks(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        assert_eq!(samples, pcm);
    }
}
