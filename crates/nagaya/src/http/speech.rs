//! The speech route `POST /v1/audio/speech` in the shape of OpenAI's speech API (spec 11.8).
//!
//! A call thread starts the request and moves each chunk to the response. The route waits for the first chunk before
//! it sends the status. Thus a request that fails before its first chunk gets a JSON error in OpenAI's shape, and a
//! failure after the `200` ends the body early. A client that goes away drops the receiver of the response, so the job
//! sees a closed queue at its next quantum and stops.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Extension, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

use super::{Council, RequestId};
use crate::speech::{Chunk, SpeechError, SpeechRequest};

/// The most characters of `input` (spec 11.8).
pub const MAX_INPUT_CHARS: usize = 4096;

/// The most bytes of a request body.
const MAX_BODY_BYTES: usize = 64 << 10;

/// The sample rate of the audio.
const SAMPLE_RATE: u32 = 24_000;

/// The format of the reply body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wav,
    Pcm,
}

/// An error in OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: Option<&'static str>,
    pub param: Option<&'static str>,
    pub message: String,
}

impl OpenAiError {
    fn invalid(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: None,
            param: Some(param),
            message: message.into(),
        }
    }

    fn server(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind: "server_error",
            code: Some(code),
            param: None,
            message: message.into(),
        }
    }

    fn respond(&self) -> Response {
        let body = json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code,
        }});
        (self.status, axum::Json(body)).into_response()
    }
}

impl From<SpeechError> for OpenAiError {
    fn from(error: SpeechError) -> Self {
        match error {
            SpeechError::Invalid { param, message } => OpenAiError::invalid(param, message),
            SpeechError::Busy(message) => OpenAiError::server(StatusCode::SERVICE_UNAVAILABLE, "busy", message),
            SpeechError::Timeout(message) => OpenAiError::server(StatusCode::GATEWAY_TIMEOUT, "timeout", message),
            SpeechError::Stopping(message) => OpenAiError::server(StatusCode::SERVICE_UNAVAILABLE, "draining", message),
            SpeechError::Failed(message) => {
                OpenAiError::server(StatusCode::INTERNAL_SERVER_ERROR, "engine_failed", message)
            }
        }
    }
}

/// A request after its checks: the served model, the request, and the format.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    pub model: String,
    pub request: SpeechRequest,
    pub format: Format,
    pub voice: Option<String>,
}

/// Check a request body against the rules of spec 11.8. `id` gives the seed of a request that sends none.
pub fn check(body: &[u8], id: &RequestId) -> Result<Checked, OpenAiError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| OpenAiError::invalid("body", format!("the body is not JSON: {error}")))?;
    let Some(object) = value.as_object() else {
        return Err(OpenAiError::invalid("body", "the body is not a JSON object"));
    };
    let text = |key: &'static str| -> Result<Option<String>, OpenAiError> {
        match object.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(other) => Err(OpenAiError::invalid(key, format!("`{key}` is {other}, not a string"))),
        }
    };
    let model = text("model")?.ok_or_else(|| OpenAiError::invalid("model", "`model` is required"))?;
    let input = text("input")?.ok_or_else(|| OpenAiError::invalid("input", "`input` is required"))?;
    if input.trim().is_empty() {
        return Err(OpenAiError::invalid(
            "input",
            "`input` is empty; send the text to speak",
        ));
    }
    let chars = input.chars().count();
    if chars > MAX_INPUT_CHARS {
        return Err(OpenAiError::invalid(
            "input",
            format!("`input` has {chars} characters, and the most is {MAX_INPUT_CHARS}; send one line or paragraph"),
        ));
    }
    let instructions = text("instructions")?.unwrap_or_default();
    if instructions.trim().is_empty() {
        return Err(OpenAiError::invalid(
            "instructions",
            "`instructions` is required: the engine designs the voice from it, for example \"A calm female voice\"",
        ));
    }
    let format = match text("response_format")?.as_deref() {
        None | Some("wav") => Format::Wav,
        Some("pcm") => Format::Pcm,
        Some(other) => {
            return Err(OpenAiError::invalid(
                "response_format",
                format!("`response_format` is `{other}`; this server sends `wav` or `pcm`"),
            ));
        }
    };
    match object.get("speed") {
        None | Some(Value::Null) => {}
        Some(speed) if speed.as_f64() == Some(1.0) => {}
        Some(other) => {
            return Err(OpenAiError::invalid(
                "speed",
                format!("`speed` is {other}; the engine has no time stretch, so send 1.0 or no `speed`"),
            ));
        }
    }
    let language = text("language")?.unwrap_or_else(|| "auto".to_owned());
    let seed = match object.get("seed") {
        None | Some(Value::Null) => {
            let hash = blake3::hash(id.0.as_bytes());
            u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("8 bytes"))
        }
        Some(value) => value
            .as_u64()
            .ok_or_else(|| OpenAiError::invalid("seed", format!("`seed` is {value}, not an unsigned integer")))?,
    };
    Ok(Checked {
        model,
        request: SpeechRequest {
            input,
            instructions,
            language,
            seed,
        },
        format,
        voice: text("voice")?,
    })
}

/// The header of a WAV stream: PCM, mono, 16 bits, 24 kHz. Both size fields are `0xFFFFFFFF`, because the length is
/// not known when the header goes (spec 11.8).
pub fn wav_header() -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&1u16.to_le_bytes());
    h[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    h[28..32].copy_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    h[32..34].copy_from_slice(&2u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    h
}

fn pcm_bytes(samples: &[i16]) -> Bytes {
    Bytes::from(samples.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())
}

/// `POST /v1/audio/speech`.
pub async fn speech(State(council): State<Arc<Council>>, Extension(id): Extension<RequestId>, body: Body) -> Response {
    let body = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            return OpenAiError {
                status: StatusCode::PAYLOAD_TOO_LARGE,
                kind: "invalid_request_error",
                code: None,
                param: None,
                message: format!("the body did not arrive whole, or it is past {MAX_BODY_BYTES} bytes: {error}"),
            }
            .respond();
        }
    };
    let checked = match check(&body, &id) {
        Ok(checked) => checked,
        Err(error) => return error.respond(),
    };
    let Some(service) = council.speech_service(&checked.model) else {
        return OpenAiError {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: Some("model_not_found"),
            param: Some("model"),
            message: format!(
                "the model `{}` is not a speech model of this server; it serves {:?}",
                checked.model,
                council.speech_models()
            ),
        }
        .respond();
    };
    if council.draining() {
        return OpenAiError::server(StatusCode::SERVICE_UNAVAILABLE, "draining", "the server is stopping").respond();
    }
    tracing::info!(
        "speech request {}: {} characters, voice {:?}, language `{}`",
        id.0,
        checked.request.input.chars().count(),
        checked.voice,
        checked.request.language
    );
    let Ok(permit) = council.permit() else {
        return OpenAiError::server(
            StatusCode::TOO_MANY_REQUESTS,
            "busy",
            "nothing ran: this server runs the most requests that it runs at once; try again later",
        )
        .respond();
    };
    let (first_tx, first_rx) = oneshot::channel::<Result<Chunk, SpeechError>>();
    let (rest_tx, rest_rx) = mpsc::channel::<Result<Chunk, SpeechError>>(1);
    let request = checked.request;
    let started = council.spawn(move || {
        let chunks = match service.start(&request) {
            Ok(chunks) => chunks,
            Err(error) => {
                let _ = first_tx.send(Err(error));
                return;
            }
        };
        let mut first = Some(first_tx);
        for item in chunks {
            let last = !matches!(item, Ok(Chunk::Audio(_)));
            let sent = match first.take() {
                Some(tx) => tx.send(item).is_ok(),
                None => rest_tx.blocking_send(item).is_ok(),
            };
            // A client that went away drops the receivers; dropping `chunks` then stops the job.
            if !sent || last {
                break;
            }
        }
        drop(permit);
    });
    if let Err(failure) = started {
        tracing::error!("speech request {}: {failure:?}", id.0);
        return OpenAiError::server(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "cannot start a call thread",
        )
        .respond();
    }
    let first = match first_rx.await {
        Ok(Ok(first)) => first,
        Ok(Err(error)) => {
            if let SpeechError::Failed(message) = &error {
                tracing::error!("speech request {}: {message}", id.0);
            }
            return OpenAiError::from(error).respond();
        }
        Err(_) => {
            return OpenAiError::server(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "the speech call ended with no result; the log has its panic",
            )
            .respond();
        }
    };
    let head = match checked.format {
        Format::Wav => Bytes::copy_from_slice(&wav_header()),
        Format::Pcm => Bytes::new(),
    };
    let first = match first {
        Chunk::Audio(samples) => Some(pcm_bytes(&samples)),
        Chunk::End => None,
    };
    let ended = first.is_none();
    let state = (Some(head), first, rest_rx, id, ended);
    let stream = futures_util::stream::unfold(state, |(head, first, mut rest, id, ended)| async move {
        if let Some(head) = head {
            return Some((Ok(head), (None, first, rest, id, ended)));
        }
        if let Some(first) = first {
            return Some((Ok(first), (None, None, rest, id, ended)));
        }
        if ended {
            return None;
        }
        match rest.recv().await {
            Some(Ok(Chunk::Audio(samples))) => Some((Ok(pcm_bytes(&samples)), (None, None, rest, id, false))),
            Some(Ok(Chunk::End)) => None,
            Some(Err(error)) => {
                tracing::error!("speech request {}: the body ends early: {error}", id.0);
                Some((
                    Err(std::io::Error::other(error.to_string())),
                    (None, None, rest, id, true),
                ))
            }
            None => {
                tracing::error!("speech request {}: the body ends early: the job sent no end", id.0);
                Some((
                    Err(std::io::Error::other("the speech job sent no end")),
                    (None, None, rest, id, true),
                ))
            }
        }
    });
    let mut response = Body::from_stream(stream).into_response();
    let kind = match checked.format {
        Format::Wav => "audio/wav",
        Format::Pcm => "audio/pcm",
    };
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(kind));
    response
}
