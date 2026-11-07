//! `POST /council/v2/embeddings` of a served embedding model (spec 11.10, gogi "Embeddings").
//!
//! The route checks the request with gogi's rules, and the family gives the ids of each input. A dry run answers with
//! the token counts. Otherwise one job embeds each input, and the route cuts and normalizes each vector.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::Response;
use gogi_proto::context::Usage;
use gogi_proto::decision::DecisionIdentity;
use gogi_proto::embeddings::{Embedding, Embeddings, EmbeddingsRequest, EmbeddingsResponse, cut};
use gogi_proto::error::ErrorType;
use gogi_proto::identity::{Capability, Media, ModelLimits, ModelStatus, ServedModel};
use gogi_proto::message::{Content, Part};

use super::errors::Failure;
use super::routes::answer;
use super::{Council, RequestId};
use crate::council::Schedule;
use crate::council::error::ServiceError;
use crate::embedding::{Cancel, EmbedError, EmbeddingService};

/// The served model of an embedding service (spec 11.10, "The served embedding model"). Its hashes are the engine's
/// after its first open since the start, and empty before.
pub fn served_model(service: &EmbeddingService, status: ModelStatus) -> ServedModel {
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
        template: service.family().template(),
        engine: config.engine.clone(),
        device: None,
        text_stop: None,
        limits: ModelLimits {
            control_vector_contexts: None,
            context_tokens: config.context_tokens,
            decision_context_tokens: 0,
            snapshot_bytes: 0,
            contexts_per_decision: 0,
            default_timeout_ms: config.timeout.as_millis() as u64,
            text_tokens: None,
        },
        embeddings: Some(Embeddings {
            tasks: service.family().tasks(),
            dimensions: config.dimensions.clone(),
            inputs_per_request: config.inputs_per_request,
        }),
        capabilities: vec![Capability::Embed, Capability::DryRun],
        // A model that only embeds writes no text, so it takes no build, decision, raw read or generation (gogi).
        media: Some(Media {
            inputs: Vec::new(),
            outputs: Vec::new(),
            text_replies: false,
        }),
    }
}

/// The served model and the service of an embedding model that a request names.
pub fn route(council: &Council, name: &str) -> Option<(Arc<EmbeddingService>, ServedModel)> {
    let service = council.embedding_service(name)?;
    let model = served_model(&service, council.model_status(service.engine_state()));
    Some((service, model))
}

/// The text of an input. The checks of gogi refused each object part, so each part is text.
fn text_of(input: &Content) -> String {
    match input {
        Content::Text(text) => text.clone(),
        Content::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                Part::Text { text } => text.as_str(),
                other => panic!("the object check let a part through: {other:?}"),
            })
            .collect(),
    }
}

/// The 413 `too_large` of an input past `context_tokens` (gogi "Embeddings"). Nothing is embedded.
fn too_large(index: usize, tokens: u64, context_tokens: u64) -> Failure {
    let mut failure = Failure::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        ErrorType::TooLarge,
        format!(
            "nothing was embedded: inputs[{index}] is {tokens} tokens with its task prompt; context_tokens is \
             {context_tokens}; split the input"
        ),
    );
    failure.body.error.param = Some("context_tokens".to_owned());
    failure.body.error.prompt_tokens = Some(tokens);
    failure.body.error.context_tokens = Some(context_tokens);
    failure
}

/// What the call of a request gave.
enum Outcome {
    /// A dry run: the token count of each input.
    Counted(Vec<u64>),
    /// The token count and the vector of each input, before any normalization.
    Embedded(Vec<u64>, Vec<Vec<f32>>),
    /// An input past `context_tokens`.
    TooLarge(Failure),
    /// The job failed.
    Failed(EmbedError),
}

/// The failure of an embedding job.
fn failure_of(council: &Council, error: EmbedError) -> Failure {
    match error {
        EmbedError::Unavailable { engine, state } => council.failed(ServiceError::Unavailable { engine, state }),
        EmbedError::Stopping(message) => Failure::from(ServiceError::Stopping).with_message(message),
        EmbedError::Timeout(message) => Failure::from(ServiceError::Timeout).with_message(message),
        EmbedError::Failed(message) => council.failed(ServiceError::internal(message)),
    }
}

/// Serve an embedding request of a served embedding model that is `ready`.
pub async fn serve(
    council: Arc<Council>,
    service: Arc<EmbeddingService>,
    model: ServedModel,
    request: EmbeddingsRequest,
    id: RequestId,
) -> Response {
    let dimensions = match request.check(&model) {
        Ok(dimensions) => dimensions,
        Err(invalid) => return Failure::from(ServiceError::Invalid(invalid)).respond(&id),
    };
    if let Err(invalid) = gogi_proto::objects::check_reads(&model, &request.objects()) {
        return Failure::from(ServiceError::Invalid(invalid)).respond(&id);
    }
    let texts: Vec<String> = request.inputs.iter().map(text_of).collect();
    let task = request.task.clone();
    let schedule = Schedule::of(request.class, request.priority);
    let dry_run = request.dry_run;
    let context_tokens = service.config().context_tokens;
    // The token lives in this future. A client that goes away drops it, and the job ends at its next quantum.
    let cancel = Cancel::new();
    let cancelled = cancel.flag();
    let worker = Arc::clone(&service);
    let result = council
        .call(move || {
            let mut ids = Vec::with_capacity(texts.len());
            for text in &texts {
                let input = worker
                    .family()
                    .ids(&task, text)
                    .map_err(|message| ServiceError::internal(format!("the family cannot read an input: {message}")))?;
                ids.push(input);
            }
            let tokens: Vec<u64> = ids.iter().map(|ids| ids.len() as u64).collect();
            if dry_run {
                return Ok(Outcome::Counted(tokens));
            }
            if let Some((index, &count)) = tokens.iter().enumerate().find(|(_, count)| **count > context_tokens) {
                return Ok(Outcome::TooLarge(too_large(index, count, context_tokens)));
            }
            let made = match worker.embed(ids, schedule, cancelled) {
                Ok(made) => made,
                Err(error) => return Ok(Outcome::Failed(error)),
            };
            match made.recv() {
                Ok(Ok(vectors)) => Ok(Outcome::Embedded(tokens, vectors)),
                Ok(Err(error)) => Ok(Outcome::Failed(error)),
                // The job ended with no word: its client went away, and the cancel ended it.
                Err(_) => Err(ServiceError::Cancelled),
            }
        })
        .await;
    drop(cancel);
    let (tokens, vectors) = match result {
        Ok(Outcome::Counted(tokens)) => (tokens, None),
        Ok(Outcome::Embedded(tokens, vectors)) => (tokens, Some(vectors)),
        Ok(Outcome::TooLarge(failure)) => return failure.respond(&id),
        Ok(Outcome::Failed(error)) => return failure_of(&council, error).respond(&id),
        Err(failure) => return failure.respond(&id),
    };
    // A zero vector has no direction, so its cut cannot be normalized (spec 11.10). The request fails, and no NaN
    // goes out.
    if let Some(vectors) = &vectors
        && let Some(index) = vectors
            .iter()
            .position(|vector| vector.iter().take(dimensions as usize).all(|value| *value == 0.0))
    {
        let message = format!(
            "nothing was sent: the engine gave a zero vector for inputs[{index}] at {dimensions} dimensions, which has \
             no direction to normalize; report the input"
        );
        return Failure::from(ServiceError::internal(message)).respond(&id);
    }
    let all: u64 = tokens.iter().sum();
    let embeddings = tokens
        .iter()
        .enumerate()
        .map(|(index, &tokens)| Embedding {
            index: index as u64,
            tokens,
            vector: vectors
                .as_ref()
                .map(|vectors| cut(&vectors[index], dimensions as usize)),
        })
        .collect();
    let response = EmbeddingsResponse {
        model: model.model.clone(),
        task: request.task.clone(),
        dimensions,
        embeddings,
        identity: DecisionIdentity {
            model: model.model.clone(),
            weight_hash: model.weight_hash.clone(),
            tokenizer_hash: model.tokenizer_hash.clone(),
            template: model.template.clone(),
            engine: model.engine.clone(),
            spec_id: None,
        },
        usage: Usage {
            prompt: all,
            kept: 0,
            fed: if vectors.is_some() { all } else { 0 },
            completion: 0,
            reasoning: 0,
            media: None,
        },
    };
    answer(&id, Ok::<_, Failure>(response))
}
