//! The job of `POST /council/v2/control-vectors` (spec 11.12, gogi "Control vectors"): a prompt pass of each context
//! with the residual tap, the means of the two sets, and the vector as canonical safetensors.

use std::sync::Arc;

use gogi_proto::Invalid;
use gogi_proto::context::{MediaCount, MediaUsage, Usage};
use gogi_proto::control::{ControlVector, ControlVectorContexts, ControlVectorRequest, ControlVectorResponse};
use gogi_proto::identity::Class;
use nagaya_abi::Kind;
use nagaya_council::control::vector_of_means;

use super::build::{ContextBody, Got, Then, Want};
use super::error::ServiceError;
use super::{Schedule, Service};

/// The work of one context: feed the whole rendering in an empty slot, and keep the residual tap of its last token.
struct TapThen {
    tokens: Vec<u32>,
    rows: Option<Vec<u8>>,
}

impl Then for TapThen {
    /// The job has no recipe, so it restores nothing.
    const NEEDS_STATE: bool = false;
    const REQUEST: &'static str = "control_vector";

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match got {
            Got::Built(recipes) if recipes.is_empty() => Ok(Want::Tap {
                tokens: std::mem::take(&mut self.tokens),
                kind: Kind::Prefill,
                lens: false,
            }),
            Got::Tapped { residual, .. } => {
                self.rows = Some(residual);
                Ok(Want::Done)
            }
            other => Err(ServiceError::internal(format!(
                "the job of a control vector got {other:?}; it builds no boundary and feeds once"
            ))),
        }
    }
}

impl Service {
    /// Make a control vector from two sets of contexts (gogi "Control vectors", spec 11.12). The route ran gogi's
    /// `validate` and `check` before. `put` holds the bytes of the vector as an object and gives its content id.
    ///
    /// Each context renders with the generation prompt and feeds in an empty slot, with no snapshot. The tap reads the
    /// residual after each layer at the last token. Each layer is the mean of `positive` less the mean of `negative`,
    /// summed in `f64` in request order. The class is `batch` unless the request names another.
    pub fn control_vector(
        &self,
        request: &ControlVectorRequest,
        put: &dyn Fn(&[u8]) -> Result<gogi_proto::ids::ContentId, ServiceError>,
    ) -> Result<ControlVectorResponse, ServiceError> {
        let Some(limit) = self.shared.limits.control.filter(|_| self.controls()) else {
            return Err(Invalid::param(
                "model",
                format!(
                    "nothing ran: model {} does not declare control_vectors; send control-vector requests to a \
                     served model whose capabilities list control_vectors",
                    self.config.model
                ),
            )
            .into());
        };
        let contexts = request.contexts();
        let mut renderings = Vec::with_capacity(contexts.len());
        for (name, context) in &contexts {
            if let Some(named) = gogi_proto::objects::objects_of(&context.messages).first() {
                let at = format!("{name}.{}", named.at);
                return Err(Invalid::param(
                    at.clone(),
                    format!(
                        "nothing ran: {at} is {} of media type {}, and a control vector reads text only",
                        named.object.id, named.object.mime
                    ),
                )
                .into());
            }
            // The generation prompt is part of the prompt: the tap reads its last token.
            let tokens = self.family.render(context, true)?.tokens;
            if tokens.len() > self.config.context_tokens {
                return Err(ServiceError::past_context(tokens.len(), self.config.context_tokens));
            }
            renderings.push(tokens);
        }
        let prompt: u64 = renderings.iter().map(|tokens| tokens.len() as u64).sum();
        let usage = |written: Option<u64>| Usage {
            prompt,
            kept: 0,
            fed: prompt,
            completion: 0,
            reasoning: 0,
            media: written.map(|bytes| MediaUsage {
                read: MediaCount::default(),
                written: MediaCount {
                    objects: 1,
                    bytes,
                    ..MediaCount::default()
                },
            }),
        };
        let mut response = ControlVectorResponse {
            model: self.config.model.clone(),
            vector: None,
            layers: limit.layers,
            width: limit.width,
            contexts: ControlVectorContexts {
                positive: request.positive.len() as u64,
                negative: request.negative.len() as u64,
            },
            identity: self.identity(),
            usage: usage(None),
        };
        if request.dry_run {
            return Ok(response);
        }
        let schedule = Schedule::of(Some(request.class.unwrap_or(Class::Batch)), request.priority);
        let row_values = limit.layers as usize * limit.width as usize;
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(renderings.len());
        for ((name, _), tokens) in contexts.iter().zip(renderings) {
            let room = tokens.len();
            let then = TapThen { tokens, rows: None };
            let shared = Arc::clone(&self.shared);
            let job = |tx| ContextBody::with(shared, Arc::from(Vec::new()), Vec::new(), room, then, tx).no_copies();
            let finished = self.run_context(job, schedule, None)?;
            let bytes = finished
                .then
                .rows
                .ok_or_else(|| ServiceError::internal(format!("the job of {name} ended with no residual rows")))?;
            if bytes.len() != row_values * 4 {
                return Err(ServiceError::internal(format!(
                    "the residual tap of {name} gave {} bytes; one row of {} layers of {} f32 values is {} bytes",
                    bytes.len(),
                    limit.layers,
                    limit.width,
                    row_values * 4
                )));
            }
            rows.push(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            );
        }
        let (positive, negative) = rows.split_at(request.positive.len());
        let positive: Vec<&[f32]> = positive.iter().map(Vec::as_slice).collect();
        let negative: Vec<&[f32]> = negative.iter().map(Vec::as_slice).collect();
        let means = vector_of_means(&positive, &negative)
            .map_err(|error| ServiceError::internal(format!("the means of the residual rows: {error}")))?;
        let width = limit.width as usize;
        let vector = ControlVector {
            model: self.config.model.clone(),
            weight_hash: self.identity.checkpoint_digest.clone(),
            layers: (0..limit.layers)
                .map(|layer| {
                    let at = layer as usize * width;
                    (layer, means[at..at + width].to_vec())
                })
                .collect(),
        };
        let bytes = vector.to_bytes();
        response.vector = Some(put(&bytes)?);
        response.usage = usage(Some(bytes.len() as u64));
        Ok(response)
    }
}
