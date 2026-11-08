//! The job of the lab route `trace` (`docs/lab.md`, "The rules of `trace`"): a generation that records the numbers of
//! each step, and the object of those numbers.

use gogi_proto::Invalid;
use gogi_proto::generation::{GenerationDone, GenerationRequest};
use nagaya_council::trace::{Direction, MAX_TOP_K, Plan, TOP_K, Trace};

use super::Service;
use super::build::{LENS_TAP, RESIDUAL_TAP};
use super::control;
use super::error::ServiceError;
use super::generations::Tracing;

/// The one lens of `trace.lens` now: the logit lens.
pub const LOGIT_LENS: &str = "logit";

/// The members of `trace` in a request, after their types were checked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceMembers {
    /// The layers, or `None` for every layer.
    pub layers: Option<Vec<u32>>,
    /// The ids of the control vectors, as the client sent them.
    pub vectors: Vec<String>,
    pub top_k: Option<u64>,
    pub lens: Option<String>,
    pub rows: bool,
}

/// The facts of the model that a trace needs and the engine does not give: the layers and the width of the residual,
/// and the end tokens (`docs/lab.md`, "The end tokens").
#[derive(Clone, Debug, PartialEq)]
pub struct ModelFacts {
    pub layers: u32,
    pub width: usize,
    pub ends: Vec<u32>,
}

fn invalid(param: &str, message: String) -> ServiceError {
    ServiceError::Invalid(Invalid::param(param, message))
}

impl Service {
    /// The plan of a trace: the checks of `trace` against the engine, the model and the objects. A fault is a 400 that
    /// names the member, or a 404 for a vector that the store does not hold. Nothing runs.
    pub fn trace_plan(&self, members: &TraceMembers, facts: &ModelFacts) -> Result<Plan, ServiceError> {
        let model = &self.config.model;
        let taps = &self.shared.limits.taps;
        if taps.iter().all(|tap| tap.name != RESIDUAL_TAP) {
            return Err(invalid(
                "model",
                format!(
                    "nothing ran: the engine of {model} has no tap `{RESIDUAL_TAP}`, so it cannot trace; send the \
                     trace to a model whose engine declares it"
                ),
            ));
        }
        let lens = match members.lens.as_deref() {
            None => false,
            Some(LOGIT_LENS) if taps.iter().any(|tap| tap.name == LENS_TAP) => true,
            Some(LOGIT_LENS) => {
                return Err(invalid(
                    "trace.lens",
                    format!("nothing ran: the engine of {model} has no tap `{LENS_TAP}`; send the trace with no lens"),
                ));
            }
            Some(other) => {
                return Err(invalid(
                    "trace.lens",
                    format!("nothing ran: trace.lens is {other:?}; the lab has the lens {LOGIT_LENS:?} only"),
                ));
            }
        };
        let layers: Vec<u32> = match &members.layers {
            None => (0..facts.layers).collect(),
            Some(list) => {
                if list.is_empty() {
                    return Err(invalid(
                        "trace.layers",
                        "nothing ran: trace.layers is empty; name a layer, or send \"all\"".to_owned(),
                    ));
                }
                for (i, layer) in list.iter().enumerate() {
                    if *layer >= facts.layers {
                        return Err(invalid(
                            "trace.layers",
                            format!(
                                "nothing ran: trace.layers names the layer {layer}, and {model} has {} layers",
                                facts.layers
                            ),
                        ));
                    }
                    if list[..i].contains(layer) {
                        return Err(invalid(
                            "trace.layers",
                            format!("nothing ran: trace.layers names the layer {layer} twice; name each layer once"),
                        ));
                    }
                }
                list.clone()
            }
        };
        let top_k = members.top_k.unwrap_or(TOP_K as u64);
        if !(1..=MAX_TOP_K as u64).contains(&top_k) {
            return Err(invalid(
                "trace.top_k",
                format!("nothing ran: trace.top_k is {top_k}; send a count from 1 to {MAX_TOP_K}"),
            ));
        }
        let weight_hash = self.identity().weight_hash;
        let mut vectors = Vec::with_capacity(members.vectors.len());
        for (i, text) in members.vectors.iter().enumerate() {
            let param = format!("trace.vectors[{i}]");
            if members.vectors[..i].contains(text) {
                return Err(invalid(
                    &param,
                    format!("nothing ran: {param} names {text} again; name each vector once"),
                ));
            }
            let quoted = gogi_proto::ids::ContentId::parse(text.clone()).map_err(|why| {
                invalid(
                    &param,
                    format!("nothing ran: {param} ({text:?}) is not a content id: {why}"),
                )
            })?;
            let id =
                nagaya_store::ContentId::from_hex(quoted.as_str().trim_start_matches("blake3:")).map_err(|error| {
                    ServiceError::internal(format!("a checked content id {text} does not parse: {error}"))
                })?;
            let vector = control::held(&param, text, &id, &weight_hash, facts.layers, facts.width, &mut |id| {
                self.object(id)
            })?;
            let direction = Direction::new(text, &vector.layers, facts.layers as usize, facts.width)
                .map_err(|error| ServiceError::internal(format!("a vector that fits the model: {error}")))?;
            vectors.push(direction);
        }
        let vocab = self.shared.vocab;
        if let Some(id) = facts.ends.iter().find(|id| **id as usize >= vocab) {
            return Err(ServiceError::internal(format!(
                "the model folder of {model} names the end token {id}, past the logit row of {vocab}"
            )));
        }
        Ok(Plan {
            model: model.clone(),
            weight_hash,
            layers,
            model_layers: facts.layers as usize,
            width: facts.width,
            vocab,
            top_k: top_k as usize,
            ends: facts.ends.clone(),
            vectors,
            lens,
            rows: members.rows,
        })
    }

    /// Sample a reply after a context and record its trace (`docs/lab.md`, "`trace`"). `put` holds the bytes of the
    /// object and gives its id. The response is the response of the same generation with no trace.
    pub fn trace(
        &self,
        request: &GenerationRequest,
        plan: Plan,
        put: &dyn Fn(&[u8]) -> Result<gogi_proto::ids::ContentId, ServiceError>,
    ) -> Result<(GenerationDone, gogi_proto::ids::ContentId), ServiceError> {
        let (done, trace) = self.generate_traced(request, None, Some(Tracing::new(Trace::new(plan))))?;
        let trace = trace.ok_or_else(|| ServiceError::internal("a traced generation gave no trace"))?;
        if trace.steps() as u64 != done.usage.completion {
            return Err(ServiceError::internal(format!(
                "the trace has {} steps, and the reply has {} tokens",
                trace.steps(),
                done.usage.completion
            )));
        }
        let id = put(&trace.to_bytes())?;
        Ok((done, id))
    }
}
