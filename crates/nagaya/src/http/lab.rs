//! The lab routes under `/nagaya/v1/lab/` (`docs/lab.md`, "Where the routes live"): `project`, a direction through the
//! head of a served model, and `trace`, a generation with its numbers.
//!
//! `project` reads the checkpoint of the model folder, not the engine. Thus it runs while the engine is unloaded. It
//! runs on a call thread, as the council routes do (spec 3.2). The rules are in `docs/lab.md`, "The rules of
//! `project`" and "The rules of `trace`".

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::response::Response;
use axum::routing::post;
use gogi_proto::Invalid;
use gogi_proto::generation::GenerationRequest;
use nagaya_council::trace::Trace;
use nagaya_mk_lfm3_bmg::lab::{Entry, Facts, Unembedding, ends_of, rank};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::{Failure, parse};
use super::routes::{answer, bytes};
use super::{Council, RequestId, Served};
use crate::council::control;
use crate::council::error::ServiceError;
use crate::council::trace::{ModelFacts, TraceMembers};

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The entries at each end of a layer when the request names no `top_k`.
pub const TOP_K: u64 = 20;

/// The most entries at each end of a layer.
pub const MAX_TOP_K: u64 = 256;

/// The one matrix of `through` now: the head, with the weight of the final norm.
pub const UNEMBED: &str = "unembed";

/// The largest trace object. The job keeps the numbers of a trace in host memory, outside the ledger, until the reply
/// ends (`docs/lab.md`, "The budget").
pub const MAX_TRACE_BYTES: u64 = 1 << 30;

/// The routes of the lab. The caller adds the fallbacks.
pub fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/nagaya/v1/lab/project", post(project))
        .route("/nagaya/v1/lab/trace", post(trace))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRequest {
    model: String,
    vector: Value,
    through: String,
    layers: Option<Vec<u32>>,
    top_k: Option<u64>,
}

#[derive(Serialize)]
struct Projected {
    model: String,
    through: String,
    layers: Vec<Layer>,
}

#[derive(Serialize)]
struct Layer {
    /// The layer of the vector, or `None` for a list with no `layers`.
    layer: Option<u32>,
    up: Vec<Scored>,
    down: Vec<Scored>,
    end: Vec<Ranked>,
}

#[derive(Serialize)]
struct Scored {
    id: u32,
    token: Option<String>,
    score: f32,
}

#[derive(Serialize)]
struct Ranked {
    id: u32,
    token: Option<String>,
    score: f32,
    rank: usize,
}

fn invalid(param: &str, message: String) -> ServiceError {
    ServiceError::Invalid(Invalid::param(param, message))
}

async fn project(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: ProjectRequest = parse(&bytes(body).await?, "project")?;
        let (served, dir) = council.lab_model(&request.model)?;
        council.call(move || run(&served, &dir, request)).await
    }
    .await;
    answer(&id, result)
}

/// A direction of a request, with the layer that the response names.
type Direction = (Option<u32>, Vec<f32>);

/// The directions of a request, each with the layer that the response names.
fn directions(served: &Served, lab: &Unembedding, request: &ProjectRequest) -> Result<Vec<Direction>, ServiceError> {
    let width = lab.width();
    match &request.vector {
        Value::String(text) => {
            let quoted = gogi_proto::ids::ContentId::parse(text.clone()).map_err(|why| {
                invalid(
                    "vector",
                    format!("nothing ran: vector {text:?} is not a content id: {why}"),
                )
            })?;
            let id =
                nagaya_store::ContentId::from_hex(quoted.as_str().trim_start_matches("blake3:")).map_err(|error| {
                    ServiceError::internal(format!("a checked content id {text} does not parse: {error}"))
                })?;
            let weight_hash = served.service.identity().weight_hash;
            let vector = control::held("vector", text, &id, &weight_hash, lab.layers(), width, &mut |id| {
                served.service.object(id)
            })?;
            let chosen: Vec<u32> = match &request.layers {
                None => vector
                    .layers
                    .iter()
                    .filter(|(_, row)| row.iter().any(|x| *x != 0.0))
                    .map(|(layer, _)| *layer)
                    .collect(),
                Some(layers) => {
                    if layers.is_empty() {
                        return Err(invalid(
                            "layers",
                            "nothing ran: layers is empty; name a layer of the vector, or send no layers".to_owned(),
                        ));
                    }
                    for (i, layer) in layers.iter().enumerate() {
                        if layers[..i].contains(layer) {
                            return Err(invalid(
                                "layers",
                                format!("nothing ran: layers names the layer {layer} twice; name each layer once"),
                            ));
                        }
                        if !vector.layers.contains_key(layer) {
                            let held: Vec<u32> = vector.layers.keys().copied().collect();
                            return Err(invalid(
                                "layers",
                                format!(
                                    "nothing ran: the vector {text} holds no layer {layer}; it holds the layers \
                                     {held:?}"
                                ),
                            ));
                        }
                    }
                    layers.clone()
                }
            };
            Ok(chosen
                .into_iter()
                .map(|layer| (Some(layer), vector.layers[&layer].clone()))
                .collect())
        }
        Value::Array(items) => {
            if items.len() != width {
                return Err(invalid(
                    "vector",
                    format!(
                        "nothing ran: vector has {} numbers, and the residual of {} has the width {width}; send \
                         {width} numbers",
                        items.len(),
                        served.service.model()
                    ),
                ));
            }
            let mut row = Vec::with_capacity(width);
            for (i, item) in items.iter().enumerate() {
                let value = item.as_f64().map(|x| x as f32).filter(|x| x.is_finite());
                let Some(value) = value else {
                    return Err(invalid(
                        "vector",
                        format!("nothing ran: vector[{i}] is {item}, not a number that is finite as a 32-bit float"),
                    ));
                };
                row.push(value);
            }
            let layer = match request.layers.as_deref() {
                None => None,
                Some([layer]) if *layer < lab.layers() => Some(*layer),
                Some([layer]) => {
                    return Err(invalid(
                        "layers",
                        format!(
                            "nothing ran: layers names the layer {layer}, and {} has {} layers",
                            served.service.model(),
                            lab.layers()
                        ),
                    ));
                }
                Some(_) => {
                    return Err(invalid(
                        "layers",
                        "nothing ran: a list of numbers is one layer; name one layer in layers, or none".to_owned(),
                    ));
                }
            };
            Ok(vec![(layer, row)])
        }
        other => Err(invalid(
            "vector",
            format!("nothing ran: vector is {other}; send a control vector id or a list of numbers"),
        )),
    }
}

/// The work of `project` on a call thread.
fn run(served: &Served, dir: &Path, request: ProjectRequest) -> Result<Projected, ServiceError> {
    if request.through != UNEMBED {
        return Err(invalid(
            "through",
            format!(
                "nothing ran: through is {:?}; the lab projects through {UNEMBED:?} only",
                request.through
            ),
        ));
    }
    let top_k = request.top_k.unwrap_or(TOP_K);
    if !(1..=MAX_TOP_K).contains(&top_k) {
        return Err(invalid(
            "top_k",
            format!("nothing ran: top_k is {top_k}; send a count from 1 to {MAX_TOP_K}"),
        ));
    }
    let lab = Unembedding::open(dir).map_err(|error| {
        ServiceError::internal(format!(
            "the lab cannot read the checkpoint of {}: {error}",
            served.service.model()
        ))
    })?;
    let chosen = directions(served, &lab, &request)?;
    let rows: Vec<&[f32]> = chosen.iter().map(|(_, row)| row.as_slice()).collect();
    let scores = lab.scores(&rows);
    let family = served.service.family();
    let token = |id: u32| -> Result<Option<String>, ServiceError> {
        if id as usize >= family.vocab_size() {
            return Ok(None);
        }
        let mut pieces = family
            .pieces(&[id])
            .map_err(|error| ServiceError::internal(format!("the family gives no piece for the id {id}: {error}")))?;
        Ok(pieces.pop())
    };
    let scored = |list: Vec<Entry>| -> Result<Vec<Scored>, ServiceError> {
        list.into_iter()
            .map(|entry| {
                Ok(Scored {
                    id: entry.id,
                    token: token(entry.id)?,
                    score: entry.score,
                })
            })
            .collect()
    };
    let mut layers = Vec::with_capacity(chosen.len());
    for ((layer, _), scores) in chosen.iter().zip(&scores) {
        let (up, down) = ends_of(scores, top_k as usize);
        let end = lab
            .ends()
            .iter()
            .map(|id| {
                Ok(Ranked {
                    id: *id,
                    token: token(*id)?,
                    score: scores[*id as usize],
                    rank: rank(scores, *id),
                })
            })
            .collect::<Result<Vec<_>, ServiceError>>()?;
        layers.push(Layer {
            layer: *layer,
            up: scored(up)?,
            down: scored(down)?,
            end,
        });
    }
    Ok(Projected {
        model: served.service.model().to_owned(),
        through: request.through,
        layers,
    })
}

/// The members of `trace`, with the types of `docs/lab.md`. A member of another type, or another member, is a 400 that
/// names it.
fn members(value: Value) -> Result<TraceMembers, ServiceError> {
    let Value::Object(map) = value else {
        return Err(invalid(
            "trace",
            format!("nothing ran: trace is {value}; send an object of the members of a trace"),
        ));
    };
    let mut members = TraceMembers::default();
    for (key, value) in map {
        let param = format!("trace.{key}");
        let wrong = |what: &str| invalid(&param, format!("nothing ran: {param} is {value}; send {what}"));
        match key.as_str() {
            "layers" => {
                members.layers = match &value {
                    Value::String(all) if all == "all" => None,
                    Value::Array(items) => Some(
                        items
                            .iter()
                            .map(|item| item.as_u64().and_then(|n| u32::try_from(n).ok()))
                            .collect::<Option<Vec<u32>>>()
                            .ok_or_else(|| wrong("\"all\" or a list of layers"))?,
                    ),
                    _ => return Err(wrong("\"all\" or a list of layers")),
                }
            }
            "vectors" => {
                members.vectors = value
                    .as_array()
                    .and_then(|items| {
                        items
                            .iter()
                            .map(|item| item.as_str().map(str::to_owned))
                            .collect::<Option<Vec<String>>>()
                    })
                    .ok_or_else(|| wrong("a list of control vector ids"))?
            }
            "top_k" => members.top_k = Some(value.as_u64().ok_or_else(|| wrong("a count"))?),
            "lens" => {
                members.lens = match &value {
                    Value::Null => None,
                    Value::String(lens) => Some(lens.clone()),
                    _ => return Err(wrong("the name of a lens")),
                }
            }
            "rows" => members.rows = value.as_bool().ok_or_else(|| wrong("true or false"))?,
            _ => {
                return Err(invalid(
                    &param,
                    format!(
                        "nothing ran: a trace has no member {key}; its members are layers, vectors, top_k, lens and \
                         rows"
                    ),
                ));
            }
        }
    }
    Ok(members)
}

async fn trace(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let mut value: Value = parse(&bytes(body).await?, "trace")?;
        let trace = value
            .as_object_mut()
            .and_then(|map| map.remove("trace"))
            .ok_or_else(|| {
                Failure::from(invalid(
                    "trace",
                    "nothing ran: a trace request is a generation request with the member trace; send trace".to_owned(),
                ))
            })?;
        let body = serde_json::to_vec(&value).expect("a JSON value encodes");
        let request: GenerationRequest = parse(&body, "trace")?;
        let members = members(trace)?;
        let (served, dir) = council.lab_model(&request.model)?;
        if request.stream == Some(true) {
            return Err(Failure::from(invalid(
                "stream",
                "nothing ran: a trace does not stream, because its object is complete only at the end of the reply; \
                 send stream: false"
                    .to_owned(),
            )));
        }
        let store = council.objects().ok_or_else(|| {
            Failure::from(invalid(
                "trace",
                "nothing ran: the server holds no objects, so it cannot keep a trace; its profile names no `objects` \
                 budget in `[store]` (spec 6.15)"
                    .to_owned(),
            ))
        })?;
        // The checks of the trace come before the status of the model: they read no engine.
        let checks = Arc::clone(&served);
        let max_tokens = request.max_tokens as usize;
        let plan = council
            .call(move || {
                let facts = Facts::open(&dir).map_err(|error| {
                    ServiceError::internal(format!(
                        "the lab cannot read the model folder of {}: {error}",
                        checks.service.model()
                    ))
                })?;
                let facts = ModelFacts {
                    layers: facts.layers,
                    width: facts.width,
                    ends: facts.ends,
                };
                let plan = checks.service.trace_plan(&members, &facts)?;
                let limit = store
                    .lock()
                    .expect("the object store lock is poisoned")
                    .object_bytes()
                    .min(MAX_TRACE_BYTES);
                let needed = Trace::object_bytes(&plan, max_tokens);
                if needed > limit {
                    let param = if plan.rows { "trace.rows" } else { "max_tokens" };
                    return Err(ServiceError::TooLarge {
                        param: param.to_owned(),
                        message: format!(
                            "nothing ran: a trace of {max_tokens} steps is {needed} bytes, and the largest trace here \
                             is {limit} bytes; send fewer max_tokens or layers, or no rows"
                        ),
                    });
                }
                Ok(plan)
            })
            .await?;
        let served = council.route_contexts(&request.model, &[&request.context], false)?;
        let inner = Arc::clone(&council);
        let (done, trace) = council
            .call(move || {
                let put = |bytes: &[u8]| {
                    let id = super::voice::seal(&inner, bytes, TRACE_MEDIA_TYPE)?;
                    Ok(gogi_proto::ids::ContentId::parse(format!("blake3:{}", id.hex()))
                        .expect("a content id makes a valid gogi id"))
                };
                served.service.trace(&request, plan, &put)
            })
            .await?;
        let mut out = serde_json::to_value(&done).expect("a generation response encodes");
        out["trace"] = Value::String(trace.as_str().to_owned());
        Ok(out)
    }
    .await;
    answer(&id, result)
}

/// The media type of a trace object: canonical safetensors (D10).
pub const TRACE_MEDIA_TYPE: &str = "application/x-safetensors";

impl Council {
    /// The served model of a lab route and its model folder. A model that this node does not serve, or one whose
    /// engine kind the lab does not read, is a 400 that names `model`. Its status does not matter.
    fn lab_model(&self, name: &str) -> Result<(Arc<Served>, PathBuf), Failure> {
        let served = self
            .models
            .iter()
            .find(|served| served.service.model() == name || served.aliases.iter().any(|alias| alias == name))
            .ok_or_else(|| {
                Failure::from(invalid(
                    "model",
                    format!("nothing ran: model {name} is not served here; identity lists the served models"),
                ))
            })?;
        let dir = self.lab.get(&served.service.config().engine).ok_or_else(|| {
            Failure::from(invalid(
                "model",
                format!(
                    "nothing ran: the lab reads no weights of {name}; it projects models of the engine kind {}",
                    crate::lfm3::KIND
                ),
            ))
        })?;
        Ok((Arc::clone(served), dir.clone()))
    }
}
