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
use nagaya_council::ensemble::{Combine, EnsembleError, combine};
use nagaya_council::latent::LatentMap;
use nagaya_council::trace::{Lens, Readout, Trace};
use nagaya_mk_lfm3_bmg::lab::{Entry, Facts, JLens, LensError, Unembedding, ends_of, rank};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::{Failure, parse};
use super::routes::{answer, bytes};
use super::{Council, RequestId, Served};
use crate::council::control;
use crate::council::error::ServiceError;
use crate::council::generations::{Latent, Soft};
use crate::council::trace::{JLENS, ModelFacts, TraceMembers};
use nagaya_council::soft::{DEFAULT_TOP, MAX_TOP, SoftMap, SoftPlan, Span};

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The entries at each end of a layer when the request names no `top_k`.
pub const TOP_K: u64 = 20;

/// The most entries at each end of a layer.
pub const MAX_TOP_K: u64 = 256;

/// The matrix of `through` that is the head, with the weight of the final norm.
pub const UNEMBED: &str = "unembed";

/// The most threads that move the rows of one step through a J-Lens.
const LENS_THREADS: usize = 8;

/// The J-Lens file of one engine, and the lens after its first read.
pub struct LensSlot {
    path: PathBuf,
    read: std::sync::Mutex<Option<Arc<JLens>>>,
}

impl LensSlot {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            read: std::sync::Mutex::new(None),
        }
    }
}

/// The readout of a J-Lens in a trace: `J_l h` for each row, then the final norm with its division, and the head.
struct JLensReadout {
    lens: Arc<JLens>,
    head: Unembedding,
}

impl Readout for JLensReadout {
    fn logits(&self, rows: &[(u32, &[f32])]) -> Vec<Vec<f32>> {
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, LENS_THREADS);
        let per = rows.len().div_ceil(threads).max(1);
        let moved: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = rows
                .chunks(per)
                .map(|part| {
                    scope.spawn(move || part.iter().map(|(l, h)| self.lens.transport(*l, h)).collect::<Vec<_>>())
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("a thread of the J-Lens panicked"))
                .collect()
        });
        let refs: Vec<&[f32]> = moved.iter().map(Vec::as_slice).collect();
        self.head.readout(&refs)
    }
}

/// The largest trace object. The job keeps the numbers of a trace in host memory, outside the ledger, until the reply
/// ends (`docs/lab.md`, "The budget").
pub const MAX_TRACE_BYTES: u64 = 1 << 30;

/// The routes of the lab. The caller adds the fallbacks.
pub fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/nagaya/v1/lab/project", post(project))
        .route("/nagaya/v1/lab/trace", post(trace))
        .route("/nagaya/v1/lab/ensemble", post(ensemble))
        .route("/nagaya/v1/lab/compose", post(super::compose::compose))
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

pub(super) fn invalid(param: &str, message: String) -> ServiceError {
    ServiceError::Invalid(Invalid::param(param, message))
}

async fn project(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: ProjectRequest = parse(&bytes(body).await?, "project")?;
        let (served, dir) = council.lab_model(&request.model)?;
        let inner = Arc::clone(&council);
        council
            .call(move || run(&served, &dir, request, &|| inner.lab_lens(&served, &dir, "through")))
            .await
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

/// The work of `project` on a call thread. `lens` reads the J-Lens of the model, for `through: "jlens"`, after the
/// checks of the request.
fn run(
    served: &Served,
    dir: &Path,
    request: ProjectRequest,
    lens: &dyn Fn() -> Result<Arc<JLens>, ServiceError>,
) -> Result<Projected, ServiceError> {
    if request.through != UNEMBED && request.through != JLENS {
        return Err(invalid(
            "through",
            format!(
                "nothing ran: through is {:?}; the lab projects through {UNEMBED:?} and {JLENS:?}",
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
    let mut chosen = directions(served, &lab, &request)?;
    if request.through == JLENS {
        let lens = lens()?;
        for (layer, row) in &mut chosen {
            let Some(layer) = layer else {
                return Err(invalid(
                    "layers",
                    "nothing ran: the J-Lens moves a row of one layer; name the layer of the list in layers".to_owned(),
                ));
            };
            *row = lens.transport(*layer, row);
        }
    }
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
        // A context that names a `parent` is whole before the decode, as in a generation (`docs/lab.md`, "Context
        // records in the lab routes").
        let body = super::routes::whole(&council, bytes(body).await?, false).await?;
        let mut value: Value = parse(&body, "trace")?;
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
        let lab = Arc::clone(&council);
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
                let jlens = match members.lens.as_deref() {
                    Some(JLENS) => {
                        let lens = lab.lab_lens(&checks, &dir, "trace.lens")?;
                        let head = Unembedding::open(&dir).map_err(|error| {
                            ServiceError::internal(format!(
                                "the lab cannot read the checkpoint of {}: {error}",
                                checks.service.model()
                            ))
                        })?;
                        Some(Lens::Cpu {
                            name: JLENS.to_owned(),
                            id: lens.id().to_owned(),
                            readout: Arc::new(JLensReadout { lens, head }),
                        })
                    }
                    _ => None,
                };
                let plan = checks.service.trace_plan(&members, &facts, jlens)?;
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

/// The members of an ensemble request (`docs/lab.md`, "`ensemble`").
const ENSEMBLE_MEMBERS: [&str; 10] = [
    "model",
    "contexts",
    "combine",
    "weights",
    "strength",
    "max_tokens",
    "sample",
    "latent",
    "soft",
    "force",
];

/// The member `latent` of an ensemble request (docs/lab.md, "Latent steps in a request"): the steps and the read. A
/// member of another type, or another member, is a 400 that names it.
fn latent_member(value: &Value) -> Result<(usize, bool, bool), Failure> {
    let refuse = |param: &str, message: String| Failure::from(invalid(param, format!("nothing ran: {message}")));
    let Value::Object(map) = value else {
        return Err(refuse(
            "latent",
            format!("latent is {value}; send an object with steps and read"),
        ));
    };
    if let Some(other) = map
        .keys()
        .find(|key| !["steps", "read", "stop"].contains(&key.as_str()))
    {
        return Err(refuse(
            &format!("latent.{other}"),
            format!("latent has no member {other}; its members are steps, read and stop"),
        ));
    }
    let most = nagaya_council::latent::MAX_STEPS;
    let steps = match map.get("steps") {
        None => 0,
        Some(steps) => steps.as_u64().filter(|steps| *steps <= most as u64).ok_or_else(|| {
            refuse(
                "latent.steps",
                format!("latent.steps is {steps}; send a whole number from 0 to {most}"),
            )
        })? as usize,
    };
    let read = match map.get("read") {
        None => false,
        Some(read) => read
            .as_bool()
            .ok_or_else(|| refuse("latent.read", format!("latent.read is {read}; send true or false")))?,
    };
    let stop = match map.get("stop") {
        None => false,
        Some(stop) => stop
            .as_bool()
            .ok_or_else(|| refuse("latent.stop", format!("latent.stop is {stop}; send true or false")))?,
    };
    Ok((steps, read, stop))
}

/// The member `soft` of an ensemble request (docs/lab.md, "Soft inputs in a request"): the map, its value, `top` and
/// `span`. A fault is a 400 that names the member.
fn soft_member(value: &Value) -> Result<SoftPlan, Failure> {
    let refuse = |param: &str, message: String| Failure::from(invalid(param, format!("nothing ran: {message}")));
    let Value::Object(map) = value else {
        return Err(refuse("soft", format!("soft is {value}; send an object with map")));
    };
    let (kind, own) = match map.get("map").and_then(Value::as_str) {
        Some("moi") => ("moi", "beta"),
        Some("gumbel") => ("gumbel", "tau"),
        _ => {
            return Err(refuse(
                "soft.map",
                format!(
                    "soft.map is {}; send \"moi\" or \"gumbel\"",
                    map.get("map").unwrap_or(&Value::Null)
                ),
            ));
        }
    };
    if let Some(other) = map
        .keys()
        .find(|key| !["map", own, "top", "span"].contains(&key.as_str()))
    {
        return Err(refuse(
            &format!("soft.{other}"),
            format!("soft with the map {kind} has no member {other}; its members are map, {own}, top and span"),
        ));
    }
    let number = |name: &str, default: f64, ok: fn(f64) -> bool, rule: &str| -> Result<f64, Failure> {
        match map.get(name) {
            None => Ok(default),
            Some(value) => value
                .as_f64()
                .filter(|x| x.is_finite() && ok(*x))
                .ok_or_else(|| refuse(&format!("soft.{name}"), format!("soft.{name} is {value}; send {rule}"))),
        }
    };
    let soft_map = match kind {
        "moi" => SoftMap::Moi {
            beta: number("beta", 1.0, |x| x >= 0.0, "a finite number of 0 or more")?,
        },
        _ => SoftMap::Gumbel {
            tau: number("tau", 0.5, |x| x > 0.0, "a finite number above 0")?,
        },
    };
    let top = match map.get("top") {
        None => DEFAULT_TOP,
        Some(top) => top
            .as_u64()
            .filter(|top| (1..=MAX_TOP as u64).contains(top))
            .ok_or_else(|| {
                refuse(
                    "soft.top",
                    format!("soft.top is {top}; send a whole number from 1 to {MAX_TOP}"),
                )
            })? as usize,
    };
    let span = match map.get("span").map(|span| (span, span.as_str())) {
        None | Some((_, Some("all"))) => Span::All,
        Some((_, Some("think"))) => Span::Think,
        Some((span, _)) => {
            return Err(refuse(
                "soft.span",
                format!("soft.span is {span}; send \"all\" or \"think\""),
            ));
        }
    };
    Ok(SoftPlan {
        map: soft_map,
        top,
        span,
    })
}

/// The echo of a soft plan in a response, with each default filled in.
fn soft_echo(plan: &SoftPlan) -> Value {
    let span = match plan.span {
        Span::All => "all",
        Span::Think => "think",
    };
    match plan.map {
        SoftMap::Moi { beta } => serde_json::json!({"map": "moi", "beta": beta, "top": plan.top, "span": span}),
        SoftMap::Gumbel { tau } => serde_json::json!({"map": "gumbel", "tau": tau, "top": plan.top, "span": span}),
    }
}

/// The map of a latent step of the model in `dir`, from its checkpoint: the final norm and the mean norm of the rows of
/// its table.
fn latent_map(served: &Served, dir: &Path) -> Result<LatentMap, ServiceError> {
    let head = Unembedding::open(dir).map_err(|error| {
        ServiceError::internal(format!(
            "the lab cannot read the checkpoint of {}: {error}",
            served.service.model()
        ))
    })?;
    Ok(LatentMap {
        layers: head.layers() as usize,
        width: head.width(),
        norm: head.norm().to_vec(),
        eps: head.eps(),
        scale: head.mean_row_norm(),
    })
}

/// The strength of `contrast` when the request sends none.
const CONTRAST_STRENGTH: f64 = 0.5;

/// One reply from joined forks (`docs/lab.md`, "`ensemble`: joined forks").
async fn ensemble(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let body = super::routes::whole(&council, bytes(body).await?, false).await?;
        let value: Value = parse(&body, "ensemble")?;
        let refuse = |param: &str, message: &str| Failure::from(invalid(param, format!("nothing ran: {message}")));
        let map = value
            .as_object()
            .ok_or_else(|| refuse("ensemble", "an ensemble request is a JSON object"))?;
        if let Some(other) = map.keys().find(|key| !ENSEMBLE_MEMBERS.contains(&key.as_str())) {
            return Err(refuse(
                other,
                &format!("an ensemble request has no member {other}; remove it"),
            ));
        }
        let model = map
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| refuse("model", "send the model"))?;
        let (lab_served, dir) = council.lab_model(model)?;
        let latent = map.get("latent").map(latent_member).transpose()?;
        let soft_plan = map.get("soft").map(soft_member).transpose()?;
        // A forced text gives the reply its tokens and their count (docs/lab.md, "The score of a text").
        let force = match map.get("force") {
            None => None,
            Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
            Some(other) => {
                return Err(refuse(
                    "force",
                    &format!("force is {other}; send the text of the reply as a string that is not empty"),
                ));
            }
        };
        if force.is_some() && map.contains_key("max_tokens") {
            return Err(refuse(
                "max_tokens",
                "a forced text gives the count of tokens of the reply; send no max_tokens",
            ));
        }
        if force.is_some() && soft_plan.is_some() {
            return Err(refuse(
                "force",
                "the score of a forced text reads plain inputs; send force or soft, not both",
            ));
        }
        let forced = match &force {
            Some(text) => Some(lab_served.service.tokens_of(text)?),
            None => None,
        };
        let contexts = map
            .get("contexts")
            .and_then(Value::as_array)
            .filter(|contexts| (1..=8).contains(&contexts.len()))
            .ok_or_else(|| refuse("contexts", "send 1 to 8 contexts"))?;
        let strength = match map.get("strength") {
            None => CONTRAST_STRENGTH,
            Some(value) => value
                .as_f64()
                .ok_or_else(|| refuse("strength", "the strength is a number"))?,
        };
        let rule = match map.get("combine").and_then(Value::as_str) {
            Some("logprob_sum") => Combine::LogprobSum,
            Some("prob_mean") => Combine::ProbMean,
            Some("contrast") => Combine::Contrast { strength },
            _ => return Err(refuse("combine", "combine is logprob_sum, prob_mean or contrast")),
        };
        let weights = match map.get("weights") {
            None => vec![1.0; contexts.len()],
            Some(Value::Array(values)) => values
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    value
                        .as_f64()
                        .ok_or_else(|| refuse(&format!("weights[{i}]"), "a weight is a number above 0"))
                })
                .collect::<Result<Vec<f64>, Failure>>()?,
            Some(_) => return Err(refuse("weights", "weights is a list of numbers, one for each context")),
        };
        // The rules of combine check the count, the weights and the strength, with one value for each context.
        let one = [0.0f32];
        if let Err(error) = combine(&vec![&one[..]; contexts.len()], &rule, &weights) {
            let param = match error {
                EnsembleError::WeightCount => "weights".to_owned(),
                EnsembleError::Weight(i) => format!("weights[{i}]"),
                EnsembleError::ContrastCount(_) => "combine".to_owned(),
                EnsembleError::Strength => "strength".to_owned(),
                other => return Err(Failure::from(ServiceError::internal(other.to_string()))),
            };
            return Err(refuse(&param, &error.to_string()));
        }
        if let Some(i) = contexts
            .iter()
            .position(|c| c.get("control") != contexts[0].get("control"))
        {
            return Err(refuse(
                &format!("contexts[{i}].control"),
                "the contexts of an ensemble share one control list, so their rows share a pass",
            ));
        }
        if let Some(i) = contexts.iter().position(|c| c.get("loop") != contexts[0].get("loop")) {
            return Err(refuse(
                &format!("contexts[{i}].loop"),
                "the contexts of an ensemble share one layer loop, so their rows share a pass",
            ));
        }
        let mut requests = Vec::with_capacity(contexts.len());
        for context in contexts {
            let mut request = serde_json::json!({"model": model, "context": context});
            for member in ["max_tokens", "sample"] {
                if let Some(value) = map.get(member) {
                    request[member] = value.clone();
                }
            }
            if let Some(tokens) = &forced {
                request["max_tokens"] = tokens.len().into();
            }
            let body = serde_json::to_vec(&request).expect("a JSON value encodes");
            requests.push(parse::<GenerationRequest>(&body, "ensemble")?);
        }
        let refs: Vec<&gogi_proto::Context> = requests.iter().map(|request| &request.context).collect();
        // The latent steps read the final norm and the table of the checkpoint. Their checks read no engine.
        let plan = match latent {
            Some((steps, read, stop)) if steps > 0 => {
                lab_served.service.takes_latent(1)?;
                let (latent_served, latent_dir) = (Arc::clone(&lab_served), dir.clone());
                let map = council.call(move || latent_map(&latent_served, &latent_dir)).await?;
                Some(Latent {
                    steps,
                    read,
                    stop,
                    map: Arc::new(map),
                })
            }
            _ => None,
        };
        // The soft inputs read the rows of the embedding table of the checkpoint.
        let soft = match soft_plan {
            Some(soft_plan) => {
                let (soft_served, soft_dir) = (Arc::clone(&lab_served), dir.clone());
                let table = council
                    .call(move || {
                        Unembedding::open(&soft_dir).map_err(|error| {
                            ServiceError::internal(format!(
                                "the lab cannot read the checkpoint of {}: {error}",
                                soft_served.service.model()
                            ))
                        })
                    })
                    .await?;
                let table = Arc::new(table);
                Some(Soft {
                    plan: soft_plan,
                    width: table.width(),
                    table: Arc::new(move |id| table.head_row(id)),
                })
            }
            None => None,
        };
        let served = council.route_contexts(model, &refs, true)?;
        let ensembled = council
            .call(move || served.service.ensemble(&requests, rule, weights, plan, soft, forced))
            .await?;
        let out = serde_json::to_value(&ensembled.done).expect("a generation response encodes");
        let mut body = serde_json::json!({"model": out["model"], "content": out["content"], "finish": out["finish"],
                                          "usage": out["usage"]});
        // The stats score the reply, and the reasoning shows what the model thought (docs/lab.md).
        for member in ["stats", "reasoning"] {
            if out.get(member).is_some_and(|value| !value.is_null()) {
                body[member] = out[member].clone();
            }
        }
        if let (Some(plan), Some((mixed, share))) = (soft_plan, ensembled.soft) {
            body["soft"] = soft_echo(&plan);
            body["soft"]["mixed"] = mixed.into();
            body["soft"]["mean_share"] = share.into();
        }
        if let Some((steps, read, stop)) = latent {
            body["latent"] = serde_json::json!({"steps": steps});
            if stop {
                body["latent"]["ran"] = ensembled.ran.unwrap_or(0).into();
            }
            if read {
                let tokens: Vec<Value> = ensembled
                    .read
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(id, text)| serde_json::json!({"id": id, "text": text}))
                    .collect();
                body["latent"]["read"] = Value::Array(tokens);
            }
        }
        Ok(body)
    }
    .await;
    answer(&id, result)
}

/// The media type of a trace object: canonical safetensors (D10).
pub const TRACE_MEDIA_TYPE: &str = "application/x-safetensors";

impl Council {
    /// The served model of a lab route and its model folder. A model that this node does not serve, or one whose
    /// engine kind the lab does not read, is a 400 that names `model`. Its status does not matter.
    pub(super) fn lab_model(&self, name: &str) -> Result<(Arc<Served>, PathBuf), Failure> {
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

    /// The J-Lens of a served model of the lab, from the file that its profile entry names. The first use reads the
    /// file, and later uses share it. No lens, or a lens of other weights, is a 400 that names `param`. A file that
    /// does not read is a 500 that names the path.
    pub(super) fn lab_lens(&self, served: &Served, dir: &Path, param: &str) -> Result<Arc<JLens>, ServiceError> {
        let model = served.service.model();
        let slot = self.lenses.get(&served.service.config().engine).ok_or_else(|| {
            invalid(
                param,
                format!(
                    "nothing ran: the profile names no J-Lens for {model}; fit one with tools/lfm3_jlens_fit.py, and \
                     name its file in [engines.lab] jlens"
                ),
            )
        })?;
        let mut read = slot.read.lock().expect("the lock of a lens is poisoned");
        if let Some(lens) = &*read {
            return Ok(Arc::clone(lens));
        }
        let facts = Facts::open(dir).map_err(|error| {
            ServiceError::internal(format!("the lab cannot read the model folder of {model}: {error}"))
        })?;
        let weight_hash = served.service.identity().weight_hash;
        let lens = JLens::read(&slot.path, &weight_hash, facts.layers, facts.width).map_err(|error| match error {
            LensError::OtherWeights(why) => invalid(param, format!("nothing ran: {why}")),
            LensError::Unread(why) | LensError::Malformed(why) => {
                ServiceError::internal(format!("the J-Lens of {model} does not read: {why}"))
            }
        })?;
        let lens = Arc::new(lens);
        *read = Some(Arc::clone(&lens));
        Ok(lens)
    }
}
