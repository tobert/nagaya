//! `POST /nagaya/v1/lab/compose`: a control vector from parts (`docs/lab.md`, "compose"). A part adds a vector or a
//! direction toward words at chosen layers, or takes a direction out. The result is a plain control vector object, so
//! every client can use it.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::response::Response;
use gogi_proto::control::{ControlVector, MEDIA_TYPE};
use nagaya_mk_lfm3_bmg::lab::{JLens, Unembedding};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::parse;
use super::lab::{UNEMBED, invalid};
use super::routes::{answer, bytes};
use super::{Council, RequestId, Served};
use crate::council::control;
use crate::council::error::ServiceError;
use crate::council::trace::JLENS as JLENS_THROUGH;

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComposeRequest {
    model: String,
    parts: Vec<Value>,
}

#[derive(Serialize)]
struct Composed {
    model: String,
    vector: String,
    layers: Vec<u32>,
    /// The norm of each layer of the result, by its layer number in decimal.
    norms: BTreeMap<String, f64>,
    /// The ids of each text of each `words` part, in the order of the parts.
    words: Vec<Word>,
}

#[derive(Serialize)]
struct Word {
    text: String,
    ids: Vec<u32>,
}

/// One part after its checks: its directions by layer, and its weight, or `None` for a remove.
struct Part {
    layers: BTreeMap<u32, Vec<f32>>,
    weight: Option<f64>,
}

pub(super) async fn compose(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: ComposeRequest = parse(&bytes(body).await?, "compose")?;
        let (served, dir) = council.lab_model(&request.model)?;
        let inner = Arc::clone(&council);
        let lens_of = move |served: &Served, dir: &Path, param: &str| inner.lab_lens(served, dir, param);
        let put_council = Arc::clone(&council);
        council
            .call(move || {
                let put = |bytes: &[u8]| super::voice::seal(&put_council, bytes, MEDIA_TYPE);
                run(&served, &dir, request, &lens_of, &put)
            })
            .await
    }
    .await;
    answer(&id, result)
}

type LensOf<'a> = dyn Fn(&Served, &Path, &str) -> Result<Arc<JLens>, ServiceError> + 'a;
type Put<'a> = dyn Fn(&[u8]) -> Result<nagaya_store::ContentId, ServiceError> + 'a;

fn run(
    served: &Served,
    dir: &Path,
    request: ComposeRequest,
    lens_of: &LensOf<'_>,
    put: &Put<'_>,
) -> Result<Composed, ServiceError> {
    if request.parts.is_empty() {
        return Err(invalid(
            "parts",
            "nothing ran: parts is empty; send a part or more".to_owned(),
        ));
    }
    if super::lab::is_inkling2(dir) {
        return Err(invalid(
            "model",
            format!(
                "nothing ran: compose reads the checkpoints of the kind {}; {} is Inkling 2",
                crate::lfm3::KIND,
                served.service.model()
            ),
        ));
    }
    let lab = Unembedding::open(dir).map_err(|error| {
        ServiceError::internal(format!(
            "the lab cannot read the checkpoint of {}: {error}",
            served.service.model()
        ))
    })?;
    let mut parts = Vec::with_capacity(request.parts.len());
    let mut words = Vec::new();
    for (i, value) in request.parts.iter().enumerate() {
        parts.push(part(served, dir, &lab, i, value, lens_of, &mut words)?);
    }
    let adds: Vec<(&BTreeMap<u32, Vec<f32>>, f64)> =
        parts.iter().filter_map(|p| p.weight.map(|w| (&p.layers, w))).collect();
    let removes: Vec<&BTreeMap<u32, Vec<f32>>> =
        parts.iter().filter(|p| p.weight.is_none()).map(|p| &p.layers).collect();
    let layers = nagaya_council::control::compose(&adds, &removes);
    if layers.is_empty() {
        return Err(invalid(
            "parts",
            "nothing ran: the parts add nothing at any layer; add a vector or words at a layer".to_owned(),
        ));
    }
    let norms = layers
        .iter()
        .map(|(l, row)| {
            let norm = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
            (l.to_string(), norm)
        })
        .collect();
    let vector = ControlVector {
        model: served.service.model().to_owned(),
        weight_hash: served.service.identity().weight_hash,
        layers,
    };
    let id = put(&vector.to_bytes())?;
    Ok(Composed {
        model: served.service.model().to_owned(),
        vector: format!("blake3:{}", id.hex()),
        layers: vector.layers.keys().copied().collect(),
        norms,
        words,
    })
}

/// Check one part and give its directions. An error names the part, such as `parts[1].weight`.
#[allow(clippy::too_many_arguments)]
fn part(
    served: &Served,
    dir: &Path,
    lab: &Unembedding,
    i: usize,
    value: &Value,
    lens_of: &LensOf<'_>,
    words: &mut Vec<Word>,
) -> Result<Part, ServiceError> {
    let at = format!("parts[{i}]");
    let Value::Object(m) = value else {
        return Err(invalid(&at, format!("nothing ran: {at} is an object, not {value}")));
    };
    let known = ["vector", "words", "through", "weight", "layers", "remove"];
    if let Some(k) = m.keys().find(|k| !known.contains(&k.as_str())) {
        let param = format!("{at}.{k}");
        return Err(invalid(
            &param,
            format!(
                "nothing ran: {at} has the unknown member {k}; a part has {}",
                known.join(", ")
            ),
        ));
    }
    let remove = match m.get("remove") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => {
            return Err(invalid(
                &format!("{at}.remove"),
                format!("nothing ran: {at}.remove is true or false, not {other}"),
            ));
        }
    };
    let weight = match (m.get("weight"), remove) {
        (Some(_), true) => {
            return Err(invalid(
                &format!("{at}.weight"),
                format!("nothing ran: {at} removes a direction, so it has no weight; leave weight out"),
            ));
        }
        (None, true) => None,
        (None, false) => Some(1.0),
        (Some(w), false) => match w.as_f64().filter(|w| w.is_finite()) {
            Some(w) => Some(w),
            None => {
                return Err(invalid(
                    &format!("{at}.weight"),
                    format!("nothing ran: {at}.weight is a finite number, not {w}"),
                ));
            }
        },
    };
    let model_layers = lab.layers();
    let layers: Option<Vec<u32>> = match m.get("layers") {
        None => None,
        Some(Value::Array(items)) if !items.is_empty() => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let layer = item.as_u64().filter(|l| *l < u64::from(model_layers)).map(|l| l as u32);
                let Some(layer) = layer.filter(|l| !out.contains(l)) else {
                    return Err(invalid(
                        &format!("{at}.layers"),
                        format!(
                            "nothing ran: {at}.layers has {item}; name each layer of 0 to {} once",
                            model_layers - 1
                        ),
                    ));
                };
                out.push(layer);
            }
            Some(out)
        }
        Some(other) => {
            return Err(invalid(
                &format!("{at}.layers"),
                format!("nothing ran: {at}.layers is a list of layers, not {other}"),
            ));
        }
    };
    let directions = match (m.get("vector"), m.get("words")) {
        (Some(Value::String(text)), None) => {
            if m.contains_key("through") {
                return Err(invalid(
                    &format!("{at}.through"),
                    format!("nothing ran: {at} adds a vector, which has no through"),
                ));
            }
            let param = format!("{at}.vector");
            let quoted = gogi_proto::ids::ContentId::parse(text.clone()).map_err(|why| {
                invalid(
                    &param,
                    format!("nothing ran: {param} {text:?} is not a content id: {why}"),
                )
            })?;
            let id =
                nagaya_store::ContentId::from_hex(quoted.as_str().trim_start_matches("blake3:")).map_err(|error| {
                    ServiceError::internal(format!("a checked content id {text} does not parse: {error}"))
                })?;
            let weight_hash = served.service.identity().weight_hash;
            let vector = control::held(&param, text, &id, &weight_hash, model_layers, lab.width(), &mut |id| {
                served.service.object(id)
            })?;
            vector
                .layers
                .into_iter()
                .filter(|(l, row)| row.iter().any(|x| *x != 0.0) && layers.as_ref().is_none_or(|ls| ls.contains(l)))
                .collect()
        }
        (None, Some(Value::Array(texts))) if !texts.is_empty() => {
            let Some(layers) = &layers else {
                return Err(invalid(
                    &format!("{at}.layers"),
                    format!("nothing ran: {at} adds words, so it names its layers"),
                ));
            };
            let through = m.get("through").and_then(Value::as_str);
            let lens = match through {
                Some(UNEMBED) => None,
                Some(JLENS_THROUGH) => Some(lens_of(served, dir, &format!("{at}.through"))?),
                _ => {
                    return Err(invalid(
                        &format!("{at}.through"),
                        format!("nothing ran: {at}.through is {UNEMBED:?} or {JLENS_THROUGH:?} for words"),
                    ));
                }
            };
            let family = served.service.family();
            let mut sum = vec![0f64; lab.width()];
            for (j, text) in texts.iter().enumerate() {
                let param = format!("{at}.words[{j}]");
                let Some(text) = text.as_str().filter(|t| !t.is_empty()) else {
                    return Err(invalid(&param, format!("nothing ran: {param} is a text, not {text}")));
                };
                let ids = family
                    .tokenize(text)
                    .map_err(|error| invalid(&param, format!("nothing ran: {param} does not tokenize: {error}")))?;
                if ids.is_empty() {
                    return Err(invalid(&param, format!("nothing ran: {param} {text:?} gives no token")));
                }
                for id in &ids {
                    for (s, x) in sum.iter_mut().zip(lab.head_row(*id)) {
                        *s += f64::from(x);
                    }
                }
                words.push(Word {
                    text: text.to_owned(),
                    ids,
                });
            }
            // The direction of the words in the space of the final norm: the weight of the norm times the rows.
            let base: Vec<f32> = sum
                .iter()
                .zip(lab.norm())
                .map(|(s, g)| (s * f64::from(*g)) as f32)
                .collect();
            let mut out = BTreeMap::new();
            for l in layers {
                let row = match &lens {
                    Some(lens) => lens.transport_back(*l, &base),
                    None => base.clone(),
                };
                let norm = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
                if norm == 0.0 {
                    return Err(invalid(
                        &format!("{at}.words"),
                        format!("nothing ran: the words of {at} give no direction at layer {l}"),
                    ));
                }
                out.insert(*l, row.iter().map(|x| (f64::from(*x) / norm) as f32).collect());
            }
            out
        }
        _ => {
            return Err(invalid(
                &at,
                format!("nothing ran: {at} has a vector or a list of words, and not both"),
            ));
        }
    };
    Ok(Part {
        layers: directions,
        weight,
    })
}
