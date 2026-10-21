//! The text routes under `/nagaya/v1` (spec 11.4a to 11.4d): `render`, `tokenize` and `detokenize`.
//!
//! The routes use the family library of a served model, and nothing else. They use no engine, no ledger charge and no
//! job. Each call runs on a call thread, as the council routes do (spec 3.2, 11.7). Each body has a required `model`,
//! and a member that the route does not know is a 400 that names it (spec 11.4b).

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::Extension;
use axum::extract::State;
use axum::response::Response;
use axum::routing::post;
use gogi_proto::context::Context;
use gogi_proto::decision::{ControlTextHit, Signals};
use nagaya_council::family::FamilyError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::{Failure, parse};
use super::routes::{answer, bytes};
use super::{Council, RequestId};
use crate::council::error::ServiceError;

/// The prefix of the routes of the nagaya, as `identity.extensions` lists it (spec 11.4f).
pub const PREFIX: &str = "/nagaya/v1";

type Shared = State<Arc<Council>>;
type Id = Extension<RequestId>;

/// The routes of the text family of extensions. The caller adds the fallbacks.
pub fn routes() -> Router<Arc<Council>> {
    Router::new()
        .route("/nagaya/v1/render", post(render))
        .route("/nagaya/v1/tokenize", post(tokenize))
        .route("/nagaya/v1/detokenize", post(detokenize))
}

/// The error of a family call. A client error with no `param` gets `param`, the member of the request at fault
/// (spec 11.4b). A family error that names a member keeps it.
fn at(param: &'static str) -> impl Fn(FamilyError) -> ServiceError {
    move |error| match error {
        FamilyError::Client(mut invalid) => {
            invalid.param.get_or_insert_with(|| param.to_owned());
            ServiceError::Invalid(invalid)
        }
        internal @ FamilyError::Internal(_) => internal.into(),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderRequest {
    model: String,
    context: Context,
    generation_prompt: Option<bool>,
}

#[derive(Serialize)]
struct Boundary {
    message: usize,
    tokens: usize,
}

#[derive(Serialize)]
struct Rendered {
    model: String,
    template: String,
    text: String,
    tokens: Vec<u32>,
    n_tokens: usize,
    end: usize,
    boundaries: Vec<Boundary>,
    signals: Signals,
}

async fn render(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: RenderRequest = parse(&bytes(body).await?, "render")?;
        let served = council.route(&request.model)?;
        council
            .call(move || {
                let family = served.service.family();
                let prompt = request.generation_prompt.unwrap_or(false);
                let rendering = family.render(&request.context, prompt).map_err(at("context"))?;
                // The tokens of a rendering come from valid text, so they make whole text. A failure is a defect.
                let text = family.detokenize(&rendering.tokens).map_err(|error| {
                    ServiceError::internal(format!("the tokens of a rendering do not make text: {error}"))
                })?;
                Ok(Rendered {
                    model: served.service.model().to_owned(),
                    template: family.template_id().to_owned(),
                    text,
                    n_tokens: rendering.tokens.len(),
                    tokens: rendering.tokens,
                    end: rendering.end,
                    boundaries: rendering
                        .boundaries
                        .iter()
                        .map(|boundary| Boundary {
                            message: boundary.message,
                            tokens: boundary.tokens,
                        })
                        .collect(),
                    signals: Signals {
                        control_text: rendering
                            .control_text
                            .into_iter()
                            .map(|hit| ControlTextHit {
                                r#where: hit.place,
                                token: hit.token,
                            })
                            .collect(),
                    },
                })
            })
            .await
    }
    .await;
    answer(&id, result)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeRequest {
    model: String,
    text: String,
    special: Option<bool>,
    pieces: Option<bool>,
}

#[derive(Serialize)]
struct Tokenized {
    model: String,
    tokens: Vec<u32>,
    n_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pieces: Option<Vec<String>>,
}

async fn tokenize(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: TokenizeRequest = parse(&bytes(body).await?, "tokenize")?;
        let served = council.route(&request.model)?;
        council
            .call(move || {
                let family = served.service.family();
                let tokens = match request.special.unwrap_or(false) {
                    true => family.tokenize_special(&request.text),
                    false => family.tokenize(&request.text),
                }
                .map_err(at("text"))?;
                let pieces = match request.pieces.unwrap_or(false) {
                    true => Some(family.pieces(&tokens).map_err(at("text"))?),
                    false => None,
                };
                Ok(Tokenized {
                    model: served.service.model().to_owned(),
                    n_tokens: tokens.len(),
                    tokens,
                    pieces,
                })
            })
            .await
    }
    .await;
    answer(&id, result)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DetokenizeRequest {
    model: String,
    /// An array of unsigned 32-bit integers. The route checks each one, so that a bad id names `tokens`.
    tokens: Value,
}

#[derive(Serialize)]
struct Detokenized {
    model: String,
    text: String,
}

/// The ids of a `tokens` member. Each one must be an unsigned 32-bit integer.
fn token_ids(value: &Value) -> Result<Vec<u32>, Failure> {
    let refuse = |message: String| Failure::from(ServiceError::Invalid(gogi_proto::Invalid::param("tokens", message)));
    let Some(items) = value.as_array() else {
        return Err(refuse(
            "nothing ran: tokens is not an array; send an array of unsigned 32-bit integers".to_owned(),
        ));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            item.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| {
                refuse(format!(
                    "nothing ran: tokens[{i}] is {item}, which is not an unsigned 32-bit integer"
                ))
            })
        })
        .collect()
}

async fn detokenize(State(council): Shared, Extension(id): Id, body: Body) -> Response {
    let result = async {
        let request: DetokenizeRequest = parse(&bytes(body).await?, "detokenize")?;
        let served = council.route(&request.model)?;
        let tokens = token_ids(&request.tokens)?;
        council
            .call(move || {
                let text = served.service.family().detokenize(&tokens).map_err(at("tokens"))?;
                Ok(Detokenized {
                    model: served.service.model().to_owned(),
                    text,
                })
            })
            .await
    }
    .await;
    answer(&id, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_error_with_no_param_gets_the_member_at_fault_and_one_with_a_param_keeps_it() {
        let ServiceError::Invalid(invalid) = at("text")(FamilyError::client("bad")) else {
            panic!("want a client error")
        };
        assert_eq!(invalid.param.as_deref(), Some("text"));
        let ServiceError::Invalid(invalid) = at("text")(FamilyError::client_param("continue", "bad")) else {
            panic!("want a client error")
        };
        assert_eq!(invalid.param.as_deref(), Some("continue"));
        assert!(matches!(
            at("text")(FamilyError::internal("x")),
            ServiceError::Internal(_)
        ));
    }
}
