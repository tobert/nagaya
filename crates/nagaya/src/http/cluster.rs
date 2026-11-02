//! The cluster route (spec 11.4a, "The cluster route"): `GET /nagaya/v1/cluster` gives the gossip view of this node.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Extension, State};
use axum::http::{StatusCode, Uri};
use axum::response::Response;
use axum::routing::get;
use gogi_proto::error::ErrorType;
use serde_json::{Value, json};

use super::errors::Failure;
use super::routes::json;
use super::{Council, RequestId};
use crate::gossip::{ClusterView, Key, NodeView};

/// The cluster route. The caller adds the fallbacks.
pub(super) fn routes() -> Router<Arc<Council>> {
    Router::new().route("/nagaya/v1/cluster", get(cluster))
}

async fn cluster(State(council): State<Arc<Council>>, Extension(id): Extension<RequestId>, uri: Uri) -> Response {
    if let Some(query) = uri.query().filter(|query| !query.is_empty()) {
        let name = query.split(['=', '&']).next().unwrap_or(query);
        let mut failure = Failure::new(
            StatusCode::BAD_REQUEST,
            ErrorType::InvalidRequest,
            format!("the cluster route takes no query, and the request has `{name}`; send it with no query"),
        );
        failure.body.error.param = Some(name.to_owned());
        return failure.respond(&id);
    }
    let Some(cluster) = council.cluster() else {
        return Failure::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            "this server has no cluster; the cluster route needs a process",
        )
        .respond(&id);
    };
    json(StatusCode::OK, &view_json(cluster.own(), cluster.state().view()))
}

/// The JSON of a gossip view (spec 11.4a). A key that a node does not write gives `null`, or an empty list for a list.
pub(crate) fn view_json(own: &str, view: &ClusterView) -> Value {
    let mut nodes: Vec<&NodeView> = view.live.iter().collect();
    nodes.sort_by(|left, right| left.name.cmp(&right.name));
    json!({
        "node": own,
        "nodes": nodes.into_iter().map(node_json).collect::<Vec<_>>(),
        "dead": view.dead,
    })
}

/// The JSON of one live node.
fn node_json(node: &NodeView) -> Value {
    let value = |key: &Key| node.value(key).unwrap_or(Value::Null);
    let engines: Vec<Value> = node
        .keys
        .keys()
        .filter_map(|key| match Key::parse(key) {
            Some(Key::Engine(name)) => Some(name),
            _ => None,
        })
        .map(|name| {
            let engine = value(&Key::Engine(name.clone()));
            let models: Vec<Value> = engine["served"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|entry| entry["model"].clone())
                .collect();
            json!({"name": name, "state": engine["state"], "models": models})
        })
        .collect();
    let model_conflicts = value(&Key::ModelConflict)["models"].clone();
    json!({
        "name": node.name,
        "id": node.id,
        "self": node.is_self,
        "gossip": node.addr.to_string(),
        "http": value(&Key::Server)["http"],
        "engines": engines,
        "conflicts": {
            "name": value(&Key::NameConflict),
            "models": if model_conflicts.is_array() { model_conflicts } else { json!([]) },
        },
        "refused": value(&Key::ProfileRefused),
    })
}
