//! The cluster route (spec 11.4a, "The cluster route"): `GET /nagaya/v1/cluster` gives the gossip view.

mod peer_support;

use std::collections::BTreeMap;

use nagaya::cluster::ClusterState;
use nagaya::gossip::{ClusterView, NodeView};
use peer_support::{Has, Node};
use serde_json::{Value, json};

fn peer(name: &str, keys: &[(&str, Value)]) -> NodeView {
    let keys: BTreeMap<String, String> = keys
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.to_string()))
        .collect();
    NodeView {
        name: name.to_owned(),
        id: format!("{name}:7:127.0.0.1:9"),
        generation: 7,
        addr: "127.0.0.1:9".parse().unwrap(),
        is_self: false,
        keys,
    }
}

#[test]
fn the_route_gives_each_live_node_its_engines_conflicts_and_refusal() {
    let a = Node::start("host-a", Has::Objects);
    let served = |model: &str| json!({"model": model, "aliases": []});
    let b = peer(
        "host-b",
        &[
            (
                "server",
                json!({"http": "127.0.0.1:8091", "limits": {"case_bytes": 1, "choice_options": 2},
                              "capabilities": []}),
            ),
            (
                "engine:zeta",
                json!({"state": "unloaded", "served": [], "http": "127.0.0.1:8091"}),
            ),
            (
                "engine:gemma",
                json!({"state": "ready", "served": [served("gemma-4")], "http": "127.0.0.1:8091"}),
            ),
            (
                "name:conflict",
                json!({"own": "host-b:7:127.0.0.1:9", "peers": ["host-b:8:127.0.0.1:10"]}),
            ),
            (
                "model:conflict",
                json!({"models": [{"model": "toy", "kept_by": "host-a"}]}),
            ),
            (
                "profile:refused",
                json!({"version": 3, "reason": "busy", "shortfall": []}),
            ),
        ],
    );
    let c = peer("host-c", &[]);
    let mut own = a.as_peer();
    own.is_self = true;
    let view = ClusterView {
        live: vec![own, b, c],
        dead: vec!["host-d:100:127.0.0.1:8092".to_owned()],
    };
    a.cluster
        .set(ClusterState::observe(&ClusterState::default(), &view, "host-a", &[]));
    let reply = a.send("GET", "/nagaya/v1/cluster", &[], &[]);
    assert_eq!(reply.status, 200, "{}", reply.text());
    let body = reply.json();
    assert_eq!(body["node"], "host-a");
    assert_eq!(body["dead"], json!(["host-d:100:127.0.0.1:8092"]));
    let names: Vec<&str> = body["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| node["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["host-a", "host-b", "host-c"]);
    let own = &body["nodes"][0];
    assert_eq!(own["self"], true);
    assert_eq!(own["http"], a.http.to_string());
    let b = &body["nodes"][1];
    assert_eq!(
        *b,
        json!({"name": "host-b", "id": "host-b:7:127.0.0.1:9", "self": false, "gossip": "127.0.0.1:9",
               "http": "127.0.0.1:8091",
               "engines": [{"name": "gemma", "state": "ready", "models": ["gemma-4"]},
                           {"name": "zeta", "state": "unloaded", "models": []}],
               "conflicts": {"name": {"own": "host-b:7:127.0.0.1:9", "peers": ["host-b:8:127.0.0.1:10"]},
                             "models": [{"model": "toy", "kept_by": "host-a"}]},
               "refused": {"version": 3, "reason": "busy", "shortfall": []}})
    );
    let c = &body["nodes"][2];
    assert_eq!(
        *c,
        json!({"name": "host-c", "id": "host-c:7:127.0.0.1:9", "self": false, "gossip": "127.0.0.1:9",
               "http": null, "engines": [], "conflicts": {"name": null, "models": []}, "refused": null})
    );
}

#[test]
fn the_route_never_forwards_and_takes_no_query() {
    let a = Node::start("host-a", Has::Objects);
    let reply = a.send("GET", "/nagaya/v1/cluster?host=host-b", &[], &[]);
    assert_eq!(reply.status, 400, "{}", reply.text());
    assert_eq!(reply.json()["error"]["param"], "host");
}
