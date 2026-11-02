//! What this node knows of the cluster from gossip: the served models of the peers, their servers, and the names that
//! a peer keeps (spec 2.3, 11.1a "The identity of the cluster").
//!
//! The main loop builds a new [`ClusterState`] from each gossip view with [`ClusterState::observe`]. The function is
//! pure, so the tests build views by hand.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;

use gogi_proto::identity::{Capability, ModelStatus, ServedModel, ServerLimits};
use serde_json::Value;

use crate::gossip::{ClusterView, Key, NodeView};

/// A served model of a peer, as gossip names it.
#[derive(Clone, Debug, PartialEq)]
pub struct PeerModel {
    /// The node name of the peer.
    pub node: String,
    /// The chitchat id of the peer, `name:generation:address`.
    pub node_id: String,
    /// The HTTP address of the engine (spec 2.3).
    pub http: SocketAddr,
    /// The engine name on the peer.
    pub engine: String,
    /// The identity entry, with the status from the state of the engine (spec 11.1).
    pub entry: ServedModel,
    /// False for a model whose node left the live set. Its status is then `unloaded`.
    pub live: bool,
    /// True when another live node has the name of this node (spec 2.3, "A duplicate node name").
    pub duplicate: bool,
}

/// The server of a live peer: its `server` value (spec 2.3).
#[derive(Clone, Debug, PartialEq)]
pub struct PeerServer {
    /// The node name of the peer.
    pub node: String,
    /// The HTTP address of the node.
    pub http: SocketAddr,
    /// The server limits of the node's own identity.
    pub limits: ServerLimits,
    /// The server capabilities of the node's own identity.
    pub capabilities: Vec<Capability>,
    /// True when another live node has the name of this node.
    pub duplicate: bool,
}

/// One clash of a served model name between this node and a peer (spec 2.3, "A served model name in the cluster").
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Clash {
    /// The `model` of the entry that lost the name.
    pub model: String,
    /// The node that lost the name.
    pub loser: String,
    /// The node with the lower name, which keeps the name.
    pub keeper: String,
}

/// The cluster as the last gossip view showed it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterState {
    /// The served models of the peers that the identity lists, by `model`. A model whose node left stays here.
    models: BTreeMap<String, PeerModel>,
    /// The servers of the live peers, in the order of their names.
    servers: Vec<PeerServer>,
    /// The `model` of each local served model whose name a peer keeps, with the keeper.
    lost: BTreeMap<String, String>,
    /// The clashes in which this node takes part.
    clashes: Vec<Clash>,
    /// The chitchat ids of the live peers.
    live: BTreeSet<String>,
    /// What could not be read from the gossip of a peer, one line each.
    problems: Vec<String>,
}

/// A served model before the names are resolved.
struct Candidate {
    node: String,
    model: Option<PeerModel>,
    entry: ServedModel,
}

impl ClusterState {
    /// The state after `view`. `own` is the name of this node, and `local` holds its served models. A model of a peer
    /// that left the live set stays from `previous`, with the status `unloaded` (spec 11.1a).
    pub fn observe(previous: &ClusterState, view: &ClusterView, own: &str, local: &[ServedModel]) -> ClusterState {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for node in &view.live {
            *counts.entry(node.name.as_str()).or_default() += 1;
        }
        let peers: Vec<&NodeView> = view.live.iter().filter(|node| !node.is_self).collect();
        let mut state = ClusterState {
            live: peers.iter().map(|node| node.id.clone()).collect(),
            ..ClusterState::default()
        };
        let mut candidates: Vec<Candidate> = local
            .iter()
            .map(|entry| Candidate {
                node: own.to_owned(),
                model: None,
                entry: entry.clone(),
            })
            .collect();
        for node in &peers {
            let duplicate = counts[node.name.as_str()] > 1;
            match server(node, duplicate) {
                Ok(Some(server)) => state.servers.push(server),
                Ok(None) => {}
                Err(problem) => state.problems.push(problem),
            }
            for (key, text) in &node.keys {
                let Some(Key::Engine(engine)) = Key::parse(key) else {
                    continue;
                };
                match engine_models(node, &engine, text, duplicate) {
                    Ok(models) => candidates.extend(models.into_iter().map(|model| Candidate {
                        node: node.name.clone(),
                        entry: model.entry.clone(),
                        model: Some(model),
                    })),
                    Err(problem) => state.problems.push(problem),
                }
            }
        }
        // The node with the lower name keeps a name (spec 2.3). The sort is stable, so this node comes first among
        // nodes of its own name, and its own engines keep their profile order.
        candidates.sort_by(|left, right| left.node.cmp(&right.node));
        let mut taken: BTreeMap<String, String> = BTreeMap::new();
        for candidate in candidates {
            let names = names_of(&candidate.entry);
            if let Some(keeper) = names.iter().find_map(|name| taken.get(name)).cloned() {
                if candidate.node == own || keeper == own {
                    state.clashes.push(Clash {
                        model: candidate.entry.model.clone(),
                        loser: candidate.node.clone(),
                        keeper: keeper.clone(),
                    });
                }
                if candidate.model.is_none() {
                    state.lost.insert(candidate.entry.model.clone(), keeper);
                }
                continue;
            }
            for name in names {
                taken.insert(name, candidate.node.clone());
            }
            if let Some(model) = candidate.model {
                state.models.insert(model.entry.model.clone(), model);
            }
        }
        // A model whose node left stays listed as `unloaded`, unless a live node has its name now (spec 11.1a).
        let live_names: BTreeSet<&str> = peers.iter().map(|node| node.name.as_str()).collect();
        for (model, kept) in &previous.models {
            if live_names.contains(kept.node.as_str()) {
                continue;
            }
            if names_of(&kept.entry).iter().any(|name| taken.contains_key(name)) {
                continue;
            }
            let mut gone = kept.clone();
            gone.live = false;
            gone.entry.status = ModelStatus::Unloaded;
            state.models.insert(model.clone(), gone);
        }
        state.clashes.sort();
        state
    }

    /// The served models of the peers that the identity lists, in byte order of `model`.
    pub fn models(&self) -> impl Iterator<Item = &PeerModel> {
        self.models.values()
    }

    /// The peer model that `name` names, as a `model` or an alias.
    pub fn holder(&self, name: &str) -> Option<&PeerModel> {
        self.models.values().find(|model| model.entry.is_named(name))
    }

    /// The served model of a live peer that `name` names, as a `model` or an alias (spec 2.3). A model whose node left
    /// the live set is not served.
    pub fn served_by_peer(&self, name: &str) -> Option<&PeerModel> {
        self.models
            .values()
            .find(|model| model.live && model.entry.is_named(name))
    }

    /// The keeper of a local served model whose name a peer keeps, by its `model`.
    pub fn keeper(&self, model: &str) -> Option<&str> {
        self.lost.get(model).map(String::as_str)
    }

    /// The local served models whose name a peer keeps: each `model` with its keeper.
    pub fn lost(&self) -> &BTreeMap<String, String> {
        &self.lost
    }

    /// The servers of the live peers.
    pub fn servers(&self) -> &[PeerServer] {
        &self.servers
    }

    /// The clashes in which this node takes part, sorted.
    pub fn clashes(&self) -> &[Clash] {
        &self.clashes
    }

    /// The chitchat ids of the live peers.
    pub fn live(&self) -> &BTreeSet<String> {
        &self.live
    }

    /// What could not be read from the gossip of a peer.
    pub fn problems(&self) -> &[String] {
        &self.problems
    }
}

/// The names of a served model: its `model` and its aliases.
fn names_of(entry: &ServedModel) -> Vec<String> {
    std::iter::once(entry.model.clone())
        .chain(entry.aliases.iter().cloned())
        .collect()
}

/// The status of a served model whose engine has this state in gossip. `fatal` gives `unloaded` (spec 11.1).
fn status_of(state: &str) -> Option<ModelStatus> {
    match state {
        "ready" => Some(ModelStatus::Ready),
        "loading" => Some(ModelStatus::Loading),
        "draining" => Some(ModelStatus::Draining),
        "unloaded" | "fatal" => Some(ModelStatus::Unloaded),
        _ => None,
    }
}

/// The `server` value of a peer, or `None` when the peer has not written it yet.
fn server(node: &NodeView, duplicate: bool) -> Result<Option<PeerServer>, String> {
    let Some(text) = node.keys.get(&Key::Server.name()) else {
        return Ok(None);
    };
    let problem = |what: String| format!("the `server` value of peer {} {what}: {text}", node.id);
    let value: Value = serde_json::from_str(text).map_err(|error| problem(format!("is not JSON ({error})")))?;
    let http = value["http"]
        .as_str()
        .and_then(|http| http.parse().ok())
        .ok_or_else(|| problem("has no `http` address".to_owned()))?;
    let limits = serde_json::from_value(value["limits"].clone())
        .map_err(|error| problem(format!("has bad `limits` ({error})")))?;
    let capabilities = serde_json::from_value(value["capabilities"].clone())
        .map_err(|error| problem(format!("has bad `capabilities` ({error})")))?;
    Ok(Some(PeerServer {
        node: node.name.clone(),
        http,
        limits,
        capabilities,
        duplicate,
    }))
}

/// The served models in the value of `engine:<engine>` of a peer, each with the status from the engine state.
fn engine_models(node: &NodeView, engine: &str, text: &str, duplicate: bool) -> Result<Vec<PeerModel>, String> {
    let problem = |what: String| format!("the value of `engine:{engine}` of peer {} {what}: {text}", node.id);
    let value: Value = serde_json::from_str(text).map_err(|error| problem(format!("is not JSON ({error})")))?;
    let Some(served) = value.get("served") else {
        // A node of an older build writes no list. It serves no model to the cluster.
        return Ok(Vec::new());
    };
    let served = served
        .as_array()
        .ok_or_else(|| problem("has a `served` that is not a list".to_owned()))?;
    if served.is_empty() {
        return Ok(Vec::new());
    }
    let state = value["state"].as_str().unwrap_or("");
    let status = status_of(state).ok_or_else(|| problem(format!("has the unknown state `{state}`")))?;
    let http = value["http"]
        .as_str()
        .and_then(|http| http.parse().ok())
        .ok_or_else(|| problem("has no `http` address".to_owned()))?;
    served
        .iter()
        .map(|entry| {
            let mut entry = entry.clone();
            let object = entry
                .as_object_mut()
                .ok_or_else(|| problem("has a served entry that is not an object".to_owned()))?;
            object.insert(
                "status".to_owned(),
                serde_json::to_value(status).expect("a status is JSON"),
            );
            let entry: ServedModel =
                serde_json::from_value(entry).map_err(|error| problem(format!("has a bad served entry ({error})")))?;
            Ok(PeerModel {
                node: node.name.clone(),
                node_id: node.id.clone(),
                http,
                engine: engine.to_owned(),
                entry,
                live: true,
                duplicate,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(model: &str, aliases: &[&str]) -> ServedModel {
        serde_json::from_value(json!({
            "model": model,
            "aliases": aliases,
            "status": "ready",
            "weight_hash": "w",
            "tokenizer_hash": "t",
            "template": "x",
            "engine": "e",
            "limits": {"context_tokens": 1, "decision_context_tokens": 1, "snapshot_bytes": 1,
                       "contexts_per_decision": 1, "default_timeout_ms": 1},
            "capabilities": ["generate"],
        }))
        .unwrap()
    }

    fn gossip_entry(model: &str, aliases: &[&str]) -> Value {
        let mut value = serde_json::to_value(entry(model, aliases)).unwrap();
        value.as_object_mut().unwrap().remove("status");
        value
    }

    /// A live node with one engine that serves `models` in the state `state`.
    fn node(name: &str, port: u16, engine: &str, state: &str, models: &[Value]) -> NodeView {
        let mut keys = BTreeMap::new();
        keys.insert(
            format!("engine:{engine}"),
            json!({"identity": null, "models": [], "served": models, "state": state,
                   "http": format!("127.0.0.1:{port}")})
            .to_string(),
        );
        keys.insert(
            "server".to_owned(),
            json!({"http": format!("127.0.0.1:{port}"),
                   "limits": {"case_bytes": 1, "choice_options": 2, "pin_bytes": port as u64},
                   "capabilities": ["pins", "classes"]})
            .to_string(),
        );
        NodeView {
            name: name.to_owned(),
            id: format!("{name}:1:127.0.0.1:{port}"),
            generation: 1,
            addr: format!("127.0.0.1:{port}").parse().unwrap(),
            is_self: false,
            keys,
        }
    }

    fn own(name: &str) -> NodeView {
        NodeView {
            name: name.to_owned(),
            id: format!("{name}:1:127.0.0.1:1"),
            generation: 1,
            addr: "127.0.0.1:1".parse().unwrap(),
            is_self: true,
            keys: BTreeMap::new(),
        }
    }

    fn view(nodes: Vec<NodeView>) -> ClusterView {
        ClusterView {
            live: nodes,
            dead: Vec::new(),
        }
    }

    fn listed(state: &ClusterState) -> Vec<(String, String, ModelStatus)> {
        state
            .models()
            .map(|model| (model.entry.model.clone(), model.node.clone(), model.entry.status))
            .collect()
    }

    #[test]
    fn the_models_of_each_live_peer_are_listed_with_the_status_of_their_engine() {
        let view = view(vec![
            own("host-a"),
            node("host-b", 2, "gemma", "ready", &[gossip_entry("gemma-4", &["gemma"])]),
            node("host-c", 3, "mk", "fatal", &[gossip_entry("qwen", &[])]),
            node("host-d", 4, "x", "loading", &[gossip_entry("x-1", &[])]),
        ]);
        let state = ClusterState::observe(&ClusterState::default(), &view, "host-a", &[]);
        assert_eq!(
            listed(&state),
            [
                ("gemma-4".to_owned(), "host-b".to_owned(), ModelStatus::Ready),
                ("qwen".to_owned(), "host-c".to_owned(), ModelStatus::Unloaded),
                ("x-1".to_owned(), "host-d".to_owned(), ModelStatus::Loading),
            ]
        );
        let gemma = state.holder("gemma").expect("an alias names the model");
        assert_eq!(gemma.http, "127.0.0.1:2".parse().unwrap());
        assert_eq!(gemma.engine, "gemma");
        assert_eq!(state.servers().len(), 3);
        assert_eq!(state.servers()[0].limits.pin_bytes, Some(2));
        assert!(state.problems().is_empty(), "{:?}", state.problems());
    }

    #[test]
    fn a_model_whose_node_leaves_stays_as_unloaded_and_one_seen_never_is_not_listed() {
        let first = view(vec![
            own("host-a"),
            node("host-b", 2, "toy", "ready", &[gossip_entry("toy-b", &[])]),
        ]);
        let seen = ClusterState::observe(&ClusterState::default(), &first, "host-a", &[]);
        let gone = ClusterState::observe(&seen, &view(vec![own("host-a")]), "host-a", &[]);
        assert_eq!(
            listed(&gone),
            [("toy-b".to_owned(), "host-b".to_owned(), ModelStatus::Unloaded)]
        );
        assert!(!gone.holder("toy-b").unwrap().live);
        assert!(gone.servers().is_empty() && gone.live().is_empty());
        // It stays through later views too.
        let later = ClusterState::observe(&gone, &view(vec![own("host-a")]), "host-a", &[]);
        assert_eq!(listed(&later), listed(&gone));
        // A peer that comes back with another profile replaces its old entries.
        let back = view(vec![
            own("host-a"),
            node("host-b", 2, "toy", "ready", &[gossip_entry("toy-c", &[])]),
        ]);
        let back = ClusterState::observe(&later, &back, "host-a", &[]);
        assert_eq!(
            listed(&back),
            [("toy-c".to_owned(), "host-b".to_owned(), ModelStatus::Ready)]
        );
        // A coordinator that starts while the peer is down never saw it.
        let fresh = ClusterState::observe(&ClusterState::default(), &view(vec![own("host-a")]), "host-a", &[]);
        assert!(listed(&fresh).is_empty());
    }

    #[test]
    fn the_node_with_the_lower_name_keeps_a_clashing_name() {
        let local = [entry("toy", &[])];
        // This node has the higher name, so it loses its copy and the peer keeps the name.
        let view_b = view(vec![
            node("host-a", 2, "toy", "ready", &[gossip_entry("toy", &[])]),
            own("host-b"),
        ]);
        let on_b = ClusterState::observe(&ClusterState::default(), &view_b, "host-b", &local);
        assert_eq!(on_b.keeper("toy"), Some("host-a"));
        assert_eq!(
            listed(&on_b),
            [("toy".to_owned(), "host-a".to_owned(), ModelStatus::Ready)]
        );
        let clash = Clash {
            model: "toy".to_owned(),
            loser: "host-b".to_owned(),
            keeper: "host-a".to_owned(),
        };
        assert_eq!(on_b.clashes(), std::slice::from_ref(&clash));
        // On the keeper, the peer's copy is not listed, and the keeper knows of the clash too.
        let view_a = view(vec![
            own("host-a"),
            node("host-b", 3, "toy", "ready", &[gossip_entry("toy", &[])]),
        ]);
        let on_a = ClusterState::observe(&ClusterState::default(), &view_a, "host-a", &local);
        assert_eq!(on_a.keeper("toy"), None);
        assert!(listed(&on_a).is_empty());
        assert_eq!(on_a.clashes(), [clash]);
    }

    #[test]
    fn an_alias_that_names_a_model_of_another_node_is_a_clash() {
        let view = view(vec![
            own("host-z"),
            node("host-a", 2, "e", "ready", &[gossip_entry("alpha", &["shared"])]),
            node("host-b", 3, "e", "ready", &[gossip_entry("shared", &[])]),
        ]);
        let state = ClusterState::observe(&ClusterState::default(), &view, "host-z", &[]);
        assert_eq!(
            listed(&state),
            [("alpha".to_owned(), "host-a".to_owned(), ModelStatus::Ready)]
        );
        assert!(state.clashes().is_empty(), "this node takes no part in the clash");
    }

    #[test]
    fn a_departed_model_that_a_live_node_now_names_is_not_listed() {
        let first = view(vec![
            own("host-a"),
            node("host-b", 2, "toy", "ready", &[gossip_entry("toy", &[])]),
        ]);
        let seen = ClusterState::observe(&ClusterState::default(), &first, "host-a", &[]);
        let local = [entry("toy", &[])];
        let gone = ClusterState::observe(&seen, &view(vec![own("host-a")]), "host-a", &local);
        assert!(listed(&gone).is_empty());
        assert_eq!(gone.keeper("toy"), None);
    }

    #[test]
    fn a_peer_with_a_duplicate_name_is_marked() {
        let mut twin = node("host-b", 3, "toy", "ready", &[gossip_entry("toy-twin", &[])]);
        twin.id = "host-b:2:127.0.0.1:3".to_owned();
        let view = view(vec![
            own("host-a"),
            node("host-b", 2, "toy", "ready", &[gossip_entry("toy-b", &[])]),
            twin,
        ]);
        let state = ClusterState::observe(&ClusterState::default(), &view, "host-a", &[]);
        assert!(state.models().all(|model| model.duplicate));
        assert!(state.servers().iter().all(|server| server.duplicate));
        assert_eq!(state.live().len(), 2);
    }

    #[test]
    fn a_value_that_does_not_read_is_a_problem_and_lists_nothing() {
        let mut bad = node("host-b", 2, "toy", "ready", &[]);
        bad.keys.insert("engine:toy".to_owned(), "{\"served\": 3}".to_owned());
        bad.keys.insert("server".to_owned(), "{\"http\": 3}".to_owned());
        let state = ClusterState::observe(&ClusterState::default(), &view(vec![own("host-a"), bad]), "host-a", &[]);
        assert!(listed(&state).is_empty());
        assert_eq!(state.problems().len(), 2, "{:?}", state.problems());
    }
}
