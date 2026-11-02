//! What a node writes to gossip: its inventory, its profile and its engines (spec 2.3).

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::time::Instant;

use gogi_proto::identity::{Capability, ServedModel, ServerLimits};
use serde_json::{Value, json};

use crate::cluster::{Clash, Cluster, ClusterState};
use crate::gossip::{ClusterView, Gossip, Key, PublishError};
use crate::inventory::Inventory;
use crate::notify::EngineState;
use crate::park::identity_digest;
use crate::process::{OpenedEngine, Ports};
use crate::profile::PlacementProfile;
use crate::reprofile::DesireAnswer;
use crate::state_feed::StateFeed;
use crate::swap::Refusal;

/// A served model as gossip holds it: the gogi identity entry without `status`, because the state of the engine
/// gives the status (spec 2.3, 11.1).
fn gossip_entry(model: &ServedModel) -> Value {
    let mut value = serde_json::to_value(model).expect("a served model is JSON data");
    value
        .as_object_mut()
        .expect("a served model is a JSON object")
        .remove("status");
    value
}

/// What the node last wrote for one engine. A new state keeps the rest.
#[derive(Clone, Debug)]
struct EngineRecord {
    /// The hex digest of the identity. The engine has none until it opens.
    identity: Option<String>,
    /// The models that the engine serves.
    models: Vec<String>,
    /// The served models of the engine, each a gogi identity entry without `status` (spec 2.3, "What a node
    /// publishes for the cluster"). The list is empty until the council attaches.
    served: Vec<Value>,
    state: EngineState,
}

/// The ids of the live peers that have `name` and are not this node's own earlier run (spec 2.3 note).
///
/// An earlier run has this node's gossip address and a lower generation. A peer with another address, or with this
/// address and a higher generation, is a duplicate.
pub fn duplicate_names(view: &ClusterView, name: &str, addr: SocketAddr, generation: u64) -> Vec<String> {
    let mut peers: Vec<String> = view
        .live
        .iter()
        .filter(|node| !node.is_self && node.name == name)
        .filter(|node| !(node.addr == addr && node.generation < generation))
        .map(|node| node.id.clone())
        .collect();
    peers.sort();
    peers
}

/// A node: its gossip and the last record of each engine.
pub struct Node {
    gossip: Gossip,
    http: SocketAddr,
    engines: BTreeMap<String, EngineRecord>,
    refused: bool,
    /// The ids of the peers that the node logged as a duplicate name.
    logged_names: BTreeSet<String>,
    /// The peers in the last `name:conflict` value. It is empty when the node wrote no such key.
    conflict: Vec<String>,
    /// The clashes of served model names that the node logged (spec 2.3).
    logged_clashes: BTreeSet<Clash>,
    /// The gossip problems of peers that the node logged.
    logged_problems: BTreeSet<String>,
    /// The entries of the last `model:conflict` value. It is empty when the node wrote no such key.
    model_conflict: BTreeMap<String, String>,
}

impl Node {
    /// Make a node. `http` is the `http_bind` value (spec 2.3).
    pub fn new(gossip: Gossip, http: SocketAddr) -> Self {
        Self {
            gossip,
            http,
            engines: BTreeMap::new(),
            refused: false,
            logged_names: BTreeSet::new(),
            conflict: Vec::new(),
            logged_clashes: BTreeSet::new(),
            logged_problems: BTreeSet::new(),
            model_conflict: BTreeMap::new(),
        }
    }

    /// The node's gossip.
    pub fn gossip(&self) -> &Gossip {
        &self.gossip
    }

    /// Write `inventory` and `profile:active`.
    pub fn publish_machine(&self, inventory: &Inventory, profile: &PlacementProfile) -> Result<(), PublishError> {
        let devices: Vec<Value> = inventory
            .devices
            .iter()
            .map(|device| {
                json!({
                    "id": device.id,
                    "name": device.name,
                    "pci_address": device.pci_address,
                    "memory_bytes": device.memory_bytes,
                    "unified": device.unified,
                    "ignored": device.ignored,
                })
            })
            .collect();
        let inventory = json!({"devices": devices, "host_bytes": inventory.host_bytes});
        self.gossip.publish(&Key::Inventory, &inventory)?;
        self.publish_profile(profile)
    }

    /// Write `profile:active`.
    pub fn publish_profile(&self, profile: &PlacementProfile) -> Result<(), PublishError> {
        let active = serde_json::to_value(profile).expect("a placement profile is JSON data");
        self.gossip.publish(&Key::ProfileActive, &active)
    }

    /// Write `profile:desired:<host>` for a live node, when `version` is above each live version for the host (spec
    /// 2.4, "The routes of a profile for another host").
    pub fn desire(&self, host: &str, version: u64, profile: &Value) -> DesireAnswer {
        let view = self.gossip.view();
        if !view.live.iter().any(|node| node.name == host) {
            return DesireAnswer::UnknownHost;
        }
        let key = Key::ProfileDesired(host.to_owned());
        let highest = view
            .live
            .iter()
            .filter_map(|node| node.value(&key))
            .filter_map(|value| value["version"].as_u64())
            .max();
        if let Some(highest) = highest.filter(|highest| version <= *highest) {
            return DesireAnswer::Conflict { highest };
        }
        match self
            .gossip
            .publish(&key, &json!({"version": version, "profile": profile}))
        {
            Ok(()) => DesireAnswer::Published,
            Err(error) => DesireAnswer::TooLarge(error.to_string()),
        }
    }

    /// Write `profile:refused` (spec 2.4 note).
    pub fn publish_refusal(&mut self, refusal: &Refusal) -> Result<(), PublishError> {
        self.gossip.publish(&Key::ProfileRefused, &refusal.to_value())?;
        self.refused = true;
        Ok(())
    }

    /// Remove `profile:refused`, when the node wrote it.
    pub fn clear_refusal(&mut self) {
        if self.refused {
            self.gossip.withdraw(&Key::ProfileRefused);
            self.refused = false;
        }
    }

    /// Write the engines of the profile as `loading`.
    pub fn engines_loading(&mut self, names: &[String]) -> Result<(), PublishError> {
        for name in names {
            let record = EngineRecord {
                identity: None,
                models: Vec::new(),
                served: Vec::new(),
                state: EngineState::Loading,
            };
            self.write_engine(name, record)?;
        }
        Ok(())
    }

    /// Write the opened engines as `ready`, with their identity digest and models.
    pub fn engines_ready(&mut self, opened: &[OpenedEngine]) -> Result<(), PublishError> {
        for engine in opened {
            let digest = blake3::Hash::from_bytes(identity_digest(&engine.identity))
                .to_hex()
                .to_string();
            let served = self
                .engines
                .get(&engine.name)
                .map(|record| record.served.clone())
                .unwrap_or_default();
            let record = EngineRecord {
                identity: Some(digest),
                models: vec![engine.identity.model.clone()],
                served,
                state: EngineState::Ready,
            };
            self.write_engine(&engine.name, record)?;
        }
        Ok(())
    }

    /// Keep the identity and the models of engines that a swap opened, with no write. The state feed writes them with
    /// the next state of the engine (spec 2.3, "The state of an engine in gossip").
    pub fn set_identity(&mut self, opened: &[OpenedEngine]) {
        for engine in opened {
            let record = self.engines.get_mut(&engine.name).unwrap_or_else(|| {
                panic!(
                    "engine `{}` has no gossip record; write loading or ready first",
                    engine.name
                )
            });
            let digest = blake3::Hash::from_bytes(identity_digest(&engine.identity));
            record.identity = Some(digest.to_hex().to_string());
            record.models = vec![engine.identity.model.clone()];
        }
    }

    /// Write the state of each engine of `names` that changed since its last write, through the feed (spec 2.3, "The
    /// state of an engine in gossip"). The feed writes one engine at most once each second.
    pub fn publish_states(
        &mut self,
        feed: &mut StateFeed,
        ports: &Ports,
        names: &[String],
        now: Instant,
    ) -> Result<(), PublishError> {
        let states: Vec<_> = names
            .iter()
            .map(|name| {
                let state = ports
                    .engine_state(name)
                    .unwrap_or_else(|| panic!("engine `{name}` of the profile has no state"));
                (name.clone(), state, self.published_state(name))
            })
            .collect();
        for (name, state) in feed.due(&states, now) {
            self.engines_state(&[name], state)?;
        }
        Ok(())
    }

    /// The state of an engine that the node wrote last, or `None` for an engine that it never wrote.
    pub fn published_state(&self, engine: &str) -> Option<EngineState> {
        self.engines.get(engine).map(|record| record.state)
    }

    /// Write a new state for engines that the node already wrote. An engine that the node did not write is a defect.
    pub fn engines_state(&mut self, names: &[String], state: EngineState) -> Result<(), PublishError> {
        for name in names {
            let mut record = self
                .engines
                .get(name)
                .unwrap_or_else(|| panic!("engine `{name}` has no gossip record; write loading or ready first"))
                .clone();
            record.state = state;
            self.write_engine(name, record)?;
        }
        Ok(())
    }

    /// Report the live peers that have this node's name. Call it often.
    ///
    /// A peer has this node's name and another chitchat id (spec 2.3 note). The node logs an error once for each such
    /// peer id, and it writes `name:conflict` while such a peer is live. It takes no other action.
    pub fn check_name(&mut self) {
        let view = self.gossip.view();
        let name = self.gossip.node_name();
        let own = self.gossip.id();
        let peers = duplicate_names(&view, name, self.gossip.local_addr(), self.gossip.generation());
        for peer in &peers {
            if self.logged_names.insert(peer.clone()) {
                tracing::error!("error: duplicate node name `{name}`: peer {peer} and this node {own} have one name");
            }
        }
        if peers == self.conflict {
            return;
        }
        if peers.is_empty() {
            self.gossip.withdraw(&Key::NameConflict);
        } else {
            let value = json!({"own": own, "peers": peers});
            if let Err(error) = self.gossip.publish(&Key::NameConflict, &value) {
                tracing::warn!("cannot publish to gossip: {error}");
                return;
            }
        }
        self.conflict = peers;
    }

    /// Write the served models of each engine that the node wrote, in its `engine:<name>` value (spec 2.3, "What a
    /// node publishes for the cluster"). An engine that `served` does not name serves no model, so its list is empty.
    /// An engine of `served` that the node did not write is a defect.
    pub fn engines_served(&mut self, served: &BTreeMap<String, Vec<ServedModel>>) -> Result<(), PublishError> {
        if let Some(engine) = served.keys().find(|engine| !self.engines.contains_key(*engine)) {
            panic!("engine `{engine}` serves a model and has no gossip record; write loading or ready first");
        }
        let names: Vec<String> = self.engines.keys().cloned().collect();
        for name in names {
            let entries = served.get(&name).map(Vec::as_slice).unwrap_or_default();
            let mut record = self.engines[&name].clone();
            record.served = entries.iter().map(gossip_entry).collect();
            self.write_engine(&name, record)?;
        }
        Ok(())
    }

    /// Write `server`: the HTTP address of the node, and the server limits and capabilities of its own identity (spec
    /// 2.3, "What a node publishes for the cluster"). A node with no engine writes it too.
    pub fn publish_server(&self, limits: &ServerLimits, capabilities: &[Capability]) -> Result<(), PublishError> {
        let value = json!({
            "http": self.http.to_string(),
            "limits": limits,
            "capabilities": capabilities,
        });
        self.gossip.publish(&Key::Server, &value)
    }

    /// Read the cluster from gossip into `cluster`. `local` holds the served models of this node. Call it often.
    ///
    /// The node logs one error for each clash of a served model name, and for each peer value that does not read. It
    /// writes `model:conflict` while a peer keeps the name of a local model (spec 2.3, "A served model name in the
    /// cluster").
    pub fn observe_cluster(&mut self, cluster: &Cluster, local: &[ServedModel]) {
        let view = self.gossip.view();
        let state = ClusterState::observe(&cluster.state(), &view, self.gossip.node_name(), local);
        for clash in state.clashes() {
            if self.logged_clashes.insert(clash.clone()) {
                tracing::error!(
                    "error: served model `{}` has one name on node {} and node {}; node {} keeps the name until a \
                     client renames one (spec 2.3)",
                    clash.model,
                    clash.keeper,
                    clash.loser,
                    clash.keeper
                );
            }
        }
        for problem in state.problems() {
            if self.logged_problems.insert(problem.clone()) {
                tracing::error!("error: {problem}; the peer's value is left out");
            }
        }
        if *state.lost() != self.model_conflict {
            if state.lost().is_empty() {
                self.gossip.withdraw(&Key::ModelConflict);
            } else {
                let models: Vec<Value> = state
                    .lost()
                    .iter()
                    .map(|(model, keeper)| json!({"model": model, "kept_by": keeper}))
                    .collect();
                if let Err(error) = self.gossip.publish(&Key::ModelConflict, &json!({"models": models})) {
                    tracing::warn!("cannot publish to gossip: {error}");
                }
            }
            self.model_conflict = state.lost().clone();
        }
        cluster.set(state);
    }

    /// Stop gossip.
    pub fn shutdown(self) {
        self.gossip.shutdown();
    }

    fn write_engine(&mut self, name: &str, record: EngineRecord) -> Result<(), PublishError> {
        let value = json!({
            "identity": record.identity,
            "models": record.models,
            "served": record.served,
            "state": record.state.as_str(),
            "http": self.http.to_string(),
        });
        self.gossip.publish(&Key::Engine(name.to_owned()), &value)?;
        self.engines.insert(name.to_owned(), record);
        Ok(())
    }
}
