//! What a node writes to gossip: its inventory, its profile and its engines (spec 2.3).

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;

use serde_json::{Value, json};

use crate::gossip::{ClusterView, Gossip, Key, PublishError};
use crate::inventory::Inventory;
use crate::notify::EngineState;
use crate::park::identity_digest;
use crate::process::OpenedEngine;
use crate::profile::PlacementProfile;
use crate::swap::Refusal;

/// What the node last wrote for one engine. A new state keeps the rest.
#[derive(Clone, Debug)]
struct EngineRecord {
    /// The hex digest of the identity. The engine has none until it opens.
    identity: Option<String>,
    /// The models that the engine serves.
    models: Vec<String>,
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
                    "memory_bytes": device.memory_bytes,
                    "unified": device.unified,
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
            let record = EngineRecord {
                identity: Some(digest),
                models: vec![engine.identity.model.clone()],
                state: EngineState::Ready,
            };
            self.write_engine(&engine.name, record)?;
        }
        Ok(())
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

    /// Stop gossip.
    pub fn shutdown(self) {
        self.gossip.shutdown();
    }

    fn write_engine(&mut self, name: &str, record: EngineRecord) -> Result<(), PublishError> {
        let value = json!({
            "identity": record.identity,
            "models": record.models,
            "state": record.state.as_str(),
            "http": self.http.to_string(),
        });
        self.gossip.publish(&Key::Engine(name.to_owned()), &value)?;
        self.engines.insert(name.to_owned(), record);
        Ok(())
    }
}
