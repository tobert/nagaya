//! What a node writes to gossip: its inventory, its profile and its engines (spec 2.3).

use std::collections::BTreeMap;
use std::net::SocketAddr;

use serde_json::{Value, json};

use crate::gossip::{Gossip, Key, PublishError};
use crate::inventory::Inventory;
use crate::notify::EngineState;
use crate::park::identity_digest;
use crate::process::OpenedEngine;
use crate::profile::PlacementProfile;

/// What the node last wrote for one engine. A new state keeps the rest.
#[derive(Clone, Debug)]
struct EngineRecord {
    /// The hex digest of the identity. The engine has none until it opens.
    identity: Option<String>,
    /// The models that the engine serves.
    models: Vec<String>,
    state: EngineState,
}

/// A node: its gossip and the last record of each engine.
pub struct Node {
    gossip: Gossip,
    http: SocketAddr,
    engines: BTreeMap<String, EngineRecord>,
}

impl Node {
    /// Make a node. `http` is the `http_bind` value (spec 2.3).
    pub fn new(gossip: Gossip, http: SocketAddr) -> Self {
        Self {
            gossip,
            http,
            engines: BTreeMap::new(),
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
        let active = serde_json::to_value(profile).expect("a placement profile is JSON data");
        self.gossip.publish(&Key::ProfileActive, &active)
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
