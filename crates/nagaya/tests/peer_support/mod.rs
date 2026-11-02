//! Helpers for the in-process tests of the routes with no model in the cluster (spec 6.15, 11.1a): councils that each
//! serve HTTP on loopback, with clusters that a test links by hand.

#![allow(dead_code, unused_imports)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

use nagaya::cluster::{Cluster, ClusterState};
use nagaya::gossip::{ClusterView, NodeView};
use nagaya::http::{Council, Served, Server};
use nagaya_store::objects::ObjectStore;
use serde_json::json;

#[path = "../cluster_support/mod.rs"]
mod cluster_support;
pub use cluster_support::{Reply, send, send_bytes};

#[path = "../council_support/mod.rs"]
mod council_support;
pub use council_support::Fixture;

/// One node: a council that serves HTTP on loopback, on its own runtime.
pub struct Node {
    pub name: String,
    pub council: Arc<Council>,
    pub cluster: Arc<Cluster>,
    pub http: SocketAddr,
    pub fixture: Option<Fixture>,
    _directory: tempfile::TempDir,
    _server: Option<Server>,
    _runtime: tokio::runtime::Runtime,
}

/// What a node serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Has {
    /// An object store and no served model.
    Objects,
    /// The toy model and no object store.
    Model,
    /// The toy model and an object store.
    Both,
}

impl Node {
    /// Start the node `name` with what `has` names.
    pub fn start(name: &str, has: Has) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let fixture = (has != Has::Objects).then(Fixture::new);
        let models = fixture
            .iter()
            .map(|fixture| Served {
                service: fixture.attach_again(),
                aliases: Vec::new(),
                device: None,
            })
            .collect();
        let cluster = Arc::new(Cluster::new(name));
        let mut council = Council::new(models, None).unwrap().with_cluster(Arc::clone(&cluster));
        if has != Has::Model {
            let store = ObjectStore::open(&directory.path().join("objects"), 1 << 20).unwrap();
            council = council.with_objects(store);
        }
        let council = Arc::new(council);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let server = Server::start(runtime.handle(), "127.0.0.1:0".parse().unwrap(), Arc::clone(&council)).unwrap();
        Self {
            name: name.to_owned(),
            http: server.local_addr(),
            council,
            cluster,
            fixture,
            _directory: directory,
            _server: Some(server),
            _runtime: runtime,
        }
    }

    /// The view of this node as a peer sees it in gossip: its `server` value (spec 2.3).
    pub fn as_peer(&self) -> NodeView {
        let local = self.council.local_identity();
        let mut keys = BTreeMap::new();
        keys.insert(
            "server".to_owned(),
            json!({"http": self.http.to_string(), "limits": local.limits, "capabilities": local.capabilities})
                .to_string(),
        );
        NodeView {
            name: self.name.clone(),
            id: format!("{}:1:{}", self.name, self.http),
            generation: 1,
            addr: self.http,
            is_self: false,
            keys,
        }
    }

    /// The view of this node as a peer sees it, with an engine `toy` that serves its toy model under the name `model`.
    /// The node does not serve that name, so a forward to it gets its 503 of a model that it does not serve.
    pub fn as_peer_serving(&self, model: &str) -> NodeView {
        let mut view = self.as_peer();
        let local = self.council.local_identity().models[0].clone();
        let mut entry = serde_json::to_value(local).unwrap();
        let object = entry.as_object_mut().unwrap();
        object.remove("status");
        object.insert("model".to_owned(), json!(model));
        view.keys.insert(
            "engine:toy".to_owned(),
            json!({"identity": null, "models": [], "served": [entry], "state": "ready",
                   "http": self.http.to_string()})
            .to_string(),
        );
        view
    }

    /// Set the cluster of this node to itself and `peers`.
    pub fn see(&self, peers: Vec<NodeView>) {
        let mut own = self.as_peer();
        own.is_self = true;
        let mut live = vec![own];
        live.extend(peers);
        let view = ClusterView { live, dead: Vec::new() };
        let local: Vec<_> = self.council.served_by_engine().into_values().flatten().collect();
        self.cluster
            .set(ClusterState::observe(&self.cluster.state(), &view, &self.name, &local));
    }

    /// Send a request to this node.
    pub fn send(&self, method: &str, path: &str, body: &[u8], headers: &[(&str, &str)]) -> Reply {
        send_bytes(self.http, method, path, body, headers)
    }
}

/// Link the nodes: each sees each other node as a live peer.
pub fn link(nodes: &[&Node]) {
    for node in nodes {
        let peers = nodes
            .iter()
            .filter(|other| other.name != node.name)
            .map(|other| other.as_peer())
            .collect();
        node.see(peers);
    }
}

/// The path of the object of `bytes`, and its id.
pub fn object(bytes: &[u8]) -> (String, String) {
    let id = format!("blake3:{}", blake3::hash(bytes).to_hex());
    (format!("/council/v2/objects/{id}"), id)
}
