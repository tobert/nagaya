//! The cluster of nagaya processes over gossip and HTTP (spec 2.3, 11.1a).
//!
//! The main loop writes a new [`ClusterState`] from each gossip view. The routes read it on the control runtime, and
//! they reach the peers through the [`call::Caller`].

pub mod call;
pub mod state;

use std::collections::BTreeSet;
use std::sync::RwLock;

pub use state::{Clash, ClusterState, PeerModel, PeerServer};

/// The cluster as the routes see it: the last state from gossip, and the calls to peers.
pub struct Cluster {
    caller: call::Caller,
    state: RwLock<ClusterState>,
    /// The chitchat ids of the live peers. A forward ends when its peer leaves this set (spec 11.1a).
    live: tokio::sync::watch::Sender<BTreeSet<String>>,
}

impl Cluster {
    /// The cluster of the node `own`, before the first gossip view.
    pub fn new(own: &str) -> Self {
        Self {
            caller: call::Caller::new(own),
            state: RwLock::new(ClusterState::default()),
            live: tokio::sync::watch::Sender::new(BTreeSet::new()),
        }
    }

    /// The calls of this node to its peers.
    pub fn caller(&self) -> &call::Caller {
        &self.caller
    }

    /// The node name of this node.
    pub fn own(&self) -> &str {
        self.caller.own()
    }

    /// The last state from gossip.
    pub fn state(&self) -> ClusterState {
        self.state.read().expect("the cluster state lock is poisoned").clone()
    }

    /// Keep a new state from gossip.
    pub fn set(&self, state: ClusterState) {
        let live = state.live().clone();
        *self.state.write().expect("the cluster state lock is poisoned") = state;
        self.live.send_if_modified(|known| {
            if *known == live {
                return false;
            }
            *known = live;
            true
        });
    }

    /// A receiver of the chitchat ids of the live peers.
    pub fn watch_live(&self) -> tokio::sync::watch::Receiver<BTreeSet<String>> {
        self.live.subscribe()
    }
}
