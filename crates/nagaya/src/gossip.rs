//! The gossip of the cluster, on the control runtime (spec 2.3, 3.2a).

use std::fmt;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chitchat::transport::UdpTransport;
use chitchat::{ChitchatConfig, ChitchatHandle, ChitchatId, FailureDetectorConfig, ProtocolVersion, spawn_chitchat};
use tokio::runtime::{Builder, Runtime};

/// The cluster id. A node of another cluster id is refused (`chitchat` handshake).
const CLUSTER_ID: &str = "nagaya";

/// How long the shutdown waits for the control runtime to end its tasks.
const RUNTIME_STOP: Duration = Duration::from_secs(2);

/// The default gossip interval.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(500);

/// How a node joins the cluster.
#[derive(Clone, Debug)]
pub struct GossipConfig {
    /// The node name (spec 2.1 note).
    pub node_name: String,
    /// The UDP address to bind. Port 0 asks the operating system for a port.
    pub bind: SocketAddr,
    /// The seed peers.
    pub seeds: Vec<SocketAddr>,
    /// How often the node starts a gossip round.
    pub interval: Duration,
    /// The failure detector (spec 2.3, phi-accrual).
    pub failure_detector: FailureDetectorConfig,
}

impl GossipConfig {
    /// A config with no seeds.
    pub fn new(node_name: &str, bind: SocketAddr) -> Self {
        Self {
            node_name: node_name.to_owned(),
            bind,
            seeds: Vec::new(),
            interval: DEFAULT_INTERVAL,
            failure_detector: FailureDetectorConfig {
                initial_interval: DEFAULT_INTERVAL * 2,
                ..FailureDetectorConfig::default()
            },
        }
    }
}

/// Why gossip did not start.
#[derive(Debug)]
pub struct GossipError {
    /// What failed.
    pub message: String,
}

impl fmt::Display for GossipError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for GossipError {}

/// The largest gossip value in bytes (spec 2.3 note).
pub const MAX_VALUE_BYTES: usize = 8192;

/// The largest gossip key in bytes.
///
/// `chitchat` 0.13.0 writes the length of a key as a `u16` (`src/serialize.rs:213`), so a key of 65536 bytes or more
/// gets a wrong length on the wire. A delta must also fit one datagram of 65507 bytes (`src/lib.rs:65`), with the
/// value of at most 8192 bytes. This bound is far below both limits.
pub const MAX_KEY_BYTES: usize = 1024;

/// A gossip key that a node may write (spec 2.3). No other key exists, so a pin or a store entry has no key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Key {
    /// `inventory`: the probed devices and tiers.
    Inventory,
    /// `profile:active`: the profile that the node runs.
    ProfileActive,
    /// `engine:<name>`: one engine.
    Engine(String),
    /// `profile:desired:<host>`: a target profile for a host.
    ProfileDesired(String),
    /// `profile:refused`: the profile that the node refused.
    ProfileRefused,
    /// `name:conflict`: the live peers that have this node's name (spec 2.3 note).
    NameConflict,
}

impl Key {
    /// The key text.
    pub fn name(&self) -> String {
        match self {
            Key::Inventory => "inventory".to_owned(),
            Key::ProfileActive => "profile:active".to_owned(),
            Key::Engine(name) => format!("engine:{name}"),
            Key::ProfileDesired(host) => format!("profile:desired:{host}"),
            Key::ProfileRefused => "profile:refused".to_owned(),
            Key::NameConflict => "name:conflict".to_owned(),
        }
    }

    /// The key for a text, or `None` when spec 2.3 has no such key.
    pub fn parse(name: &str) -> Option<Key> {
        match name {
            "inventory" => Some(Key::Inventory),
            "profile:active" => Some(Key::ProfileActive),
            "profile:refused" => Some(Key::ProfileRefused),
            "name:conflict" => Some(Key::NameConflict),
            _ => name
                .strip_prefix("engine:")
                .filter(|engine| !engine.is_empty())
                .map(|engine| Key::Engine(engine.to_owned()))
                .or_else(|| {
                    name.strip_prefix("profile:desired:")
                        .filter(|host| !host.is_empty())
                        .map(|host| Key::ProfileDesired(host.to_owned()))
                }),
        }
    }
}

/// Why a value was not published.
#[derive(Debug, PartialEq, Eq)]
pub enum PublishError {
    /// The key has more bytes than the bound.
    KeyTooLong {
        /// The key.
        key: String,
        /// The size of the key in bytes.
        bytes: usize,
    },
    /// The value has more bytes than the bound.
    TooLarge {
        /// The key.
        key: String,
        /// The size of the value in bytes.
        bytes: usize,
    },
}

impl fmt::Display for PublishError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PublishError::KeyTooLong { key, bytes } => write!(
                formatter,
                "the gossip key `{key}` has {bytes} bytes; the bound is {MAX_KEY_BYTES} bytes"
            ),
            PublishError::TooLarge { key, bytes } => write!(
                formatter,
                "the value of gossip key `{key}` has {bytes} bytes; the bound is {MAX_VALUE_BYTES} bytes"
            ),
        }
    }
}

impl std::error::Error for PublishError {}

/// One node in a cluster view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeView {
    /// The node name.
    pub name: String,
    /// The chitchat id: `name:generation:address`.
    pub id: String,
    /// The generation of the node: the start time in milliseconds since 1970.
    pub generation: u64,
    /// The gossip address of the node.
    pub addr: SocketAddr,
    /// True for this node.
    pub is_self: bool,
    /// The keys of the node and their values, as JSON text.
    pub keys: std::collections::BTreeMap<String, String>,
}

impl NodeView {
    /// The value of a key, parsed.
    pub fn value(&self, key: &Key) -> Option<serde_json::Value> {
        let text = self.keys.get(&key.name())?;
        serde_json::from_str(text).ok()
    }
}

/// The cluster as one node sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterView {
    /// The nodes that the failure detector holds live, with this node.
    pub live: Vec<NodeView>,
    /// The names of the nodes that the failure detector marked dead (spec 2.3). A dead node is not live.
    pub dead: Vec<String>,
}

impl fmt::Display for ClusterView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.dead.is_empty() {
            writeln!(formatter, "dead {:?}", self.dead)?;
        }
        for node in &self.live {
            let keys: Vec<&str> = node.keys.keys().map(String::as_str).collect();
            writeln!(formatter, "live {} {} keys {keys:?}", node.name, node.addr)?;
        }
        Ok(())
    }
}

/// A node's gossip. It owns the control runtime (spec 3.2a).
pub struct Gossip {
    name: String,
    addr: SocketAddr,
    runtime: Runtime,
    handle: ChitchatHandle,
}

impl Gossip {
    /// Start gossip on a new control runtime.
    pub fn start(config: GossipConfig) -> Result<Self, GossipError> {
        let error = |message: String| GossipError { message };
        let bind = reserve_port(config.bind).map_err(|source| {
            error(format!(
                "cannot bind the gossip address {}: {source}; check `gossip_bind`",
                config.bind
            ))
        })?;
        let generation = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|source| error(format!("the clock is before 1970: {source}")))?
            .as_millis() as u64;
        let chitchat = ChitchatConfig {
            chitchat_id: ChitchatId::new(config.node_name.as_str(), generation, bind),
            cluster_id: CLUSTER_ID.to_owned(),
            gossip_interval: config.interval,
            listen_addr: bind,
            seed_nodes: config.seeds.iter().map(SocketAddr::to_string).collect(),
            failure_detector_config: config.failure_detector,
            marked_for_deletion_grace_period: Duration::from_secs(3600),
            catchup_callback: None,
            extra_liveness_predicate: None,
            protocol_version: ProtocolVersion::V0,
        };
        let runtime = Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("nagaya-control")
            .enable_all()
            .build()
            .map_err(|source| error(format!("cannot start the control runtime: {source}")))?;
        let handle = runtime
            .block_on(spawn_chitchat(chitchat, Vec::new(), &UdpTransport))
            .map_err(|source| error(format!("cannot start gossip on {bind}: {source}")))?;
        Ok(Self {
            name: config.node_name,
            addr: bind,
            runtime,
            handle,
        })
    }

    /// The node name.
    pub fn node_name(&self) -> &str {
        &self.name
    }

    /// The chitchat id of this node: `name:generation:address`.
    pub fn id(&self) -> String {
        format!("{:?}", self.handle.chitchat_id())
    }

    /// The generation of this node: its start time in milliseconds since 1970.
    pub fn generation(&self) -> u64 {
        self.handle.chitchat_id().generation_id
    }

    /// The UDP address that gossip is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// The cluster as this node sees it now.
    pub fn view(&self) -> ClusterView {
        self.runtime.block_on(self.handle.with_chitchat(|chitchat| {
            let own = chitchat.self_chitchat_id().clone();
            let mut live: Vec<NodeView> = chitchat
                .live_nodes()
                .map(|id| NodeView {
                    name: id.node_id.to_string(),
                    id: format!("{id:?}"),
                    generation: id.generation_id,
                    addr: id.gossip_advertise_addr,
                    is_self: *id == own,
                    keys: chitchat
                        .node_state(id)
                        .map(|state| {
                            state
                                .key_values()
                                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect();
            live.sort_by(|left, right| left.name.cmp(&right.name));
            let mut dead: Vec<String> = chitchat.dead_nodes().map(|id| id.node_id.to_string()).collect();
            dead.sort();
            ClusterView { live, dead }
        }))
    }

    /// Report each change of the live peers to `report`, one line for each change.
    ///
    /// The lines are `gossip: peer <id> is live`, `gossip: peer <id> keys <list>` and `gossip: peer <id> is dead`.
    /// The `<id>` is the chitchat id, `name:generation:address`, so two peers with one name stay two peers. The
    /// task runs on the control runtime.
    pub fn watch_peers(&self, report: impl Fn(String) + Send + 'static) {
        let own = self.handle.chitchat_id().clone();
        let mut watcher = self
            .runtime
            .block_on(self.handle.with_chitchat(|chitchat| chitchat.live_nodes_watcher()));
        self.runtime.spawn(async move {
            let mut known: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
            loop {
                let now: std::collections::BTreeMap<String, String> = watcher
                    .borrow_and_update()
                    .iter()
                    .filter(|(id, _)| **id != own)
                    .map(|(id, state)| {
                        let keys: Vec<&str> = state.key_values().map(|(key, _)| key).collect();
                        (format!("{id:?}"), keys.join(","))
                    })
                    .collect();
                for (id, keys) in &now {
                    match known.get(id) {
                        None => {
                            report(format!("gossip: peer {id} is live"));
                            report(format!("gossip: peer {id} keys {keys}"));
                        }
                        Some(before) if before != keys => report(format!("gossip: peer {id} keys {keys}")),
                        Some(_) => {}
                    }
                }
                for id in known.keys().filter(|id| !now.contains_key(*id)) {
                    report(format!("gossip: peer {id} is dead"));
                }
                known = now;
                if watcher.changed().await.is_err() {
                    return;
                }
            }
        });
    }

    /// Publish a value under one of this node's keys. The value is compact JSON.
    pub fn publish(&self, key: &Key, value: &serde_json::Value) -> Result<(), PublishError> {
        let name = key.name();
        if name.len() > MAX_KEY_BYTES {
            return Err(PublishError::KeyTooLong {
                bytes: name.len(),
                key: name,
            });
        }
        let text = value.to_string();
        if text.len() > MAX_VALUE_BYTES {
            return Err(PublishError::TooLarge {
                key: name,
                bytes: text.len(),
            });
        }
        self.runtime.block_on(
            self.handle
                .with_chitchat(|chitchat| chitchat.self_node_state().set(name.clone(), text.clone())),
        );
        Ok(())
    }

    /// Remove a key of this node. A key that the node did not write is a defect.
    pub fn withdraw(&self, key: &Key) {
        let name = key.name();
        self.runtime.block_on(
            self.handle
                .with_chitchat(|chitchat| chitchat.self_node_state().delete(&name)),
        );
    }

    /// Stop gossip and the control runtime. The gossip port is free after this call.
    pub fn shutdown(self) {
        let Gossip { runtime, handle, .. } = self;
        if let Err(error) = runtime.block_on(handle.shutdown()) {
            eprintln!("gossip stopped with an error: {error}");
        }
        runtime.shutdown_timeout(RUNTIME_STOP);
    }
}

/// Give port 0 a port from the operating system. The tests bind port 0. A fixed port stays as it is.
///
/// `chitchat` opens the socket itself and does not report the port, so the node asks for a free port first.
fn reserve_port(bind: SocketAddr) -> std::io::Result<SocketAddr> {
    if bind.port() != 0 {
        return Ok(bind);
    }
    UdpSocket::bind(bind)?.local_addr()
}
