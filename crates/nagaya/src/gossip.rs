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

/// One node in a cluster view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeView {
    /// The node name.
    pub name: String,
    /// The gossip address of the node.
    pub addr: SocketAddr,
    /// True for this node.
    pub is_self: bool,
}

/// The cluster as one node sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterView {
    /// The nodes that the failure detector holds live, with this node.
    pub live: Vec<NodeView>,
}

impl fmt::Display for ClusterView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for node in &self.live {
            writeln!(formatter, "live {} {}", node.name, node.addr)?;
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
                    addr: id.gossip_advertise_addr,
                    is_self: *id == own,
                })
                .collect();
            live.sort_by(|left, right| left.name.cmp(&right.name));
            ClusterView { live }
        }))
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
