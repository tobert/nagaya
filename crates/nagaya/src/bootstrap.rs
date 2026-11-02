//! The bootstrap config.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::inventory::is_pci_address;

/// The default for the stop park's time budget (spec 10.3, ds4-recipe: 10 s).
const DEFAULT_PARK_BUDGET_SECS: u64 = 10;

/// Where Linux keeps the host name. It is the default node name (spec 2.1 note).
const HOST_NAME_FILE: &str = "/proc/sys/kernel/hostname";

/// An error in the bootstrap config.
#[derive(Debug)]
pub enum BootstrapError {
    /// The file could not be read.
    Read {
        /// The file path.
        path: PathBuf,
        /// The read error.
        source: std::io::Error,
    },
    /// The text is not a valid bootstrap config.
    Parse {
        /// The parser's message.
        message: String,
    },
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BootstrapError::Read { path, source } => {
                write!(
                    formatter,
                    "cannot read the bootstrap config `{}`: {source}",
                    path.display()
                )
            }
            BootstrapError::Parse { message } => {
                write!(formatter, "the bootstrap config is not valid: {message}")
            }
        }
    }
}

impl std::error::Error for BootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BootstrapError::Read { source, .. } => Some(source),
            BootstrapError::Parse { .. } => None,
        }
    }
}

/// The bootstrap config (spec 2.1, 11.6). It holds the facts that the process needs before it probes the machine.
/// An unknown key is an error, so a misspelled key stops the process.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Bootstrap {
    /// The HTTP listen address (spec 11.6).
    pub http_bind: SocketAddr,
    /// The gossip listen address (spec 11.6).
    pub gossip_bind: SocketAddr,
    /// The store directory (spec 2.1).
    pub store_dir: PathBuf,
    /// The gossip seed peers (spec 2.1).
    #[serde(default)]
    pub seed_peers: Vec<SocketAddr>,
    /// The time budget for the stop park, in seconds (spec 10.3).
    #[serde(default = "default_park_budget_secs")]
    pub park_budget_secs: u64,
    /// The node name: the `<host>` of the gossip keys (spec 2.1 note, 2.3). Without it, the host name.
    #[serde(default)]
    pub node_name: Option<String>,
    /// The PCI addresses of the devices that get no work (spec 2.1, "Ignored devices"). By default, none.
    #[serde(default)]
    pub ignore_devices: Vec<String>,
}

impl Bootstrap {
    /// Parse a bootstrap config from TOML text.
    pub fn parse(text: &str) -> Result<Self, BootstrapError> {
        let config: Self = toml::from_str(text).map_err(|error| BootstrapError::Parse {
            message: error.to_string(),
        })?;
        if config.gossip_bind.ip().is_unspecified() {
            // The node advertises this address to its peers (spec 2.3), so a wildcard address can reach no peer.
            return Err(BootstrapError::Parse {
                message: format!(
                    "gossip_bind `{}` is an unspecified address that no peer can reach; use the address of this machine",
                    config.gossip_bind
                ),
            });
        }
        if config.http_bind.ip().is_unspecified() {
            // The node publishes this address in `server` and `engine:<name>` (spec 2.3), so a wildcard address can
            // reach no peer (spec 2.1, Proposed).
            return Err(BootstrapError::Parse {
                message: format!(
                    "http_bind `{}` is an unspecified address that no peer can reach; use the address of this machine",
                    config.http_bind
                ),
            });
        }
        if let Some(name) = &config.node_name {
            check_node_name(name)?;
        }
        for address in &config.ignore_devices {
            if !is_pci_address(address) {
                return Err(BootstrapError::Parse {
                    message: format!(
                        "ignore_devices holds `{address}`, which is not a PCI address; write it as `dddd:bb:dd.f` in hex, for example `0000:80:00.0`"
                    ),
                });
            }
        }
        Ok(config)
    }

    /// The node name. It is `node_name`, or else the host name of the machine.
    pub fn node_name(&self) -> Result<String, BootstrapError> {
        if let Some(name) = &self.node_name {
            return Ok(name.clone());
        }
        let host = std::fs::read_to_string(HOST_NAME_FILE).map_err(|source| BootstrapError::Read {
            path: PathBuf::from(HOST_NAME_FILE),
            source,
        })?;
        let host = host.trim();
        check_node_name(host)?;
        Ok(host.to_owned())
    }

    /// Read a bootstrap config from a TOML file.
    pub fn read(path: &Path) -> Result<Self, BootstrapError> {
        let text = std::fs::read_to_string(path).map_err(|source| BootstrapError::Read {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text)
    }
}

/// The default time budget for the stop park, in seconds.
fn default_park_budget_secs() -> u64 {
    DEFAULT_PARK_BUDGET_SECS
}

/// The most bytes in a node name, so each gossip key that holds it fits 1024 bytes (spec 2.1, 2.3).
const MAX_NODE_NAME_BYTES: usize = 128;

/// Refuse a node name that a gossip key cannot hold. A key is `profile:desired:<host>`, so a name has no colon.
fn check_node_name(name: &str) -> Result<(), BootstrapError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_NODE_NAME_BYTES
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'));
    if valid {
        Ok(())
    } else {
        Err(BootstrapError::Parse {
            message: format!(
                "node_name `{name}` is not valid; use 1 to {MAX_NODE_NAME_BYTES} bytes of letters, digits, `.`, `_` and `-`"
            ),
        })
    }
}
