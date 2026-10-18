//! The bootstrap config.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The default for the stop park's time budget (spec 10.3, ds4-recipe: 10 s).
const DEFAULT_PARK_BUDGET_SECS: u64 = 10;

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
}

impl Bootstrap {
    /// Parse a bootstrap config from TOML text.
    pub fn parse(text: &str) -> Result<Self, BootstrapError> {
        toml::from_str(text).map_err(|error| BootstrapError::Parse {
            message: error.to_string(),
        })
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
