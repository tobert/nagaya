//! A desired profile from a peer: the order of versions, admission, persistence and refusal (spec 2.4, 3.3).

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use nagaya_ledger::LedgerError;

use crate::admission::AdmitError;
use crate::gossip::{ClusterView, Key};
use crate::process::Process;
use crate::profile::{PlacementProfile, ProfileError, TierSpec};

/// The file that holds the handled version, next to `profile.toml` (spec 2.4 note).
pub const VERSION_FILE: &str = "profile.version";

/// The value of `profile:desired:<host>`: a version and a profile (spec 2.4 note).
pub fn desired_value(version: u64, profile: &PlacementProfile) -> Value {
    json!({"version": version, "profile": serde_json::to_value(profile).expect("a placement profile is JSON data")})
}

/// A desired profile that a peer offers.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// The peer that holds the value.
    pub from: String,
    /// The version.
    pub version: u64,
    /// The profile, not yet parsed.
    pub profile: Value,
}

/// The shortfall of one budget, in bytes (spec 2.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shortfall {
    /// The tiers that share the budget.
    pub tiers: Vec<TierSpec>,
    /// The bytes that the budget lacks.
    pub bytes: u64,
}

/// A profile that the node refuses. The active profile stays (spec 2.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The version of the refused profile.
    pub version: u64,
    /// Why the node refuses it.
    pub reason: String,
    /// The shortfall for each short budget. It is empty when the profile does not parse.
    pub shortfall: Vec<Shortfall>,
}

impl Refusal {
    /// The value of `profile:refused` (spec 2.4 note).
    pub fn to_value(&self) -> Value {
        let shortfall: Vec<Value> = self
            .shortfall
            .iter()
            .map(|entry| {
                let tiers: Vec<String> = entry.tiers.iter().map(TierSpec::to_string).collect();
                json!({"tiers": tiers, "bytes": entry.bytes})
            })
            .collect();
        json!({"version": self.version, "reason": self.reason, "shortfall": shortfall})
    }
}

/// What the node did with a desired profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The profile differs from the active profile. The node persisted it and must stop with exit 4 (spec 3.3).
    Restart,
    /// The profile equals the active profile. The node changed nothing and keeps running.
    Unchanged,
    /// The node refused the profile.
    Refused(Refusal),
}

/// Why a desired profile cannot be handled. The process stops, because it cannot keep its order (spec 8.5).
#[derive(Debug)]
pub enum SwapError {
    /// The version file cannot be read or written.
    Version {
        /// The file path.
        path: PathBuf,
        /// What failed.
        message: String,
    },
    /// The profile file cannot be written.
    Profile(ProfileError),
}

impl std::fmt::Display for SwapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwapError::Version { path, message } => {
                write!(formatter, "cannot use the version file `{}`: {message}", path.display())
            }
            SwapError::Profile(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SwapError {}

/// The order of desired profiles for one host.
pub struct Swap {
    store_dir: PathBuf,
    host: String,
    handled: u64,
    active: PlacementProfile,
    reported: HashSet<String>,
}

impl Swap {
    /// Open the swap state of a store directory. A missing version file means version 0.
    pub fn open(store_dir: &Path, host: &str, active: PlacementProfile) -> Result<Self, SwapError> {
        let path = store_dir.join(VERSION_FILE);
        let handled = match std::fs::read_to_string(&path) {
            Ok(text) => text.trim().parse::<u64>().map_err(|error| SwapError::Version {
                path: path.clone(),
                message: format!("{error}; the file holds one whole number, or delete the file"),
            })?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => {
                return Err(SwapError::Version {
                    path,
                    message: error.to_string(),
                });
            }
        };
        Ok(Self {
            store_dir: store_dir.to_owned(),
            host: host.to_owned(),
            handled,
            active,
            reported: HashSet::new(),
        })
    }

    /// The highest version that the node handled.
    pub fn handled(&self) -> u64 {
        self.handled
    }

    /// The profile that the node runs.
    pub fn active(&self) -> &PlacementProfile {
        &self.active
    }

    /// The newest desired profile for this host in the view, when its version is above the handled version.
    pub fn candidate(&mut self, view: &ClusterView) -> Option<Candidate> {
        let key = Key::ProfileDesired(self.host.clone()).name();
        let mut best: Option<Candidate> = None;
        for node in &view.live {
            let Some(text) = node.keys.get(&key) else { continue };
            let parsed: Result<Value, String> = serde_json::from_str(text).map_err(|error| error.to_string());
            let version = parsed.as_ref().map_err(Clone::clone).and_then(|value| {
                value["version"]
                    .as_u64()
                    .ok_or_else(|| "it has no whole number `version`".to_owned())
            });
            let (value, version) = match (parsed, version) {
                (Ok(value), Ok(version)) => (value, version),
                (_, Err(reason)) | (Err(reason), _) => {
                    if self.reported.insert(format!("{}\n{text}", node.name)) {
                        eprintln!("ignoring the desired profile from peer {}: {reason}", node.name);
                    }
                    continue;
                }
            };
            if best.as_ref().is_none_or(|best| version > best.version) {
                best = Some(Candidate {
                    from: node.name.clone(),
                    version,
                    profile: value["profile"].clone(),
                });
            }
        }
        best.filter(|candidate| candidate.version > self.handled)
    }

    /// Admit a candidate with the ledger, persist it, and say what the node does next (spec 3.3).
    pub fn handle(&mut self, candidate: Candidate, process: &Process) -> Result<Outcome, SwapError> {
        let version = candidate.version;
        // The node handles a version once, also when it refuses the version.
        self.handled = version;
        let refuse = |reason: String, shortfall: Vec<Shortfall>| {
            Ok(Outcome::Refused(Refusal {
                version,
                reason,
                shortfall,
            }))
        };
        let profile: PlacementProfile = match serde_json::from_value(candidate.profile) {
            Ok(profile) => profile,
            Err(error) => return refuse(format!("the profile does not parse: {error}"), Vec::new()),
        };
        if !profile.engines.is_empty() && profile.store.is_none() {
            return refuse(
                "the profile has engines and no `store.nvme` budget; add a [store] table".to_owned(),
                Vec::new(),
            );
        }
        if profile == self.active {
            // The ledger already holds this profile. Another admission would not match the running engines.
            self.write_version(version)?;
            return Ok(Outcome::Unchanged);
        }
        match process.admit(profile.clone()) {
            Ok(_) => {}
            Err(AdmitError::Ledger(LedgerError::Refused(refusal))) => {
                let shortfall = refusal
                    .shortfall
                    .into_iter()
                    .map(|(tiers, bytes)| Shortfall {
                        tiers: tiers.into_iter().map(TierSpec::from).collect(),
                        bytes,
                    })
                    .collect();
                return refuse("the profile does not fit its budgets".to_owned(), shortfall);
            }
            Err(error) => return refuse(error.to_string(), Vec::new()),
        }
        profile.write(&self.store_dir).map_err(SwapError::Profile)?;
        self.write_version(version)?;
        self.active = profile;
        Ok(Outcome::Restart)
    }

    /// Write the handled version atomically: a temporary file, then a rename.
    fn write_version(&self, version: u64) -> Result<(), SwapError> {
        let path = self.store_dir.join(VERSION_FILE);
        let temporary = self.store_dir.join(format!("{VERSION_FILE}.tmp"));
        let failed = |source: std::io::Error| SwapError::Version {
            path: path.clone(),
            message: source.to_string(),
        };
        let mut file = std::fs::File::create(&temporary).map_err(failed)?;
        writeln!(file, "{version}").map_err(failed)?;
        file.sync_all().map_err(failed)?;
        drop(file);
        std::fs::rename(&temporary, &path).map_err(failed)?;
        std::fs::File::open(&self.store_dir)
            .and_then(|directory| directory.sync_all())
            .map_err(failed)
    }
}
