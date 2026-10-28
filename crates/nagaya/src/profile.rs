//! The placement profile (spec 2.4).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use nagaya_abi::{DeviceId, Tier};
use serde::{Deserialize, Serialize};

/// The name of the accepted profile file in the store directory.
pub const PROFILE_FILE: &str = "profile.toml";

/// The longest engine name in bytes. The gossip key `engine:<name>` stays within `MAX_KEY_BYTES`.
pub const MAX_ENGINE_NAME_BYTES: usize = 128;

const _: () = assert!("engine:".len() + MAX_ENGINE_NAME_BYTES <= crate::gossip::MAX_KEY_BYTES);

/// Check an engine name when a profile is read. The name goes into a gossip key, so it has the characters of a node
/// name (spec 2.1 note) and a length bound.
fn engine_name<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let name = String::deserialize(deserializer)?;
    check_engine_name(&name).map_err(serde::de::Error::custom)?;
    Ok(name)
}

fn check_engine_name(name: &str) -> Result<(), String> {
    if name.len() > MAX_ENGINE_NAME_BYTES {
        return Err(format!(
            "engine name `{name}` has {} bytes; the bound is {MAX_ENGINE_NAME_BYTES} bytes",
            name.len()
        ));
    }
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "engine name `{name}` is not valid; use letters, digits, `.`, `_` and `-`, and at least one character"
        ))
    }
}

/// A tier as the profile writes it: `host` or `device(i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierSpec(pub Tier);

impl From<Tier> for TierSpec {
    fn from(tier: Tier) -> Self {
        TierSpec(tier)
    }
}

impl From<TierSpec> for Tier {
    fn from(spec: TierSpec) -> Self {
        spec.0
    }
}

impl std::fmt::Display for TierSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Tier::Host => write!(formatter, "host"),
            Tier::Device(id) => write!(formatter, "device({id})"),
        }
    }
}

impl std::str::FromStr for TierSpec {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text == "host" {
            return Ok(TierSpec(Tier::Host));
        }
        let index = text
            .strip_prefix("device(")
            .and_then(|rest| rest.strip_suffix(')'))
            .ok_or_else(|| format!("tier `{text}` is not valid; write `host` or `device(<n>)`"))?;
        let id = index
            .parse::<DeviceId>()
            .map_err(|_| format!("tier `{text}` is not valid; write `host` or `device(<n>)`"))?;
        Ok(TierSpec(Tier::Device(id)))
    }
}

impl Serialize for TierSpec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TierSpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// One engine in the profile.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnginePlacement {
    /// The engine name. It has the characters of a node name and at most `MAX_ENGINE_NAME_BYTES` bytes.
    #[serde(deserialize_with = "engine_name")]
    pub name: String,
    /// The engine kind, which selects the factory.
    pub kind: String,
    /// The devices of the engine's device group.
    pub devices: Vec<DeviceId>,
    /// The engine's own config (ABI "Rust shape").
    #[serde(default)]
    pub config: toml::Table,
    /// The files of the engine's family library (spec 11.3, "The engine registry"). The family opens from this
    /// table and from the engine identity.
    #[serde(default)]
    pub family: toml::Table,
    /// The engine's share of the spare memory (D9).
    pub share: u64,
    /// True when the profile pins the engine's identity (spec 12.1).
    #[serde(default)]
    pub pin_identity: bool,
    /// When the engine is loaded (spec 3.3a). The default is `resident`, and a resident entry writes no key.
    #[serde(default, skip_serializing_if = "Residency::is_resident")]
    pub residency: Residency,
    /// The seconds with no job after which an on-demand engine unloads (spec 3.3a). Only an on-demand entry has it;
    /// without it, the time is [`DEFAULT_IDLE_UNLOAD_S`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_unload_s: Option<u64>,
}

/// When an engine is loaded (spec 3.3a).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    /// The engine opens at start. It unloads only while a swap gives its memory to another engine.
    #[default]
    Resident,
    /// The engine opens at a load (spec 3.3b), and it unloads after its idle time.
    OnDemand,
}

impl Residency {
    /// True for `resident`, the default.
    pub fn is_resident(&self) -> bool {
        *self == Residency::Resident
    }
}

/// The seconds with no job after which an on-demand engine unloads, when its entry names none (spec 3.3a).
pub const DEFAULT_IDLE_UNLOAD_S: u64 = 60;

/// One budget in the profile (spec 4.1).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpec {
    /// The tiers that share this budget.
    pub tiers: Vec<TierSpec>,
    /// The byte target.
    pub bytes: u64,
}

/// The store's budgets in the profile (spec 4.1 rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoreSpec {
    /// The NVMe budget in bytes.
    pub nvme: u64,
    /// The host-memory budget in bytes. Without it, the budget is the unassigned host bytes (spec 4.4 step 4).
    #[serde(default)]
    pub host: Option<u64>,
    /// The most interval copies that the store keeps in host memory (spec 6.13). Zero, the default, means the
    /// store refuses them.
    #[serde(default)]
    pub interval_copies: u32,
    /// The budget of objects in bytes, apart from `nvme` (spec 6.15). Without it, the process holds no objects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objects: Option<u64>,
}

/// A machine's placement profile (spec 2.4).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementProfile {
    /// The engines.
    pub engines: Vec<EnginePlacement>,
    /// The budgets.
    pub budgets: Vec<BudgetSpec>,
    /// The store's budgets. A profile with engines must name them (spec 4.1 rule).
    #[serde(default)]
    pub store: Option<StoreSpec>,
}

/// A profile that cannot be read, parsed or written.
#[derive(Debug)]
pub enum ProfileError {
    /// The file could not be read.
    Read {
        /// The file path.
        path: PathBuf,
        /// The read error.
        source: std::io::Error,
    },
    /// The text is not a valid profile.
    Parse {
        /// The parser's message.
        message: String,
    },
    /// The profile cannot be written as TOML.
    Serialize {
        /// The serializer's message.
        message: String,
    },
    /// The file could not be written.
    Write {
        /// The file path.
        path: PathBuf,
        /// The write error.
        source: std::io::Error,
    },
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::Read { path, source } => {
                write!(
                    formatter,
                    "cannot read the placement profile `{}`: {source}",
                    path.display()
                )
            }
            ProfileError::Parse { message } => {
                write!(formatter, "the placement profile is not valid: {message}")
            }
            ProfileError::Serialize { message } => {
                write!(formatter, "cannot make the placement profile text: {message}")
            }
            ProfileError::Write { path, source } => {
                write!(
                    formatter,
                    "cannot write the placement profile `{}`: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for ProfileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProfileError::Read { source, .. } | ProfileError::Write { source, .. } => Some(source),
            ProfileError::Parse { .. } | ProfileError::Serialize { .. } => None,
        }
    }
}

impl PlacementProfile {
    /// Parse a profile from TOML text.
    pub fn parse(text: &str) -> Result<Self, ProfileError> {
        toml::from_str(text).map_err(|error| ProfileError::Parse {
            message: error.to_string(),
        })
    }

    /// Read a profile from a TOML file.
    pub fn read(path: &Path) -> Result<Self, ProfileError> {
        let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Read {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Read the accepted profile of a machine from its store directory.
    pub fn read_active(store_dir: &Path) -> Result<Self, ProfileError> {
        Self::read(&store_dir.join(PROFILE_FILE))
    }

    /// Write this profile as the accepted profile of a machine, atomically: a temporary file, then a rename.
    pub fn write(&self, store_dir: &Path) -> Result<PathBuf, ProfileError> {
        let path = store_dir.join(PROFILE_FILE);
        let temporary = store_dir.join(format!("{PROFILE_FILE}.tmp"));
        let text = self.to_toml()?;
        let write_error = |source| ProfileError::Write {
            path: temporary.clone(),
            source,
        };
        let mut file = std::fs::File::create(&temporary).map_err(write_error)?;
        file.write_all(text.as_bytes()).map_err(write_error)?;
        file.sync_all().map_err(write_error)?;
        drop(file);
        std::fs::rename(&temporary, &path).map_err(|source| ProfileError::Write {
            path: path.clone(),
            source,
        })?;
        std::fs::File::open(store_dir)
            .and_then(|directory| directory.sync_all())
            .map_err(|source| ProfileError::Write {
                path: path.clone(),
                source,
            })?;
        Ok(path)
    }

    /// Make the profile's TOML text.
    pub fn to_toml(&self) -> Result<String, ProfileError> {
        toml::to_string(self).map_err(|error| ProfileError::Serialize {
            message: error.to_string(),
        })
    }
}
