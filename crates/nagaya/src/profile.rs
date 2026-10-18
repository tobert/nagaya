//! The placement profile (spec 2.4).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use nagaya_abi::{DeviceId, Tier};
use serde::{Deserialize, Serialize};

/// The name of the accepted profile file in the store directory.
pub const PROFILE_FILE: &str = "profile.toml";

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
    /// The engine name.
    pub name: String,
    /// The engine kind, which selects the factory.
    pub kind: String,
    /// The devices of the engine's device group.
    pub devices: Vec<DeviceId>,
    /// The engine's own config (ABI "Rust shape").
    #[serde(default)]
    pub config: toml::Table,
    /// The engine's share of the spare memory (D9).
    pub share: u64,
    /// True when the profile pins the engine's identity (spec 12.1).
    #[serde(default)]
    pub pin_identity: bool,
}

/// One budget in the profile (spec 4.1).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpec {
    /// The tiers that share this budget.
    pub tiers: Vec<TierSpec>,
    /// The byte target.
    pub bytes: u64,
}

/// A machine's placement profile (spec 2.4).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementProfile {
    /// The engines.
    pub engines: Vec<EnginePlacement>,
    /// The budgets.
    pub budgets: Vec<BudgetSpec>,
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
