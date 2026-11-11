//! The pins of council owners (gogi "Snapshots and pins", spec 6.10, 11.3c).
//!
//! gogi names an owner with text, and the store names an owner with a `u64`. The book keeps the text and the
//! snapshots of each owner. The store keeps the pins themselves, so eviction sees them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nagaya_store::ContentId;
use serde::{Deserialize, Serialize};

use super::error::ServiceError;

/// The file in the store directory that holds the owners of the pins.
pub const PINS_FILE: &str = "pins.json";

/// The owner key that the store gets for an owner name: the first 8 bytes of its BLAKE3 hash, little-endian.
pub fn owner_key(owner: &str) -> u64 {
    let hash = blake3::hash(owner.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

/// What one owner pins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerPins {
    /// The content ids that the owner pinned, which the store held at the time.
    pub snapshots: Vec<ContentId>,
    /// The ids of the context records that the owner pinned, which the server held at the time (spec 6.17). A book
    /// of an older nagaya has none.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// The end of the pins, in milliseconds of the service clock.
    pub expires_ms: u64,
}

/// The owners of the pins.
#[derive(Debug)]
pub struct PinBook {
    path: PathBuf,
    owners: BTreeMap<String, OwnerPins>,
}

impl PinBook {
    /// Read the book from the store directory. A missing file is an empty book.
    pub fn open(store_dir: &Path) -> Result<Self, ServiceError> {
        let path = store_dir.join(PINS_FILE);
        let owners = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| {
                ServiceError::internal(format!(
                    "the file `{}` is not a book of pins: {error}. Remove the file",
                    path.display()
                ))
            })?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(error) => {
                return Err(ServiceError::internal(format!(
                    "cannot read `{}`: {error}",
                    path.display()
                )));
            }
        };
        Ok(Self { path, owners })
    }

    /// Write the book through a temporary file and a rename.
    pub fn save(&self) -> Result<(), ServiceError> {
        let temporary = self.path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(&self.owners).map_err(|error| ServiceError::internal(error.to_string()))?;
        std::fs::write(&temporary, bytes)
            .and_then(|()| std::fs::rename(&temporary, &self.path))
            .map_err(|error| ServiceError::internal(format!("cannot write `{}`: {error}", self.path.display())))
    }

    /// The pins of an owner.
    pub fn get(&self, owner: &str) -> Option<&OwnerPins> {
        self.owners.get(owner)
    }

    /// Set the pins of an owner, or remove them when the set is empty.
    pub fn set(&mut self, owner: &str, pins: Option<OwnerPins>) {
        match pins {
            Some(pins) => {
                self.owners.insert(owner.to_owned(), pins);
            }
            None => {
                self.owners.remove(owner);
            }
        }
    }

    /// Every owner and its pins.
    pub fn owners(&self) -> &BTreeMap<String, OwnerPins> {
        &self.owners
    }

    /// The ids of the context records that a pin keeps at the time `now_ms`. A pin that ended keeps nothing.
    pub fn records(&self, now_ms: u64) -> std::collections::BTreeSet<String> {
        self.owners
            .values()
            .filter(|pins| pins.expires_ms > now_ms)
            .flat_map(|pins| pins.contexts.iter().cloned())
            .collect()
    }

    /// The name of the owner that has this key, when the book knows it.
    pub fn name_of(&self, key: u64) -> Option<&str> {
        self.owners
            .keys()
            .find(|owner| owner_key(owner) == key)
            .map(String::as_str)
    }
}
