//! Held specs: content-addressed, and kept in the store directory (gogi "Specs", spec 11.1).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use gogi_proto::SpecId;
use gogi_proto::spec::{HeldSpec, SpecRules, parse_spec};
use gogi_proto::spec_id_of;
use serde_json::Value;

use super::error::ServiceError;

/// The directory under the store directory that holds the specs.
pub const SPEC_DIR: &str = "specs";

/// The specs that the service holds.
///
/// Each spec is a file `specs/<spec id>.json`, such as `specs/blake3:5d61….json`.
#[derive(Debug)]
pub struct Specs {
    dir: PathBuf,
    rules: SpecRules,
    held: BTreeMap<String, Value>,
}

fn file_name(id: &str) -> String {
    format!("{id}.json")
}

/// Whether a file name is the form of a spec file before gogi's BLAKE3 ids: the 64 hex digits of a SHA-256 id, with
/// no scheme.
fn is_old_name(name: &str) -> bool {
    name.strip_suffix(".json")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

fn io_error(what: &str, path: &Path, error: std::io::Error) -> ServiceError {
    ServiceError::internal(format!("cannot {what} the spec file `{}`: {error}", path.display()))
}

impl Specs {
    /// Open the specs under a store directory, and read every file. A file that the rules refuse, or whose name is
    /// not the id of its content, is an error: the open stops (spec 8.5). A file with the bare hex name of an older
    /// nagaya takes the name of the spec id of its content.
    pub fn open(store_dir: &Path, rules: SpecRules) -> Result<Self, ServiceError> {
        let dir = store_dir.join(SPEC_DIR);
        std::fs::create_dir_all(&dir).map_err(|error| io_error("make", &dir, error))?;
        let mut held = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).map_err(|error| io_error("list", &dir, error))? {
            let path = entry.map_err(|error| io_error("list", &dir, error))?.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()).map(str::to_owned) else {
                return Err(ServiceError::internal(format!(
                    "the specs directory holds a file with a name that is not text: `{}`",
                    path.display()
                )));
            };
            if name.ends_with(".tmp") {
                // A write that stopped before its rename. The spec was never held.
                std::fs::remove_file(&path).map_err(|error| io_error("remove", &path, error))?;
                continue;
            }
            let bytes = std::fs::read(&path).map_err(|error| io_error("read", &path, error))?;
            let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
                ServiceError::internal(format!(
                    "the spec file `{}` in the specs directory is not JSON: {error}. Remove the file",
                    path.display()
                ))
            })?;
            let id = spec_id_of(&value).map_err(|error| {
                ServiceError::internal(format!(
                    "the spec file `{}` in the specs directory: {error}",
                    path.display()
                ))
            })?;
            if file_name(&id) != name && is_old_name(&name) {
                // The content names the spec, so the file takes the name of its spec id.
                let new = dir.join(file_name(&id));
                std::fs::rename(&path, &new).map_err(|error| io_error("rename", &path, error))?;
            } else if file_name(&id) != name {
                return Err(ServiceError::internal(format!(
                    "the spec file `{}` in the specs directory holds the spec {id}, not the spec its name says. \
                     Remove the file",
                    path.display()
                )));
            }
            parse_spec(&value, &rules).map_err(|invalid| {
                ServiceError::internal(format!(
                    "the spec file `{}` in the specs directory breaks the rules: {}. Remove the file",
                    path.display(),
                    invalid.message
                ))
            })?;
            held.insert(id, value);
        }
        Ok(Self { dir, rules, held })
    }

    /// Hold a spec. The same spec gets the same id. A spec that breaks the rules is held by no one.
    pub fn hold(&mut self, value: Value) -> Result<HeldSpec, ServiceError> {
        parse_spec(&value, &self.rules)?;
        let id = spec_id_of(&value).map_err(|message| ServiceError::Invalid(gogi_proto::Invalid::new(message)))?;
        if !self.held.contains_key(&id) {
            self.write(&id, &value)?;
            self.held.insert(id.clone(), value.clone());
        }
        Ok(HeldSpec {
            spec_id: SpecId::parse(id).map_err(ServiceError::internal)?,
            spec: value,
        })
    }

    /// Write a spec file. The write goes to a temporary file first, so a crash leaves no half file.
    fn write(&self, id: &str, value: &Value) -> Result<(), ServiceError> {
        let path = self.dir.join(file_name(id));
        let temporary = self.dir.join(format!("{}.tmp", file_name(id)));
        let bytes = serde_json::to_vec(value).map_err(|error| ServiceError::internal(error.to_string()))?;
        let mut file = std::fs::File::create(&temporary).map_err(|error| io_error("make", &temporary, error))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| io_error("write", &temporary, error))?;
        std::fs::rename(&temporary, &path).map_err(|error| io_error("rename", &path, error))
    }

    /// The held spec with this id.
    pub fn get(&self, id: &str) -> Result<HeldSpec, ServiceError> {
        let value = self
            .held
            .get(id)
            .ok_or_else(|| ServiceError::NotFound(format!("no spec {id}: post the spec first")))?;
        Ok(HeldSpec {
            spec_id: SpecId::parse(id).map_err(ServiceError::internal)?,
            spec: value.clone(),
        })
    }

    /// The held specs, in order of id.
    pub fn list(&self) -> Vec<HeldSpec> {
        self.held
            .iter()
            .map(|(id, value)| HeldSpec {
                spec_id: SpecId::parse(id.clone()).expect("a held spec has a valid id"),
                spec: value.clone(),
            })
            .collect()
    }

    /// Drop a spec, and remove its file.
    pub fn drop_spec(&mut self, id: &str) -> Result<(), ServiceError> {
        if !self.held.contains_key(id) {
            return Err(ServiceError::NotFound(format!("no spec {id}: it is not held")));
        }
        let path = self.dir.join(file_name(id));
        std::fs::remove_file(&path).map_err(|error| io_error("remove", &path, error))?;
        self.held.remove(id);
        Ok(())
    }
}
