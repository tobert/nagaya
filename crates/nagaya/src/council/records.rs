//! Context records: the contexts that the server read, each under its context id (gogi "Context records", spec 6.17).
//!
//! A record is the file `contexts/<context id>.json` in the store directory. The book keeps the ids in memory, in
//! the order of their last name, and it reads the text of a record from its file.
//!
//! The record of a reply can have a head note: the file `context-heads/<context id>.json`, which names the decoded
//! head of the reply (spec 6.14, 6.17).

use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use gogi_proto::{Context, ContextId, context_id, record_text};
use indexmap::IndexSet;
use nagaya_store::RecipeId;
use serde::{Deserialize, Serialize};

use super::error::ServiceError;

/// The directory under the store directory that holds the records.
pub const RECORD_DIR: &str = "contexts";

/// The directory under the store directory that holds the head notes.
pub const HEAD_DIR: &str = "context-heads";

/// The most records that the server holds: `limits.context_records` (spec 6.17).
pub const CONTEXT_RECORDS: usize = 10_000;

/// A seal of the state of a decoded head, as its note keeps it (spec 6.17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealNote {
    /// The tokens of the state at the seal.
    pub tokens: usize,
    /// For a decoded head, the tokens of its prompt. A boundary has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<usize>,
}

/// The note of the decoded head of a reply (spec 6.14, 6.17). A build computes the recipe again from the seals and
/// its own tokens, so it trusts no token of the note.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadNote {
    /// The recipe of the head.
    pub recipe: RecipeId,
    /// The tokens of the head.
    pub tokens: usize,
    /// Each seal of the state of the head, in order. The last one is the head.
    pub seals: Vec<SealNote>,
}

/// The records that the server holds.
#[derive(Debug)]
pub struct Records {
    dir: PathBuf,
    /// The directory of the head notes.
    heads: PathBuf,
    limit: usize,
    /// The ids of the held records. The least recently named record is first.
    held: IndexSet<String>,
}

fn file_name(id: &str) -> String {
    format!("{id}.json")
}

fn io_error(what: &str, path: &Path, error: std::io::Error) -> ServiceError {
    ServiceError::internal(format!("cannot {what} the record file `{}`: {error}", path.display()))
}

impl Records {
    /// Open the records under a store directory, with the limit of the server.
    pub fn open(store_dir: &Path) -> Result<Self, ServiceError> {
        Self::open_with(store_dir, CONTEXT_RECORDS)
    }

    /// Open the records under a store directory. The book holds at most `limit` records that no pin keeps.
    ///
    /// The open lists the files and reads none of them: a read checks its record (spec 6.17). The records take the
    /// order of their file times, the oldest first. A file whose name is not a record name is an error.
    pub fn open_with(store_dir: &Path, limit: usize) -> Result<Self, ServiceError> {
        let dir = store_dir.join(RECORD_DIR);
        std::fs::create_dir_all(&dir).map_err(|error| io_error("make the directory of", &dir, error))?;
        let mut found: Vec<(SystemTime, String)> = Vec::new();
        for entry in std::fs::read_dir(&dir).map_err(|error| io_error("list", &dir, error))? {
            let entry = entry.map_err(|error| io_error("list", &dir, error))?;
            let path = entry.path();
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            if name.ends_with(".tmp") {
                // A write that stopped before its rename. The record was never held.
                std::fs::remove_file(&path).map_err(|error| io_error("remove", &path, error))?;
                continue;
            }
            let id = name
                .strip_suffix(".json")
                .filter(|id| gogi_proto::content_id_problem(id).is_none())
                .ok_or_else(|| {
                    ServiceError::internal(format!(
                        "the file `{}` in the records directory does not have the name of a context record, \
                         `<context id>.json`. Remove the file",
                        path.display()
                    ))
                })?;
            let written = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .map_err(|error| io_error("read the time of", &path, error))?;
            found.push((written, id.to_owned()));
        }
        found.sort();
        let held: IndexSet<String> = found.into_iter().map(|(_, id)| id).collect();
        let heads = store_dir.join(HEAD_DIR);
        std::fs::create_dir_all(&heads).map_err(|error| io_error("make the directory of", &heads, error))?;
        for entry in std::fs::read_dir(&heads).map_err(|error| io_error("list", &heads, error))? {
            let path = entry.map_err(|error| io_error("list", &heads, error))?.path();
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            let id = name.strip_suffix(".json");
            // A write that stopped before its rename, or the note of a record that the book does not hold.
            if name.ends_with(".tmp") || id.is_some_and(|id| !held.contains(id) && content_id_name(id)) {
                std::fs::remove_file(&path).map_err(|error| io_error("remove", &path, error))?;
                continue;
            }
            if !id.is_some_and(content_id_name) {
                return Err(ServiceError::internal(format!(
                    "the file `{}` in the directory of the head notes does not have the name of a head note, \
                     `<context id>.json`. Remove the file",
                    path.display()
                )));
            }
        }
        Ok(Self {
            dir,
            heads,
            limit,
            held,
        })
    }

    /// Keep the note of the decoded head of a held record (spec 6.17). A second note takes the place of the first.
    /// A record that the book does not hold is a defect of the caller.
    pub fn note_head(&mut self, id: &str, note: &HeadNote) -> Result<(), ServiceError> {
        if !self.held.contains(id) {
            return Err(ServiceError::internal(format!(
                "the context record {id} is not held, so it takes no head note"
            )));
        }
        let bytes = serde_json::to_vec(note)
            .map_err(|error| ServiceError::internal(format!("cannot write the head note of {id}: {error}")))?;
        write_file(&self.heads, id, &bytes)
    }

    /// The note of the decoded head of a record, or `None` when the record has none. The read does not name the
    /// record.
    ///
    /// A file that is not a note is removed with a warning, and the read is a miss (spec 6.17).
    pub fn head(&mut self, id: &str) -> Result<Option<HeadNote>, ServiceError> {
        if !self.held.contains(id) {
            return Ok(None);
        }
        let path = self.heads.join(file_name(id));
        let problem = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<HeadNote>(&bytes) {
                Ok(note) => return Ok(Some(note)),
                Err(error) => error.to_string(),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error("read", &path, error)),
        };
        tracing::warn!(
            "the head note of the context record {id} is dropped: its text is not a note: {problem}. The next build \
             of that context feeds the reply again"
        );
        remove_file(&path)?;
        Ok(None)
    }

    /// The most records that the book holds when no pin keeps one.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The count of the held records.
    pub fn len(&self) -> usize {
        self.held.len()
    }

    /// True when the book holds no record.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    /// True when the book holds a record with this id. This does not name the record.
    pub fn holds(&self, id: &str) -> bool {
        self.held.contains(id)
    }

    /// Hold the record of a context, and give its id. The hold names the record.
    ///
    /// Past the limit, the book drops the least recently named records. It never drops a record of `pinned`, and it
    /// never drops the record of this hold.
    pub fn hold(&mut self, context: &Context, pinned: &BTreeSet<String>) -> Result<ContextId, ServiceError> {
        let id = context_id(context);
        if self.held.shift_remove(id.as_str()) {
            self.held.insert(id.to_string());
            return Ok(id);
        }
        self.write(id.as_str(), record_text(context).as_bytes())?;
        self.held.insert(id.to_string());
        let over = self.held.len().saturating_sub(self.limit);
        let dropped: Vec<String> = self
            .held
            .iter()
            .filter(|held| held.as_str() != id.as_str() && !pinned.contains(*held))
            .take(over)
            .cloned()
            .collect();
        for old in dropped {
            self.remove(&old)?;
        }
        Ok(id)
    }

    /// The context of a held record, or `None` when the book does not hold the id. The read names the record.
    ///
    /// The read checks the file: its text must be a context whose id is the name. A file that fails is removed with
    /// a warning, and the read is a miss (spec 6.17).
    pub fn get(&mut self, id: &str) -> Result<Option<Context>, ServiceError> {
        if !self.held.contains(id) {
            return Ok(None);
        }
        let path = self.dir.join(file_name(id));
        let problem = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<Context>(&bytes) {
                Ok(context) if context_id(&context).as_str() == id => {
                    self.held.shift_remove(id);
                    self.held.insert(id.to_owned());
                    return Ok(Some(context));
                }
                Ok(context) => format!("its text has the context id {}", context_id(&context)),
                Err(error) => format!("its text is not a context: {error}"),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => "the file is gone".to_owned(),
            Err(error) => return Err(io_error("read", &path, error)),
        };
        tracing::warn!(
            "the context record {id} is dropped: {problem}. A client that names it gets a 404 and sends the context \
             whole"
        );
        self.remove(id)?;
        Ok(None)
    }

    /// Write a record file. The write goes to a temporary file first, so a crash leaves no half file.
    fn write(&self, id: &str, bytes: &[u8]) -> Result<(), ServiceError> {
        write_file(&self.dir, id, bytes)
    }

    /// Drop a record, and remove its file and its head note. A file that is gone is not an error.
    fn remove(&mut self, id: &str) -> Result<(), ServiceError> {
        self.held.shift_remove(id);
        remove_file(&self.heads.join(file_name(id)))?;
        remove_file(&self.dir.join(file_name(id)))
    }
}

/// True when `id` has the form of a context id.
fn content_id_name(id: &str) -> bool {
    gogi_proto::content_id_problem(id).is_none()
}

/// Write the file of `id` in `dir`. The write goes to a temporary file first, so a crash leaves no half file.
fn write_file(dir: &Path, id: &str, bytes: &[u8]) -> Result<(), ServiceError> {
    let path = dir.join(file_name(id));
    let temporary = dir.join(format!("{}.tmp", file_name(id)));
    let mut file = std::fs::File::create(&temporary).map_err(|error| io_error("make", &temporary, error))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| io_error("write", &temporary, error))?;
    std::fs::rename(&temporary, &path).map_err(|error| io_error("rename", &path, error))
}

/// Remove a file. A file that is gone is not an error.
fn remove_file(path: &Path) -> Result<(), ServiceError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error("remove", path, error)),
    }
}

/// True when a context of a request body names a `parent`: `context`, or a context of `contexts`, `positive` or
/// `negative` (gogi `resolve_request`). A body with none goes to its route as its bytes came (spec 11.13).
pub fn names_parent(body: &serde_json::Value) -> bool {
    let one = body.get("context").into_iter();
    let lists = ["contexts", "positive", "negative"]
        .into_iter()
        .filter_map(|list| body.get(list)?.as_array())
        .flatten();
    one.chain(lists).any(|context| context.get("parent").is_some())
}
