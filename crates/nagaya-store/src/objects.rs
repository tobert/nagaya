//! Objects: bytes and a media type that a client or an engine gives to the store, named by the content id of the
//! bytes (spec 6.15, gogi "Objects and keeps").
//!
//! Objects have their own budget, apart from the snapshots. Half of it is for keeps, so a put always fits after the
//! store evicts cached objects. The store evicts the cached object that was used least recently first. Each object is
//! a file `<dir>/<the first 2 hex digits>/<the other 62>`, and the index `<dir>/index.json` holds the media types, the
//! times of use and the keeps. The store has no clock: each call takes `now_ms` from the caller.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ContentId;

/// The largest object: 8 GiB (spec 6.15), or half the budget when that is less.
pub const OBJECT_BYTES_MAX: u64 = 8 << 30;

/// The name of the index file.
const INDEX: &str = "index.json";

/// The directory of bytes that are not held yet.
const STAGING: &str = "staging";

/// An error of the object store.
#[derive(Debug, PartialEq, Eq)]
pub enum ObjectError {
    /// A put or a keep set is past a limit, named as gogi names it (`object_bytes` or `keep_bytes`).
    TooLarge {
        /// The limit.
        limit: &'static str,
        /// The bytes that the request asks for.
        bytes: u64,
        /// The bytes that the limit allows.
        max: u64,
    },
    /// The file system failed.
    Io(String),
}

impl std::fmt::Display for ObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectError::TooLarge { limit, bytes, max } => {
                write!(f, "nothing is held: {bytes} bytes are past {limit}, which is {max}")
            }
            ObjectError::Io(message) => f.write_str(message),
        }
    }
}

fn io(what: &str, path: &Path, error: std::io::Error) -> ObjectError {
    ObjectError::Io(format!("cannot {what} `{}`: {error}", path.display()))
}

fn is_hex_id(hex: &str) -> bool {
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// An object that the store holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Held {
    mime: String,
    bytes: u64,
    used_ms: u64,
}

/// The set of objects of one owner, until an expiry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Keep {
    objects: Vec<String>,
    expires_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Index {
    /// Hex of a content id to the object.
    objects: BTreeMap<String, Held>,
    /// An owner to its keep.
    keeps: BTreeMap<String, Keep>,
}

/// The answer of a commit: whether the object is new, and the media type that the store holds for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committed {
    /// `false` when the store held the object before: a put of it is a 200, not a 201.
    pub created: bool,
    /// The media type of the first put.
    pub mime: String,
}

/// A held object: its file, media type and size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    /// The file that holds the bytes.
    pub path: PathBuf,
    /// The media type of its first put.
    pub mime: String,
    /// Its size.
    pub bytes: u64,
}

/// The objects of one machine and their keeps.
#[derive(Debug)]
pub struct ObjectStore {
    dir: PathBuf,
    budget: u64,
    index: Index,
    next_stage: u64,
    /// The bytes that the held objects take.
    used: u64,
    /// The objects whose file this process hashed: each one that it committed, and each one that a read checked.
    verified: BTreeSet<String>,
}

impl ObjectStore {
    /// Open the objects under `dir` with a budget in bytes.
    ///
    /// A file that is missing, or whose size is not the size in the index, drops its object, as an eviction does: a
    /// client puts it again. A keep forgets a dropped object, and keeps past `keep_bytes` drop, the one that expires
    /// first first. An index that does not read moves aside, and the store starts empty. Staged bytes from a put that
    /// stopped, and files that the index does not name, are removed. Objects are a cache of what clients sent, so each
    /// of these costs a put, never a wrong answer.
    pub fn open(dir: &Path, budget: u64) -> Result<Self, ObjectError> {
        std::fs::create_dir_all(dir.join(STAGING)).map_err(|e| io("make", dir, e))?;
        let path = dir.join(INDEX);
        let mut index: Index = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(index) => index,
                Err(e) => {
                    let aside = dir.join(format!("{INDEX}.damaged"));
                    tracing::error!(
                        "the object index `{}` is not an index ({e}); it moves to `{}`, and the store starts empty",
                        path.display(),
                        aside.display()
                    );
                    std::fs::rename(&path, &aside).map_err(|e| io("move", &path, e))?;
                    Index::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Index::default(),
            Err(e) => return Err(io("read", &path, e)),
        };
        index.objects.retain(|hex, held| {
            let whole = is_hex_id(hex) && std::fs::metadata(object_path(dir, hex)).is_ok_and(|m| m.len() == held.bytes);
            if !whole {
                tracing::warn!(object = %hex, "an object file is missing or has the wrong size; the object is dropped");
            }
            whole
        });
        let held: BTreeSet<String> = index.objects.keys().cloned().collect();
        for keep in index.keeps.values_mut() {
            keep.objects.retain(|hex| held.contains(hex));
        }
        index.keeps.retain(|_, keep| !keep.objects.is_empty());
        let used = index.objects.values().map(|held| held.bytes).sum();
        let mut store = Self {
            dir: dir.to_path_buf(),
            budget,
            index,
            next_stage: 0,
            used,
            verified: BTreeSet::new(),
        };
        store.clamp_keeps();
        store.sweep()?;
        store.save()?;
        Ok(store)
    }

    /// The largest object: [`OBJECT_BYTES_MAX`], or half the budget when that is less.
    pub fn object_bytes(&self) -> u64 {
        OBJECT_BYTES_MAX.min(self.budget / 2)
    }

    /// The most bytes that all keeps together hold: half the budget.
    pub fn keep_bytes(&self) -> u64 {
        self.budget / 2
    }

    /// A new path for the bytes of a put. The caller writes and hashes them there, then commits or removes them.
    pub fn staging_path(&mut self) -> PathBuf {
        self.next_stage += 1;
        self.dir
            .join(STAGING)
            .join(format!("{}-{}.tmp", std::process::id(), self.next_stage))
    }

    /// Hold the staged bytes under `id`. The caller hashed them, so their content id is `id`. A held object keeps the
    /// media type of its first put. The staged bytes go in every case.
    pub fn commit(
        &mut self,
        id: &ContentId,
        mime: &str,
        staged: &Path,
        bytes: u64,
        now_ms: u64,
    ) -> Result<Committed, ObjectError> {
        let hex = id.hex();
        if let Some(held) = self.index.objects.get_mut(&hex) {
            held.used_ms = now_ms;
            let mime = held.mime.clone();
            std::fs::remove_file(staged).map_err(|e| io("remove", staged, e))?;
            return Ok(Committed { created: false, mime });
        }
        let held = self.hold(&hex, mime, staged, bytes, now_ms);
        if held.is_err() {
            let _ = std::fs::remove_file(staged);
        }
        held?;
        Ok(Committed {
            created: true,
            mime: mime.to_string(),
        })
    }

    fn hold(&mut self, hex: &str, mime: &str, staged: &Path, bytes: u64, now_ms: u64) -> Result<(), ObjectError> {
        if bytes > self.object_bytes() {
            return Err(ObjectError::TooLarge {
                limit: "object_bytes",
                bytes,
                max: self.object_bytes(),
            });
        }
        self.evict_for(bytes)?;
        let file = object_path(&self.dir, hex);
        let parent = file.parent().expect("an object path has a directory");
        std::fs::create_dir_all(parent).map_err(|e| io("make", parent, e))?;
        std::fs::rename(staged, &file).map_err(|e| io("rename", staged, e))?;
        crate::sync_dir(parent).map_err(|e| io("sync", parent, e))?;
        self.index.objects.insert(
            hex.to_string(),
            Held {
                mime: mime.to_string(),
                bytes,
                used_ms: now_ms,
            },
        );
        self.used += bytes;
        self.verified.insert(hex.to_string());
        self.save()
    }

    /// The held object `id`, which counts as a use. The first read after an open hashes the file: a file that does not
    /// hash to its id drops its object, as an eviction does, so the store never serves bytes under another id (spec
    /// 7.3: "same size is not same bytes").
    pub fn get(&mut self, id: &ContentId, now_ms: u64) -> Result<Option<Found>, ObjectError> {
        let hex = id.hex();
        let Some(held) = self.index.objects.get(&hex).cloned() else {
            return Ok(None);
        };
        let path = object_path(&self.dir, &hex);
        if !self.verified.contains(&hex) {
            let whole = match std::fs::File::open(&path) {
                Ok(file) => {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update_reader(file).map_err(|e| io("read", &path, e))?;
                    hasher.finalize().to_hex().as_str() == hex
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
                Err(e) => return Err(io("open", &path, e)),
            };
            if !whole {
                tracing::warn!(object = %hex, "an object file does not hash to its id; the object is dropped");
                self.forget(&hex)?;
                return Ok(None);
            }
            self.verified.insert(hex.clone());
        }
        if let Some(held) = self.index.objects.get_mut(&hex) {
            held.used_ms = now_ms;
        }
        Ok(Some(Found {
            path,
            mime: held.mime,
            bytes: held.bytes,
        }))
    }

    /// The bytes of the held object `id`, which does not count as a use.
    pub fn bytes_of(&self, id: &ContentId) -> Option<u64> {
        self.index.objects.get(&id.hex()).map(|held| held.bytes)
    }

    /// The media type of the held object `id`, which does not count as a use.
    pub fn mime_of(&self, id: &ContentId) -> Option<&str> {
        self.index.objects.get(&id.hex()).map(|held| held.mime.as_str())
    }

    /// Replace the keep of `owner` with the held objects of `ids`, until `now_ms + ttl_ms`. The answer says for each
    /// id whether the store holds it. A set that holds nothing removes the keep. A set past `keep_bytes`, with every
    /// other owner's keeps, keeps nothing.
    pub fn put_keep(
        &mut self,
        owner: &str,
        ids: &[ContentId],
        ttl_ms: u64,
        now_ms: u64,
    ) -> Result<Vec<bool>, ObjectError> {
        self.expire(now_ms);
        let held: Vec<bool> = ids
            .iter()
            .map(|id| self.index.objects.contains_key(&id.hex()))
            .collect();
        let kept: Vec<String> = ids
            .iter()
            .zip(&held)
            .filter(|(_, h)| **h)
            .map(|(id, _)| id.hex())
            .collect();
        if kept.is_empty() {
            self.index.keeps.remove(owner);
            self.save()?;
            return Ok(held);
        }
        let mut all: BTreeSet<&String> = self
            .index
            .keeps
            .iter()
            .filter(|(o, _)| o.as_str() != owner)
            .flat_map(|(_, k)| k.objects.iter())
            .collect();
        all.extend(kept.iter());
        let bytes: u64 = all
            .iter()
            .filter_map(|hex| self.index.objects.get(*hex))
            .map(|held| held.bytes)
            .sum();
        if bytes > self.keep_bytes() {
            return Err(ObjectError::TooLarge {
                limit: "keep_bytes",
                bytes,
                max: self.keep_bytes(),
            });
        }
        self.index.keeps.insert(
            owner.to_string(),
            Keep {
                objects: kept,
                expires_ms: now_ms.saturating_add(ttl_ms),
            },
        );
        self.save()?;
        Ok(held)
    }

    /// The held objects that `owner` keeps, as content ids, and the milliseconds the keep has left.
    pub fn keep(&mut self, owner: &str, now_ms: u64) -> Option<(Vec<ContentId>, u64)> {
        self.expire(now_ms);
        let keep = self.index.keeps.get(owner)?;
        let ids = keep
            .objects
            .iter()
            .filter_map(|hex| ContentId::from_hex(hex).ok())
            .collect();
        Some((ids, keep.expires_ms.saturating_sub(now_ms)))
    }

    /// Remove the keep of `owner`. `false` when it keeps nothing.
    pub fn drop_keep(&mut self, owner: &str, now_ms: u64) -> Result<bool, ObjectError> {
        self.expire(now_ms);
        let dropped = self.index.keeps.remove(owner).is_some();
        if dropped {
            self.save()?;
        }
        Ok(dropped)
    }

    /// The bytes that the store holds.
    pub fn used(&self) -> u64 {
        self.used
    }

    fn expire(&mut self, now_ms: u64) {
        self.index.keeps.retain(|_, keep| keep.expires_ms > now_ms);
    }

    fn kept(&self) -> BTreeSet<String> {
        self.index
            .keeps
            .values()
            .flat_map(|keep| keep.objects.iter().cloned())
            .collect()
    }

    /// Drop keeps, the one that expires first first, until the kept objects fit `keep_bytes`. A smaller budget at an
    /// open can leave more kept bytes than its half.
    fn clamp_keeps(&mut self) {
        loop {
            let kept: u64 = self
                .kept()
                .iter()
                .filter_map(|hex| self.index.objects.get(hex))
                .map(|held| held.bytes)
                .sum();
            if kept <= self.keep_bytes() {
                return;
            }
            let Some(first) = self
                .index
                .keeps
                .iter()
                .min_by_key(|(_, keep)| keep.expires_ms)
                .map(|(owner, _)| owner.clone())
            else {
                return;
            };
            tracing::warn!(owner = %first, "the kept objects are past keep_bytes for this budget; the keep that expires first is dropped");
            self.index.keeps.remove(&first);
        }
    }

    /// Drop an object: its file, its entry, and its place in each keep.
    fn forget(&mut self, hex: &str) -> Result<(), ObjectError> {
        let file = object_path(&self.dir, hex);
        match std::fs::remove_file(&file) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io("remove", &file, e)),
        }
        if let Some(held) = self.index.objects.remove(hex) {
            self.used -= held.bytes;
        }
        self.verified.remove(hex);
        for keep in self.index.keeps.values_mut() {
            keep.objects.retain(|kept| kept != hex);
        }
        self.index.keeps.retain(|_, keep| !keep.objects.is_empty());
        self.save()
    }

    /// Evict cached objects, the least recently used first, until `bytes` more fit the budget.
    fn evict_for(&mut self, bytes: u64) -> Result<(), ObjectError> {
        let kept = self.kept();
        let mut cached: Vec<(u64, String)> = self
            .index
            .objects
            .iter()
            .filter(|(hex, _)| !kept.contains(*hex))
            .map(|(hex, held)| (held.used_ms, hex.clone()))
            .collect();
        cached.sort();
        let mut cached = cached.into_iter();
        while self.used + bytes > self.budget {
            let Some((_, hex)) = cached.next() else {
                // keep_bytes and object_bytes are each at most half the budget, and the open clamps the keeps, so
                // this cannot happen.
                return Err(ObjectError::TooLarge {
                    limit: "object_bytes",
                    bytes,
                    max: self.budget.saturating_sub(self.used),
                });
            };
            let file = object_path(&self.dir, &hex);
            std::fs::remove_file(&file).map_err(|e| io("remove", &file, e))?;
            if let Some(held) = self.index.objects.remove(&hex) {
                self.used -= held.bytes;
            }
            self.verified.remove(&hex);
        }
        Ok(())
    }

    /// Remove staged bytes and object files that the index does not name.
    fn sweep(&self) -> Result<(), ObjectError> {
        let staging = self.dir.join(STAGING);
        for entry in std::fs::read_dir(&staging)
            .map_err(|e| io("list", &staging, e))?
            .flatten()
        {
            std::fs::remove_file(entry.path()).map_err(|e| io("remove", &entry.path(), e))?;
        }
        for entry in std::fs::read_dir(&self.dir)
            .map_err(|e| io("list", &self.dir, e))?
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.len() != 2 || !entry.path().is_dir() {
                continue;
            }
            for file in std::fs::read_dir(entry.path())
                .map_err(|e| io("list", &entry.path(), e))?
                .flatten()
            {
                let hex = format!("{name}{}", file.file_name().to_string_lossy());
                if !self.index.objects.contains_key(&hex) {
                    std::fs::remove_file(file.path()).map_err(|e| io("remove", &file.path(), e))?;
                }
            }
        }
        Ok(())
    }

    /// Write the index by a synced temporary file and a rename, then sync the directory, so a crash leaves the old
    /// index or the new one.
    fn save(&self) -> Result<(), ObjectError> {
        let path = self.dir.join(INDEX);
        let temporary = self.dir.join(format!("{INDEX}.tmp"));
        let bytes = serde_json::to_vec(&self.index)
            .map_err(|e| ObjectError::Io(format!("cannot encode the object index: {e}")))?;
        crate::write_synced(&temporary, &bytes).map_err(|e| io("write", &temporary, e))?;
        std::fs::rename(&temporary, &path).map_err(|e| io("rename", &temporary, e))?;
        crate::sync_dir(&self.dir).map_err(|e| io("sync", &self.dir, e))
    }
}

fn object_path(dir: &Path, hex: &str) -> PathBuf {
    dir.join(&hex[..2]).join(&hex[2..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_of(bytes: &[u8]) -> ContentId {
        ContentId::from_hex(&blake3::hash(bytes).to_hex()).unwrap()
    }

    /// Stage bytes as a route does, and commit them.
    fn put(store: &mut ObjectStore, bytes: &[u8], mime: &str, now_ms: u64) -> Result<Committed, ObjectError> {
        let staged = store.staging_path();
        std::fs::write(&staged, bytes).unwrap();
        store.commit(&id_of(bytes), mime, &staged, bytes.len() as u64, now_ms)
    }

    #[test]
    fn a_put_holds_the_bytes_under_their_id_and_a_second_put_keeps_the_first_media_type() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ObjectStore::open(dir.path(), 1000).unwrap();
        assert_eq!(
            put(&mut store, b"hello", "audio/wav", 1).unwrap(),
            Committed {
                created: true,
                mime: "audio/wav".into()
            }
        );
        assert_eq!(
            put(&mut store, b"hello", "image/png", 2).unwrap(),
            Committed {
                created: false,
                mime: "audio/wav".into()
            }
        );
        let found = store.get(&id_of(b"hello"), 3).unwrap().unwrap();
        assert_eq!(
            (std::fs::read(&found.path).unwrap(), found.mime.as_str(), found.bytes),
            (b"hello".to_vec(), "audio/wav", 5)
        );
        assert_eq!(
            std::fs::read_dir(dir.path().join(STAGING)).unwrap().count(),
            0,
            "no staged bytes stay"
        );
    }

    #[test]
    fn the_limits_are_halves_of_the_budget_and_8_gib_at_most() {
        let dir = tempfile::tempdir().unwrap();
        let small = ObjectStore::open(dir.path(), 1000).unwrap();
        assert_eq!((small.object_bytes(), small.keep_bytes()), (500, 500));
        let big = ObjectStore::open(dir.path(), 200 << 30).unwrap();
        assert_eq!((big.object_bytes(), big.keep_bytes()), (8 << 30, 100 << 30));
        let mut store = ObjectStore::open(dir.path(), 10).unwrap();
        assert_eq!(
            put(&mut store, b"123456", "a/b", 1),
            Err(ObjectError::TooLarge {
                limit: "object_bytes",
                bytes: 6,
                max: 5
            })
        );
        assert!(store.get(&id_of(b"123456"), 2).unwrap().is_none());
    }

    #[test]
    fn a_put_evicts_the_cached_object_used_least_recently_and_never_a_kept_one() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ObjectStore::open(dir.path(), 20).unwrap();
        for (bytes, at) in [(b"aaaa", 1), (b"bbbb", 2), (b"cccc", 3), (b"dddd", 4)] {
            put(&mut store, bytes, "a/b", at).unwrap();
        }
        store.put_keep("k", &[id_of(b"aaaa")], 1000, 5).unwrap();
        store.get(&id_of(b"bbbb"), 6).unwrap().unwrap();
        // 16 of 20 bytes are held. 8 more need 4 free bytes, so the store evicts one object: cccc, the cached object
        // that was used least recently. aaaa is kept, and bbbb and dddd were used later.
        put(&mut store, b"eeeeeeee", "a/b", 7).unwrap();
        assert!(store.get(&id_of(b"aaaa"), 8).unwrap().is_some(), "a kept object stays");
        assert!(
            store.get(&id_of(b"bbbb"), 8).unwrap().is_some(),
            "a recent use keeps an object"
        );
        assert!(store.get(&id_of(b"cccc"), 8).unwrap().is_none());
        assert!(
            store.get(&id_of(b"dddd"), 8).unwrap().is_some(),
            "the store evicts no more than it needs"
        );
        assert_eq!(store.used(), 20);
    }

    #[test]
    fn a_keep_replaces_the_owners_set_reports_unheld_ids_and_expires() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ObjectStore::open(dir.path(), 100).unwrap();
        put(&mut store, b"one", "a/b", 1).unwrap();
        put(&mut store, b"two", "a/b", 1).unwrap();
        assert_eq!(
            store.put_keep("k", &[id_of(b"one"), id_of(b"never")], 100, 10).unwrap(),
            [true, false]
        );
        assert_eq!(store.keep("k", 20).unwrap(), (vec![id_of(b"one")], 90));
        store.put_keep("k", &[id_of(b"two")], 100, 30).unwrap();
        assert_eq!(
            store.keep("k", 30).unwrap().0,
            [id_of(b"two")],
            "a put replaces the set"
        );
        assert!(store.keep("k", 130).is_none(), "a keep expires");
        assert!(!store.drop_keep("k", 131).unwrap());
    }

    #[test]
    fn keeps_past_keep_bytes_keep_nothing_and_an_owner_does_not_count_its_old_set() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ObjectStore::open(dir.path(), 20).unwrap();
        put(&mut store, b"aaaaaa", "a/b", 1).unwrap();
        put(&mut store, b"bbbbbb", "a/b", 1).unwrap();
        store.put_keep("one", &[id_of(b"aaaaaa")], 100, 2).unwrap();
        assert_eq!(
            store.put_keep("two", &[id_of(b"bbbbbb")], 100, 2),
            Err(ObjectError::TooLarge {
                limit: "keep_bytes",
                bytes: 12,
                max: 10
            })
        );
        assert!(store.keep("two", 3).is_none());
        store.put_keep("one", &[id_of(b"bbbbbb")], 100, 3).unwrap();
    }

    #[test]
    fn a_cached_object_survives_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        put(
            &mut ObjectStore::open(dir.path(), 100).unwrap(),
            b"cached",
            "image/png",
            1,
        )
        .unwrap();
        let mut store = ObjectStore::open(dir.path(), 100).unwrap();
        assert_eq!(store.get(&id_of(b"cached"), 2).unwrap().unwrap().mime, "image/png");
    }

    #[test]
    fn the_store_comes_back_after_a_restart_and_drops_a_damaged_file() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut store = ObjectStore::open(dir.path(), 100).unwrap();
            put(&mut store, b"whole", "audio/wav", 1).unwrap();
            put(&mut store, b"cut short", "audio/wav", 1).unwrap();
            store
                .put_keep("k", &[id_of(b"whole"), id_of(b"cut short")], 1000, 1)
                .unwrap();
            let stray = store.staging_path();
            std::fs::write(stray, b"a put that stopped").unwrap();
        }
        let short = object_path(dir.path(), &id_of(b"cut short").hex());
        std::fs::write(&short, b"cut").unwrap();
        let mut store = ObjectStore::open(dir.path(), 100).unwrap();
        assert_eq!(store.get(&id_of(b"whole"), 2).unwrap().unwrap().mime, "audio/wav");
        assert!(
            store.get(&id_of(b"cut short"), 2).unwrap().is_none(),
            "a file of the wrong size drops its object"
        );
        assert!(!short.exists(), "and its file goes");
        assert_eq!(
            store.keep("k", 2).unwrap().0,
            [id_of(b"whole")],
            "a keep forgets a dropped object"
        );
        assert_eq!(
            std::fs::read_dir(dir.path().join(STAGING)).unwrap().count(),
            0,
            "staged bytes of a stopped put go"
        );
    }

    #[test]
    fn a_torn_index_moves_aside_and_the_store_starts_empty() {
        let dir = tempfile::tempdir().unwrap();
        put(&mut ObjectStore::open(dir.path(), 100).unwrap(), b"lost", "a/b", 1).unwrap();
        std::fs::write(dir.path().join(INDEX), b"{\"objects\": {\"ab").unwrap();
        let mut store = ObjectStore::open(dir.path(), 100).unwrap();
        assert!(store.get(&id_of(b"lost"), 2).unwrap().is_none());
        assert_eq!(store.used(), 0);
        assert!(dir.path().join(format!("{INDEX}.damaged")).exists());
        put(&mut store, b"new", "a/b", 3).unwrap();
    }

    #[test]
    fn a_file_with_the_right_size_and_other_bytes_is_never_served() {
        let dir = tempfile::tempdir().unwrap();
        put(&mut ObjectStore::open(dir.path(), 100).unwrap(), b"original", "a/b", 1).unwrap();
        std::fs::write(object_path(dir.path(), &id_of(b"original").hex()), b"flipped!").unwrap();
        let mut store = ObjectStore::open(dir.path(), 100).unwrap();
        assert!(
            store.get(&id_of(b"original"), 2).unwrap().is_none(),
            "the first read hashes the file"
        );
        assert_eq!(store.used(), 0);
    }

    #[test]
    fn a_smaller_budget_drops_the_keep_that_expires_first_and_a_bad_key_drops() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut store = ObjectStore::open(dir.path(), 100).unwrap();
            put(&mut store, b"aaaaaaaaaa", "a/b", 1).unwrap();
            put(&mut store, b"bbbbbbbbbb", "a/b", 1).unwrap();
            store.put_keep("soon", &[id_of(b"aaaaaaaaaa")], 10, 1).unwrap();
            store.put_keep("late", &[id_of(b"bbbbbbbbbb")], 1000, 1).unwrap();
        }
        let mut index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join(INDEX)).unwrap()).unwrap();
        index["objects"]["zz"] = serde_json::json!({"mime": "a/b", "bytes": 1, "used_ms": 1});
        std::fs::write(dir.path().join(INDEX), serde_json::to_vec(&index).unwrap()).unwrap();
        // A budget of 30 keeps at most 15 bytes, so one keep of 10 bytes stays: the one that expires last.
        let mut store = ObjectStore::open(dir.path(), 30).unwrap();
        assert!(store.keep("soon", 2).is_none());
        assert_eq!(store.keep("late", 2).unwrap().0, [id_of(b"bbbbbbbbbb")]);
        assert_eq!(store.used(), 20, "the entry with a bad key is gone");
        put(&mut store, b"cccccccccc", "a/b", 3).unwrap();
    }
}
