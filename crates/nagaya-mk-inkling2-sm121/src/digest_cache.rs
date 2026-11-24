//! The memo and the cache file of the digest of the checkpoint (engine-abi "The GB10 Inkling 2 binding", "The host
//! spec"; **Proposed**).
//!
//! A read of the whole checkpoint takes 15.9 s. The open and `describe` of the factory both need the digest, so they
//! share it in two ways:
//! - A memo in the process keeps each digest by its key.
//! - A cache file in the cache folder keeps it across starts. One small file belongs to one checkpoint.
//!
//! The key holds the facts of the weights file that any write changes: the canonical path, `st_dev`, `st_ino`, the
//! size, and `mtime` and `ctime` in nanoseconds. A user tool cannot set `ctime`, so a changed file misses. The key
//! also holds the BLAKE3 of the text of `config.json` and the version of the digest key.
//!
//! The cache never guesses. A key that differs, a missing file and a file that does not parse each give a miss, and a
//! miss computes the digest. An I/O error of the cache folder is a warning and not a failure.

use std::ffi::OsString;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Value, json};

/// The version of the digest and of the cache file. Change it with the digest key or with the format of the file.
pub const VERSION: &str = "nagaya inkling2 checkpoint v1; cache file v1";

/// The facts that decide whether a stored digest still names the checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key(Value);

impl Key {
    /// Read the key of the weights file `weights` and the text of `config.json`.
    pub fn of(config_text: &str, weights: &Path) -> Result<Self, String> {
        let path =
            std::fs::canonicalize(weights).map_err(|e| format!("cannot resolve `{}`: {e}", weights.display()))?;
        let meta = std::fs::metadata(&path).map_err(|e| format!("cannot stat `{}`: {e}", path.display()))?;
        let nanos = |secs: i64, nsec: i64| i128::from(secs) * 1_000_000_000 + i128::from(nsec);
        // Numbers are strings, so that a large value never loses bits in the JSON of the file.
        Ok(Self(json!({
            "path": path.to_string_lossy(),
            "dev": meta.dev().to_string(),
            "ino": meta.ino().to_string(),
            "size": meta.size().to_string(),
            "mtime_ns": nanos(meta.mtime(), meta.mtime_nsec()).to_string(),
            "ctime_ns": nanos(meta.ctime(), meta.ctime_nsec()).to_string(),
            "config_blake3": blake3::hash(config_text.as_bytes()).to_hex().to_string(),
            "version": VERSION,
        })))
    }

    /// The name of the cache file: it depends on the path only, so a changed file replaces its own file.
    fn file_name(&self) -> String {
        let path = self.0["path"].as_str().unwrap_or_default();
        format!("{}.json", &blake3::hash(path.as_bytes()).to_hex().as_str()[..32])
    }

    fn text(&self) -> String {
        self.0.to_string()
    }
}

/// The memo of one process: the digest of each key that it saw.
#[derive(Default)]
pub struct Memo(Mutex<Vec<(String, String)>>);

impl Memo {
    fn get(&self, key: &Key) -> Option<String> {
        let text = key.text();
        let seen = self.0.lock().expect("the lock of the memo");
        seen.iter()
            .find(|(seen_key, _)| *seen_key == text)
            .map(|(_, digest)| digest.clone())
    }

    fn put(&self, key: &Key, digest: &str) {
        let text = key.text();
        let mut seen = self.0.lock().expect("the lock of the memo");
        seen.retain(|(seen_key, _)| *seen_key != text);
        seen.push((text, digest.to_owned()));
    }
}

/// The memo of the process.
pub static PROCESS_MEMO: Memo = Memo(Mutex::new(Vec::new()));

/// The folder of the digest files: `$XDG_CACHE_HOME/nagaya/digests`, or `~/.cache/nagaya/digests`. An empty variable
/// counts as not set. With neither variable the cache has no folder.
pub fn cache_dir(xdg_cache_home: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let non_empty = |value: Option<OsString>| value.filter(|value| !value.is_empty()).map(PathBuf::from);
    let base = non_empty(xdg_cache_home).or_else(|| non_empty(home).map(|home| home.join(".cache")))?;
    Some(base.join("nagaya").join("digests"))
}

/// The folder of the digest files that the environment names.
pub fn cache_dir_from_env() -> Option<PathBuf> {
    cache_dir(std::env::var_os("XDG_CACHE_HOME"), std::env::var_os("HOME"))
}

/// True for a digest in the form that the engine writes: `blake3:` and 64 hexadecimal digits.
fn is_digest(text: &str) -> bool {
    text.strip_prefix("blake3:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

/// The digest in the file of `key`, or `None` for each kind of miss.
fn read_file(dir: &Path, key: &Key) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(key.file_name())).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    if value.get("key") != Some(&key.0) {
        return None;
    }
    let digest = value.get("digest")?.as_str()?;
    is_digest(digest).then(|| digest.to_owned())
}

/// Write the file of `key` through a temporary file and a rename, so a reader never sees half a file.
fn write_file(dir: &Path, key: &Key, digest: &str) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot make the folder `{}`: {e}", dir.display()))?;
    let name = key.file_name();
    let temp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let body = json!({ "key": key.0, "digest": digest }).to_string();
    std::fs::write(&temp, body).map_err(|e| format!("cannot write `{}`: {e}", temp.display()))?;
    let target = dir.join(&name);
    std::fs::rename(&temp, &target).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        format!("cannot rename `{}` to `{}`: {e}", temp.display(), target.display())
    })
}

/// The digest of the checkpoint. `compute` reads the whole file, and it runs only on a miss of the memo and of the
/// file in `dir`. With `dir` set to `None` only the memo works.
///
/// The key is read before `compute`. A file that changes during the read gets a new `ctime`, so the digest that
/// `compute` gave stays under the old key, and no later call finds it.
pub fn cached_digest(
    memo: &Memo,
    dir: Option<&Path>,
    config_text: &str,
    weights: &Path,
    compute: impl FnOnce() -> Result<String, String>,
) -> Result<String, String> {
    let key = Key::of(config_text, weights)?;
    if let Some(digest) = memo.get(&key) {
        return Ok(digest);
    }
    if let Some(digest) = dir.and_then(|dir| read_file(dir, &key)) {
        memo.put(&key, &digest);
        return Ok(digest);
    }
    let digest = compute()?;
    memo.put(&key, &digest);
    if let Some(dir) = dir
        && let Err(message) = write_file(dir, &key, &digest)
    {
        eprintln!(
            "nagaya: warning: the digest of the checkpoint is not cached: {message}. The open goes on with the digest \
             that it computed. Check the folder `{}`",
            dir.display()
        );
    }
    Ok(digest)
}
