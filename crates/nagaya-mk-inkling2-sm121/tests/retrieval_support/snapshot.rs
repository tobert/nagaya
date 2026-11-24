//! The snapshot files of the retrieval test. A file holds the parts of one slot after the prefill of a context, in the
//! stored bytes of one cache format. The key of a file is the digest of the engine identity, the hash of the context
//! tokens and the format. A snapshot of one format never stands in for another.
//!
//! The file is the bytes of the parts one after another, then a trailer of JSON, then the length of the trailer and
//! the magic. The trailer holds the key, the size of each part and the BLAKE3 of the bytes. A writer makes a temporary
//! file and renames it when it is whole. A reader checks the size and the key when it opens, and the BLAKE3 when it ends.
//! A file that fails a check is an error: the test never prefills again in its place.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"NGYSNAP1";
const TAIL: u64 = 16;

/// The key of a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    /// The digest of the engine identity.
    pub identity: [u8; 32],
    /// The BLAKE3 of the context tokens, as `u32` values in little-endian bytes.
    pub tokens: [u8; 32],
    pub format: String,
}

impl Key {
    pub fn file_name(&self) -> String {
        format!(
            "snap-{}-{}-{}.bin",
            self.format,
            hex(&self.tokens)[..16].to_owned(),
            hex(&self.identity)[..16].to_owned()
        )
    }

    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.file_name())
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({"identity": hex(&self.identity), "tokens": hex(&self.tokens), "format": self.format})
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The hash of the context tokens.
pub fn token_hash(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for token in tokens {
        hasher.update(&token.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// One part in a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartMeta {
    pub id: usize,
    pub name: String,
    /// The first position of a row part, and 0 for a fixed part.
    pub start: u64,
    /// The rows of a row part, and 0 for a fixed part.
    pub rows: u64,
    pub len: u64,
}

fn io(path: &Path, what: &str, e: std::io::Error) -> String {
    format!("{}: cannot {what}: {e}", path.display())
}

pub struct Writer {
    key: Key,
    final_path: PathBuf,
    temp: PathBuf,
    file: Option<File>,
    hasher: blake3::Hasher,
    parts: Vec<PartMeta>,
    bytes: u64,
    /// The count of positions that the slot held.
    pub n: u64,
    /// Facts that the reader gives back: the test keeps there the hash of the first logits.
    pub note: serde_json::Value,
}

impl Writer {
    pub fn create(dir: &Path, key: &Key, n: u64) -> Result<Writer, String> {
        std::fs::create_dir_all(dir).map_err(|e| io(dir, "make the folder", e))?;
        let final_path = key.path(dir);
        let temp = dir.join(format!(".{}.tmp-{}", key.file_name(), std::process::id()));
        let file = File::create(&temp).map_err(|e| io(&temp, "create the temporary file", e))?;
        Ok(Writer {
            key: key.clone(),
            final_path,
            temp,
            file: Some(file),
            hasher: blake3::Hasher::new(),
            parts: Vec::new(),
            bytes: 0,
            n,
            note: serde_json::Value::Null,
        })
    }

    pub fn add(&mut self, id: usize, name: &str, start: u64, rows: u64, bytes: &[u8]) -> Result<(), String> {
        self.file
            .as_mut()
            .expect("a writer that is not finished has its file")
            .write_all(bytes)
            .map_err(|e| io(&self.temp, "write", e))?;
        self.hasher.update(bytes);
        self.parts.push(PartMeta {
            id,
            name: name.to_owned(),
            start,
            rows,
            len: bytes.len() as u64,
        });
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    /// Write the trailer, make the file durable and rename it to its name. Give its size.
    pub fn finish(mut self) -> Result<u64, String> {
        let trailer = serde_json::json!({
            "version": 1,
            "key": self.key.json(),
            "n": self.n,
            "note": self.note,
            "data_len": self.bytes,
            "data_blake3": hex(self.hasher.finalize().as_bytes()),
            "parts": self.parts.iter().map(|p| serde_json::json!({
                "id": p.id, "name": p.name, "start": p.start, "rows": p.rows, "len": p.len
            })).collect::<Vec<_>>(),
        });
        let trailer = serde_json::to_vec(&trailer).expect("the trailer is JSON");
        let mut file = self.file.take().expect("a writer that is not finished has its file");
        file.write_all(&trailer)
            .and_then(|()| file.write_all(&(trailer.len() as u64).to_le_bytes()))
            .and_then(|()| file.write_all(MAGIC))
            .and_then(|()| file.sync_all())
            .map_err(|e| io(&self.temp, "finish", e))?;
        drop(file);
        std::fs::rename(&self.temp, &self.final_path).map_err(|e| io(&self.final_path, "rename the file", e))?;
        if let Some(dir) = self.final_path.parent() {
            File::open(dir)
                .and_then(|d| d.sync_all())
                .map_err(|e| io(dir, "sync the folder", e))?;
        }
        Ok(self.bytes + trailer.len() as u64 + TAIL)
    }
}

impl Drop for Writer {
    /// A writer that was not finished leaves no file.
    fn drop(&mut self) {
        if self.file.is_some() {
            self.file = None;
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

pub struct Reader {
    path: PathBuf,
    file: File,
    hasher: blake3::Hasher,
    want: String,
    pub parts: Vec<PartMeta>,
    pub n: u64,
    pub note: serde_json::Value,
    next: usize,
}

impl Reader {
    /// Open the snapshot of `key`. `Ok(None)` means that no file has the name of the key. A file that fails a check is
    /// an error.
    pub fn open(dir: &Path, key: &Key) -> Result<Option<Reader>, String> {
        let path = key.path(dir);
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(&path, "open", e)),
        };
        let size = file.metadata().map_err(|e| io(&path, "stat", e))?.len();
        if size < TAIL {
            return Err(format!(
                "{}: the file has {size} bytes, fewer than its tail; delete it to make the snapshot again",
                path.display()
            ));
        }
        let mut tail = [0u8; TAIL as usize];
        file.seek(SeekFrom::End(-(TAIL as i64)))
            .and_then(|_| file.read_exact(&mut tail))
            .map_err(|e| io(&path, "read the tail", e))?;
        if &tail[8..] != MAGIC {
            return Err(format!(
                "{}: the file has no end mark; it is cut or not a snapshot. Delete it to make the snapshot again",
                path.display()
            ));
        }
        let trailer_len = u64::from_le_bytes(tail[..8].try_into().unwrap());
        if trailer_len > size - TAIL {
            return Err(format!("{}: the trailer is longer than the file", path.display()));
        }
        let mut trailer = vec![0u8; trailer_len as usize];
        file.seek(SeekFrom::Start(size - TAIL - trailer_len))
            .and_then(|_| file.read_exact(&mut trailer))
            .map_err(|e| io(&path, "read the trailer", e))?;
        let json: serde_json::Value = serde_json::from_slice(&trailer)
            .map_err(|e| format!("{}: the trailer is not JSON: {e}", path.display()))?;
        if json["key"] != key.json() {
            return Err(format!(
                "{}: the key inside the file is not the key in its name: {}",
                path.display(),
                json["key"]
            ));
        }
        let data_len = json["data_len"].as_u64().ok_or("the trailer has no data_len")?;
        if data_len + trailer_len + TAIL != size {
            return Err(format!(
                "{}: the file has {size} bytes, and its trailer says {} bytes. Delete it to make the snapshot again",
                path.display(),
                data_len + trailer_len + TAIL
            ));
        }
        let parts: Vec<PartMeta> = json["parts"]
            .as_array()
            .ok_or("the trailer has no parts")?
            .iter()
            .map(|p| PartMeta {
                id: p["id"].as_u64().unwrap() as usize,
                name: p["name"].as_str().unwrap().to_owned(),
                start: p["start"].as_u64().unwrap(),
                rows: p["rows"].as_u64().unwrap(),
                len: p["len"].as_u64().unwrap(),
            })
            .collect();
        if parts.iter().map(|p| p.len).sum::<u64>() != data_len {
            return Err(format!("{}: the parts do not add up to the data", path.display()));
        }
        file.seek(SeekFrom::Start(0)).map_err(|e| io(&path, "seek", e))?;
        Ok(Some(Reader {
            path,
            file,
            hasher: blake3::Hasher::new(),
            want: json["data_blake3"]
                .as_str()
                .ok_or("the trailer has no hash")?
                .to_owned(),
            parts,
            n: json["n"].as_u64().ok_or("the trailer has no n")?,
            note: json["note"].clone(),
            next: 0,
        }))
    }

    /// Read the next part into `buf` and give its meta. The parts come in the order of the file.
    pub fn read_next(&mut self, buf: &mut Vec<u8>) -> Result<PartMeta, String> {
        let meta = self
            .parts
            .get(self.next)
            .ok_or_else(|| format!("{}: no part is left", self.path.display()))?
            .clone();
        buf.resize(meta.len as usize, 0);
        self.file
            .read_exact(buf)
            .map_err(|e| io(&self.path, &format!("read part {}", meta.id), e))?;
        self.hasher.update(buf);
        self.next += 1;
        Ok(meta)
    }

    /// Check that the reader gave every part and that the BLAKE3 of the bytes is the one that the writer kept.
    pub fn finish(self) -> Result<(), String> {
        if self.next != self.parts.len() {
            return Err(format!(
                "{}: the restore read {} of {} parts",
                self.path.display(),
                self.next,
                self.parts.len()
            ));
        }
        let got = hex(self.hasher.finalize().as_bytes());
        if got != self.want {
            return Err(format!(
                "{}: the BLAKE3 of the bytes is {got}, and the file says {}. The file is damaged; delete it to make \
                 the snapshot again",
                self.path.display(),
                self.want
            ));
        }
        Ok(())
    }
}
