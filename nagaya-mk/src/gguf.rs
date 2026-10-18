//! The GGUF header reader and the spot digest of the weight shards.
//!
//! mk's `weight_hash` is a spot check: it hashes each shard's size and every byte before the tensor data. The spec
//! labels it a spot check until we have a logical checkpoint digest (spec 12.2).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// The GGUF magic, "GGUF" in little-endian byte order.
const MAGIC: u32 = 0x4655_4747;

/// The default data alignment (the GGUF default).
const DEFAULT_ALIGNMENT: u32 = 32;

/// The largest metadata key that the reader stores, in bytes.
const MAX_KEY: u64 = 4096;

/// What the header reader learns about one shard.
#[derive(Debug)]
pub(crate) struct ShardHeader {
    /// The size of the file in bytes.
    pub size: u64,
    /// The first byte of the tensor data.
    pub data_offset: u64,
}

/// A streaming reader with a position and a bound.
struct Cursor<R> {
    reader: R,
    position: u64,
    size: u64,
    path: String,
}

impl<R: Read> Cursor<R> {
    fn new(reader: R, size: u64, path: &Path) -> Self {
        Self {
            reader,
            position: 0,
            size,
            path: path.display().to_string(),
        }
    }

    fn take(&mut self, count: u64) -> Result<(), String> {
        if count > self.size.saturating_sub(self.position) {
            return Err(format!(
                "{}: the header ends before byte {}",
                self.path,
                self.position + count
            ));
        }
        // Read the bytes in chunks, so a corrupt length cannot allocate a large buffer.
        let mut buffer = [0u8; 64 * 1024];
        let mut left = count;
        while left > 0 {
            let want = left.min(buffer.len() as u64) as usize;
            self.reader
                .read_exact(&mut buffer[..want])
                .map_err(|error| format!("{}: read at byte {}: {error}", self.path, self.position))?;
            self.position += want as u64;
            left -= want as u64;
        }
        Ok(())
    }

    fn read(&mut self, buffer: &mut [u8]) -> Result<(), String> {
        if buffer.len() as u64 > self.size.saturating_sub(self.position) {
            return Err(format!(
                "{}: the header ends before byte {}",
                self.path,
                self.position + buffer.len() as u64
            ));
        }
        self.reader
            .read_exact(buffer)
            .map_err(|error| format!("{}: read at byte {}: {error}", self.path, self.position))?;
        self.position += buffer.len() as u64;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut bytes = [0u8; 4];
        self.read(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut bytes = [0u8; 8];
        self.read(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn string(&mut self) -> Result<String, String> {
        let length = self.u64()?;
        if length > MAX_KEY {
            return Err(format!(
                "{}: a metadata key of {length} bytes is larger than {MAX_KEY}",
                self.path
            ));
        }
        let mut bytes = vec![0u8; length as usize];
        self.read(&mut bytes)?;
        String::from_utf8(bytes).map_err(|error| format!("{}: a metadata key is not UTF-8: {error}", self.path))
    }

    fn skip_string(&mut self) -> Result<(), String> {
        let length = self.u64()?;
        self.take(length)
    }

    fn skip_value(&mut self, kind: u32) -> Result<(), String> {
        match kind {
            0..=1 | 7 => self.take(1),
            2..=3 => self.take(2),
            4..=6 => self.take(4),
            8 => self.skip_string(),
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if count > self.size.saturating_sub(self.position) {
                    return Err(format!(
                        "{}: an array of {count} values is larger than the file",
                        self.path
                    ));
                }
                for _ in 0..count {
                    self.skip_value(element)?;
                }
                Ok(())
            }
            10..=12 => self.take(8),
            other => Err(format!("{}: metadata type {other} is unknown", self.path)),
        }
    }
}

/// Read one shard's GGUF header and return its data offset.
pub(crate) fn read_header(path: &Path) -> Result<ShardHeader, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("{}: {error}", path.display()))?
        .len();
    if size < 24 {
        return Err(format!(
            "{}: a file of {size} bytes is too small for a GGUF header",
            path.display()
        ));
    }
    let mut cursor = Cursor::new(BufReader::new(file), size, path);

    let magic = cursor.u32()?;
    if magic != MAGIC {
        return Err(format!("{}: bad GGUF magic {magic:#010x}", path.display()));
    }
    let version = cursor.u32()?;
    if version != 3 {
        return Err(format!("{}: GGUF version {version} is not 3", path.display()));
    }
    let tensor_count = cursor.u64()?;
    let metadata_count = cursor.u64()?;

    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let key = cursor.string()?;
        let kind = cursor.u32()?;
        if key == "general.alignment" {
            if kind != 4 {
                return Err(format!("{}: general.alignment is not a u32", path.display()));
            }
            let value = cursor.u32()?;
            if value == 0 || !value.is_power_of_two() {
                return Err(format!(
                    "{}: general.alignment {value} is not a power of two",
                    path.display()
                ));
            }
            alignment = value;
        } else {
            cursor.skip_value(kind)?;
        }
    }

    for _ in 0..tensor_count {
        cursor.skip_string()?;
        let dimensions = cursor.u32()?;
        if !(1..=4).contains(&dimensions) {
            return Err(format!("{}: a tensor has {dimensions} dimensions", path.display()));
        }
        cursor.take(8 * dimensions as u64)?;
        cursor.take(4 + 8)?;
    }

    let remainder = cursor.position % alignment as u64;
    let data_offset = if remainder == 0 {
        cursor.position
    } else {
        cursor.position + (alignment as u64 - remainder)
    };
    if data_offset > size {
        return Err(format!(
            "{}: the data offset {data_offset} is past the file size {size}",
            path.display()
        ));
    }
    Ok(ShardHeader { size, data_offset })
}

/// The spot digest: each shard's size and every byte before its tensor data, in the given order.
pub(crate) fn spot_digest(shards: &[std::path::PathBuf]) -> Result<String, String> {
    let mut hash = Sha256::new();
    for path in shards {
        let header = read_header(path)?;
        hash.update(header.size.to_le_bytes());
        let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let mut left = header.data_offset;
        let mut buffer = vec![0u8; 1 << 20];
        while left > 0 {
            let want = left.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..want]).map_err(|error| {
                format!(
                    "{}: read at byte {}: {error}",
                    path.display(),
                    header.data_offset - left
                )
            })?;
            hash.update(&buffer[..want]);
            left -= want as u64;
        }
    }
    Ok(hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::{read_header, spot_digest};

    /// Write a minimal GGUF v3 file with one metadata key and one tensor.
    fn write_gguf(path: &std::path::Path, magic: u32, alignment: u32) {
        let mut file = std::fs::File::create(path).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&magic.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one metadata key
        let key = b"general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        bytes.extend_from_slice(&alignment.to_le_bytes());
        let name = b"x";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one dimension
        bytes.extend_from_slice(&4u64.to_le_bytes()); // 4 elements
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
        // Padding to the 64-byte alignment and a little tensor data.
        bytes.resize(154, 0);
        file.write_all(&bytes).unwrap();
    }

    #[test]
    fn a_header_reports_its_aligned_data_offset() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("shard.gguf");
        write_gguf(&path, 0x4655_4747, 32);
        let header = read_header(&path).unwrap();
        // 24 header bytes + (8 + 17 + 4 + 4) + (8 + 1 + 4 + 8 + 4 + 8) = 90, aligned to 96.
        assert_eq!(header.data_offset, 96);
        assert_eq!(header.size, 154);
    }

    #[test]
    fn a_bad_magic_is_an_error() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("shard.gguf");
        write_gguf(&path, 0xdead_beef, 32);
        let error = read_header(&path).unwrap_err();
        assert!(error.contains("bad GGUF magic"), "{error}");
    }

    #[test]
    fn the_digest_covers_the_size_and_the_header() {
        let directory = tempfile::tempdir().unwrap();
        let left = directory.path().join("left.gguf");
        let right = directory.path().join("right.gguf");
        write_gguf(&left, 0x4655_4747, 32);
        write_gguf(&right, 0x4655_4747, 32);
        let left_digest = spot_digest(std::slice::from_ref(&left)).unwrap();
        let right_digest = spot_digest(std::slice::from_ref(&right)).unwrap();
        assert_eq!(left_digest, right_digest);
        // A different alignment changes the header bytes and the data offset, so the digest changes.
        write_gguf(&right, 0x4655_4747, 64);
        assert_ne!(left_digest, spot_digest(&[right]).unwrap());
    }
}
