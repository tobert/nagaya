//! The GGUF reader of the Qwen3.8 engine: the shards' metadata, their tensor records, and the read-only maps of the files.
//!
//! The reader is a port of the reference's `src/gguf/gguf.cpp` (reference commit `555a2a3`). It keeps the reference's checks: a file that does not hold
//! what it promises is an error that names the file and the byte. The family library parses the metadata values
//! (`nagaya_text_qwen38::gguf::Metadata`), so the workspace has one parser of GGUF values.
//!
//! The reader also gives the spot digest of the weight shards. The reference's `weight_hash` is a spot check: it hashes each
//! shard's size and every byte before the tensor data. The spec labels it a spot check until we have a logical
//! checkpoint digest (spec 12.2).

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use nagaya_text_qwen38::gguf::{Metadata, Value};
use sha2::{Digest, Sha256};

use crate::dequant::GgmlType;
#[cfg(test)]
use crate::dequant::dequant_f32;

/// The default data alignment (the GGUF default).
const DEFAULT_ALIGNMENT: u64 = 32;

/// The longest tensor name that the reader takes, in bytes. GGUF names are at most 64 bytes.
const MAX_NAME: u64 = 4096;

/// The bytes of the fixed header: the magic, the version and the two counts.
const FIXED_HEADER: usize = 24;

/// What the header reader learns about one shard.
#[derive(Debug)]
pub(crate) struct ShardHeader {
    /// The size of the file in bytes.
    pub size: u64,
    /// The first byte of the tensor data.
    pub data_offset: u64,
}

/// One tensor record, checked against its shard.
///
/// `ne[0]` is the innermost dimension, as the file stores it. A row is `ne[0]` values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TensorInfo {
    pub name: String,
    pub kind: GgmlType,
    pub n_dims: u32,
    pub ne: [u64; 4],
    pub n_elements: u64,
    pub n_bytes: u64,
    /// The index of the shard in the given order.
    pub shard: usize,
    /// The first byte of the tensor in its shard file.
    pub offset: u64,
}

impl TensorInfo {
    /// The count of rows of `ne[0]` values.
    #[cfg(test)]
    pub(crate) fn rows(&self) -> u64 {
        self.n_elements / self.ne[0]
    }
}

/// A read-only private map of a whole file.
///
/// The map is never written. A file that another process truncates while it is mapped stops this process with
/// SIGBUS, as it did for the reference's C++ reader.
struct Map {
    ptr: *mut c_void,
    len: usize,
    file: File,
}

// SAFETY: the map is read-only for its whole life, so threads can share it.
unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Map {
    fn open(path: &Path) -> Result<Map, String> {
        let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let size = file
            .metadata()
            .map_err(|error| format!("{}: {error}", path.display()))?
            .len();
        if size < FIXED_HEADER as u64 {
            return Err(format!(
                "{}: a file of {size} bytes is too small for a GGUF header",
                path.display()
            ));
        }
        let len =
            usize::try_from(size).map_err(|_| format!("{}: {size} bytes do not fit in memory", path.display()))?;
        // SAFETY: a new private, read-only map of an open file. The kernel chooses the address.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap {}: {}", path.display(), std::io::Error::last_os_error()));
        }
        Ok(Map { ptr, len, file })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the map holds `len` readable bytes until `drop`.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: the map came from `mmap` with this length, and nothing borrows it past `drop`.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// A bounded cursor over the bytes of one shard. A read past the end is an error with the path and the byte.
struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
    path: &'a Path,
}

impl<'a> Cursor<'a> {
    fn error(&self, message: impl std::fmt::Display) -> String {
        format!("{}: byte {}: {message}", self.path.display(), self.position)
    }

    fn take(&mut self, count: u64) -> Result<&'a [u8], String> {
        let left = (self.bytes.len() - self.position) as u64;
        if count > left {
            return Err(self.error(format!(
                "a read of {count} bytes runs past the end of the file. The file is cut short."
            )));
        }
        let start = self.position;
        self.position += count as usize;
        Ok(&self.bytes[start..self.position])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("take gives 4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("take gives 8 bytes"),
        ))
    }

    fn name(&mut self) -> Result<String, String> {
        let length = self.u64()?;
        if length > MAX_NAME {
            return Err(self.error(format!("a tensor name of {length} bytes is longer than {MAX_NAME}")));
        }
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|error| self.error(format!("a tensor name is not UTF-8: {error}")))
    }
}

/// What one shard's header holds.
struct Header {
    data_offset: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
}

/// The integer of a metadata value, or `None` for a value that is not an integer or does not fit in an `i64`.
fn integer(value: &Value) -> Option<i64> {
    match *value {
        Value::U8(v) => Some(i64::from(v)),
        Value::I8(v) => Some(i64::from(v)),
        Value::U16(v) => Some(i64::from(v)),
        Value::I16(v) => Some(i64::from(v)),
        Value::U32(v) => Some(i64::from(v)),
        Value::I32(v) => Some(i64::from(v)),
        Value::U64(v) => i64::try_from(v).ok(),
        Value::I64(v) => Some(v),
        _ => None,
    }
}

/// Parse the header of the shard at index `shard`: the metadata, the tensor records, and the data offset.
///
/// Each tensor must have a known type, whole blocks, an aligned offset, and data inside the file.
fn parse_header(bytes: &[u8], path: &Path, shard: usize) -> Result<Header, String> {
    if bytes.len() < FIXED_HEADER {
        return Err(format!(
            "{}: a file of {} bytes is too small for a GGUF header",
            path.display(),
            bytes.len()
        ));
    }
    let size = bytes.len() as u64;
    // The family library reads the magic, the version, the counts and the metadata. It stops after the last entry.
    let mut rest = bytes;
    let metadata = Metadata::read(&mut rest, size).map_err(|error| format!("{}: {error}", path.display()))?;
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
    let mut cursor = Cursor {
        bytes,
        position: bytes.len() - rest.len(),
        path,
    };

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(Value::U32(value)) if value.is_power_of_two() => u64::from(*value),
        Some(Value::U32(value)) => {
            return Err(format!(
                "{}: general.alignment {value} is not a power of two",
                path.display()
            ));
        }
        Some(_) => return Err(format!("{}: general.alignment is not a u32", path.display())),
    };

    struct Record {
        name: String,
        n_dims: u32,
        ne: [u64; 4],
        kind: u32,
        offset: u64,
    }
    let mut records = Vec::new();
    for _ in 0..tensor_count {
        let name = cursor.name()?;
        let n_dims = cursor.u32()?;
        if !(1..=4).contains(&n_dims) {
            return Err(cursor.error(format!("the tensor {name} has {n_dims} dimensions, not 1 to 4")));
        }
        let mut ne = [1u64; 4];
        for dimension in ne.iter_mut().take(n_dims as usize) {
            *dimension = cursor.u64()?;
        }
        let kind = cursor.u32()?;
        let offset = cursor.u64()?;
        records.push(Record {
            name,
            n_dims,
            ne,
            kind,
            offset,
        });
    }

    let position = cursor.position as u64;
    let data_offset = position.div_ceil(alignment) * alignment;
    if data_offset > size {
        return Err(format!(
            "{}: the data offset {data_offset} is past the file size {size}",
            path.display()
        ));
    }

    let fail = |message: String| format!("{}: {message}", path.display());
    let mut tensors = Vec::with_capacity(records.len());
    for record in records {
        let name = record.name;
        let kind = GgmlType::from_id(record.kind).ok_or_else(|| {
            fail(format!(
                "the tensor {name} has the unknown ggml type id {}",
                record.kind
            ))
        })?;
        let n_elements = record
            .ne
            .iter()
            .try_fold(1u64, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| {
                fail(format!(
                    "the element count of the tensor {name} does not fit in 64 bits"
                ))
            })?;
        if n_elements == 0 || !n_elements.is_multiple_of(kind.block_values()) {
            return Err(fail(format!(
                "the tensor {name} has {n_elements} values, which are not whole blocks of {} ({})",
                kind.block_values(),
                kind.name()
            )));
        }
        let n_bytes = n_elements / kind.block_values() * kind.block_bytes();
        if !record.offset.is_multiple_of(alignment) {
            return Err(fail(format!(
                "the offset {} of the tensor {name} is not aligned to {alignment}",
                record.offset
            )));
        }
        let start = data_offset.checked_add(record.offset);
        if start.is_none_or(|start| start > size || n_bytes > size - start) {
            return Err(fail(format!(
                "the data of the tensor {name} at offset {} for {n_bytes} bytes runs past the file size {size}",
                record.offset
            )));
        }
        tensors.push(TensorInfo {
            name,
            kind,
            n_dims: record.n_dims,
            ne: record.ne,
            n_elements,
            n_bytes,
            shard,
            offset: data_offset + record.offset,
        });
    }
    Ok(Header {
        data_offset,
        metadata,
        tensors,
    })
}

/// Read one shard's GGUF header and return its data offset.
pub(crate) fn read_header(path: &Path) -> Result<ShardHeader, String> {
    let map = Map::open(path)?;
    let header = parse_header(map.bytes(), path, 0)?;
    Ok(ShardHeader {
        size: map.len as u64,
        data_offset: header.data_offset,
    })
}

/// One shard of a model: its map and where its tensor data starts.
struct Shard {
    map: Map,
    path: PathBuf,
    #[cfg(test)]
    data_offset: u64,
}

/// A model across every shard of a split GGUF, with each shard mapped read-only.
///
/// The maps hold the tensor data. The reader never copies it into anonymous memory: the shards hold 94 GB on gfx1151host.
pub(crate) struct Model {
    shards: Vec<Shard>,
    tensors: Vec<TensorInfo>,
    /// The index in `tensors` of each name.
    index: HashMap<String, usize>,
    /// The metadata of each shard.
    metadata: Vec<Metadata>,
    /// The shard of each metadata key, except the keys `split.*`.
    key_shard: HashMap<String, usize>,
}

impl Model {
    /// Open the shards of one model, in split order.
    ///
    /// Each shard must hold `split.count`, equal to the count of paths, and `split.no`, equal to its place in the
    /// list. A key other than `split.*` must be in one shard only. A tensor name must be in one shard only.
    pub(crate) fn open(paths: &[PathBuf]) -> Result<Model, String> {
        if paths.is_empty() {
            return Err("the model has no shard paths; give the shards in split order".to_owned());
        }
        let mut model = Model {
            shards: Vec::new(),
            tensors: Vec::new(),
            index: HashMap::new(),
            metadata: Vec::new(),
            key_shard: HashMap::new(),
        };
        for (place, path) in paths.iter().enumerate() {
            let map = Map::open(path)?;
            let header = parse_header(map.bytes(), path, place)?;
            let split = |key: &str| {
                header
                    .metadata
                    .get(key)
                    .and_then(integer)
                    .ok_or_else(|| format!("{}: the integer key {key} is missing", path.display()))
            };
            let count = split("split.count")?;
            if usize::try_from(count) != Ok(paths.len()) {
                return Err(format!(
                    "{}: split.count is {count}, but {} shard paths were given",
                    path.display(),
                    paths.len()
                ));
            }
            let number = split("split.no")?;
            if usize::try_from(number) != Ok(place) {
                return Err(format!(
                    "{}: split.no is {number}, but the shard is at place {place} in the list. Give the shards in \
                     split order.",
                    path.display()
                ));
            }
            for (key, _) in header.metadata.iter() {
                // The keys split.* differ from shard to shard, so they stay out of the model's keys.
                if key.starts_with("split.") {
                    continue;
                }
                if let Some(first) = model.key_shard.insert(key.to_owned(), place) {
                    return Err(format!(
                        "{}: the key {key} is also in {}. Each key other than split.* must be in one shard.",
                        path.display(),
                        paths[first].display()
                    ));
                }
            }
            for tensor in header.tensors {
                if model.index.insert(tensor.name.clone(), model.tensors.len()).is_some() {
                    return Err(format!(
                        "{}: the tensor name {} is in two shards",
                        path.display(),
                        tensor.name
                    ));
                }
                model.tensors.push(tensor);
            }
            model.metadata.push(header.metadata);
            model.shards.push(Shard {
                map,
                path: path.clone(),
                #[cfg(test)]
                data_offset: header.data_offset,
            });
        }
        Ok(model)
    }

    /// The shard paths, in split order.
    pub(crate) fn shard_paths(&self) -> impl Iterator<Item = &Path> {
        self.shards.iter().map(|shard| shard.path.as_path())
    }

    /// Each tensor, in file order: the first shard's tensors, then the second shard's, and so on.
    pub(crate) fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The index of the tensor `name`.
    #[cfg(test)]
    pub(crate) fn find(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// The bytes of tensor `index`, in its shard's map.
    pub(crate) fn tensor_bytes(&self, index: usize) -> &[u8] {
        let tensor = &self.tensors[index];
        let start = tensor.offset as usize;
        &self.shards[tensor.shard].map.bytes()[start..start + tensor.n_bytes as usize]
    }

    /// The value of the metadata key `key`, or `None` when no shard holds it.
    fn value(&self, key: &str) -> Option<&Value> {
        let shard = *self.key_shard.get(key)?;
        self.metadata[shard].get(key)
    }

    /// The integer value of `key`.
    pub(crate) fn int(&self, key: &str) -> Result<i64, String> {
        let value = self
            .value(key)
            .ok_or_else(|| format!("the GGUF key {key} is missing"))?;
        integer(value).ok_or_else(|| format!("the GGUF key {key} is not an integer: {value:?}"))
    }

    /// The values of `key`, an array of integers.
    pub(crate) fn int_array(&self, key: &str) -> Result<Vec<i64>, String> {
        match self.value(key) {
            None => Err(format!("the GGUF key {key} is missing")),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| integer(item).ok_or_else(|| format!("the GGUF array {key} holds {item:?}, not an integer")))
                .collect(),
            Some(other) => Err(format!("the GGUF key {key} is not an array: {other:?}")),
        }
    }

    /// Dequantize rows `[row0, row1)` of tensor `index` into `out`, which holds `(row1 - row0) * ne[0]` values.
    ///
    /// The tests compare these rows with the reference's C++ copy. The engine calls `dequant_f32` on its PLE rows directly.
    #[cfg(test)]
    pub(crate) fn dequant_rows(&self, index: usize, row0: u64, row1: u64, out: &mut [f32]) -> Result<(), String> {
        let tensor = &self.tensors[index];
        let ne0 = tensor.ne[0];
        if !ne0.is_multiple_of(tensor.kind.block_values()) {
            return Err(format!(
                "the tensor {}: ne[0] = {ne0} is not whole blocks of {} ({})",
                tensor.name,
                tensor.kind.block_values(),
                tensor.kind.name()
            ));
        }
        if row1 < row0 || row1 > tensor.rows() {
            return Err(format!(
                "the tensor {}: the rows [{row0}, {row1}) are not inside its {} rows",
                tensor.name,
                tensor.rows()
            ));
        }
        let values = ((row1 - row0) * ne0) as usize;
        if out.len() != values {
            return Err(format!(
                "the tensor {}: {} values of output do not hold {values}",
                tensor.name,
                out.len()
            ));
        }
        let row_bytes = ne0 / tensor.kind.block_values() * tensor.kind.block_bytes();
        let bytes = &self.tensor_bytes(index)[(row0 * row_bytes) as usize..(row1 * row_bytes) as usize];
        dequant_f32(tensor.kind, bytes, out);
        Ok(())
    }

    /// Ask the kernel to drop the page-cache pages that lie wholly inside `[p, p + n)`.
    ///
    /// The range must be inside one shard's map. A page shared with bytes outside the range stays, so a neighbour
    /// tensor keeps its edge pages: the engine reads the PLE table after the load. The call drops this map's view of
    /// the pages first (`MADV_DONTNEED`), because the kernel keeps a mapped page. Then it advises the file
    /// (`POSIX_FADV_DONTNEED`). The kernel can keep a page that another process maps. The map stays valid, and a
    /// later read faults the page in from the file.
    pub(crate) fn drop_cached(&self, p: *const u8, n: u64) -> Result<(), String> {
        let address = p as usize;
        let shard = self
            .shards
            .iter()
            .find(|shard| (shard.map.ptr as usize..shard.map.ptr as usize + shard.map.len).contains(&address))
            .ok_or_else(|| format!("drop_cached: the range at {address:#x} is not inside the map of a shard"))?;
        let low = (address - shard.map.ptr as usize) as u64;
        let size = shard.map.len as u64;
        if n > size - low {
            return Err(format!(
                "drop_cached: the range [{low}, +{n}) runs past the {size} bytes of {}",
                shard.path.display()
            ));
        }
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let Some((first, end)) = inward_pages(low, n, page) else {
            return Ok(());
        };
        // SAFETY: [first, end) is inside the map, and the map is private and read-only, so its pages fault in again.
        let code = unsafe {
            libc::madvise(
                (shard.map.ptr as *mut u8).add(first as usize) as *mut c_void,
                (end - first) as usize,
                libc::MADV_DONTNEED,
            )
        };
        if code != 0 {
            return Err(format!(
                "drop_cached: madvise {}: {}",
                shard.path.display(),
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: the fd is open for the map's life.
        let code = unsafe {
            libc::posix_fadvise(
                shard.map.file.as_raw_fd(),
                first as libc::off_t,
                (end - first) as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        if code != 0 {
            return Err(format!(
                "drop_cached: posix_fadvise {}: {}",
                shard.path.display(),
                std::io::Error::from_raw_os_error(code)
            ));
        }
        Ok(())
    }

    /// The data offset of each shard, in split order.
    #[cfg(test)]
    pub(crate) fn data_offsets(&self) -> Vec<u64> {
        self.shards.iter().map(|shard| shard.data_offset).collect()
    }

    /// The size of each shard, in split order.
    #[cfg(test)]
    pub(crate) fn sizes(&self) -> Vec<u64> {
        self.shards.iter().map(|shard| shard.map.len as u64).collect()
    }
}

/// The whole pages of `page` bytes inside `[low, low + n)`, as `[first, end)`, or `None` when no page is whole.
fn inward_pages(low: u64, n: u64, page: u64) -> Option<(u64, u64)> {
    let first = low.div_ceil(page) * page;
    let end = (low + n) / page * page;
    (end > first).then_some((first, end))
}

/// The text keys of a shard's GGUF metadata that the engine identity holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextKeys {
    /// The hash of the tokenizer keys.
    pub tokenizer_hash: String,
    /// The text of the key `tokenizer.chat_template`.
    pub template: String,
}

/// The tokenizer hash and the chat template in a shard's GGUF metadata, for the engine identity.
///
/// The text library computes the hash (`nagaya_text_qwen38::tokenizer_hash`), so the identity and the library
/// agree. The metadata is in the first shard. The hash covers these keys of the file: `tokenizer.ggml.model`, `.pre`,
/// `.tokens`, `.token_type` and `.merges`, which the file must hold, and `.bos_token_id`, `.eos_token_id`,
/// `.eot_token_id`, `.padding_token_id` and `.unknown_token_id`, which it may hold. The template is the text of
/// `tokenizer.chat_template`, as the text library reads it. The file must hold it as a string.
pub(crate) fn text_keys(shard: &Path) -> Result<TextKeys, String> {
    let metadata = nagaya_text_qwen38::gguf::Metadata::open(shard)
        .map_err(|error| format!("the tokenizer of {} is not readable: {error}", shard.display()))?;
    let tokenizer_hash = nagaya_text_qwen38::tokenizer_hash(&metadata)
        .map_err(|error| format!("the tokenizer hash of {} is not available: {error}", shard.display()))?;
    let template = match metadata.get("tokenizer.chat_template") {
        Some(nagaya_text_qwen38::gguf::Value::Str(text)) => text.clone(),
        Some(_) => {
            return Err(format!(
                "the key tokenizer.chat_template of {} is not a string. Use a GGUF made for this family.",
                shard.display()
            ));
        }
        None => {
            return Err(format!(
                "the key tokenizer.chat_template of {} is missing. Use a GGUF made for this family.",
                shard.display()
            ));
        }
    };
    Ok(TextKeys {
        tokenizer_hash,
        template,
    })
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

    use std::path::{Path, PathBuf};

    use super::{Model, inward_pages, read_header, spot_digest, text_keys};

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

    /// Write a GGUF v3 file with no tensors and one metadata value: an array nested `depth` times.
    fn write_nested_gguf(path: &std::path::Path, depth: u32) {
        fn nest(bytes: &mut Vec<u8>, depth: u32) {
            let element = if depth > 1 { 9u32 } else { 4u32 };
            bytes.extend_from_slice(&element.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes()); // one element
            if depth > 1 {
                nest(bytes, depth - 1);
            } else {
                bytes.extend_from_slice(&0x2au32.to_le_bytes());
            }
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one metadata key
        let key = b"nested";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        nest(&mut bytes, depth);
        let padded = bytes.len().div_ceil(32) * 32;
        bytes.resize(padded, 0);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_metadata_array_that_nests_too_deep_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let shallow = directory.path().join("shallow.gguf");
        write_nested_gguf(&shallow, 4);
        assert!(read_header(&shallow).is_ok(), "a shallow array is fine");
        // A crafted file must not exhaust the stack. The family library refuses an array at depth 16, so 17 nested
        // arrays fail. The bound is deeper than any key that the engine writes.
        let deep = directory.path().join("deep.gguf");
        write_nested_gguf(&deep, 17);
        let error = read_header(&deep).unwrap_err();
        assert!(error.contains("nest"), "{error}");
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
        assert!(error.contains("magic") && error.contains("GGUF"), "{error}");
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
    fn gguf_string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }

    /// Write a GGUF v3 file with the tokenizer keys of a three-token vocabulary. `skip` leaves out one key.
    fn write_tokenizer_gguf(path: &std::path::Path, skip: Option<&str>) {
        write_text_gguf(path, skip, "{{ messages }}");
    }

    /// The same file, with the given chat template.
    fn write_text_gguf(path: &std::path::Path, skip: Option<&str>, template: &str) {
        let strings = |items: &[&str]| {
            let mut out = 9u32.to_le_bytes().to_vec();
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                out.extend(gguf_string(item));
            }
            out
        };
        let mut types = 9u32.to_le_bytes().to_vec();
        types.extend_from_slice(&5u32.to_le_bytes());
        types.extend_from_slice(&3u64.to_le_bytes());
        for kind in [1i32, 1, 3] {
            types.extend_from_slice(&kind.to_le_bytes());
        }
        let text = |value: &str| {
            let mut out = 8u32.to_le_bytes().to_vec();
            out.extend(gguf_string(value));
            out
        };
        let number = |value: u32| {
            let mut out = 4u32.to_le_bytes().to_vec();
            out.extend_from_slice(&value.to_le_bytes());
            out
        };
        let entries: Vec<(&str, Vec<u8>)> = vec![
            ("tokenizer.ggml.model", text("gpt2")),
            ("tokenizer.ggml.pre", text("qwen35")),
            ("tokenizer.ggml.tokens", strings(&["a", "b", "<|x|>"])),
            ("tokenizer.ggml.token_type", types),
            ("tokenizer.ggml.merges", strings(&["a b"])),
            ("tokenizer.chat_template", text(template)),
            ("tokenizer.ggml.bos_token_id", number(7)),
            ("tokenizer.ggml.eos_token_id", number(8)),
        ];
        let entries: Vec<_> = entries.into_iter().filter(|(key, _)| Some(*key) != skip).collect();
        let mut bytes = 0x4655_4747u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            bytes.extend(gguf_string(key));
            bytes.extend(value);
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// The hash that the text library gives this file. It is the golden value of that library's own test, which
    /// an independent Python writer made.
    const FIXTURE_HASH: &str = "sha256:018a61bfe25a6ed49d01bfd9c5f1415e6f99f365646354005fa617aed73a59a8";

    #[test]
    fn the_tokenizer_hash_of_a_shard_is_the_hash_of_the_text_library() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.gguf");
        write_tokenizer_gguf(&path, None);
        let from_mk = text_keys(&path).unwrap().tokenizer_hash;
        let metadata = nagaya_text_qwen38::gguf::Metadata::open(&path).unwrap();
        assert_eq!(from_mk, nagaya_text_qwen38::tokenizer_hash(&metadata).unwrap());
        assert_eq!(from_mk, FIXTURE_HASH);
    }

    #[test]
    fn a_shard_without_a_tokenizer_key_is_an_error_that_names_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.gguf");
        write_tokenizer_gguf(&path, Some("tokenizer.ggml.merges"));
        let error = text_keys(&path).expect_err("a file without merges has no hash");
        assert!(error.contains("tokenizer.ggml.merges"), "{error}");
        assert!(error.contains(path.to_str().unwrap()), "{error}");
    }

    #[test]
    fn a_missing_shard_has_no_tokenizer_hash() {
        let error = text_keys(std::path::Path::new("/nonexistent/shard.gguf")).expect_err("a missing file");
        assert!(error.contains("/nonexistent/shard.gguf"), "{error}");
    }

    #[test]
    fn the_template_is_the_text_of_the_chat_template_key() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.gguf");
        write_text_gguf(&path, None, "<|im_start|>{{ x }}");
        assert_eq!(text_keys(&path).unwrap().template, "<|im_start|>{{ x }}");
    }

    #[test]
    fn a_shard_without_a_chat_template_is_an_error_that_names_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.gguf");
        write_text_gguf(&path, Some("tokenizer.chat_template"), "unused");
        let error = text_keys(&path).expect_err("a file without a template has no identity");
        assert!(error.contains("tokenizer.chat_template"), "{error}");
        assert!(error.contains(path.to_str().unwrap()), "{error}");
    }

    #[test]
    fn two_files_that_differ_only_in_the_chat_template_have_two_identities() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("one.gguf");
        let second = directory.path().join("two.gguf");
        write_text_gguf(&first, None, "template one");
        write_text_gguf(&second, None, "template two");
        let shards = vec![first.clone()];
        let build = crate::binding::Build {
            library_sha256: "sha",
            compiler: "clang",
            hip: "hip",
        };
        let identity_of = |path: &std::path::Path| {
            crate::spec::identity(&shards, &build, true, true, "spot", &text_keys(path).unwrap())
        };
        let (one, two) = (identity_of(&first), identity_of(&second));
        assert_eq!(one.tokenizer_hash, two.tokenizer_hash);
        assert_ne!(one.template, two.template);
        assert_ne!(one, two);
        assert_eq!(one.canary_hash, "");
    }

    /// One metadata entry for `build`: the key, the GGUF value type and the value bytes.
    type Entry = (String, u32, Vec<u8>);

    fn u16_entry(key: &str, value: u16) -> Entry {
        (key.to_owned(), 2, value.to_le_bytes().to_vec())
    }

    fn u32_entry(key: &str, value: u32) -> Entry {
        (key.to_owned(), 4, value.to_le_bytes().to_vec())
    }

    fn i64_array_entry(key: &str, values: &[i64]) -> Entry {
        let mut bytes = 11u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        (key.to_owned(), 9, bytes)
    }

    /// One tensor record for `build`: the name, the dimensions, the ggml type id and the offset in the data.
    struct Record {
        name: String,
        ne: Vec<u64>,
        kind: u32,
        offset: u64,
    }

    fn record(name: &str, ne: &[u64], kind: u32, offset: u64) -> Record {
        Record {
            name: name.to_owned(),
            ne: ne.to_vec(),
            kind,
            offset,
        }
    }

    /// A GGUF v3 file with these entries and records, its data aligned to `alignment`, then `data`.
    fn build(entries: &[Entry], records: &[Record], alignment: u64, data: &[u8]) -> Vec<u8> {
        let mut bytes = 0x4655_4747u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, kind, value) in entries {
            bytes.extend(gguf_string(key));
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(value);
        }
        for record in records {
            bytes.extend(gguf_string(&record.name));
            bytes.extend_from_slice(&(record.ne.len() as u32).to_le_bytes());
            for dimension in &record.ne {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&record.kind.to_le_bytes());
            bytes.extend_from_slice(&record.offset.to_le_bytes());
        }
        bytes.resize(bytes.len().div_ceil(alignment as usize) * alignment as usize, 0);
        bytes.extend_from_slice(data);
        bytes
    }

    /// The split keys of shard `number` of `count`.
    fn split(number: u16, count: u16) -> Vec<Entry> {
        vec![u16_entry("split.no", number), u16_entry("split.count", count)]
    }

    /// One Q8_0 block: the scale `d` as f16 bits, then 32 values.
    fn q8_block(d: u16, values: impl Fn(usize) -> i8) -> Vec<u8> {
        let mut block = d.to_le_bytes().to_vec();
        block.extend((0..32).map(|j| values(j) as u8));
        block
    }

    /// Write three shards: the first holds the keys and an F32 tensor, the second two Q8_0 tensors of two rows each,
    /// and the third holds a key and no tensor.
    fn three_shards(directory: &Path) -> Vec<PathBuf> {
        let mut first_entries = split(0, 3);
        first_entries.push(u32_entry("qwen4exp.ple.eos_token_id", 248_044));
        first_entries.push(i64_array_entry("qwen4exp.ple.layer_multipliers", &[3, -5, 7]));
        let f32_data: Vec<u8> = [1.5f32, -2.0, 0.25, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let first = build(&first_entries, &[record("norm.weight", &[4], 0, 0)], 32, &f32_data);

        // 0x3800 is 0.5 in f16. Each tensor is two rows of one block. The second tensor starts at the aligned 96.
        let mut q8_data = Vec::new();
        q8_data.extend(q8_block(0x3800, |j| j as i8 - 16));
        q8_data.extend(q8_block(0x3800, |j| -(j as i8)));
        q8_data.resize(96, 0);
        q8_data.extend(q8_block(0x3c00, |j| j as i8));
        q8_data.extend(q8_block(0x3c00, |_| 1));
        let mut second_entries = split(1, 3);
        second_entries.push(u32_entry("general.alignment", 32));
        let second = build(
            &second_entries,
            &[record("a.weight", &[32, 2], 8, 0), record("b.weight", &[32, 2], 8, 96)],
            32,
            &q8_data,
        );

        let mut third_entries = split(2, 3);
        third_entries.push(u32_entry("general.name", 1));
        let third = build(&third_entries, &[], 32, &[]);

        [first, second, third]
            .into_iter()
            .enumerate()
            .map(|(place, bytes)| {
                let path = directory.join(format!("shard-{place}.gguf"));
                std::fs::write(&path, bytes).unwrap();
                path
            })
            .collect()
    }

    #[test]
    fn a_model_lists_the_tensors_of_every_shard_in_file_order() {
        let directory = tempfile::tempdir().unwrap();
        let paths = three_shards(directory.path());
        let model = Model::open(&paths).unwrap();
        let names: Vec<(&str, usize, u64, u64)> = model
            .tensors()
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.shard, tensor.n_bytes, tensor.rows()))
            .collect();
        assert_eq!(
            names,
            [
                ("norm.weight", 0, 16, 1),
                ("a.weight", 1, 68, 2),
                ("b.weight", 1, 68, 2)
            ]
        );
        let data = model.data_offsets();
        // A tensor's offset is its first byte in the file: the data offset plus the offset of its record.
        assert_eq!(model.tensors()[0].offset, data[0]);
        assert_eq!(model.tensors()[1].offset, data[1]);
        assert_eq!(model.tensors()[2].offset, data[1] + 96);
        assert_eq!(model.sizes()[2], data[2], "the third shard holds no tensor data");
        assert_eq!(model.find("b.weight"), Some(2));
        assert_eq!(model.find("c.weight"), None);
        assert_eq!(
            model.tensor_bytes(0),
            [1.5f32, -2.0, 0.25, 8.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        );
        assert_eq!(model.shard_paths().collect::<Vec<_>>(), paths);
    }

    #[test]
    fn a_model_reads_the_keys_of_any_shard_but_not_the_split_keys() {
        let directory = tempfile::tempdir().unwrap();
        let model = Model::open(&three_shards(directory.path())).unwrap();
        assert_eq!(model.int("qwen4exp.ple.eos_token_id").unwrap(), 248_044);
        assert_eq!(model.int("general.name").unwrap(), 1, "a key of the third shard");
        assert_eq!(model.int_array("qwen4exp.ple.layer_multipliers").unwrap(), [3, -5, 7]);
        let error = model.int("split.no").unwrap_err();
        assert!(error.contains("missing"), "{error}");
        let error = model.int("qwen4exp.ple.layer_multipliers").unwrap_err();
        assert!(error.contains("not an integer"), "{error}");
        let error = model.int_array("general.alignment").unwrap_err();
        assert!(error.contains("not an array"), "{error}");
    }

    #[test]
    fn a_row_range_dequantizes_to_the_scale_times_the_values() {
        let directory = tempfile::tempdir().unwrap();
        let model = Model::open(&three_shards(directory.path())).unwrap();
        let mut both = vec![0f32; 64];
        model.dequant_rows(1, 0, 2, &mut both).unwrap();
        let want: Vec<f32> = (0..32)
            .map(|j| (j as f32 - 16.0) * 0.5)
            .chain((0..32).map(|j| -(j as f32) * 0.5))
            .collect();
        assert_eq!(both, want);
        let mut second = vec![0f32; 32];
        model.dequant_rows(2, 1, 2, &mut second).unwrap();
        assert_eq!(second, vec![1.0f32; 32]);
        let mut norm = vec![0f32; 4];
        model.dequant_rows(0, 0, 1, &mut norm).unwrap();
        assert_eq!(norm, [1.5, -2.0, 0.25, 8.0]);

        let error = model.dequant_rows(1, 1, 3, &mut both).unwrap_err();
        assert!(error.contains("not inside its 2 rows"), "{error}");
        let error = model.dequant_rows(1, 0, 1, &mut both).unwrap_err();
        assert!(error.contains("do not hold 32"), "{error}");
    }

    /// Write shards with `edit` applied to the entries and records of each, and give the error of `Model::open`.
    fn open_error(edit: impl Fn(usize, &mut Vec<Entry>, &mut Vec<Record>)) -> String {
        let directory = tempfile::tempdir().unwrap();
        let paths: Vec<PathBuf> = (0..2)
            .map(|place| {
                let mut entries = split(place as u16, 2);
                let mut records = vec![record(&format!("t{place}"), &[4], 0, 0)];
                edit(place, &mut entries, &mut records);
                let path = directory.path().join(format!("{place}.gguf"));
                std::fs::write(&path, build(&entries, &records, 32, &[0u8; 64])).unwrap();
                path
            })
            .collect();
        assert!(Model::open(&paths).is_err(), "the edit must make the shards invalid");
        Model::open(&paths).err().unwrap()
    }

    #[test]
    fn a_model_whose_split_keys_do_not_match_the_list_is_refused() {
        let error = open_error(|_, entries, _| entries[1] = u16_entry("split.count", 3));
        assert!(error.contains("split.count is 3, but 2 shard paths"), "{error}");
        let error = open_error(|place, entries, _| entries[0] = u16_entry("split.no", 1 - place as u16));
        assert!(
            error.contains("split.no is 1") && error.contains("split order"),
            "{error}"
        );
        let error = open_error(|place, entries, _| {
            if place == 1 {
                entries.remove(0);
            }
        });
        assert!(error.contains("split.no is missing"), "{error}");
    }

    #[test]
    fn a_key_or_a_tensor_in_two_shards_is_refused() {
        let error = open_error(|_, entries, _| entries.push(u32_entry("general.alignment", 32)));
        assert!(error.contains("the key general.alignment is also in"), "{error}");
        let error = open_error(|_, _, records| records[0].name = "same".to_owned());
        assert!(error.contains("the tensor name same is in two shards"), "{error}");
    }

    #[test]
    fn a_tensor_record_that_the_file_does_not_hold_is_refused() {
        type Edit = fn(&mut Record);
        let cases: [(&str, Edit); 6] = [
            ("unknown ggml type id 1", |r| r.kind = 1),
            ("not aligned to 32", |r| r.offset = 16),
            ("runs past the file size", |r| r.offset = 64),
            ("not whole blocks of 32 (Q8_0)", |r| r.kind = 8),
            ("has 5 dimensions", |r| r.ne = vec![1, 1, 1, 1, 4]),
            ("has 0 values", |r| r.ne = vec![0]),
        ];
        for (needle, edit) in cases {
            let error = open_error(|place, _, records| {
                if place == 1 {
                    edit(&mut records[0]);
                }
            });
            assert!(error.contains(needle), "want {needle:?}: {error}");
            assert!(error.contains("1.gguf"), "the error names the shard: {error}");
        }
    }

    #[test]
    fn an_alignment_that_is_not_a_power_of_two_is_refused() {
        let error = open_error(|_, entries, _| entries.push(u32_entry("general.alignment", 48)));
        assert!(error.contains("general.alignment 48 is not a power of two"), "{error}");
        let error = open_error(|_, entries, _| entries.push(u16_entry("general.alignment", 32)));
        assert!(error.contains("general.alignment is not a u32"), "{error}");
    }

    #[test]
    fn a_drop_takes_only_the_whole_pages_inside_the_range() {
        assert_eq!(inward_pages(0, 8192, 4096), Some((0, 8192)));
        assert_eq!(inward_pages(1, 8192, 4096), Some((4096, 8192)));
        assert_eq!(
            inward_pages(1, 8190, 4096),
            None,
            "a range that holds no whole page drops nothing"
        );
        assert_eq!(inward_pages(4095, 4097, 4096), Some((4096, 8192)));
        assert_eq!(inward_pages(4096, 4095, 4096), None);
    }

    #[test]
    fn a_drop_outside_the_maps_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let model = Model::open(&three_shards(directory.path())).unwrap();
        let bytes = model.tensor_bytes(1);
        model.drop_cached(bytes.as_ptr(), bytes.len() as u64).unwrap();
        let outside = [0u8; 8];
        let error = model.drop_cached(outside.as_ptr(), 8).unwrap_err();
        assert!(error.contains("not inside the map of a shard"), "{error}");
        let error = model.drop_cached(bytes.as_ptr(), 1 << 20).unwrap_err();
        assert!(error.contains("runs past"), "{error}");
        // The map stays valid after a drop: a read faults the pages in again.
        assert_eq!(model.tensor_bytes(1)[..2], [0x00, 0x38]);
    }
}
