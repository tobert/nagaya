//! Minimal read-only safetensors header and tensor slice reader.
//!
//! Reading the header reads no weight bytes, so engine `plan` calls stay fast and need no GPU (spec 2.4).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::Deserialize;

/// Metadata for one tensor in a safetensors file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte offset from the start of the tensor data section (after the 8-byte length and JSON header).
    pub data_offset: u64,
    /// Byte length of the tensor payload.
    pub nbytes: u64,
}

/// Parsed header of one safetensors file.
#[derive(Clone, Debug)]
pub struct SafeTensorsHeader {
    pub data_start: u64,
    pub file_len: u64,
    pub tensors: BTreeMap<String, TensorMeta>,
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Maximum header length accepted (64 MiB), preventing malformed files from allocating unbounded RAM.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

impl SafeTensorsHeader {
    /// Read and validate only the JSON header of `path`. Reads no weight bytes.
    pub fn read(path: &Path) -> Result<Self, String> {
        let mut file =
            File::open(path).map_err(|e| format!("cannot open safetensors file `{}`: {e}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("cannot stat safetensors file `{}`: {e}", path.display()))?
            .len();
        if file_len < 8 {
            return Err(format!(
                "safetensors file `{}` is truncated ({file_len} bytes, need at least 8)",
                path.display()
            ));
        }
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| format!("cannot read header length of `{}`: {e}", path.display()))?;
        let header_len = u64::from_le_bytes(len_buf);
        if header_len > MAX_HEADER_BYTES || 8u64.saturating_add(header_len) > file_len {
            return Err(format!(
                "safetensors file `{}` has invalid header length {header_len} for file size {file_len}",
                path.display()
            ));
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("cannot read header of `{}`: {e}", path.display()))?;
        let data_start = 8 + header_len;
        let data_len = file_len - data_start;

        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("invalid JSON header in `{}`: {e}", path.display()))?;
        let mut tensors = BTreeMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let entry: RawEntry = serde_json::from_value(value)
                .map_err(|e| format!("invalid tensor entry `{name}` in `{}`: {e}", path.display()))?;
            let [start, end] = entry.data_offsets;
            if start > end || end > data_len {
                return Err(format!(
                    "tensor `{name}` in `{}` has out-of-bounds offsets [{start}, {end}] for data size {data_len}",
                    path.display()
                ));
            }
            tensors.insert(
                name.clone(),
                TensorMeta {
                    name,
                    dtype: entry.dtype,
                    shape: entry.shape,
                    data_offset: start,
                    nbytes: end - start,
                },
            );
        }
        Ok(Self {
            data_start,
            file_len,
            tensors,
        })
    }

    /// Read the raw payload bytes of one tensor from `file`.
    pub fn read_tensor(&self, file: &mut File, name: &str) -> Result<Vec<u8>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor `{name}` is missing from the safetensors header"))?;
        let abs = self.data_start + meta.data_offset;
        file.seek(SeekFrom::Start(abs))
            .map_err(|e| format!("cannot seek to tensor `{name}` at offset {abs}: {e}"))?;
        let mut buf = vec![0u8; meta.nbytes as usize];
        file.read_exact(&mut buf)
            .map_err(|e| format!("cannot read tensor `{name}` ({} bytes): {e}", meta.nbytes))?;
        Ok(buf)
    }
}
