//! Zero-copy `safetensors` reader over `mmap` (or in-memory buffers).
//!
//! The header is parsed with `serde_json`. Every tensor must have a known dtype, offsets
//! inside the file, and a byte length equal to its element count times its dtype size;
//! anything else is an error rather than a guess.

#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I8,
    U8,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Dtype::F32),
            "F16" => Some(Dtype::F16),
            "BF16" => Some(Dtype::BF16),
            "I8" => Some(Dtype::I8),
            "U8" => Some(Dtype::U8),
            _ => None,
        }
    }

    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I8 | Dtype::U8 => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

#[derive(serde::Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensorsView<'a> {
    pub raw_bytes: &'a [u8],
    pub header_len: usize,
    pub data_offset: usize,
    pub tensors: HashMap<String, TensorMeta>,
    /// sha256 of the 8-byte length prefix and JSON header: tensor names, dtypes, shapes, and offsets.
    pub header_sha256: [u8; 32],
}

impl<'a> SafeTensorsView<'a> {
    /// Builds an in-memory `.safetensors` buffer for tests:
    /// `[8-byte little-endian header length N][N bytes JSON header][tensor bytes]`
    pub fn build_synthetic_safetensors(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut current_offset = 0usize;
        for (name, dtype_str, shape, bytes) in entries {
            let start = current_offset;
            current_offset += bytes.len();
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype_str, "shape": shape, "data_offsets": [start, current_offset]}),
            );
        }
        let mut json = serde_json::Value::Object(header).to_string();
        while json.len() % 8 != 0 {
            json.push(' ');
        }
        let mut out = Vec::with_capacity(8 + json.len() + current_offset);
        out.extend_from_slice(&(json.len() as u64).to_le_bytes());
        out.extend_from_slice(json.as_bytes());
        for (_, _, _, bytes) in entries {
            out.extend_from_slice(bytes);
        }
        out
    }

    pub fn parse(raw_bytes: &'a [u8]) -> Result<Self, String> {
        if raw_bytes.len() < 8 {
            return Err("buffer too small for the 8-byte header length".to_string());
        }
        let header_len = u64::from_le_bytes(raw_bytes[0..8].try_into().unwrap()) as usize;
        let data_offset = 8usize
            .checked_add(header_len)
            .filter(|&o| o <= raw_bytes.len())
            .ok_or("truncated safetensors header")?;
        let data_len = raw_bytes.len() - data_offset;

        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&raw_bytes[8..data_offset])
            .map_err(|e| format!("invalid safetensors JSON header: {}", e))?;

        let mut tensors = HashMap::with_capacity(header.len());
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry: RawTensorEntry =
                serde_json::from_value(value).map_err(|e| format!("tensor {}: bad header entry: {}", name, e))?;
            let dtype = Dtype::parse(&entry.dtype)
                .ok_or_else(|| format!("tensor {}: unsupported dtype {}", name, entry.dtype))?;
            let [start, end] = entry.data_offsets;
            if start > end || end > data_len {
                return Err(format!(
                    "tensor {}: offsets {}..{} outside {} data bytes",
                    name, start, end, data_len
                ));
            }
            let want = entry.shape.iter().product::<usize>() * dtype.size_bytes();
            if end - start != want {
                return Err(format!(
                    "tensor {}: {} bytes but shape {:?} x {:?} needs {}",
                    name,
                    end - start,
                    entry.shape,
                    dtype,
                    want
                ));
            }
            tensors.insert(
                name.clone(),
                TensorMeta {
                    name,
                    dtype,
                    shape: entry.shape,
                    data_start: start,
                    data_end: end,
                },
            );
        }

        Ok(Self {
            raw_bytes,
            header_len,
            data_offset,
            tensors,
            header_sha256: Sha256::digest(&raw_bytes[..data_offset]).into(),
        })
    }

    /// Byte range of tensor `name` within the whole file.
    pub fn tensor_range(&self, name: &str) -> Option<std::ops::Range<usize>> {
        let meta = self.tensors.get(name)?;
        Some(self.data_offset + meta.data_start..self.data_offset + meta.data_end)
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<&'a [u8]> {
        self.raw_bytes.get(self.tensor_range(name)?)
    }

    /// The bytes of `name`, which must be `BF16` with exactly `shape`.
    pub fn bf16_tensor(&self, name: &str, shape: &[usize]) -> Result<&'a [u8], String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {}", name))?;
        if meta.dtype != Dtype::BF16 || meta.shape != shape {
            return Err(format!(
                "tensor {}: expected BF16 {:?}, found {:?} {:?}",
                name, shape, meta.dtype, meta.shape
            ));
        }
        Ok(self.tensor_bytes(name).expect("range checked in parse"))
    }
}

/// Convert IEEE bfloat16 (`u16`) to `f32` (exact zero-cost 16-bit left shift)
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        length: usize,
        prot: std::ffi::c_int,
        flags: std::ffi::c_int,
        fd: std::ffi::c_int,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, length: usize) -> std::ffi::c_int;
}

const PROT_READ: std::ffi::c_int = 0x1;
const MAP_PRIVATE: std::ffi::c_int = 0x02;
const MAP_FAILED: *mut std::ffi::c_void = !0usize as *mut std::ffi::c_void;

/// Zero-copy POSIX `mmap`-backed `.safetensors` file on Samsung 9100 PRO NVMe.
/// Maps multi-gigabyte checkpoints (e.g. 47 GB `model-00001-of-00002.safetensors`)
/// in microseconds without heap allocation.
pub struct MappedSafeTensors {
    ptr: *mut std::ffi::c_void,
    len: usize,
    _file: std::fs::File,
}

unsafe impl Send for MappedSafeTensors {}
unsafe impl Sync for MappedSafeTensors {}

impl MappedSafeTensors {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let len = file
            .metadata()
            .map_err(|e| format!("Failed to stat {}: {}", path.display(), e))?
            .len() as usize;
        if len < 8 {
            return Err(format!("SafeTensors file {} too small ({} bytes)", path.display(), len));
        }
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr == MAP_FAILED {
            return Err(format!("mmap failed on {} ({} bytes)", path.display(), len));
        }
        Ok(Self { ptr, len, _file: file })
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    pub fn view(&self) -> Result<SafeTensorsView<'_>, String> {
        SafeTensorsView::parse(self.as_bytes())
    }
}

impl Drop for MappedSafeTensors {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.ptr != MAP_FAILED && self.len > 0 {
            unsafe {
                let _ = munmap(self.ptr, self.len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_buffer_and_checks_shapes() {
        let w = [0u8; 2 * 3 * 2];
        let buf = SafeTensorsView::build_synthetic_safetensors(&[("w", "BF16", &[2, 3], &w)]);
        let v = SafeTensorsView::parse(&buf).unwrap();
        assert_eq!(v.bf16_tensor("w", &[2, 3]).unwrap().len(), 12);
        assert!(v.bf16_tensor("w", &[3, 2]).is_err());
        assert!(v.bf16_tensor("missing", &[2, 3]).is_err());
    }

    #[test]
    fn rejects_unknown_dtype_and_length_mismatch() {
        let w = [0u8; 8];
        let buf = SafeTensorsView::build_synthetic_safetensors(&[("w", "F8_E4M3", &[8], &w)]);
        assert!(
            SafeTensorsView::parse(&buf)
                .err()
                .unwrap()
                .contains("unsupported dtype")
        );
        let buf = SafeTensorsView::build_synthetic_safetensors(&[("w", "BF16", &[8], &w)]);
        assert!(SafeTensorsView::parse(&buf).err().unwrap().contains("needs 16"));
    }
}
