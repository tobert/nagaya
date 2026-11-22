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

/// The header object as a list of pairs, so that a repeated key is visible. A map would keep only the last one.
struct Entries(Vec<(String, serde_json::Value)>);

impl<'de> Deserialize<'de> for Entries {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Entries;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Entries, A::Error> {
                let mut pairs = Vec::new();
                while let Some(pair) = map.next_entry::<String, serde_json::Value>()? {
                    pairs.push(pair);
                }
                Ok(Entries(pairs))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

/// Bits in one value of a safetensors dtype, or `None` for a dtype that this reader does not know.
fn dtype_bits(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 64,
        "F32" | "I32" | "U32" => 32,
        "F16" | "BF16" | "I16" | "U16" => 16,
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" | "BOOL" => 8,
        "F6_E2M3" | "F6_E3M2" => 6,
        "F4" => 4,
        _ => return None,
    })
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

        let raw: Entries = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("invalid JSON header in `{}`: {e}", path.display()))?;
        let mut tensors = BTreeMap::new();
        for (name, value) in raw.0 {
            if name == "__metadata__" {
                continue;
            }
            if tensors.contains_key(&name) {
                return Err(format!(
                    "the header of `{}` names the tensor `{name}` two times. Write the file again with one entry \
                     for each tensor",
                    path.display()
                ));
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
            let bits = dtype_bits(&entry.dtype).ok_or_else(|| {
                format!(
                    "tensor `{name}` in `{}` has the dtype `{}`, which this reader does not know. Add the dtype to \
                     `dtype_bits` if it is real",
                    path.display(),
                    entry.dtype
                )
            })?;
            let count = entry
                .shape
                .iter()
                .try_fold(1u64, |acc, &dim| acc.checked_mul(dim as u64))
                .ok_or_else(|| format!("tensor `{name}` in `{}` has a shape that overflows", path.display()))?;
            let want_bits = count.checked_mul(bits);
            if want_bits != (end - start).checked_mul(8) {
                return Err(format!(
                    "tensor `{name}` in `{}` has the shape {:?} of {} values and the dtype {}, which need {} bits, but \
                     its offsets [{start}, {end}] hold {} bytes. Write the file again with the right offsets",
                    path.display(),
                    entry.shape,
                    count,
                    entry.dtype,
                    want_bits.map_or("more than 2^64".to_owned(), |b| b.to_string()),
                    end - start
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
        let mut spans: Vec<(u64, u64, &str)> = tensors
            .values()
            .filter(|t| t.nbytes > 0)
            .map(|t| (t.data_offset, t.data_offset + t.nbytes, t.name.as_str()))
            .collect();
        spans.sort_unstable();
        for pair in spans.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(format!(
                    "tensors `{}` and `{}` in `{}` overlap: [{}, {}] and [{}, {}]. Write the file again with \
                     separate offsets",
                    pair[0].2,
                    pair[1].2,
                    path.display(),
                    pair[0].0,
                    pair[0].1,
                    pair[1].0,
                    pair[1].1
                ));
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a safetensors file with `header` and `data_len` zero bytes, and read its header.
    fn read(name: &str, header: &str, data_len: usize) -> Result<SafeTensorsHeader, String> {
        let dir = std::env::temp_dir().join(format!("nagaya-mk-cuda-st-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + data_len, 0);
        std::fs::write(&path, &bytes).unwrap();
        let result = SafeTensorsHeader::read(&path);
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    #[test]
    fn a_shape_that_does_not_fit_its_bytes_is_refused() {
        // 2 x 3 values of F32 need 24 bytes.
        let bad = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,16]}}"#;
        let error = read("shape", bad, 16).unwrap_err();
        assert!(error.contains("`w`") && error.contains("16 bytes"), "{error}");
        assert!(error.contains("192 bits"), "{error}");
        let good = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        assert_eq!(read("shape-ok", good, 24).unwrap().tensors["w"].nbytes, 24);
        // A scalar has an empty shape and one value.
        let scalar = r#"{"s":{"dtype":"BF16","shape":[],"data_offsets":[0,2]}}"#;
        assert!(read("scalar", scalar, 2).is_ok());
        // Two F4 values fill one byte.
        let packed = r#"{"p":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}}"#;
        assert!(read("packed", packed, 2).is_ok());
    }

    #[test]
    fn an_unknown_dtype_is_refused() {
        let header = r#"{"w":{"dtype":"Q9","shape":[1],"data_offsets":[0,1]}}"#;
        let error = read("dtype", header, 1).unwrap_err();
        assert!(error.contains("Q9") && error.contains("does not know"), "{error}");
    }

    #[test]
    fn overlapping_tensors_are_refused_and_touching_ones_are_not() {
        let overlap = r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}"#;
        let error = read("overlap", overlap, 12).unwrap_err();
        assert!(
            error.contains("`a`") && error.contains("`b`") && error.contains("overlap"),
            "{error}"
        );
        let touching = r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[4],"data_offsets":[8,12]},"e":{"dtype":"U8","shape":[0],"data_offsets":[8,8]}}"#;
        assert_eq!(read("touching", touching, 12).unwrap().tensors.len(), 3);
    }

    #[test]
    fn a_repeated_key_is_refused() {
        let header = r#"{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"w":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        let error = read("dup", header, 2).unwrap_err();
        assert!(error.contains("`w`") && error.contains("two times"), "{error}");
    }
}
