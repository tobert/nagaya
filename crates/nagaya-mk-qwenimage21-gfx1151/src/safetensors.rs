//! A strict reader of safetensors files. It reads the header, checks each tensor, and reads one tensor at a time
//! with positioned reads, so a load holds one tensor in host memory at a time.
//!
//! Each tensor must have a known dtype, a shape whose bytes match its range, and a range inside the data that no
//! other tensor overlaps. Anything else is an error, not a guess (spec 8.5).

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// The dtype of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
    I64,
    U8,
}

impl Dtype {
    fn parse(text: &str) -> Option<Dtype> {
        match text {
            "F32" => Some(Dtype::F32),
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "I64" => Some(Dtype::I64),
            "U8" => Some(Dtype::U8),
            _ => None,
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::I64 => 8,
            Dtype::U8 => 1,
        }
    }
}

/// One tensor of a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// The range of its bytes, from the start of the file.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// An error of a safetensors file.
#[derive(Debug)]
pub struct SafetensorsError {
    pub path: PathBuf,
    pub what: String,
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the safetensors file {}: {}", self.path.display(), self.what)
    }
}

impl std::error::Error for SafetensorsError {}

/// An open safetensors file and its checked header.
pub struct Safetensors {
    path: PathBuf,
    file: File,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl Safetensors {
    /// Open `path` and check its header.
    pub fn open(path: &Path) -> Result<Safetensors, SafetensorsError> {
        let error = |what: String| SafetensorsError {
            path: path.to_owned(),
            what,
        };
        let file = File::open(path).map_err(|e| error(format!("cannot open: {e}")))?;
        let size = file.metadata().map_err(|e| error(format!("cannot stat: {e}")))?.len();
        let mut length = [0u8; 8];
        file.read_exact_at(&mut length, 0)
            .map_err(|e| error(format!("cannot read the header length: {e}")))?;
        let header_len = u64::from_le_bytes(length);
        if header_len > 100 << 20 || 8 + header_len > size {
            return Err(error(format!(
                "a header of {header_len} bytes does not fit a file of {size}"
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|e| error(format!("cannot read the header: {e}")))?;
        let json: serde_json::Value =
            serde_json::from_slice(&header).map_err(|e| error(format!("the header is not JSON: {e}")))?;
        let object = json
            .as_object()
            .ok_or_else(|| error("the header is not an object".to_owned()))?;
        let data = 8 + header_len;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                for (key, text) in value
                    .as_object()
                    .ok_or_else(|| error("__metadata__ is not an object".to_owned()))?
                {
                    let text = text
                        .as_str()
                        .ok_or_else(|| error(format!("the metadata `{key}` is not a string")))?;
                    metadata.insert(key.clone(), text.to_owned());
                }
                continue;
            }
            let bad = |what: &str| error(format!("the tensor `{name}`: {what}"));
            let dtype = value["dtype"]
                .as_str()
                .and_then(Dtype::parse)
                .ok_or_else(|| bad(&format!("the dtype {} is not known", value["dtype"])))?;
            let shape: Vec<usize> = value["shape"]
                .as_array()
                .ok_or_else(|| bad("no shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| bad("a dimension is not a number"))?;
            let offsets = value["data_offsets"].as_array().ok_or_else(|| bad("no data_offsets"))?;
            let (Some(begin), Some(end)) = (
                offsets.first().and_then(|v| v.as_u64()),
                offsets.get(1).and_then(|v| v.as_u64()),
            ) else {
                return Err(bad("the data_offsets are not two numbers"));
            };
            let bytes = shape.iter().product::<usize>() as u64 * dtype.bytes() as u64;
            if offsets.len() != 2 || end < begin || end - begin != bytes || data + end > size {
                return Err(bad(&format!(
                    "the range {begin}..{end} does not hold {bytes} bytes inside the data of {} bytes",
                    size - data
                )));
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start: data + begin,
                    end: data + end,
                },
            );
        }
        let mut ranges: Vec<(u64, u64, &String)> = tensors.iter().map(|(n, t)| (t.start, t.end, n)).collect();
        ranges.sort();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(error(format!(
                    "the tensors `{}` and `{}` overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
        Ok(Safetensors {
            path: path.to_owned(),
            file,
            tensors,
            metadata,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The name and the information of each tensor, in name order.
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// The `__metadata__` of the header.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// The information of `name`, or an error that names it.
    pub fn info(&self, name: &str) -> Result<&TensorInfo, SafetensorsError> {
        self.tensors.get(name).ok_or_else(|| SafetensorsError {
            path: self.path.clone(),
            what: format!("it has no tensor `{name}`"),
        })
    }

    /// The bytes of `name`, after a check of its dtype and shape.
    pub fn read(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<Vec<u8>, SafetensorsError> {
        let info = self.info(name)?;
        if info.dtype != dtype || info.shape != shape {
            return Err(SafetensorsError {
                path: self.path.clone(),
                what: format!(
                    "the tensor `{name}` is {:?} {:?}, and the engine needs {dtype:?} {shape:?}",
                    info.dtype, info.shape
                ),
            });
        }
        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        self.file
            .read_exact_at(&mut bytes, info.start)
            .map_err(|e| SafetensorsError {
                path: self.path.clone(),
                what: format!("cannot read the tensor `{name}`: {e}"),
            })?;
        Ok(bytes)
    }
}

/// The bytes of a safetensors file of `(name, dtype, shape, bytes)` entries, in order, with `metadata`. Tests use it.
pub fn write_bytes(entries: &[(&str, &str, &[usize], &[u8])], metadata: &[(&str, &str)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let meta: serde_json::Map<String, serde_json::Value> = metadata
            .iter()
            .map(|(k, v)| ((*k).to_owned(), serde_json::Value::String((*v).to_owned())))
            .collect();
        header.insert("__metadata__".to_owned(), serde_json::Value::Object(meta));
    }
    let mut offset = 0;
    for (name, dtype, shape, bytes) in entries {
        header.insert(
            (*name).to_owned(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}),
        );
        offset += bytes.len();
    }
    let text = serde_json::to_vec(&header).expect("a header serializes");
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&text);
    for (_, _, _, bytes) in entries {
        out.extend_from_slice(bytes);
    }
    out
}
