//! Memory-mapped safetensors checkpoints (single file or `model.safetensors.index.json` shards)
//! with one tensor table parsed at open.

#![allow(dead_code)]

use crate::safetensors::{Dtype, MappedSafeTensors};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

struct TensorEntry {
    shard: usize,
    range: Range<usize>,
    dtype: Dtype,
    shape: Vec<usize>,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    shards: Vec<MappedSafeTensors>,
    tensors: HashMap<String, TensorEntry>,
    /// sha256 over each shard's header sha256, in shard-name order: tensor names, dtypes, shapes,
    /// and offsets, not weight values.
    pub header_hash: [u8; 32],
}

impl Checkpoint {
    /// Opens `model.safetensors.index.json` and its shards, or a lone `model.safetensors`.
    pub fn open_dir(dir: &Path) -> Result<Self, String> {
        let index_path = dir.join("model.safetensors.index.json");
        let shard_files: Vec<String> = if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct RawIndex {
                weight_map: HashMap<String, String>,
            }
            let raw = std::fs::read_to_string(&index_path)
                .map_err(|e| format!("Failed to read {}: {}", index_path.display(), e))?;
            let parsed: RawIndex =
                serde_json::from_str(&raw).map_err(|e| format!("Failed to parse {}: {}", index_path.display(), e))?;
            let mut files: Vec<String> = parsed.weight_map.into_values().collect();
            files.sort();
            files.dedup();
            files
        } else {
            vec!["model.safetensors".to_string()]
        };
        let paths: Vec<PathBuf> = shard_files.iter().map(|f| dir.join(f)).collect();
        Self::open_files(dir, &paths)
    }

    pub fn open_files(dir: &Path, paths: &[PathBuf]) -> Result<Self, String> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        let mut hasher = Sha256::new();
        for (i, path) in paths.iter().enumerate() {
            let mapped = MappedSafeTensors::open(path)?;
            {
                let view = mapped.view().map_err(|e| format!("{}: {}", path.display(), e))?;
                hasher.update(view.header_sha256);
                for (name, meta) in &view.tensors {
                    let entry = TensorEntry {
                        shard: i,
                        range: view.tensor_range(name).unwrap(),
                        dtype: meta.dtype,
                        shape: meta.shape.clone(),
                    };
                    if tensors.insert(name.clone(), entry).is_some() {
                        return Err(format!("tensor {} appears in more than one shard", name));
                    }
                }
            }
            shards.push(mapped);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            shards,
            tensors,
            header_hash: hasher.finalize().into(),
        })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn entry(&self, name: &str) -> Result<&TensorEntry, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("{}: missing tensor {}", self.dir.display(), name))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        Ok(&self.entry(name)?.shape)
    }

    /// The bytes of `name`, which must be `BF16` with exactly `shape`.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<&[u8], String> {
        let e = self.entry(name)?;
        if e.dtype != Dtype::BF16 || e.shape != shape {
            return Err(format!(
                "{}: tensor {} expected BF16 {:?}, found {:?} {:?}",
                self.dir.display(),
                name,
                shape,
                e.dtype,
                e.shape
            ));
        }
        Ok(&self.shards[e.shard].as_bytes()[e.range.clone()])
    }

    /// The values of an `F32` tensor and its shape.
    pub fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let e = self.entry(name)?;
        if e.dtype != Dtype::F32 {
            return Err(format!(
                "{}: tensor {} is {:?}, not F32",
                self.dir.display(),
                name,
                e.dtype
            ));
        }
        let bytes = &self.shards[e.shard].as_bytes()[e.range.clone()];
        let values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((values, e.shape.clone()))
    }
}
