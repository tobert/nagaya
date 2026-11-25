//! The facts of the engine identity that come from the model folder: the digest of the checkpoint (spec 12.2) and the
//! facts of the family library (spec 11.3, 12.1). They need no GPU.
//!
//! The digest is the digest of the B70 engine (`nagaya_mk_inkling2_bmg::engine::checkpoint_digest`): the two engines
//! read one checkpoint, so they name it with one digest. This module reads the file in chunks, so that it needs no
//! map of 19 GB. Proposed: the two engines may share one function later.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use nagaya_mk_cuda::safetensors::{SafeTensorsHeader, TensorMeta};

/// The key of the digest. The B70 engine uses the same key.
const DIGEST_KEY: &str = "nagaya inkling2 checkpoint v1";

/// The bytes that one read takes from the file: 16 MiB.
const CHUNK_BYTES: usize = 16 << 20;

/// The facts of the identity that come from the family library: the tokenizer hash and the template (spec 11.3,
/// 12.1). The registry compares the hash with the family's at the start, so both come from one function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextFacts {
    pub tokenizer_hash: String,
    pub template: String,
}

/// The text facts of the model folder `dir`, from the family library of its files.
pub fn text_facts(dir: &Path) -> Result<TextFacts, String> {
    let family = nagaya_text_inkling2::Inkling2Family::from_dir(dir)
        .map_err(|e| format!("cannot build the Inkling 2 family from {}: {e}", dir.display()))?;
    Ok(TextFacts {
        tokenizer_hash: family.tokenizer_hash().to_owned(),
        template: family.template().to_owned(),
    })
}

/// The name of a safetensors dtype in the digest. It is the name that the B70 reader gives to the dtype, so the two
/// engines write the same digest. A dtype that the B70 reader does not know is an error.
fn dtype_name(tensor: &str, dtype: &str) -> Result<&'static str, String> {
    Ok(match dtype {
        "F32" => "F32",
        "F16" => "F16",
        "BF16" => "BF16",
        "I8" => "I8",
        "U8" => "U8",
        "F8_E4M3" => "F8E4M3",
        "I64" => "I64",
        other => {
            return Err(format!(
                "tensor `{tensor}` has the dtype `{other}`, which the digest of the checkpoint does not know. Use \
                 a checkpoint of Inkling 2 Mini or of the variable-loop research model"
            ));
        }
    })
}

/// The hashes that a worker made, each with the index of its tensor.
type Hashes = Vec<(usize, [u8; 32])>;

/// The BLAKE3 hash of the bytes of one tensor, read from `file` in chunks of `CHUNK_BYTES`.
fn hash_tensor(file: &File, start: u64, meta: &TensorMeta) -> Result<[u8; 32], String> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; CHUNK_BYTES.min(meta.nbytes as usize).max(1)];
    let mut done = 0u64;
    while done < meta.nbytes {
        let take = (meta.nbytes - done).min(buffer.len() as u64) as usize;
        file.read_exact_at(&mut buffer[..take], start + meta.data_offset + done)
            .map_err(|e| format!("cannot read tensor `{}`: {e}", meta.name))?;
        hasher.update(&buffer[..take]);
        done += take as u64;
    }
    Ok(*hasher.finalize().as_bytes())
}

/// The logical digest of the checkpoint (spec 12.2): BLAKE3 with the key `nagaya inkling2 checkpoint v1` over the
/// canonical `config.json` and each tensor of the file `weights`, in bytewise order of its name. A tensor adds its
/// name, its dtype, its shape and the BLAKE3 of its bytes. The tensors of the tower of images are in it too, so step
/// 13 does not change the digest. The two models share every tensor, so `config.json` makes their digests differ.
pub fn checkpoint_digest(config_text: &str, weights: &Path) -> Result<String, String> {
    checkpoint_digest_with(
        config_text,
        weights,
        std::thread::available_parallelism().map_or(4, |n| n.get()).min(16),
    )
}

/// `checkpoint_digest` through the memo of the process and the cache file (`digest_cache`). The first call for a file
/// reads it. A later call for the same file, in this process or in another, reads nothing.
pub fn checkpoint_digest_cached(config_text: &str, weights: &Path) -> Result<String, String> {
    crate::digest_cache::cached_digest(
        &crate::digest_cache::PROCESS_MEMO,
        crate::digest_cache::cache_dir_from_env().as_deref(),
        config_text,
        weights,
        || checkpoint_digest(config_text, weights),
    )
}

/// `checkpoint_digest` with `workers` threads. The digest does not depend on the count.
pub fn checkpoint_digest_with(config_text: &str, weights: &Path, workers: usize) -> Result<String, String> {
    let config: serde_json::Value =
        serde_json::from_str(config_text).map_err(|e| format!("config.json is not JSON: {e}"))?;
    let header = SafeTensorsHeader::read(weights)?;
    // The map is sorted by the bytes of the name.
    let metas: Vec<&TensorMeta> = header.tensors.values().collect();
    let dtypes = metas
        .iter()
        .map(|meta| dtype_name(&meta.name, &meta.dtype))
        .collect::<Result<Vec<_>, _>>()?;
    let file = File::open(weights).map_err(|e| format!("cannot open `{}`: {e}", weights.display()))?;
    let next = AtomicUsize::new(0);
    let results: Vec<Result<Hashes, String>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..workers.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(meta) = metas.get(index) else {
                            return Ok(done);
                        };
                        done.push((index, hash_tensor(&file, header.data_start, meta)?));
                    }
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().expect("a hash thread panicked"))
            .collect()
    });
    let mut hashes: Vec<[u8; 32]> = vec![[0; 32]; metas.len()];
    for result in results {
        for (index, hash) in result? {
            hashes[index] = hash;
        }
    }
    let mut hasher = blake3::Hasher::new_derive_key(DIGEST_KEY);
    let canonical = canonical_json(&config);
    hasher.update(&(canonical.len() as u64).to_le_bytes());
    hasher.update(canonical.as_bytes());
    for ((meta, dtype), hash) in metas.iter().zip(&dtypes).zip(&hashes) {
        hasher.update(&(meta.name.len() as u64).to_le_bytes());
        hasher.update(meta.name.as_bytes());
        hasher.update(&(dtype.len() as u64).to_le_bytes());
        hasher.update(dtype.as_bytes());
        hasher.update(&(meta.shape.len() as u64).to_le_bytes());
        for dim in &meta.shape {
            hasher.update(&(*dim as u64).to_le_bytes());
        }
        hasher.update(hash);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

/// JSON with the keys of each object in bytewise order and no whitespace.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let items: Vec<String> = keys
                .into_iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        serde_json::Value::String(key.clone()),
                        canonical_json(&map[key])
                    )
                })
                .collect();
            format!("{{{}}}", items.join(","))
        }
        serde_json::Value::Array(items) => {
            let items: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", items.join(","))
        }
        other => other.to_string(),
    }
}
