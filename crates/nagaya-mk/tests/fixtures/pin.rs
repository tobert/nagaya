//! A fake mk installation: a library that is not an ELF file, three GGUF shards, and their paths.
//!
//! Each pin test is its own test binary, because `open()` caches its result for the life of the process. Thus no
//! other test reads the environment that `set_env` changes.
//!
//! A test uses some of these items only. The others are not dead code.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The SHA-256 of a file, as lowercase hex.
pub fn file_sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap();
    Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Write a minimal GGUF v3 file with no tensors and no metadata. It is 32 bytes, so its data starts at byte 32.
fn write_shard(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(32, 0);
    std::fs::write(path, bytes).unwrap();
}

/// A fake mk installation under a temporary directory.
pub struct FakeInstall {
    /// The temporary directory. It is removed when this value drops.
    pub directory: tempfile::TempDir,
    /// The file at `MK_ENGINE_LIB`. It is not an ELF file, so a load of it fails after the pin check.
    pub library: PathBuf,
    /// The three files at `MK_SHARDS`.
    pub shards: Vec<PathBuf>,
}

impl FakeInstall {
    /// Write the library and the three shards.
    pub fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let library = directory.path().join("libmkengine.so");
        std::fs::write(&library, b"this is not an ELF file\n").unwrap();
        let shards: Vec<PathBuf> = (0..3)
            .map(|index| directory.path().join(format!("shard{index}.gguf")))
            .collect();
        for path in &shards {
            write_shard(path);
        }
        Self {
            directory,
            library,
            shards,
        }
    }

    /// The library's SHA-256, as lowercase hex.
    pub fn digest(&self) -> String {
        file_sha256(&self.library)
    }

    /// Set `MK_ENGINE_LIB` and `MK_SHARDS`, and set `MK_ENGINE_LIB_SHA256` to `pin` or remove it.
    ///
    /// # Safety
    /// The caller must be the only test in this process that changes the environment.
    pub unsafe fn set_env(&self, pin: Option<&str>) {
        unsafe {
            std::env::set_var("MK_ENGINE_LIB", &self.library);
            let joined = self
                .shards
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<String>>()
                .join(":");
            std::env::set_var("MK_SHARDS", joined);
            match pin {
                Some(pin) => std::env::set_var("MK_ENGINE_LIB_SHA256", pin),
                None => std::env::remove_var("MK_ENGINE_LIB_SHA256"),
            }
        }
    }
}
