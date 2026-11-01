//! The checkpoint hash of the weight files (spec 12.1, 12.2), as the SDXL engine hashes its checkpoint.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// The workers of a file hash. Each holds one chunk at a time, so a hash holds at most 1 GiB of the file.
const HASH_WORKERS: u64 = 4;

/// The BLAKE3 hash of a file, as the SDXL engine hashes its checkpoint: the hash of the hashes of its chunks of 256
/// MiB, in order. Four workers read the chunks, so the memory of a hash stays at 1 GiB. Before 2026-11-01 each chunk
/// had its own thread, and a file of 17.5 GB held 66 chunks at once.
pub fn file_hash(path: &Path) -> Result<blake3::Hash, String> {
    file_hash_in(path, 256 << 20)
}

/// The hash of `file_hash` with chunks of `chunk` bytes. The tests use small chunks.
pub fn file_hash_in(path: &Path, chunk: u64) -> Result<blake3::Hash, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    let chunks = size.div_ceil(chunk).max(1);
    let workers: Vec<Result<Vec<(u64, blake3::Hash)>, String>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..HASH_WORKERS.min(chunks))
            .map(|worker| {
                let file = &file;
                scope.spawn(move || {
                    let mut out = Vec::new();
                    let mut bytes = Vec::new();
                    let mut i = worker;
                    while i < chunks {
                        let start = i * chunk;
                        bytes.resize((size - start).min(chunk) as usize, 0);
                        file.read_exact_at(&mut bytes, start)
                            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                        out.push((i, blake3::hash(&bytes)));
                        i += HASH_WORKERS;
                    }
                    Ok(out)
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().expect("a hash thread panicked"))
            .collect()
    });
    let mut hashes = Vec::with_capacity(chunks as usize);
    for worker in workers {
        hashes.extend(worker?);
    }
    hashes.sort_by_key(|(i, _)| *i);
    assert_eq!(
        hashes.len() as u64,
        chunks,
        "each chunk of {} has a hash",
        path.display()
    );
    let mut all = blake3::Hasher::new();
    for (_, hash) in hashes {
        all.update(hash.as_bytes());
    }
    Ok(all.finalize())
}
