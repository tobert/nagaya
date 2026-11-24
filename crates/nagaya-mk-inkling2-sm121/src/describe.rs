//! The identity and the limits of an open, with no load of weights, and the problem that an open would meet now
//! (spec 11.1, 3.3b; engine-abi "The GB10 Inkling 2 binding", "The engine").
//!
//! The rule of the memory check needs no GPU. The two calls that use it open the GPU and its cubin, and they read no
//! weights.

use crate::health::MIN_FREE_AFTER_BYTES;

/// The problem of an open that the free memory cannot hold, or `None`.
///
/// `planned` is the device bytes of the plan. `free` is the free bytes of the GPU now, and `total` is its total.
/// `freed` is the most that the unload of a swap returns. The open fits when the free bytes with `freed` hold the plan
/// and the floor of the engine's memory rule. The message names each number.
pub fn open_memory_problem(planned: u64, free: u64, total: u64, freed: u64) -> Option<String> {
    let available = free.saturating_add(freed);
    if available >= planned.saturating_add(MIN_FREE_AFTER_BYTES) {
        return None;
    }
    Some(format!(
        "the engine needs {planned} bytes of GPU memory for its weights, its slots and its work, and the machine needs \
         {MIN_FREE_AFTER_BYTES} bytes free after the open. The GPU has {free} bytes free of {total}, and the unload of \
         the swap frees at most {freed} bytes. Name fewer slots or slots of fewer positions in `slots`, set `cache` \
         to `bf16` or `e4m3`, name the loaded engines of the group in `unload`, or stop the programs that hold GPU \
         memory"
    ))
}

#[cfg(feature = "cuda")]
pub use gpu::{describe, open_problem};

#[cfg(feature = "cuda")]
mod gpu {
    use nagaya_abi::{Identity, Limits};
    use nagaya_mk_cuda::driver::CudaDevice;

    use super::open_memory_problem;
    use crate::engine::{device_bytes, identity_of};
    use crate::identity;
    use crate::spec::{self, Config, Model, WEIGHTS_FILE};

    /// The identity and the limits that an open of `config` on the GPU at `pci` gives, with no load (spec 11.1).
    ///
    /// It reads the model folder and the digest of the checkpoint through the cache. It opens the GPU, for the driver
    /// version and the count of multiprocessors, and it loads no weights. The limits are the plan.
    pub fn describe(config: &Config, pci: &str) -> Result<(Identity, Limits), String> {
        config.check()?;
        let dir = &config.model_dir;
        let limits = spec::plan(config)?;
        let model = Model::read(dir)?;
        let device = CudaDevice::open_by_pci(pci, config.cache.cubin())?;
        let config_path = dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("cannot read `{}`: {e}", config_path.display()))?;
        let facts = identity::text_facts(dir)?;
        let digest = identity::checkpoint_digest_cached(&config_text, &dir.join(WEIGHTS_FILE))?;
        let identity = identity_of(
            model,
            digest,
            facts,
            config,
            (device.driver_version(), device.sm_count()),
        );
        Ok((identity, limits))
    }

    /// The problem that an open of `config` on the GPU at `pci` would meet now, or `None` (spec 3.3b).
    ///
    /// It compares the planned device bytes with the free bytes of the GPU and `freed_bytes`, with the floor of the
    /// engine's memory rule. It reads no weights. A failure of the plan or of the GPU is a problem too.
    pub fn open_problem(config: &Config, pci: &str, freed_bytes: u64) -> Option<String> {
        let check = || -> Result<Option<String>, String> {
            let limits = spec::plan(config)?;
            let device = CudaDevice::open_by_pci(pci, config.cache.cubin())?;
            let (free, total) = device.memory_info()?;
            Ok(open_memory_problem(
                device_bytes(&limits),
                free as u64,
                total as u64,
                freed_bytes,
            ))
        };
        check().unwrap_or_else(Some)
    }
}
