//! The identity and the limits of an open, with no load of weights (spec 11.1; engine-abi "The GB10 Inkling 2
//! binding", "The engine"). The call opens the GPU and its cubin, and it reads no weights.

#[cfg(feature = "cuda")]
pub use gpu::describe;

#[cfg(feature = "cuda")]
mod gpu {
    use nagaya_abi::{Identity, Limits};
    use nagaya_mk_cuda::driver::CudaDevice;

    use crate::engine::identity_of;
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
}
