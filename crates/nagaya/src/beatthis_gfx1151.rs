//! The Beat This! engine on one gfx1151 card as a factory kind (spec 11.19). The plan needs no ROCm, so the host
//! tests cover it. The factory opens the card only in a build with the feature `hip`.

use nagaya_abi::Limits;

use crate::admission::{GroupDevice, PlanError};
use crate::beatthis::weights;

/// The engine kind in a placement profile (spec 11.19).
pub const KIND: &str = "mk-beatthis-gfx1151";

/// The limits of the engine on one card. It reads the header of the weights file, and no weight bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    if devices.len() != 1 {
        return Err(PlanError {
            message: format!(
                "the engine `{KIND}` runs on one card, and its profile names {} devices: name one device",
                devices.len()
            ),
        });
    }
    let path = weights(table)?;
    let config = nagaya_mk_beatthis_cpu::engine::config_of(&path).map_err(|message| PlanError { message })?;
    Ok(nagaya_mk_beatthis_gfx1151::limits(&config))
}

#[cfg(feature = "hip")]
mod factory {
    use nagaya_abi::{Engine, Limits};
    use nagaya_mk_beatthis_gfx1151::engine::{BeatThisGpuEngine, OpenFailure};

    use super::KIND;
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};
    use crate::beatthis::weights;

    /// The factory of the Beat This! engine on gfx1151.
    pub struct BeatThisGfx1151Factory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for BeatThisGfx1151Factory {
        /// The served model is the engine name (spec 11.19).
        fn served_names(&self, engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
            Some(vec![engine.to_owned()])
        }

        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Upload the prepared weights to the card. The machine has one gfx1151 card, which is the current device.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            super::plan(table, devices).map_err(|e| open_error(e.message))?;
            let path = weights(table).map_err(|e| open_error(e.message))?;
            // A canary refusal stops the start with exit 3 and no restart (spec 8.1, 10.4).
            let engine = BeatThisGpuEngine::open(&path).map_err(|failure| match failure {
                OpenFailure::Canary(refusal) => OpenError {
                    kind: OpenErrorKind::Canary,
                    message: refusal.to_string(),
                },
                OpenFailure::Other(message) => open_error(format!("`{KIND}` cannot open: {message}")),
            })?;
            let identity = nagaya_abi::Engine::identity(&engine);
            tracing::info!(
                "the Beat This! engine passed its canary at load canary_hash={} checkpoint={} target=gfx1151",
                identity.canary_hash,
                identity.checkpoint_digest
            );
            Ok(Box::new(engine))
        }
    }
}

#[cfg(feature = "hip")]
pub use factory::BeatThisGfx1151Factory;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_needs_one_device() {
        let e = plan(&"weights = \"/m/x\"".parse().unwrap(), &[]).unwrap_err();
        assert!(
            e.message.contains("one card") && e.message.contains("0 devices"),
            "{}",
            e.message
        );
    }
}
