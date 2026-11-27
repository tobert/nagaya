//! The EmbeddingGemma 2 engine on one B70 card as a factory kind, and its embedding service (spec 11.10).
//!
//! The config parse, the plan and the family need no card, so the host tests cover them. The factory opens the card
//! only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::Limits;
use nagaya_family_embeddinggemma2::EmbeddingGemma2Family;
use nagaya_mk_embeddinggemma2_bmg::Config;

use crate::admission::{GroupDevice, PlanError};
use crate::embedding::{EmbedFamily, EmbeddingConfig, EmbeddingService};
use crate::process::Ports;
use crate::profile::PlacementProfile;

/// The engine kind in a placement profile (spec 11.10).
pub const KIND: &str = "mk-embeddinggemma2-bmg";

/// The variable that names the checkpoint directory when a table has no key `model_dir`.
pub const MODEL_VARIABLE: &str = "EMBEDDINGGEMMA2_MODEL_DIR";

/// The checkpoint directory of a table: its key `model_dir`, or `EMBEDDINGGEMMA2_MODEL_DIR`. Any other key is an
/// error (ABI "Rust shape").
pub fn model_dir(table: &toml::Table, what: &str) -> Result<PathBuf, PlanError> {
    let fail = |message: String| PlanError { message };
    if let Some(key) = table.keys().find(|key| key.as_str() != "model_dir") {
        return Err(fail(format!(
            "{what} has the key `{key}`, which the engine does not know; the key is `model_dir`"
        )));
    }
    match table.get("model_dir") {
        Some(value) => Ok(PathBuf::from(value.as_str().ok_or_else(|| {
            fail(format!(
                "the key `model_dir` of {what} is not a string; write the path of the checkpoint directory"
            ))
        })?)),
        None => Ok(PathBuf::from(std::env::var_os(MODEL_VARIABLE).ok_or_else(|| {
            fail(format!(
                "{what} has no key `model_dir`, and {MODEL_VARIABLE} is not set; name the checkpoint directory"
            ))
        })?)),
    }
}

/// Read the config of the engine.
pub fn config(table: &toml::Table) -> Result<Config, PlanError> {
    Ok(Config {
        model_dir: model_dir(table, &format!("the config of `{KIND}`"))?,
    })
}

/// Refuse a group of any size but one device.
fn check_devices(devices: &[GroupDevice]) -> Result<(), PlanError> {
    if devices.len() == 1 {
        return Ok(());
    }
    Err(PlanError {
        message: format!(
            "the engine `{KIND}` runs on one card, but its profile names {}: name one device in `devices`",
            devices.len()
        ),
    })
}

/// The limits of the engine for its config and its device. It reads `config.json` and the tensor headers, and no
/// weight bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    check_devices(devices)?;
    let config = config(table)?;
    nagaya_mk_embeddinggemma2_bmg::plan(&config).map_err(|message| PlanError { message })
}

/// The family of the engine as an embedding family (spec 11.3, 11.10).
pub struct Family(pub EmbeddingGemma2Family);

impl EmbedFamily for Family {
    fn tasks(&self) -> Vec<String> {
        self.0.tasks().into_iter().map(str::to_owned).collect()
    }

    fn ids(&self, task: &str, text: &str) -> Result<Vec<u32>, String> {
        self.0.ids(task, text).map_err(|error| error.to_string())
    }

    fn template(&self) -> String {
        self.0.template_id().to_owned()
    }
}

/// The embedding service of each engine of this kind in the profile. The served model is the engine's name. The
/// family opens from the checkpoint, so a service exists also while its engine is unloaded (spec 3.3a).
pub fn embedding_services(profile: &PlacementProfile, ports: &Ports) -> Result<Vec<EmbeddingService>, String> {
    let mut services = Vec::new();
    for placement in profile.engines.iter().filter(|placement| placement.kind == KIND) {
        let what = format!("the family table of `{}`", placement.name);
        let dir = model_dir(&placement.family, &what).map_err(|error| error.message)?;
        let family = EmbeddingGemma2Family::from_dir(&dir).map_err(|error| {
            format!(
                "cannot build the EmbeddingGemma 2 family of engine `{}` from `{}`: {error}",
                placement.name,
                dir.display()
            )
        })?;
        let config = EmbeddingConfig::new(&placement.name, &placement.name);
        let service = EmbeddingService::new(config, Arc::new(Family(family)), ports.clone())?;
        tracing::info!(
            "serving embedding model `{}` on engine `{}`",
            placement.name,
            placement.name
        );
        services.push(service);
    }
    Ok(services)
}

#[cfg(feature = "xe")]
pub use factory::EmbeddingGemma2Factory;

#[cfg(feature = "xe")]
mod factory {
    use nagaya_abi::{Engine, Limits};
    use nagaya_mk_embeddinggemma2_bmg::EmbeddingGemma2Engine;

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the EmbeddingGemma 2 engine.
    pub struct EmbeddingGemma2Factory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for EmbeddingGemma2Factory {
        /// The served model is the engine name (spec 2.3, "The check of a name").
        fn served_names(&self, engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
            Some(vec![engine.to_owned()])
        }

        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Upload the text tower to the card at the PCI address of the device, and open the engine on it.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let engine = EmbeddingGemma2Engine::open(config, &devices[0].pci_address)
                .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
            Ok(Box::new(engine))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> toml::Table {
        text.parse().unwrap()
    }

    #[test]
    fn the_config_takes_model_dir_and_refuses_another_key() {
        assert_eq!(
            config(&table("model_dir = \"/eg2\"")).unwrap().model_dir,
            PathBuf::from("/eg2")
        );
        let unknown = config(&table("model_dir = \"/eg2\"\nslots = 2")).unwrap_err();
        assert!(unknown.message.contains("`slots`"), "{}", unknown.message);
        let number = config(&table("model_dir = 3")).unwrap_err();
        assert!(number.message.contains("`model_dir`"), "{}", number.message);
    }

    #[test]
    fn the_kind_is_the_one_of_the_spec() {
        assert_eq!(KIND, "mk-embeddinggemma2-bmg");
    }
}
