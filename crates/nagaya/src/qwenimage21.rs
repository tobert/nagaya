//! The Qwen-Image 2.1 engine on gfx1151 as a factory kind, and its image service (spec 11.9c, "The served Qwen-Image
//! model").
//!
//! The config parse, the plan and the family need no GPU, so the host tests cover them. The factory opens the engine
//! only in a build with the feature `hip`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{ImageChunk, Limits};
use nagaya_mk_qwenimage21_gfx1151::plan::{self, Config};
use nagaya_text_qwenimage21::{Error as FamilyError, QwenImage21Family};

use crate::admission::{GroupDevice, PlanError};
use crate::image::{ImageConfig, ImageFamily, ImageForm, ImageService};
use crate::process::Ports;
use crate::profile::PlacementProfile;

/// The engine kind in a placement profile.
pub const KIND: &str = "qwenimage21";

/// Read the config of the engine. The engine refuses a key that it does not know (ABI "Rust shape").
pub fn config(table: &toml::Table) -> Result<Config, PlanError> {
    Config::from_toml(table).map_err(|message| PlanError {
        message: format!("the config of `{KIND}`: {message}"),
    })
}

/// Refuse a group of any size but one device.
fn check_devices(devices: &[GroupDevice]) -> Result<(), PlanError> {
    if devices.len() == 1 {
        return Ok(());
    }
    Err(PlanError {
        message: format!(
            "the engine `{KIND}` runs on one GPU, but its profile names {}: name one device in `devices`",
            devices.len()
        ),
    })
}

/// The limits of the engine for its config and its device. It reads the header of each weight file, and no weight
/// bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    check_devices(devices)?;
    let config = config(table)?;
    plan::plan(&config).map_err(|message| PlanError {
        message: format!("`{KIND}` cannot plan: {message}"),
    })
}

/// The family of the engine as an image family of the tokens form (spec 11.9c).
pub struct Family(pub QwenImage21Family);

impl ImageFamily for Family {
    fn form(&self) -> ImageForm {
        ImageForm::Tokens
    }

    fn prompt(&self, _text: &str) -> Result<(Vec<ImageChunk>, u64), String> {
        Err("is not a prompt of chunks: the Qwen-Image family makes tokens".to_owned())
    }

    fn tokens(&self, text: &str) -> Result<(Vec<u32>, u32), String> {
        let ids = self.0.ids(text).map_err(|error| match error {
            FamilyError::SpecialToken { token } => {
                format!("holds the special token `{token}`; write it another way")
            }
            FamilyError::TooLong { ids, max } => {
                format!("makes a template of {ids} ids, more than {max}; give a shorter prompt")
            }
            other => other.to_string(),
        })?;
        Ok((ids, nagaya_text_qwenimage21::CUT as u32))
    }
}

/// The base folder of the family of a placement: the `model` key of its config, which holds `processor/`.
fn family_dir(table: &toml::Table, engine: &str) -> Result<PathBuf, String> {
    let config = config(table).map_err(|error| format!("engine `{engine}`: {}", error.message))?;
    Ok(config.model)
}

/// The image service of each engine of this kind in the profile. The served model is the engine's name. The family
/// opens from the base folder, so a service exists also while its engine is unloaded (spec 3.3a).
pub fn image_services(profile: &PlacementProfile, ports: &Ports) -> Result<Vec<ImageService>, String> {
    let mut services = Vec::new();
    for placement in profile.engines.iter().filter(|placement| placement.kind == KIND) {
        let dir = family_dir(&placement.config, &placement.name)?;
        let family = QwenImage21Family::from_dir(&dir).map_err(|error| {
            format!(
                "cannot build the Qwen-Image family of engine `{}` from `{}`: {error}",
                placement.name,
                dir.display()
            )
        })?;
        let mut config = ImageConfig::new(&placement.name, &placement.name);
        config.context_tokens = nagaya_mk_qwenimage21_gfx1151::plan::MAX_IDS as u64;
        let service = ImageService::new(config, Arc::new(Family(family)), ports.clone())?;
        tracing::info!(
            "serving image model `{}` on engine `{}`",
            placement.name,
            placement.name
        );
        services.push(service);
    }
    Ok(services)
}

#[cfg(feature = "hip")]
pub use factory::QwenImage21Factory;

#[cfg(feature = "hip")]
mod factory {
    use nagaya_abi::{Engine, Limits};
    use nagaya_mk_qwenimage21_gfx1151::engine::QwenImageEngine;

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the Qwen-Image engine.
    pub struct QwenImage21Factory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for QwenImage21Factory {
        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Load the weights into the device of the thread, and open the engine on it. HIP's device is the device
        /// that is current on the calling thread.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let engine = QwenImageEngine::open(&config)
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
    fn the_config_names_the_files_and_refuses_an_unknown_key() {
        let config = config(&table("model = \"/q\"\nencoder = \"/e\"\nlora = \"/l\"")).unwrap();
        assert_eq!(config.lora, Some((PathBuf::from("/l"), 1.0)));
        let error = super::config(&table("model = \"/q\"\nencoder = \"/e\"\nclip_skip = 2")).unwrap_err();
        assert!(error.message.contains("`clip_skip`"), "{}", error.message);
    }

    #[test]
    fn the_engine_runs_on_one_device() {
        assert!(check_devices(&[]).is_err());
    }
}
