//! The SDXL engine on one B70 card as a factory kind, and its image service (spec 11.9, "The served image model").
//!
//! The config parse, the plan and the family need no card, so the host tests cover them. The factory opens the card
//! only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{ImageChunk, Limits};
use nagaya_mk_sdxl_bmg::Config;
use nagaya_mk_sdxl_bmg::spec::KIND as ENGINE_KIND;
use nagaya_text_sdxl::SdxlFamily;

use crate::admission::{GroupDevice, PlanError};
use crate::image::{ImageConfig, ImageFamily, ImageService};
use crate::process::Ports;
use crate::profile::PlacementProfile;

/// The engine kind in a placement profile.
pub const KIND: &str = ENGINE_KIND;

/// The variable that names the checkpoint directory when a table has no key `model_dir`.
pub const MODEL_VARIABLE: &str = "SDXL_MODEL_DIR";

/// The start id, the end id, and the pad id of the second encoder. They are not tokens of the prompt.
const NOT_TOKENS: [u32; 3] = [49406, 49407, 0];

/// The checkpoint directory of a table: its key `model_dir`, or `SDXL_MODEL_DIR`.
fn model_dir(table: &toml::Table, what: &str) -> Result<PathBuf, PlanError> {
    let fail = |message: String| PlanError { message };
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

/// Read the config of the engine. The engine refuses a key that it does not know (ABI "Rust shape").
pub fn config(table: &toml::Table) -> Result<Config, PlanError> {
    let what = format!("the config of `{KIND}`");
    if let Some(key) = table
        .keys()
        .find(|key| !["model_dir", "checkpoint"].contains(&key.as_str()))
    {
        return Err(PlanError {
            message: format!(
                "{what} has the key `{key}`, which the engine does not know; the keys are `model_dir` and \
                 `checkpoint`"
            ),
        });
    }
    let checkpoint = match table.get("checkpoint") {
        None => None,
        Some(value) => Some(PathBuf::from(value.as_str().ok_or_else(|| PlanError {
            message: format!(
                "the key `checkpoint` of {what} is not a string; write the path of a single .safetensors file"
            ),
        })?)),
    };
    Ok(Config {
        model_dir: model_dir(table, &what)?,
        checkpoint,
    })
}

/// The family table of an engine of this kind: its checkpoint directory and its default `clip_skip`.
pub fn family_config(table: &toml::Table) -> Result<(PathBuf, u32), PlanError> {
    let what = format!("the family table of `{KIND}`");
    if let Some(key) = table
        .keys()
        .find(|key| !["model_dir", "clip_skip"].contains(&key.as_str()))
    {
        return Err(PlanError {
            message: format!("{what} has the key `{key}`; the keys are `model_dir` and `clip_skip`"),
        });
    }
    let clip_skip = match table.get("clip_skip") {
        None => 2,
        Some(value) => value
            .as_integer()
            .filter(|v| (2..=3).contains(v))
            .ok_or_else(|| PlanError {
                message: format!("`clip_skip` in {what} is {value}; give 2 or 3"),
            })? as u32,
    };
    Ok((model_dir(table, &what)?, clip_skip))
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

/// The limits of the engine for its config and its device. It reads the header of each weight file, and no weight
/// bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    check_devices(devices)?;
    let config = config(table)?;
    nagaya_mk_sdxl_bmg::plan(&config).map_err(|message| PlanError { message })
}

/// The family of the engine as an image family (spec 11.3, 11.9).
pub struct Family(pub SdxlFamily);

impl ImageFamily for Family {
    fn prompt(&self, text: &str) -> Result<(Vec<ImageChunk>, u64), String> {
        let prompt = self.0.prompt(text).map_err(|error| error.to_string())?;
        let tokens = prompt
            .chunks
            .iter()
            .flat_map(|chunk| chunk.ids1.iter())
            .filter(|id| !NOT_TOKENS.contains(id))
            .count() as u64;
        let chunks = prompt
            .chunks
            .into_iter()
            .map(|chunk| ImageChunk {
                ids1: chunk.ids1.to_vec(),
                ids2: chunk.ids2.to_vec(),
                weights: chunk.weights.to_vec(),
            })
            .collect();
        Ok((chunks, tokens))
    }
}

/// The image service of each engine of this kind in the profile. The served model is the engine's name. The family
/// opens from the checkpoint, so a service exists also while its engine is unloaded (spec 3.3a).
pub fn image_services(profile: &PlacementProfile, ports: &Ports) -> Result<Vec<ImageService>, String> {
    let mut services = Vec::new();
    for placement in profile.engines.iter().filter(|placement| placement.kind == KIND) {
        let (dir, clip_skip) = family_config(&placement.family).map_err(|error| error.message)?;
        let family = SdxlFamily::from_dir(&dir).map_err(|error| {
            format!(
                "cannot build the SDXL family of engine `{}` from `{}`: {error}",
                placement.name,
                dir.display()
            )
        })?;
        let mut config = ImageConfig::new(&placement.name, &placement.name);
        config.clip_skip = clip_skip;
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

#[cfg(feature = "xe")]
pub use factory::SdxlFactory;

#[cfg(feature = "xe")]
mod factory {
    use nagaya_abi::{Engine, Limits};
    use nagaya_mk_sdxl_bmg::SdxlEngine;

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the SDXL engine.
    pub struct SdxlFactory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for SdxlFactory {
        /// The served model is the engine name (spec 2.3, "The check of a name").
        fn served_names(&self, engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
            Some(vec![engine.to_owned()])
        }

        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Upload the weights to the card at the PCI address of the device, and open the engine on it.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let engine = SdxlEngine::open(config, &devices[0].pci_address)
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
    fn the_config_takes_a_folder_and_a_single_file() {
        let folder = config(&table("model_dir = \"/sdxl\"")).unwrap();
        assert_eq!((folder.model_dir, folder.checkpoint), (PathBuf::from("/sdxl"), None));
        let single = config(&table(
            "model_dir = \"/sdxl\"\ncheckpoint = \"/civitai/pony.safetensors\"",
        ))
        .unwrap();
        assert_eq!(single.checkpoint, Some(PathBuf::from("/civitai/pony.safetensors")));
    }

    #[test]
    fn the_config_refuses_an_unknown_key_and_a_checkpoint_that_is_no_path() {
        let unknown = config(&table("model_dir = \"/sdxl\"\nvae = \"/vae\"")).unwrap_err();
        assert!(unknown.message.contains("`vae`"), "{}", unknown.message);
        let number = config(&table("model_dir = \"/sdxl\"\ncheckpoint = 3")).unwrap_err();
        assert!(number.message.contains("`checkpoint`"), "{}", number.message);
    }

    /// `clip_skip` counts as A1111 does: 2 is the penultimate layer (spec 11.9a, Decided 2026-10-29).
    #[test]
    fn the_family_table_takes_clip_skip_2_or_3_with_the_default_2() {
        assert_eq!(family_config(&table("model_dir = \"/sdxl\"")).unwrap().1, 2);
        assert_eq!(
            family_config(&table("model_dir = \"/sdxl\"\nclip_skip = 3")).unwrap().1,
            3
        );
        for skip in [1, 4] {
            let error = family_config(&table(&format!("model_dir = \"/sdxl\"\nclip_skip = {skip}"))).unwrap_err();
            assert!(error.message.contains("give 2 or 3"), "{}", error.message);
        }
    }
}
