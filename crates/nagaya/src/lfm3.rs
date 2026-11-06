//! The LFM3 engine on one B70 card as a factory kind, and the opener of its family library (spec 2.4, 11.3, 11.11;
//! engine-abi "The first LFM3 binding").
//!
//! One kind serves `LFM3-1.7B` and `LFM3-Omni-2B`, so the config names the model folder, and the folder has no default.
//! The config parse, the plan and the family opener need no card, so the host tests cover them. The factory opens the
//! card only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{Identity, Limits};
use nagaya_mk_lfm3_bmg::spec::{self, Config};

use crate::admission::{GroupDevice, PlanError};
use crate::serving::OpenedFamily;

/// The engine kind in a placement profile.
pub const KIND: &str = spec::KIND;

/// The model folder: the key `model_dir` of `table`. It must be a directory. The message names the key.
pub fn model_dir(table: &toml::Table, what: &str) -> Result<PathBuf, String> {
    let dir = match table.get("model_dir") {
        Some(value) => PathBuf::from(value.as_str().ok_or_else(|| {
            format!("the key `model_dir` of {what} is not a string; write the path of the model folder")
        })?),
        None => {
            return Err(format!(
                "{what} has no key `model_dir`; name the folder of LFM3-1.7B or LFM3-Omni-2B, because one kind serves \
                 both"
            ));
        }
    };
    if !dir.is_dir() {
        return Err(format!(
            "the model folder `{}` of {what} is not a directory",
            dir.display()
        ));
    }
    Ok(dir)
}

/// Read the config of the engine. The engine refuses a key that it does not know (ABI "Rust shape").
pub fn config(table: &toml::Table) -> Result<Config, PlanError> {
    let fail = |message: String| PlanError { message };
    let what = format!("the config of `{KIND}`");
    if let Some(key) = table
        .keys()
        .find(|key| !matches!(key.as_str(), "model_dir" | "slots" | "max_block"))
    {
        return Err(fail(format!(
            "{what} has the key `{key}`, which the engine does not know; the keys are `model_dir`, `slots` and \
             `max_block`"
        )));
    }
    let model_dir = model_dir(table, &what).map_err(fail)?;
    let count = |key: &str, value: &toml::Value| {
        value
            .as_integer()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| fail(format!("`{key}` in {what} is {value}, not a count")))
    };
    let slots = match table.get("slots") {
        Some(value) => count("slots", value)?,
        None => {
            return Err(fail(format!(
                "{what} needs `slots`, the count of slots of {} positions each",
                spec::SLOT_TOKENS
            )));
        }
    };
    let max_block = match table.get("max_block") {
        None => spec::DEFAULT_MAX_BLOCK,
        Some(value) => count("max_block", value)?,
    };
    let config = Config {
        model_dir,
        slots,
        max_block,
    };
    config.check().map_err(|message| fail(format!("{what}: {message}")))?;
    Ok(config)
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
    spec::plan(&config).map_err(|message| PlanError { message })
}

/// The LFM3 family from the model folder: the key `model_dir` of `[engines.family]`. The registry compares its
/// tokenizer hash with the identity (spec 11.3).
pub fn open_family(identity: &Identity, table: &toml::Table) -> Result<OpenedFamily, String> {
    let dir = model_dir(table, "the table `[engines.family]`")?;
    let family = nagaya_text_lfm3::Lfm3Family::from_dir(&dir)
        .map_err(|error| format!("cannot build the LFM3 family from `{}`: {error}", dir.display()))?;
    let model = match family.kind() {
        nagaya_text_lfm3::Kind::Text => spec::Model::Text,
        nagaya_text_lfm3::Kind::Omni => spec::Model::Omni,
    };
    if model.name() != identity.model {
        return Err(format!(
            "the family folder `{}` holds `{}`, and the engine serves `{}`; name the folder of the engine",
            dir.display(),
            model.name(),
            identity.model
        ));
    }
    let hash = family.tokenizer_hash().to_owned();
    Ok(OpenedFamily {
        family: Arc::new(family),
        tokenizer_hash: Some(hash),
    })
}

#[cfg(feature = "xe")]
pub use factory::Lfm3Factory;

#[cfg(feature = "xe")]
mod factory {
    use nagaya_abi::{Engine, Limits};
    use nagaya_mk_lfm3_bmg::Lfm3Engine;

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the LFM3 engine.
    pub struct Lfm3Factory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for Lfm3Factory {
        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Upload the backbone to the card at the PCI address of the device, and open the engine on it. The served
        /// name comes from `config.json` of the folder, so the kind gives no name before the open.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let engine = Lfm3Engine::open(config, &devices[0].pci_address)
                .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
            Ok(Box::new(engine))
        }
    }
}
