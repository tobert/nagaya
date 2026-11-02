//! The Gemma 4 engine on two B70 cards as a factory kind, and the opener of its family library (spec 2.4, 11.3;
//! engine-abi "The first Gemma 4 binding").
//!
//! The config parse and the family opener need no card, so the host tests cover them. The factory opens the cards only
//! in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{ArenaPart, Identity, Limits};
use nagaya_mk_gemma4_bmg::spec::{self, Config};

use crate::admission::{GroupDevice, PlanError};
use crate::serving::OpenedFamily;

/// The engine kind in a placement profile.
pub const KIND: &str = spec::KIND;

/// The variable that names the checkpoint directory when a table has no key `model_dir`.
pub const MODEL_VARIABLE: &str = "GEMMA4_MODEL_DIR";

/// The checkpoint directory: the key `model_dir` of `table`, else the variable `GEMMA4_MODEL_DIR`. It must be a
/// directory. The message names the key and the variable.
pub fn model_dir(table: &toml::Table, what: &str) -> Result<PathBuf, String> {
    let dir = match table.get("model_dir") {
        Some(value) => PathBuf::from(value.as_str().ok_or_else(|| {
            format!("the key `model_dir` of {what} is not a string; write the path of the checkpoint directory")
        })?),
        None => PathBuf::from(std::env::var_os(MODEL_VARIABLE).ok_or_else(|| {
            format!("{what} has no key `model_dir`, and {MODEL_VARIABLE} is not set; name the checkpoint directory")
        })?),
    };
    if !dir.is_dir() {
        return Err(format!(
            "the checkpoint directory `{}` of {what} is not a directory",
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
    let slots = table
        .get("slots")
        .ok_or_else(|| {
            fail(format!(
                "{what} needs `slots`, the positions of each slot, for example [4096, 65536]"
            ))
        })?
        .as_array()
        .ok_or_else(|| fail(format!("`slots` in {what} is not an array of positions")))?
        .iter()
        .map(|value| {
            value
                .as_integer()
                .and_then(|slot| usize::try_from(slot).ok())
                .ok_or_else(|| {
                    fail(format!(
                        "`slots` in {what} holds {value}, which is not a count of positions"
                    ))
                })
        })
        .collect::<Result<Vec<usize>, PlanError>>()?;
    let max_block = match table.get("max_block") {
        None => spec::DEFAULT_MAX_BLOCK,
        Some(value) => value
            .as_integer()
            .and_then(|block| usize::try_from(block).ok())
            .ok_or_else(|| fail(format!("`max_block` in {what} is {value}, not a count of tokens")))?,
    };
    let config = Config {
        model_dir,
        slots,
        max_block,
    };
    config.check().map_err(|message| fail(format!("{what}: {message}")))?;
    Ok(config)
}

/// Refuse a group of any size but two devices: stage 0 runs on the first, stage 1 on the second.
fn check_devices(devices: &[GroupDevice]) -> Result<(), PlanError> {
    if devices.len() == 2 {
        return Ok(());
    }
    Err(PlanError {
        message: format!(
            "the engine `{KIND}` runs on two cards, but its profile names {}: name two devices in `devices`",
            devices.len()
        ),
    })
}

/// The limits of the engine for its config and its devices. It reads no weight bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    check_devices(devices)?;
    let config = config(table)?;
    spec::plan(&config).map_err(|message| PlanError { message })
}

/// The parts of the arena of the engine: a slab for each layer on the card of its stage, and the head slab on the
/// second card (spec 10.2). It reads no weight bytes.
pub fn arena_parts(devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
    check_devices(devices)?;
    Ok(spec::arena_parts())
}

/// The Gemma 4 family from the checkpoint directory: the key `model_dir` of `[engines.family]`, else
/// `GEMMA4_MODEL_DIR`. The registry compares its tokenizer hash with the identity (spec 11.3).
pub fn open_family(identity: &Identity, table: &toml::Table) -> Result<OpenedFamily, String> {
    let _ = identity;
    let dir = model_dir(table, "the table `[engines.family]`")?;
    let family = nagaya_text_gemma4::Gemma4Family::from_dir(&dir)
        .map_err(|error| format!("cannot build the Gemma 4 family from `{}`: {error}", dir.display()))?;
    let hash = family.tokenizer_hash().to_owned();
    Ok(OpenedFamily {
        family: Arc::new(family),
        tokenizer_hash: Some(hash),
    })
}

#[cfg(feature = "xe")]
pub use factory::Gemma4Factory;

#[cfg(feature = "xe")]
mod factory {
    use std::os::fd::OwnedFd;

    use nagaya_abi::{ArenaFd, ArenaPart, CheckStrength, Engine, Limits};
    use nagaya_mk_gemma4_bmg::engine::{LoadError, Loaded};

    use super::{KIND, check_devices, config};
    use crate::admission::{AttachError, Attached, Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the Gemma 4 engine.
    pub struct Gemma4Factory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for Gemma4Factory {
        /// The served model is the fixed model of the kind (spec 2.3, "The check of a name").
        fn served_names(&self, _engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
            Some(vec![nagaya_mk_gemma4_bmg::spec::MODEL.to_owned()])
        }

        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Load the weights on the cards at the PCI addresses of the group, and open one engine on them.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let addresses: Vec<String> = devices.iter().map(|device| device.pci_address.clone()).collect();
            let loaded = Loaded::load(&config.model_dir, &addresses)
                .map_err(|message| open_error(format!("`{KIND}` cannot load: {message}")))?;
            let engine = nagaya_mk_gemma4_bmg::engine::open(&loaded, &config)
                .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
            Ok(Box::new(engine))
        }

        fn arena_parts(&self, _table: &toml::Table, devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
            super::arena_parts(devices)
        }

        /// Spot-check the stored slabs on the cards of the group, and open one engine on them (spec 10.2).
        fn attach(
            &self,
            table: &toml::Table,
            devices: &[GroupDevice],
            arenas: Vec<ArenaFd>,
        ) -> Result<Attached, AttachError> {
            let fail = |message: String| AttachError::Open(open_error(message));
            check_devices(devices).map_err(|error| fail(error.message))?;
            let config = config(table).map_err(|error| fail(error.message))?;
            let addresses: Vec<String> = devices.iter().map(|device| device.pci_address.clone()).collect();
            let mut stored: [Vec<(String, OwnedFd)>; 2] = [Vec::new(), Vec::new()];
            for arena in arenas {
                let stage = stored
                    .get_mut(arena.device)
                    .ok_or_else(|| AttachError::Mismatch(format!("a slab is on device {} of two", arena.device)))?;
                stage.push((arena.part, arena.fd));
            }
            let loaded = Loaded::attach(&config.model_dir, &addresses, stored).map_err(|error| match error {
                LoadError::Mismatch(why) => AttachError::Mismatch(why),
                LoadError::Other(message) => fail(format!("`{KIND}` cannot attach: {message}")),
            })?;
            let engine = nagaya_mk_gemma4_bmg::engine::open(&loaded, &config)
                .map_err(|message| fail(format!("`{KIND}` cannot open: {message}")))?;
            Ok(Attached {
                engine: Box::new(engine),
                check: CheckStrength::Spot,
            })
        }
    }
}
