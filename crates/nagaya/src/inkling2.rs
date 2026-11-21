//! The Inkling 2 engine on one B70 card as a factory kind, and the opener of its family library (spec 2.4, 11.3,
//! 11.18; engine-abi "Inkling 2").
//!
//! One kind serves `Inkling-2-Mini` and `TML-d24-varloop-research`, so the config names the model folder, and the
//! folder has no default. The config parse, the plan and the family opener need no card, so the host tests cover them.
//! The factory opens the card only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in
//! that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{Identity, Limits};
use nagaya_mk_inkling2_bmg::spec::{self, Config};

use crate::admission::{GroupDevice, PlanError};
use crate::serving::OpenedFamily;

/// The engine kind in a placement profile.
pub const KIND: &str = spec::KIND;

/// The variable that names a model folder of Inkling 2 for the family, when `[engines.family]` names none. Both
/// models share the files of the family, so either folder serves (spec 11.18).
pub const MODEL_VARIABLE: &str = "INKLING2_MODEL_DIR";

/// The keys of the engine config.
const KEYS: [&str; 4] = ["model_dir", "slots", "max_block", "extra_runs"];

/// The path `dir` as a model folder of `what`. It must be a directory.
fn folder(dir: PathBuf, what: &str) -> Result<PathBuf, String> {
    if !dir.is_dir() {
        return Err(format!(
            "the model folder `{}` of {what} is not a directory",
            dir.display()
        ));
    }
    Ok(dir)
}

/// The string at the key `model_dir` of `table`, or `None` with no key.
fn model_dir_key(table: &toml::Table, what: &str) -> Result<Option<PathBuf>, String> {
    table
        .get("model_dir")
        .map(|value| {
            value.as_str().map(PathBuf::from).ok_or_else(|| {
                format!("the key `model_dir` of {what} is not a string; write the path of the model folder")
            })
        })
        .transpose()
}

/// The model folder of the engine: the key `model_dir` of `table`. The key has no default, because one kind serves
/// two models.
pub fn model_dir(table: &toml::Table, what: &str) -> Result<PathBuf, String> {
    let dir = model_dir_key(table, what)?.ok_or_else(|| {
        format!(
            "{what} has no key `model_dir`; name the folder of Inkling-2-Mini or TML-d24-varloop-research, because \
             one kind serves both"
        )
    })?;
    folder(dir, what)
}

/// The model folder of the family: the key `model_dir` of `[engines.family]`, or `INKLING2_MODEL_DIR`.
pub fn family_dir(table: &toml::Table) -> Result<PathBuf, String> {
    let what = "the table `[engines.family]`";
    let dir = match model_dir_key(table, what)? {
        Some(dir) => dir,
        None => PathBuf::from(std::env::var_os(MODEL_VARIABLE).ok_or_else(|| {
            format!("{what} has no key `model_dir`, and {MODEL_VARIABLE} is not set; name a model folder of Inkling 2")
        })?),
    };
    folder(dir, what)
}

/// Read the config of the engine. The engine refuses a key that it does not know (ABI "Rust shape").
pub fn config(table: &toml::Table) -> Result<Config, PlanError> {
    let fail = |message: String| PlanError { message };
    let what = format!("the config of `{KIND}`");
    if let Some(key) = table.keys().find(|key| !KEYS.contains(&key.as_str())) {
        return Err(fail(format!(
            "{what} has the key `{key}`, which the engine does not know; the keys are `model_dir`, `slots`, \
             `max_block` and `extra_runs`"
        )));
    }
    let model_dir = model_dir(table, &what).map_err(fail)?;
    let count = |key: &str, value: &toml::Value| {
        value
            .as_integer()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| fail(format!("`{key}` in {what} is {value}, not a count")))
    };
    // A count gives slots of the default size. A list gives the positions of each slot (engine-abi "The host spec").
    let slots = match table.get("slots") {
        Some(toml::Value::Array(list)) => list
            .iter()
            .map(|value| count("slots", value))
            .collect::<Result<Vec<_>, _>>()?,
        Some(value @ toml::Value::Integer(_)) => vec![spec::SLOT_TOKENS; count("slots", value)?],
        Some(value) => {
            return Err(fail(format!(
                "`slots` in {what} is {value}; give a count of slots of {} positions, or a list of positions",
                spec::SLOT_TOKENS
            )));
        }
        None => {
            return Err(fail(format!(
                "{what} needs `slots`, a count of slots of {} positions each, or a list of positions",
                spec::SLOT_TOKENS
            )));
        }
    };
    let max_block = match table.get("max_block") {
        None => spec::DEFAULT_MAX_BLOCK,
        Some(value) => count("max_block", value)?,
    };
    // The reserve of each slot holds the runs of the cycles after the first (engine-abi "The host spec").
    let extra_runs = match table.get("extra_runs") {
        None => spec::DEFAULT_EXTRA_RUNS,
        Some(value) => count("extra_runs", value)?,
    };
    let config = Config {
        model_dir,
        slots,
        max_block,
        extra_runs,
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

/// The served model names of an engine of this kind: its engine name (spec 11.18).
pub fn served_names(engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
    Some(vec![engine.to_owned()])
}

/// The Inkling 2 family from a model folder (`family_dir`). The registry compares its tokenizer hash with the identity
/// (spec 11.3). The served model is the engine name.
pub fn open_family(identity: &Identity, table: &toml::Table) -> Result<OpenedFamily, String> {
    let _ = identity;
    let dir = family_dir(table)?;
    let family = nagaya_text_inkling2::Inkling2Family::from_dir(&dir)
        .map_err(|error| format!("cannot build the Inkling 2 family from `{}`: {error}", dir.display()))?;
    let hash = family.tokenizer_hash().to_owned();
    Ok(OpenedFamily {
        family: Arc::new(family),
        tokenizer_hash: Some(hash),
        speech: None,
        served: crate::serving::ServedName::Engine,
    })
}

#[cfg(feature = "xe")]
pub use factory::Inkling2Factory;

#[cfg(feature = "xe")]
mod factory {
    use std::sync::Mutex;

    use nagaya_abi::{Engine, Identity, Limits};
    use nagaya_mk_inkling2_bmg::Inkling2Engine;
    use nagaya_mk_inkling2_bmg::spec::Config;

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the Inkling 2 engine. It keeps each identity that it described, by the config and the card, so
    /// an open checks that it reports the same identity (spec 11.1).
    #[derive(Default)]
    pub struct Inkling2Factory {
        described: Mutex<Vec<((Config, String), Identity)>>,
    }

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for Inkling2Factory {
        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// The served model is the engine name (spec 11.18).
        fn served_names(&self, engine: &str, config: &toml::Table) -> Option<Vec<String>> {
            super::served_names(engine, config)
        }

        /// The identity and the limits of an open on the card of the group, from the model folder, sysfs and the
        /// Level Zero driver, with no load (spec 11.1). It reads every tensor for the checkpoint digest.
        fn describe(
            &self,
            table: &toml::Table,
            devices: &[GroupDevice],
        ) -> Result<Option<(Identity, Limits)>, PlanError> {
            check_devices(devices)?;
            let config = config(table)?;
            let pci = devices[0].pci_address.clone();
            let (identity, limits) =
                nagaya_mk_inkling2_bmg::engine::describe(&config, &pci).map_err(|message| PlanError {
                    message: format!("`{KIND}` cannot describe: {message}"),
                })?;
            let key = (config, pci);
            let mut described = self.described.lock().expect("the lock of the descriptions");
            described.retain(|(seen, _)| *seen != key);
            described.push((key, identity.clone()));
            Ok(Some((identity, limits)))
        }

        /// Upload the text model to the card at the PCI address of the device, and open the engine on it.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let pci = devices[0].pci_address.clone();
            let engine = Inkling2Engine::open(config.clone(), &pci)
                .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
            let key = (config, pci);
            let described = self.described.lock().expect("the lock of the descriptions");
            if let Some((_, identity)) = described.iter().find(|(seen, _)| *seen == key)
                && *identity != engine.identity()
            {
                return Err(open_error(format!(
                    "`{KIND}` opened with another identity than its description: the description gave {identity:?}, \
                     and the open gave {:?}. The model folder or the driver changed after the start; restart the \
                     process",
                    engine.identity()
                )));
            }
            Ok(Box::new(engine))
        }
    }
}
