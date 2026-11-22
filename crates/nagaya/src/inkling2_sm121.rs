//! The Inkling 2 engine on one NVIDIA GB10 (`sm_121`) GPU as a factory kind (spec 2.4, 11.3, 11.18;
//! engine-abi "The GB10 Inkling 2 binding").
//!
//! The config parse and plan need no GPU, so the host tests cover them. `Registry::compiled_in` registers the kind
//! when the feature `cuda` is enabled.

use std::path::PathBuf;

use nagaya_abi::Limits;
use nagaya_mk_inkling2_sm121::spec::{self, Config};

use crate::admission::{GroupDevice, PlanError};

/// Engine kind name in a placement profile.
pub const KIND: &str = spec::KIND;

/// Allowed keys of the engine config table.
const KEYS: [&str; 4] = ["model_dir", "slots", "max_block", "extra_runs"];

/// Read `model_dir` from `table` and verify that it names an existing directory.
pub fn model_dir(table: &toml::Table, what: &str) -> Result<PathBuf, String> {
    let dir = table
        .get("model_dir")
        .ok_or_else(|| {
            format!("{what} has no key `model_dir`; name the folder of Inkling-2-Mini or TML-d24-varloop-research")
        })?
        .as_str()
        .map(PathBuf::from)
        .ok_or_else(|| format!("the key `model_dir` of {what} is not a string; write the path of the model folder"))?;
    if !dir.is_dir() {
        return Err(format!(
            "the model folder `{}` of {what} is not a directory",
            dir.display()
        ));
    }
    Ok(dir)
}

/// Parse and validate the engine config table, refusing unknown keys.
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

/// Plan `Limits` for `table` and `devices` without reading weight bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    check_devices(devices)?;
    let config = config(table)?;
    spec::plan(&config).map_err(|message| PlanError { message })
}

/// Served model names for this kind: the engine name (spec 11.18).
pub fn served_names(engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
    Some(vec![engine.to_owned()])
}

#[cfg(feature = "cuda")]
pub use factory::Inkling2Sm121Factory;

#[cfg(feature = "cuda")]
mod factory {
    use nagaya_abi::{Engine, Limits};

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// Factory for `mk-inkling2-sm121`.
    #[derive(Default)]
    pub struct Inkling2Sm121Factory;

    impl Factory for Inkling2Sm121Factory {
        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        fn served_names(&self, engine: &str, config: &toml::Table) -> Option<Vec<String>> {
            super::served_names(engine, config)
        }

        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            let err = |message: String| OpenError {
                kind: OpenErrorKind::Other,
                message,
            };
            check_devices(devices).map_err(|e| err(e.message))?;
            let _config = config(table).map_err(|e| err(e.message))?;
            Err(err(format!(
                "`{KIND}` cannot open yet: the device ops are step 9 of milestone 37 and the engine is step 11. \
                 Remove this engine from the profile until step 11 is done"
            )))
        }
    }
}
