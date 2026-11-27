//! The Inkling 2 engine on one NVIDIA GB10 (`sm_121`) GPU as a factory kind (spec 2.4, 11.3, 11.18;
//! engine-abi "The GB10 Inkling 2 binding").
//!
//! The config parse and plan need no GPU, so the host tests cover them. `Registry::compiled_in` registers the kind
//! when the feature `cuda` is enabled.

use std::path::PathBuf;

use nagaya_abi::{Identity, Limits};
use nagaya_mk_inkling2_sm121::spec::{self, Config};

use crate::admission::{GroupDevice, PlanError};

/// Engine kind name in a placement profile.
pub const KIND: &str = spec::KIND;

/// Allowed keys of the engine config table.
const KEYS: [&str; 6] = ["model_dir", "slots", "max_block", "extra_runs", "cache", "taps"];

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
             `max_block`, `extra_runs`, `cache` and `taps`"
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
    let cache = match table.get("cache") {
        None => spec::DEFAULT_CACHE,
        Some(toml::Value::String(name)) => {
            spec::parse_cache(name).map_err(|message| fail(format!("{what}: {message}")))?
        }
        Some(value) => {
            return Err(fail(format!(
                "`cache` in {what} is {value}; write `\"f32\"`, `\"bf16\"` or `\"e4m3\"`"
            )));
        }
    };
    let taps = match table.get("taps") {
        None => false,
        Some(toml::Value::Boolean(on)) => *on,
        Some(value) => {
            return Err(fail(format!("`taps` in {what} is {value}; write `true` or `false`")));
        }
    };
    let config = Config {
        model_dir,
        slots,
        max_block,
        extra_runs,
        cache,
        taps,
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

/// The refusal of an open whose identity differs from its description, or `None` (spec 11.1). The process serves a
/// text engine on demand from the description, so the engine that opens must report the same facts.
pub fn identity_mismatch(described: &Identity, opened: &Identity) -> Option<String> {
    (described != opened).then(|| {
        format!(
            "`{KIND}` opened with another identity than its description: the description gave {described:?}, and the \
             open gave {opened:?}. The model folder, the build or the driver changed after the start; restart the \
             process"
        )
    })
}

#[cfg(feature = "cuda")]
pub use factory::Inkling2Sm121Factory;

#[cfg(feature = "cuda")]
mod factory {
    use std::sync::Mutex;

    use nagaya_abi::{Engine, Identity, Limits};
    use nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine;
    use nagaya_mk_inkling2_sm121::spec::Config;

    use super::{KIND, check_devices, config, identity_mismatch};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// Factory for `mk-inkling2-sm121`. It keeps each identity that it described, by the config and the GPU, so an
    /// open checks that it reports the same identity (spec 11.1).
    #[derive(Default)]
    pub struct Inkling2Sm121Factory {
        described: Mutex<Vec<((Config, String), Identity)>>,
    }

    impl Inkling2Sm121Factory {
        /// Change each stored description, so that the next open meets another identity than its description. A test
        /// uses it to reach the refusal, because a real change needs a new build or a new driver.
        #[doc(hidden)]
        pub fn corrupt_descriptions_for_test(&self) {
            for (_, identity) in self.described.lock().expect("the lock of the descriptions").iter_mut() {
                identity.checkpoint_digest.push_str("-changed");
            }
        }
    }

    impl Factory for Inkling2Sm121Factory {
        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        fn served_names(&self, engine: &str, config: &toml::Table) -> Option<Vec<String>> {
            super::served_names(engine, config)
        }

        /// The identity and the limits of an open on the GPU of the group, with no load of weights (spec 11.1). It
        /// reads the digest of the checkpoint through the cache of the digest.
        fn describe(
            &self,
            table: &toml::Table,
            devices: &[GroupDevice],
        ) -> Result<Option<(Identity, Limits)>, PlanError> {
            check_devices(devices)?;
            let config = config(table)?;
            let pci = devices[0].pci_address.clone();
            let (identity, limits) =
                nagaya_mk_inkling2_sm121::describe::describe(&config, &pci).map_err(|message| PlanError {
                    message: format!("`{KIND}` cannot describe: {message}"),
                })?;
            let key = (config, pci);
            let mut described = self.described.lock().expect("the lock of the descriptions");
            described.retain(|(seen, _)| *seen != key);
            described.push((key, identity.clone()));
            Ok(Some((identity, limits)))
        }

        /// The problem that an open on the GPU of the group would meet now (spec 3.3b). It reads no weights.
        fn open_problem(&self, table: &toml::Table, devices: &[GroupDevice], freed_bytes: u64) -> Option<String> {
            if let Err(error) = check_devices(devices) {
                return Some(error.message);
            }
            let config = match config(table) {
                Ok(config) => config,
                Err(error) => return Some(error.message),
            };
            nagaya_mk_inkling2_sm121::describe::open_problem(&config, &devices[0].pci_address, freed_bytes)
        }

        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            let err = |message: String| OpenError {
                kind: OpenErrorKind::Other,
                message,
            };
            check_devices(devices).map_err(|e| err(e.message))?;
            let config = config(table).map_err(|e| err(e.message))?;
            let pci = devices[0].pci_address.clone();
            let engine = Inkling2Sm121Engine::open(config.clone(), &pci).map_err(err)?;
            let key = (config, pci);
            let described = self.described.lock().expect("the lock of the descriptions");
            if let Some((_, identity)) = described.iter().find(|(seen, _)| *seen == key)
                && let Some(message) = identity_mismatch(identity, &engine.identity())
            {
                return Err(err(message));
            }
            Ok(Box::new(engine))
        }
    }
}
