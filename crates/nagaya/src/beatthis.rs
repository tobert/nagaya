//! The Beat This! engine of the CPU group as a factory kind (spec 3.2c, 11.19). It needs no GPU, so every build
//! registers it.

use std::path::PathBuf;

use nagaya_abi::{Engine, Limits};
use nagaya_mk_beatthis_cpu::engine::{self, BeatThisEngine};

use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

/// The engine kind in a placement profile (spec 11.19).
pub const KIND: &str = "mk-beatthis-cpu";

/// The weights file of the engine's config: its one key `weights`, the path of a converted checkpoint.
pub fn weights(table: &toml::Table) -> Result<PathBuf, PlanError> {
    let fail = |message: String| PlanError { message };
    if let Some(key) = table.keys().find(|key| key.as_str() != "weights") {
        return Err(fail(format!(
            "the config of `{KIND}` has the key `{key}`, which the engine does not know; the key is `weights`"
        )));
    }
    let value = table.get("weights").ok_or_else(|| {
        fail(format!(
            "the config of `{KIND}` has no key `weights`; name a checkpoint converted by tools/beatthis_golden.sh"
        ))
    })?;
    value.as_str().map(PathBuf::from).ok_or_else(|| {
        fail(format!(
            "the key `weights` of `{KIND}` is not a string; write the path of the file"
        ))
    })
}

/// The limits of the engine on `threads` workers. It reads the header of the weights file, and no weight bytes
/// (spec 2.4).
pub fn plan_cpu(table: &toml::Table, threads: u32) -> Result<Limits, PlanError> {
    let path = weights(table)?;
    let config = engine::config_of(&path).map_err(|message| PlanError { message })?;
    Ok(engine::limits(&config, threads as usize))
}

/// The factory of the Beat This! engine.
pub struct BeatThisFactory;

fn open_error(message: String) -> OpenError {
    OpenError {
        kind: OpenErrorKind::Other,
        message,
    }
}

impl Factory for BeatThisFactory {
    /// The served model is the engine name (spec 11.19).
    fn served_names(&self, engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
        Some(vec![engine.to_owned()])
    }

    /// The process plans an engine of the CPU group with `plan_cpu`, so a call here names devices.
    fn plan(&self, _table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Err(PlanError {
            message: format!(
                "the engine `{KIND}` runs in the CPU group, and its profile names {} devices: write `devices = []`",
                devices.len()
            ),
        })
    }

    fn plan_cpu(&self, table: &toml::Table, threads: u32) -> Result<Limits, PlanError> {
        plan_cpu(table, threads)
    }

    /// The process opens an engine of the CPU group with `open_cpu`, so a call here is a profile with devices, which
    /// `plan` refused.
    fn open(&self, _table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Err(open_error(format!(
            "the engine `{KIND}` runs in the CPU group, and it got {} devices: write `devices = []`",
            devices.len()
        )))
    }

    fn open_cpu(&self, table: &toml::Table, threads: u32) -> Result<Box<dyn Engine>, OpenError> {
        let path = weights(table).map_err(|e| open_error(e.message))?;
        let engine = BeatThisEngine::open(&path, threads)
            .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
        Ok(Box::new(engine))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> toml::Table {
        text.parse().unwrap()
    }

    #[test]
    fn the_config_has_one_key_weights() {
        assert_eq!(
            weights(&table("weights = \"/m/final0.safetensors\"")).unwrap(),
            PathBuf::from("/m/final0.safetensors")
        );
        assert!(weights(&table("")).unwrap_err().message.contains("no key `weights`"));
        assert!(
            weights(&table("weights = 3"))
                .unwrap_err()
                .message
                .contains("not a string")
        );
        assert!(
            weights(&table("weights = \"a\"\nthreads = 2"))
                .unwrap_err()
                .message
                .contains("`threads`")
        );
    }

    #[test]
    fn a_plan_with_devices_is_refused() {
        let device = GroupDevice {
            id: 0,
            name: "gpu".into(),
            pci_address: "0000:00:00.0".into(),
            memory_bytes: 1,
        };
        let e = BeatThisFactory
            .plan(&table("weights = \"/m/x\""), &[device])
            .unwrap_err();
        assert!(
            e.message.contains("runs in the CPU group") && e.message.contains("devices = []"),
            "{}",
            e.message
        );
    }

    #[test]
    fn open_with_devices_is_refused() {
        let e = BeatThisFactory.open(&table("weights = \"/m/x\""), &[]).err().unwrap();
        assert!(e.message.contains("CPU group"), "{}", e.message);
    }
}
