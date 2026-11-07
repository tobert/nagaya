//! The mk engine as a factory kind (spec 2.4; `docs/engine-abi.md`, "v0 mk binding choices").
//!
//! This module exists in a build with the feature `hip`, which links mk's engine statically. The factory reads
//! `MK_SHARDS` from the environment, as the unit sets it. It opens no device until `open`.

use nagaya_abi::{Engine, Identity, Limits};
use nagaya_mk_qwen38_gfx1151::{MkError, MkPaths};

use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

/// The engine kind in a placement profile.
pub const KIND: &str = "mk-qwen38-gfx1151";

/// The key in the engine config that gives the size of mk's weight arena, in bytes.
///
/// mk knows the arena size only after the weights load, and `plan` reads no weights (spec 2.4). The profile gives the
/// size. `open` checks it against the real arena.
pub const WEIGHTS_KEY: &str = "weights_bytes";

/// The factory of the mk engine.
pub struct MkFactory;

impl Factory for MkFactory {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        check_devices(devices)?;
        check_install()?;
        Ok(nagaya_mk_qwen38_gfx1151::plan(weights_bytes(config)?))
    }

    fn open(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        check_devices(devices).map_err(|error| OpenError {
            kind: OpenErrorKind::Other,
            message: error.message,
        })?;
        let planned = nagaya_mk_qwen38_gfx1151::plan(weights_bytes(config).map_err(|error| OpenError {
            kind: OpenErrorKind::Other,
            message: error.message,
        })?);
        let engine = nagaya_mk_qwen38_gfx1151::open().map_err(open_error)?;
        weights_problem(&planned, &engine.limits())?;
        // The council may serve this engine from its description (spec 11.1). Another identity would give the
        // council's recipes the wrong root, so it is an error.
        let (identity, _) = nagaya_mk_qwen38_gfx1151::describe().map_err(open_error)?;
        if identity != engine.identity() {
            return Err(OpenError {
                kind: OpenErrorKind::Other,
                message: "the opened mk engine has another identity than its description; the shards or the \
                          knobs changed after the start, so restart the process"
                    .to_owned(),
            });
        }
        Ok(Box::new(engine))
    }

    /// mk runs on the one HIP device of its thread, so the description reads no device.
    fn describe(
        &self,
        config: &toml::Table,
        _devices: &[GroupDevice],
    ) -> Result<Option<(Identity, Limits)>, PlanError> {
        let planned = nagaya_mk_qwen38_gfx1151::plan(weights_bytes(config)?);
        let (identity, limits) = nagaya_mk_qwen38_gfx1151::describe().map_err(|error| PlanError {
            message: error.message().to_owned(),
        })?;
        weights_problem(&planned, &limits).map_err(|error| PlanError { message: error.message })?;
        Ok(Some((identity, limits)))
    }
}

/// Refuse a group of any size but one device. The mk engine runs on the one HIP device of its thread.
///
/// The engine does not select the device by its PCI address yet (engine ABI "The factory gets its devices").
fn check_devices(devices: &[GroupDevice]) -> Result<(), PlanError> {
    if devices.len() == 1 {
        return Ok(());
    }
    Err(PlanError {
        message: format!(
            "the mk engine runs on exactly one device, but its profile names {}: name one device in `devices`",
            devices.len()
        ),
    })
}

/// Read the weights size from the engine config. The engine refuses a key that it does not know (ABI "Rust shape").
fn weights_bytes(config: &toml::Table) -> Result<u64, PlanError> {
    let fail = |message: String| PlanError { message };
    if let Some(key) = config.keys().find(|key| key.as_str() != WEIGHTS_KEY) {
        return Err(fail(format!(
            "the config of `{KIND}` has the key `{key}`, which the engine does not know; the only key is `{WEIGHTS_KEY}`"
        )));
    }
    let value = config.get(WEIGHTS_KEY).ok_or_else(|| {
        fail(format!(
            "the config of `{KIND}` needs `{WEIGHTS_KEY}`, the size of mk's weight arena in bytes; \
             a start with a wrong value reports the real size"
        ))
    })?;
    match value.as_integer().and_then(|bytes| u64::try_from(bytes).ok()) {
        Some(bytes) if bytes > 0 => Ok(bytes),
        _ => Err(fail(format!(
            "`{WEIGHTS_KEY}` in the config of `{KIND}` must be a whole number of bytes above 0; it is {value}"
        ))),
    }
}

/// Check the environment and the files that it names. Each message names the variable.
///
/// A start that fails here is a refusal: exit 3 (spec 10.4). A failure inside `open` for a missing file would be a
/// crash, and systemd would start the process again.
fn check_install() -> Result<(), PlanError> {
    let paths = MkPaths::from_env().map_err(|error| PlanError {
        message: error.message().to_owned(),
    })?;
    for shard in &paths.shards {
        if !shard.is_file() {
            return Err(PlanError {
                message: format!(
                    "MK_SHARDS names `{}`, which is not a file; give the three GGUF shards of the model",
                    shard.display()
                ),
            });
        }
    }
    Ok(())
}

/// Map an error of the mk binding to an open error. Each one is a crash (spec 10.4): the binding has no refusal.
pub fn open_error(error: MkError) -> OpenError {
    OpenError {
        kind: OpenErrorKind::Other,
        message: error.message().to_owned(),
    }
}

/// Refuse an engine whose weights term differs from the plan that the ledger admitted.
///
/// The message gives the real size, so the operator can write it into the profile. The ledger holds the planned
/// bytes, so a different arena breaks the budget (spec 4.3).
pub fn weights_problem(planned: &Limits, opened: &Limits) -> Result<(), OpenError> {
    let weights = |limits: &Limits| {
        limits
            .memory
            .mandatory
            .iter()
            .find(|term| term.name == "weights")
            .map(|term| term.bytes)
    };
    let (Some(planned), Some(opened)) = (weights(planned), weights(opened)) else {
        return Err(OpenError {
            kind: OpenErrorKind::Other,
            message: "an mk limits table has no weights term".to_owned(),
        });
    };
    if planned == opened {
        return Ok(());
    }
    Err(OpenError {
        kind: OpenErrorKind::PartSize,
        message: format!(
            "mk's weight arena is {opened} bytes, and the profile plans {planned} bytes; \
             set `{WEIGHTS_KEY} = {opened}` in the config of the engine"
        ),
    })
}
