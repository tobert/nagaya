//! The Qwen3-TTS VoiceDesign engine on one B70 card as a factory kind, and its speech service (spec 2.4, 11.8;
//! engine-abi "Speech").
//!
//! The config parse, the plan and the family need no card, so the host tests cover them. The factory opens the card
//! only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{Limits, PromptPair};
use nagaya_mk_qwen3tts_bmg::Config;
use nagaya_mk_qwen3tts_bmg::spec::{KIND as ENGINE_KIND, MODEL};
use nagaya_text_qwen3tts::{Error as FamilyError, Qwen3TtsFamily};

use crate::admission::{GroupDevice, PlanError};
use crate::process::Ports;
use crate::profile::PlacementProfile;
use crate::speech::{SpeechConfig, SpeechError, SpeechFamily, SpeechService};

/// The engine kind in a placement profile.
pub const KIND: &str = ENGINE_KIND;

/// The variable that names the checkpoint directory when a table has no key `model_dir`.
pub const MODEL_VARIABLE: &str = "QWEN3TTS_MODEL_DIR";

/// The keys of the config, each with its default. `model_dir` has no default.
const DEFAULTS: [(&str, u32); 4] = [
    ("slots", 2),
    // 1,500 frames are 120 s of audio.
    ("max_frames", 1500),
    ("max_prompt", 512),
    ("frames_per_quantum", 8),
];

/// Read the config of the engine. The engine refuses a key that it does not know (ABI "Rust shape").
pub fn config(table: &toml::Table) -> Result<Config, PlanError> {
    let fail = |message: String| PlanError { message };
    let what = format!("the config of `{KIND}`");
    if let Some(key) = table
        .keys()
        .find(|key| key.as_str() != "model_dir" && DEFAULTS.iter().all(|(name, _)| name != key))
    {
        return Err(fail(format!(
            "{what} has the key `{key}`, which the engine does not know; the keys are `model_dir`, `slots`, \
             `max_frames`, `max_prompt` and `frames_per_quantum`"
        )));
    }
    let model_dir = match table.get("model_dir") {
        Some(value) => PathBuf::from(value.as_str().ok_or_else(|| {
            fail(format!(
                "the key `model_dir` of {what} is not a string; write the path of the checkpoint directory"
            ))
        })?),
        None => PathBuf::from(std::env::var_os(MODEL_VARIABLE).ok_or_else(|| {
            fail(format!(
                "{what} has no key `model_dir`, and {MODEL_VARIABLE} is not set; name the checkpoint directory"
            ))
        })?),
    };
    let mut values = [0u32; 4];
    for (value, (key, default)) in values.iter_mut().zip(DEFAULTS) {
        *value = match table.get(key) {
            None => default,
            Some(given) => given
                .as_integer()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| fail(format!("`{key}` in {what} is {given}, not a count")))?,
        };
    }
    let [slots, max_frames, max_prompt, frames_per_quantum] = values;
    let config = Config {
        model_dir,
        slots,
        max_frames,
        max_prompt,
        frames_per_quantum,
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

/// The limits of the engine for its config and its device. It reads no weight bytes (spec 2.4).
pub fn plan(table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
    check_devices(devices)?;
    let config = config(table)?;
    nagaya_mk_qwen3tts_bmg::plan(&config).map_err(|message| PlanError { message })
}

/// The family of the engine as a speech family (spec 11.3, 11.8).
pub struct Family(pub Qwen3TtsFamily);

impl SpeechFamily for Family {
    fn prompt(&self, text: &str, instructions: &str, language: &str) -> Result<Vec<PromptPair>, SpeechError> {
        let positions = self
            .0
            .prompt(text, instructions, language)
            .map_err(|error| match error {
                FamilyError::Language { .. } => SpeechError::Invalid {
                    param: "language",
                    message: error.to_string(),
                },
                other => SpeechError::Failed(other.to_string()),
            })?;
        Ok(positions
            .into_iter()
            .map(|p| PromptPair {
                text: p.text,
                codec: p.codec,
            })
            .collect())
    }
}

/// The speech service of each engine of this kind in the profile. The family opens from the checkpoint of the
/// engine, so a service exists also while its engine is unloaded (spec 3.3a).
pub fn speech_services(profile: &PlacementProfile, ports: &Ports) -> Result<Vec<SpeechService>, String> {
    let mut services = Vec::new();
    for placement in profile.engines.iter().filter(|placement| placement.kind == KIND) {
        let config = config(&placement.config).map_err(|error| error.message)?;
        let family = Qwen3TtsFamily::from_dir(&config.model_dir).map_err(|error| {
            format!(
                "cannot build the Qwen3-TTS family of engine `{}` from `{}`: {error}",
                placement.name,
                config.model_dir.display()
            )
        })?;
        let service = SpeechService::new(
            SpeechConfig::new(MODEL, &placement.name, config.slots),
            Arc::new(Family(family)),
            ports.clone(),
        )?;
        tracing::info!("serving speech model `{MODEL}` on engine `{}`", placement.name);
        services.push(service);
    }
    Ok(services)
}

#[cfg(feature = "xe")]
pub use factory::Qwen3TtsFactory;

#[cfg(feature = "xe")]
mod factory {
    use nagaya_abi::{Engine, Limits};
    use nagaya_mk_qwen3tts_bmg::Qwen3TtsEngine;

    use super::{KIND, check_devices, config};
    use crate::admission::{Factory, GroupDevice, OpenError, OpenErrorKind, PlanError};

    /// The factory of the Qwen3-TTS engine.
    pub struct Qwen3TtsFactory;

    fn open_error(message: String) -> OpenError {
        OpenError {
            kind: OpenErrorKind::Other,
            message,
        }
    }

    impl Factory for Qwen3TtsFactory {
        fn plan(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            super::plan(table, devices)
        }

        /// Upload the weights to the card at the PCI address of the device, and open the engine on it.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let engine = Qwen3TtsEngine::open(config, &devices[0].pci_address)
                .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
            Ok(Box::new(engine))
        }
    }
}
