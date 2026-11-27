//! The Qwen3-TTS VoiceDesign engine on one B70 card as a factory kind, and its speech service (spec 2.4, 11.8;
//! engine-abi "Speech").
//!
//! The config parse, the plan and the family need no card, so the host tests cover them. The factory opens the card
//! only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::Arc;

use nagaya_abi::{Limits, PromptPair, SpeechPosition};
use nagaya_family_qwen3tts::{Error as FamilyError, Qwen3TtsFamily, SpeechItem, SpeechPosition as TextPosition};
use nagaya_mk_qwen3tts_bmg::Config;
use nagaya_mk_qwen3tts_bmg::spec::{KIND as ENGINE_KIND, MODEL};

use crate::admission::{GroupDevice, PlanError};
use crate::process::Ports;
use crate::profile::PlacementProfile;
use crate::speech::{ContextItem, ContextRendering, SpeechConfig, SpeechError, SpeechFamily, SpeechService};

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
    fn context(
        &self,
        voice: Option<&str>,
        items: &[ContextItem],
        language: &str,
    ) -> Result<ContextRendering, SpeechError> {
        let failed = |error: FamilyError| match error {
            FamilyError::Language { .. } => SpeechError::Invalid {
                param: "output.options.language",
                message: error.to_string(),
            },
            other => SpeechError::Failed(other.to_string()),
        };
        let voice = voice.map(|text| self.0.voice_ids(text)).transpose().map_err(failed)?;
        let lines: Vec<Option<Vec<u32>>> = items
            .iter()
            .map(|item| match item {
                ContextItem::Line(text) => self.0.line_ids(text).map(Some),
                ContextItem::Audio(_) => Ok(None),
            })
            .collect::<Result<_, _>>()
            .map_err(failed)?;
        let rendered: Vec<SpeechItem> = items
            .iter()
            .zip(&lines)
            .map(|(item, ids)| match (item, ids) {
                (ContextItem::Audio(frames), _) => SpeechItem::Audio(frames),
                (ContextItem::Line(_), Some(ids)) => SpeechItem::Line(ids),
                (ContextItem::Line(_), None) => unreachable!("a line has its ids"),
            })
            .collect();
        let rendering = self
            .0
            .ids()
            .context(voice.as_deref(), &rendered, language)
            .map_err(|error| match error {
                FamilyError::Context { item, message } => SpeechError::Invalid {
                    param: "messages",
                    message: format!(
                        "message {} of the context: {message}",
                        item + usize::from(voice.is_some())
                    ),
                },
                other => failed(other),
            })?;
        Ok(ContextRendering {
            positions: rendering
                .positions
                .into_iter()
                .map(|position| match position {
                    TextPosition::Pair(p) => SpeechPosition::Pair(PromptPair {
                        text: p.text,
                        codec: p.codec,
                    }),
                    TextPosition::Frame(codes) => SpeechPosition::Frame(codes.to_vec()),
                })
                .collect(),
            ends: rendering.ends,
        })
    }
}

/// The engines of this kind in the profile, each with its served model. The council reads and pins their snapshots
/// (spec 11.8a).
pub fn peers(profile: &PlacementProfile) -> Vec<(String, String)> {
    profile
        .engines
        .iter()
        .filter(|placement| placement.kind == KIND)
        .map(|placement| (placement.name.clone(), MODEL.to_owned()))
        .collect()
}

/// The speech service of each engine of this kind in the profile. The family opens from the checkpoint of the
/// engine, so a service exists also while its engine is unloaded (spec 3.3a).
pub fn speech_services(
    profile: &PlacementProfile,
    ports: &Ports,
    store_dir: &std::path::Path,
) -> Result<Vec<SpeechService>, String> {
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
        let mut speech = SpeechConfig::new(MODEL, &placement.name, config.slots);
        speech.context_positions = u64::from(config.max_positions());
        let limits = nagaya_mk_qwen3tts_bmg::plan(&config)
            .map_err(|message| format!("cannot plan `{}`: {message}", placement.name))?;
        speech.snapshot_bytes = limits
            .parts
            .iter()
            .map(|part| part.dtype.element_bytes() * part.shape.iter().product::<u64>())
            .sum();
        let service = SpeechService::new(speech, Arc::new(Family(family)), ports.clone())?
            .with_codes(crate::speech::codes::CodeBook::open(store_dir)?);
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
        /// The served model is the fixed model of the kind (spec 2.3, "The check of a name").
        fn served_names(&self, _engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
            Some(vec![super::MODEL.to_owned()])
        }

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
