//! The LFM3 engine on one B70 card as a factory kind, and the opener of its family library (spec 2.4, 11.3, 11.11;
//! engine-abi "The first LFM3 binding").
//!
//! One kind serves `LFM3-1.7B` and `LFM3-Omni-2B`, so the config names the model folder, and the folder has no default.
//! The config parse, the plan and the family opener need no card, so the host tests cover them. The factory opens the
//! card only in a build with the feature `xe`. `Registry::compiled_in` registers the kind in that build only.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nagaya_abi::{Identity, Limits};
use nagaya_mk_lfm3_bmg::spec::{self, Config};

use crate::admission::{GroupDevice, PlanError};
use crate::council::speech_out::SpeechOut;
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
        .find(|key| !matches!(key.as_str(), "model_dir" | "slots" | "max_block" | "extra_runs"))
    {
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
    // A count gives slots of the default size. A list gives the positions of each slot (engine-abi "Slot sizes").
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
    // The reserve of extra runs in each slot, for layer loops (engine-abi "Layer loops of LFM3"). With 0, the engine
    // runs no loop.
    let extra_runs = match table.get("extra_runs") {
        None => 0,
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

/// The objects of the process, as the LFM3 family reads the bytes of a media part (spec 6.15, 11.11).
pub struct StoreObjects(Arc<Mutex<nagaya_store::objects::ObjectStore>>);

impl StoreObjects {
    /// A reader of `store`.
    pub fn new(store: Arc<Mutex<nagaya_store::objects::ObjectStore>>) -> Self {
        Self(store)
    }
}

impl nagaya_family_lfm3::Objects for StoreObjects {
    /// The bytes of a held object. An object that the store does not hold, or whose file went, is none: the family
    /// then refuses the part, and the client puts it again (gogi "Objects and keeps").
    fn bytes(&self, id: &str) -> Option<Vec<u8>> {
        let id = nagaya_store::ContentId::from_hex(id.strip_prefix("blake3:")?).ok()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the clock is before the epoch")
            .as_millis() as u64;
        let found = match self.0.lock().expect("the object store lock is poisoned").get(&id, now) {
            Ok(found) => found?,
            Err(error) => {
                tracing::warn!("cannot read the object {}: {error}", id.hex());
                return None;
            }
        };
        std::fs::read(&found.path).ok()
    }
}

/// The served model names of an engine of this kind (spec 11.11, "The served name of an LFM3 engine").
/// The engine name is the served name, so two engines of one checkpoint serve two names.
pub fn served_names(engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
    Some(vec![engine.to_owned()])
}

/// The LFM3 family from the model folder: the key `model_dir` of `[engines.family]`. The registry compares its
/// tokenizer hash with the identity (spec 11.3).
pub fn open_family(identity: &Identity, table: &toml::Table) -> Result<OpenedFamily, String> {
    open_family_with(identity, table, None)
}

/// Like [`open_family`], with the objects of the process: the family of Omni reads the media parts from them.
pub fn open_family_with(
    identity: &Identity,
    table: &toml::Table,
    objects: Option<&crate::serving::ObjectReader>,
) -> Result<OpenedFamily, String> {
    let dir = model_dir(table, "the table `[engines.family]`")?;
    let mut family = nagaya_family_lfm3::Lfm3Family::from_dir(&dir)
        .map_err(|error| format!("cannot build the LFM3 family from `{}`: {error}", dir.display()))?;
    if let Some(objects) = objects {
        family = family.with_objects(Arc::new(StoreObjects::new(Arc::clone(objects))));
    }
    let model = match family.kind() {
        nagaya_family_lfm3::Kind::Text => spec::Model::Text,
        nagaya_family_lfm3::Kind::Omni => spec::Model::Omni,
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
    let speech = match model {
        spec::Model::Omni => Some(speech_out(&family, &dir)?),
        spec::Model::Text => None,
    };
    Ok(OpenedFamily {
        family: Arc::new(family),
        tokenizer_hash: Some(hash),
        speech,
        served: crate::serving::ServedName::Engine,
    })
}

/// The speech path of Omni (spec 11.11): the markers from the family, and the blocks of an interleaved reply from
/// `config.json`. The ids of `config.json` must be the tokens of the family, or the folder is refused.
fn speech_out(family: &nagaya_family_lfm3::Lfm3Family, dir: &std::path::Path) -> Result<SpeechOut, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| format!("cannot read `{}`: {error}", path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("`{}` is not JSON: {error}", path.display()))?;
    let number = |key: &str| {
        config[key].as_u64().ok_or_else(|| {
            format!(
                "`{}` has no number `{key}`, which the speech of Omni needs",
                path.display()
            )
        })
    };
    let token = |text: &str, key: &str| -> Result<u32, String> {
        use nagaya_council::family::Family;
        let ids = family
            .tokenize_special(text)
            .map_err(|error| format!("the tokenizer of `{}` cannot read {text}: {error}", dir.display()))?;
        let want = number(key)?;
        match ids.as_slice() {
            [id] if u64::from(*id) == want => Ok(*id),
            _ => Err(format!(
                "`{}` names {key} {want}, and the tokenizer gives {ids:?} for {text}; name the folder of one checkpoint",
                path.display()
            )),
        }
    };
    let [audio_start, mixed_start] = family.text_suppressed();
    let speech = SpeechOut {
        audio_start: token("<|audio_start|>", "audio_start_token_id")?,
        mixed_start: token("<|mixed_start|>", "mixed_start_token_id")?,
        text_end: token("<|text_end|>", "text_end_token_id")?,
        audio_out: token("<|audio_out|>", "audio_out_token_id")?,
        n_text: number("interleaved_n_text")? as usize,
        n_audio: number("interleaved_n_audio")? as usize,
    };
    if [speech.audio_start, speech.mixed_start] != [audio_start, mixed_start] {
        return Err(format!(
            "the family of `{}` has other markers than its config; this is a defect of the family",
            dir.display()
        ));
    }
    if speech.n_text == 0 || speech.n_audio == 0 {
        return Err(format!(
            "`{}` names a block of 0 tokens or 0 frames; an interleaved reply needs at least 1 of each",
            path.display()
        ));
    }
    Ok(speech)
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

        /// The served model is the engine name (spec 11.11, "The served name of an LFM3 engine").
        fn served_names(&self, engine: &str, config: &toml::Table) -> Option<Vec<String>> {
            super::served_names(engine, config)
        }

        /// Upload the backbone to the card at the PCI address of the device, and open the engine on it.
        fn open(&self, table: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            check_devices(devices).map_err(|error| open_error(error.message))?;
            let config = config(table).map_err(|error| open_error(error.message))?;
            let engine = Lfm3Engine::open(config, &devices[0].pci_address)
                .map_err(|message| open_error(format!("`{KIND}` cannot open: {message}")))?;
            Ok(Box::new(engine))
        }
    }
}
