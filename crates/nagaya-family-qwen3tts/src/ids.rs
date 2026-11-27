//! The ids of the checkpoint's `config.json` that the prompt and the decode need.

use crate::Error;
use serde_json::Value;

/// The language name that lets the model choose the language. Its prompt has no language id.
pub const AUTO: &str = "auto";

/// The size of the codec id range at the end of the codec vocabulary that the decode suppresses, except end-of-speech.
/// Source: `Qwen3TTSForConditionalGeneration.generate`, `suppress_tokens`.
const SUPPRESS_COUNT: u32 = 1024;

/// Ids of the text vocabulary, from the top level of `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIds {
    /// `tts_pad_token_id`: the text of a position that has only codec content.
    pub pad: u32,
    /// `tts_bos_token_id`: the text of the last codec prefill position.
    pub bos: u32,
    /// `tts_eos_token_id`: the text after the last text token of the request.
    pub eos: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub assistant: u32,
}

/// Ids of the codec vocabulary, from `talker_config` of `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecIds {
    pub pad: u32,
    /// The start of speech.
    pub bos: u32,
    /// The end of speech.
    pub eos: u32,
    /// The think tag of a prompt with a language id.
    pub think: u32,
    /// The think tag of a prompt with no language id.
    pub nothink: u32,
    pub think_bos: u32,
    pub think_eos: u32,
    /// The number of codec ids.
    pub vocab_size: u32,
}

/// The ids of a checkpoint: text ids, codec ids and the language ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechIds {
    pub text: TextIds,
    pub codec: CodecIds,
    /// The languages that a request can name, in order of name, with their codec ids.
    languages: Vec<(String, u32)>,
}

fn object<'a>(v: &'a Value, path: &str) -> Result<&'a serde_json::Map<String, Value>, Error> {
    v.as_object()
        .ok_or_else(|| Error::Checkpoint(format!("config.json: {path} is not an object")))
}

fn member<'a>(v: &'a Value, parent: &str, key: &str) -> Result<&'a Value, Error> {
    let path = if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    };
    object(v, if parent.is_empty() { "the file" } else { parent })?
        .get(key)
        .ok_or_else(|| Error::Checkpoint(format!("config.json has no key {path}")))
}

fn id_value(v: &Value, path: &str) -> Result<u32, Error> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::Checkpoint(format!("config.json: {path} is {v}, not an unsigned 32-bit integer")))
}

fn id(v: &Value, parent: &str, key: &str) -> Result<u32, Error> {
    let path = if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    };
    id_value(member(v, parent, key)?, &path)
}

fn require(v: &Value, key: &str, wanted: &str) -> Result<(), Error> {
    match member(v, "", key)?.as_str() {
        Some(s) if s == wanted => Ok(()),
        _ => Err(Error::Checkpoint(format!(
            "config.json: {key} is {}, and this family reads {wanted:?} only",
            member(v, "", key)?
        ))),
    }
}

impl SpeechIds {
    /// Read the ids from the text of `config.json`. The model must be `qwen3_tts` of the type `voice_design`.
    pub fn from_config(text: &str) -> Result<SpeechIds, Error> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| Error::Checkpoint(format!("config.json is not JSON ({e})")))?;
        require(&v, "model_type", "qwen3_tts")?;
        require(&v, "tts_model_type", "voice_design")?;
        let text = TextIds {
            pad: id(&v, "", "tts_pad_token_id")?,
            bos: id(&v, "", "tts_bos_token_id")?,
            eos: id(&v, "", "tts_eos_token_id")?,
            im_start: id(&v, "", "im_start_token_id")?,
            im_end: id(&v, "", "im_end_token_id")?,
            assistant: id(&v, "", "assistant_token_id")?,
        };
        let talker = member(&v, "", "talker_config")?;
        let t = "talker_config";
        let codec = CodecIds {
            pad: id(talker, t, "codec_pad_id")?,
            bos: id(talker, t, "codec_bos_id")?,
            eos: id(talker, t, "codec_eos_token_id")?,
            think: id(talker, t, "codec_think_id")?,
            nothink: id(talker, t, "codec_nothink_id")?,
            think_bos: id(talker, t, "codec_think_bos_id")?,
            think_eos: id(talker, t, "codec_think_eos_id")?,
            vocab_size: id(talker, t, "vocab_size")?,
        };
        if codec.vocab_size < SUPPRESS_COUNT {
            return Err(Error::Checkpoint(format!(
                "config.json: talker_config.vocab_size is {}, below the {SUPPRESS_COUNT} ids of the suppress range",
                codec.vocab_size
            )));
        }
        let path = "talker_config.codec_language_id";
        let mut languages = Vec::new();
        for (name, value) in object(member(talker, t, "codec_language_id")?, path)? {
            let language_id = id_value(value, &format!("{path}.{name}"))?;
            // The reference lists no dialect as a language of a request (`supported_languages`).
            if !name.contains("dialect") {
                languages.push((name.clone(), language_id));
            }
        }
        languages.sort();
        Ok(SpeechIds { text, codec, languages })
    }

    /// The language names that a request can give: `auto`, then the names of the checkpoint in order.
    pub fn languages(&self) -> Vec<&str> {
        std::iter::once(AUTO)
            .chain(self.languages.iter().map(|(name, _)| name.as_str()))
            .collect()
    }

    /// The codec id of a language, or `None` for `auto`. The reference compares the lower case of the name.
    pub fn language_id(&self, name: &str) -> Result<Option<u32>, Error> {
        let lower = name.to_lowercase();
        if lower == AUTO {
            return Ok(None);
        }
        match self.languages.iter().find(|(known, _)| *known == lower) {
            Some((_, language_id)) => Ok(Some(*language_id)),
            None => Err(Error::Language {
                given: name.to_string(),
                known: self.languages().iter().map(|s| s.to_string()).collect(),
            }),
        }
    }

    /// The codec id that ends the speech. The decode stops when codebook 0 gives it.
    pub fn codec_end(&self) -> u32 {
        self.codec.eos
    }

    /// The text id that the decode adds to each generated position after the prompt, in non-streaming mode.
    pub fn trailing_text(&self) -> u32 {
        self.text.pad
    }

    /// The codec ids that codebook 0 never samples: the last 1,024 ids, except the end of speech.
    pub fn suppressed_codecs(&self) -> Vec<u32> {
        (self.codec.vocab_size - SUPPRESS_COUNT..self.codec.vocab_size)
            .filter(|&i| i != self.codec.eos)
            .collect()
    }
}
