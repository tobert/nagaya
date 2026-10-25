//! The family of Qwen3-TTS: the tokenizer and the ids of one checkpoint.

use crate::tokenizer::Tokenizer;
use crate::{Error, SpeechIds};
use std::path::Path;

/// The family library of one Qwen3-TTS VoiceDesign checkpoint.
pub struct Qwen3TtsFamily {
    tokenizer: Tokenizer,
    ids: SpeechIds,
}

impl std::fmt::Debug for Qwen3TtsFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsFamily")
            .field("ids", &self.ids)
            .finish_non_exhaustive()
    }
}

impl Qwen3TtsFamily {
    /// Build the family from the checkpoint directory: `config.json`, `vocab.json`, `merges.txt` and
    /// `tokenizer_config.json`. The text ids of `config.json` must be the ids of their tokens.
    pub fn from_dir(dir: &Path) -> Result<Qwen3TtsFamily, Error> {
        let config_path = dir.join("config.json");
        let config = std::fs::read_to_string(&config_path).map_err(|source| Error::Io {
            path: config_path.display().to_string(),
            source,
        })?;
        let ids = SpeechIds::from_config(&config)?;
        let tokenizer = Tokenizer::from_dir(dir)?;
        let t = &ids.text;
        let named = [
            ("tts_pad_token_id", "<tts_pad>", t.pad),
            ("tts_bos_token_id", "<tts_text_bos>", t.bos),
            ("tts_eos_token_id", "<tts_text_eod>", t.eos),
            ("im_start_token_id", "<|im_start|>", t.im_start),
            ("im_end_token_id", "<|im_end|>", t.im_end),
            ("assistant_token_id", "assistant", t.assistant),
        ];
        for (key, text, id) in named {
            let given = tokenizer.token_id(text);
            if given != Some(id) {
                return Err(Error::Checkpoint(format!(
                    "config.json: {key} is {id}, but the tokenizer gives {text:?} the id {given:?}"
                )));
            }
        }
        Ok(Qwen3TtsFamily { tokenizer, ids })
    }

    /// The token ids of `text`, as the reference processor gives them. The text of an added token becomes that token.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.tokenizer.encode(text)
    }

    /// The ids of the checkpoint, with the constants that the decode needs.
    pub fn ids(&self) -> &SpeechIds {
        &self.ids
    }
}
