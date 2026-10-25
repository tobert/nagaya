//! The prompt of a speech request: the layout of `Qwen3TTSForConditionalGeneration.generate` for VoiceDesign, with
//! `non_streaming_mode=True` (`qwen_tts/core/models/modeling_qwen3_tts.py`, about lines 2070 to 2240).
//!
//! The reference builds one embedding row for each position. A row is the projected text embedding of a text id,
//! plus the codec embedding of a codec id, or one of the two alone. This module gives the two ids of each row.

use crate::{Error, SpeechIds};

/// The number of tokens of `<|im_start|>assistant\n` at the start of the assistant ids (reference `input_id[:, :3]`).
const ROLE_LEN: usize = 3;
/// The number of tokens of `<|im_end|>\n<|im_start|>assistant\n` at the end (reference `input_id[:, -5:]`).
const TAIL_LEN: usize = 5;

/// One position of a prompt. The engine sums the projected text embedding and the codec embedding of the ids that
/// are present (engine-abi "Speech").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptPosition {
    pub text: Option<u32>,
    pub codec: Option<u32>,
}

fn text_only(id: u32) -> PromptPosition {
    PromptPosition {
        text: Some(id),
        codec: None,
    }
}

fn both(text: u32, codec: u32) -> PromptPosition {
    PromptPosition {
        text: Some(text),
        codec: Some(codec),
    }
}

impl SpeechIds {
    /// The prompt positions of a request, from its token ids.
    ///
    /// - `instructions`: the ids of `<|im_start|>user\n{instructions}<|im_end|>\n`, or `None` for no instructions.
    /// - `assistant`: the ids of `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    /// - `language`: `auto` or a language of the checkpoint, in any case.
    pub fn positions(
        &self,
        instructions: Option<&[u32]>,
        assistant: &[u32],
        language: &str,
    ) -> Result<Vec<PromptPosition>, Error> {
        let language_id = self.language_id(language)?;
        self.check_template(assistant)?;
        let (t, c) = (&self.text, &self.codec);
        let mut out = Vec::new();
        // The instructions and the role are text only.
        out.extend(instructions.unwrap_or(&[]).iter().map(|&id| text_only(id)));
        out.extend(assistant[..ROLE_LEN].iter().map(|&id| text_only(id)));
        // The codec prefill. Its last two codec ids are pad and bos. The reference pairs each codec id but the last
        // with tts_pad, and the pad with tts_bos. The bos is not here: non-streaming mode moves it to the end.
        let prefill = match language_id {
            None => vec![c.nothink, c.think_bos, c.think_eos],
            Some(id) => vec![c.think, c.think_bos, id, c.think_eos],
        };
        out.extend(prefill.iter().map(|&codec| both(t.pad, codec)));
        out.push(both(t.bos, c.pad));
        // Non-streaming mode: each text token and tts_eos with the codec pad, then tts_pad with the codec bos.
        let text = &assistant[ROLE_LEN..assistant.len() - TAIL_LEN];
        out.extend(text.iter().map(|&id| both(id, c.pad)));
        out.push(both(t.eos, c.pad));
        out.push(both(t.pad, c.bos));
        Ok(out)
    }

    /// Check the ids that the reference slices off without a check: the role at the start and the tail at the end.
    fn check_template(&self, assistant: &[u32]) -> Result<(), Error> {
        let t = &self.text;
        let n = assistant.len();
        let fits = n >= ROLE_LEN + TAIL_LEN
            && assistant[0] == t.im_start
            && assistant[1] == t.assistant
            && assistant[n - 5] == t.im_end
            && assistant[n - 3] == t.im_start
            && assistant[n - 2] == t.assistant;
        if fits {
            Ok(())
        } else {
            Err(Error::Prompt(format!(
                "the assistant ids {assistant:?} do not have the form of the assistant template"
            )))
        }
    }
}
