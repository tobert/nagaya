//! The rendering of a speech context (spec 11.8a): the voice design, then each line and the frames of its audio.

use crate::prompt::PromptPosition;
use crate::{Error, SpeechIds};

/// The codes of one frame: one code of each of the 16 codebooks.
pub type Frame = [u32; 16];

/// One position of a rendered speech context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechPosition {
    /// A text position: a text id and a codec id, either of which can be absent.
    Pair(PromptPosition),
    /// A frame of audio that the model made, which the engine feeds as it fed the frame.
    Frame(Frame),
}

/// One message of a speech context after the voice design, with its text as token ids.
#[derive(Debug, Clone, Copy)]
pub enum SpeechItem<'a> {
    /// A line: the ids of `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    Line(&'a [u32]),
    /// The audio of the line before it, as the frames that the model made.
    Audio(&'a [Frame]),
}

/// The positions of a speech context, and the position count at the end of each message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendering {
    pub positions: Vec<SpeechPosition>,
    /// The position count after the voice design, when the context has one, then after each item.
    pub ends: Vec<usize>,
}

impl SpeechIds {
    /// Render a speech context in one language (spec 11.8a). `voice` is the ids of
    /// `<|im_start|>user\n{system}<|im_end|>\n`. An `Audio` item must follow a `Line`.
    pub fn context(&self, voice: Option<&[u32]>, items: &[SpeechItem], language: &str) -> Result<Rendering, Error> {
        let mut positions = Vec::new();
        let mut ends = Vec::new();
        if let Some(voice) = voice {
            positions.extend(voice.iter().map(|&id| {
                SpeechPosition::Pair(PromptPosition {
                    text: Some(id),
                    codec: None,
                })
            }));
            ends.push(positions.len());
        }
        let mut after_line = false;
        let mut after_audio = false;
        for (item, message) in items.iter().enumerate() {
            match message {
                SpeechItem::Line(assistant) => {
                    if after_audio {
                        // The codec end of the audio before, so the head after a line holds the rows of the engine.
                        positions.push(SpeechPosition::Pair(PromptPosition {
                            text: Some(self.text.pad),
                            codec: Some(self.codec.eos),
                        }));
                    }
                    let line = self.positions(None, assistant, language)?;
                    positions.extend(line.into_iter().map(SpeechPosition::Pair));
                    (after_line, after_audio) = (true, false);
                }
                SpeechItem::Audio(frames) => {
                    if !after_line {
                        return Err(Error::Context {
                            item,
                            message: "audio must follow the line that it speaks".to_owned(),
                        });
                    }
                    positions.extend(frames.iter().map(|f| SpeechPosition::Frame(*f)));
                    (after_line, after_audio) = (false, true);
                }
            }
            ends.push(positions.len());
        }
        Ok(Rendering { positions, ends })
    }
}
