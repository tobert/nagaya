//! The reply parser of the Gemma 4 family: mkviolet's reply splitter (`Splitter` in `src/mk_api.rs`, council wire)
//! behind the `ReplyParser` trait (spec 11.3).
//!
//! The caller pushes every sampled token, the stop token too. A stop token ends the reply, adds no text, and makes
//! `ParsedReply::ended` true. A marker token adds no text: a piece is display text.

use std::sync::Arc;

use gogi_proto::generation::Phase;
use nagaya_council::family::{FamilyError, ParsedReply, ReplyCall, ReplyParser, ReplyStart, ReplyStep};

use super::Gemma4Family;
use crate::call::parse_tool_call;
use crate::tokenizer::Tokenizer;

/// The ids of the template's markers that split a reply into its parts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Marks {
    pub channel_open: u32,
    pub channel_close: u32,
    pub tool_open: u32,
    pub tool_close: u32,
}

/// The name of the thought channel, as the template writes it after `<|channel>`.
const CHANNEL_NAME: &str = "thought";

pub(crate) fn reply_parser(family: &Gemma4Family, start: ReplyStart) -> Box<dyn ReplyParser> {
    Box::new(Gemma4ReplyParser {
        tokenizer: family.shared_tokenizer(),
        marks: family.marks,
        stop: family.stop.clone(),
        phase: if start.thinking {
            Phase::Reasoning
        } else {
            Phase::Content
        },
        self_opened: false,
        think_sent: 0,
        think: String::new(),
        answer: String::new(),
        call: None,
        calls: Vec::new(),
        segment: Vec::new(),
        segment_len: 0,
        ended: false,
    })
}

struct Gemma4ReplyParser {
    tokenizer: Arc<Tokenizer>,
    marks: Marks,
    stop: Vec<u32>,
    phase: Phase,
    /// A thought channel that the model opened itself before any answer, as Gemma 4 does after a tool result with
    /// thinking off. Its text is reasoning, and its markers are not shown.
    self_opened: bool,
    /// Of the reasoning of a self-opened channel, the bytes that the steps gave.
    think_sent: usize,
    think: String,
    answer: String,
    /// The body of the open call.
    call: Option<String>,
    calls: Vec<ReplyCall>,
    /// The tokens of the current run of text, decoded together, so that a character split across tokens comes out
    /// whole.
    segment: Vec<u32>,
    segment_len: usize,
    /// True after a stop token.
    ended: bool,
}

impl Gemma4ReplyParser {
    /// Start a new run of text. The bytes of a character that the old run did not finish are dropped, as in mkviolet
    /// (`Splitter::new_segment`): a marker gives no text, and the bytes belong to the phase before it (spec 11.3).
    fn new_segment(&mut self) {
        self.segment.clear();
        self.segment_len = 0;
    }

    /// Give the text of the current run that ended inside a character and was held back for a later token.
    fn flush_segment(&mut self) -> Option<ReplyStep> {
        // `push` decoded these tokens without an error, so an error here is a defect of the tokenizer (spec 8.5).
        let text = self
            .tokenizer
            .decode(&self.segment)
            .expect("the tokens of a run decode again as they decoded at push");
        let delta = text[self.segment_len.min(text.len())..].to_string();
        if delta.is_empty() {
            return None;
        }
        self.segment_len = text.len();
        self.append_delta(delta)
    }

    fn append_delta(&mut self, delta: String) -> Option<ReplyStep> {
        match self.phase {
            Phase::Reasoning => self.think.push_str(&delta),
            Phase::Content => self.answer.push_str(&delta),
            Phase::Call => {
                if let Some(body) = self.call.as_mut() {
                    body.push_str(&delta);
                }
            }
            Phase::Unknown => unreachable!("the parser never sets the phase Unknown"),
        }
        let piece = if self.self_opened && self.phase == Phase::Reasoning {
            self.visible_think()
        } else {
            delta
        };
        Some(ReplyStep {
            phase: self.phase,
            piece,
        })
    }

    fn marker(&mut self, phase: Phase) -> ReplyStep {
        self.new_segment();
        ReplyStep {
            phase,
            piece: String::new(),
        }
    }

    fn close_call(&mut self, body: String, closed: bool) {
        self.calls.push(match (closed, parse_tool_call(&body)) {
            (true, Ok((name, input))) => ReplyCall::Parsed { name, input },
            (true, Err((_, error))) => ReplyCall::Malformed { text: body, error },
            (false, _) => ReplyCall::Malformed {
                text: body,
                error: "the call did not close with <tool_call|>".to_string(),
            },
        });
    }

    /// The text that a token of a self-opened channel shows: the reasoning after the channel's name.
    fn visible_think(&mut self) -> String {
        let visible = if CHANNEL_NAME.starts_with(self.think.as_str()) {
            ""
        } else {
            self.think.strip_prefix(CHANNEL_NAME).unwrap_or(&self.think)
        };
        let piece = visible.get(self.think_sent..).unwrap_or("").to_string();
        self.think_sent = visible.len();
        piece
    }
}

impl ReplyParser for Gemma4ReplyParser {
    fn push(&mut self, token: u32) -> Result<ReplyStep, FamilyError> {
        if self.ended {
            return Err(FamilyError::internal(format!(
                "token {token} came after the stop token: the reply is over"
            )));
        }
        if token as usize >= self.tokenizer.len() {
            return Err(FamilyError::internal(format!(
                "token {token} is not in the vocabulary of {} tokens",
                self.tokenizer.len()
            )));
        }
        if self.stop.contains(&token) {
            // The run stays, so `finish` gives a character that it did not finish in the tail, in this phase.
            self.ended = true;
            return Ok(ReplyStep {
                phase: self.phase,
                piece: String::new(),
            });
        }
        let m = self.marks;
        if token == m.channel_open && self.phase == Phase::Content && self.answer.trim().is_empty() {
            self.phase = Phase::Reasoning;
            self.self_opened = true;
            return Ok(self.marker(Phase::Reasoning));
        }
        if token == m.channel_close && self.phase == Phase::Reasoning {
            self.phase = Phase::Content;
            return Ok(self.marker(Phase::Reasoning));
        }
        // A call that the model opens inside its thought ends the thought, as the template renders the calls of a
        // reply after its reasoning.
        if token == m.tool_open {
            self.phase = Phase::Call;
            self.call = Some(String::new());
            return Ok(self.marker(Phase::Call));
        }
        if token == m.tool_close && self.phase == Phase::Call {
            if let Some(body) = self.call.take() {
                self.close_call(body, true);
            }
            // Text after a closed call is content.
            self.phase = Phase::Content;
            return Ok(self.marker(Phase::Content));
        }
        self.segment.push(token);
        let text = self
            .tokenizer
            .decode(&self.segment)
            .map_err(|e| FamilyError::internal(e.to_string()))?;
        if text.ends_with('\u{FFFD}') {
            // The token ends inside a character. A later token completes it.
            return Ok(ReplyStep {
                phase: self.phase,
                piece: String::new(),
            });
        }
        let delta = text[self.segment_len.min(text.len())..].to_string();
        self.segment_len = text.len();
        let step = self.append_delta(delta).unwrap_or(ReplyStep {
            phase: self.phase,
            piece: String::new(),
        });
        Ok(step)
    }

    fn ended(&self) -> bool {
        self.ended
    }

    fn phase(&self) -> Phase {
        self.phase
    }

    fn finish(mut self: Box<Self>, _max_tokens: bool) -> ParsedReply {
        let mut tail = Vec::new();
        if let Some(step) = self.flush_segment()
            && !step.piece.is_empty()
        {
            tail.push(step);
        }
        if let Some(body) = self.call.take() {
            self.close_call(body, false);
        }
        let reasoning = if self.self_opened {
            self.think.strip_prefix(CHANNEL_NAME).unwrap_or(&self.think).to_string()
        } else {
            std::mem::take(&mut self.think)
        };
        ParsedReply {
            content: std::mem::take(&mut self.answer),
            reasoning,
            calls: std::mem::take(&mut self.calls),
            ended: self.ended,
            tail,
        }
    }
}

#[cfg(test)]
mod tests {
    use gogi_proto::generation::Phase;
    use nagaya_council::family::{ParsedReply, ReplyCall, ReplyParser};

    use super::{Gemma4ReplyParser, Marks};
    use crate::tokenizer::Tokenizer;
    use std::sync::Arc;

    const MINI_TOKENIZER: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "<|turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 3, "content": "<turn|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 4, "content": "<|channel>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 5, "content": "<channel|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 6, "content": "<|tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 7, "content": "<tool_call|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 8, "content": "<|tool_response>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {"type": "ByteFallback"},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": {
                "<bos>": 0,
                "<eos>": 1,
                "<|turn>": 2,
                "<turn|>": 3,
                "<|channel>": 4,
                "<channel|>": 5,
                "<|tool_call>": 6,
                "<tool_call|>": 7,
                "<|tool_response>": 8,
                "hi": 9,
                "<0xF0>": 10,
                "<0x9F>": 11
            },
            "merges": []
        }
    }"#;

    /// The tokens of `MINI_TOKENIZER`: `hi`, the first byte of a four-byte character, the markers and a stop token.
    const HI: u32 = 9;
    const SPLIT: u32 = 10;
    const CHANNEL_OPEN: u32 = 4;
    const CHANNEL_CLOSE: u32 = 5;
    const TOOL_OPEN: u32 = 6;
    const TOOL_CLOSE: u32 = 7;
    const EOS: u32 = 1;

    fn parser(thinking: bool) -> Box<dyn ReplyParser> {
        let tokenizer = Arc::new(Tokenizer::from_bytes(MINI_TOKENIZER.as_bytes()).unwrap());
        Box::new(Gemma4ReplyParser {
            tokenizer,
            marks: Marks {
                channel_open: CHANNEL_OPEN,
                channel_close: CHANNEL_CLOSE,
                tool_open: TOOL_OPEN,
                tool_close: TOOL_CLOSE,
            },
            stop: vec![EOS, 2, 3, 8],
            phase: if thinking { Phase::Reasoning } else { Phase::Content },
            self_opened: false,
            think_sent: 0,
            think: String::new(),
            answer: String::new(),
            call: None,
            calls: Vec::new(),
            segment: Vec::new(),
            segment_len: 0,
            ended: false,
        })
    }

    /// Push the tokens and finish. Then check that the pieces of each part join to its text in the reply, with the
    /// tail on the last step, as the service streams them (spec 11.3c).
    fn run(thinking: bool, tokens: &[u32]) -> ParsedReply {
        let mut parser = parser(thinking);
        let mut steps = Vec::new();
        for &token in tokens {
            steps.push(parser.push(token).unwrap());
        }
        let reply = parser.finish(tokens.last() != Some(&EOS));
        for step in &reply.tail {
            assert_eq!(
                step.phase,
                steps.last().unwrap().phase,
                "the tail has the phase of the last token"
            );
        }
        let joined = |phase: Phase| -> String {
            steps
                .iter()
                .chain(&reply.tail)
                .filter(|step| step.phase == phase)
                .map(|step| step.piece.as_str())
                .collect()
        };
        assert_eq!(
            joined(Phase::Reasoning),
            reply.reasoning,
            "the reasoning pieces join to the reasoning"
        );
        assert_eq!(
            joined(Phase::Content),
            reply.content,
            "the content pieces join to the content"
        );
        let call_text: String = reply
            .calls
            .iter()
            .map(|call| match call {
                ReplyCall::Malformed { text, .. } => text.as_str(),
                ReplyCall::Parsed { .. } => panic!("the test calls do not parse"),
            })
            .collect();
        assert_eq!(joined(Phase::Call), call_text, "the call pieces join to the call text");
        reply
    }

    #[test]
    fn a_reply_ending_inside_a_character_flushes_its_held_text_on_finish() {
        let mut parser = parser(false);
        assert_eq!(parser.push(HI).unwrap().piece, "hi");
        // `SPLIT` (`<0xF0>`) is the start of a four-byte UTF-8 character and decodes to U+FFFD on its own.
        assert_eq!(parser.push(SPLIT).unwrap().piece, "");
        let reply = parser.finish(true);
        assert_eq!(reply.content, "hi\u{FFFD}");
        assert_eq!(reply.tail.len(), 1);
        assert_eq!(reply.tail[0].piece, "\u{FFFD}");
    }

    #[test]
    fn a_stop_token_after_an_unfinished_character_gives_it_in_the_tail() {
        let reply = run(false, &[HI, SPLIT, EOS]);
        assert!(reply.ended);
        assert_eq!(reply.content, "hi\u{FFFD}");
        assert_eq!(reply.tail.len(), 1, "{:?}", reply.tail);
        assert_eq!(
            (reply.tail[0].phase, reply.tail[0].piece.as_str()),
            (Phase::Content, "\u{FFFD}")
        );
    }

    #[test]
    fn a_phase_marker_drops_an_unfinished_character_from_every_part() {
        // `<channel|>` after held reasoning.
        let reply = run(true, &[HI, SPLIT, CHANNEL_CLOSE, HI]);
        assert_eq!((reply.reasoning.as_str(), reply.content.as_str()), ("hi", "hi"));
        // `<|channel>` after held content, which opens a thought.
        let reply = run(false, &[SPLIT, CHANNEL_OPEN, HI, CHANNEL_CLOSE, HI]);
        assert_eq!((reply.reasoning.as_str(), reply.content.as_str()), ("hi", "hi"));
        // `<|tool_call>` after held content.
        let reply = run(false, &[HI, SPLIT, TOOL_OPEN, HI, TOOL_CLOSE, HI]);
        assert_eq!(reply.content, "hihi");
        // `<tool_call|>` after a held byte of the call.
        let reply = run(false, &[TOOL_OPEN, HI, SPLIT, TOOL_CLOSE, HI]);
        assert_eq!(reply.content, "hi");
        assert!(
            matches!(&reply.calls[..], [ReplyCall::Malformed { text, .. }] if text == "hi"),
            "{:?}",
            reply.calls
        );
    }
}
