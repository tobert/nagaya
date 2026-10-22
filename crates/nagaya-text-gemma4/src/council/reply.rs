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
    fn new_segment(&mut self) {
        self.segment.clear();
        self.segment_len = 0;
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
            self.ended = true;
            return Ok(self.marker(self.phase));
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
        Ok(ReplyStep {
            phase: self.phase,
            piece,
        })
    }

    fn ended(&self) -> bool {
        self.ended
    }

    fn phase(&self) -> Phase {
        self.phase
    }

    fn finish(mut self: Box<Self>, _max_tokens: bool) -> ParsedReply {
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
            tail: Vec::new(),
        }
    }
}
