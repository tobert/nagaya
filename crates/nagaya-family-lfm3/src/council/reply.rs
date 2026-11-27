//! The reply parser of the LFM3 family (spec 11.3, 11.11).
//!
//! The parser reads tokens. Each marker of a reply is one added token: `<think>`, `</think>`, `<tool_call>` and
//! `</tool_call>`. The caller pushes every sampled token, the stop token too. A stop token ends the reply and adds no
//! text.
//!
//! - With thinking on, the reply starts in the reasoning, and `</think>` ends it. `<think>` before any content or
//!   call opens the reasoning again, as after a prompt with thinking off.
//! - `<tool_call>` in the content opens a call, and `</tool_call>` closes it. The call text includes both markers.
//!   The text between them reads as `crate::call` says.
//! - A marker that opens or closes the reasoning gives an empty piece. Any other token gives its text, once its
//!   characters are whole. A marker drops the bytes of a character that the text before it did not finish, as in the
//!   Gemma 4 library (spec 11.3, "The Gemma 4 choices").
//! - The content and the reasoning lose the white space at their ends, as the reference strips them. A reply that
//!   goes on from an open message keeps the start of its first part, and a reply cut by `max_tokens` keeps the end of
//!   its last part (gogi "Continuations").

use std::sync::Arc;

use gogi_proto::context::Tool;
use gogi_proto::generation::Phase;
use nagaya_council::family::{FamilyError, ParsedReply, ReplyCall, ReplyParser, ReplyStart, ReplyStep};
use nagaya_jinja::pystr::strip;

use super::{Kind, Lfm3Family, Marks};
use crate::call::{ToolParams, parse_call};
use crate::tokenizer::Tokenizer;

pub(crate) fn reply_parser(family: &Lfm3Family, start: ReplyStart, tools: &[Tool]) -> Box<dyn ReplyParser> {
    // Omni does not reason, so its reply never starts in the reasoning (spec 11.11).
    let thinking = start.thinking && family.kind == Kind::Text;
    Box::new(Lfm3ReplyParser {
        tokenizer: family.shared_tokenizer(),
        marks: family.marks,
        stop: family.stop.clone(),
        tools: ToolParams::new(tools),
        first: if thinking { Phase::Reasoning } else { Phase::Content },
        phase: if thinking { Phase::Reasoning } else { Phase::Content },
        open: start.open,
        reasoning: String::new(),
        content: String::new(),
        call: None,
        calls: Vec::new(),
        held: Vec::new(),
        ended: false,
    })
}

struct Lfm3ReplyParser {
    tokenizer: Arc<Tokenizer>,
    marks: Marks,
    stop: Vec<u32>,
    tools: ToolParams,
    /// The phase of the first token.
    first: Phase,
    phase: Phase,
    /// True when the reply goes on from an open message.
    open: bool,
    reasoning: String,
    content: String,
    /// The text of the open call, from its `<tool_call>`.
    call: Option<String>,
    calls: Vec<ReplyCall>,
    /// The bytes of a character that is not whole yet.
    held: Vec<u8>,
    ended: bool,
}

/// The longest start of `bytes` that is whole UTF-8 text, as text, and the rest. Bytes that can never be text become
/// U+FFFD, so only the start of an unfinished character stays in the rest.
fn whole_text(bytes: &[u8]) -> (String, Vec<u8>) {
    let mut out = String::new();
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(text) => {
                out.push_str(text);
                return (out, Vec::new());
            }
            Err(e) => {
                let (good, bad) = rest.split_at(e.valid_up_to());
                out.push_str(std::str::from_utf8(good).expect("the start is valid"));
                match e.error_len() {
                    Some(n) => {
                        out.push('\u{FFFD}');
                        rest = &bad[n..];
                    }
                    None => return (out, bad.to_vec()),
                }
            }
        }
    }
}

impl Lfm3ReplyParser {
    /// Add text to the part of the phase now.
    fn add(&mut self, text: &str) -> ReplyStep {
        match self.phase {
            Phase::Reasoning => self.reasoning.push_str(text),
            Phase::Content => self.content.push_str(text),
            Phase::Call => self
                .call
                .as_mut()
                .expect("the phase Call has an open call")
                .push_str(text),
            Phase::Unknown => unreachable!("the parser never sets the phase Unknown"),
        }
        ReplyStep {
            phase: self.phase,
            piece: text.to_string(),
        }
    }

    /// A marker of the template: the held bytes go, and the phase may change.
    fn marker(&mut self, token: u32) -> Option<ReplyStep> {
        let m = self.marks;
        let quiet = |phase| ReplyStep {
            phase,
            piece: String::new(),
        };
        let step = match self.phase {
            Phase::Reasoning if token == m.think_close => {
                self.phase = Phase::Content;
                quiet(Phase::Reasoning)
            }
            Phase::Content if token == m.think_open && self.content.trim().is_empty() && self.calls.is_empty() => {
                self.phase = Phase::Reasoning;
                quiet(Phase::Reasoning)
            }
            Phase::Content if token == m.call_open => {
                self.held.clear();
                self.phase = Phase::Call;
                self.call = Some(String::new());
                return Some(self.add("<tool_call>"));
            }
            Phase::Call if token == m.call_close => {
                self.held.clear();
                let step = self.add("</tool_call>");
                let text = self.call.take().expect("the phase Call has an open call");
                let body = &text["<tool_call>".len()..text.len() - "</tool_call>".len()];
                self.calls.push(match parse_call(body, &self.tools) {
                    Ok((name, input)) => ReplyCall::Parsed { name, input },
                    Err(error) => ReplyCall::Malformed { text, error },
                });
                self.phase = Phase::Content;
                return Some(step);
            }
            _ => return None,
        };
        self.held.clear();
        Some(step)
    }
}

impl ReplyParser for Lfm3ReplyParser {
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
            // The held bytes stay, so `finish` gives them in the tail, in this phase.
            self.ended = true;
            return Ok(ReplyStep {
                phase: self.phase,
                piece: String::new(),
            });
        }
        if let Some(step) = self.marker(token) {
            return Ok(step);
        }
        let bytes = self
            .tokenizer
            .token_bytes(token)
            .map_err(|e| FamilyError::internal(e.to_string()))?;
        self.held.extend(bytes);
        let (text, rest) = whole_text(&self.held);
        self.held = rest;
        Ok(self.add(&text))
    }

    fn ended(&self) -> bool {
        self.ended
    }

    fn phase(&self) -> Phase {
        self.phase
    }

    fn finish(mut self: Box<Self>, max_tokens: bool) -> ParsedReply {
        let mut tail = Vec::new();
        if !self.held.is_empty() {
            self.held.clear();
            tail.push(self.add("\u{FFFD}"));
        }
        if let Some(text) = self.call.take() {
            self.calls.push(ReplyCall::Malformed {
                text,
                error: "the call did not close with </tool_call>".to_string(),
            });
        }
        // A part keeps its start when the reply goes on from an open message and the part is the first, and its end
        // when `max_tokens` cut the reply in that part.
        let last = self.phase;
        let keep_end = max_tokens && !self.ended;
        let trim = |text: &str, phase: Phase| {
            let left = !(self.open && phase == self.first);
            let right = !(keep_end && phase == last);
            strip(text, None, left, right)
        };
        ParsedReply {
            content: trim(&self.content, Phase::Content),
            reasoning: trim(&self.reasoning, Phase::Reasoning),
            calls: std::mem::take(&mut self.calls),
            ended: self.ended,
            tail,
        }
    }
}
