//! The reply parser of the Inkling 2 family (spec 11.3, milestone 35 step 3).
//!
//! Follows the `response_template` in `tokenizer_config.json`:
//! - `<|message_model|>` starts a block (with optional header text before `<|content_*|>`).
//! - `<|content_thinking|>...<|end_message|>` is reasoning.
//! - `<|content_text|>`, `<|content_tool_error|>`, and `<|content_invoke_tool_text|>` up to `<|end_message|>` are content.
//! - `<|content_invoke_tool_json|>{"name":...,"args":...}<|end_message|>` is a tool call.
//! - `<|content_model_end_sampling|>` ends the turn.
//! - Blocks repeat and whitespace is not stripped (`strip: false`).

use std::sync::Arc;

use gogi_proto::context::Tool;
use gogi_proto::generation::Phase;
use nagaya_council::family::{FamilyError, ParsedReply, ReplyCall, ReplyParser, ReplyStart, ReplyStep};
use serde_json::{Map, Value};

use super::{Inkling2Family, Marks};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Idle,
    Header,
    Thinking,
    Content,
    Call,
}

pub(crate) fn reply_parser(family: &Inkling2Family, start: ReplyStart, _tools: &[Tool]) -> Box<dyn ReplyParser> {
    // A reply after the generation prompt starts inside `<|message_model|>`, so its header text is not content.
    let state = match (start.open, start.thinking) {
        (true, true) => State::Thinking,
        (true, false) => State::Idle,
        (false, _) => State::Header,
    };
    Box::new(Inkling2ReplyParser {
        tokenizer: family.shared_tokenizer(),
        marks: family.marks,
        stop: family.stop.clone(),
        state,
        reasoning: String::new(),
        content: String::new(),
        call: None,
        calls: Vec::new(),
        held: Vec::new(),
        ended: false,
    })
}

struct Inkling2ReplyParser {
    tokenizer: Arc<Tokenizer>,
    marks: Marks,
    stop: Vec<u32>,
    state: State,
    reasoning: String,
    content: String,
    call: Option<String>,
    calls: Vec<ReplyCall>,
    held: Vec<u8>,
    ended: bool,
}

/// The longest start of `bytes` that is whole UTF-8 text, as text, and the rest.
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

fn has_oversized_int_literal(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                    } else if bytes[i] == b'"' {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                i += 1;
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit() || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
                {
                    i += 1;
                }
                let tok = &text[start..i];
                if !tok.contains(['.', 'e', 'E']) && tok.parse::<i64>().is_err() && tok.parse::<u64>().is_err() {
                    return true;
                }
            }
            _ => i += 1,
        }
    }
    false
}

fn parse_tool_json(text: &str) -> Result<(String, Map<String, Value>), String> {
    if has_oversized_int_literal(text) {
        return Err("tool call JSON holds an integer outside 64 bits".to_string());
    }
    let v: Value = serde_json::from_str(text).map_err(|e| format!("invalid tool call JSON: {e}"))?;
    let obj = v
        .as_object()
        .ok_or_else(|| "tool call JSON must be an object".to_string())?;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "tool call JSON must have a string \"name\"".to_string())?
        .to_string();
    let args = match obj.get("args") {
        Some(Value::Object(m)) => m.clone(),
        Some(other) => return Err(format!("tool call \"args\" must be an object, got {other}")),
        None => Map::new(),
    };
    Ok((name, args))
}

impl Inkling2ReplyParser {
    fn phase_of(state: State) -> Phase {
        match state {
            State::Thinking => Phase::Reasoning,
            State::Call => Phase::Call,
            State::Idle | State::Header | State::Content => Phase::Content,
        }
    }

    fn add(&mut self, text: &str) -> ReplyStep {
        match self.state {
            State::Header => {
                return ReplyStep {
                    phase: Phase::Content,
                    piece: String::new(),
                };
            }
            State::Thinking => self.reasoning.push_str(text),
            State::Idle | State::Content => self.content.push_str(text),
            State::Call => self
                .call
                .as_mut()
                .expect("the Call state has an open call")
                .push_str(text),
        }
        ReplyStep {
            phase: Self::phase_of(self.state),
            piece: text.to_string(),
        }
    }

    fn marker(&mut self, token: u32) -> Option<ReplyStep> {
        let m = self.marks;
        let quiet = |phase| ReplyStep {
            phase,
            piece: String::new(),
        };
        if token == m.message_model {
            self.held.clear();
            self.state = State::Header;
            return Some(quiet(Phase::Content));
        }
        if token == m.content_thinking {
            self.held.clear();
            self.state = State::Thinking;
            return Some(quiet(Phase::Reasoning));
        }
        if token == m.content_text || token == m.content_tool_error || token == m.content_invoke_tool_text {
            self.held.clear();
            self.state = State::Content;
            return Some(quiet(Phase::Content));
        }
        if token == m.content_invoke_tool_json {
            self.held.clear();
            self.state = State::Call;
            self.call = Some(String::new());
            return Some(quiet(Phase::Call));
        }
        if token == m.end_message {
            self.held.clear();
            let prev = Self::phase_of(self.state);
            if self.state == State::Call {
                let text = self.call.take().expect("the Call state has an open call");
                self.calls.push(match parse_tool_json(&text) {
                    Ok((name, input)) => ReplyCall::Parsed { name, input },
                    Err(error) => ReplyCall::Malformed { text, error },
                });
            }
            self.state = State::Idle;
            return Some(quiet(prev));
        }
        None
    }
}

impl ReplyParser for Inkling2ReplyParser {
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
            return Ok(ReplyStep {
                phase: self.phase(),
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
        Self::phase_of(self.state)
    }

    fn finish(mut self: Box<Self>, _max_tokens: bool) -> ParsedReply {
        let mut tail = Vec::new();
        if !self.held.is_empty() {
            self.held.clear();
            let step = self.add("\u{FFFD}");
            if !step.piece.is_empty() {
                tail.push(step);
            }
        }
        if let Some(text) = self.call.take() {
            self.calls.push(ReplyCall::Malformed {
                text,
                error: "the call did not close with <|end_message|>".to_string(),
            });
        }
        ParsedReply {
            content: self.content,
            reasoning: self.reasoning,
            calls: self.calls,
            ended: self.ended,
            tail,
        }
    }
}
