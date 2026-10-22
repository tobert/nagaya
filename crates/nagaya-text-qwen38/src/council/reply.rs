//! The reply parser of the Qwen3.8 family: the raw parser of `crate::reply` behind the `ReplyParser` trait (spec 11.3).
//!
//! The caller pushes every sampled token, the stop token too. The stop token ends the reply. It adds no text, and
//! `ParsedReply::ended` is true. A reply that stops for another reason has `ended` false, and the caller says whether
//! `max_tokens` or a full context cut it.

use std::sync::Arc;

use gogi_proto::context::Tool;
use gogi_proto::generation::Phase;
use nagaya_council::family::{FamilyError, ParsedReply, ReplyCall, ReplyParser, ReplyStart, ReplyStep};
use serde_json::{Map, Number, Value};

use super::messages::tools_json;
use super::{Qwen38Family, internal, refusal};
use crate::json::Json;
use crate::reply::{Ending, Keep, Phase as RawPhase, ReplyParser as Raw, TokenEvent, ToolCall};
use crate::{Error, Tokenizer};

/// The end of the thinking. The raw parser releases it as the last text of the reasoning.
const THINK_END: &str = "</think>";

pub(crate) fn reply_parser(
    family: &Qwen38Family,
    start: ReplyStart,
    tools: &[Tool],
) -> Result<Box<dyn ReplyParser>, FamilyError> {
    // A tool list that mk's parser refuses is the client's: the same tools made the context.
    let raw = Raw::new(family.shared_tokenizer(), start.thinking, tools_json(tools).as_ref()).map_err(refusal)?;
    Ok(Box::new(Qwen38ReplyParser {
        raw,
        open: start.open,
        stop: family.stop_ids().to_vec(),
        vocab: family.tokenizer().len(),
        ended: false,
    }))
}

struct Qwen38ReplyParser {
    raw: Raw<Arc<Tokenizer>>,
    /// True when the reply goes on from an open message: the start of its first part stays as it is.
    open: bool,
    stop: Vec<u32>,
    vocab: usize,
    ended: bool,
}

fn phase_of(raw: RawPhase) -> Phase {
    match raw {
        RawPhase::Reasoning => Phase::Reasoning,
        RawPhase::Content => Phase::Content,
        RawPhase::Call => Phase::Call,
    }
}

/// The text of the released pieces that belong to `phase`. The end of the thinking is not part of the reasoning.
fn text_in(released: &[(RawPhase, String)], phase: RawPhase) -> String {
    released
        .iter()
        .filter(|(p, _)| *p == phase)
        .map(|(_, text)| match phase {
            RawPhase::Reasoning => text.strip_suffix(THINK_END).unwrap_or(text),
            _ => text,
        })
        .collect()
}

/// The step of a token. It has one phase, the phase of the last text that the token released. Text that the token
/// released in an earlier phase is in the fields of the whole reply, but in no step.
fn step_of(event: &TokenEvent) -> ReplyStep {
    ReplyStep {
        phase: phase_of(event.phase),
        piece: text_in(&event.released, event.phase),
    }
}

impl ReplyParser for Qwen38ReplyParser {
    fn push(&mut self, token: u32) -> Result<ReplyStep, FamilyError> {
        if self.ended {
            return Err(FamilyError::internal(format!(
                "token {token} came after the stop token: the reply is over"
            )));
        }
        if token as usize >= self.vocab {
            return Err(FamilyError::internal(format!(
                "{token} is not a token: ids run 0 to {}",
                self.vocab - 1
            )));
        }
        if self.stop.contains(&token) {
            self.ended = true;
            return Ok(ReplyStep {
                phase: phase_of(self.raw.phase()),
                piece: String::new(),
            });
        }
        let event = self.raw.push(token).map_err(internal)?;
        Ok(step_of(&event))
    }

    fn ended(&self) -> bool {
        self.ended
    }

    fn phase(&self) -> Phase {
        phase_of(self.raw.phase())
    }

    fn finish(self: Box<Self>, max_tokens: bool) -> ParsedReply {
        let ending = match (self.ended, max_tokens) {
            (true, _) => Ending::StopToken,
            (false, true) => Ending::MaxTokens,
            (false, false) => Ending::ContextFull,
        };
        let ended = self.ended;
        let keep = Keep {
            start: self.open,
            end: ending == Ending::MaxTokens,
        };
        let reply = self.raw.finish_keeping(ending, keep);
        let tail = reply
            .tail
            .iter()
            .map(|(phase, text)| ReplyStep {
                phase: phase_of(*phase),
                piece: text_in(&[(*phase, text.clone())], *phase),
            })
            .collect();
        ParsedReply {
            content: reply.content,
            reasoning: reply.reasoning,
            calls: reply.tool_calls.into_iter().map(call_of).collect(),
            ended,
            tail,
        }
    }
}

/// A call of the council API. The raw parser keeps the members in the order of the text, and so does this.
fn call_of(call: ToolCall) -> ReplyCall {
    match call {
        ToolCall::Parsed { name, input, raw } => match object_of(&input) {
            Ok(input) => ReplyCall::Parsed { name, input },
            // The model wrote a call that parses, but the API cannot carry it. Keep its text and say why.
            Err(error) => ReplyCall::Malformed { text: raw, error },
        },
        ToolCall::Malformed { raw, error, .. } => ReplyCall::Malformed { text: raw, error },
    }
}

fn object_of(input: &Json) -> Result<Map<String, Value>, String> {
    match value_of(input)? {
        Value::Object(map) => Ok(map),
        other => Err(format!("the input of a call must be an object, got {other}")),
    }
}

/// The JSON value of the raw parser as a `serde_json` value. A number that a JSON number of the API cannot hold
/// is an error, so no value changes on the way.
fn value_of(json: &Json) -> Result<Value, String> {
    Ok(match json {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Int(i) => match (i64::try_from(*i), u64::try_from(*i)) {
            (Ok(i), _) => Value::Number(i.into()),
            (_, Ok(u)) => Value::Number(u.into()),
            _ => {
                return Err(format!(
                    "the integer {i} does not fit in 64 bits. The council API carries no larger integer."
                ));
            }
        },
        Json::Float(f) => {
            Value::Number(Number::from_f64(*f).ok_or_else(|| format!("the number {f} is not a finite JSON number"))?)
        }
        Json::Str(s) => Value::String(s.clone()),
        Json::Array(items) => Value::Array(items.iter().map(value_of).collect::<Result<_, _>>()?),
        Json::Object(pairs) => Value::Object(
            pairs
                .iter()
                .map(|(k, v)| Ok((k.clone(), value_of(v)?)))
                .collect::<Result<_, String>>()?,
        ),
    })
}

impl Qwen38Family {
    /// The ids of the tokens that end a reply: `<|im_end|>` and `<|endoftext|>` (mk `STOP_IDS`).
    pub(crate) fn stop_ids(&self) -> &[u32] {
        &self.stop
    }
}

/// The stop tokens of the family, from the vocabulary. A model without them is not a Qwen3.8 model.
pub(super) fn stop_tokens(tokenizer: &Tokenizer) -> Result<Vec<u32>, Error> {
    ["<|im_end|>", "<|endoftext|>"]
        .into_iter()
        .map(|text| {
            tokenizer.token_id(text).ok_or_else(|| {
                Error::Vocab(format!(
                    "the vocabulary has no token {text}. Use a GGUF made for this family."
                ))
            })
        })
        .collect()
}
