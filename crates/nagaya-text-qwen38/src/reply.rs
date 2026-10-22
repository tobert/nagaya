//! The reply parser: the tokens of a reply to its phases, its parts and its tool calls.
//!
//! This is a port of `ReplyTracker` in mk's `service/chatfmt.py`, with the token loop of `Generation._loop` in mk's
//! `service/generate.py`. The format is the format of the Qwen3.8 chat template:
//!
//! - With thinking on, the text of the reply starts inside the thinking, and `</think>` ends it. With thinking off,
//!   the text starts in the content.
//! - A tool call is `<tool_call>`, `<function=NAME>`, some `<parameter=NAME>` blocks, `</function>` and
//!   `</tool_call>`. A parameter value is the raw text between the tags. The declared type of the parameter says
//!   whether the value is a string or JSON.
//!
//! The parser makes no call ids. The caller names each call (council API, "Generations").

use crate::json::Json;
use crate::pystr::{repr, strip};
use crate::{Error, Tokenizer, chars, detok::Utf8Stream};
use nagaya_jinja::truthy;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;

const THINK_END: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FUNC_CLOSE: &str = "</function>";
const PARAM_CLOSE: &str = "\n</parameter>";

/// The part of the reply that a piece of text belongs to. mk calls these `think`, `answer` and `tool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Reasoning,
    Content,
    Call,
}

/// How the sampling of the reply stopped. The caller knows this. The parser does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// The model sampled a stop token. Do not push the stop token.
    StopToken,
    /// The reply reached `max_tokens`.
    MaxTokens,
    /// The prompt and the reply filled the context.
    ContextFull,
}

/// Why the reply ended, in the words of the council API (`finish`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn with no tool call.
    EndTurn,
    /// The model ended its turn after one or more tool calls. A call that does not parse counts.
    ToolUse,
    MaxTokens,
    ContextFull,
}

/// The ends of the reply that `ReplyParser::finish_keeping` keeps as they are. mk trims both ends of each part. gogi
/// keeps two ends (gogi "Continuations").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Keep {
    /// Keep the white space at the start of the first part. A continuation joins its open text and the new text,
    /// and the server trims nothing at the seam.
    pub start: bool,
    /// Keep the white space at the end of the part that the reply is in when it stops. A reply that ends with
    /// `max_tokens` keeps it, so the client can send it back exactly.
    pub end: bool,
}

/// A tool call of the reply. The caller adds the id.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCall {
    /// A call that parsed. `input` is a `Json::Object` with the members in the order of the text. `raw` is the call
    /// as written.
    Parsed { name: String, input: Json, raw: String },
    /// A call that did not parse, as written. `name` is `None` when the text has no readable function tag.
    Malformed {
        name: Option<String>,
        raw: String,
        error: String,
    },
}

/// What one token gives.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenEvent {
    /// The phase of the last text that the tracker released for this token. When it released none, the phase now.
    pub phase: Phase,
    /// The text of this token: the bytes of the token, once the character is complete. It is empty while a
    /// character is not complete. It includes marker text that the tracker holds back or releases.
    pub piece: String,
    /// What mk's `ReplyTracker.feed` released for this token. The text of a marker that is not complete stays out.
    /// The end of thinking, `</think>`, is the last text of the reasoning.
    pub released: Vec<(Phase, String)>,
}

/// The whole reply.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    /// The thinking, without white space at either end. Thinking that never ended is all reasoning.
    pub reasoning: String,
    /// The content before the first tool call, without white space at either end.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish: Finish,
    /// The text that the reply released after its last token: what the last character released, and the text that
    /// the tracker held back as a possible marker. mk's `ReplyTracker.tail` has only the second part.
    pub tail: Vec<(Phase, String)>,
}

/// The parser of one reply. Push the token ids in order, then call `finish`.
#[derive(Debug)]
pub struct ReplyParser<T> {
    tokenizer: T,
    utf8: Utf8Stream,
    tracker: Tracker,
}

impl<T: Deref<Target = Tokenizer>> ReplyParser<T> {
    /// Start a reply. `tokenizer` is a reference to the tokenizer, or a shared handle that outlives the parser.
    /// `thinking` is true when the prompt ends inside the thinking. `tools` is the list of tools of the request, in
    /// the form of the chat API. With no tools or an empty list, `<tool_call>` is plain text.
    pub fn new(tokenizer: T, thinking: bool, tools: Option<&Json>) -> Result<ReplyParser<T>, Error> {
        let specs = match tools.filter(|t| truthy(t)) {
            Some(tools) => Some(ToolSpecs::from_tools(tools)?),
            None => None,
        };
        Ok(ReplyParser {
            tokenizer,
            utf8: Utf8Stream::new(),
            tracker: Tracker::new(thinking, specs),
        })
    }

    /// The phase now, after everything pushed. Text held back as a possible marker counts for this phase.
    pub fn phase(&self) -> Phase {
        self.tracker.phase
    }

    /// Add the next token of the reply. A stop token is not part of the reply: do not push it.
    pub fn push(&mut self, id: u32) -> Result<TokenEvent, Error> {
        let bytes = self.tokenizer.token_piece(id, false)?;
        let piece = self.utf8.push(&bytes);
        let released = self.tracker.feed(&piece);
        let phase = released.last().map_or(self.tracker.phase, |s| s.0);
        Ok(TokenEvent { phase, piece, released })
    }

    /// End the reply. Bytes of a character that is not complete become U+FFFD.
    pub fn finish(self, ending: Ending) -> Reply {
        self.finish_keeping(ending, Keep::default())
    }

    /// End the reply as `finish` does, but keep the ends that `keep` names.
    pub fn finish_keeping(mut self, ending: Ending, keep: Keep) -> Reply {
        let flush = self.utf8.flush();
        let mut tail = if flush.is_empty() {
            Vec::new()
        } else {
            self.tracker.feed(&flush)
        };
        let (reasoning, content, tool_calls, held) = self.tracker.finish(keep);
        tail.extend(held);
        let finish = match ending {
            Ending::StopToken if tool_calls.is_empty() => Finish::EndTurn,
            Ending::StopToken => Finish::ToolUse,
            Ending::MaxTokens => Finish::MaxTokens,
            Ending::ContextFull => Finish::ContextFull,
        };
        Reply {
            reasoning,
            content,
            tool_calls,
            finish,
            tail,
        }
    }
}

// ---------------------------------------------------------------- tool schemas

/// A JSON schema type. The order is the alphabetical order of the names, as mk sorts them in its messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Ty {
    Array,
    Boolean,
    Integer,
    Null,
    Number,
    Object,
    String,
}

const ALL_TYPES: [Ty; 7] = [
    Ty::Array,
    Ty::Boolean,
    Ty::Integer,
    Ty::Null,
    Ty::Number,
    Ty::Object,
    Ty::String,
];

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::Array => "array",
            Ty::Boolean => "boolean",
            Ty::Integer => "integer",
            Ty::Null => "null",
            Ty::Number => "number",
            Ty::Object => "object",
            Ty::String => "string",
        }
    }

    fn from_name(name: &str) -> Option<Ty> {
        ALL_TYPES.into_iter().find(|t| t.name() == name)
    }
}

type Types = BTreeSet<Ty>;

#[derive(Debug)]
struct Spec {
    props: BTreeMap<String, Types>,
    required: Vec<String>,
}

/// What the reply parser needs from the tools of a request: the parameters, their types, and the required ones.
#[derive(Debug)]
struct ToolSpecs(BTreeMap<String, Spec>);

fn tools_error(message: String) -> Error {
    Error::Tools(message)
}

fn member<'a>(object: &'a Json, key: &str) -> Option<&'a Json> {
    match object {
        Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

/// `_types_of` of mk. A schema that is not an object, or that names no type, allows every type.
fn types_of(schema: &Json) -> Result<Types, Error> {
    if !matches!(schema, Json::Object(_)) {
        return Ok(ALL_TYPES.into_iter().collect());
    }
    let mut names: Vec<&str> = Vec::new();
    match member(schema, "type") {
        Some(Json::Str(s)) => names.push(s),
        Some(Json::Array(a)) => {
            for item in a {
                match item {
                    Json::Str(s) => names.push(s),
                    other => {
                        return Err(tools_error(format!(
                            "a schema type in a list must be a string, got {other:?}. Write the type as text."
                        )));
                    }
                }
            }
        }
        _ => {}
    }
    let mut found = Types::new();
    let mut unknown: Vec<&str> = Vec::new();
    for name in names {
        match Ty::from_name(name) {
            Some(t) => {
                found.insert(t);
            }
            None => unknown.push(name),
        }
    }
    for key in ["anyOf", "oneOf"] {
        match member(schema, key) {
            Some(Json::Array(subs)) => {
                for sub in subs {
                    found.extend(types_of(sub)?);
                }
            }
            Some(other) if truthy(other) => {
                return Err(tools_error(format!(
                    "{key} must be an array of schemas, got {other:?}."
                )));
            }
            _ => {}
        }
    }
    if found.is_empty() && unknown.is_empty() {
        return Ok(ALL_TYPES.into_iter().collect());
    }
    if !unknown.is_empty() {
        unknown.sort_unstable();
        unknown.dedup();
        return Err(tools_error(format!(
            "unsupported JSON schema type(s) {}",
            repr_list(unknown.into_iter())
        )));
    }
    Ok(found)
}

impl ToolSpecs {
    /// `_specs` of mk. A tool is `{"function": {...}}` or the function itself.
    fn from_tools(tools: &Json) -> Result<ToolSpecs, Error> {
        let Json::Array(tools) = tools else {
            return Err(tools_error(format!(
                "the tools must be an array, got {tools:?}. Send a list of tools."
            )));
        };
        let mut out = BTreeMap::new();
        for tool in tools {
            let function = member(tool, "function").unwrap_or(tool);
            let name = match member(function, "name") {
                Some(Json::Str(name)) => name,
                _ => {
                    return Err(tools_error(format!(
                        "a tool needs a function with a name, got {tool:?}"
                    )));
                }
            };
            let empty = Json::Object(Vec::new());
            let params = member(function, "parameters").filter(|p| truthy(p)).unwrap_or(&empty);
            if !matches!(params, Json::Object(_)) {
                return Err(tools_error(format!(
                    "tool {}: parameters must be an object, got {params:?}",
                    repr(name)
                )));
            }
            let mut props = BTreeMap::new();
            match member(params, "properties").filter(|p| truthy(p)) {
                None => {}
                Some(Json::Object(pairs)) => {
                    for (key, schema) in pairs {
                        props.insert(key.clone(), types_of(schema)?);
                    }
                }
                Some(_) => {
                    return Err(tools_error(format!(
                        "tool {}: parameters.properties must be an object",
                        repr(name)
                    )));
                }
            }
            let required = match member(params, "required").filter(|r| truthy(r)) {
                None => Vec::new(),
                Some(Json::Array(items)) => items
                    .iter()
                    .map(|item| match item {
                        Json::Str(s) => Ok(s.clone()),
                        other => Err(tools_error(format!(
                            "tool {}: each name in required must be a string, got {other:?}",
                            repr(name)
                        ))),
                    })
                    .collect::<Result<_, _>>()?,
                Some(other) => {
                    return Err(tools_error(format!(
                        "tool {}: required must be an array of names, got {other:?}",
                        repr(name)
                    )));
                }
            };
            out.insert(name.clone(), Spec { props, required });
        }
        Ok(ToolSpecs(out))
    }
}

/// Python's `repr` of a list of strings.
fn repr_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    format!("[{}]", items.map(repr).collect::<Vec<_>>().join(", "))
}

fn types_text(types: &Types) -> String {
    repr_list(types.iter().map(|t| t.name()))
}

// ---------------------------------------------------------------- values

fn python_type_name(v: &Json) -> &'static str {
    match v {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Float(_) => "float",
        Json::Str(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

fn is_type(v: &Json, t: Ty) -> bool {
    match t {
        Ty::Integer => matches!(v, Json::Int(_)),
        Ty::Number => matches!(v, Json::Int(_) | Json::Float(_)),
        Ty::Boolean => matches!(v, Json::Bool(_)),
        Ty::Null => matches!(v, Json::Null),
        Ty::Object => matches!(v, Json::Object(_)),
        Ty::Array => matches!(v, Json::Array(_)),
        Ty::String => false,
    }
}

/// `_convert` of mk. The error text names the JSON error in the words of this library, not of Python.
fn convert(value: &str, types: &Types) -> Result<Json, String> {
    let only_string = types.len() == 1 && types.contains(&Ty::String);
    if only_string {
        return Ok(Json::Str(value.to_string()));
    }
    let v = match Json::parse(value) {
        Ok(v) => v,
        Err(e) => {
            if types.contains(&Ty::String) {
                return Ok(Json::Str(value.to_string()));
            }
            let why = match Error::from(e) {
                Error::Json(m) => m,
                other => other.to_string(),
            };
            return Err(format!("not JSON ({why})"));
        }
    };
    if types.iter().any(|&t| t != Ty::String && is_type(&v, t)) {
        return Ok(v);
    }
    if types.contains(&Ty::String) {
        return Ok(Json::Str(value.to_string()));
    }
    Err(format!(
        "JSON {} {} is not one of {}",
        python_type_name(&v),
        repr(value),
        types_text(types)
    ))
}

// ---------------------------------------------------------------- tool calls

struct Malformed {
    why: String,
    name: Option<String>,
}

fn skip_space(text: &str, pos: usize) -> usize {
    pos + text[pos..].len() - text[pos..].trim_start_matches(chars::is_python_space).len()
}

/// The tag `<{word}NAME>` and a newline at `pos`. NAME has no `>` and no newline, and it is not empty.
fn named_tag<'a>(text: &'a str, pos: usize, open: &str) -> Option<(&'a str, usize)> {
    let rest = text[pos..].strip_prefix(open)?;
    let end = rest.find(['>', '\n'])?;
    if end == 0 || !rest[end..].starts_with(">\n") {
        return None;
    }
    Some((&rest[..end], pos + open.len() + end + 2))
}

/// `_parse_one` of mk. The call starts at `pos` with `<tool_call>`. A call that cannot be read runs to the next
/// `<tool_call>`, or to the end, so that the rest of the reply is still read. Returns the end of the call.
fn parse_one(text: &str, pos: usize, specs: &ToolSpecs) -> (usize, ToolCall) {
    let (end, name, params) = match read_call(text, pos) {
        Ok(read) => read,
        Err(m) => {
            let end = text[pos + 1..].find(CALL_OPEN).map_or(text.len(), |at| pos + 1 + at);
            let call = ToolCall::Malformed {
                name: m.name,
                raw: text[pos..end].to_string(),
                error: m.why,
            };
            return (end, call);
        }
    };
    let raw = &text[pos..end];
    match check_call(&name, &params, specs) {
        Ok(input) => (
            end,
            ToolCall::Parsed {
                name,
                input,
                raw: raw.to_string(),
            },
        ),
        Err(error) => (
            end,
            ToolCall::Malformed {
                name: Some(name),
                raw: raw.to_string(),
                error,
            },
        ),
    }
}

type Params<'a> = Vec<(&'a str, &'a str)>;

/// The tags of the call: its name and its parameters as written. Returns the end of the call.
fn read_call(text: &str, pos: usize) -> Result<(usize, String, Params<'_>), Malformed> {
    let open_end = skip_space(text, pos + CALL_OPEN.len());
    let Some((name, mut i)) = named_tag(text, open_end, "<function=") else {
        return Err(Malformed {
            why: "expected <function=NAME> after <tool_call>".to_string(),
            name: None,
        });
    };
    let bad = |why: String| Malformed {
        why,
        name: Some(name.to_string()),
    };
    let mut params: Params = Vec::new();
    loop {
        i = skip_space(text, i);
        if text[i..].starts_with(FUNC_CLOSE) {
            i += FUNC_CLOSE.len();
            break;
        }
        let Some((pname, value_start)) = named_tag(text, i, "<parameter=") else {
            return Err(bad("expected <parameter=NAME> or </function>".to_string()));
        };
        let mut j = value_start;
        let (value_end, after) = loop {
            let Some(at) = text[j..].find(PARAM_CLOSE) else {
                return Err(bad(format!(
                    "parameter {} is not closed (no </parameter>)",
                    repr(pname)
                )));
            };
            let value_end = j + at;
            let after = skip_space(text, value_end + PARAM_CLOSE.len());
            if text[after..].starts_with("<parameter=") || text[after..].starts_with(FUNC_CLOSE) {
                break (value_end, after);
            }
            // The "</parameter>" was part of the value.
            j = value_end + 1;
        };
        params.push((pname, &text[value_start..value_end]));
        i = after;
    }
    i = skip_space(text, i);
    if !text[i..].starts_with(CALL_CLOSE) {
        return Err(bad("tool call is not closed (no </tool_call>)".to_string()));
    }
    Ok((i + CALL_CLOSE.len(), name.to_string(), params))
}

/// The checks of mk after a call is read: the function exists, each parameter is declared once and has the right
/// type, and the required parameters are there. Returns the input as a `Json::Object` in the order of the text.
fn check_call(name: &str, params: &Params, specs: &ToolSpecs) -> Result<Json, String> {
    let Some(spec) = specs.0.get(name) else {
        return Err(format!(
            "unknown function {} (tools: {})",
            repr(name),
            repr_list(specs.0.keys().map(String::as_str))
        ));
    };
    let mut arguments: Vec<(String, Json)> = Vec::new();
    for &(pname, value) in params {
        if arguments.iter().any(|(k, _)| k == pname) {
            return Err(format!("parameter {} given twice", repr(pname)));
        }
        let Some(types) = spec.props.get(pname) else {
            return Err(format!(
                "parameter {} is not declared by {} (declared: {})",
                repr(pname),
                repr(name),
                repr_list(spec.props.keys().map(String::as_str))
            ));
        };
        let json = convert(value, types).map_err(|e| {
            let names: Vec<&str> = types.iter().map(|t| t.name()).collect();
            format!("parameter {} ({}): {e}", repr(pname), names.join("|"))
        })?;
        arguments.push((pname.to_string(), json));
    }
    let missing: Vec<&str> = spec
        .required
        .iter()
        .map(String::as_str)
        .filter(|r| !arguments.iter().any(|(k, _)| k == r))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "missing required parameter(s) {} of {}",
            repr_list(missing.into_iter()),
            repr(name)
        ));
    }
    Ok(Json::Object(arguments))
}

/// `_parse_calls` of mk: the text from the first `<tool_call>` to the end. Text between calls is an error entry.
fn parse_calls(text: &str, specs: &ToolSpecs) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut pos = 0;
    loop {
        pos = skip_space(text, pos);
        if pos >= text.len() {
            return calls;
        }
        if text[pos..].starts_with(CALL_OPEN) {
            let (end, call) = parse_one(text, pos, specs);
            calls.push(call);
            pos = end;
        } else {
            let end = text[pos..].find(CALL_OPEN).map_or(text.len(), |at| pos + at);
            calls.push(ToolCall::Malformed {
                name: None,
                raw: text[pos..end].to_string(),
                error: "text outside a tool call".to_string(),
            });
            pos = end;
        }
    }
}

// ---------------------------------------------------------------- streaming

/// The length of the longest proper prefix of `marker` that `buf` ends with.
fn marker_prefix_len(buf: &str, marker: &str) -> usize {
    (1..marker.len().min(buf.len() + 1))
        .rev()
        .find(|&k| buf.ends_with(&marker[..k]))
        .unwrap_or(0)
}

/// Join neighbors that have the same phase.
fn coalesce(pieces: Vec<(Phase, String)>) -> Vec<(Phase, String)> {
    let mut out: Vec<(Phase, String)> = Vec::new();
    for (phase, text) in pieces {
        match out.last_mut() {
            Some(last) if last.0 == phase => last.1.push_str(&text),
            _ => out.push((phase, text)),
        }
    }
    out
}

/// `ReplyTracker` of mk, without the prefill. It holds back only the text that could still become a marker:
/// `</think>` while thinking, and `<tool_call>` in the content when the request has tools.
#[derive(Debug)]
struct Tracker {
    thinking: bool,
    specs: Option<ToolSpecs>,
    phase: Phase,
    /// The text that could still become a marker.
    held: String,
    /// All the text fed.
    text: String,
}

impl Tracker {
    fn new(thinking: bool, specs: Option<ToolSpecs>) -> Tracker {
        Tracker {
            thinking,
            specs,
            phase: if thinking { Phase::Reasoning } else { Phase::Content },
            held: String::new(),
            text: String::new(),
        }
    }

    fn marker(&self) -> Option<&'static str> {
        match self.phase {
            Phase::Reasoning => Some(THINK_END),
            Phase::Content if self.specs.is_some() => Some(CALL_OPEN),
            _ => None,
        }
    }

    fn feed(&mut self, text: &str) -> Vec<(Phase, String)> {
        self.text.push_str(text);
        self.held.push_str(text);
        let mut out: Vec<(Phase, String)> = Vec::new();
        while !self.held.is_empty() {
            let Some(marker) = self.marker() else {
                out.push((self.phase, std::mem::take(&mut self.held)));
                break;
            };
            if let Some(i) = self.held.find(marker) {
                if i > 0 {
                    out.push((self.phase, self.held[..i].to_string()));
                }
                if self.phase == Phase::Reasoning {
                    // The end marker is the last of the thinking.
                    out.push((Phase::Reasoning, marker.to_string()));
                    self.held = self.held[i + marker.len()..].to_string();
                    self.phase = Phase::Content;
                } else {
                    // The start marker is the first of the call text.
                    self.held = self.held[i..].to_string();
                    self.phase = Phase::Call;
                }
                continue;
            }
            let hold = marker_prefix_len(&self.held, marker);
            let emit = self.held.len() - hold;
            if emit > 0 {
                out.push((self.phase, self.held[..emit].to_string()));
            }
            self.held = self.held[emit..].to_string();
            break;
        }
        coalesce(out)
    }

    /// The parts of the reply, and the text that was held back. Each part loses the white space at its ends, but for
    /// the ends that `keep` names.
    fn finish(self, keep: Keep) -> (String, String, Vec<ToolCall>, Vec<(Phase, String)>) {
        let phase = self.phase;
        let held = if self.held.is_empty() {
            Vec::new()
        } else {
            vec![(self.phase, self.held)]
        };
        // `first` is true for the part that the reply started in, and `last` for the part it stopped in.
        let trim = |s: &str, first: bool, last: bool| strip(s, None, !(first && keep.start), !(last && keep.end));
        let (reasoning, rest) = if self.thinking {
            let Some(i) = self.text.find(THINK_END) else {
                return (trim(&self.text, true, true), String::new(), Vec::new(), held);
            };
            (trim(&self.text[..i], true, false), &self.text[i + THINK_END.len()..])
        } else {
            (String::new(), self.text.as_str())
        };
        let mut content = rest;
        let mut calls = Vec::new();
        if let Some(specs) = &self.specs
            && let Some(c) = rest.find(CALL_OPEN)
        {
            content = &rest[..c];
            calls = parse_calls(&rest[c..], specs);
        }
        (
            reasoning,
            trim(content, !self.thinking, phase == Phase::Content),
            calls,
            held,
        )
    }
}
