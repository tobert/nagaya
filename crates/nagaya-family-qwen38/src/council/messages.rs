//! The messages and the tools of a gogi context, in the shape that the Qwen3.8 chat template reads.

use gogi_proto::context::Tool;
use gogi_proto::message::{Message, Role};
use serde_json::{Map, Value};

use crate::json::Json;

/// What the library writes before the content of a failed tool result. The template has no form for a failure.
pub(crate) const ERROR_PREFIX: &str = "Error: ";

/// A `serde_json` value in the library's own JSON type, with the order of the members kept.
pub(crate) fn json_of(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Json::Int(i128::from(i)),
            (None, Some(u)) => Json::Int(i128::from(u)),
            (None, None) => Json::Float(n.as_f64().expect("a number that is not an integer is a float")),
        },
        Value::String(s) => Json::Str(s.clone()),
        Value::Array(items) => Json::Array(items.iter().map(json_of).collect()),
        Value::Object(map) => object_of(map),
    }
}

pub(crate) fn object_of(map: &Map<String, Value>) -> Json {
    Json::Object(map.iter().map(|(k, v)| (k.clone(), json_of(v))).collect())
}

fn text(s: &str) -> Json {
    Json::Str(s.to_string())
}

/// The tools as the template reads them: `{type: function, function: {name, description, parameters}}`.
/// A context with no tools gives `None`.
pub(crate) fn tools_json(tools: &[Tool]) -> Option<Json> {
    if tools.is_empty() {
        return None;
    }
    Some(Json::Array(
        tools
            .iter()
            .map(|t| {
                let mut function = vec![("name".to_string(), text(&t.name))];
                if let Some(d) = &t.description {
                    function.push(("description".to_string(), text(d)));
                }
                function.push(("parameters".to_string(), object_of(&t.input_schema)));
                Json::Object(vec![
                    ("type".to_string(), text("function")),
                    ("function".to_string(), Json::Object(function)),
                ])
            })
            .collect(),
    ))
}

/// A message as the template reads it. The template calls the thinking of a message `reasoning_content`, and it
/// reads the input of a call as `arguments`. The template matches a result to its call by order, so no id is sent.
pub(crate) fn message_json(m: &Message) -> Json {
    let mut o = vec![("role".to_string(), text(m.role.as_str()))];
    let failed = m.is_error && m.role == Role::Tool;
    match (nagaya_council::family::content_text(m), failed) {
        (Some(c), true) => o.push(("content".to_string(), Json::Str(format!("{ERROR_PREFIX}{c}")))),
        (None, true) => o.push(("content".to_string(), text(ERROR_PREFIX))),
        (Some(c), false) => o.push(("content".to_string(), text(&c))),
        (None, false) => {}
    }
    if let Some(r) = &m.reasoning {
        o.push(("reasoning_content".to_string(), text(r)));
    }
    if !m.tool_calls.is_empty() {
        let calls = m
            .tool_calls
            .iter()
            .map(|c| {
                Json::Object(vec![
                    ("name".to_string(), text(&c.name)),
                    ("arguments".to_string(), object_of(&c.input)),
                ])
            })
            .collect();
        o.push(("tool_calls".to_string(), Json::Array(calls)));
    }
    Json::Object(o)
}

/// A message with a role and a content.
pub(crate) fn turn_json(role: &str, content: &str) -> Json {
    Json::Object(vec![
        ("role".to_string(), text(role)),
        ("content".to_string(), text(content)),
    ])
}

/// Every string of a message that the library renders: its content, its reasoning, and what its calls say.
pub(crate) fn message_strings(m: &Message) -> Vec<&str> {
    fn walk<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
        match v {
            Value::String(s) => out.push(s),
            Value::Array(items) => items.iter().for_each(|i| walk(i, out)),
            Value::Object(map) => map.iter().for_each(|(k, v)| {
                out.push(k);
                walk(v, out);
            }),
            _ => {}
        }
    }
    let mut out: Vec<&str> = Vec::new();
    out.extend(nagaya_council::family::text_pieces(m));
    out.extend(m.reasoning.as_deref());
    for call in &m.tool_calls {
        out.push(&call.name);
        for (k, v) in &call.input {
            out.push(k);
            walk(v, &mut out);
        }
    }
    out
}

fn role_of(message: &Json) -> Option<&str> {
    match message {
        Json::Object(o) => match o.iter().find(|(k, _)| k == "role") {
            Some((_, Json::Str(role))) => Some(role),
            _ => None,
        },
        _ => None,
    }
}

/// The text content of a message, or an empty string when it has none.
pub(crate) fn content_of(message: &Json) -> &str {
    match message {
        Json::Object(o) => match o.iter().find(|(k, _)| k == "content") {
            Some((_, Json::Str(content))) => content,
            _ => "",
        },
        _ => "",
    }
}

/// The indexes of the system messages that follow a message of another role. The template merges the system
/// messages at the start, and it refuses these.
pub(crate) fn later_system_messages(messages: &[Json]) -> Vec<usize> {
    let leading = messages.iter().take_while(|m| role_of(m) == Some("system")).count();
    (leading..messages.len())
        .filter(|&i| role_of(&messages[i]) == Some("system"))
        .collect()
}

fn mentions(json: &Json, needle: &str) -> bool {
    match json {
        Json::Str(s) => s.contains(needle),
        Json::Array(items) => items.iter().any(|i| mentions(i, needle)),
        Json::Object(o) => o.iter().any(|(k, v)| k.contains(needle) || mentions(v, needle)),
        _ => false,
    }
}

/// The start of the stand-in text of a later system message. No string of the messages or the tools holds it.
pub(crate) fn stand_in_prefix(messages: &[Json], tools: Option<&Json>) -> String {
    let mut prefix = "nagaya-system-turn-".to_string();
    while messages.iter().chain(tools).any(|j| mentions(j, &prefix)) {
        prefix.insert(0, 'x');
    }
    prefix
}
