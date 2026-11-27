//! The messages and the tools of a gogi context, in the shape that Google's template reads.

use gogi_proto::context::Tool;
use gogi_proto::message::{Message, Role};
use nagaya_jinja::Json;
use serde_json::{Map, Value};

/// What the library writes before the content of a failed tool result. The first Gemma 4 engine gave the template the
/// map
/// `{"error": content}`, and the template refuses a map in Python's jinja2 (spec 11.3, "A failed tool result").
pub(crate) const ERROR_PREFIX: &str = "Error: ";

/// A `serde_json` value in the interpreter's JSON type, with the order of the members kept.
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
pub(crate) fn tools_json(tools: &[Tool]) -> Vec<Json> {
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
        .collect()
}

/// A message as the template reads it. The template matches a result to its call by `tool_call_id`.
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
                    ("id".to_string(), text(&c.id)),
                    (
                        "function".to_string(),
                        Json::Object(vec![
                            ("name".to_string(), text(&c.name)),
                            ("arguments".to_string(), object_of(&c.input)),
                        ]),
                    ),
                ])
            })
            .collect();
        o.push(("tool_calls".to_string(), Json::Array(calls)));
    }
    if let Some(id) = &m.tool_call_id {
        o.push(("tool_call_id".to_string(), text(id)));
    }
    Json::Object(o)
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
