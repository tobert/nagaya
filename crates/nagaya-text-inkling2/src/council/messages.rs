//! The messages and the tools of a gogi context, in the shape that Inkling 2's chat template reads.

use gogi_proto::context::Tool;
use gogi_proto::message::{Content, Message, Part, Role};
use nagaya_council::family::FamilyError;
use nagaya_jinja::Json;
use serde_json::{Map, Value};

/// What the library writes before the content of a failed tool result. The template has no form for a failure
/// (spec 11.3, "One form in every family").
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

fn pair(k: &str, v: Json) -> (String, Json) {
    (k.to_string(), v)
}

/// The tools as the template reads them: `{type: function, function: {name, description, parameters}}`.
pub(crate) fn tools_json(tools: &[Tool]) -> Vec<Json> {
    tools
        .iter()
        .map(|t| {
            let mut function = vec![pair("name", text(&t.name))];
            if let Some(d) = &t.description {
                function.push(pair("description", text(d)));
            }
            function.push(pair("parameters", object_of(&t.input_schema)));
            Json::Object(vec![
                pair("type", text("function")),
                pair("function", Json::Object(function)),
            ])
        })
        .collect()
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn content_json(m: &Message, index: usize, failed: bool) -> Result<Option<Json>, FamilyError> {
    let prefix = if failed { ERROR_PREFIX } else { "" };
    Ok(match &m.content {
        None if failed => Some(text(ERROR_PREFIX)),
        None => None,
        Some(Content::Text(s)) => Some(Json::Str(format!("{prefix}{s}"))),
        Some(Content::Parts(parts)) if parts.iter().all(|p| matches!(p, Part::Text { .. })) => {
            let joined: String = parts
                .iter()
                .map(|p| match p {
                    Part::Text { text } => text.as_str(),
                    Part::Object { .. } => unreachable!("the guard keeps text parts only"),
                })
                .collect();
            Some(Json::Str(format!("{prefix}{joined}")))
        }
        Some(Content::Parts(parts)) => {
            // engine-abi "The images of Inkling 2", "The parts of a message": each part in order, so each text part
            // is a block of its own, as in the reference.
            let mut list = Vec::with_capacity(parts.len() + 1);
            if failed {
                list.push(Json::Object(vec![
                    pair("type", text("text")),
                    pair("text", text(ERROR_PREFIX)),
                ]));
            }
            for (j, p) in parts.iter().enumerate() {
                list.push(match p {
                    Part::Text { text: t } => Json::Object(vec![pair("type", text("text")), pair("text", text(t))]),
                    Part::Object { mime, .. } => {
                        check_image(m.role, index, j, mime)?;
                        Json::Object(vec![pair("type", text("image"))])
                    }
                });
            }
            Some(Json::Array(list))
        }
    })
}

/// The request member of a part, for an error.
pub(crate) fn part_param(message: usize, part: usize) -> String {
    format!("messages[{message}].content[{part}]")
}

/// Check that an object part is an image of a role that the template reads, or give the client error that names it.
fn check_image(role: Role, message: usize, part: usize, mime: &str) -> Result<(), FamilyError> {
    let param = part_param(message, part);
    if !crate::media::MEDIA_INPUTS.contains(&mime) {
        return Err(FamilyError::client_param(
            param,
            format!(
                "nothing ran: Inkling 2 reads {}, not {mime}. Send one of those.",
                crate::media::MEDIA_INPUTS.join(" and ")
            ),
        ));
    }
    if !matches!(role, Role::User | Role::Tool) {
        return Err(FamilyError::client_param(
            param,
            format!(
                "nothing ran: Inkling 2 reads an image in a user or a tool message, not in a {} message. Move the \
                 image to a user message.",
                role_str(role)
            ),
        ));
    }
    Ok(())
}

/// An image part of a context: its place, its object and its media type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePart {
    /// The index of the message, and the index of the part in its content.
    pub message: usize,
    pub part: usize,
    /// The content id of the object.
    pub id: String,
    pub mime: String,
}

impl ImagePart {
    /// The request member of the part, for an error.
    pub fn param(&self) -> String {
        part_param(self.message, self.part)
    }
}

/// The image parts of the messages, in the order of the rendering. A part that the family cannot read is a client
/// error that names it.
pub(crate) fn image_parts(messages: &[Message]) -> Result<Vec<ImagePart>, FamilyError> {
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let Some(Content::Parts(parts)) = &m.content else {
            continue;
        };
        for (j, p) in parts.iter().enumerate() {
            if let Part::Object { id, mime } = p {
                check_image(m.role, i, j, mime)?;
                out.push(ImagePart {
                    message: i,
                    part: j,
                    id: id.clone(),
                    mime: mime.clone(),
                });
            }
        }
    }
    Ok(out)
}

/// The messages as the template reads them.
pub(crate) fn messages_json(messages: &[Message]) -> Result<Vec<Json>, FamilyError> {
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let failed = m.role == Role::Tool && m.is_error;
            let mut pairs = vec![pair("role", text(role_str(m.role)))];
            if let Some(content) = content_json(m, i, failed)? {
                pairs.push(pair("content", content));
            }
            if let Some(reasoning) = &m.reasoning {
                pairs.push(pair("reasoning_content", text(reasoning)));
            }
            if let Some(id) = &m.tool_call_id {
                pairs.push(pair("tool_call_id", text(id)));
            }
            if !m.tool_calls.is_empty() {
                let calls = m
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        Json::Object(vec![
                            pair("id", text(&tc.id)),
                            pair("type", text("function")),
                            pair(
                                "function",
                                Json::Object(vec![
                                    pair("name", text(&tc.name)),
                                    pair("arguments", object_of(&tc.input)),
                                ]),
                            ),
                        ])
                    })
                    .collect();
                pairs.push(pair("tool_calls", Json::Array(calls)));
            }
            Ok(Json::Object(pairs))
        })
        .collect()
}

/// All client strings in a message, for control-text reporting.
pub(crate) fn message_strings(m: &Message) -> Vec<&str> {
    fn walk_value<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
        match v {
            Value::String(s) => out.push(s.as_str()),
            Value::Array(a) => a.iter().for_each(|x| walk_value(x, out)),
            Value::Object(m) => {
                for (k, val) in m {
                    out.push(k.as_str());
                    walk_value(val, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    match &m.content {
        Some(Content::Text(s)) => out.push(s.as_str()),
        Some(Content::Parts(parts)) => {
            for p in parts {
                if let Part::Text { text } = p {
                    out.push(text.as_str());
                }
            }
        }
        None => {}
    }
    if let Some(r) = &m.reasoning {
        out.push(r.as_str());
    }
    for tc in &m.tool_calls {
        out.push(tc.name.as_str());
        for (k, v) in &tc.input {
            out.push(k.as_str());
            walk_value(v, &mut out);
        }
    }
    out
}
