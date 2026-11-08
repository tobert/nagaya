//! The messages and the tools of a gogi context, in the shape that the LFM3 chat templates read.

use gogi_proto::context::Tool;
use gogi_proto::message::{Content, Message, Part, Role};
use nagaya_council::family::FamilyError;
use nagaya_jinja::Json;
use serde_json::{Map, Value};

/// What the library writes before the content of a failed tool result. The templates have no form for a failure
/// (spec 11.3, "One form in every family").
pub(crate) const ERROR_PREFIX: &str = "Error: ";

/// The kind of a media part, and the placeholder that the template writes for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    /// An image of a message that the model reads: `<|image|>`.
    Image,
    /// A clip of a message that the model reads: `<|audio|>`.
    Audio,
    /// A clip of an `assistant` message, which the model spoke: `<|audio_out|>`.
    AudioOut,
    /// A video of a message that the model reads: `<|video|>`. It expands to its pairs of frames and its sound
    /// (spec 11.11, "The video of a request").
    Video,
}

impl MediaKind {
    /// The `type` of the part, as the template reads it.
    pub(crate) fn part_type(self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Audio => "audio",
            MediaKind::AudioOut => "audio_out",
            MediaKind::Video => "video",
        }
    }

    /// The text of the placeholder token.
    pub fn placeholder(self) -> &'static str {
        match self {
            MediaKind::Image => "<|image|>",
            MediaKind::Audio => "<|audio|>",
            MediaKind::AudioOut => "<|audio_out|>",
            MediaKind::Video => "<|video|>",
        }
    }
}

/// A media part of a context: its place and its kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaPart {
    /// The index of the message.
    pub message: usize,
    /// The index of the part in the message's content.
    pub part: usize,
    pub kind: MediaKind,
    /// The object id of the part.
    pub id: String,
    /// The media type of the part.
    pub mime: String,
}

/// The request member of a part, for an error.
fn part_param(message: usize, part: usize) -> String {
    format!("messages[{message}].content[{part}]")
}

/// The kind of an object part, or the client error that names it. Only Omni reads media.
pub(crate) fn media_kind(
    omni: bool,
    role: Role,
    message: usize,
    part: usize,
    mime: &str,
) -> Result<MediaKind, FamilyError> {
    let param = part_param(message, part);
    if !omni {
        return Err(FamilyError::client_param(
            param,
            "the served model lfm3 reads text only. Send the media part to lfm3-omni.",
        ));
    }
    match (role, mime) {
        (Role::Assistant, "audio/wav") => Ok(MediaKind::AudioOut),
        (Role::Assistant, _) => Err(FamilyError::client_param(
            param,
            format!("an assistant message holds audio/wav only, not {mime}. Remove the part."),
        )),
        (_, "image/png" | "image/jpeg") => Ok(MediaKind::Image),
        (_, "audio/wav") => Ok(MediaKind::Audio),
        (_, "video/mp4" | "video/webm") if cfg!(feature = "video") => Ok(MediaKind::Video),
        (_, "video/mp4" | "video/webm") => Err(FamilyError::client_param(
            param,
            "this build of the server reads no video: it has no feature `video`. Send the frames as images.",
        )),
        _ => Err(FamilyError::client_param(
            param,
            format!("lfm3-omni reads image/png, image/jpeg and audio/wav, not {mime}. Send one of those."),
        )),
    }
}

/// The media parts of the messages, in the order of the rendering.
pub(crate) fn media_parts(omni: bool, messages: &[Message]) -> Result<Vec<MediaPart>, FamilyError> {
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let Some(Content::Parts(parts)) = &m.content else {
            continue;
        };
        for (j, p) in parts.iter().enumerate() {
            if let Part::Object { id, mime } = p {
                out.push(MediaPart {
                    message: i,
                    part: j,
                    kind: media_kind(omni, m.role, i, j, mime)?,
                    id: id.clone(),
                    mime: mime.clone(),
                });
            }
        }
    }
    Ok(out)
}

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

/// The content of a message as the template reads it: a string, or a list of parts when the message holds media.
/// Text parts alone join into one string, as gogi reads a string as one text part.
fn content_json(m: &Message, index: usize, omni: bool, failed: bool) -> Result<Option<Json>, FamilyError> {
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
            let mut list = Vec::new();
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
                        let kind = media_kind(omni, m.role, index, j, mime)?;
                        Json::Object(vec![pair("type", text(kind.part_type()))])
                    }
                });
            }
            Some(Json::Array(list))
        }
    })
}

/// A message as the template reads it. The template merges a run of results of one call by `tool_call_id`.
pub(crate) fn message_json(m: &Message, index: usize, omni: bool) -> Result<Json, FamilyError> {
    let mut o = vec![pair("role", text(m.role.as_str()))];
    let failed = m.is_error && m.role == Role::Tool;
    if let Some(c) = content_json(m, index, omni, failed)? {
        o.push(pair("content", c));
    }
    if let Some(r) = &m.reasoning {
        o.push(pair("reasoning_content", text(r)));
    }
    if !m.tool_calls.is_empty() {
        let calls = m
            .tool_calls
            .iter()
            .map(|c| {
                Json::Object(vec![
                    pair("id", text(&c.id)),
                    pair("type", text("function")),
                    pair(
                        "function",
                        Json::Object(vec![
                            pair("name", text(&c.name)),
                            pair("arguments", object_of(&c.input)),
                        ]),
                    ),
                ])
            })
            .collect();
        o.push(pair("tool_calls", Json::Array(calls)));
    }
    if let Some(id) = &m.tool_call_id {
        o.push(pair("tool_call_id", text(id)));
    }
    Ok(Json::Object(o))
}

/// The messages as the template reads them.
pub(crate) fn messages_json(messages: &[Message], omni: bool) -> Result<Vec<Json>, FamilyError> {
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| message_json(m, i, omni))
        .collect()
}

/// A message with a role and a content.
pub(crate) fn turn_json(role: &str, content: &str) -> Json {
    Json::Object(vec![pair("role", text(role)), pair("content", text(content))])
}

/// Every string of a message that the library renders: its text, its reasoning, and what its calls say.
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
    match &m.content {
        Some(Content::Text(s)) => out.push(s),
        Some(Content::Parts(parts)) => out.extend(parts.iter().filter_map(|p| match p {
            Part::Text { text } => Some(text.as_str()),
            Part::Object { .. } => None,
        })),
        None => {}
    }
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
