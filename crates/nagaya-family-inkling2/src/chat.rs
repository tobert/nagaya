//! Inkling 2 chat rendering: the model's `chat_template.jinja` in the nagaya's interpreter, with the escapes of
//! `crate::tokenizer`.
//!
//! Every string that a client sends is content. Before the render, each marker text in it becomes a sentinel. After
//! the render, each sentinel tokenizes as the plain text it stands for. Thus the template's own markers are the only
//! marker tokens (spec 11.3, "Escapes").
//!
//! The template writes no start token and no generation prompt; the model writes `<|message_model|>` as its first
//! token. It writes `Thinking effort level: <n>` from `reasoning_effort` (default 0.9).

use std::sync::Arc;

use nagaya_jinja::{Json, Template};

use crate::Error;
use crate::tokenizer::Tokenizer;

/// The members that the template compares with fixed words, such as `assistant` or `image`. They are not escaped. The
/// library writes them, not the client.
const NAMING_KEYS: [&str; 2] = ["role", "type"];

/// The reasoning effort level passed to the Inkling 2 chat template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Value(f64),
}

impl Effort {
    fn to_json(self) -> Json {
        match self {
            Effort::None => Json::Str("none".to_string()),
            Effort::Minimal => Json::Str("minimal".to_string()),
            Effort::Low => Json::Str("low".to_string()),
            Effort::Medium => Json::Str("medium".to_string()),
            Effort::High => Json::Str("high".to_string()),
            Effort::XHigh => Json::Str("xhigh".to_string()),
            Effort::Max => Json::Str("max".to_string()),
            Effort::Value(v) => Json::Float(v),
        }
    }
}

/// The template variables of a render, other than the messages and the tools.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Options {
    /// `reasoning_effort`. When `None`, the template uses its default (0.9).
    pub effort: Option<Effort>,
}

/// A rendered chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    /// The text with the client's marker text as sentinels.
    pub escaped: String,
    /// The text, with the marker text of the content back in place.
    pub text: String,
    /// The token ids. Only the markers of the template are marker tokens.
    pub tokens: Vec<u32>,
}

/// A model's chat template, ready to render, with the tokenizer of the same folder.
pub struct Chat {
    template: Template,
    source: String,
    tokenizer: Arc<Tokenizer>,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Chat")
    }
}

impl Chat {
    /// Compile the template `source` for `tokenizer`.
    pub fn new(source: &str, tokenizer: Arc<Tokenizer>) -> Result<Chat, Error> {
        for marker in ["<|message_model|>", "<|content_model_end_sampling|>", "<|end_message|>"] {
            if tokenizer.token_id(marker).is_none() {
                return Err(Error::Model(format!("the vocabulary has no {marker}")));
            }
        }
        Ok(Chat {
            template: Template::from_source(source)?,
            source: source.to_string(),
            tokenizer,
        })
    }

    /// The template text, as the model folder holds it. The identity's `template` is this text (spec 12.1).
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// The same JSON with every string escaped, keys too.
    fn escape_json(&self, j: &Json) -> Result<Json, Error> {
        Ok(match j {
            Json::Str(s) => Json::Str(self.tokenizer.escape(s)?),
            Json::Array(a) => Json::Array(a.iter().map(|x| self.escape_json(x)).collect::<Result<_, _>>()?),
            Json::Object(o) => Json::Object(
                o.iter()
                    .map(|(k, v)| {
                        let v = if NAMING_KEYS.contains(&k.as_str()) {
                            check_naming(v)?;
                            v.clone()
                        } else {
                            self.escape_json(v)?
                        };
                        Ok((self.tokenizer.escape(k)?, v))
                    })
                    .collect::<Result<_, Error>>()?,
            ),
            other => other.clone(),
        })
    }

    /// Render template-shaped `messages` and `tools`, with the escapes and the default effort.
    pub fn render(&self, messages: &[Json], tools: &[Json]) -> Result<Rendered, Error> {
        self.render_with(messages, tools, Options::default())
    }

    /// Render as [`Chat::render`] does, with explicit template [`Options`].
    pub fn render_with(&self, messages: &[Json], tools: &[Json], options: Options) -> Result<Rendered, Error> {
        let tools: Vec<Json> = tools.iter().map(|t| self.escape_json(t)).collect::<Result<_, _>>()?;
        let messages: Vec<Json> = messages.iter().map(|m| self.escape_json(m)).collect::<Result<_, _>>()?;
        let mut variables = vec![("messages", Json::Array(messages))];
        if !tools.is_empty() {
            variables.push(("tools", Json::Array(tools)));
        }
        if let Some(effort) = options.effort {
            variables.push(("reasoning_effort", effort.to_json()));
        }
        let escaped = self.template.render(&variables)?;
        self.finish(escaped)
    }

    /// The text and the tokens of escaped text.
    pub fn finish(&self, escaped: String) -> Result<Rendered, Error> {
        let tokens = self.tokenizer.encode_escaped(&escaped)?;
        let text = self.tokenizer.unescape(&escaped)?;
        Ok(Rendered { escaped, text, tokens })
    }
}

/// A naming member is not escaped, so it must hold no noncharacter: one could forge a sentinel.
fn check_naming(v: &Json) -> Result<(), Error> {
    match v {
        Json::Str(s) => crate::tokenizer::check_text(s),
        _ => Ok(()),
    }
}
