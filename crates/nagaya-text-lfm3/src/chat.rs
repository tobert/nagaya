//! LFM3 chat rendering: the model's `chat_template.jinja` in the nagaya's interpreter, with the escapes of
//! `crate::tokenizer`.
//!
//! Every string that a client sends is content. Before the render, each marker text in it becomes a sentinel. After
//! the render, each sentinel tokenizes as the plain text it stands for. Thus the template's own markers are the only
//! marker tokens (spec 11.3, "Escapes").

use std::sync::Arc;

use nagaya_jinja::{Json, Template};

use crate::Error;
use crate::tokenizer::Tokenizer;

/// The text of `<|startoftext|>`. The template writes it first, as `bos_token`.
pub const BOS: &str = "<|startoftext|>";

/// The template variable `preserve_thinking`. An assistant message keeps its reasoning after a later user message,
/// so a held prefix stays a prefix (spec 11.11, "The choices of the LFM3 family").
const PRESERVE_THINKING: bool = true;

/// The members that the template compares with fixed words, such as `assistant` or `image`. They are not escaped. The
/// library writes them, not the client.
const NAMING_KEYS: [&str; 2] = ["role", "type"];

/// The template variables of a render, other than the messages and the tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// `enable_thinking`.
    pub thinking: bool,
    /// `add_generation_prompt`.
    pub generation_prompt: bool,
    /// `preserve_thinking`. The library always sets it.
    pub preserve_thinking: bool,
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
    bos: u32,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Chat")
    }
}

impl Chat {
    /// Compile the template `source` for `tokenizer`. A tokenizer with no `<|startoftext|>` is not an LFM3 tokenizer.
    pub fn new(source: &str, tokenizer: Arc<Tokenizer>) -> Result<Chat, Error> {
        let bos = tokenizer
            .token_id(BOS)
            .ok_or_else(|| Error::Model(format!("the vocabulary has no {BOS}")))?;
        Ok(Chat {
            template: Template::from_source(source)?,
            source: source.to_string(),
            tokenizer,
            bos,
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

    /// Render template-shaped `messages` and `tools` (spec 11.11), with the escapes. The text is the escaped text of
    /// the template, ready for more escaped text after it.
    pub fn render(
        &self,
        messages: &[Json],
        tools: &[Json],
        thinking: bool,
        generation_prompt: bool,
    ) -> Result<Rendered, Error> {
        let options = Options {
            thinking,
            generation_prompt,
            preserve_thinking: PRESERVE_THINKING,
        };
        self.render_with(messages, tools, options)
    }

    /// Render as [`Chat::render`] does, with all the template variables that the library sets. A test compares other
    /// values with the reference.
    pub fn render_with(&self, messages: &[Json], tools: &[Json], options: Options) -> Result<Rendered, Error> {
        let tools: Vec<Json> = tools.iter().map(|t| self.escape_json(t)).collect::<Result<_, _>>()?;
        let messages: Vec<Json> = messages.iter().map(|m| self.escape_json(m)).collect::<Result<_, _>>()?;
        let mut variables = vec![("messages", Json::Array(messages))];
        if !tools.is_empty() {
            variables.push(("tools", Json::Array(tools)));
        }
        variables.extend([
            ("add_generation_prompt", Json::Bool(options.generation_prompt)),
            ("enable_thinking", Json::Bool(options.thinking)),
            ("preserve_thinking", Json::Bool(options.preserve_thinking)),
            ("bos_token", Json::Str(BOS.to_string())),
        ]);
        let escaped = self.template.render(&variables)?;
        self.finish(escaped)
    }

    /// The text and the tokens of escaped text that starts with `<|startoftext|>`.
    pub fn finish(&self, escaped: String) -> Result<Rendered, Error> {
        let tokens = self.tokenizer.encode_escaped(&escaped)?;
        if tokens.first() != Some(&self.bos) {
            return Err(Error::Template(format!(
                "the rendered prompt does not start with {BOS}. The chat template changed shape."
            )));
        }
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
