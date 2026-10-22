//! Gemma 4 chat rendering: Google's template in the nagaya's interpreter, with the escapes of mkviolet `src/chat.rs`.
//!
//! Every string that a client sends is content. Before the render, each special-token text in it (`<|turn>`,
//! `<|channel>`, ...) becomes a sentinel of Unicode noncharacters. After the render, each sentinel is tokenized as the
//! plain text it stands for. Thus the template's own markers are the only special tokens. A noncharacter in client
//! text is refused (spec 11.3, "Escapes").
//!
//! With thinking on, the generation prompt also opens the thought channel (`<|channel>thought\n`), so the reply starts
//! inside the thinking. This is mkviolet's addition to the template. The template opens the channel itself only after
//! a tool response, and with thinking off it closes an empty thought channel.

use std::sync::Arc;

use nagaya_jinja::{Json, Template};

use crate::tokenizer::{BOS, Tokenizer, check_text};
use crate::{CHAT_TEMPLATE, Error};

/// The thought channel that a reply with thinking on starts in.
pub const THOUGHT_OPEN: &str = "<|channel>thought\n";

/// The template variable `preserve_thinking`. An assistant message with tool calls keeps its thinking after a later
/// user message. Thus an agent turn renders the same when the next prompt arrives, and a held prefix stays a prefix.
const PRESERVE_THINKING: bool = true;

const SENTINEL_OPEN: char = '\u{FDD0}';
const SENTINEL_CLOSE: char = '\u{FDD1}';

/// The members of a template message that name a thing and do not say it. They are not content, so they are not
/// escaped, and the `id` of a call matches the `tool_call_id` of its result as written.
const NAMING_KEYS: [&str; 3] = ["role", "id", "tool_call_id"];

/// A rendered chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    /// The text, with the special-token text of the content back in place.
    pub text: String,
    /// The token ids. Only the markers of the template are special tokens.
    pub tokens: Vec<u32>,
}

/// The special-token text of client strings, swapped for sentinels while the template runs.
struct Escaper<'a> {
    specials: &'a [String],
    originals: Vec<String>,
}

impl Escaper<'_> {
    /// `s` with each special-token text replaced by a sentinel that names its original.
    fn escape(&mut self, s: &str) -> Result<String, Error> {
        check_text(s)?;
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        'outer: while let Some(c) = rest.chars().next() {
            for special in self.specials {
                if rest.starts_with(special.as_str()) {
                    out.push(SENTINEL_OPEN);
                    out.push_str(&self.originals.len().to_string());
                    out.push(SENTINEL_CLOSE);
                    self.originals.push(special.clone());
                    rest = &rest[special.len()..];
                    continue 'outer;
                }
            }
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
        Ok(out)
    }

    /// The same JSON with every string escaped, keys too.
    fn escape_json(&mut self, j: &Json) -> Result<Json, Error> {
        Ok(match j {
            Json::Str(s) => Json::Str(self.escape(s)?),
            Json::Array(a) => Json::Array(a.iter().map(|x| self.escape_json(x)).collect::<Result<_, _>>()?),
            Json::Object(o) => Json::Object(
                o.iter()
                    .map(|(k, v)| Ok((self.escape(k)?, self.escape_json(v)?)))
                    .collect::<Result<_, Error>>()?,
            ),
            other => other.clone(),
        })
    }

    /// A message with its content escaped and its naming members as written.
    fn escape_message(&mut self, m: &Json) -> Result<Json, Error> {
        let Json::Object(o) = m else {
            return Err(Error::Template("a message is not an object".to_string()));
        };
        Ok(Json::Object(
            o.iter()
                .map(|(k, v)| {
                    let v = if NAMING_KEYS.contains(&k.as_str()) {
                        v.clone()
                    } else {
                        self.escape_json(v)?
                    };
                    Ok((k.clone(), v))
                })
                .collect::<Result<_, Error>>()?,
        ))
    }
}

fn member<'a>(m: &'a Json, key: &str) -> Option<&'a Json> {
    match m {
        Json::Object(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

/// Split each assistant message that has both text and tool calls into two: the text with its thinking, then the
/// calls. Google's template renders the text of a message after its calls and their results, and the model writes the
/// text first. Two assistant messages in a row render as one model turn, so the split puts the text where the model
/// wrote it (mkviolet `split_text_and_calls`).
fn split_text_and_calls(messages: Vec<Json>) -> Vec<Json> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let has_calls = matches!(member(&m, "tool_calls"), Some(Json::Array(c)) if !c.is_empty());
        let has_text = matches!(member(&m, "content"), Some(Json::Str(c)) if !c.trim().is_empty());
        let assistant = matches!(member(&m, "role"), Some(Json::Str(r)) if r == "assistant");
        match (&m, assistant && has_calls && has_text) {
            (Json::Object(o), true) => {
                let text = o.iter().filter(|(k, _)| k != "tool_calls").cloned().collect();
                let calls = o
                    .iter()
                    .filter(|(k, _)| !matches!(k.as_str(), "content" | "reasoning_content" | "reasoning"))
                    .cloned()
                    .collect();
                out.push(Json::Object(text));
                out.push(Json::Object(calls));
            }
            _ => out.push(m),
        }
    }
    out
}

/// Google's template, ready to render, with the tokenizer of its checkpoint.
pub struct Chat {
    template: Template,
    tokenizer: Arc<Tokenizer>,
    bos: u32,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Chat")
    }
}

impl Chat {
    /// Compile the template for `tokenizer`. A tokenizer with no `<bos>` is not a Gemma 4 tokenizer.
    pub fn new(tokenizer: Arc<Tokenizer>) -> Result<Chat, Error> {
        let bos = tokenizer
            .token_id(BOS)
            .ok_or_else(|| Error::Tokenizer("the vocabulary has no <bos>. Use a Gemma 4 tokenizer.".to_string()))?;
        Ok(Chat {
            template: Template::from_source(CHAT_TEMPLATE)?,
            tokenizer,
            bos,
        })
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// Render template-shaped `messages` (role, content, reasoning_content, tool_calls with id and function,
    /// tool_call_id) and `tools`, and tokenize the result.
    pub fn render(
        &self,
        messages: &[Json],
        tools: &[Json],
        thinking: bool,
        generation_prompt: bool,
    ) -> Result<Rendered, Error> {
        self.render_then(messages, tools, thinking, generation_prompt, &[])
    }

    /// Render `messages` with the generation prompt, then add the open text of a reply as the model writes it
    /// (spec 11.3c, "The Gemma 4 rendering"). Each `(text, marker)` pair adds the text as content, then the marker as
    /// template text. The whole text is tokenized as one string, so a field can merge with the text before it.
    pub fn render_open(
        &self,
        messages: &[Json],
        tools: &[Json],
        thinking: bool,
        open: &[(&str, &str)],
    ) -> Result<Rendered, Error> {
        self.render_then(messages, tools, thinking, true, open)
    }

    fn render_then(
        &self,
        messages: &[Json],
        tools: &[Json],
        thinking: bool,
        generation_prompt: bool,
        open: &[(&str, &str)],
    ) -> Result<Rendered, Error> {
        let mut esc = Escaper {
            specials: self.tokenizer.specials(),
            originals: Vec::new(),
        };
        let tools: Vec<Json> = tools.iter().map(|t| esc.escape_json(t)).collect::<Result<_, _>>()?;
        let messages: Vec<Json> = messages
            .iter()
            .map(|m| esc.escape_message(m))
            .collect::<Result<_, _>>()?;
        let mut variables = vec![("messages", Json::Array(split_text_and_calls(messages)))];
        if !tools.is_empty() {
            variables.push(("tools", Json::Array(tools)));
        }
        variables.extend([
            ("add_generation_prompt", Json::Bool(generation_prompt)),
            ("enable_thinking", Json::Bool(thinking)),
            ("preserve_thinking", Json::Bool(PRESERVE_THINKING)),
            ("bos_token", Json::Str(BOS.to_string())),
        ]);
        let mut escaped = self.template.render(&variables)?;
        // After a tool response the template opens the thought channel itself.
        if thinking && generation_prompt && !escaped.ends_with(THOUGHT_OPEN) {
            escaped.push_str(THOUGHT_OPEN);
        }
        for (text, marker) in open {
            escaped.push_str(&esc.escape(text)?);
            escaped.push_str(marker);
        }
        let rendered = self.tokenize_escaped(&escaped, &esc.originals)?;
        if rendered.tokens.first() != Some(&self.bos) {
            return Err(Error::Template(
                "the rendered prompt does not start with <bos>. The chat template changed shape.".to_string(),
            ));
        }
        Ok(rendered)
    }

    /// Tokenize rendered text: template text with special tokens on, and each sentinel as the plain text it stands
    /// for.
    fn tokenize_escaped(&self, text: &str, originals: &[String]) -> Result<Rendered, Error> {
        let tok = &self.tokenizer;
        let mut tokens = Vec::new();
        let mut readable = String::with_capacity(text.len());
        let mut rest = text;
        while !rest.is_empty() {
            let Some(at) = rest.find(SENTINEL_OPEN) else {
                tokens.extend(tok.encode(rest)?);
                readable.push_str(rest);
                break;
            };
            let (before, after) = rest.split_at(at);
            if !before.is_empty() {
                tokens.extend(tok.encode(before)?);
                readable.push_str(before);
            }
            let after = &after[SENTINEL_OPEN.len_utf8()..];
            let bad = || Error::Template("the template changed a sentinel of the escapes".to_string());
            let close = after.find(SENTINEL_CLOSE).ok_or_else(bad)?;
            let index: usize = after[..close].parse().map_err(|_| bad())?;
            let original = originals.get(index).ok_or_else(bad)?;
            tokens.extend(tok.encode_content(original)?);
            readable.push_str(original);
            rest = &after[close + SENTINEL_CLOSE.len_utf8()..];
        }
        Ok(Rendered { text: readable, tokens })
    }
}
