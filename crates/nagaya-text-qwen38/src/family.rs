//! The family library of Qwen3.8: the tokenizer and the chat template together.
//!
//! This is `render_messages` of the reference's `service/common.py`. Every string in the messages and the tools is content. Its
//! special token text is escaped before the template runs, so that the markers of the template are the only special
//! tokens in the ids.

use crate::Error;
use crate::gguf::Metadata;
use crate::json::Json;
use crate::template::{ChatTemplate, RenderOptions};
use crate::tokenizer::Tokenizer;
use nagaya_jinja::truthy;
use std::path::Path;
use std::sync::Arc;

/// The result of a render.
#[derive(Debug)]
pub struct Rendered {
    /// The text that the template made, with an escape character where content held a special token's text.
    pub escaped: String,
    /// The same text with each escape character replaced by the text it stands for.
    pub text: String,
    /// The token ids of the rendered text. Only the markers of the template are special tokens.
    pub ids: Vec<u32>,
}

/// The text rules of one Qwen3.8 model.
#[derive(Debug)]
pub struct Qwen38 {
    tokenizer: Arc<Tokenizer>,
    template: ChatTemplate,
}

impl Qwen38 {
    /// Build the family library from the metadata of the GGUF file at `path`.
    pub fn from_gguf(path: &Path) -> Result<Qwen38, Error> {
        Qwen38::from_metadata(&Metadata::open(path)?)
    }

    pub fn from_metadata(m: &Metadata) -> Result<Qwen38, Error> {
        Ok(Qwen38 {
            tokenizer: Arc::new(Tokenizer::from_metadata(m)?),
            template: ChatTemplate::from_metadata(m)?,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tokenizer, shared. A reply parser that outlives a borrow holds this.
    pub fn shared_tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    /// Render a chat. A noncharacter in the messages or the tools is an error, and so is a refusal of the template.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, options: &RenderOptions) -> Result<Rendered, Error> {
        let messages = self.escape_tree(messages)?;
        let tools = match tools.filter(|t| truthy(t)) {
            Some(tools) => Some(self.escape_tree(tools)?),
            None => None,
        };
        let escaped = self.template.render(&messages, tools.as_ref(), options)?;
        let ids = self.tokenizer.encode_escaped(&escaped)?;
        let text = self.tokenizer.unescape(&escaped)?;
        Ok(Rendered { escaped, text, ids })
    }

    /// The same JSON with every string, and every key, escaped.
    fn escape_tree(&self, j: &Json) -> Result<Json, Error> {
        Ok(match j {
            Json::Str(s) => Json::Str(self.tokenizer.escape(s)?),
            Json::Array(a) => Json::Array(a.iter().map(|x| self.escape_tree(x)).collect::<Result<_, _>>()?),
            Json::Object(o) => Json::Object(
                o.iter()
                    .map(|(k, v)| Ok((self.tokenizer.escape(k)?, self.escape_tree(v)?)))
                    .collect::<Result<_, Error>>()?,
            ),
            other => other.clone(),
        })
    }
}
