//! The chat template of Qwen3.8: the Jinja source in the GGUF, run by `nagaya-jinja` with the variables of the
//! reference.

use crate::Error;
use crate::gguf::{Metadata, Value as MetaValue};
use crate::json::Json;
use nagaya_jinja::{Template, truthy};

/// What a render takes besides the messages and the tools.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// The template variable `enable_thinking`.
    pub thinking: bool,
    /// The template variable `add_generation_prompt`.
    pub generation_prompt: bool,
    /// The template variable `reasoning_effort`. `None` leaves the variable out.
    pub reasoning_effort: Option<String>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            thinking: true,
            generation_prompt: true,
            reasoning_effort: None,
        }
    }
}

/// A chat template, ready to render.
#[derive(Debug)]
pub struct ChatTemplate {
    template: Template,
}

impl ChatTemplate {
    /// Compile the template source. A syntax error is an error here, not at the first render.
    pub fn from_source(source: &str) -> Result<ChatTemplate, Error> {
        Ok(ChatTemplate {
            template: Template::from_source(source)?,
        })
    }

    /// Compile the template in the key `tokenizer.chat_template`.
    pub fn from_metadata(m: &Metadata) -> Result<ChatTemplate, Error> {
        match m.get("tokenizer.chat_template") {
            Some(MetaValue::Str(source)) => ChatTemplate::from_source(source),
            Some(_) => Err(Error::Metadata(
                "tokenizer.chat_template is not a string. Use a GGUF made for this family.".into(),
            )),
            None => Err(Error::Metadata(
                "the key tokenizer.chat_template is missing. Use a GGUF made for this family.".into(),
            )),
        }
    }

    /// The text for `messages`, a JSON array. `tools` that are empty count as no tools, as in the reference.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, options: &RenderOptions) -> Result<String, Error> {
        if !matches!(messages, Json::Array(_)) {
            return Err(Error::Template("the messages must be an array".to_string()));
        }
        let mut variables: Vec<(&str, Json)> = vec![
            ("messages", messages.clone()),
            ("add_generation_prompt", Json::Bool(options.generation_prompt)),
            ("enable_thinking", Json::Bool(options.thinking)),
        ];
        if let Some(tools) = tools.filter(|t| truthy(t)) {
            variables.push(("tools", tools.clone()));
        }
        if let Some(effort) = &options.reasoning_effort {
            variables.push(("reasoning_effort", Json::Str(effort.clone())));
        }
        Ok(self.template.render(&variables)?)
    }
}
