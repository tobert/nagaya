//! The Inkling 2 family library for the council (spec 11.3, 11.3a, milestone 35 step 3).

mod messages;
mod read;
mod reply;
mod test_set;

use std::path::Path;
use std::sync::Arc;

use gogi_proto::context::{Context, Tool};
use gogi_proto::message::Message;
use nagaya_council::family::{
    Boundary, ControlHit, DecisionInput, Family, FamilyError, RawRead, ReadPlan, Rendering, ReplyParser, ReplyStart,
};
use nagaya_council::template_id::{council_hash, hash16, template_id};
use nagaya_jinja::Json;

use crate::Error;
use crate::chat::{Chat, Rendered};
use crate::tokenizer::{Tokenizer, check_text};

pub const FAMILY: &str = "inkling2";

pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TEMPLATE_FILE: &str = "chat_template.jinja";
pub const GENERATION_FILE: &str = "generation_config.json";
pub const CONFIG_FILE: &str = "config.json";

#[derive(Clone, Debug)]
pub struct Files {
    pub tokenizer: Vec<u8>,
    pub template: String,
    pub generation: String,
    pub config: String,
}

impl Files {
    pub fn read(dir: &Path) -> Result<Files, Error> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|source| Error::Io {
                path: path.display().to_string(),
                source,
            })
        };
        let text =
            |name: &str| String::from_utf8(read(name)?).map_err(|_| Error::Model(format!("{name} is not UTF-8 text")));
        Ok(Files {
            tokenizer: read(TOKENIZER_FILE)?,
            template: text(TEMPLATE_FILE)?,
            generation: text(GENERATION_FILE)?,
            config: text(CONFIG_FILE)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Marks {
    pub message_model: u32,
    pub content_thinking: u32,
    pub content_text: u32,
    pub content_tool_error: u32,
    pub content_invoke_tool_text: u32,
    pub content_invoke_tool_json: u32,
    pub end_message: u32,
}

pub struct Inkling2Family {
    chat: Chat,
    vocab_size: usize,
    letters: Vec<u32>,
    bare_letters: Vec<u32>,
    stop: Vec<u32>,
    marks: Marks,
    template_id: String,
}

impl std::fmt::Debug for Inkling2Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inkling2Family")
            .field("vocab_size", &self.vocab_size)
            .field("template_id", &self.template_id)
            .finish()
    }
}

pub(crate) fn client(e: Error) -> FamilyError {
    FamilyError::client(e.to_string())
}

pub(crate) fn internal(e: Error) -> FamilyError {
    FamilyError::internal(e.to_string())
}

fn tokens_error(e: Error) -> FamilyError {
    match e {
        Error::TokenId { .. } | Error::Bytes(_) => FamilyError::client_param("tokens", e.to_string()),
        other => internal(other),
    }
}

pub(crate) fn refusal(e: Error) -> FamilyError {
    match e {
        Error::Template(_) | Error::Reserved { .. } | Error::Json(_) => client(e),
        other => internal(other),
    }
}

fn unpadded_vocab_size_of(config: &str, tokenizer: &Tokenizer) -> Result<usize, Error> {
    let config: serde_json::Value =
        serde_json::from_str(config).map_err(|e| Error::Model(format!("config.json is not JSON: {e}")))?;
    let size = config["text_config"]["unpadded_vocab_size"]
        .as_u64()
        .ok_or_else(|| Error::Model("config.json has no text_config.unpadded_vocab_size".to_string()))?
        as usize;
    if size != tokenizer.len() {
        return Err(Error::Model(format!(
            "unpadded_vocab_size {size} does not match tokenizer size {}",
            tokenizer.len()
        )));
    }
    Ok(size)
}

fn stop_of(generation: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>, Error> {
    let config: serde_json::Value = serde_json::from_str(generation)
        .map_err(|e| Error::Model(format!("generation_config.json is not JSON: {e}")))?;
    let ids: Vec<&serde_json::Value> = match &config["eos_token_id"] {
        serde_json::Value::Array(list) => list.iter().collect(),
        one @ serde_json::Value::Number(_) => vec![one],
        other => {
            return Err(Error::Model(format!(
                "generation_config.json has eos_token_id {other}, not an id or a list of ids"
            )));
        }
    };
    let stop: Vec<u32> = ids
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&id| (id as usize) < tokenizer.len())
                .ok_or_else(|| Error::Model(format!("generation_config.json has eos_token_id {v}")))
        })
        .collect::<Result<_, _>>()?;
    if stop.is_empty() {
        return Err(Error::Model("generation_config.json has no eos_token_id".to_string()));
    }
    Ok(stop)
}

fn mark(tokenizer: &Tokenizer, text: &str) -> Result<u32, Error> {
    tokenizer
        .token_id(text)
        .ok_or_else(|| Error::Model(format!("the tokenizer has no {text}")))
}

fn letters_with(tokenizer: &Tokenizer, leading: &str) -> Result<Vec<u32>, Error> {
    (b'A'..=b'Z')
        .map(|b| {
            let s = format!("{leading}{}", char::from(b));
            let tokens = tokenizer.encode_content(&s)?;
            match tokens[..] {
                [id] => Ok(id),
                _ => Err(Error::Model(format!("{s:?} is {} tokens, expected 1", tokens.len()))),
            }
        })
        .collect()
}

impl Inkling2Family {
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        Self::from_files(Files::read(dir)?)
    }

    pub fn from_files(files: Files) -> Result<Self, Error> {
        let tokenizer = Arc::new(Tokenizer::from_bytes(&files.tokenizer)?);
        let vocab_size = unpadded_vocab_size_of(&files.config, &tokenizer)?;
        let stop = stop_of(&files.generation, &tokenizer)?;
        let marks = Marks {
            message_model: mark(&tokenizer, "<|message_model|>")?,
            content_thinking: mark(&tokenizer, "<|content_thinking|>")?,
            content_text: mark(&tokenizer, "<|content_text|>")?,
            content_tool_error: mark(&tokenizer, "<|content_tool_error|>")?,
            content_invoke_tool_text: mark(&tokenizer, "<|content_invoke_tool_text|>")?,
            content_invoke_tool_json: mark(&tokenizer, "<|content_invoke_tool_json|>")?,
            end_message: mark(&tokenizer, "<|end_message|>")?,
        };
        let letters = letters_with(&tokenizer, " ")?;
        let bare_letters = letters_with(&tokenizer, "")?;
        let chat = Chat::new(&files.template, Arc::clone(&tokenizer))?;
        let mut family = Inkling2Family {
            chat,
            vocab_size,
            letters,
            bare_letters,
            stop,
            marks,
            template_id: String::new(),
        };
        let c_hash = council_hash(&test_set::renders(&family));
        let t_hash = hash16(files.template.as_bytes());
        family.template_id = template_id(&c_hash, FAMILY, &t_hash);
        Ok(family)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.chat.tokenizer()
    }

    pub(crate) fn shared_tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(self.chat.tokenizer())
    }

    pub fn tokenizer_hash(&self) -> &str {
        self.tokenizer().hash()
    }

    pub(crate) fn run(&self, messages: &[Json], tools: &[Json]) -> Result<Rendered, FamilyError> {
        self.chat.render(messages, tools).map_err(refusal)
    }

    pub(crate) fn run_open(
        &self,
        before: &[Json],
        open: &Message,
        tools: &[Json],
        thinking: bool,
    ) -> Result<Rendered, FamilyError> {
        let prompt = self.run(before, tools)?;
        let mut escaped = prompt.escaped;
        escaped.push_str("<|message_model|>");
        if thinking {
            let r = open.reasoning.as_deref().unwrap_or("");
            if !r.is_empty() || open.content.is_none() {
                escaped.push_str("<|content_thinking|>");
                escaped.push_str(&self.escaped(r)?);
            }
            if let Some(c) = open.text() {
                if !r.is_empty() {
                    escaped.push_str("<|end_message|>");
                }
                escaped.push_str("<|content_text|>");
                escaped.push_str(&self.escaped(c)?);
            }
        } else {
            escaped.push_str("<|content_text|>");
            escaped.push_str(&self.escaped(open.text().unwrap_or(""))?);
        }
        self.chat.finish(escaped).map_err(refusal)
    }

    fn check_seams(&self, before: &[Json], open: &Message, tools: &[Json], thinking: bool) -> Result<(), FamilyError> {
        for seam in gogi_proto::continuation::seams(open) {
            let start = self.run_open(before, &seam.before, tools, thinking)?.tokens;
            let with = self.run_open(before, &seam.with, tools, thinking)?.tokens;
            gogi_proto::continuation::check_seam(seam.field, &start, &with, |ids| {
                self.tokenizer().decode_lossy(ids).unwrap_or_default()
            })?;
        }
        Ok(())
    }

    pub(crate) fn escaped(&self, text: &str) -> Result<String, FamilyError> {
        self.tokenizer().escape(text).map_err(refusal)
    }

    pub(crate) fn encode(&self, escaped: &str) -> Result<Vec<u32>, FamilyError> {
        self.tokenizer().encode_escaped(escaped).map_err(refusal)
    }

    pub(crate) fn hits(&self, place: &str, text: &str) -> Vec<ControlHit> {
        self.tokenizer()
            .control_hits(text)
            .into_iter()
            .map(|token| ControlHit {
                place: place.to_string(),
                token,
            })
            .collect()
    }
}

impl Family for Inkling2Family {
    fn render(&self, context: &Context, _generation_prompt: bool) -> Result<Rendering, FamilyError> {
        context.check()?;
        let tools = messages::tools_json(&context.tools);
        let msgs = messages::messages_json(&context.messages)?;
        let open = context.open();
        let base = match open {
            Some(open) => {
                let before = &msgs[..msgs.len() - 1];
                self.check_seams(before, open, &tools, context.thinking)?;
                self.run_open(before, open, &tools, context.thinking)?
            }
            None => self.run(&msgs, &tools)?,
        };
        let end = base.tokens.len();
        let last = msgs.len() - 1;
        let mut boundaries: Vec<Boundary> = Vec::new();
        for (i, message) in context.messages.iter().enumerate().take(last) {
            if !message.snap {
                continue;
            }
            let part = self.run(&msgs[..=i], &tools)?;
            let after = boundaries.last().map_or(0, |b| b.tokens);
            if base.tokens.starts_with(&part.tokens) && part.tokens.len() > after && part.tokens.len() < end {
                boundaries.push(Boundary {
                    message: i,
                    tokens: part.tokens.len(),
                });
            }
        }
        boundaries.push(Boundary {
            message: last,
            tokens: end,
        });
        let control_text = context
            .messages
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                let place = format!("message:{i}");
                messages::message_strings(m)
                    .into_iter()
                    .flat_map(|s| self.hits(&place, s))
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(Rendering {
            tokens: base.tokens,
            end,
            boundaries,
            control_text,
            media: Vec::new(),
        })
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        read::decision_read(self, input)
    }

    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
        read::raw_read(self, letters, suffixes)
    }

    fn reply_parser(&self, start: ReplyStart, tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        Ok(reply::reply_parser(self, start, tools))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        check_text(text)
            .and_then(|()| self.tokenizer().encode_content(text))
            .map_err(refusal)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        check_text(text)
            .and_then(|()| self.tokenizer().encode(text))
            .map_err(refusal)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        self.tokenizer().decode_whole(tokens).map_err(tokens_error)
    }

    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError> {
        tokens
            .iter()
            .map(|&id| self.tokenizer().decode_lossy(&[id]).map_err(tokens_error))
            .collect()
    }

    fn template_id(&self) -> &str {
        &self.template_id
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn thinks(&self) -> bool {
        true
    }
}
