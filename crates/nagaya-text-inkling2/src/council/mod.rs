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
use crate::chat::{Effort, Options};
use crate::tokenizer::{Tokenizer, check_text};

pub const FAMILY: &str = "inkling2";

pub use messages::ImagePart;

/// The objects of this machine, as the family reads them for the image parts of a context (spec 6.15, 11.11).
pub trait Objects: Send + Sync {
    /// The bytes of the object with the content id `id`, or none when this machine does not hold it.
    fn bytes(&self, id: &str) -> Option<Vec<u8>>;
}

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
    /// The reader of the objects of this machine. A family without one refuses each image part.
    objects: Option<Arc<dyn Objects>>,
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
        let placeholder = mark(&tokenizer, "<|unused_200054|>")?;
        if placeholder != crate::media::IMAGE_TOKEN {
            return Err(Error::Model(format!(
                "the image placeholder <|unused_200054|> has the id {placeholder}, and the engine reads {}",
                crate::media::IMAGE_TOKEN
            )));
        }
        let chat = Chat::new(&files.template, Arc::clone(&tokenizer))?;
        let mut family = Inkling2Family {
            chat,
            vocab_size,
            letters,
            bare_letters,
            stop,
            marks,
            template_id: String::new(),
            objects: None,
        };
        let c_hash = council_hash(&test_set::renders(&family));
        let t_hash = hash16(files.template.as_bytes());
        family.template_id = template_id(&c_hash, FAMILY, &t_hash);
        Ok(family)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.chat.tokenizer()
    }

    /// The family with a reader of the objects of this machine. The process gives it, so that a render can read the
    /// image parts of a context.
    pub fn with_objects(mut self, objects: Arc<dyn Objects>) -> Inkling2Family {
        self.objects = Some(objects);
        self
    }

    /// The image parts of a context, in the order of the rendering.
    pub fn image_parts(&self, context: &Context) -> Result<Vec<ImagePart>, FamilyError> {
        messages::image_parts(&context.messages)
    }

    /// The bytes of the object `id` that the request member `param` names. An absent reader and an absent object
    /// are client errors that name the member.
    fn object_bytes(&self, id: &str, param: &str) -> Result<Vec<u8>, FamilyError> {
        let Some(objects) = &self.objects else {
            return Err(FamilyError::client_param(
                param,
                "nothing ran: this server gives the Inkling 2 family no objects, so it cannot read an image part. \
                 Send text only.",
            ));
        };
        objects.bytes(id).ok_or_else(|| {
            FamilyError::client_param(
                param,
                format!("nothing ran: the object {id} is not on this server. Upload it, then send the request again."),
            )
        })
    }

    /// The grid of an image part from the header of its object.
    fn part_grid(&self, part: &ImagePart) -> Result<crate::media::Grid, FamilyError> {
        self.grid_named(part, &part.param())
    }

    /// The grid of an image part, with `param` as the request member that an error names.
    fn grid_named(&self, part: &ImagePart, param: &str) -> Result<crate::media::Grid, FamilyError> {
        let bytes = self.object_bytes(&part.id, param)?;
        crate::media::grid(&bytes, &part.mime).map_err(|e| media_error(param, e))
    }

    /// The record and the rows of one span of a rendering. The object can leave the store after the render, so an
    /// absent object is a client error. A count that the object does not give is a defect of the caller.
    fn media_input(
        &self,
        span: &nagaya_council::family::MediaSpan,
    ) -> Result<nagaya_council::family::MediaInput, FamilyError> {
        self.media_input_named(span, &format!("the object {}", span.id))
    }

    /// The record and the rows of one span, with `param` as the request member that an error names.
    fn media_input_named(
        &self,
        span: &nagaya_council::family::MediaSpan,
        param: &str,
    ) -> Result<nagaya_council::family::MediaInput, FamilyError> {
        let bytes = self.object_bytes(&span.id, param)?;
        let rows = crate::media::prepare(&bytes, &span.mime).map_err(|e| media_error(param, e))?;
        if rows.positions() != span.count {
            return Err(FamilyError::internal(format!(
                "the span of {} has {} positions, and its rows give {}",
                span.id,
                span.count,
                rows.positions()
            )));
        }
        Ok(nagaya_council::family::MediaInput {
            span: span.clone(),
            meta: rows.record(span.first),
            rows: rows.bytes().into(),
        })
    }

    pub(crate) fn shared_tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(self.chat.tokenizer())
    }

    pub fn tokenizer_hash(&self) -> &str {
        self.tokenizer().hash()
    }

    /// The chat template text of the folder. The identity's `template` is this text (spec 12.1).
    pub fn template(&self) -> &str {
        self.chat.source()
    }

    /// Render the messages with the effort that `thinking` chooses: with thinking off the template writes the effort
    /// "none", and with thinking on its default, 0.9 (milestone 35 question 4).
    pub(crate) fn run(&self, messages: &[Json], tools: &[Json], thinking: bool) -> Result<Rendered, FamilyError> {
        let effort = if thinking { None } else { Some(Effort::None) };
        self.chat
            .render_with(messages, tools, Options { effort })
            .map_err(refusal)
    }

    pub(crate) fn run_open(
        &self,
        before: &[Json],
        open: &Message,
        tools: &[Json],
        thinking: bool,
    ) -> Result<Rendered, FamilyError> {
        let prompt = self.run(before, tools, thinking)?;
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
    /// The generation prompt is `<|message_model|>`. The template has none, and the model writes this token first: the
    /// reference gives it a probability of 1 - 6e-12 after the prompt "chat". The council needs a generation prompt for
    /// the first logit row (spec 11.18). An open message takes its place.
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
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
            None => self.run(&msgs, &tools, context.thinking)?,
        };
        let end = base.tokens.len();
        let last = msgs.len() - 1;
        let mut boundaries: Vec<Boundary> = Vec::new();
        for (i, message) in context.messages.iter().enumerate().take(last) {
            if !message.snap {
                continue;
            }
            let part = self.run(&msgs[..=i], &tools, context.thinking)?;
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
        let mut tokens = base.tokens;
        if generation_prompt && open.is_none() {
            tokens.push(self.marks.message_model);
        }
        let rendering = Rendering {
            tokens,
            end,
            boundaries,
            control_text,
            media: Vec::new(),
        };
        // engine-abi "The images of Inkling 2", "The placeholder": the header of each part gives its count.
        let parts = self.image_parts(context)?;
        if parts.is_empty() {
            return Ok(rendering);
        }
        let grids = parts
            .iter()
            .map(|p| self.part_grid(p))
            .collect::<Result<Vec<_>, FamilyError>>()?;
        expand(rendering, &parts, &grids)
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        read::decision_read(self, input, &[])
    }

    fn decision_read_with_media(
        &self,
        input: &DecisionInput<'_>,
        case_media: &[gogi_proto::objects::ObjectRef],
    ) -> Result<Box<dyn ReadPlan>, FamilyError> {
        read::decision_read(self, input, case_media)
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

    fn media_inputs(&self) -> Vec<String> {
        crate::media::MEDIA_INPUTS.iter().map(|m| (*m).to_owned()).collect()
    }

    /// A latent position holds `<|endoftext|>`, the `pad_token_id` of `generation_config.json` (engine-abi "The lab of
    /// Inkling 2"). The record is the kind 5, the count, the first position, then the width of a row and three 0.
    fn latent_record(&self, first: usize, count: usize, width: usize) -> Option<(u32, Vec<u8>)> {
        let pad = self.tokenizer().token_id(LATENT_PLACEHOLDER)?;
        let word = |value: usize| u32::try_from(value).expect("a latent span fits in 32 bits");
        let mut record = Vec::with_capacity(32);
        record.extend(5u32.to_le_bytes());
        record.extend(word(count).to_le_bytes());
        record.extend((first as u64).to_le_bytes());
        for value in [width, 0, 0, 0] {
            record.extend(word(value).to_le_bytes());
        }
        Some((pad, record))
    }

    fn media_rows(&self, rendering: &Rendering) -> Result<Vec<nagaya_council::family::MediaInput>, FamilyError> {
        rendering.media.iter().map(|span| self.media_input(span)).collect()
    }
}

/// The error of an image part whose bytes the family refused. A media type that the family does not read is a
/// defect: `image_parts` refuses it first.
fn media_error(param: &str, e: crate::media::MediaError) -> FamilyError {
    match e {
        crate::media::MediaError::Mime(_) => FamilyError::internal(format!("{param} {e}")),
        other => FamilyError::client_param(param, format!("nothing ran: {param} {other}.")),
    }
}

/// Put one placeholder for each image token in place of the one that the template writes for each part
/// (`processing_inkling2.py:37`), and move the end and the boundaries with it. The template writes the placeholder
/// once for each part, and client text cannot spell it, so a count that differs is a defect of the template.
fn expand(rendering: Rendering, parts: &[ImagePart], grids: &[crate::media::Grid]) -> Result<Rendering, FamilyError> {
    let placeholder = crate::media::IMAGE_TOKEN;
    let mut tokens = Vec::with_capacity(rendering.tokens.len());
    // For each position of the template's tokens, the position in the expanded tokens.
    let mut moved = Vec::with_capacity(rendering.tokens.len() + 1);
    let mut media = Vec::with_capacity(parts.len());
    for &token in &rendering.tokens {
        moved.push(tokens.len());
        if token != placeholder {
            tokens.push(token);
            continue;
        }
        let n = media.len();
        let (Some(part), Some(grid)) = (parts.get(n), grids.get(n)) else {
            return Err(FamilyError::internal(
                "the rendering holds more image placeholders than the context has image parts. The chat template \
                 changed shape.",
            ));
        };
        media.push(nagaya_council::family::MediaSpan {
            first: tokens.len(),
            count: grid.tokens(),
            mime: part.mime.clone(),
            id: part.id.clone(),
        });
        tokens.extend(std::iter::repeat_n(placeholder, grid.tokens()));
    }
    moved.push(tokens.len());
    if media.len() != parts.len() {
        return Err(FamilyError::internal(format!(
            "the context has {} image parts, and the template wrote {} placeholders",
            parts.len(),
            media.len()
        )));
    }
    Ok(Rendering {
        end: moved[rendering.end],
        boundaries: rendering
            .boundaries
            .iter()
            .map(|b| Boundary {
                message: b.message,
                tokens: moved[b.tokens],
            })
            .collect(),
        control_text: rendering.control_text,
        tokens,
        media,
    })
}

/// The placeholder of a latent position: `<|endoftext|>`, the `pad_token_id` of `generation_config.json`.
pub const LATENT_PLACEHOLDER: &str = "<|endoftext|>";
