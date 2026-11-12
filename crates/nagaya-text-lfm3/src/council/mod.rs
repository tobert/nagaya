//! The LFM3 family library for the council (spec 11.3, 11.3a, 11.11): nagaya's family interface on this crate.
//!
//! A context renders with the chat template of the model folder (`crate::chat`). A decision puts a spec, a case and
//! a letter menu after the context. This is mkviolet's rendering (`src/template.rs`) in LFM3's ChatML form, as the
//! Qwen3.8 library writes it:
//!
//! ```text
//! context messages (the chat template), or <|startoftext|> for a decision with no context
//! <|im_start|>user\n{instructions}\n\nAnswer each question below about the {label} in the next message. ...
//!   Question {n}. {instructions}\n{A}. {option}: {means}\n ...<|im_end|>\n                         spec layer
//! <|im_start|>user\n{label}:\n{case}<|im_end|>\n
//! <|im_start|>assistant\n<think></think>Answer to question                                          case
//!  {n} ({first sentence of the question}):                                  one read for each question
//! ```
//!
//! A queue read gives the case as the result of the context's last call, and writes the call to the queue tool in
//! the template's form: `<tool_call>\n<function={queue}>\n<parameter=answers>\n{"{id}": "` and the bare letter.
//!
//! One library serves both models. The folder says which model it is (`config.json`, `architectures`). Omni does not
//! reason, and only Omni reads media (spec 11.11). The family reads the bytes of a media part through the object
//! reader that the process gives (`Lfm3Family::with_objects`). The header of each part gives its count of positions,
//! and `Lfm3Family::prepare` also gives the rows that the engine reads.

mod media;
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

pub use media::{MediaRendering, MediaSpan};

/// The objects of this machine, as the family reads them for the media parts of a context (spec 6.15, 11.11).
pub trait Objects: Send + Sync {
    /// The bytes of the object with the content id `id`, or none when this machine does not hold it.
    fn bytes(&self, id: &str) -> Option<Vec<u8>>;
}

/// What the family hands the engine for one media span: its place in the tokens and its rows (engine-abi "LFM3").
#[derive(Clone, Debug, PartialEq)]
pub struct MediaInput {
    /// The kind, the first position and the count of positions of the part.
    pub span: MediaSpan,
    /// The prepared rows of the part, in f32.
    pub rows: crate::media::MediaRows,
}

/// A rendering with its media parts at their full size, and the rows of each part.
#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    pub rendering: Rendering,
    /// The inputs of the media parts, in the order of the rendering.
    pub inputs: Vec<MediaInput>,
}
pub use messages::{MediaKind, MediaPart};

/// The name of the family in the identity and in the template id.
pub const FAMILY: &str = "lfm3";

/// The files of a model folder that the library reads.
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TEMPLATE_FILE: &str = "chat_template.jinja";
pub const GENERATION_FILE: &str = "generation_config.json";
pub const CONFIG_FILE: &str = "config.json";

/// The end of the thinking. A read and a reply with thinking off start after `<think></think>`.
const THINK_CLOSE: &str = "</think>";

/// The two LFM3 models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `LFM3-Omni-2B`: text, images and audio in, text and speech out. It does not reason.
    Omni,
    /// `LFM3-1.7B`: text in and out, with thinking.
    Text,
}

/// How a generation writes its reply (spec 11.11, "The generation modes").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Text only. The sampler never picks a marker of another mode.
    Text,
    /// Speech only, after `<|audio_start|>`.
    Speech,
    /// Text and speech in turn, after `<|mixed_start|>`.
    Interleaved,
}

/// The files of a model folder, as bytes and text.
#[derive(Clone, Debug)]
pub struct Files {
    pub tokenizer: Vec<u8>,
    pub template: String,
    pub generation: String,
    pub config: String,
}

impl Files {
    /// Read the files of the folder `dir`.
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

/// The ids of the markers that a reply or a mode uses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Marks {
    pub think_open: u32,
    pub think_close: u32,
    pub call_open: u32,
    pub call_close: u32,
    pub audio_start: u32,
    pub mixed_start: u32,
    /// The markers around a video and around each of its pairs (spec 11.11, "The video of a request").
    pub vision_start: u32,
    pub vision_end: u32,
}

/// The LFM3 family library of one model folder.
pub struct Lfm3Family {
    kind: Kind,
    chat: Chat,
    /// The token of each letter with a leading space, `A` first. This is the menu of a plain read.
    letters: Vec<u32>,
    /// The token of each bare letter, `A` first. A queue read uses these: its letter follows a quote.
    bare_letters: Vec<u32>,
    /// The tokens that end a reply: `eos_token_id` of `generation_config.json`.
    stop: Vec<u32>,
    marks: Marks,
    /// The placeholder token of each media kind.
    placeholders: [(MediaKind, u32); 4],
    template_id: String,
    /// The reader of the objects of this machine. A family without one refuses each media part.
    objects: Option<Arc<dyn Objects>>,
    /// The codes of the speech outputs of the service, for an `assistant` audio part (spec 11.11, "Read it back").
    codes: std::sync::OnceLock<Arc<dyn nagaya_council::family::OutputCodes>>,
    /// The prepared videos of the last few video parts, by content id. A render, its rows and a read each need them,
    /// and a decode is slow.
    videos: std::sync::Mutex<Vec<(String, Arc<crate::media::video::Video>)>>,
}

/// The prepared videos that a family keeps.
const VIDEOS_KEPT: usize = 4;

/// Decode and prepare a video part that `param` names.
#[cfg(feature = "video")]
fn prepare_video(bytes: &[u8], param: &str) -> Result<crate::media::video::Video, FamilyError> {
    crate::media::video::prepare(bytes).map_err(|e| FamilyError::client_param(param, format!("{param} {e}.")))
}

/// A build with no feature `video` reads no video. `media_kind` refuses a video part first, so this is a guard.
#[cfg(not(feature = "video"))]
fn prepare_video(_bytes: &[u8], param: &str) -> Result<crate::media::video::Video, FamilyError> {
    Err(FamilyError::client_param(
        param,
        "this build of the server reads no video: it has no feature `video`.",
    ))
}

impl std::fmt::Debug for Lfm3Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lfm3Family")
            .field("kind", &self.kind)
            .field("template_id", &self.template_id)
            .finish()
    }
}

/// A client error from a library error: the text or the messages are at fault.
pub(crate) fn client(e: Error) -> FamilyError {
    FamilyError::client(e.to_string())
}

/// A library error for a failure that the client did not cause.
pub(crate) fn internal(e: Error) -> FamilyError {
    FamilyError::internal(e.to_string())
}

/// An error of the tokens that a client sent: an id outside the vocabulary, or bytes that are not whole text.
fn tokens_error(e: Error) -> FamilyError {
    match e {
        Error::TokenId { .. } | Error::Bytes(_) => FamilyError::client_param("tokens", e.to_string()),
        other => internal(other),
    }
}

/// A refusal of the template or of the escapes on client input is the client's. Any other error is ours.
pub(crate) fn refusal(e: Error) -> FamilyError {
    match e {
        Error::Template(_) | Error::Reserved { .. } | Error::Json(_) => client(e),
        other => internal(other),
    }
}

/// The model of a folder, from the first entry of `architectures` in `config.json`.
fn kind_of(config: &str) -> Result<Kind, Error> {
    let config: serde_json::Value =
        serde_json::from_str(config).map_err(|e| Error::Model(format!("config.json is not JSON: {e}")))?;
    match config["architectures"][0].as_str() {
        Some("Lfm3OmniForConditionalGeneration") => Ok(Kind::Omni),
        Some("Lfm3ForCausalLM") => Ok(Kind::Text),
        other => Err(Error::Model(format!(
            "config.json names the architecture {other:?}, which is not LFM3-Omni-2B or LFM3-1.7B"
        ))),
    }
}

/// The stop tokens: `eos_token_id` of `generation_config.json`, one id or a list. They must be tokens of the
/// vocabulary.
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
        .into_iter()
        .map(|v| {
            v.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < tokenizer.len())
                .ok_or_else(|| Error::Model(format!("eos_token_id {v} is not a token of tokenizer.json")))
        })
        .collect::<Result<_, _>>()?;
    if stop.is_empty() {
        return Err(Error::Model("generation_config.json names no eos_token_id".to_string()));
    }
    Ok(stop)
}

impl Lfm3Family {
    /// Build the family from a model folder. The registry calls this with the folder that the profile names.
    pub fn from_dir(dir: &Path) -> Result<Lfm3Family, Error> {
        Lfm3Family::new(&Files::read(dir)?)
    }

    /// Build the family from the files of a folder, and compute its template id from its test renders.
    pub fn new(files: &Files) -> Result<Lfm3Family, Error> {
        let kind = kind_of(&files.config)?;
        let tokenizer = Arc::new(Tokenizer::from_bytes(&files.tokenizer)?);
        let (letters, bare_letters) = letter_tokens(&tokenizer)?;
        let id = |text: &str| {
            tokenizer
                .token_id(text)
                .ok_or_else(|| Error::Model(format!("the vocabulary has no {text}")))
        };
        let marks = Marks {
            think_open: id("<think>")?,
            think_close: id(THINK_CLOSE)?,
            call_open: id("<tool_call>")?,
            call_close: id("</tool_call>")?,
            audio_start: id("<|audio_start|>")?,
            mixed_start: id("<|mixed_start|>")?,
            vision_start: id("<|vision_start|>")?,
            vision_end: id("<|vision_end|>")?,
        };
        let placeholders = [
            MediaKind::Image,
            MediaKind::Audio,
            MediaKind::AudioOut,
            MediaKind::Video,
        ]
        .map(|k| id(k.placeholder()).map(|t| (k, t)));
        let [a, b, c, d] = placeholders;
        let placeholders = [a?, b?, c?, d?];
        let stop = stop_of(&files.generation, &tokenizer)?;
        let mut family = Lfm3Family {
            kind,
            chat: Chat::new(&files.template, tokenizer)?,
            letters,
            bare_letters,
            stop,
            marks,
            placeholders,
            template_id: String::new(),
            objects: None,
            videos: std::sync::Mutex::new(Vec::new()),
            codes: std::sync::OnceLock::new(),
        };
        let council = council_hash(&test_set::renders(&family));
        family.template_id = template_id(&council, FAMILY, &hash16(files.template.as_bytes()));
        Ok(family)
    }

    /// The family with a reader of the objects of this machine. The process gives it, so that a render can read
    /// the media parts of a context.
    pub fn with_objects(mut self, objects: Arc<dyn Objects>) -> Lfm3Family {
        self.objects = Some(objects);
        self
    }

    /// Which of the two models this folder holds.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The tokenizer hash: `sha256:` and the 64 hex digits of the file (spec 12.3). The registry compares it with the
    /// `tokenizer_hash` of the engine identity.
    pub fn tokenizer_hash(&self) -> &str {
        self.tokenizer().hash()
    }

    /// The chat template text of the folder. The identity's `template` is this text (spec 12.1).
    pub fn template(&self) -> &str {
        self.chat.source()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.chat.tokenizer()
    }

    /// The chat template of the folder, with its escapes.
    pub fn chat(&self) -> &Chat {
        &self.chat
    }

    /// The tokenizer, shared with a reply parser that outlives a borrow.
    pub(crate) fn shared_tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(self.chat.tokenizer())
    }

    /// The tokens that end a reply.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// The number of letters that a question can use: one token for each, no two the same.
    pub fn letter_count(&self) -> usize {
        self.letters.len()
    }

    /// The token that opens a reply of `mode`, after the whole generation prompt. A text reply has none. `lfm3` does
    /// not speak, so a speech mode is a client error that names `output`.
    pub fn mode_marker(&self, mode: Mode) -> Result<Option<u32>, FamilyError> {
        match (mode, self.kind) {
            (Mode::Text, _) => Ok(None),
            (_, Kind::Text) => Err(FamilyError::client_param(
                "output",
                "the served model lfm3 writes text only. Ask lfm3-omni for speech.",
            )),
            (Mode::Speech, Kind::Omni) => Ok(Some(self.marks.audio_start)),
            (Mode::Interleaved, Kind::Omni) => Ok(Some(self.marks.mixed_start)),
        }
    }

    /// The tokens that the sampler of a text reply never picks: the markers of the speech modes.
    pub fn text_suppressed(&self) -> [u32; 2] {
        [self.marks.audio_start, self.marks.mixed_start]
    }

    /// Whether a context renders with thinking. Omni does not reason: its served model does not declare `thinking`,
    /// and a context with `thinking: true` is a 400 that names `thinking` (spec 11.11, gogi "Contexts").
    pub(crate) fn thinking(&self, context: &Context) -> Result<bool, FamilyError> {
        match self.kind {
            Kind::Text => Ok(context.thinking),
            Kind::Omni if context.thinking => Err(FamilyError::client_param(
                "thinking",
                "nothing ran: lfm3-omni does not think; send thinking: false",
            )),
            Kind::Omni => Ok(false),
        }
    }

    /// Render messages and tools with the chat template.
    pub(crate) fn run(
        &self,
        messages: &[Json],
        tools: &[Json],
        thinking: bool,
        generation_prompt: bool,
    ) -> Result<Rendered, FamilyError> {
        self.chat
            .render(messages, tools, thinking, generation_prompt)
            .map_err(refusal)
    }

    /// Render messages and an open last message (gogi "Continuations"): the generation prompt of the messages before
    /// it, then the open text as the model writes it. With thinking on, the prompt ends with `<think>`. The reasoning
    /// follows, and a `content` member closes it with `</think>`. With thinking off, the prompt closed it. This is the
    /// text that the template writes for the message when it is closed, up to its end.
    pub(crate) fn run_open(
        &self,
        before: &[Json],
        open: &Message,
        tools: &[Json],
        thinking: bool,
    ) -> Result<Rendered, FamilyError> {
        let prompt = self.run(before, tools, thinking, true)?;
        let mut escaped = prompt.escaped;
        if thinking {
            escaped.push_str(&self.escaped(open.reasoning.as_deref().unwrap_or(""))?);
            if open.content.is_some() {
                escaped.push_str(THINK_CLOSE);
            }
        }
        // An open message's content is a string: gogi's check_open refuses parts.
        escaped.push_str(&self.escaped(open.text().unwrap_or(""))?);
        self.chat.finish(escaped).map_err(refusal)
    }

    /// The seam check of the reference's `prefill` (gogi "Continuations"): each open field that is not empty must tokenize on its
    /// own. The rendering up to the start of the field must be a token prefix of the rendering through its end.
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

    /// Client text as escaped content.
    pub(crate) fn escaped(&self, text: &str) -> Result<String, FamilyError> {
        self.tokenizer().escape(text).map_err(refusal)
    }

    /// The tokens of escaped text.
    pub(crate) fn encode(&self, escaped: &str) -> Result<Vec<u32>, FamilyError> {
        self.tokenizer().encode_escaped(escaped).map_err(refusal)
    }

    /// The control text of `text`, found at `place`.
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

    /// The rendering of a context in the token space of the template: each media part is one placeholder token.
    pub(crate) fn render_placeholders(
        &self,
        context: &Context,
        generation_prompt: bool,
    ) -> Result<Rendering, FamilyError> {
        context.check()?;
        let omni = self.kind == Kind::Omni;
        let thinking = self.thinking(context)?;
        let tools = messages::tools_json(&context.tools);
        let msgs = messages::messages_json(&context.messages, omni)?;
        let open = context.open();
        let base = match open {
            Some(open) => {
                let before = &msgs[..msgs.len() - 1];
                self.check_seams(before, open, &tools, thinking)?;
                self.run_open(before, open, &tools, thinking)?
            }
            None => self.run(&msgs, &tools, thinking, false)?,
        };
        let end = base.tokens.len();
        // An open message takes the place of the generation prompt (gogi "Continuations").
        let tokens = if generation_prompt && open.is_none() {
            let whole = self.run(&msgs, &tools, thinking, true)?.tokens;
            if !whole.starts_with(&base.tokens) {
                return Err(FamilyError::internal(
                    "the generation prompt does not extend the rendering of the context. The chat template changed shape.",
                ));
            }
            whole
        } else {
            base.tokens.clone()
        };
        // A marked message is a boundary when its rendering is a prefix of the whole (gogi "Contexts"). It must also
        // add tokens, and it comes after the boundary before it.
        let last = msgs.len() - 1;
        let mut boundaries: Vec<Boundary> = Vec::new();
        for (i, message) in context.messages.iter().enumerate().take(last) {
            if !message.snap {
                continue;
            }
            let part = self.run(&msgs[..=i], &tools, thinking, false)?;
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
            tokens,
            end,
            boundaries,
            control_text,
            media: Vec::new(),
        })
    }

    /// The media parts of a context, in the order of the rendering. The caller gives a count of positions for each,
    /// in this order, to [`Lfm3Family::render_media`]. A part that this model cannot read is a client error that
    /// names it.
    pub fn media_parts(&self, context: &Context) -> Result<Vec<MediaPart>, FamilyError> {
        messages::media_parts(self.kind == Kind::Omni, &context.messages)
    }

    /// Render a context with its media. `counts` holds the count of positions of each part of
    /// [`Lfm3Family::media_parts`], in that order. Each placeholder token of the template becomes that many
    /// positions, as Liquid's processor writes them (spec 11.11).
    pub fn render_media(
        &self,
        context: &Context,
        generation_prompt: bool,
        counts: &[usize],
    ) -> Result<MediaRendering, FamilyError> {
        let parts = self.media_parts(context)?;
        let rendering = self.render_placeholders(context, generation_prompt)?;
        media::expand(self, rendering, &parts, counts)
    }

    /// The placeholder token of a media kind.
    pub(crate) fn placeholder_token(&self, kind: MediaKind) -> u32 {
        self.placeholders
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, t)| *t)
            .expect("the family has a placeholder of each kind")
    }

    /// The media type of a part in its recipe item (tag 9). A video part also names the versions of the decoder,
    /// because a new decoder can give other rows from the same bytes (spec 11.11, Decided 2026-11-08).
    pub(crate) fn recipe_mime(&self, part: &MediaPart) -> String {
        // The speech of an assistant part enters the recipe as its codes do (spec 11.11, "Read it back").
        if part.kind == MediaKind::AudioOut
            && let Ok(codes) = self.output_codes(part)
        {
            let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
            return format!("{};codes={}", part.mime, gogi_proto::canon::content_id(&bytes));
        }
        #[cfg(feature = "video")]
        if part.kind == MediaKind::Video {
            return format!("{};{}", part.mime, crate::media::decode::versions());
        }
        // A WAV that the resampler made names the resampler too, as a video names its decoder.
        #[cfg(feature = "video")]
        if part.kind == MediaKind::Audio
            && let Ok(bytes) = self.part_bytes(part)
            && crate::media::audio::read_wav(&bytes).is_err()
        {
            return format!("{};{}", part.mime, crate::media::decode::resampler_version());
        }
        part.mime.clone()
    }

    /// The prepared video of the object `id`, from the family's few kept videos or from a decode. `param` names the
    /// part in an error.
    fn video(&self, id: &str, param: &str) -> Result<Arc<crate::media::video::Video>, FamilyError> {
        if let Some((_, video)) = self
            .videos
            .lock()
            .expect("the video lock")
            .iter()
            .find(|(k, _)| k == id)
        {
            return Ok(Arc::clone(video));
        }
        let bytes = self.object_bytes(id, param)?;
        let video = Arc::new(prepare_video(&bytes, param)?);
        let mut kept = self.videos.lock().expect("the video lock");
        kept.retain(|(k, _)| k != id);
        kept.push((id.to_owned(), Arc::clone(&video)));
        if kept.len() > VIDEOS_KEPT {
            kept.remove(0);
        }
        Ok(video)
    }

    /// The pieces of a video in the token space of the template: for each pair, its time, its markers, its video
    /// tokens and its run of sound (spec 11.11, "The video of a request"). It copies no rows.
    fn video_pieces(&self, video: &crate::media::video::Video) -> Result<Vec<media::Piece>, FamilyError> {
        let mut pieces = Vec::new();
        for (k, (time, pair)) in video.times.iter().zip(&video.pairs).enumerate() {
            pieces.push(media::Piece::Tokens(
                self.tokenizer().encode_content(time).map_err(internal)?,
            ));
            pieces.push(media::Piece::Tokens(vec![self.marks.vision_start]));
            pieces.push(media::Piece::Span {
                token: self.placeholder_token(MediaKind::Video),
                count: pair.positions(),
            });
            pieces.push(media::Piece::Tokens(vec![self.marks.vision_end]));
            if let Some((_, bounds)) = &video.sound
                && bounds[k + 1] > bounds[k]
            {
                pieces.push(media::Piece::Span {
                    token: self.placeholder_token(MediaKind::Audio),
                    count: bounds[k + 1] - bounds[k],
                });
            }
        }
        Ok(pieces)
    }

    /// The rows of each span of a video, in the order of its pieces: the rows of each pair, and the run of each pair
    /// that has sound.
    fn video_rows(&self, video: &crate::media::video::Video) -> Vec<crate::media::MediaRows> {
        let clip = video
            .sound
            .as_ref()
            .map(|(clip, bounds)| (Arc::new(clip.clone()), bounds));
        let mut rows = Vec::new();
        for (k, pair) in video.pairs.iter().enumerate() {
            rows.push(crate::media::MediaRows::Image(pair.clone()));
            if let Some((clip, bounds)) = &clip
                && bounds[k + 1] > bounds[k]
            {
                rows.push(crate::media::MediaRows::Run(crate::media::video::AudioRun {
                    clip: Arc::clone(clip),
                    start: bounds[k],
                    count: bounds[k + 1] - bounds[k],
                }));
            }
        }
        rows
    }

    /// The codes of the speech of an `assistant` audio part, with the end frame: the kept codes of its output. A part
    /// whose codes the service does not hold is a 404 that names it (spec 11.11, "Read it back").
    fn output_codes(&self, part: &MediaPart) -> Result<Vec<u32>, FamilyError> {
        let param = part_param(part);
        let codes = self
            .codes
            .get()
            .and_then(|book| book.codes(&part.id))
            .ok_or_else(|| FamilyError::NotHeld {
                param: param.clone(),
                message: format!(
                    "{param} names {}, and this server holds no codes of that output: it is not an output of this \
                 served model, or the server dropped its codes. Send the reply's text instead.",
                    part.id
                ),
            })?;
        if !codes.len().is_multiple_of(crate::media::SPEECH_CODES) || codes.iter().any(|c| *c > crate::media::END_CODE)
        {
            return Err(FamilyError::internal(format!(
                "the kept codes of {} are not frames of 12 codes below 2017",
                part.id
            )));
        }
        let mut codes = codes;
        codes.extend([crate::media::END_CODE; crate::media::SPEECH_CODES]);
        Ok(codes)
    }

    /// The layout of a media part: one span of its count for an image or a clip, from its header, and the pieces of
    /// a video.
    pub(crate) fn layout(&self, part: &MediaPart) -> Result<Vec<media::Piece>, FamilyError> {
        if part.kind == MediaKind::AudioOut {
            return Ok(vec![media::Piece::Span {
                token: self.placeholder_token(MediaKind::AudioOut),
                count: self.output_codes(part)?.len() / crate::media::SPEECH_CODES,
            }]);
        }
        if part.kind == MediaKind::Video {
            let video = self.video(&part.id, &part_param(part))?;
            return self.video_pieces(&video);
        }
        let bytes = self.part_bytes(part)?;
        let count = crate::media::positions(&bytes, &part.mime).map_err(|e| media_error(part, e))?;
        Ok(vec![media::Piece::Span {
            token: self.placeholder_token(part.kind),
            count,
        }])
    }

    /// The rows of each span of a media part, in order: one for an image or a clip, and one for each pair and each
    /// run of a video.
    fn part_rows(&self, part: &MediaPart) -> Result<Vec<crate::media::MediaRows>, FamilyError> {
        if part.kind == MediaKind::AudioOut {
            return Ok(vec![crate::media::MediaRows::Codes(self.output_codes(part)?)]);
        }
        if part.kind == MediaKind::Video {
            let video = self.video(&part.id, &part_param(part))?;
            return Ok(self.video_rows(&video));
        }
        let bytes = self.part_bytes(part)?;
        let header = crate::media::positions(&bytes, &part.mime).map_err(|e| media_error(part, e))?;
        let prepared = crate::media::prepare(&bytes, &part.mime).map_err(|e| media_error(part, e))?;
        if prepared.positions() != header {
            return Err(FamilyError::internal(format!(
                "media part {} of message {} has {} positions from its header and {} from its rows",
                part.part,
                part.message,
                header,
                prepared.positions()
            )));
        }
        Ok(vec![prepared])
    }

    /// Render a context with its media, and prepare the rows of each media part for the engine.
    /// The header of each part gives its count of positions, and the whole part must give the same count.
    pub fn prepare(&self, context: &Context, generation_prompt: bool) -> Result<Prepared, FamilyError> {
        let parts = self.media_parts(context)?;
        let mut rows = Vec::new();
        let mut layouts = Vec::with_capacity(parts.len());
        for part in &parts {
            layouts.push(self.layout(part)?);
            rows.extend(self.part_rows(part)?);
        }
        let rendering = self.render_placeholders(context, generation_prompt)?;
        let MediaRendering { rendering, spans } = media::expand_layouts(self, rendering, &parts, &layouts)?;
        let inputs = spans
            .into_iter()
            .zip(rows)
            .map(|(span, rows)| {
                let matches = matches!(
                    (span.kind, &rows),
                    (MediaKind::Image | MediaKind::Video, crate::media::MediaRows::Image(_))
                        | (MediaKind::Audio, crate::media::MediaRows::Audio(_))
                        | (MediaKind::Video, crate::media::MediaRows::Run(_))
                );
                assert!(matches, "a span of {:?} got rows of another kind", span.kind);
                MediaInput { span, rows }
            })
            .collect();
        Ok(Prepared { rendering, inputs })
    }

    /// The bytes of a media part from the object reader. A part that the family cannot read yet, an absent reader
    /// and an absent object are client errors that name the part.
    fn part_bytes(&self, part: &MediaPart) -> Result<Vec<u8>, FamilyError> {
        let param = part_param(part);
        if part.kind == MediaKind::AudioOut {
            return Err(FamilyError::client_param(
                param,
                "this build cannot read an assistant audio part yet. Its positions come from the codes of a speech \
                 output, which a later step keeps. Remove the part.",
            ));
        }
        let Some(objects) = &self.objects else {
            return Err(FamilyError::client_param(
                param,
                "this server gives the LFM3 family no objects, so it cannot read a media part. Send text only.",
            ));
        };
        objects.bytes(&part.id).ok_or_else(|| {
            FamilyError::client_param(
                param,
                format!(
                    "the object {} is not on this server. Upload it, then send the request again.",
                    part.id
                ),
            )
        })
    }

    /// The bytes of the object `id` that the request member `param` names, such as `case_media[0]`. An absent reader
    /// and an absent object are client errors that name it.
    fn object_bytes(&self, id: &str, param: &str) -> Result<Vec<u8>, FamilyError> {
        let Some(objects) = &self.objects else {
            return Err(FamilyError::client_param(
                param,
                "this server gives the LFM3 family no objects, so it cannot read a media part. Send text only.",
            ));
        };
        objects.bytes(id).ok_or_else(|| {
            FamilyError::client_param(
                param,
                format!("the object {id} is not on this server. Upload it, then send the request again."),
            )
        })
    }

    /// The media kind of a placeholder token, or none.
    pub(crate) fn placeholder_kind(&self, token: u32) -> Option<MediaKind> {
        self.placeholders.iter().find(|(_, t)| *t == token).map(|(k, _)| *k)
    }
}

/// The request member of a media part, for an error.
fn part_param(part: &MediaPart) -> String {
    format!("messages[{}].content[{}]", part.message, part.part)
}

/// The error of a media part whose bytes the preprocessing refused. A media type that the family does not read is a
/// defect: `media_parts` refuses it first.
fn media_error(part: &MediaPart, e: crate::media::MediaError) -> FamilyError {
    let param = part_param(part);
    match e {
        crate::media::MediaError::Mime(_) => FamilyError::internal(format!("{param} {e}")),
        other => FamilyError::client_param(param.clone(), format!("{param} {other}.")),
    }
}

/// The tokens of the letters `A` to `Z`, with a leading space and bare. The run stops at the first letter that fails.
/// A letter fails when it is not one token in both forms, or when an earlier letter has its token. A run of fewer
/// than two letters is an error.
fn letter_tokens(tokenizer: &Tokenizer) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let one = |text: &str| -> Result<Option<u32>, Error> {
        Ok(match tokenizer.encode_content(text)?.as_slice() {
            [id] => Some(*id),
            _ => None,
        })
    };
    let (mut spaced, mut bare): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    for letter in 'A'..='Z' {
        match (one(&format!(" {letter}"))?, one(&letter.to_string())?) {
            (Some(s), Some(b)) if !spaced.contains(&s) && !bare.contains(&b) => {
                spaced.push(s);
                bare.push(b);
            }
            _ => break,
        }
    }
    if spaced.len() < 2 {
        return Err(Error::Tokenizer(format!(
            "only {} of the letters A to Z are one token each. A question needs at least two.",
            spaced.len()
        )));
    }
    Ok((spaced, bare))
}

impl Family for Lfm3Family {
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
        // spec 11.11, "The choices of the LFM3 family": the header of each part gives its count of positions.
        let parts = self.media_parts(context)?;
        if parts.is_empty() {
            return self.render_placeholders(context, generation_prompt);
        }
        let layouts = parts
            .iter()
            .map(|part| self.layout(part))
            .collect::<Result<Vec<_>, FamilyError>>()?;
        let rendering = self.render_placeholders(context, generation_prompt)?;
        Ok(media::expand_layouts(self, rendering, &parts, &layouts)?.rendering)
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        read::decision_read(self, input, &[])
    }

    fn decision_read_with_media(
        &self,
        input: &DecisionInput<'_>,
        case_media: &[gogi_proto::objects::ObjectRef],
    ) -> Result<Box<dyn ReadPlan>, FamilyError> {
        if self.kind == Kind::Text && !case_media.is_empty() {
            return Err(FamilyError::client_param(
                "case_media[0]",
                "the served model lfm3 reads text only. Send the decision with case_media to lfm3-omni.",
            ));
        }
        read::decision_read(self, input, case_media)
    }

    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
        read::raw_read(self, letters, suffixes)
    }

    fn reply_parser(&self, start: ReplyStart, tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        Ok(reply::reply_parser(self, start, tools))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        check_text(text).map_err(client)?;
        self.tokenizer().encode_content(text).map_err(internal)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        check_text(text).map_err(client)?;
        self.tokenizer().encode(text).map_err(internal)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        self.tokenizer().decode_whole(tokens).map_err(tokens_error)
    }

    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError> {
        tokens
            .iter()
            .map(|&id| {
                let bytes = self.tokenizer().token_bytes(id).map_err(tokens_error)?;
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            })
            .collect()
    }

    fn template_id(&self) -> &str {
        &self.template_id
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer().len()
    }

    /// `lfm3` thinks before its reply. Omni does not reason.
    fn thinks(&self) -> bool {
        self.kind == Kind::Text
    }

    fn media_inputs(&self) -> Vec<String> {
        match self.kind {
            Kind::Omni => media_inputs().iter().map(|mime| (*mime).to_owned()).collect(),
            Kind::Text => Vec::new(),
        }
    }

    fn media_rows(&self, rendering: &Rendering) -> Result<Vec<nagaya_council::family::MediaInput>, FamilyError> {
        self.inputs_of(&rendering.media)
    }

    /// Both models take latent rows at positions of `<|pad|>`, the `pad_token_id` of each config.
    fn latent_record(&self, first: usize, count: usize, width: usize) -> Option<(u32, Vec<u8>)> {
        let pad = self.tokenizer().token_id(PAD)?;
        Some((pad, latent_record(first, count, width)))
    }

    fn read_codes(&self, codes: Arc<dyn nagaya_council::family::OutputCodes>) {
        // A service gives its book once, at its attach.
        let _ = self.codes.set(codes);
    }
}

/// The media types that Omni reads (spec 11.11, "The served LFM3 models"), and the video types of a build with the
/// feature `video` ("The video of a request").
pub fn media_inputs() -> &'static [&'static str] {
    if cfg!(feature = "video") {
        &["image/png", "image/jpeg", "audio/wav", "video/mp4", "video/webm"]
    } else {
        &["image/png", "image/jpeg", "audio/wav"]
    }
}

/// The bytes of one record of `meta` (engine-abi "The LFM3 vision binding"): the kind, the count of positions, the
/// first position as a `u64`, and four words of the shape.
pub const RECORD_BYTES: usize = 32;

/// The record of a span in `meta`, little-endian. An image has the kind 1 and its grid. A clip has the kind 2 and its
/// count of mel frames. `nagaya-mk-lfm3-bmg` reads the same layout (`media::decode`).
pub fn media_record(first: usize, rows: &crate::media::MediaRows) -> Vec<u8> {
    let (kind, shape) = match rows {
        crate::media::MediaRows::Image(image) => (1u32, [image.grid[0], image.grid[1], image.grid[2], 0]),
        crate::media::MediaRows::Audio(audio) => (2u32, [audio.frames, 0, 0, 0]),
        // A run of a clip: the frames of the whole clip, and the first encoder row of the run (spec 11.11).
        crate::media::MediaRows::Run(run) => (3u32, [run.clip.frames, run.start, 0, 0]),
        // Speech: 12 codes for each frame (spec 11.11, "Read it back").
        crate::media::MediaRows::Codes(_) => (4u32, [crate::media::SPEECH_CODES, 0, 0, 0]),
    };
    let word = |value: usize| u32::try_from(value).expect("a shape of a media part fits in 32 bits");
    let mut record = Vec::with_capacity(RECORD_BYTES);
    record.extend(kind.to_le_bytes());
    record.extend(word(rows.positions()).to_le_bytes());
    record.extend((first as u64).to_le_bytes());
    for value in shape {
        record.extend(word(value).to_le_bytes());
    }
    record
}

/// The placeholder of a latent position (engine-abi "Latent rows of LFM3").
pub const PAD: &str = "<|pad|>";

/// The record of a latent span in `meta`: the kind 5, then the width of a row and three 0. `nagaya-mk-lfm3-bmg` reads
/// the same layout (`media::decode`).
pub fn latent_record(first: usize, count: usize, width: usize) -> Vec<u8> {
    let word = |value: usize| u32::try_from(value).expect("a latent span fits in 32 bits");
    let mut record = Vec::with_capacity(RECORD_BYTES);
    record.extend(5u32.to_le_bytes());
    record.extend(word(count).to_le_bytes());
    record.extend((first as u64).to_le_bytes());
    for value in [width, 0, 0, 0] {
        record.extend(word(value).to_le_bytes());
    }
    record
}

impl Lfm3Family {
    /// The record and the rows of each span of `spans`, in order. The spans of a video come in the order of its
    /// layout, so the n-th span of a video takes the n-th rows of its layout.
    pub(crate) fn inputs_of(
        &self,
        spans: &[nagaya_council::family::MediaSpan],
    ) -> Result<Vec<nagaya_council::family::MediaInput>, FamilyError> {
        let mut out = Vec::with_capacity(spans.len());
        let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        // The rows of each video, made once for all its spans.
        let mut made: std::collections::HashMap<&str, Vec<crate::media::MediaRows>> = std::collections::HashMap::new();
        for span in spans {
            // The speech of an assistant part reads the kept codes of its output (`recipe_mime`).
            if span.mime.starts_with("audio/wav;codes=") {
                let part = MediaPart {
                    message: 0,
                    part: 0,
                    kind: MediaKind::AudioOut,
                    id: span.id.clone(),
                    mime: "audio/wav".to_owned(),
                };
                let rows = crate::media::MediaRows::Codes(self.output_codes(&part)?);
                out.push(nagaya_council::family::MediaInput {
                    span: span.clone(),
                    meta: media_record(span.first, &rows),
                    rows: rows.bytes().into(),
                });
                continue;
            }
            // The recipe media type of a video span starts with its own type (`recipe_mime`).
            if !span.mime.starts_with("video/") {
                out.push(self.media_input(span)?);
                continue;
            }
            if !made.contains_key(span.id.as_str()) {
                let video = self.video(&span.id, &format!("the object {}", span.id))?;
                made.insert(span.id.as_str(), self.video_rows(&video));
            }
            let rows = &made[span.id.as_str()];
            let n = seen.entry(span.id.as_str()).or_insert(0);
            let rows = &rows[*n % rows.len()];
            *n += 1;
            if rows.positions() != span.count {
                return Err(FamilyError::internal(format!(
                    "the span of {} has {} positions, and its rows give {}",
                    span.id,
                    span.count,
                    rows.positions()
                )));
            }
            let values: Vec<u8> = rows.bytes();
            out.push(nagaya_council::family::MediaInput {
                span: span.clone(),
                meta: media_record(span.first, rows),
                rows: values.into(),
            });
        }
        Ok(out)
    }

    /// The record and the rows of one span of a rendering. The object can leave the store after the render, so an
    /// absent object is a client error. A count that the object does not give is a defect of the caller.
    fn media_input(
        &self,
        span: &nagaya_council::family::MediaSpan,
    ) -> Result<nagaya_council::family::MediaInput, FamilyError> {
        let objects = self.objects.as_ref().ok_or_else(|| {
            FamilyError::internal("a rendering holds a media span, and the family has no object reader")
        })?;
        let bytes = objects.bytes(&span.id).ok_or_else(|| {
            FamilyError::client(format!(
                "nothing ran: the object {} left the server after the request named it. Upload it, then send the \
                 request again.",
                span.id
            ))
        })?;
        // The recipe media type can name the resampler after a `;`: the rows read the type itself.
        let mime = span.mime.split(';').next().unwrap_or(&span.mime);
        let rows = crate::media::prepare(&bytes, mime)
            .map_err(|e| FamilyError::client(format!("nothing ran: the object {} {e}.", span.id)))?;
        if rows.positions() != span.count {
            return Err(FamilyError::internal(format!(
                "the media span of {} has {} positions, and its object gives {}",
                span.id,
                span.count,
                rows.positions()
            )));
        }
        let values: Vec<u8> = rows.values().iter().flat_map(|value| value.to_le_bytes()).collect();
        Ok(nagaya_council::family::MediaInput {
            span: span.clone(),
            meta: media_record(span.first, &rows),
            rows: values.into(),
        })
    }
}
