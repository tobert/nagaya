//! The Gemma 4 family library for the council (spec 11.3, 11.3a): nagaya's family interface on this crate.
//!
//! A context renders with Google's chat template (`crate::chat`). A decision puts a spec, a case and a letter menu
//! after the context. This is mkviolet's rendering (`src/template.rs`, `mkv-letters-1`) in Gemma 4's chat format with
//! thinking off:
//!
//! ```text
//! context (the chat template, no generation prompt), or <bos> for a decision with no context
//! <|turn>user\n{instructions}\n\nAnswer each question below about the {label} in the next message. ...
//!   Question {n}. {instructions}\n{A}. {option}: {means}\n ...<turn|>\n                            spec layer
//! <|turn>user\n{label}:\n{case}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Answer to question     case
//!  {n} ({first sentence of the question}):                                 one read for each question
//! ```
//!
//! The letter token is the letter with a leading space. Template text is tokenized with special tokens on, and client
//! text with them off. A queue read gives the case as the result of the context's last call. The model then writes
//! the call to the queue tool in the template's form (mkviolet `queue_reads`):
//!
//! ```text
//! context messages, then a tool message that holds the case, with the generation prompt
//! <channel|> (thinking on only), then <|tool_call>call:{queue}{answers:{
//! {id}:<|"|>  then the bare letter token, then <|"|>, and `,` before the next question
//! ```
//!
//! Where mkviolet and gogi's contract disagree, gogi wins. Each such place has a comment with the word "gogi".

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

use crate::chat::{Chat, Rendered};
use crate::tokenizer::{Tokenizer, check_text};
use crate::{CHAT_TEMPLATE, Error};

/// The name of the family in the identity and in the template id.
pub const FAMILY: &str = "gemma4";

/// The marker that closes the thought channel. An open message with a `content` member writes it after the reasoning.
const CHANNEL_CLOSE: &str = "<channel|>";

/// The file of the tokenizer in a checkpoint directory.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The Gemma 4 family library.
pub struct Gemma4Family {
    chat: Chat,
    /// The token of each letter with a leading space, `A` first. This is the menu of a plain read.
    letters: Vec<u32>,
    /// The token of each bare letter, `A` first. A queue read uses these: its letter follows a quote.
    bare_letters: Vec<u32>,
    /// The tokens that end a reply: `<turn|>`, `<eos>`, `<|tool_response>` and `<|turn>` (mkviolet `mk_api.rs`).
    stop: Vec<u32>,
    /// The markers that split a reply into its parts.
    marks: reply::Marks,
    template_id: String,
}

impl std::fmt::Debug for Gemma4Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4Family")
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

impl Gemma4Family {
    /// Build the family from a checkpoint directory. It reads `tokenizer.json` there (spec 12.3). The registry
    /// calls this with the directory that the profile names.
    pub fn from_dir(dir: &Path) -> Result<Gemma4Family, Error> {
        Gemma4Family::from_tokenizer_file(&dir.join(TOKENIZER_FILE))
    }

    /// Build the family from a tokenizer file, and compute its template id from its test renders.
    pub fn from_tokenizer_file(path: &Path) -> Result<Gemma4Family, Error> {
        Gemma4Family::new(Tokenizer::from_file(path)?)
    }

    /// Build the family from a tokenizer.
    pub fn new(tokenizer: Tokenizer) -> Result<Gemma4Family, Error> {
        let tokenizer = Arc::new(tokenizer);
        let (letters, bare_letters) = letter_tokens(&tokenizer)?;
        let id = |text: &str| {
            tokenizer
                .token_id(text)
                .ok_or_else(|| Error::Tokenizer(format!("the vocabulary has no {text}. Use a Gemma 4 tokenizer.")))
        };
        let stop = vec![id("<turn|>")?, id("<eos>")?, id("<|tool_response>")?, id("<|turn>")?];
        let marks = reply::Marks {
            channel_open: id("<|channel>")?,
            channel_close: id("<channel|>")?,
            tool_open: id("<|tool_call>")?,
            tool_close: id("<tool_call|>")?,
        };
        let mut family = Gemma4Family {
            chat: Chat::new(tokenizer)?,
            letters,
            bare_letters,
            stop,
            marks,
            template_id: String::new(),
        };
        let council = council_hash(&test_set::renders(&family));
        family.template_id = template_id(&council, FAMILY, &hash16(CHAT_TEMPLATE.as_bytes()));
        Ok(family)
    }

    /// The tokenizer hash: `sha256:` and the 64 hex digits of the file (spec 12.3). The registry compares it with the
    /// `tokenizer_hash` of the engine identity.
    pub fn tokenizer_hash(&self) -> &str {
        self.tokenizer().hash()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.chat.tokenizer()
    }

    /// The tokenizer, shared with a reply parser that outlives a borrow.
    pub(crate) fn shared_tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(self.chat.tokenizer())
    }

    /// The number of letters that a question can use: one token for each, no two the same.
    pub fn letter_count(&self) -> usize {
        self.letters.len()
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

    /// Render messages and an open last message (spec 11.3c, "The Gemma 4 rendering"): the generation prompt of the
    /// messages before it, then the open text as the model writes it. A `content` member closes the thinking with
    /// `<channel|>` when thinking is on. With thinking off, the prompt closed it.
    pub(crate) fn run_open(
        &self,
        before: &[Json],
        open: &Message,
        tools: &[Json],
        thinking: bool,
    ) -> Result<Rendered, FamilyError> {
        let reasoning = open.reasoning.as_deref().unwrap_or("");
        let close = if thinking && open.content.is_some() {
            CHANNEL_CLOSE
        } else {
            ""
        };
        let content = nagaya_council::family::content_text(open).unwrap_or("");
        self.chat
            .render_open(before, tools, thinking, &[(reasoning, close), (content, "")])
            .map_err(refusal)
    }

    /// The seam check of mk's `prefill` (gogi "Continuations"): each open field that is not empty must tokenize on its
    /// own. The rendering up to the start of the field must be a token prefix of the rendering through its end.
    fn check_seams(&self, before: &[Json], open: &Message, tools: &[Json], thinking: bool) -> Result<(), FamilyError> {
        for seam in gogi_proto::continuation::seams(open) {
            let start = self.run_open(before, &seam.before, tools, thinking)?.tokens;
            let with = self.run_open(before, &seam.with, tools, thinking)?.tokens;
            gogi_proto::continuation::check_seam(seam.field, &start, &with, |ids| self.lossy_text(ids))?;
        }
        Ok(())
    }

    /// The text of tokens, with U+FFFD for bytes that are not whole characters. An error message shows it.
    fn lossy_text(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.tokenizer().token_bytes(id).unwrap_or_default())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The tokens of template text: its marker text gives the marker tokens.
    pub(crate) fn tpl(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.tokenizer().encode(text).map_err(internal)
    }

    /// The tokens of client text: its marker text stays text. A noncharacter is the client's error.
    pub(crate) fn content(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        check_text(text).map_err(client)?;
        self.tokenizer().encode_content(text).map_err(internal)
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
}

/// The tokens of the letters `A` to `Z`, with a leading space and bare. The run stops at the first letter that fails.
/// A letter fails when it is not one token in both forms, or when an earlier letter has its token. A run of fewer
/// than two letters is an error.
fn letter_tokens(tokenizer: &Tokenizer) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let one = |text: &str| -> Result<Option<u32>, Error> {
        Ok(match tokenizer.encode(text)?.as_slice() {
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

impl Family for Gemma4Family {
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
        context.check()?;
        let tools = messages::tools_json(&context.tools);
        let msgs: Vec<Json> = context.messages.iter().map(messages::message_json).collect();
        let thinking = context.thinks();
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
        // A marked message is a boundary when its rendering is a prefix of the whole (gogi "Contexts"). The template
        // drops the thinking of a text answer before a later user message, so such a message is no boundary.
        let last = msgs.len() - 1;
        let mut boundaries: Vec<Boundary> = Vec::new();
        for (i, message) in context.messages.iter().enumerate().take(last) {
            if !message.snap {
                continue;
            }
            let part = self.run(&msgs[..=i], &tools, thinking, false)?;
            let new = boundaries.last().is_none_or(|b| b.tokens < part.tokens.len());
            if base.tokens.starts_with(&part.tokens) && new {
                boundaries.push(Boundary {
                    message: i,
                    tokens: part.tokens.len(),
                });
            }
        }
        if boundaries.last().is_some_and(|b| b.tokens == end) {
            boundaries.pop();
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
        })
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        read::decision_read(self, input)
    }

    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
        read::raw_read(self, letters, suffixes)
    }

    fn reply_parser(&self, start: ReplyStart, _tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        // The Gemma 4 call form writes its own types, so the parser needs no schema.
        Ok(reply::reply_parser(self, start))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.content(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        self.tokenizer().decode_whole(tokens).map_err(tokens_error)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        check_text(text).map_err(client)?;
        self.tokenizer().encode(text).map_err(internal)
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
}
