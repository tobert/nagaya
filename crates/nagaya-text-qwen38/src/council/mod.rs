//! The Qwen3.8 family library for the council (spec 11.3, 11.3a): nagaya's family interface on this crate.
//!
//! A context renders with the chat template of the GGUF. The template runs with the reasoning effort `medium`, as
//! mk's council does, so a context renders the same with thinking on and off except for its generation prompt.
//! Client text is content: its special token text is escaped before the template runs and tokenized as text.
//!
//! A decision puts a spec, a case and a letter menu after the context. This is mkviolet's rendering (mkviolet
//! `src/template.rs`) in Qwen3.8's chat format. A plain read is:
//!
//! ```text
//! context messages (the chat template)
//! <|im_start|>user\n{instructions}\n\nAnswer each question below about the {label} in the next message. ...
//!   Question {n}. {instructions}\n{A}. {option}: {means}\n ...<|im_end|>\n                         spec layer
//! <|im_start|>user\n{label}:\n{case}<|im_end|>\n
//! <|im_start|>assistant\n<think>\n\n</think>\n\nAnswer to question                                  case
//!  {n} ({first sentence of the question}):                                  one read for each question
//! ```
//!
//! The letter token is the letter with a leading space, as in mkviolet. A queue read gives the case as the result of
//! the context's last call, and writes the call to the queue tool in the template's form:
//!
//! ```text
//! context messages, then a tool message that holds the case
//! <|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function={queue}>\n<parameter=answers>\n
//! {"{id}": "  then the bare letter token, then `", "{id}": "` for the next question
//! ```
//!
//! Where mkviolet and gogi's contract disagree, gogi wins. Each such place has a comment with the word "gogi".

mod messages;
mod read;
mod reply;
mod test_set;
#[cfg(test)]
mod tests;

use gogi_proto::context::{Context, Tool};
use nagaya_council::family::{
    Boundary, ControlHit, DecisionInput, Family, FamilyError, RawRead, ReadPlan, Rendering, ReplyParser,
};
use nagaya_council::template_id::{hash16, template_id};

use crate::gguf::{Metadata, Value};
use crate::json::Json;
use crate::template::RenderOptions;
use crate::{Error, Qwen38, Rendered, Tokenizer};

/// The name of the family in the template id.
const FAMILY: &str = "qwen38";

/// The reasoning effort of every render. At `medium` the template writes no effort line (mk `council_v1.py`).
const EFFORT: &str = "medium";

/// Where in the library the test hook changes tokens. The hook exists in test builds only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    /// The tokens of a rendered context that ends with a generation prompt.
    Context,
    /// The tokens of a letter or a suffix of a raw read.
    Raw,
    /// The prefix of a read plan.
    Prefix,
    /// The tokens that a step of a read plan feeds.
    Feed,
}

/// The test hook: it changes tokens that the library renders. Test builds only.
pub(crate) type HookFn = fn(Site, &mut [u32]);

/// The Qwen3.8 family library.
pub struct Qwen38Family {
    model: Qwen38,
    /// The token of each letter with a leading space, `A` first. This is the menu of a plain read.
    letters: Vec<u32>,
    /// The token of each bare letter, `A` first. A queue read uses these: its letter follows an opening quote.
    bare_letters: Vec<u32>,
    /// The tokens that end a reply (`<|im_end|>` and `<|endoftext|>`).
    stop: Vec<u32>,
    template_id: String,
    #[cfg(test)]
    pub(crate) hook: Option<HookFn>,
}

/// A client error from a library error: the text or the messages are at fault.
pub(crate) fn client(e: Error) -> FamilyError {
    FamilyError::client(e.to_string())
}

/// A library error for a failure that the client did not cause.
pub(crate) fn internal(e: Error) -> FamilyError {
    FamilyError::internal(e.to_string())
}

/// A refusal of the template or of the tokenizer on client input is the client's. Any other error is ours.
fn refusal(e: Error) -> FamilyError {
    match e {
        Error::Template(_) | Error::Reserved { .. } | Error::Tools(_) | Error::Json(_) => client(e),
        other => internal(other),
    }
}

impl Qwen38Family {
    /// Build the family from the GGUF file at `path`.
    pub fn from_gguf(path: &std::path::Path) -> Result<Qwen38Family, Error> {
        Qwen38Family::from_metadata(&Metadata::open(path)?)
    }

    /// Build the family from GGUF metadata, and compute its template id from its test renders.
    pub fn from_metadata(m: &Metadata) -> Result<Qwen38Family, Error> {
        Qwen38Family::new(m, None)
    }

    /// Build the family with a hook that changes the tokens it renders. Tests use this to see the id change.
    #[cfg(test)]
    pub(crate) fn with_hook(m: &Metadata, hook: HookFn) -> Result<Qwen38Family, Error> {
        Qwen38Family::new(m, Some(hook))
    }

    #[cfg_attr(not(test), allow(unused_variables))]
    fn new(m: &Metadata, hook: Option<HookFn>) -> Result<Qwen38Family, Error> {
        let source = match m.get("tokenizer.chat_template") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(Error::Metadata(
                    "tokenizer.chat_template is missing or is not a string. Use a GGUF made for this family.".into(),
                ));
            }
        };
        let model = Qwen38::from_metadata(m)?;
        let (letters, bare_letters) = letter_tokens(model.tokenizer())?;
        let stop = reply::stop_tokens(model.tokenizer())?;
        let mut family = Qwen38Family {
            model,
            letters,
            bare_letters,
            stop,
            template_id: String::new(),
            #[cfg(test)]
            hook,
        };
        let council = nagaya_council::template_id::council_hash(&test_set::renders(&family));
        family.template_id = template_id(&council, FAMILY, &hash16(source.as_bytes()));
        Ok(family)
    }

    #[cfg_attr(not(test), allow(unused_variables))]
    pub(crate) fn tweak(&self, site: Site, tokens: &mut [u32]) {
        #[cfg(test)]
        if let Some(hook) = self.hook {
            hook(site, tokens);
        }
    }

    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        self.model.tokenizer()
    }

    /// The tokenizer, shared with a read plan or a reply parser that outlives a borrow.
    pub(crate) fn shared_tokenizer(&self) -> std::sync::Arc<Tokenizer> {
        self.model.shared_tokenizer()
    }

    /// The number of letters that a question can use: one token for each, no two the same.
    pub fn letter_count(&self) -> usize {
        self.letters.len()
    }

    /// Render messages with the chat template. The effort is `medium`.
    pub(crate) fn run(
        &self,
        messages: &[Json],
        tools: Option<&Json>,
        thinking: bool,
        generation_prompt: bool,
    ) -> Result<Rendered, FamilyError> {
        let options = RenderOptions {
            thinking,
            generation_prompt,
            reasoning_effort: Some(EFFORT.to_string()),
        };
        self.model
            .render(&Json::Array(messages.to_vec()), tools, &options)
            .map_err(refusal)
    }

    /// Client text as escaped content: its special token text cannot open or close a turn.
    pub(crate) fn content(&self, text: &str) -> Result<String, FamilyError> {
        self.tokenizer().escape(text).map_err(refusal)
    }

    /// The tokens of escaped text. The template's own markers are the only special tokens.
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
}

/// The tokens of the letters `A` to `Z`, with a leading space and bare. The run stops at the first letter that is
/// not one token of its own in both forms, or whose token an earlier letter has. A run of fewer than two letters is
/// an error.
fn letter_tokens(tokenizer: &Tokenizer) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let one = |text: &str| -> Result<Option<u32>, Error> {
        let ids = tokenizer.encode_escaped(&tokenizer.escape(text)?)?;
        Ok(match ids.as_slice() {
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
        return Err(Error::Vocab(format!(
            "only {} of the letters A to Z are one token each. A question needs at least two.",
            spaced.len()
        )));
    }
    Ok((spaced, bare))
}

impl Family for Qwen38Family {
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
        context.check()?;
        let tools = messages::tools_json(&context.tools);
        let messages: Vec<Json> = context.messages.iter().map(messages::message_json).collect();
        let thinking = context.thinks();
        let base = self.run(&messages, tools.as_ref(), thinking, false)?;
        let end = base.ids.len();
        let mut tokens = if generation_prompt {
            let whole = self.run(&messages, tools.as_ref(), thinking, true)?.ids;
            if !whole.starts_with(&base.ids) {
                return Err(FamilyError::internal(
                    "the generation prompt does not extend the rendering of the context. The chat template changed shape.",
                ));
            }
            whole
        } else {
            base.ids.clone()
        };
        // A marked message is a boundary when its rendering is a prefix of the whole (gogi "Contexts").
        let last = messages.len() - 1;
        let mut boundaries = Vec::new();
        for (i, message) in context.messages.iter().enumerate().take(last) {
            if !message.snap {
                continue;
            }
            let part = self.run(&messages[..=i], tools.as_ref(), thinking, false)?;
            if base.ids.starts_with(&part.ids) {
                boundaries.push(Boundary {
                    message: i,
                    tokens: part.ids.len(),
                });
            }
        }
        boundaries.push(Boundary {
            message: last,
            tokens: end,
        });
        if generation_prompt {
            self.tweak(Site::Context, &mut tokens);
        }
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

    fn reply_parser(&self, thinking: bool, tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        reply::reply_parser(self, thinking, tools)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.encode(&self.content(text)?)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        self.tokenizer().decode(tokens).map_err(|e| match e {
            Error::TokenId { .. } => FamilyError::client_param("tokens", e.to_string()),
            Error::Bytes(_) => {
                FamilyError::client_param("tokens", format!("{e}. Send the tokens of whole characters."))
            }
            other => internal(other),
        })
    }

    fn template_id(&self) -> &str {
        &self.template_id
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer().len()
    }
}
