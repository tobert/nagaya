//! The interface of a family library (spec 11.3, 11.3a).
//!
//! A family library holds the text rules of one model family. The coordinator holds one `dyn Family` for each
//! served model. A family has no I/O and no engine: it turns contexts, specs and cases into tokens, and tokens into
//! text and replies. The engine feeds the tokens and gives the logits back (engine-abi, "Logit outputs").
//!
//! The routes use the interface in these ways:
//!
//! - **Contexts and snapshots.** [`Family::render`] gives the tokens of a context and the boundaries where a
//!   snapshot may be kept. A context whose last message is open renders without the end of that message
//!   (gogi "Continuations").
//! - **Decisions.** [`Family::decision_read`] gives a [`ReadPlan`]. The caller feeds the plan's tokens, reads the
//!   letter logits that the plan names, and gives them back until the plan is done.
//! - **Raw reads.** [`Family::raw_read`] gives the letter token ids and the tokens of each suffix.
//! - **Generations.** [`Family::reply_parser`] labels the tokens of a reply and parses the whole reply.
//! - **Extensions.** [`Family::tokenize`], [`Family::tokenize_special`], [`Family::detokenize`] and [`Family::pieces`]
//!   serve the `/nagaya/v1` routes (spec 11.4a).
//! - **Identity.** [`Family::template_id`] and [`Family::vocab_size`] go in the council identity.

use std::fmt;

use gogi_proto::Invalid;
use gogi_proto::context::{Context, Tool};
use gogi_proto::generation::Phase;
use gogi_proto::spec::Spec;
use serde_json::{Map, Value};

/// Why a family library refused a request.
#[derive(Clone, Debug, PartialEq)]
pub enum FamilyError {
    /// The client's input is wrong: a bad context, a queue schema that does not match, a letter that is not one
    /// token. The route answers with a 400 and the text of the [`Invalid`].
    Client(Invalid),
    /// The library or its caller has a defect. The route answers with a 500 `internal`, and the process stops
    /// (spec 8.5, 11.5).
    Internal(String),
}

impl FamilyError {
    /// A client error without a request field.
    pub fn client(message: impl Into<String>) -> Self {
        FamilyError::Client(Invalid::new(message))
    }

    /// A client error that names the request field at fault.
    pub fn client_param(param: impl Into<String>, message: impl Into<String>) -> Self {
        FamilyError::Client(Invalid::param(param, message))
    }

    /// An internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        FamilyError::Internal(message.into())
    }

    /// True when the client caused the error.
    pub fn is_client(&self) -> bool {
        matches!(self, FamilyError::Client(_))
    }
}

impl fmt::Display for FamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FamilyError::Client(invalid) => write!(f, "{}", invalid.message),
            FamilyError::Internal(message) => write!(f, "internal error in a family library: {message}"),
        }
    }
}

impl std::error::Error for FamilyError {}

impl From<Invalid> for FamilyError {
    fn from(invalid: Invalid) -> Self {
        FamilyError::Client(invalid)
    }
}

/// A boundary in a rendering: the end of a message that the client marked, where a snapshot may be kept.
///
/// A family keeps a boundary only when the rendering of the messages up to it is a prefix of the whole rendering.
/// The end of the context is always a boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boundary {
    /// The index of the message that ends here.
    pub message: usize,
    /// The token count at the end of that message.
    pub tokens: usize,
}

/// A piece of client text that spells a control token of the family (gogi "Terms": control text).
///
/// The family tokenizes the text as content, so the hit never opens or closes a turn. The route reports it in
/// `signals.control_text`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlHit {
    /// Where the text is: `case`, `spec`, `suffix:<i>`, or `message:<i>` for a message of a context.
    pub place: String,
    /// The control token that the text spells.
    pub token: String,
}

/// The tokens of a context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendering {
    /// The tokens of the context, then the generation prompt when it was asked for. An open last message takes the
    /// place of the generation prompt, so the tokens end with its open text.
    pub tokens: Vec<u32>,
    /// The token count before the generation prompt. It equals `tokens.len()` when there is none.
    pub end: usize,
    /// The boundaries that the rendering keeps, in order. The last one is at `end`.
    pub boundaries: Vec<Boundary>,
    /// The control text in the context's messages.
    pub control_text: Vec<ControlHit>,
}

/// One asked question of a decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asked {
    /// The index of the question in the spec's `questions`.
    pub question: usize,
    /// The positions in the question's answer keys that the read keeps, in increasing order. A position is the
    /// position of the option in the spec, so a letter does not change when the options narrow.
    pub options: Vec<usize>,
}

/// The input of a decision read.
#[derive(Clone, Copy, Debug)]
pub struct DecisionInput<'a> {
    /// The context and its rendering, or `None` for a decision with no context. The rendering has no generation
    /// prompt. The family renders it again when the read needs a different one.
    pub context: Option<(&'a Context, &'a Rendering)>,
    /// The held spec.
    pub spec: &'a Spec,
    /// The case, as text.
    pub case: &'a str,
    /// The asked questions, in the order of the response.
    pub asked: &'a [Asked],
}

/// How a step continues the model state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Continues {
    /// The step feeds its tokens after the plan's prefix. Steps of this kind do not see each other.
    Prefix,
    /// The step feeds its tokens after the step before it. A queue read goes this way.
    PreviousStep,
}

/// One step of a read: tokens to feed, then letters to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadStep {
    /// The index in `asked` of the question that this step reads.
    pub question: usize,
    /// Where the step starts.
    pub continues: Continues,
    /// The tokens to feed. After them the caller takes the logits of the last position.
    pub feed: Vec<u32>,
    /// The token id of the letter of each kept option, in the order of `Asked::options`.
    pub letters: Vec<u32>,
}

/// What a plan gives when it is done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadDone {
    /// The SHA-256 of the text that the read was conditioned on, in lowercase hexadecimal (gogi "Decisions").
    pub rendered_sha256: String,
}

/// What a plan gives next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Advance {
    /// Feed these tokens and read these letters.
    Step(ReadStep),
    /// The read is complete.
    Done(ReadDone),
}

/// The plan of a decision read, one context at a time.
///
/// In a queue context a later question is read after the most likely letter of the one before (gogi "Decisions").
/// Thus the plan needs the letter log probabilities of each step before it gives the next one.
pub trait ReadPlan: Send {
    /// The tokens to feed after the context's rendering, before the first step. All the context's tokens when the
    /// read has no context. The caller may keep a snapshot after them.
    fn prefix(&self) -> &[u32];

    /// The control text in the spec and in the case.
    fn control_text(&self) -> &[ControlHit];

    /// Give the next step. `last` is `None` for the first call. Then it holds the log probabilities of the letters of
    /// the step before, in the order of its `letters`. A call after `Done` is an internal error. So is a first call
    /// with a `last`, a later call without one, and a `last` of the wrong length.
    fn advance(&mut self, last: Option<&[f64]>) -> Result<Advance, FamilyError>;
}

/// The input of a raw read, before its contexts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawRead {
    /// The token id of each letter, in the order of the request. A letter must be one token, and no two are equal.
    pub letters: Vec<u32>,
    /// The tokens of each suffix. Marker text in a suffix becomes the marker token (gogi "Raw reads").
    pub suffixes: Vec<Vec<u32>>,
    /// The control text in the suffixes.
    pub control_text: Vec<ControlHit>,
}

/// One token of a reply, labeled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplyStep {
    /// The part of the reply that the token belongs to.
    pub phase: Phase,
    /// The text that the token adds to its part. It may be empty.
    pub piece: String,
}

/// A call in a reply.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplyCall {
    /// A call that parsed.
    Parsed {
        /// The tool's name.
        name: String,
        /// The input, in the key order that the model wrote.
        input: Map<String, Value>,
    },
    /// A call that did not parse: its text as written, and why it failed. The parser makes no call id.
    Malformed { text: String, error: String },
}

/// A whole reply.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedReply {
    /// The content.
    pub content: String,
    /// The reasoning.
    pub reasoning: String,
    /// The calls, in order.
    pub calls: Vec<ReplyCall>,
    /// True when the model ended the turn.
    pub ended: bool,
    /// The pieces that the reply released after its last token, such as text that the parser held back as a
    /// possible marker. They belong to the parts above, but no `push` gave them.
    pub tail: Vec<ReplyStep>,
}

/// Where a reply starts (gogi "Continuations").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplyStart {
    /// True when the first token of the reply is in its reasoning: the generation prompt opened the thinking, or the
    /// open message has no `content` member and the context thinks.
    pub thinking: bool,
    /// True when the reply goes on from an open message. The parser then trims nothing at the start of the reply,
    /// because `done` joins the open text and the new text with nothing trimmed at the seam.
    pub open: bool,
}

impl ReplyStart {
    /// A reply after the generation prompt.
    pub fn prompt(thinking: bool) -> Self {
        ReplyStart { thinking, open: false }
    }

    /// Where the reply after a context starts. An open message without a `content` member continues inside the
    /// thinking. A `content` member closes the thinking, even when it is empty.
    pub fn of(context: &Context) -> Self {
        match context.open() {
            Some(open) => ReplyStart {
                thinking: context.thinks() && open.content.is_none(),
                open: true,
            },
            None => ReplyStart::prompt(context.thinks()),
        }
    }
}

/// Reads the tokens of one reply, in order, after the generation prompt or after an open message. It reads only the
/// sampled tokens, so the text of an open message never changes a phase.
pub trait ReplyParser: Send {
    /// Label one token. A token outside the vocabulary, or one after the end of the reply, is an internal error.
    fn push(&mut self, token: u32) -> Result<ReplyStep, FamilyError>;

    /// True when the last token ended the reply, such as a stop token. The caller samples no more tokens, and a
    /// later `push` is an internal error.
    fn ended(&self) -> bool;

    /// The phase that the next token would have: `call` inside a call, `reasoning` inside the thinking, else
    /// `content`. A reply that ends with `max_tokens` reports it as `done.phase`.
    fn phase(&self) -> Phase;

    /// End the reply and give its parts. With `max_tokens` true, the reply reached `max_tokens`: the parser keeps
    /// the white space at the end of the part that it was in, so the client can send the reply back exactly.
    fn finish(self: Box<Self>, max_tokens: bool) -> ParsedReply;
}

/// The text rules of one model family. The trait is object safe: a coordinator holds a `dyn Family` for each served
/// model.
pub trait Family: Send + Sync {
    /// Render a context. A context that the family cannot render is [`FamilyError::Client`].
    ///
    /// An open last message (`continue: true`) renders as a reply up to its open text, without its end and without
    /// a generation prompt. Thus `end` is the length of the tokens. The family checks that each open field tokenizes
    /// on its own (gogi `continuation::seams`). A field that does not is [`FamilyError::Client`] that names
    /// `continue`.
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError>;

    /// Plan a decision read of one context, or of no context. A queue schema that does not match the asked
    /// questions is [`FamilyError::Client`].
    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError>;

    /// Tokenize the letters and the suffixes of a raw read.
    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError>;

    /// Start a reply parser. `start` says where the reply starts. `tools` are the tools of the context: a family that
    /// reads a call against the schema of its tool needs them. A tool list that the family cannot read is
    /// [`FamilyError::Client`].
    fn reply_parser(&self, start: ReplyStart, tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError>;

    /// Tokenize client text as content. Text that spells a control token stays text.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError>;

    /// Tokenize text in which the text of a marker is the marker token. This is the rule of a raw-read suffix (gogi
    /// "Raw reads"). A text that the family cannot tokenize is [`FamilyError::Client`].
    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError>;

    /// Turn tokens into text. A token outside the vocabulary, or tokens that do not make whole text, are
    /// [`FamilyError::Client`] with the `param` `tokens`.
    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError>;

    /// The text of each token alone, in order. A piece is the bytes of one token, and a token can end inside a
    /// character. Then the piece has U+FFFD in place of the bad bytes. The piece of a marker is its text. A token
    /// outside the vocabulary is [`FamilyError::Client`] with the `param` `tokens`.
    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError>;

    /// The template id: `nagaya-council:<hash>+<family>-chat:<hash>` (spec 11.3a).
    fn template_id(&self) -> &str;

    /// The number of tokens in the vocabulary. A logit row has this many elements.
    fn vocab_size(&self) -> usize;
}
