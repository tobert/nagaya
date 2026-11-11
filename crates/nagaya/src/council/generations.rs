//! Generations (gogi "Generations"): a reply sampled after a context, with the sampler of spec 11.2.

use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::mpsc::{Sender, SyncSender};
use std::time::Instant;

use gogi_proto::Invalid;
use gogi_proto::context::{Context, Usage};
use gogi_proto::generation::{
    Finish, GenerationDone, GenerationRequest, Phase, ReplyStats, ReplyToolCall, TokenEvent, TokenStats,
};
use gogi_proto::message::Message;
use nagaya_abi::Kind;
use nagaya_council::build::{BoundaryRecipe, range_recipe, recipes_with_cuts};
use nagaya_council::family::{
    Boundary, Family, FamilyError, ParsedReply, Rendering, ReplyCall, ReplyParser, ReplyStart,
};
use nagaya_council::latent::{LatentMap, top};
use nagaya_council::sampler::{Settings, pick_except};
use nagaya_council::trace::Trace;
use nagaya_store::{ContentId, Reason, RecipeId};

use super::MediaRead;
use super::build::{ContextBody, Cut, Extension, Finished, Got, JobControl, Shared, Then, Want};
use super::error::ServiceError;
use super::{Schedule, Service, snapshot_id};

/// The most tokens that a generation feeds after its reply to hold the head (gogi "Generations": "at most a few
/// tokens of feeding").
pub const HEAD_FEED: usize = 8;

/// The stats of one part of a reply, from the untempered probability and the entropy of each token.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PartStats {
    tokens: u64,
    logprob_sum: f64,
    logprob_min: f64,
    entropy_sum: f64,
    entropy_max: f64,
}

impl PartStats {
    pub(super) fn push(&mut self, p: f64, h: f64) {
        let logprob = p.ln();
        if self.tokens == 0 {
            self.logprob_min = logprob;
            self.entropy_max = h;
        }
        self.tokens += 1;
        self.logprob_sum += logprob;
        self.logprob_min = self.logprob_min.min(logprob);
        self.entropy_sum += h;
        self.entropy_max = self.entropy_max.max(h);
    }

    /// The stats in gogi's shape. A part with no token has none.
    pub(super) fn finish(&self) -> Option<TokenStats> {
        (self.tokens > 0).then(|| TokenStats {
            tokens: self.tokens,
            mean_logprob: self.logprob_sum / self.tokens as f64,
            min_logprob: self.logprob_min,
            mean_entropy: self.entropy_sum / self.tokens as f64,
            max_entropy: self.entropy_max,
        })
    }
}

/// The trace of a generation (`docs/lab.md`, "`trace`"): its records, and the taps of the step that the job fed last.
pub(crate) struct Tracing {
    trace: Trace,
    /// The residual tap of the last logit row, until the sample of that row records it.
    residual: Option<Vec<f32>>,
    /// The tap `lens` of the last logit row, when the trace has a lens.
    lens: Option<Vec<f32>>,
}

impl Tracing {
    pub fn new(trace: Trace) -> Self {
        Self {
            trace,
            residual: None,
            lens: None,
        }
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The latent steps of a generation (docs/lab.md, "Latent steps in a request").
#[derive(Clone, Debug)]
pub(crate) struct Latent {
    /// The steps after the last token of the prompt, 1 to [`nagaya_council::latent::MAX_STEPS`].
    pub steps: usize,
    /// True when the generation keeps the top token of the logit row of each step.
    pub read: bool,
    /// The map from the residual tap to the next input row.
    pub map: Arc<LatentMap>,
}

/// The latent steps of a generation while they run.
struct LatentRun {
    plan: Latent,
    /// The latent rows that the slot holds.
    fed: usize,
    /// The top token of the logit row of each step, when the plan reads them.
    read: Vec<u32>,
}

/// Where a generation is.
enum Stage {
    /// The build is running.
    Building,
    /// The feed of the prompt with the residual tap, then the latent steps.
    Latent,
    /// The feed of the generation prompt and of each sampled token.
    Sampling,
    /// The feed of the tokens after the reply, then the snapshot of the head.
    Head { recipe: RecipeId, boundary: Boundary },
    /// The reply and the head are complete.
    Done,
}

/// What a generation gives back.
pub(crate) struct Reply {
    /// The sampled tokens.
    pub tokens: Vec<u32>,
    /// The tokens of the reasoning part.
    pub reasoning: u64,
    pub finish: Finish,
    pub parsed: ParsedReply,
    pub reasoning_stats: Option<TokenStats>,
    pub content_stats: Option<TokenStats>,
    /// The head after the reply, when the job held it.
    pub head: Option<(Boundary, ContentId)>,
    /// The phase of the next token, for a reply that ended with `max_tokens`.
    pub phase: Option<Phase>,
}

/// The work of a generation after its build: feed the generation prompt, sample each token, and hold the head
/// after the reply when it costs at most [`HEAD_FEED`] tokens.
pub(crate) struct GenerateThen {
    family: Arc<dyn Family>,
    context: Context,
    /// The open message of the context, for a continuation (gogi "Continuations").
    open: Option<Message>,
    root: RecipeId,
    /// The seals of the state after the build: the boundaries of the build, and the cut and the seals of its chain
    /// when the build started from one.
    recipes: Vec<BoundaryRecipe>,
    /// The rendering that the build makes: the context without the generation prompt, or the context before the open
    /// message of a continuation.
    rendering: Rendering,
    /// The tokens of the prompt: the context, then the generation prompt or the open text.
    prompt: Vec<u32>,
    /// The tokens that the build makes. The generation feeds the rest of the prompt for its first logit row.
    end: usize,
    settings: Settings,
    max_tokens: usize,
    context_tokens: usize,
    /// The most tokens that the slot of the job holds.
    room: usize,
    parser: Option<Box<dyn ReplyParser>>,
    events: Option<SyncSender<TokenEvent>>,
    started: Instant,
    stage: Stage,
    sampled: Vec<u32>,
    reasoning: u64,
    stats: [PartStats; 2],
    finish: Option<Finish>,
    parsed: Option<ParsedReply>,
    head: Option<(Boundary, ContentId)>,
    /// False on an engine with a feed rule: the reply rows come from decode quanta, so no head is kept (spec 6.14).
    keeps_head: bool,
    /// The phase of the next token after the last sampled token.
    next_phase: Option<Phase>,
    /// The trace, when the request asks for one. It reads each logit row and its taps, and it changes no want but
    /// the taps of a feed.
    tracing: Option<Tracing>,
    /// The tokens that the reply never picks: the markers of the speech modes of a family that speaks (spec 11.11).
    dropped: Vec<u32>,
    /// The latent steps before the first token of the reply, when the request asks for them.
    latent: Option<LatentRun>,
}

impl GenerateThen {
    /// The latent rows that the slot holds before the reply.
    fn latent_steps(&self) -> usize {
        self.latent.as_ref().map_or(0, |run| run.plan.steps)
    }

    /// One latent step after a tapped pass: read its logit row, then feed the next row (docs/lab.md, "Latent steps in a
    /// request"). The pass of the prompt gives no read, because the reply's first token comes after the last step.
    fn latent_step(&mut self, row: &[f32], residual: &[u8]) -> Result<Want, ServiceError> {
        let position = self.prompt.len();
        let run = self
            .latent
            .as_mut()
            .expect("a generation in its latent stage has latent steps");
        if run.fed > 0 && run.plan.read {
            run.read
                .push(top(row).ok_or_else(|| ServiceError::internal("a latent step gave an empty logit row"))?);
        }
        let next = run
            .plan
            .map
            .row(residual)
            .map_err(|error| ServiceError::internal(format!("latent step {}: {error}", run.fed + 1)))?;
        let position = position + run.fed;
        run.fed += 1;
        let tap = run.fed < run.plan.steps;
        let (token, meta) = self
            .family
            .latent_record(position, 1, run.plan.map.width)
            .ok_or_else(|| ServiceError::internal("the family lost its latent rows after the request check"))?;
        Ok(Want::Latent {
            token,
            meta,
            row: next,
            position,
            tap,
            kind: Kind::Decode,
        })
    }

    /// Take the logit row of the last latent step: read it, and start the reply. A generation in another stage keeps
    /// its row as it is. The ensemble calls it before it combines the row.
    pub(crate) fn end_latent(&mut self, row: &[f32]) -> Result<(), ServiceError> {
        if !matches!(self.stage, Stage::Latent) {
            return Ok(());
        }
        let run = self
            .latent
            .as_mut()
            .expect("a generation in its latent stage has latent steps");
        if run.fed != run.plan.steps {
            return Err(ServiceError::internal(format!(
                "a latent stage ended after {} of {} steps",
                run.fed, run.plan.steps
            )));
        }
        if run.plan.read {
            run.read
                .push(top(row).ok_or_else(|| ServiceError::internal("a latent step gave an empty logit row"))?);
        }
        self.stage = Stage::Sampling;
        Ok(())
    }

    /// The top token of each latent step, when the request read them.
    pub(crate) fn latent_read(&self) -> Option<&[u32]> {
        self.latent
            .as_ref()
            .filter(|run| run.plan.read)
            .map(|run| run.read.as_slice())
    }

    /// The reply, once the work is done.
    pub fn reply(self) -> Reply {
        assert!(
            matches!(self.stage, Stage::Done),
            "a generation gives its reply when it is done"
        );
        Reply {
            tokens: self.sampled,
            reasoning: self.reasoning,
            finish: self.finish.expect("a done generation has a finish"),
            parsed: self.parsed.expect("a done generation has a parsed reply"),
            reasoning_stats: self.stats[0].finish(),
            content_stats: self.stats[1].finish(),
            head: self.head,
            phase: self.next_phase.filter(|_| self.finish == Some(Finish::MaxTokens)),
        }
    }

    /// Sample the next token from a logit row, and give it to the parser and to the stream.
    fn sample(&mut self, row: &[f32]) -> Result<Want, ServiceError> {
        let position = self.prompt.len() + self.latent_steps() + self.sampled.len();
        let chosen = pick_except(row, &self.settings, position as u64, &self.dropped)
            .map_err(|error| ServiceError::internal(format!("the sample at position {position} failed: {error}")))?;
        if let Some(tracing) = &mut self.tracing {
            let residual = tracing
                .residual
                .take()
                .ok_or_else(|| ServiceError::internal("a traced logit row came with no residual tap"))?;
            let lens = tracing.lens.take();
            tracing
                .trace
                .record(chosen.token, row, &residual, lens.as_deref())
                .map_err(|error| ServiceError::internal(format!("the trace at position {position}: {error}")))?;
        }
        let parser = self
            .parser
            .as_mut()
            .expect("a parser reads the reply while it is sampled");
        let step = parser.push(chosen.token)?;
        let ended = parser.ended();
        self.sampled.push(chosen.token);
        match step.phase {
            Phase::Reasoning => {
                self.reasoning += 1;
                self.stats[0].push(chosen.p, chosen.h);
            }
            Phase::Content => self.stats[1].push(chosen.p, chosen.h),
            Phase::Call | Phase::Unknown => {}
        }
        let n = self.sampled.len();
        // The reply ends at this token when the parser ended it, or when it reaches a limit. A full context goes
        // before `max_tokens` (gogi "Generations", `finish`).
        let end = if ended {
            Some(None)
        } else if self.prompt.len() + self.latent_steps() + n >= self.context_tokens {
            Some(Some(Finish::ContextFull))
        } else if n == self.max_tokens {
            Some(Some(Finish::MaxTokens))
        } else {
            None
        };
        let mut piece = step.piece;
        let parsed = match end {
            Some(cut) => {
                let max_tokens = cut == Some(Finish::MaxTokens);
                let parser = self.parser.take().expect("a reply finishes once");
                self.next_phase = Some(parser.phase());
                let parsed = parser.finish(max_tokens);
                piece.push_str(&tail_text(&parsed, step.phase)?);
                Some(parsed)
            }
            None => None,
        };
        if let Some(events) = &self.events {
            let seconds = self.started.elapsed().as_secs_f64();
            let event = TokenEvent {
                id: chosen.token,
                piece,
                phase: step.phase,
                p: chosen.p,
                h: chosen.h,
                n: n as u64,
                tps: (seconds > 0.0).then(|| n as f64 / seconds),
            };
            // A closed stream cancels the reply (gogi "Streams").
            if events.send(event).is_err() {
                return Ok(Want::Cancel);
            }
        }
        match (end, parsed) {
            (Some(cut), Some(parsed)) => self.finish(cut, parsed),
            _ => Ok(self.row_feed(vec![chosen.token], Kind::Decode)),
        }
    }

    /// The feed of tokens whose last logit row the generation samples. A trace names the taps on its last pass. The
    /// passes are the same with or without the taps.
    fn row_feed(&self, tokens: Vec<u32>, kind: Kind) -> Want {
        match &self.tracing {
            Some(tracing) => Want::Tap {
                tokens,
                kind,
                lens: tracing.trace.plan().lens == nagaya_council::trace::Lens::Logit,
            },
            None => Want::Feed {
                tokens,
                logits: true,
                kind,
            },
        }
    }

    /// The trace, once the work is done.
    pub fn take_trace(&mut self) -> Option<Trace> {
        self.tracing.take().map(|tracing| tracing.trace)
    }

    /// End the reply. `cut` names why a reply that did not end its turn stopped. `parsed` is the reply that the
    /// parser read.
    ///
    /// The reply of a continuation is the whole message: the open text as sent, then the new text, with nothing
    /// trimmed at the seam (gogi "Continuations"). The head after a reply is the end of its closed rendering. The head
    /// after a reply that ended with `max_tokens` is the seam of its open rendering (spec 11.3c).
    fn finish(&mut self, cut: Option<Finish>, mut parsed: ParsedReply) -> Result<Want, ServiceError> {
        let finish = match cut {
            Some(cut) => cut,
            None if parsed.calls.is_empty() => Finish::EndTurn,
            None => Finish::ToolUse,
        };
        self.finish = Some(finish);
        if let Some(open) = &self.open {
            parsed.reasoning.insert_str(0, open.reasoning.as_deref().unwrap_or(""));
            parsed.content.insert_str(0, open.text().unwrap_or(""));
        }
        let done = GenerationDone {
            finish,
            phase: self.next_phase,
            ..reply_done(&parsed, &self.call_ids(parsed.calls.len()))
        };
        let reply = match finish {
            Finish::MaxTokens => done.as_open_message(),
            _ => done.as_message(),
        };
        self.parsed = Some(parsed);
        // Decided 2026-10-21 (spec 6.14): a generation on an engine with a feed rule holds no head after its reply.
        if !self.keeps_head {
            return Ok(self.done());
        }
        let Some(reply) = reply else {
            return Ok(self.done());
        };
        let mut context = self.context.clone();
        if self.open.is_some() {
            context.messages.pop();
        }
        context.messages.push(reply);
        let rendering = match self.family.render(&context, false) {
            Ok(rendering) => rendering,
            // A reply that the template does not render has no head.
            // A cut reply that fails its seam check has none.
            Err(FamilyError::Client(_)) => return Ok(self.done()),
            Err(error) => return Err(error.into()),
        };
        if context.open().is_some() {
            return Ok(self.cut_head(&rendering, context.messages.len() - 1));
        }
        // The state has the boundaries of the build, the cut and the seam that it started from too.
        let inserted: Vec<Boundary> = self
            .recipes
            .iter()
            .map(|recipe| recipe.boundary)
            .filter(|boundary| self.rendering.boundaries.iter().all(|b| b.tokens != boundary.tokens))
            .collect();
        let recipes = recipes_with_cuts(self.root, &rendering, &inserted)?;
        // The slot holds the prompt and each sampled token but the last.
        let fed = self.prompt.len() + self.sampled.len() - 1;
        let tokens = &rendering.tokens[..rendering.end];
        let in_slot = tokens.len() >= fed
            && tokens[..self.prompt.len()] == self.prompt[..]
            && tokens[self.prompt.len()..fed] == self.sampled[..self.sampled.len() - 1];
        let same_boundaries =
            recipes.len() == self.recipes.len() + 1 && recipes.iter().zip(&self.recipes).all(|(a, b)| a == b);
        if !same_boundaries || !in_slot || tokens.len() - fed > HEAD_FEED || tokens.len() > self.room {
            return Ok(self.done());
        }
        let last: &BoundaryRecipe = recipes.last().expect("a rendering has a boundary");
        self.stage = Stage::Head {
            recipe: last.recipe,
            boundary: last.boundary,
        };
        if tokens.len() == fed {
            return Ok(Want::Snapshot {
                recipe: last.recipe,
                reason: Reason::PrefillDone,
            });
        }
        Ok(Want::Feed {
            tokens: tokens[fed..].to_vec(),
            logits: false,
            kind: Kind::Prefill,
        })
    }

    /// Hold the head of a cut reply: a sealed snapshot at the seam of its open rendering, the position before its last
    /// token (spec 11.3c). A snapshot holds no logit row, so a continuation from the head feeds that last token for
    /// its first logit row. `rendering` is the open rendering of the reply, the message at index `message`.
    ///
    /// The recipe chains from the last seal of the state: the build feeds the prompt and samples with no seal. The
    /// slot holds the prompt and each sampled token but the last, and the job feeds the rest up to the seam.
    fn cut_head(&mut self, rendering: &Rendering, message: usize) -> Want {
        let tokens = &rendering.tokens[..rendering.end];
        let fed = self.prompt.len() + self.sampled.len() - 1;
        let Some(seam) = tokens.len().checked_sub(1) else {
            return self.done();
        };
        let (base, at) = self
            .recipes
            .last()
            .map_or((self.root, 0), |recipe| (recipe.recipe, recipe.boundary.tokens));
        let in_slot = seam >= fed
            && tokens[..self.prompt.len()] == self.prompt[..]
            && tokens[self.prompt.len()..fed] == self.sampled[..self.sampled.len() - 1];
        if !in_slot || seam - fed > HEAD_FEED || at >= seam {
            return self.done();
        }
        // The family made the spans of this rendering, so a bad span is a defect of the family (spec 8.5).
        let recipe = range_recipe(&base, tokens, &rendering.media, at, seam)
            .unwrap_or_else(|error| panic!("the rendering of a reply has a bad media span: {error}"));
        self.stage = Stage::Head {
            recipe,
            boundary: Boundary { message, tokens: seam },
        };
        if seam == fed {
            return Want::Snapshot {
                recipe,
                reason: Reason::PrefillDone,
            };
        }
        Want::Feed {
            tokens: tokens[fed..seam].to_vec(),
            logits: false,
            kind: Kind::Prefill,
        }
    }

    /// Feed the prompt after the build: the generation prompt, or the open message of a continuation.
    fn feed_prompt(&mut self) -> Result<Want, ServiceError> {
        let tokens = self.prompt[self.end..].to_vec();
        if tokens.is_empty() {
            return Err(ServiceError::internal(
                "the family rendered no generation prompt, so the reply has no first logit row",
            ));
        }
        // With latent steps, the pass of the last prompt token gives the residual of the first latent row.
        if self.latent.is_some() {
            self.stage = Stage::Latent;
            return Ok(Want::Tap {
                tokens,
                kind: Kind::Prefill,
                lens: false,
            });
        }
        Ok(self.row_feed(tokens, Kind::Prefill))
    }

    fn done(&mut self) -> Want {
        self.stage = Stage::Done;
        Want::Done
    }

    /// The id of each call of the reply. The server names the calls (gogi "Generations"). An id is a hash of the
    /// prompt and the reply, with the index of the call. Thus a reply that replays has the same ids, and the
    /// calls of a context with other tokens have other ids.
    fn call_ids(&self, count: usize) -> Vec<String> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nagaya call v1");
        hasher.update(&(self.prompt.len() as u64).to_le_bytes());
        for token in self.prompt.iter().chain(&self.sampled) {
            hasher.update(&token.to_le_bytes());
        }
        let hash = hasher.finalize();
        let stem: String = hash.as_bytes()[..8].iter().map(|byte| format!("{byte:02x}")).collect();
        (0..count).map(|index| format!("call_{stem}_{index}")).collect()
    }
}

/// The text that a parser released after the last token, for the step of that token.
///
/// gogi: "A part's pieces join to its text in done". A token event has one phase, so the tail joins the last step
/// only in the phase of that step. Each marker of a family is one token, so a family never releases text of
/// another phase after the last token. Such text is a defect of the family library (spec 8.5).
pub(super) fn tail_text(parsed: &ParsedReply, phase: Phase) -> Result<String, ServiceError> {
    let mut text = String::new();
    for step in &parsed.tail {
        if step.piece.is_empty() {
            continue;
        }
        if step.phase != phase {
            return Err(ServiceError::internal(format!(
                "the reply parser released {:?} in the phase {:?} after a last token of the phase {phase:?}; \
                 a token event has one phase",
                step.piece, step.phase
            )));
        }
        text.push_str(&step.piece);
    }
    Ok(text)
}

/// The reply in gogi's shape, with the defaults of the fields that the caller sets. Its message form is the assistant
/// message that a client appends. A reply with a call that did not parse has none: a client sends that call back as
/// text.
fn reply_done(parsed: &ParsedReply, ids: &[String]) -> GenerationDone {
    GenerationDone {
        model: String::new(),
        content: parsed.content.clone(),
        reasoning: parsed.reasoning.clone(),
        tool_calls: tool_calls(parsed, ids),
        finish: Finish::EndTurn,
        usage: Usage::default(),
        head: None,
        context: None,
        snapshots: Vec::new(),
        stats: None,
        decisions: Vec::new(),
        seed: None,
        phase: None,
        outputs: Vec::new(),
    }
}

/// The calls of a reply in gogi's shape, with their ids.
pub(super) fn tool_calls(parsed: &ParsedReply, ids: &[String]) -> Vec<ReplyToolCall> {
    parsed
        .calls
        .iter()
        .zip(ids)
        .map(|(call, id)| match call {
            ReplyCall::Parsed { name, input } => ReplyToolCall::Parsed {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                stats: None,
            },
            ReplyCall::Malformed { text, error } => ReplyToolCall::Unparsed {
                id: id.clone(),
                name: None,
                raw: text.clone(),
                error: error.clone(),
                stats: None,
            },
        })
        .collect()
}

impl Then for GenerateThen {
    const NEEDS_STATE: bool = true;
    const REQUEST: &'static str = "generation";

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match (&self.stage, got) {
            (Stage::Building, Got::Built(recipes)) => {
                self.stage = Stage::Sampling;
                self.recipes = recipes;
                self.feed_prompt()
            }
            // A continuation from a held seal in its open text, such as the head of a cut reply, feeds the rest.
            (Stage::Building, Got::Resumed(recipes)) => {
                self.stage = Stage::Sampling;
                self.end = recipes.last().expect("a resumed build has a seal").boundary.tokens;
                self.recipes = recipes;
                self.feed_prompt()
            }
            (Stage::Latent, Got::Tapped { row, residual, .. }) => self.latent_step(&row, &residual),
            (Stage::Latent, Got::Fed(Some(row))) => {
                self.end_latent(&row)?;
                self.sample(&row)
            }
            (Stage::Sampling, Got::Fed(Some(row))) if self.tracing.is_none() => self.sample(&row),
            (Stage::Sampling, Got::Tapped { row, residual, lens }) if self.tracing.is_some() => {
                let tracing = self.tracing.as_mut().expect("the arm holds a trace");
                tracing.residual = Some(f32s(&residual));
                tracing.lens = lens.as_deref().map(f32s);
                self.sample(&row)
            }
            (Stage::Head { recipe, .. }, Got::Fed(None)) => Ok(Want::Snapshot {
                recipe: *recipe,
                reason: Reason::PrefillDone,
            }),
            (Stage::Head { boundary, .. }, Got::Held(id)) => {
                self.head = id.map(|id| (*boundary, id));
                Ok(self.done())
            }
            (_, got) => Err(ServiceError::internal(format!(
                "a generation got {got:?} where it did not expect it"
            ))),
        }
    }
}

/// A seed for a request that sends none. It comes from the hasher keys of the standard library, which the
/// operating system seeds.
pub(super) fn fresh_seed() -> u64 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(0);
    hasher.finish()
}

impl Service {
    /// Sample a reply after a context (gogi "Generations").
    ///
    /// The route chose this service by the request's `model`. The generation builds its context, as a build does,
    /// and samples from the head. Each sampled token goes to `events` as it is sampled, when the caller streams. A
    /// full channel holds the reply back, and the reply drops no event. A closed receiver cancels the reply: the call
    /// gives [`ServiceError::Cancelled`] and holds no head.
    pub fn generate(
        &self,
        request: &GenerationRequest,
        events: Option<SyncSender<TokenEvent>>,
    ) -> Result<GenerationDone, ServiceError> {
        let (done, trace) = self.generate_traced(request, events, None)?;
        assert!(trace.is_none(), "a generation with no trace gives none");
        Ok(done)
    }

    /// Sample a reply after a context, as [`Service::generate`] does, and record its trace when `tracing` is given
    /// (`docs/lab.md`, "`trace`"). The trace changes no token and no member of the response.
    pub(crate) fn generate_traced(
        &self,
        request: &GenerationRequest,
        events: Option<SyncSender<TokenEvent>>,
        tracing: Option<Tracing>,
    ) -> Result<(GenerationDone, Option<Trace>), ServiceError> {
        let Prepared { then, parts, facts } = self.prepare(request, events, tracing, None)?;
        // The checks passed, so the record of the context is held, also when the job fails (spec 11.13).
        self.hold_record(&request.context)?;
        let schedule = Schedule::of(request.class, request.priority);
        let finished = self.run_context(|tx| parts.body(then, tx), schedule, None)?;
        let (mut done, trace) = self.answer(finished, &facts);
        self.name_reply(&request.context, &mut done)?;
        Ok((done, trace))
    }

    /// Check the latent steps of a request against the model (docs/lab.md, "Latent steps in a request"). A model with
    /// no tap `residual`, or a family with no latent rows, is a 400 that names `latent`.
    pub(crate) fn check_latent(&self, plan: &Latent, traced: bool) -> Result<(), ServiceError> {
        if traced {
            return Err(ServiceError::internal(
                "a generation with a trace takes no latent steps",
            ));
        }
        // The route sends a plan of 1 to MAX_STEPS steps. Another count is a defect of the route.
        if plan.steps == 0 || plan.steps > nagaya_council::latent::MAX_STEPS {
            return Err(ServiceError::internal(format!(
                "a plan of latent steps has {} steps; the route sends 1 to {}",
                plan.steps,
                nagaya_council::latent::MAX_STEPS
            )));
        }
        self.takes_latent(plan.map.width)
    }

    /// Check that the model takes latent rows of `width` values: its engine has the tap `residual`, and its family has
    /// latent rows. A fault is a 400 that names `latent`.
    pub(crate) fn takes_latent(&self, width: usize) -> Result<(), ServiceError> {
        let model = &self.config.model;
        if self
            .shared
            .limits
            .taps
            .iter()
            .all(|tap| tap.name != super::build::RESIDUAL_TAP)
        {
            return Err(Invalid::param(
                "latent",
                format!(
                    "nothing ran: the engine of {model} has no tap `{}`, so it takes no latent steps; send the \
                     request with no latent",
                    super::build::RESIDUAL_TAP
                ),
            )
            .into());
        }
        if self.family.latent_record(0, 1, width).is_none() {
            return Err(Invalid::param(
                "latent",
                format!("nothing ran: the family of {model} has no latent rows; send the request with no latent"),
            )
            .into());
        }
        Ok(())
    }

    /// Check a generation request, render its context, and plan its job (spec 11.3c). The work of the job is a
    /// [`GenerateThen`]; the ensemble of `docs/lab.md` wraps it.
    pub(crate) fn prepare(
        &self,
        request: &GenerationRequest,
        events: Option<SyncSender<TokenEvent>>,
        tracing: Option<Tracing>,
        latent: Option<Latent>,
    ) -> Result<Prepared, ServiceError> {
        request.validate()?;
        let read = self.check_objects(&gogi_proto::objects::objects_of(&request.context.messages))?;
        if let Some(output) = &request.output {
            return Err(Invalid::param(
                "output.mime",
                format!(
                    "nothing ran: output.mime is {}, and {} writes text only; send the generation without output",
                    output.mime, self.config.model
                ),
            )
            .into());
        }
        if !request.decisions.is_empty() {
            return Err(Invalid::param(
                "decisions",
                "riding decisions need the ride capability, which this model does not have",
            )
            .into());
        }
        let context_tokens = self.config.context_tokens;
        let max_tokens = request.max_tokens as usize;
        if max_tokens > context_tokens {
            return Err(Invalid::param("max_tokens", format!("max_tokens is 1 to {context_tokens}")).into());
        }
        let sample = request.sample.clone().unwrap_or_default();
        let defaults = self.config.sample;
        let settings = Settings {
            temperature: sample.temperature.unwrap_or(defaults.temperature),
            top_p: sample.top_p.unwrap_or(defaults.top_p),
            top_k: sample.top_k.unwrap_or(defaults.top_k),
            min_p: sample.min_p.unwrap_or(defaults.min_p),
            seed: sample.seed.unwrap_or_else(fresh_seed),
        };
        settings
            .validate()
            .map_err(|error| Invalid::param("sample", error.to_string()))?;
        let context = &request.context;
        // An open message renders in place of the generation prompt, and the family checks its seams.
        let rendering = self.family.render(context, true)?;
        let prompt = rendering.tokens.len();
        if let Some(plan) = &latent {
            self.check_latent(plan, tracing.is_some())?;
        }
        let steps = latent.as_ref().map_or(0, |plan| plan.steps);
        if prompt + steps >= context_tokens {
            return Err(ServiceError::ContextFull {
                prompt_tokens: (prompt + steps) as u64,
                context_tokens: context_tokens as u64,
            });
        }
        // The build makes the context and the generation feeds the generation prompt. A continuation builds the
        // context before its open message and feeds the open message (spec 11.3c).
        let built = match context.open() {
            Some(_) => self.open_plan(context, &rendering)?,
            None => Rendering {
                tokens: rendering.tokens[..rendering.end].to_vec(),
                end: rendering.end,
                boundaries: rendering.boundaries.clone(),
                control_text: Vec::new(),
                media: rendering.media.clone(),
            },
        };
        let steer = self.steer(context, "")?;
        let root = steer.root;
        // A plan with no boundary builds nothing, and the generation feeds the whole prompt.
        let recipes = super::plan_recipes(root, &built)?;
        let end = built.end;
        let room = (prompt + steps + max_tokens + HEAD_FEED).min(context_tokens);
        let tokens: Arc<[u32]> = built.tokens[..end].into();
        let (cuts, extension) = match context.open() {
            Some(_) => self.starts(root, context, &rendering.tokens, &built)?,
            None => (self.cuts(root, context, &built)?, None),
        };
        // The rows of each span of the prompt. A prompt with no build plan feeds its spans in the work.
        let media = self.media_rows(&rendering)?;
        let then = GenerateThen {
            family: Arc::clone(&self.family),
            context: context.clone(),
            open: context.open().cloned(),
            root,
            recipes: recipes.clone(),
            rendering: built,
            prompt: rendering.tokens.clone(),
            end,
            settings,
            max_tokens,
            context_tokens,
            room,
            parser: Some(self.family.reply_parser(ReplyStart::of(context), &context.tools)?),
            events,
            started: Instant::now(),
            stage: Stage::Building,
            sampled: Vec::new(),
            reasoning: 0,
            stats: [PartStats::default(); 2],
            finish: None,
            parsed: None,
            head: None,
            // A head names the tokens of the slot, and a latent row is not a token. So latent steps hold no head.
            keeps_head: self.shared.limits.feed.is_none() && latent.is_none(),
            next_phase: None,
            tracing,
            dropped: self
                .config
                .speech
                .as_ref()
                .map(|speech| speech.markers().to_vec())
                .unwrap_or_default(),
            latent: latent.map(|plan| LatentRun {
                plan,
                fed: 0,
                read: Vec::new(),
            }),
        };
        let parts = JobParts {
            shared: Arc::clone(&self.shared),
            tokens,
            recipes,
            room,
            media,
            cuts,
            extension,
            root,
            control: steer.control,
        };
        Ok(Prepared {
            then,
            parts,
            facts: Facts { prompt, settings, read },
        })
    }

    /// The response of a generation from its finished job.
    pub(crate) fn answer(
        &self,
        mut finished: Finished<GenerateThen>,
        facts: &Facts,
    ) -> (GenerationDone, Option<Trace>) {
        let Facts { prompt, settings, read } = facts.clone();
        let ids = finished
            .then
            .call_ids(finished.then.parsed.as_ref().map_or(0, |p| p.calls.len()));
        let trace = finished.then.take_trace();
        let reply = finished.then.reply();
        let mut held = super::held_boundaries(&finished.outcome);
        held.extend(reply.head);
        let kept = finished.outcome.kept as u64;
        let snapshots = self.infos(&held);
        // The store can evict the head before the list is read. A response names only a head that it lists.
        let head = reply
            .head
            .map(|(_, id)| snapshot_id(&id))
            .filter(|head| snapshots.iter().any(|snapshot| snapshot.id == *head));
        let done = GenerationDone {
            model: self.config.model.clone(),
            content: reply.parsed.content.clone(),
            reasoning: reply.parsed.reasoning.clone(),
            tool_calls: tool_calls(&reply.parsed, &ids),
            finish: reply.finish,
            usage: Usage {
                prompt: prompt as u64,
                kept,
                fed: prompt as u64 - kept,
                completion: reply.tokens.len() as u64,
                reasoning: reply.reasoning,
                media: read,
            },
            head,
            context: None,
            snapshots,
            stats: Some(ReplyStats {
                reasoning: reply.reasoning_stats,
                content: reply.content_stats,
            }),
            decisions: Vec::new(),
            seed: Some(settings.seed),
            phase: reply.phase,
            outputs: Vec::new(),
        };
        (done, trace)
    }
}

/// A generation before its job: the work, the parts of the body, and the facts of the response.
pub(crate) struct Prepared {
    pub(crate) then: GenerateThen,
    pub(crate) parts: JobParts,
    pub(crate) facts: Facts,
}

/// The facts of a generation that its response names.
#[derive(Clone)]
pub(crate) struct Facts {
    pub(crate) prompt: usize,
    pub(crate) settings: Settings,
    pub(crate) read: MediaRead,
}

/// What the body of a generation job takes, other than its work.
pub(crate) struct JobParts {
    shared: Arc<Shared>,
    tokens: Arc<[u32]>,
    recipes: Vec<BoundaryRecipe>,
    room: usize,
    media: Arc<[nagaya_council::family::MediaInput]>,
    cuts: Vec<Cut>,
    extension: Option<Extension>,
    root: RecipeId,
    control: Option<JobControl>,
}

impl JobParts {
    /// The body of the job, with `then` as its work.
    pub(crate) fn body<T: Then>(self, then: T, tx: Sender<Result<Finished<T>, ServiceError>>) -> ContextBody<T> {
        ContextBody::with(self.shared, self.tokens, self.recipes, self.room, then, tx)
            .media(self.media)
            .cuts(self.cuts)
            .extension(self.extension)
            .control(self.root, self.control)
    }
}
