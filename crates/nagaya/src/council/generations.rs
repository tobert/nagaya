//! Generations (gogi "Generations"): a reply sampled after a context, with the sampler of spec 11.2.

use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::Instant;

use gogi_proto::Invalid;
use gogi_proto::context::{Context, Usage};
use gogi_proto::generation::{
    Finish, GenerationDone, GenerationRequest, Phase, ReplyStats, ReplyToolCall, TokenEvent, TokenStats,
};
use nagaya_abi::Kind;
use nagaya_council::build::{BoundaryRecipe, boundary_recipes, recipes_with_cut};
use nagaya_council::family::{Boundary, Family, FamilyError, ParsedReply, Rendering, ReplyCall, ReplyParser};
use nagaya_council::sampler::{Settings, pick};
use nagaya_store::{ContentId, Reason, RecipeId};

use super::build::{ContextBody, Got, Then, Want};
use super::error::ServiceError;
use super::{Schedule, Service, snapshot_id};

/// The most tokens that a generation feeds after its reply to hold the head (gogi "Generations": "at most a few
/// tokens of feeding").
pub const HEAD_FEED: usize = 8;

/// The stats of one part of a reply, from the untempered probability and the entropy of each token.
#[derive(Clone, Copy, Debug, Default)]
struct PartStats {
    tokens: u64,
    logprob_sum: f64,
    logprob_min: f64,
    entropy_sum: f64,
    entropy_max: f64,
}

impl PartStats {
    fn push(&mut self, p: f64, h: f64) {
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
    fn finish(&self) -> Option<TokenStats> {
        (self.tokens > 0).then(|| TokenStats {
            tokens: self.tokens,
            mean_logprob: self.logprob_sum / self.tokens as f64,
            min_logprob: self.logprob_min,
            mean_entropy: self.entropy_sum / self.tokens as f64,
            max_entropy: self.entropy_max,
        })
    }
}

/// Where a generation is.
enum Stage {
    /// The build is running.
    Building,
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
}

/// The work of a generation after its build: feed the generation prompt, sample each token, and hold the head
/// after the reply when it costs at most [`HEAD_FEED`] tokens.
pub(crate) struct GenerateThen {
    family: Arc<dyn Family>,
    context: Context,
    root: RecipeId,
    /// The recipes of the boundaries of the context. A build that starts from a cut gives them the cut too.
    recipes: Vec<RecipeId>,
    /// The rendering of the context, without the generation prompt.
    rendering: Rendering,
    /// The cut that the build started from.
    cut: Option<Boundary>,
    /// The tokens of the prompt: the context, then the generation prompt.
    prompt: Vec<u32>,
    /// The tokens of the context, without the generation prompt.
    end: usize,
    settings: Settings,
    max_tokens: usize,
    context_tokens: usize,
    /// The most tokens that the slot of the job holds.
    room: usize,
    parser: Option<Box<dyn ReplyParser>>,
    events: Option<Sender<TokenEvent>>,
    started: Instant,
    stage: Stage,
    sampled: Vec<u32>,
    reasoning: u64,
    stats: [PartStats; 2],
    finish: Option<Finish>,
    parsed: Option<ParsedReply>,
    head: Option<(Boundary, ContentId)>,
}

impl GenerateThen {
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
        }
    }

    /// Sample the next token from a logit row, and give it to the parser and to the stream.
    fn sample(&mut self, row: &[f32]) -> Result<Want, ServiceError> {
        let position = self.prompt.len() + self.sampled.len();
        let chosen = pick(row, &self.settings, position as u64)
            .map_err(|error| ServiceError::internal(format!("the sample at position {position} failed: {error}")))?;
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
        // The reply ends at this token when the parser ended it, or when it reaches a limit.
        let end = if ended {
            Some(None)
        } else if n == self.max_tokens {
            Some(Some(Finish::MaxTokens))
        } else if self.prompt.len() + n >= self.context_tokens {
            Some(Some(Finish::ContextFull))
        } else {
            None
        };
        let mut piece = step.piece;
        let parsed = match end {
            Some(_) => {
                let parsed = self.parser.take().expect("a reply finishes once").finish();
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
            _ => Ok(Want::Feed {
                tokens: vec![chosen.token],
                logits: true,
                kind: Kind::Decode,
            }),
        }
    }

    /// End the reply. `cut` names why a reply that did not end its turn stopped. `parsed` is the whole reply.
    fn finish(&mut self, cut: Option<Finish>, parsed: ParsedReply) -> Result<Want, ServiceError> {
        self.finish = Some(match cut {
            Some(cut) => cut,
            None if parsed.calls.is_empty() => Finish::EndTurn,
            None => Finish::ToolUse,
        });
        let reply = reply_message(&parsed, &self.call_ids(parsed.calls.len()));
        self.parsed = Some(parsed);
        let Some(reply) = reply else {
            return Ok(self.done());
        };
        let mut context = self.context.clone();
        context.messages.push(reply);
        let rendering = match self.family.render(&context, false) {
            Ok(rendering) => rendering,
            // A reply that the template does not render has no head.
            Err(FamilyError::Client(_)) => return Ok(self.done()),
            Err(error) => return Err(error.into()),
        };
        // The state has the boundaries of the build, the cut too.
        let recipes = match self.cut {
            Some(cut) => recipes_with_cut(self.root, &rendering, cut)?,
            None => boundary_recipes(self.root, &rendering)?,
        };
        // The slot holds the prompt and each sampled token but the last.
        let fed = self.prompt.len() + self.sampled.len() - 1;
        let tokens = &rendering.tokens[..rendering.end];
        let same_boundaries =
            recipes.len() == self.recipes.len() + 1 && recipes.iter().zip(&self.recipes).all(|(a, b)| a.recipe == *b);
        let in_slot = tokens.len() >= fed
            && tokens[..self.prompt.len()] == self.prompt[..]
            && tokens[self.prompt.len()..fed] == self.sampled[..self.sampled.len() - 1];
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
fn tail_text(parsed: &ParsedReply, phase: Phase) -> Result<String, ServiceError> {
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

/// The reply as the assistant message that a client appends. A reply with a call that did not parse has none: a
/// client sends that call back as text.
fn reply_message(parsed: &ParsedReply, ids: &[String]) -> Option<gogi_proto::message::Message> {
    let done = GenerationDone {
        model: String::new(),
        content: parsed.content.clone(),
        reasoning: parsed.reasoning.clone(),
        tool_calls: tool_calls(parsed, ids),
        finish: Finish::EndTurn,
        usage: Usage::default(),
        head: None,
        snapshots: Vec::new(),
        stats: None,
        decisions: Vec::new(),
        seed: None,
    };
    done.as_message()
}

/// The calls of a reply in gogi's shape, with their ids.
fn tool_calls(parsed: &ParsedReply, ids: &[String]) -> Vec<ReplyToolCall> {
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

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match (&self.stage, got) {
            (Stage::Building, Got::Built(cut)) => {
                self.stage = Stage::Sampling;
                if let Some(cut) = cut {
                    self.cut = Some(cut);
                    self.recipes = recipes_with_cut(self.root, &self.rendering, cut)?
                        .iter()
                        .map(|recipe| recipe.recipe)
                        .collect();
                }
                let tokens = self.prompt[self.end..].to_vec();
                if tokens.is_empty() {
                    return Err(ServiceError::internal(
                        "the family rendered no generation prompt, so the reply has no first logit row",
                    ));
                }
                Ok(Want::Feed {
                    tokens,
                    logits: true,
                    kind: Kind::Prefill,
                })
            }
            (Stage::Sampling, Got::Fed(Some(row))) => self.sample(&row),
            (Stage::Head { recipe, .. }, Got::Fed(None)) => Ok(Want::Snapshot {
                recipe: *recipe,
                reason: Reason::PrefillDone,
            }),
            (Stage::Head { boundary, .. }, Got::Held(id)) => {
                self.head = Some((*boundary, id));
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
fn fresh_seed() -> u64 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(0);
    hasher.finish()
}

impl Service {
    /// Sample a reply after a context (gogi "Generations").
    ///
    /// The route chose this service by the request's `model`. The generation builds its context, as a build does,
    /// and samples from the head. Each sampled token goes to `events` as it is sampled, when the caller streams. A
    /// closed receiver cancels the reply: the call gives [`ServiceError::Cancelled`] and holds no head.
    pub fn generate(
        &self,
        request: &GenerationRequest,
        events: Option<Sender<TokenEvent>>,
    ) -> Result<GenerationDone, ServiceError> {
        request.validate()?;
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
        let rendering = self.family.render(context, true)?;
        let prompt = rendering.tokens.len();
        if prompt >= context_tokens {
            return Err(ServiceError::ContextFull {
                prompt_tokens: prompt as u64,
                context_tokens: context_tokens as u64,
            });
        }
        let recipes = boundary_recipes(self.root, &rendering)?;
        let end = rendering.end;
        let room = (prompt + max_tokens + HEAD_FEED).min(context_tokens);
        let then = GenerateThen {
            family: Arc::clone(&self.family),
            context: context.clone(),
            root: self.root,
            recipes: recipes.iter().map(|recipe| recipe.recipe).collect(),
            rendering: Rendering {
                tokens: rendering.tokens[..end].to_vec(),
                end,
                boundaries: rendering.boundaries.clone(),
                control_text: Vec::new(),
            },
            cut: None,
            prompt: rendering.tokens.clone(),
            end,
            settings,
            max_tokens,
            context_tokens,
            room,
            parser: Some(self.family.reply_parser(context.thinks(), &context.tools)?),
            events,
            started: Instant::now(),
            stage: Stage::Building,
            sampled: Vec::new(),
            reasoning: 0,
            stats: [PartStats::default(); 2],
            finish: None,
            parsed: None,
            head: None,
        };
        let tokens: Arc<[u32]> = rendering.tokens[..end].into();
        let shared = Arc::clone(&self.shared);
        let cuts = self.cuts(context, &rendering)?;
        let job = |tx| ContextBody::with(shared, tokens, recipes.clone(), room, then, tx).cuts(cuts);
        let finished = self.run_context(job, Schedule::of(request.class, request.priority), None)?;
        let ids = finished
            .then
            .call_ids(finished.then.parsed.as_ref().map_or(0, |p| p.calls.len()));
        let reply = finished.then.reply();
        let mut held = super::held_boundaries(&finished.outcome);
        held.extend(reply.head);
        let kept = finished.outcome.kept as u64;
        Ok(GenerationDone {
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
            },
            head: reply.head.map(|(_, id)| snapshot_id(&id)),
            snapshots: self.infos(&held),
            stats: Some(ReplyStats {
                reasoning: reply.reasoning_stats,
                content: reply.content_stats,
            }),
            decisions: Vec::new(),
            seed: Some(settings.seed),
        })
    }
}
