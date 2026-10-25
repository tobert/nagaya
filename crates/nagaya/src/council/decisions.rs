//! Decisions (gogi "Decisions"): a case read after each context, with the reads pooled.

use std::sync::Arc;
use std::time::{Duration, Instant};

use gogi_proto::context::{Context, Usage};
use gogi_proto::decision::{DecisionRequest, DecisionResponse, Read};
use gogi_proto::spec::parse_spec;
use gogi_proto::{Invalid, math};
use nagaya_abi::Kind;
use nagaya_council::build::boundary_recipes;
use nagaya_council::decision::{answers, asked, case_text};
use nagaya_council::family::{Advance, Continues, DecisionInput, ReadPlan, ReadStep};
use nagaya_council::read::log_probs;

use super::build::{ContextBody, Got, Then, Want};
use super::error::ServiceError;
use super::reads::signals;
use super::{Schedule, Service, snapshot_id};

/// The work of a decision read after its build: the plan's prefix, then each step of the plan.
///
/// A step that continues from the prefix starts from the state after the prefix. The first such step marks that
/// state, and each later one puts it back.
pub(crate) struct DecisionThen {
    plan: Box<dyn ReadPlan>,
    /// The letter log probabilities of each asked question, by its index in the asked questions.
    pub logprobs: Vec<Option<Vec<f64>>>,
    /// The hash of the text that the read was conditioned on.
    pub rendered_hash: Option<String>,
    /// The tokens that the read fed after the build.
    pub fed: usize,
    /// True while the slot holds the state after the prefix.
    at_prefix: bool,
    marked: bool,
    /// The step that waits for a mark or a rewind.
    waiting: Option<ReadStep>,
    /// The step whose logit row the feed gives.
    reading: Option<ReadStep>,
}

impl DecisionThen {
    fn new(plan: Box<dyn ReadPlan>, asked: usize) -> Self {
        Self {
            plan,
            logprobs: vec![None; asked],
            rendered_hash: None,
            fed: 0,
            at_prefix: false,
            marked: false,
            waiting: None,
            reading: None,
        }
    }

    fn feed(&mut self, step: ReadStep) -> Want {
        self.fed += step.feed.len();
        self.at_prefix = false;
        let tokens = step.feed.clone();
        self.reading = Some(step);
        Want::Feed {
            tokens,
            logits: true,
            kind: Kind::Read,
        }
    }

    fn advance(&mut self, last: Option<&[f64]>) -> Result<Want, ServiceError> {
        match self.plan.advance(last)? {
            Advance::Done(done) => {
                self.rendered_hash = Some(done.rendered_hash);
                Ok(Want::Done)
            }
            Advance::Step(step) => {
                if step.feed.is_empty() {
                    return Err(ServiceError::internal(format!(
                        "the read step of question {} feeds no token, so it has no logit row",
                        step.question
                    )));
                }
                if step.question >= self.logprobs.len() {
                    return Err(ServiceError::internal(format!(
                        "a read step names question {}, but the decision asks {}",
                        step.question,
                        self.logprobs.len()
                    )));
                }
                match step.continues {
                    Continues::PreviousStep => Ok(self.feed(step)),
                    Continues::Prefix if self.at_prefix && self.marked => Ok(self.feed(step)),
                    Continues::Prefix if self.at_prefix => {
                        self.marked = true;
                        self.waiting = Some(step);
                        Ok(Want::Mark)
                    }
                    Continues::Prefix if self.marked => {
                        self.waiting = Some(step);
                        Ok(Want::Rewind)
                    }
                    Continues::Prefix => Err(ServiceError::internal(
                        "a read step continues from the prefix, but the plan left the prefix before a mark",
                    )),
                }
            }
        }
    }
}

impl Then for DecisionThen {
    const NEEDS_STATE: bool = true;
    const REQUEST: &'static str = "decision";

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match got {
            Got::Built(_) => {
                let prefix = self.plan.prefix().to_vec();
                if prefix.is_empty() {
                    self.at_prefix = true;
                    return self.advance(None);
                }
                self.fed += prefix.len();
                Ok(Want::Feed {
                    tokens: prefix,
                    logits: false,
                    kind: Kind::Read,
                })
            }
            Got::Fed(None) => {
                self.at_prefix = true;
                self.advance(None)
            }
            // A decision reads no open message, so its build has no extension.
            Got::Resumed(_) => Err(ServiceError::internal(
                "a decision build restored a seal of an open text",
            )),
            Got::Copied => {
                self.at_prefix = true;
                let step = self.waiting.take().expect("a mark or a rewind has a step that waits");
                Ok(self.feed(step))
            }
            Got::Fed(Some(row)) => {
                let step = self.reading.take().expect("a logit row belongs to a step");
                let lp = log_probs(&row).map_err(|error| ServiceError::internal(error.to_string()))?;
                let letters: Vec<f64> = step.letters.iter().map(|&letter| lp[letter as usize]).collect();
                self.logprobs[step.question] = Some(letters.clone());
                self.advance(Some(&letters))
            }
            Got::Held(_) => Err(ServiceError::internal("a decision read holds no snapshot of its own")),
        }
    }
}

impl Service {
    /// Read a case after each context, and pool the reads (gogi "Decisions").
    ///
    /// The route chose this service by the request's `model`. Each read builds its context, as a build does, and
    /// reads from the head. The reads run one after another, in request order. A decision past its `timeout_ms` is
    /// [`ServiceError::Timeout`]. Numbers that fail `gogi_proto::math::verify` are an internal error, never a
    /// response.
    pub fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, ServiceError> {
        let started = Instant::now();
        request.validate()?;
        let mut objects = Self::objects_in(&request.contexts);
        objects.extend(
            request
                .case_media
                .iter()
                .enumerate()
                .map(|(i, o)| (format!("case_media[{i}]"), o.clone())),
        );
        self.reads_text_only(&objects)?;
        if request.think.is_some_and(|think| think.tokens > 0) {
            return Err(Invalid::param(
                "think",
                "think above 0 tokens needs the think capability, which this model does not have",
            )
            .into());
        }
        if request.contexts.len() > self.config.contexts_per_decision {
            return Err(Invalid::param(
                "contexts",
                format!(
                    "the decision has {} contexts; this model reads at most {}",
                    request.contexts.len(),
                    self.config.contexts_per_decision
                ),
            )
            .into());
        }
        let held = self.spec(&request.spec_id)?;
        let spec = parse_spec(&held.spec, &self.config.spec_rules)?;
        let asked = asked(&spec, request.ask.as_deref(), request.options.as_ref())?;
        let case = case_text(&request.case);
        if case.len() > self.config.case_bytes {
            return Err(ServiceError::TooLarge {
                param: "case_bytes".to_owned(),
                message: format!(
                    "nothing was read: the case is {} bytes, and case_bytes is {}; send a shorter case",
                    case.len(),
                    self.config.case_bytes
                ),
            });
        }
        let timeout = request.timeout_ms.unwrap_or(self.config.default_timeout_ms);
        let deadline = started + Duration::from_millis(timeout);
        let schedule = Schedule::of(request.class, request.priority);
        let contexts: Vec<Option<&Context>> = if request.contexts.is_empty() {
            vec![None]
        } else {
            request.contexts.iter().map(Some).collect()
        };
        let mut reads = Vec::with_capacity(contexts.len());
        let mut numbers = Vec::with_capacity(contexts.len());
        let mut usage = Usage::default();
        let mut control = None;
        for (index, context) in contexts.into_iter().enumerate() {
            if Instant::now() >= deadline {
                return Err(ServiceError::Timeout);
            }
            let rendering = context.map(|c| self.family.render(c, false)).transpose()?;
            let end = rendering.as_ref().map_or(0, |r| r.end);
            if end > self.config.decision_context_tokens {
                return Err(ServiceError::TooLarge {
                    param: "decision_context_tokens".to_owned(),
                    message: format!(
                        "nothing was read: context {index} has {end} tokens, and a decision reads at most {}; \
                         shorten the context",
                        self.config.decision_context_tokens
                    ),
                });
            }
            let plan = self.family.decision_read(&DecisionInput {
                context: context.zip(rendering.as_ref()),
                spec: &spec,
                case: &case,
                asked: &asked,
            })?;
            if control.is_none() {
                control = Some(signals(plan.control_text()));
            }
            let room = end + plan.prefix().len();
            let (tokens, recipes, cuts): (Arc<[u32]>, _, _) = match (&rendering, context) {
                (Some(r), Some(c)) => (
                    r.tokens[..r.end].into(),
                    boundary_recipes(self.root, r)?,
                    self.cuts(c, r)?,
                ),
                _ => (Arc::from(Vec::new()), Vec::new(), Vec::new()),
            };
            let then = DecisionThen::new(plan, asked.len());
            let shared = Arc::clone(&self.shared);
            let job = |tx| ContextBody::with(shared, tokens, recipes, room, then, tx).cuts(cuts);
            let finished = self.run_context(job, schedule, Some(deadline))?;
            let then = finished.then;
            let kept = finished.outcome.kept;
            let fed = end - kept + then.fed;
            usage.prompt += (kept + fed) as u64;
            usage.kept += kept as u64;
            usage.fed += fed as u64;
            let read: Vec<Vec<f64>> = then
                .logprobs
                .into_iter()
                .enumerate()
                .map(|(slot, letters)| {
                    letters.ok_or_else(|| {
                        ServiceError::internal(format!(
                            "the read plan never read question {}",
                            spec.questions[asked[slot].question].id
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            numbers.push(read);
            reads.push(Read {
                context: context.map(|_| index as u64),
                snapshot: context.and(
                    finished
                        .outcome
                        .held
                        .last()
                        .copied()
                        .flatten()
                        .map(|id| snapshot_id(&id)),
                ),
                described: None,
                answers: Default::default(),
                rendered_hash: then
                    .rendered_hash
                    .ok_or_else(|| ServiceError::internal("a read plan ended with no rendered hash"))
                    .and_then(|hash| gogi_proto::ContentId::parse(hash).map_err(ServiceError::internal))?,
                thought: None,
                tokens: None,
                ms: None,
            });
        }
        let pool = request.pool.clone().unwrap_or_default();
        let answers = answers(&spec, &asked, &numbers, &pool)
            .map_err(|error| ServiceError::internal(format!("the reads cannot be pooled: {error}")))?;
        for (read, answers) in reads.iter_mut().zip(answers.reads) {
            read.answers = answers;
        }
        let mut identity = self.identity();
        identity.spec_id = Some(held.spec_id.clone());
        let response = DecisionResponse {
            model: self.config.model.clone(),
            answers: answers.pooled,
            reads,
            pool: answers.pool,
            signals: control,
            identity,
            usage,
            queue_ms: None,
            ms: Some(started.elapsed().as_secs_f64() * 1000.0),
        };
        // A number that fails the check is a defect, never an answer (gogi "Errors").
        math::verify(&response, &request.expect())
            .map_err(|mismatch| ServiceError::internal(format!("a decision failed its check: {mismatch}")))?;
        Ok(response)
    }
}
