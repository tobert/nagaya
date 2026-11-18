//! Decisions (gogi "Decisions"): a case read after each context, with the reads pooled.

use std::sync::Arc;
use std::time::{Duration, Instant};

use gogi_proto::context::{Context, Usage};
use gogi_proto::decision::{DecisionRequest, DecisionResponse, Read};
use gogi_proto::spec::{QuestionKind, parse_spec};
use gogi_proto::{Invalid, math};
use nagaya_abi::Kind;
use nagaya_council::build::boundary_recipes;
use nagaya_council::decision::{answers, asked, case_text};
use nagaya_council::family::{
    Advance, Boundary, Continues, DecisionInput, MediaInput, ReadPlan, ReadStep, Rendering, WriteStep,
};
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
    /// True when the build fed the prefix with the context, because the prefix ends at a head (spec 11.11, "The plain
    /// read of the LFM3 family").
    prefix_built: bool,
    /// True while the slot holds the state after the prefix.
    at_prefix: bool,
    marked: bool,
    /// The step that waits for a mark or a rewind.
    waiting: Option<Waiting>,
    /// The step whose logit row the feed gives.
    reading: Option<ReadStep>,
    /// The `text` question that the model writes now, and the tokens of its answer so far.
    writing: Option<(WriteStep, Vec<u32>)>,
    /// True while the feed after a written answer runs.
    closing: bool,
    /// The letters of the last step, until the plan takes them.
    last: Option<Vec<f64>>,
    /// Each written answer: the index of its question in the spec, and its text.
    pub described: Vec<(usize, String)>,
    /// The tokens that the model wrote.
    pub written: usize,
}

/// A step that starts from the state after the prefix, or after the last written answer.
enum Waiting {
    Read(ReadStep),
    Write(WriteStep),
}

/// The most likely token of a logit row that is not in `never`. The lowest id wins a tie (spec 11.11, "Text questions
/// of the LFM3 family").
fn most_likely(row: &[f32], never: &[u32]) -> Result<u32, ServiceError> {
    let mut best: Option<(u32, f32)> = None;
    for (id, &logit) in row.iter().enumerate() {
        if logit.is_nan() {
            return Err(ServiceError::internal(format!("the logit of token {id} is NaN")));
        }
        let id = id as u32;
        if !never.contains(&id) && best.is_none_or(|(_, top)| logit > top) {
            best = Some((id, logit));
        }
    }
    best.map(|(id, _)| id)
        .ok_or_else(|| ServiceError::internal("a logit row has no candidate token"))
}

impl DecisionThen {
    fn new(plan: Box<dyn ReadPlan>, asked: usize, prefix_built: bool) -> Self {
        Self {
            plan,
            prefix_built,
            logprobs: vec![None; asked],
            rendered_hash: None,
            fed: 0,
            at_prefix: false,
            marked: false,
            waiting: None,
            reading: None,
            writing: None,
            closing: false,
            last: None,
            described: Vec::new(),
            written: 0,
        }
    }

    /// Start the answer of a `text` question from the state after the prefix or after the answer before.
    fn start(&mut self, step: WriteStep) -> Want {
        self.fed += step.feed.len();
        self.at_prefix = false;
        let tokens = step.feed.clone();
        self.writing = Some((step, Vec::new()));
        Want::Feed {
            tokens,
            logits: true,
            kind: Kind::Read,
        }
    }

    /// Take the logit row after the turn of a `text` question, or after a token of its answer.
    fn write(&mut self, row: &[f32]) -> Result<Want, ServiceError> {
        let (step, mut tokens) = self.writing.take().expect("a row of an answer belongs to a write");
        let token = most_likely(row, &step.never)?;
        let stopped = step.stop.contains(&token);
        if !stopped {
            tokens.push(token);
        }
        if !stopped && tokens.len() < step.max_tokens {
            self.writing = Some((step, tokens));
            self.fed += 1;
            return Ok(Want::Feed {
                tokens: vec![token],
                logits: true,
                kind: Kind::Read,
            });
        }
        // The answer is whole. The slot holds every token of it but the last, when `max_tokens` ended it.
        let written = self.plan.wrote(&tokens)?;
        self.written += tokens.len();
        self.described.push((step.question, written.text));
        let mut feed: Vec<u32> = if stopped { Vec::new() } else { vec![token] };
        feed.extend(written.close);
        if feed.is_empty() {
            return Err(ServiceError::internal("the plan closes a written answer with no token"));
        }
        self.fed += feed.len();
        self.closing = true;
        Ok(Want::Feed {
            tokens: feed,
            logits: false,
            kind: Kind::Read,
        })
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
        // A `text` question comes before the step after it. The plan takes the letters of the last step later.
        if let Some(step) = self.plan.write()? {
            if step.feed.is_empty() || step.max_tokens == 0 {
                return Err(ServiceError::internal(format!(
                    "the write of question {} feeds no token or writes no token",
                    step.question
                )));
            }
            if let Some(last) = last {
                self.last = Some(last.to_vec());
            }
            return match (self.at_prefix, self.marked) {
                (true, _) => Ok(self.start(step)),
                (false, true) => {
                    self.waiting = Some(Waiting::Write(step));
                    Ok(Want::Rewind)
                }
                (false, false) => Err(ServiceError::internal(
                    "a text question starts from the prefix, but the plan left the prefix before a mark",
                )),
            };
        }
        let kept = self.last.take();
        let last = last.or(kept.as_deref());
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
                        self.waiting = Some(Waiting::Read(step));
                        Ok(Want::Mark)
                    }
                    Continues::Prefix if self.marked => {
                        self.waiting = Some(Waiting::Read(step));
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
                if prefix.is_empty() || self.prefix_built {
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
                // After the close of a written answer, the mark of the state before it is stale.
                if self.closing {
                    self.closing = false;
                    self.marked = false;
                }
                self.at_prefix = true;
                self.advance(None)
            }
            // A decision reads no open message, so its build has no extension.
            Got::Resumed(_) => Err(ServiceError::internal(
                "a decision build restored a seal of an open text",
            )),
            Got::Copied => {
                self.at_prefix = true;
                match self.waiting.take().expect("a mark or a rewind has a step that waits") {
                    Waiting::Read(step) => Ok(self.feed(step)),
                    Waiting::Write(step) => Ok(self.start(step)),
                }
            }
            Got::Fed(Some(row)) if self.writing.is_some() => self.write(&row),
            Got::Fed(Some(row)) => {
                let step = self.reading.take().expect("a logit row belongs to a step");
                let lp = log_probs(&row).map_err(|error| ServiceError::internal(error.to_string()))?;
                let letters: Vec<f64> = step.letters.iter().map(|&letter| lp[letter as usize]).collect();
                self.logprobs[step.question] = Some(letters.clone());
                self.advance(Some(&letters))
            }
            Got::Held(_) => Err(ServiceError::internal("a decision read holds no snapshot of its own")),
            Got::Tapped { .. } => Err(ServiceError::internal("a decision read names no tap")),
            Got::Woken => Err(ServiceError::internal("a decision read never waits")),
            Got::Frame(_) | Got::Samples(_) => Err(ServiceError::internal("a decision read makes no audio")),
        }
    }
}

/// The rendering of a context and the prefix of `plan` after it, with a boundary at the end of the prefix. The prefix
/// is the case turn, which counts as the message after the last message of the context.
fn with_case_turn(rendering: Option<&Rendering>, context: Option<&Context>, plan: &dyn ReadPlan) -> Rendering {
    let (mut tokens, mut boundaries, mut media) = match rendering {
        Some(r) => (r.tokens[..r.end].to_vec(), r.boundaries.clone(), r.media.clone()),
        None => (Vec::new(), Vec::new(), Vec::new()),
    };
    tokens.extend_from_slice(plan.prefix());
    boundaries.push(Boundary {
        message: context.map_or(0, |c| c.messages.len()),
        tokens: tokens.len(),
    });
    media.extend(plan.media().into_iter().map(|input| input.span));
    Rendering {
        end: tokens.len(),
        tokens,
        boundaries,
        control_text: Vec::new(),
        media,
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
                .map(|(i, o)| gogi_proto::objects::Named {
                    at: format!("case_media[{i}]"),
                    object: o.clone(),
                    assistant: false,
                }),
        );
        let read = self.check_objects(&objects)?;
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
        let writes = spec
            .questions
            .iter()
            .any(|q| matches!(q.kind, QuestionKind::Text { .. }));
        if writes && !self.family.describes() {
            return Err(Invalid::param(
                "spec_id",
                "nothing was read: the spec holds a text question, which needs the describe capability, and this \
                 model does not have it; use a spec with no text question, or a model that declares describe",
            )
            .into());
        }
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
        // Each control list is checked before the first read runs (spec 11.12).
        let steers = contexts
            .iter()
            .enumerate()
            .map(|(index, context)| {
                context
                    .map(|c| self.steer(c, &format!("contexts[{index}].")))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        for ((index, context), steer) in contexts.into_iter().enumerate().zip(steers) {
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
            // The case media follow the case in the read (spec 11.11, "Media in a decision").
            let plan = self.family.decision_read_with_media(
                &DecisionInput {
                    context: context.zip(rendering.as_ref()),
                    spec: &spec,
                    case: &case,
                    asked: &asked,
                },
                &request.case_media,
            )?;
            if control.is_none() {
                control = Some(signals(plan.control_text()));
            }
            let room = end + plan.prefix().len() + plan.room();
            // A prefix that ends at a head is built with the context, so the store holds the head of the case and a
            // later decision on the same case starts from it (spec 11.11, "The plain read of the LFM3 family").
            let head = plan.prefix_is_head() && !plan.prefix().is_empty();
            let built = end + if head { plan.prefix().len() } else { 0 };
            let (root, job_control, layer_loop) = match steer {
                Some(steer) => (steer.root, steer.control, steer.layer_loop),
                None => (self.root, None, None),
            };
            let (mut tokens, mut recipes, cuts, media): (Arc<[u32]>, _, _, _) = match (&rendering, context) {
                (Some(r), Some(c)) => (
                    r.tokens[..r.end].into(),
                    boundary_recipes(root, r)?,
                    self.cuts(root, c, r)?,
                    self.media_rows(r)?,
                ),
                _ => (Arc::from(Vec::new()), Vec::new(), Vec::new(), Arc::from(Vec::new())),
            };
            if head {
                let whole = with_case_turn(rendering.as_ref(), context, plan.as_ref());
                recipes = boundary_recipes(root, &whole)?;
                tokens = whole.tokens.into();
            }
            // The rows of the read's own spans follow the rows of the context, in the order of the positions.
            let media: Arc<[MediaInput]> = media.iter().cloned().chain(plan.media()).collect();
            let then = DecisionThen::new(plan, asked.len(), head);
            let shared = Arc::clone(&self.shared);
            let job = |tx| {
                ContextBody::with(shared, tokens, recipes, room, then, tx)
                    .media(media)
                    .cuts(cuts)
                    .control(root, job_control, layer_loop)
            };
            let finished = self.run_context(job, schedule, Some(deadline))?;
            let then = finished.then;
            let kept = finished.outcome.kept;
            let fed = built - kept + then.fed;
            usage.prompt += (kept + fed) as u64;
            usage.kept += kept as u64;
            usage.fed += fed as u64;
            usage.completion += then.written as u64;
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
                // The read names the head of its context. With a built prefix, the last boundary is the head of the case.
                snapshot: context.and(
                    finished
                        .outcome
                        .held
                        .iter()
                        .rev()
                        .nth(usize::from(head))
                        .copied()
                        .flatten()
                        .map(|id| snapshot_id(&id)),
                ),
                // The written answers, by question id, in the order of the spec (spec 11.11).
                described: writes.then(|| {
                    then.described
                        .iter()
                        .map(|(question, text)| (spec.questions[*question].id.clone(), text.clone()))
                        .collect()
                }),
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
        usage.media = read;
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
        // A decision holds the record of each context that it read, and it names none (spec 11.13).
        for context in &request.contexts {
            self.hold_record(context)?;
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::most_likely;

    #[test]
    fn the_most_likely_token_is_the_lowest_id_of_the_top_logit_outside_never() {
        let row = [0.5, 2.0, 2.0, 1.0];
        assert_eq!(most_likely(&row, &[]).unwrap(), 1, "the lowest id wins a tie");
        assert_eq!(most_likely(&row, &[1]).unwrap(), 2);
        assert_eq!(most_likely(&row, &[1, 2]).unwrap(), 3);
        assert!(most_likely(&row, &[0, 1, 2, 3]).is_err(), "no candidate is a defect");
        assert!(most_likely(&[0.0, f32::NAN], &[]).is_err(), "a NaN is a defect");
    }
}
