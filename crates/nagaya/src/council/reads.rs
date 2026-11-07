//! Raw reads (gogi "Raw reads"): the letter log probabilities after each suffix, read after each context.

use std::sync::Arc;

use gogi_proto::Invalid;
use gogi_proto::context::Usage;
use gogi_proto::decision::{ControlTextHit, Signals};
use gogi_proto::math::log_sum_exp;
use gogi_proto::reads::{RawRead, ReadsRequest, ReadsResponse, TopToken};
use indexmap::IndexMap;
use nagaya_abi::Kind;
use nagaya_council::build::boundary_recipes;
use nagaya_council::family::{ControlHit, FamilyError};
use nagaya_council::read::{log_probs, top};

use super::build::{ContextBody, Got, Then, Want};
use super::error::ServiceError;
use super::{Schedule, Service, snapshot_id};

/// The top tokens of a read when the request names no `top`.
pub const DEFAULT_TOP: usize = 8;

/// The most top tokens that a read lists.
pub const MAX_TOP: usize = 64;

/// The numbers of one suffix after one context.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SuffixRead {
    /// The log probability of each letter, in request order.
    pub logprobs: Vec<f64>,
    /// The most likely tokens.
    pub top: Vec<(u32, f64)>,
}

/// The work of a raw read after its build: each suffix after the context's end.
///
/// The state after the context is marked when there is more than one suffix, and each later suffix starts from it.
pub(crate) struct RawThen {
    suffixes: Arc<Vec<Vec<u32>>>,
    letters: Arc<Vec<u32>>,
    top: usize,
    /// The reads so far, one for each suffix.
    pub reads: Vec<SuffixRead>,
}

impl RawThen {
    fn feed(&self) -> Want {
        Want::Feed {
            tokens: self.suffixes[self.reads.len()].clone(),
            logits: true,
            kind: Kind::Read,
        }
    }
}

impl Then for RawThen {
    const NEEDS_STATE: bool = true;
    const REQUEST: &'static str = "read";

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match got {
            Got::Built(_) if self.suffixes.len() > 1 => Ok(Want::Mark),
            Got::Built(_) | Got::Copied => Ok(self.feed()),
            Got::Fed(Some(row)) => {
                let lp = log_probs(&row).map_err(|error| ServiceError::internal(error.to_string()))?;
                let letters = self.letters.iter().map(|&letter| lp[letter as usize]).collect();
                self.reads.push(SuffixRead {
                    logprobs: letters,
                    top: top(&lp, self.top),
                });
                Ok(if self.reads.len() == self.suffixes.len() {
                    Want::Done
                } else {
                    Want::Rewind
                })
            }
            other => Err(ServiceError::internal(format!(
                "a raw read got {other:?} where it needs a logit row"
            ))),
        }
    }
}

/// The control text of a request, in the shape of gogi's signals.
pub(crate) fn signals(hits: &[ControlHit]) -> Signals {
    Signals {
        control_text: hits
            .iter()
            .map(|hit| ControlTextHit {
                r#where: hit.place.clone(),
                token: hit.token.clone(),
            })
            .collect(),
    }
}

impl Service {
    /// Read letter log probabilities after each suffix, after each context (gogi "Raw reads").
    ///
    /// The route chose this service by the request's `model`. Each read builds its context, as a build does, and
    /// reads from the head.
    pub fn read(&self, request: &ReadsRequest) -> Result<ReadsResponse, ServiceError> {
        request.validate()?;
        let read = self.check_objects(&Self::objects_in(&request.contexts))?;
        let count = request.top.unwrap_or(DEFAULT_TOP);
        if count > MAX_TOP {
            return Err(Invalid::param("top", format!("top is {count}; a read lists at most {MAX_TOP} tokens")).into());
        }
        let raw = self.family.raw_read(&request.letters, &request.suffixes)?;
        if let Some(index) = raw.suffixes.iter().position(Vec::is_empty) {
            return Err(Invalid::param("suffixes", format!("suffix {index} has no token")).into());
        }
        // `request.validate()` above refuses an empty `suffixes` (gogi `ReadsRequest::validate`), and a family gives one
        // token list for each suffix. The test `a_read_with_no_suffix_is_a_400_before_any_family_code_runs` checks it.
        let longest = raw.suffixes.iter().map(Vec::len).max().expect("a request has a suffix");
        let suffixes = Arc::new(raw.suffixes);
        let letters = Arc::new(raw.letters);
        let schedule = Schedule::of(request.class, request.priority);
        let mut reads = Vec::new();
        let mut usage = Usage::default();
        for (index, context) in request.contexts.iter().enumerate() {
            let rendering = self.family.render(context, false)?;
            let end = rendering.end;
            if end + longest > self.config.context_tokens {
                return Err(ServiceError::past_context(end + longest, self.config.context_tokens));
            }
            let recipes = boundary_recipes(self.root, &rendering)?;
            let tokens: Arc<[u32]> = rendering.tokens[..end].into();
            let then = RawThen {
                suffixes: Arc::clone(&suffixes),
                letters: Arc::clone(&letters),
                top: count,
                reads: Vec::new(),
            };
            let shared = Arc::clone(&self.shared);
            let cuts = self.cuts(context, &rendering)?;
            let media = self.media_rows(&rendering)?;
            let job = |tx| {
                ContextBody::with(shared, tokens, recipes, end + longest, then, tx)
                    .media(media)
                    .cuts(cuts)
            };
            let finished = self.run_context(job, schedule, None)?;
            let head = finished
                .outcome
                .held
                .last()
                .copied()
                .flatten()
                .map(|id| snapshot_id(&id));
            for (suffix, numbers) in finished.then.reads.into_iter().enumerate() {
                // The first suffix keeps what the build kept. Each later one starts from the context's end.
                let kept = if suffix == 0 { finished.outcome.kept } else { end };
                let prompt = end + suffixes[suffix].len();
                usage.prompt += prompt as u64;
                usage.kept += kept as u64;
                usage.fed += (prompt - kept) as u64;
                let logprobs: IndexMap<String, f64> = request
                    .letters
                    .iter()
                    .cloned()
                    .zip(numbers.logprobs.iter().copied())
                    .collect();
                let top = numbers
                    .top
                    .iter()
                    .map(|&(id, logprob)| {
                        Ok(TopToken {
                            token: self.piece(id)?,
                            id: Some(id),
                            logprob,
                        })
                    })
                    .collect::<Result<Vec<_>, ServiceError>>()?;
                reads.push(RawRead {
                    context: index as u64,
                    snapshot: head.clone(),
                    suffix: suffix as u64,
                    mass: log_sum_exp(&numbers.logprobs),
                    logprobs,
                    top,
                });
            }
        }
        usage.media = read;
        Ok(ReadsResponse {
            model: self.config.model.clone(),
            reads,
            signals: Some(signals(&raw.control_text)),
            usage,
        })
    }

    /// The text of one token, for a client to read. A token that is not whole text alone, such as one byte of a
    /// longer character, shows as U+FFFD. The `id` beside it names the token exactly.
    fn piece(&self, id: u32) -> Result<String, ServiceError> {
        match self.family.detokenize(&[id]) {
            Ok(text) => Ok(text),
            Err(FamilyError::Client(_)) => Ok("\u{FFFD}".to_owned()),
            Err(error) => Err(error.into()),
        }
    }
}
