//! The lab route `ensemble` (`docs/lab.md`, "`ensemble`: joined forks"). One job holds K contexts. At each step it
//! combines their logit rows into one row, and the generation of context 0 samples one token from it. Each context
//! feeds that token, and the K rows of the next step come from one pass.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};

use gogi_proto::Invalid;
use gogi_proto::generation::{GenerationDone, GenerationRequest};
use nagaya_abi::{Engine, Kind, PassError, PassOut, Segment};
use nagaya_council::ensemble::{Combine, combine};
use nagaya_sched::{Item, JobState};
use nagaya_telemetry::EngineState;

use super::build::{ContextBody, Finished, Got, Then, Want};
use super::error::ServiceError;
use super::generations::{GenerateThen, Latent, Soft};
use super::{Charge, Schedule, Service, submit};
use crate::jobs::{JobBody, Ran};

/// What the contexts of one step share.
#[derive(Default)]
struct Ballot {
    /// The logit row of each context at this step.
    rows: Vec<Option<Vec<f32>>>,
    /// The combined row, from the owner to the lead.
    combined: Option<Vec<f32>>,
    /// The token that each context feeds next, from the lead.
    next: Option<u32>,
    /// True when the reply of the lead ended.
    ended: bool,
}

fn lock(ballot: &Mutex<Ballot>) -> MutexGuard<'_, Ballot> {
    ballot.lock().expect("the ballot lock is poisoned")
}

/// The work of one context of an ensemble. Each context builds and feeds its prompt as its generation does. Context 0
/// leads: its generation samples from the combined row, and it ends the reply.
pub(crate) struct EnsembleThen {
    index: usize,
    generation: GenerateThen,
    ballot: Arc<Mutex<Ballot>>,
}

impl EnsembleThen {
    /// The generation of the context, with its reply when the lead is done.
    pub(crate) fn into_generation(self) -> GenerateThen {
        self.generation
    }
}

impl Then for EnsembleThen {
    const NEEDS_STATE: bool = true;
    const REQUEST: &'static str = "ensemble";

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match got {
            // A row waits for the rows of the other contexts.
            // The row of the last latent step ends the latent stage of the context first.
            Got::Fed(Some(row)) => {
                self.generation.end_latent(&row)?;
                lock(&self.ballot).rows[self.index] = Some(row);
                Ok(Want::Wait)
            }
            Got::Woken if self.index == 0 => {
                let row = lock(&self.ballot)
                    .combined
                    .take()
                    .ok_or_else(|| ServiceError::internal("the lead of an ensemble woke with no combined row"))?;
                let want = self.generation.next(Got::Fed(Some(row)))?;
                let mut ballot = lock(&self.ballot);
                match &want {
                    Want::Feed {
                        tokens,
                        logits: true,
                        kind: Kind::Decode,
                    } if tokens.len() == 1 => ballot.next = Some(tokens[0]),
                    _ => ballot.ended = true,
                }
                Ok(want)
            }
            Got::Woken => {
                let mut ballot = lock(&self.ballot);
                if ballot.ended {
                    return Ok(Want::Done);
                }
                let token = ballot
                    .next
                    .ok_or_else(|| ServiceError::internal("a context of an ensemble woke before the lead chose"))?;
                ballot.rows[self.index] = None;
                Ok(Want::Feed {
                    tokens: vec![token],
                    logits: true,
                    kind: Kind::Decode,
                })
            }
            // The build, the feed of the prompt, the latent steps, and the head of the lead's reply are the
            // generation's. A latent step is a tapped pass of one context, in the build turn of that context.
            got => self.generation.next(got),
        }
    }
}

/// The result of an ensemble job: the finished job of the lead, whose generation holds the reply.
type Result0 = Result<Finished<EnsembleThen>, ServiceError>;

/// The phase of an ensemble job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// The build of context `i` runs. The contexts before it wait at their first row.
    Building(usize),
    /// Each context waits at its row, or feeds the token of the last step in one pass.
    Stepping,
    /// The reply ended. The lead runs its last quanta alone, such as the feed and the snapshot of its head.
    Ending,
}

/// The job of an ensemble: K context jobs that it runs itself, as one job of the scheduler.
pub(crate) struct EnsembleBody {
    contexts: Vec<ContextBody<EnsembleThen>>,
    results: Vec<Receiver<Result0>>,
    ballot: Arc<Mutex<Ballot>>,
    rule: Combine,
    weights: Vec<f64>,
    engine: usize,
    phase: Phase,
    /// The segments of the pass of a step that runs, one for each context, for the split of its rows.
    stepping: Vec<Segment>,
    tx: Option<Sender<Result0>>,
}

impl EnsembleBody {
    /// Send the result of the job once.
    fn send(&mut self, result: Result0) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
    }

    /// A context failed: its body sent the error to its own receiver. Give it to the caller.
    fn failed(&mut self, index: usize) -> Ran {
        let error = match self.results[index].try_recv() {
            Ok(Err(error)) => error,
            Ok(Ok(_)) => ServiceError::internal("a context of an ensemble finished where it failed"),
            Err(_) => ServiceError::internal("a context of an ensemble failed and gave no error"),
        };
        self.send(Err(error));
        Ran::Failed
    }

    /// What a quantum of the context `index` gave, in the phase of the job.
    fn after(&mut self, index: usize, ran: Ran, engines: &mut [Box<dyn Engine>]) -> Ran {
        match ran {
            Ran::Failed => return self.failed(index),
            Ran::Cancelled => {
                self.send(Err(ServiceError::Cancelled));
                return Ran::Cancelled;
            }
            Ran::Done if self.contexts[index].waiting() => {}
            Ran::Done if self.contexts[index].finished() => {
                // Only the lead finishes by itself, after the reply ended.
                assert_eq!(
                    self.phase,
                    Phase::Ending,
                    "a context of an ensemble finished before the reply ended"
                );
                return self.finish();
            }
            // The quantum ran, and the context runs the quanta that it has.
            more => return more,
        }
        match self.phase {
            Phase::Building(i) if i + 1 < self.contexts.len() => {
                self.phase = Phase::Building(i + 1);
                Ran::More(vec![Kind::Copy])
            }
            Phase::Building(_) | Phase::Stepping => {
                self.phase = Phase::Stepping;
                if self.contexts.iter().all(ContextBody::waiting) {
                    self.step(engines)
                } else {
                    Ran::Done
                }
            }
            Phase::Ending => unreachable!("no context waits after the reply ended"),
        }
    }

    /// Every context waits at its row: combine the rows, wake the lead to sample, then wake the others to feed.
    fn step(&mut self, engines: &mut [Box<dyn Engine>]) -> Ran {
        let combined = {
            let mut ballot = lock(&self.ballot);
            let rows: Vec<Vec<f32>> = ballot
                .rows
                .iter_mut()
                .map(|row| row.take().expect("each context that waits gave its row"))
                .collect();
            let rows: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
            combine(&rows, &self.rule, &self.weights)
        };
        match combined {
            Ok(row) => lock(&self.ballot).combined = Some(row),
            Err(error) => {
                self.send(Err(ServiceError::internal(format!(
                    "the rows of an ensemble do not combine: {error}"
                ))));
                return Ran::Failed;
            }
        }
        let lead = self.contexts[0].wake(engines);
        if matches!(lead, Ran::Failed) {
            return self.failed(0);
        }
        let ended = lock(&self.ballot).ended;
        for index in 1..self.contexts.len() {
            match self.contexts[index].wake(engines) {
                Ran::Failed => return self.failed(index),
                Ran::Done if ended => {}
                Ran::More(kinds) if !ended && kinds == [Kind::Decode] => {}
                other => {
                    self.send(Err(ServiceError::internal(format!(
                        "context {index} of an ensemble gave {other:?} at a step"
                    ))));
                    return Ran::Failed;
                }
            }
        }
        if ended {
            self.phase = Phase::Ending;
            self.contexts.truncate(1);
            return match lead {
                Ran::Done if self.contexts[0].finished() => self.finish(),
                more => more,
            };
        }
        match lead {
            Ran::More(kinds) if kinds == [Kind::Decode] => Ran::More(kinds),
            other => {
                self.send(Err(ServiceError::internal(format!(
                    "the lead of an ensemble gave {other:?} at a step"
                ))));
                Ran::Failed
            }
        }
    }

    /// The lead finished: give its job to the caller.
    fn finish(&mut self) -> Ran {
        match self.results[0].try_recv() {
            Ok(result) => self.send(result),
            Err(_) => self.send(Err(ServiceError::internal(
                "the lead of an ensemble finished and gave nothing",
            ))),
        }
        Ran::Done
    }

    /// The contexts that a quantum of this phase runs: one at a build and at the end, and each one at a step.
    fn running(&self) -> std::ops::Range<usize> {
        match self.phase {
            Phase::Building(i) => i..i + 1,
            Phase::Stepping => 0..self.contexts.len(),
            Phase::Ending => 0..1,
        }
    }

    /// The end of a pass of a step: each context gets its own rows, then the step runs.
    fn end_step(&mut self, kind: Kind, result: Result<PassOut, PassError>, engines: &mut [Box<dyn Engine>]) -> Ran {
        let segments = std::mem::take(&mut self.stepping);
        let outs = crate::jobs::split(result, &segments);
        let mut last = Ran::Done;
        for (index, out) in outs.into_iter().enumerate() {
            let ran = self.contexts[index].end_pass(kind, out, engines);
            match ran {
                Ran::Done if self.contexts[index].waiting() => {}
                Ran::Failed => return self.failed(index),
                other => last = other,
            }
        }
        match last {
            Ran::Done => self.step(engines),
            other => {
                self.send(Err(ServiceError::internal(format!(
                    "a context of an ensemble gave {other:?} after a step"
                ))));
                Ran::Failed
            }
        }
    }
}

impl JobBody for EnsembleBody {
    fn request(&self) -> &'static str {
        EnsembleThen::REQUEST
    }

    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        if self.phase == Phase::Stepping {
            // The pass of a step, on a group that runs whole quanta.
            let segments = match self.begin_pass(kind, engines) {
                Ok(segments) => segments,
                Err(ran) => return ran,
            };
            let result = engines[self.engine].pass(&segments, &[]);
            return self.end_pass(kind, result, engines);
        }
        let index = self.running().start;
        let ran = self.contexts[index].run(kind, engines);
        self.after(index, ran, engines)
    }

    fn item(&self, kind: Kind) -> Item {
        match self.phase {
            Phase::Stepping => Item::Block(kind),
            _ => self.contexts[self.running().start].item(kind),
        }
    }

    fn begin_pass(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        let mut segments = Vec::new();
        for index in self.running() {
            match self.contexts[index].begin_pass(kind, engines) {
                Ok(mut more) if self.phase == Phase::Stepping => {
                    assert_eq!(more.len(), 1, "a context of a step gives one segment");
                    segments.push(more.pop().expect("one segment"));
                }
                Ok(more) => segments.extend(more),
                Err(Ran::Failed) => return Err(self.failed(index)),
                Err(ran) => return Err(ran),
            }
        }
        if self.phase == Phase::Stepping {
            self.stepping = segments.clone();
        }
        Ok(segments)
    }

    fn end_pass(&mut self, kind: Kind, result: Result<PassOut, PassError>, engines: &mut [Box<dyn Engine>]) -> Ran {
        if self.phase == Phase::Stepping {
            return self.end_step(kind, result, engines);
        }
        let index = self.running().start;
        let ran = self.contexts[index].end_pass(kind, result, engines);
        self.after(index, ran, engines)
    }

    fn ended(&mut self, state: JobState) {
        let error = match state {
            JobState::Expired => ServiceError::Timeout,
            _ => ServiceError::Cancelled,
        };
        self.send(Err(error));
    }

    fn unavailable(&mut self, engine: &str, state: EngineState) {
        self.send(Err(ServiceError::Unavailable {
            engine: engine.to_owned(),
            state,
        }));
    }
}

impl Service {
    /// Generate one reply from K contexts (`docs/lab.md`, "`ensemble`"). Each request is a generation request of one
    /// context; they name the same model and the same `max_tokens`. The reply, its finish and its usage are those of
    /// context 0, whose generation samples from the combined row.
    pub(crate) fn ensemble(
        &self,
        requests: &[GenerationRequest],
        rule: Combine,
        weights: Vec<f64>,
        latent: Option<Latent>,
        soft: Option<Soft>,
    ) -> Result<Ensembled, ServiceError> {
        if let Some(soft) = soft {
            return self.soft_generation(requests, &weights, latent, soft);
        }
        let count = requests.len();
        let limits = &self.shared.limits;
        // Forks join in batched passes by the rule of the scheduler's joins.
        let batched = crate::jobs::joins_decode(limits);
        if count > 1 && !batched {
            return Err(Invalid::param(
                "model",
                format!(
                    "nothing ran: {} has no batched passes, so its ensemble holds one context; send one context",
                    self.config.model
                ),
            )
            .into());
        }
        let most = if batched { limits.max_segments as usize } else { 1 };
        if count == 0 || count > most {
            return Err(Invalid::param(
                "contexts",
                format!(
                    "nothing ran: an ensemble on {} holds 1 to {most} contexts",
                    self.config.model
                ),
            )
            .into());
        }
        let ballot = Arc::new(Mutex::new(Ballot {
            rows: vec![None; count],
            ..Ballot::default()
        }));
        let mut contexts = Vec::with_capacity(count);
        let mut results = Vec::with_capacity(count);
        let mut facts = None;
        for (index, request) in requests.iter().enumerate() {
            // Each context takes the latent steps. Only context 0 reads them (docs/lab.md, "Latent steps").
            let plan = latent.clone().map(|plan| Latent {
                read: plan.read && index == 0,
                ..plan
            });
            let mut prepared = self.prepare(request, None, None, plan)?;
            facts.get_or_insert(prepared.facts.clone());
            // The contexts of an ensemble share each pass of a step, so the lead holds no decoded head (spec 6.14).
            prepared.then.shared_pass();
            let then = EnsembleThen {
                index,
                generation: prepared.then,
                ballot: Arc::clone(&ballot),
            };
            let (tx, rx) = mpsc::channel();
            contexts.push(prepared.parts.body(then, tx));
            results.push(rx);
        }
        let facts = facts.expect("an ensemble has a context");
        let mut charges = Vec::with_capacity(count);
        for _ in 0..count {
            charges.push(Charge::take(&self.ports, &self.config.engine)?);
        }
        let (tx, rx) = mpsc::channel();
        let body = EnsembleBody {
            contexts,
            results,
            ballot,
            rule,
            weights,
            engine: self.shared.engine_index,
            phase: Phase::Building(0),
            stepping: Vec::new(),
            tx: Some(tx),
        };
        let schedule = Schedule::of(requests[0].class, requests[0].priority);
        let result = submit(
            &self.ports,
            (self.group, self.index),
            schedule,
            None,
            Kind::Copy,
            Box::new(body),
        )
        .and_then(|()| rx.recv().map_err(|_| ServiceError::Cancelled)?);
        for charge in charges {
            charge.release()?;
        }
        let Finished { outcome, then } = result?;
        let generation = then.into_generation();
        let read = match generation.latent_read() {
            Some(ids) => Some(ids.iter().copied().zip(self.family.pieces(ids)?).collect()),
            None => None,
        };
        let finished = Finished {
            outcome,
            then: generation,
        };
        // An ensemble holds the record of each context, and it names none (`docs/lab.md`, "Context records in the
        // lab routes").
        for request in requests {
            self.hold_record(&request.context)?;
        }
        Ok(Ensembled {
            done: self.answer(finished, &facts).0,
            read,
            soft: None,
        })
    }

    /// An ensemble with soft inputs: one context with the weight 1, which runs as a generation (docs/lab.md, "Soft
    /// inputs in a request"). Its rows share no pass, so it needs none of the steps of the ensemble.
    fn soft_generation(
        &self,
        requests: &[GenerationRequest],
        weights: &[f64],
        latent: Option<Latent>,
        soft: Soft,
    ) -> Result<Ensembled, ServiceError> {
        let [request] = requests else {
            return Err(Invalid::param(
                "soft",
                format!(
                    "nothing ran: soft inputs take one context, and the request has {}; send one context",
                    requests.len()
                ),
            )
            .into());
        };
        if weights != [1.0] {
            return Err(Invalid::param(
                "weights",
                "nothing ran: soft inputs take one context with the weight 1; send no weights",
            )
            .into());
        }
        self.takes_soft(soft.width)?;
        let (finished, facts) = self.run_soft(request, latent, soft)?;
        let read = match finished.then.latent_read() {
            Some(ids) => Some(ids.iter().copied().zip(self.family.pieces(ids)?).collect()),
            None => None,
        };
        let soft = finished.then.soft_stats();
        // An ensemble holds the record of its context, and it names none (docs/lab.md, "Context records in the lab
        // routes").
        self.hold_record(&request.context)?;
        Ok(Ensembled {
            done: self.answer(finished, &facts).0,
            read,
            soft,
        })
    }
}

/// The response of an ensemble: the reply of context 0, and the top token of each latent step of context 0 when the
/// request read them.
pub(crate) struct Ensembled {
    pub(crate) done: GenerationDone,
    pub(crate) read: Option<Vec<(u32, String)>>,
    /// The count of soft rows and their mean share, when the request asked for soft inputs.
    pub(crate) soft: Option<(usize, f64)>,
}
