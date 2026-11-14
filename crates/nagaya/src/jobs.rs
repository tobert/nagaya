//! The jobs of a device group: the scheduler's state machine, driven on the group thread (spec 3.2, 5).
//!
//! A [`JobBody`] holds the work of one request. The group thread asks [`GroupJobs`] for one quantum at a time. The
//! scheduler chooses the quantum (`ObservedScheduler::step`), and the body runs it. The scheduler never sees the body.
//!
//! The group thread records what it runs (spec 9.2, 9.4, 9.6). Each job has a span `nagaya.job`. Each quantum that is
//! not decode has a span `nagaya.quantum` under it, and each quantum has its two durations. A quantum that fails
//! logs the change of the machine's counters.

use std::collections::BTreeMap;
use std::time::Instant;

use nagaya_abi::{Engine, Kind, Limits, PassError, PassOut, Segment};
use nagaya_sched::{Class, Completion, Event, Item, Job, JobId, JobState, ObservedScheduler, Quantum, Scheduler};
use nagaya_telemetry::EngineState;
use tracing::Span;

use crate::evidence::{self, Machine};

/// What a body reports after one quantum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ran {
    /// The quantum ran. The job has no new quantum to add.
    Done,
    /// The quantum ran, and the job has these quanta to add after the ones it has.
    More(Vec<Kind>),
    /// The quantum ran, and the job runs these quanta before the ones it has, for example an interval copy (spec
    /// 6.13).
    First(Vec<Kind>),
    /// The quantum failed. The job ends, and the body has told its caller why.
    Failed,
    /// The quantum ran, and the body cancels its job, for example because its client went away. The job ends as
    /// cancelled (spec 5.9).
    Cancelled,
}

/// The work of one job.
pub trait JobBody: Send {
    /// Run the quantum of the given kind. `engines` are the engines of the group, in profile order.
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran;

    /// Hear that the job reached a terminal state. The group calls it once, before it drops the body. A job can end
    /// with no quantum that ran, for example when it passed its deadline (spec 5.8).
    fn ended(&mut self, _state: JobState) {}

    /// Hear that the job ended before it ran, because its engine has this state and not `ready` (spec 3.3a). The group
    /// calls it once, instead of `ended`, and then it drops the body.
    fn unavailable(&mut self, engine: &str, state: EngineState);

    /// The kind of request that the job serves, for the span `nagaya.job` (spec 9.2). The set of values is bounded.
    fn request(&self) -> &'static str {
        "job"
    }

    /// The item of a quantum of `kind` on an engine with stages (engine-abi D1). A `Block` is a pass that the group
    /// runs stage by stage through [`JobBody::begin_pass`] and [`JobBody::end_pass`]. The default is a `Whole` copy
    /// quantum, which [`JobBody::run`] runs while it holds every device.
    fn item(&self, kind: Kind) -> Item {
        Item::Whole(kind)
    }

    /// True when the next quantum of `kind` is one decode row that can share a pass with the rows of other jobs
    /// (engine-abi "LFM3", "Batched passes"). Then the group gives its segment through [`JobBody::begin_pass`] and its
    /// rows through [`JobBody::end_pass`], even on an engine with no stages. The default is false.
    fn joinable(&self, _kind: Kind) -> bool {
        false
    }

    /// Hear that the pass that began shares its pass with the row of another job. The driver calls it before
    /// [`JobBody::end_pass`]. A job on an engine with a feed rule then keeps no row of that pass in the store (spec
    /// 6.14, "No head from a shared pass"). The default does nothing.
    fn shared_pass(&mut self) {}

    /// Begin a pass of `kind` on an engine with stages: give its segments. A body whose `item` gives a `Block`
    /// implements it. An error is `Ran::Failed` or `Ran::Cancelled`, and the body told its caller why.
    fn begin_pass(&mut self, kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        panic!("a {kind:?} quantum of this job is not a pass; its item must not be a Block")
    }

    /// Prepare the end of the oldest pass that began, from the result that [`JobBody::end_pass`] gets next. The work
    /// uses no engine, and it changes nothing that another body reads. The driver calls it for the bodies of one
    /// shared pass at the same time, each on a row thread (spec 3.2, "The row threads"). A body keeps what it
    /// prepared, and its `end_pass` gives the same result with it as with none. The default does nothing.
    fn prepare_end(&mut self, _kind: Kind, _result: &Result<PassOut, PassError>) {}

    /// End the oldest pass that began, with what its last stage gave. The passes of one job end in order.
    fn end_pass(&mut self, kind: Kind, _result: Result<PassOut, PassError>, _engines: &mut [Box<dyn Engine>]) -> Ran {
        panic!("a {kind:?} quantum of this job is not a pass; its item must not be a Block")
    }
}

/// A job that a request brings to a group.
pub struct JobSpec {
    /// The class of the job (spec 5.3).
    pub class: Class,
    /// The integer priority (spec 5.4).
    pub priority: i64,
    /// The time after which the job runs no more quanta (spec 5.8).
    pub deadline: Option<Instant>,
    /// The kind of the first quantum. The body adds the others when it runs the first one.
    pub first: Kind,
    /// The engine of the job: its place in the engines of the group, in profile order. A job for an engine that is
    /// not `ready` ends at once (spec 3.3a).
    pub engine: usize,
    /// The work.
    pub body: Box<dyn JobBody>,
}

/// The state of the jobs of one group.
pub struct GroupJobs {
    group: u32,
    /// The names of the engines of the group, in profile order. A job span, its quanta and their metrics name the
    /// engine of the job (spec 9.2, "The engine of a job").
    names: Vec<String>,
    machine: Machine,
    epoch: Instant,
    scheduler: ObservedScheduler,
    bodies: BTreeMap<JobId, Box<dyn JobBody>>,
    /// The engine of each job that has not ended, as its place in the group (spec 3.3a).
    engines: BTreeMap<JobId, usize>,
    /// The span `nagaya.job` of each job that has not ended.
    spans: BTreeMap<JobId, Span>,
    /// The jobs that arrived since the last quantum. Their inputs become present when the next quantum is chosen.
    fresh: Vec<JobId>,
    /// The quantum that ran last, until the scheduler learns that it ended. The end event chooses the next quantum.
    finished: Option<(u64, Completion)>,
    next_id: JobId,
}

impl GroupJobs {
    /// Make the jobs of a group. The scheduler serves one device group (spec 5.5).
    ///
    /// `names` are the names of the engines of the group, in profile order. The set of values of the attribute
    /// `engine` stays bounded by the profile. The joined name is the name of the group (spec 9.2).
    pub fn new(group: u32, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        let joined = names.join("+");
        Self {
            group,
            machine: Machine::real(),
            epoch: Instant::now(),
            scheduler: ObservedScheduler::new(Scheduler::default(), BTreeMap::from([(group, joined)])),
            names,
            bodies: BTreeMap::new(),
            engines: BTreeMap::new(),
            spans: BTreeMap::new(),
            fresh: Vec::new(),
            finished: None,
            next_id: 1,
        }
    }

    /// Read the counters of `machine` before a quantum (spec 9.6). The default is the machine that runs this process.
    pub fn machine(mut self, machine: Machine) -> Self {
        self.machine = machine;
        self
    }

    /// True when no job waits or runs.
    pub fn idle(&self) -> bool {
        self.bodies.is_empty() && self.finished.is_none()
    }

    /// True when a job of the engine at `engine` in the group has not ended (spec 3.3a).
    pub fn holds(&self, engine: usize) -> bool {
        self.engines.values().any(|held| *held == engine)
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn event(&mut self, event: Event) {
        // A wrong event is a defect of this driver, and the state is gone (spec 8.5).
        let chosen = self
            .scheduler
            .step(event)
            .expect("the group driver sent an event that does not fit");
        assert!(
            chosen.is_none() || self.scheduler.state().running == chosen,
            "the scheduler chose a quantum that it does not hold as running"
        );
    }

    /// Add a job. It waits for the next call of [`GroupJobs::run_one`]. The span of the job is the child of the span
    /// that is current.
    pub fn add(&mut self, spec: JobSpec) -> JobId {
        self.add_under(spec, Span::current())
    }

    /// Add a job as [`GroupJobs::add`] does. The span of the job is the child of `parent`: the span of the request that
    /// brought the job (spec 9.2).
    pub fn add_under(&mut self, spec: JobSpec, parent: Span) -> JobId {
        let id = self.next_id;
        self.next_id += 1;
        let now_ms = self.now_ms();
        let deadline_ms = spec
            .deadline
            .map(|deadline| deadline.saturating_duration_since(self.epoch).as_millis() as u64);
        let job = Job {
            id,
            class: spec.class,
            priority: spec.priority,
            arrival_ms: now_ms,
            deadline_ms,
            engine: self.group,
            quanta: [spec.first].into_iter().collect(),
            admitted: false,
            inputs_present: false,
            state: JobState::Ready,
            balance_ms: 0,
            cancel_requested: false,
            checkpoint_interval_ms: None,
        };
        let engine = self.names.get(spec.engine).unwrap_or_else(|| {
            panic!(
                "a job names engine {} of group {}, which has {} engines",
                spec.engine,
                self.group,
                self.names.len()
            )
        });
        let span = tracing::info_span!(
            parent: &parent,
            "nagaya.job",
            engine = engine.as_str(),
            class = spec.class.label(),
            request = spec.body.request(),
            outcome = tracing::field::Empty,
        );
        self.scheduler.set_job_span(id, span.clone());
        self.scheduler.set_job_engine(id, engine.clone());
        self.spans.insert(id, span);
        self.event(Event::Arrive { now_ms, job });
        // The ledger admitted the context before the job came, and a pass never allocates (spec 4.3).
        self.event(Event::Admit { now_ms, job: id });
        self.bodies.insert(id, spec.body);
        self.engines.insert(id, spec.engine);
        self.fresh.push(id);
        id
    }

    /// Choose and run one quantum. It gives false when no quantum was ready.
    ///
    /// The choice happens here, after every arrival so far, so a job that arrived while the last quantum ran can go
    /// first (spec 5.3). The end of the last quantum reaches the scheduler here too.
    pub fn run_one(&mut self, engines: &mut [Box<dyn Engine>]) -> bool {
        for id in std::mem::take(&mut self.fresh) {
            if let Some(job) = self.scheduler.state_mut().jobs.iter_mut().find(|job| job.id == id) {
                job.inputs_present = true;
            }
        }
        // The facts of the engines come before the event that chooses the next quantum.
        self.learn_joins(engines);
        let now_ms = self.now_ms();
        match self.finished.take() {
            Some((duration_ms, outcome)) => self.event(Event::Complete {
                now_ms,
                duration_ms,
                outcome,
            }),
            None => self.event(Event::Tick { now_ms }),
        }
        self.forget_ended_jobs();
        let Some(quantum) = self.scheduler.state().running.clone() else {
            return false;
        };
        if quantum.segments.len() > 1 {
            self.run_joined(quantum, engines);
            return true;
        }
        let id = quantum.segments[0];
        let body = self.bodies.get_mut(&id).expect("a chosen job has a body");
        // Reading the counters costs four files, so a decode quantum skips it (spec 9.6).
        let watch = (quantum.kind != Kind::Decode).then(|| self.machine.watch());
        let span = self.scheduler.running_span().cloned();
        let started = Instant::now();
        // The spans of the engine nest under the span of the quantum (spec 9.2).
        let ran = match &span {
            Some(span) => span.in_scope(|| body.run(quantum.kind, engines)),
            None => body.run(quantum.kind, engines),
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        if matches!(ran, Ran::Failed) {
            let failure = format!(
                "{} quantum of a {} request on engine {}",
                format!("{:?}", quantum.kind).to_lowercase(),
                body.request(),
                self.names[self.engines[&id]]
            );
            match watch {
                Some(watch) => watch.failed(&failure),
                None => evidence::failed_unwatched(&failure),
            };
        }
        let outcome = self.apply(id, ran);
        self.finished = Some((duration_ms, outcome));
        true
    }

    /// Tell the scheduler which engines of the group batch decode rows (`joins_decode`). The driver joins only decode
    /// quanta. The group is
    /// the scheduler's engine, so its limit is the largest of its engines. A joined quantum still runs each engine's
    /// rows in passes of at most that engine's limit.
    fn learn_joins(&mut self, engines: &[Box<dyn Engine>]) {
        let max_segments = engines
            .iter()
            .map(|engine| engine.limits())
            .filter(joins_decode)
            .map(|limits| limits.max_segments)
            .max();
        let facts = max_segments.map(|max_segments| nagaya_sched::Engine {
            id: self.group,
            segments_exact: false,
            max_segments,
            joins_reads: false,
            joins_decode: true,
        });
        let state = self.scheduler.state_mut();
        state.engines.retain(|engine| engine.id != self.group);
        state.engines.extend(facts);
    }

    /// Run a quantum of several decode segments. A body that can join gives its segment, and one pass runs the plain
    /// rows of each engine that share a control list, up to that engine's limit. Every other body runs alone in this
    /// quantum. Each body gets its own rows, and each job ends or goes on by its own result.
    fn run_joined(&mut self, quantum: Quantum, engines: &mut [Box<dyn Engine>]) {
        let kind = quantum.kind;
        assert_eq!(kind, Kind::Decode, "the driver joins only decode quanta");
        let span = self.scheduler.running_span().cloned();
        let _entered = span.as_ref().map(|span| span.enter());
        let started = Instant::now();
        let mut jobs: Vec<(JobId, usize, Box<dyn JobBody>)> = quantum
            .segments
            .iter()
            .map(|id| {
                (
                    *id,
                    self.engines[id],
                    self.bodies.remove(id).expect("a chosen job has a body"),
                )
            })
            .collect();
        let results = run_joined_bodies(kind, &mut jobs, engines);
        for (id, _, body) in jobs {
            self.bodies.insert(id, body);
        }
        let duration_ms = started.elapsed().as_millis() as u64;
        for (id, ran) in results {
            if matches!(ran, Ran::Failed) {
                let failure = format!(
                    "decode quantum of a {} request on engine {}",
                    self.bodies[&id].request(),
                    self.names[self.engines[&id]]
                );
                evidence::failed_unwatched(&failure);
                self.scheduler
                    .state_mut()
                    .fail_running(id)
                    .expect("a job of the joined quantum is running");
                continue;
            }
            self.apply(id, ran);
        }
        self.finished = Some((duration_ms, Completion::Done));
    }

    /// Give the scheduler what a body's quantum added, and the outcome of the quantum for that job.
    fn apply(&mut self, id: JobId, ran: Ran) -> Completion {
        match ran {
            Ran::Done => Completion::Done,
            Ran::More(kinds) => {
                self.scheduler
                    .state_mut()
                    .append_quanta(id, &kinds)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            Ran::First(kinds) => {
                self.scheduler
                    .state_mut()
                    .prepend_quanta(id, &kinds)
                    .expect("a job that ran a quantum is running");
                Completion::Done
            }
            Ran::Failed => Completion::Failed,
            Ran::Cancelled => {
                // The quantum ran to its end. A cancel of a running job takes effect when it completes (spec 5.9).
                self.event(Event::Cancel {
                    now_ms: self.now_ms(),
                    job: id,
                });
                Completion::Done
            }
        }
    }

    /// Drop the jobs that reached a terminal state, and their bodies. A job that leaves takes its balance (spec 5.4).
    fn forget_ended_jobs(&mut self) {
        let ended: Vec<(JobId, JobState)> = self
            .scheduler
            .state()
            .jobs
            .iter()
            .filter(|job| job.state.is_terminal())
            .map(|job| (job.id, job.state))
            .collect();
        self.scheduler.state_mut().jobs.retain(|job| !job.state.is_terminal());
        for (id, state) in ended {
            if let Some(span) = self.spans.remove(&id) {
                span.record("outcome", state.label());
            }
            let mut body = self.bodies.remove(&id).expect("a job that ended has a body");
            self.engines.remove(&id);
            body.ended(state);
        }
    }

    /// Cancel every job that has not ended (spec 10.3, 5.9). It gives the number of jobs that it dropped.
    ///
    /// A body that drops its result sender tells its caller that the job ended with no result.
    pub fn cancel_all(&mut self) -> usize {
        let count = self.bodies.len();
        self.bodies.clear();
        self.engines.clear();
        self.fresh.clear();
        for (_, span) in std::mem::take(&mut self.spans) {
            span.record("outcome", JobState::Cancelled.label());
        }
        // The last quantum ran. Its end still counts, although no job follows it.
        self.scheduler.abandon(self.finished.take());
        count
    }
}

/// A body of a shared pass, with the result that its `end_pass` gets next.
type PassEnd<'a> = (&'a mut Box<dyn JobBody>, &'a Result<PassOut, PassError>);

/// Run one joined decode quantum of `jobs`, each with the place of its engine in `engines`. A body that can join
/// gives its segment, and one pass runs the plain rows of each engine that share a control list, up to that engine's
/// limit. Every other body runs alone. Each body gets its own rows. The results are in the order of `jobs` for the
/// bodies that ran alone or failed to begin, then in the order of the passes (engine-abi "LFM3", "Batched passes").
pub fn run_joined_bodies(
    kind: Kind,
    jobs: &mut [(JobId, usize, Box<dyn JobBody>)],
    engines: &mut [Box<dyn Engine>],
) -> Vec<(JobId, Ran)> {
    let mut results: Vec<(JobId, Ran)> = Vec::new();
    let mut begun: Vec<(JobId, usize, Segment)> = Vec::new();
    for (id, engine, body) in jobs.iter_mut() {
        if !(matches!(body.item(kind), Item::Block(_)) && body.joinable(kind)) {
            results.push((*id, body.run(kind, engines)));
            continue;
        }
        match body.begin_pass(kind, engines) {
            Ok(mut segments) => {
                assert_eq!(segments.len(), 1, "a body that joins gives one segment");
                begun.push((*id, *engine, segments.pop().expect("one segment")));
            }
            Err(ran) => results.push((*id, ran)),
        }
    }
    for batch in batches(begun, engines) {
        let engine = batch[0].1;
        let segments: Vec<Segment> = batch.iter().map(|(_, _, segment)| segment.clone()).collect();
        let result = engines[engine].pass(&segments, &[]);
        let each = split(result, &segments);
        // The host work of each row that uses no engine runs at the same time, so the card waits for the slowest
        // row and not for their sum. A body of one pass is in one batch only.
        let mut ends: Vec<PassEnd> = jobs
            .iter_mut()
            .filter_map(|(job, _, body)| {
                let place = batch.iter().position(|(id, _, _)| id == job)?;
                Some((body, &each[place]))
            })
            .collect();
        assert_eq!(ends.len(), batch.len(), "a segment has its job");
        crate::rows::each(&mut ends, |(body, result)| body.prepare_end(kind, result));
        drop(ends);
        for ((id, _, _), result) in batch.iter().zip(each) {
            let (_, _, body) = jobs
                .iter_mut()
                .find(|(job, _, _)| job == id)
                .expect("a segment has its job");
            if segments.len() > 1 {
                body.shared_pass();
            }
            results.push((*id, body.end_pass(kind, result, engines)));
        }
    }
    results
}

/// True when an engine joins the decode rows of several jobs in one pass, with more than one segment. An engine with
/// a feed rule joins them by D11 rule 6 (engine-abi "LFM3", "Batched passes"). An engine that declares
/// `segments_exact` joins them by spec 5.7 ("The first joins of an engine with `segments_exact`").
pub(crate) fn joins_decode(limits: &Limits) -> bool {
    (limits.feed.is_some() || limits.segments_exact) && limits.max_segments > 1
}

/// The passes of a joined quantum. The plain rows of one engine with one control list share a pass, up to the
/// engine's limit; a plain row is one token with its logit row and no media. Every other segment has a pass alone.
fn batches(begun: Vec<(JobId, usize, Segment)>, engines: &[Box<dyn Engine>]) -> Vec<Vec<(JobId, usize, Segment)>> {
    let mut out: Vec<Vec<(JobId, usize, Segment)>> = Vec::new();
    // Only the engines of these segments: on a device thread, the place of an engine elsewhere has no limits.
    let mut limits: BTreeMap<usize, u32> = BTreeMap::new();
    for (_, engine, _) in &begun {
        limits.entry(*engine).or_insert_with(|| {
            let limits = engines[*engine].limits();
            if joins_decode(&limits) { limits.max_segments } else { 1 }
        });
    }
    for item in begun {
        let (_, engine, segment) = &item;
        let plain = segment.positions.len() == 1
            && segment.outputs.logit_rows.len() == 1
            && segment.meta.is_empty()
            && matches!(segment.inputs, nagaya_abi::SegmentInputs::Tokens(_));
        let joins = out.iter_mut().find(|batch| {
            let (_, other, first) = &batch[0];
            plain
                && other == engine
                && batch.len() < limits[engine] as usize
                && first.positions.len() == 1
                && first.outputs.logit_rows.len() == 1
                && first.meta.is_empty()
                && matches!(first.inputs, nagaya_abi::SegmentInputs::Tokens(_))
                && first.control == segment.control
        });
        match joins {
            Some(batch) => batch.push(item),
            None => out.push(vec![item]),
        }
    }
    out
}

/// The result of each segment of a pass: its own logit rows, in segment order (engine-abi "Logits"). An error goes to
/// each segment. A pass whose outputs are not one `logits` for each logit row breaks the ABI, and the driver stops
/// (spec 8.5).
pub(crate) fn split(result: Result<PassOut, PassError>, segments: &[Segment]) -> Vec<Result<PassOut, PassError>> {
    let out = match result {
        Ok(out) => out,
        Err(error) => return segments.iter().map(|_| Err(error.clone())).collect(),
    };
    if segments.len() == 1 {
        return vec![Ok(out)];
    }
    // A pass that poisoned or timed out gives its status to each segment, as a pass alone does, with no rows.
    if out.status != nagaya_abi::PassStatus::Ok {
        return segments
            .iter()
            .map(|_| {
                Ok(PassOut {
                    status: out.status,
                    ms: out.ms,
                    outputs: Vec::new(),
                })
            })
            .collect();
    }
    let rows: usize = segments.iter().map(|s| s.outputs.logit_rows.len()).sum();
    assert!(
        out.outputs.len() == rows && out.outputs.iter().all(|o| o.name == "logits"),
        "a pass of {} segments gave {} outputs, not one `logits` for each of {rows} logit rows",
        segments.len(),
        out.outputs.len()
    );
    let mut outputs = out.outputs.into_iter();
    segments
        .iter()
        .map(|segment| {
            Ok(PassOut {
                status: out.status,
                ms: out.ms,
                outputs: outputs.by_ref().take(segment.outputs.logit_rows.len()).collect(),
            })
        })
        .collect()
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use nagaya_abi::{OutputsWanted, PassStatus, SegmentInputs, State};

    fn segment(slot: u32) -> Segment {
        Segment {
            control: Vec::new(),
            state: State::Slot(slot),
            inputs: SegmentInputs::Tokens(vec![1]),
            positions: vec![4],
            meta: Vec::new(),
            outputs: OutputsWanted { logit_rows: vec![4] },
        }
    }

    /// A joined pass that poisoned gives its status to each segment with no rows, as a pass alone does; it does not
    /// stop the driver.
    #[test]
    fn a_poisoned_joined_pass_gives_each_segment_its_status() {
        let segments = [segment(0), segment(1), segment(2)];
        let out = PassOut {
            status: PassStatus::Poison,
            ms: 3,
            outputs: Vec::new(),
        };
        let each = split(Ok(out), &segments);
        assert_eq!(each.len(), 3);
        for result in each {
            let out = result.unwrap();
            assert_eq!((out.status, out.outputs.len()), (PassStatus::Poison, 0));
        }
    }
}
