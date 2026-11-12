//! The group driver batches the decode quanta of an engine with a feed rule and more than one segment (milestone 27
//! step 5, engine-abi "LFM3", "Batched passes"). Each job gives its segment, one pass runs them all, and each job gets
//! its own rows. A job that cannot join runs alone in the same quantum, and one job can fail alone.

use std::sync::{Arc, Mutex};

use nagaya::jobs::{GroupJobs, JobBody, JobSpec, Ran};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Dtype, Engine, FeedRule, Identity, Kind, Limits, Output, OutputsWanted, PassError, PassOut, PassStatus, Segment,
    SegmentInputs, State, Tap,
};
use nagaya_sched::{Class, Item};

/// The segment count of each pass that the engine ran.
type Passes = Arc<Mutex<Vec<usize>>>;

/// An engine that gives, for each logit row, one output `logits` of two values: the slot and the position.
struct Rows {
    limits: Limits,
    identity: Identity,
    passes: Passes,
}

impl Rows {
    /// An engine with a feed rule when `feed` is true, or one that declares `segments_exact` when `exact` is true.
    fn new(feed: bool, exact: bool, max_segments: u32, passes: &Passes) -> Self {
        assert!(
            !(feed && exact),
            "an engine with a feed rule declares no segments_exact (engine-abi D11 rule 4)"
        );
        let fake = FakeAlways::new();
        let mut limits = fake.limits();
        limits.feed = feed.then_some(FeedRule { block: 512 });
        limits.segments_exact = exact;
        limits.max_segments = max_segments;
        Self {
            limits,
            identity: fake.identity(),
            passes: Arc::clone(passes),
        }
    }
}

impl Engine for Rows {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        assert!(segments.len() <= self.limits.max_segments as usize);
        self.passes.lock().unwrap().push(segments.len());
        let mut outputs = Vec::new();
        for segment in segments {
            let State::Slot(slot) = segment.state;
            for position in &segment.outputs.logit_rows {
                let bytes = [slot as f32, *position as f32]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                outputs.push(Output {
                    name: "logits".to_owned(),
                    dtype: Dtype::F32,
                    bytes,
                });
            }
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 1,
            outputs,
        })
    }

    fn last_error(&self) -> Option<String> {
        None
    }
}

/// What a job saw: the rows that each of its passes gave, as (slot, position).
type Seen = Arc<Mutex<Vec<(f32, f32)>>>;

/// A job of `steps` decode quanta on its own slot. Each quantum feeds one token at the next position.
struct Decoder {
    slot: u32,
    position: u64,
    steps: usize,
    /// True when the body can join a batch; false runs each quantum alone with `run`.
    joins: bool,
    /// The step, from 0, whose end fails the job.
    fail_at: Option<usize>,
    done: usize,
    seen: Seen,
}

impl Decoder {
    fn segment(&self) -> Segment {
        Segment {
            control: Vec::new(),
            state: State::Slot(self.slot),
            inputs: SegmentInputs::Tokens(vec![7]),
            positions: vec![self.position],
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: vec![self.position],
            },
        }
    }

    fn took(&mut self, out: PassOut) -> Ran {
        assert_eq!(out.outputs.len(), 1, "a job gets its own row only");
        let values: Vec<f32> = out.outputs[0]
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        self.seen.lock().unwrap().push((values[0], values[1]));
        let step = self.done;
        self.done += 1;
        self.position += 1;
        if self.fail_at == Some(step) {
            return Ran::Failed;
        }
        if self.done < self.steps {
            Ran::More(vec![Kind::Decode])
        } else {
            Ran::Done
        }
    }
}

impl JobBody for Decoder {
    fn unavailable(&mut self, engine: &str, _state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is not ready, and this test keeps it ready");
    }

    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        assert_eq!(kind, Kind::Decode);
        let out = engines[0].pass(&[self.segment()], &[]).expect("a pass alone");
        self.took(out)
    }

    fn item(&self, kind: Kind) -> Item {
        Item::Block(kind)
    }

    fn joinable(&self, kind: Kind) -> bool {
        self.joins && kind == Kind::Decode
    }

    fn begin_pass(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        Ok(vec![self.segment()])
    }

    fn end_pass(&mut self, _kind: Kind, result: Result<PassOut, PassError>, _engines: &mut [Box<dyn Engine>]) -> Ran {
        self.took(result.expect("the batch ran"))
    }
}

fn decoder(slot: u32, steps: usize, joins: bool, fail_at: Option<usize>) -> (Decoder, Seen) {
    let seen = Seen::default();
    let body = Decoder {
        slot,
        position: 10 + u64::from(slot),
        steps,
        joins,
        fail_at,
        done: 0,
        seen: Arc::clone(&seen),
    };
    (body, seen)
}

fn add(jobs: &mut GroupJobs, body: Decoder) -> u64 {
    jobs.add(JobSpec {
        class: Class::Interactive,
        priority: 0,
        first: Kind::Decode,
        deadline: None,
        engine: 0,
        body: Box::new(body),
    })
}

fn run_all(jobs: &mut GroupJobs, engines: &mut [Box<dyn Engine>]) {
    let mut guard = 0;
    while !jobs.idle() {
        jobs.run_one(engines);
        guard += 1;
        assert!(guard < 100, "the jobs never ended");
    }
}

/// Each job saw its own slot at each of its positions, in order.
fn saw_its_own_rows(seen: &Seen, slot: u32, steps: usize) {
    let want: Vec<(f32, f32)> = (0..steps)
        .map(|i| (slot as f32, (10 + slot as usize + i) as f32))
        .collect();
    assert_eq!(*seen.lock().unwrap(), want, "slot {slot}");
}

#[test]
fn the_decode_quanta_of_a_batching_engine_share_one_pass_and_each_job_gets_its_rows() {
    let passes = Passes::default();
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Rows::new(true, false, 8, &passes))];
    let mut jobs = GroupJobs::new(0, ["lfm3"]);
    let mut seen = Vec::new();
    for slot in 0..3 {
        let (body, s) = decoder(slot, 4, true, None);
        add(&mut jobs, body);
        seen.push(s);
    }
    run_all(&mut jobs, &mut engines);
    assert_eq!(*passes.lock().unwrap(), [3, 3, 3, 3]);
    for (slot, s) in seen.iter().enumerate() {
        saw_its_own_rows(s, slot as u32, 4);
    }
}

/// An engine that declares `segments_exact` and no feed rule, as the Qwen3.8 engine does, joins its decode quanta too
/// (spec 5.7, "The first joins of an engine with `segments_exact`").
#[test]
fn the_decode_quanta_of_an_engine_with_segments_exact_share_one_pass() {
    let passes = Passes::default();
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Rows::new(false, true, 16, &passes))];
    let mut jobs = GroupJobs::new(0, ["qwen38"]);
    let mut seen = Vec::new();
    for slot in 0..3 {
        let (body, s) = decoder(slot, 4, true, None);
        add(&mut jobs, body);
        seen.push(s);
    }
    run_all(&mut jobs, &mut engines);
    assert_eq!(*passes.lock().unwrap(), [3, 3, 3, 3]);
    for (slot, s) in seen.iter().enumerate() {
        saw_its_own_rows(s, slot as u32, 4);
    }
}

#[test]
fn an_engine_with_no_feed_rule_or_one_segment_runs_each_decode_alone() {
    for (feed, exact, max) in [(false, false, 8), (true, false, 1), (false, true, 1)] {
        let passes = Passes::default();
        let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Rows::new(feed, exact, max, &passes))];
        let mut jobs = GroupJobs::new(0, ["toy"]);
        for slot in 0..3 {
            add(&mut jobs, decoder(slot, 2, true, None).0);
        }
        run_all(&mut jobs, &mut engines);
        assert_eq!(
            *passes.lock().unwrap(),
            [1; 6],
            "feed {feed}, exact {exact}, {max} segments"
        );
    }
}

#[test]
fn a_job_that_cannot_join_runs_alone_in_the_quantum_and_one_job_fails_alone() {
    let passes = Passes::default();
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Rows::new(true, false, 8, &passes))];
    let mut jobs = GroupJobs::new(0, ["lfm3"]);
    let (a, seen_a) = decoder(0, 3, true, None);
    let (b, seen_b) = decoder(1, 3, false, None);
    let (c, seen_c) = decoder(2, 3, true, Some(0));
    add(&mut jobs, a);
    add(&mut jobs, b);
    add(&mut jobs, c);
    run_all(&mut jobs, &mut engines);
    // Quantum 1: b alone, then a and c in one pass. Then c failed, so a joins nothing, and b runs alone.
    assert_eq!(*passes.lock().unwrap(), [1, 2, 1, 1, 1, 1]);
    saw_its_own_rows(&seen_a, 0, 3);
    saw_its_own_rows(&seen_b, 1, 3);
    saw_its_own_rows(&seen_c, 2, 1);
}
