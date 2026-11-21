//! The group driver batches the decode quanta of an engine with a feed rule and more than one segment (milestone 27
//! step 5, engine-abi "LFM3", "Batched passes"). Each job gives its segment, one pass runs them all, and each job gets
//! its own rows. A job that cannot join runs alone in the same quantum, and one job can fail alone.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use nagaya::jobs::{GroupJobs, JobBody, JobSpec, Ran};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Dtype, Engine, FeedRule, Identity, Kind, LayerLoop, Limits, Output, OutputsWanted, PassError, PassOut, PassStatus,
    Segment, SegmentInputs, State, Tap,
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
        // An engine runs one loop in a pass (engine-abi "Layer loops").
        assert!(
            segments.iter().all(|s| s.layer_loop == segments[0].layer_loop),
            "a pass mixes layer loops: {:?}",
            segments.iter().map(|s| s.layer_loop).collect::<Vec<_>>()
        );
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

/// Where the rows of one pass meet: the count of bodies that prepare the end of a pass now.
#[derive(Default)]
struct Meet {
    inside: Mutex<usize>,
    arrived: Condvar,
}

/// What a body does when the driver prepares the end of its pass.
struct Prepares {
    meet: Arc<Meet>,
    /// The count of bodies that the body waits for, with itself.
    members: usize,
    /// The row that the body prepared, until its pass ends.
    row: Option<Vec<u8>>,
    /// True for each prepared row that met the rows of the other members in time.
    met: Arc<Mutex<Vec<bool>>>,
}

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
    /// When it is set, the end of each pass must come after its preparation, with the same row.
    prepares: Option<Prepares>,
    /// The layer loop of each segment of the job.
    layer_loop: Option<LayerLoop>,
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
            layer_loop: self.layer_loop,
            cycles: None,
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

    fn prepare_end(&mut self, _kind: Kind, result: &Result<PassOut, PassError>) {
        let Some(prepares) = &mut self.prepares else {
            return;
        };
        let out = result.as_ref().expect("the batch ran");
        assert!(prepares.row.is_none(), "the driver prepared two ends before one end");
        prepares.row = Some(out.outputs[0].bytes.clone());
        let mut inside = prepares.meet.inside.lock().unwrap();
        *inside += 1;
        prepares.meet.arrived.notify_all();
        // A driver that prepares the rows in turn never has all members inside at one time.
        let (mut inside, wait) = prepares
            .meet
            .arrived
            .wait_timeout_while(inside, Duration::from_secs(2), |inside| *inside < prepares.members)
            .unwrap();
        prepares.met.lock().unwrap().push(!wait.timed_out());
        if wait.timed_out() {
            *inside -= 1;
        }
    }

    fn end_pass(&mut self, _kind: Kind, result: Result<PassOut, PassError>, _engines: &mut [Box<dyn Engine>]) -> Ran {
        let out = result.expect("the batch ran");
        if let Some(prepares) = &mut self.prepares {
            let row = prepares
                .row
                .take()
                .expect("the driver prepares the end of a shared pass");
            assert_eq!(
                row, out.outputs[0].bytes,
                "the body prepared the row that its pass ends with"
            );
            // The last member to end its pass empties the meeting place for the next pass.
            *prepares.meet.inside.lock().unwrap() = 0;
        }
        self.took(out)
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
        prepares: None,
        layer_loop: None,
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

/// Decode rows join only with rows of the same layer loop, because an engine runs one loop in a pass (spec 11.16).
/// Two jobs of one loop share a pass, and the two jobs with no loop share another.
///
/// Falsified by a driver that joins rows of any loop: the engine sees a pass that mixes loops.
#[test]
fn decode_rows_join_only_with_rows_of_the_same_layer_loop() {
    let passes = Passes::default();
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Rows::new(true, false, 8, &passes))];
    let mut jobs = GroupJobs::new(0, ["lfm3"]);
    let mut seen = Vec::new();
    let looped = Some(LayerLoop {
        start: 10,
        end: 14,
        count: 2,
    });
    for (slot, layer_loop) in [(0, looped), (1, None), (2, looped), (3, None)] {
        let (mut body, s) = decoder(slot, 3, true, None);
        body.layer_loop = layer_loop;
        add(&mut jobs, body);
        seen.push(s);
    }
    run_all(&mut jobs, &mut engines);
    assert_eq!(*passes.lock().unwrap(), [2, 2, 2, 2, 2, 2]);
    for (slot, s) in seen.iter().enumerate() {
        saw_its_own_rows(s, slot as u32, 3);
    }
}

/// The driver prepares the ends of the rows of one pass at the same time, each on a row thread, before the first end
/// (spec 3.2, "The row threads"). Then each pass ends in the order of its segments, as before.
///
/// Falsified by a driver that prepares the rows in turn: no row meets the others.
#[test]
fn the_rows_of_a_shared_pass_prepare_their_ends_at_the_same_time() {
    let passes = Passes::default();
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Rows::new(true, false, 8, &passes))];
    let mut jobs = GroupJobs::new(0, ["lfm3"]);
    let meet = Arc::new(Meet::default());
    let met = Arc::new(Mutex::new(Vec::new()));
    let mut seen = Vec::new();
    for slot in 0..4 {
        let (mut body, s) = decoder(slot, 3, true, None);
        body.prepares = Some(Prepares {
            meet: Arc::clone(&meet),
            members: 4,
            row: None,
            met: Arc::clone(&met),
        });
        add(&mut jobs, body);
        seen.push(s);
    }
    run_all(&mut jobs, &mut engines);
    assert_eq!(*passes.lock().unwrap(), [4, 4, 4]);
    assert_eq!(
        *met.lock().unwrap(),
        [true; 12],
        "each row of each pass met the three others"
    );
    for (slot, s) in seen.iter().enumerate() {
        saw_its_own_rows(s, slot as u32, 3);
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
