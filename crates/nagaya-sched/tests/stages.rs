//! The scheduler of a group with stages (engine-abi D1, "The hold of the slot", spec 5.1-5.9).

use nagaya_abi::Kind;
use nagaya_sched::{
    Class, Completion, Device, Item, JobState, StageError, StageEvent, StageJob, StageQuantum, StageScheduler,
    StagedEngine, Work, stage_step,
};

/// Two stages on devices 0 and 1, two residual buffers, and a bound of 10 ms.
fn scheduler() -> StageScheduler {
    StageScheduler::new(StagedEngine {
        id: 0,
        devices: vec![0, 1],
        buffers: 2,
        max_quantum_ms: 10,
    })
}

struct Run {
    state: Option<StageScheduler>,
    now: u64,
}

impl Run {
    fn new() -> Self {
        Self {
            state: Some(scheduler()),
            now: 0,
        }
    }

    fn step(&mut self, event: StageEvent) -> Vec<StageQuantum> {
        let (state, started) = stage_step(self.state.take().unwrap(), event).unwrap();
        self.state = Some(state);
        started
    }

    fn arrive(&mut self, job: StageJob) -> Vec<StageQuantum> {
        self.step(StageEvent::Arrive { now_ms: self.now, job })
    }

    fn complete(&mut self, device: Device, ms: u64) -> Vec<StageQuantum> {
        self.complete_with(device, ms, Completion::Done)
    }

    fn complete_with(&mut self, device: Device, ms: u64, outcome: Completion) -> Vec<StageQuantum> {
        self.now += ms;
        self.step(StageEvent::Complete {
            now_ms: self.now,
            device,
            duration_ms: ms,
            outcome,
        })
    }

    fn state(&self) -> &StageScheduler {
        self.state.as_ref().unwrap()
    }

    fn job(&self, id: u64) -> &StageJob {
        self.state().jobs().iter().find(|job| job.id == id).unwrap()
    }

    fn on(&self, device: Device) -> Option<(u64, Work)> {
        self.state()
            .running_on(device)
            .map(|quantum| (quantum.job, quantum.work))
    }
}

fn job(id: u64, class: Class, items: &[Item]) -> StageJob {
    StageJob::new(id, class, id, items.iter().copied())
}

fn stage(block: u64, stage: usize) -> Work {
    Work::Stage { block, stage }
}

fn started(quanta: &[StageQuantum]) -> Vec<(u64, Work, Vec<Device>)> {
    quanta
        .iter()
        .map(|quantum| (quantum.job, quantum.work, quantum.devices.clone()))
        .collect()
}

const PREFILL: Item = Item::Block(Kind::Prefill);

#[test]
fn stage_one_of_a_block_starts_only_after_stage_zero_ends() {
    let mut run = Run::new();
    assert_eq!(
        started(&run.arrive(job(1, Class::Interactive, &[PREFILL]))),
        vec![(1, stage(0, 0), vec![0])]
    );
    assert_eq!(run.on(1), None, "stage 1 waits for the residual of stage 0");
    assert_eq!(started(&run.complete(0, 5)), vec![(1, stage(0, 1), vec![1])]);
    assert_eq!(run.job(1).state, JobState::Running);
    assert!(run.complete(1, 5).is_empty());
    assert_eq!(run.job(1).state, JobState::Done);
}

#[test]
fn two_blocks_of_one_job_run_on_both_devices_at_once() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Batch, &[PREFILL, PREFILL, PREFILL]));
    let quanta = run.complete(0, 5);
    assert_eq!(
        started(&quanta),
        vec![(1, stage(1, 0), vec![0]), (1, stage(0, 1), vec![1])],
        "block 1 takes device 0 while block 0 runs on device 1"
    );
    assert_eq!((run.on(0), run.on(1)), (Some((1, stage(1, 0))), Some((1, stage(0, 1)))));
    // Stage 1 of block 1 waits for stage 1 of block 0 (rule 2). Block 2 waits for a residual buffer.
    assert!(run.complete(0, 5).is_empty());
    assert_eq!(
        started(&run.complete(1, 6)),
        vec![(1, stage(2, 0), vec![0]), (1, stage(1, 1), vec![1])]
    );
    assert!(
        run.complete(0, 1).is_empty(),
        "stage 1 of block 2 waits for stage 1 of block 1"
    );
    assert_eq!(started(&run.complete(1, 5)), vec![(1, stage(2, 1), vec![1])]);
    run.complete(1, 5);
    assert_eq!(run.job(1).state, JobState::Done);
}

#[test]
fn a_block_waits_for_a_free_residual_buffer() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Batch, &[PREFILL, PREFILL, PREFILL]));
    run.complete(0, 1); // block 0 goes to device 1, block 1 to device 0
    // Block 0 holds a buffer on device 1, and block 1 holds the second one. Block 2 must wait.
    assert!(run.complete(0, 1).is_empty(), "both buffers of the boundary are held");
    assert_eq!(run.on(0), None);
    let quanta = run.complete(1, 20);
    assert_eq!(
        started(&quanta),
        vec![(1, stage(2, 0), vec![0]), (1, stage(1, 1), vec![1])]
    );
}

#[test]
fn a_council_read_takes_device_zero_while_a_generation_runs_stage_one() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Interactive, &[Item::Block(Kind::Decode)]));
    run.complete(0, 2);
    assert_eq!(run.on(1), Some((1, stage(0, 1))));
    let quanta = run.arrive(job(2, Class::Interactive, &[Item::Block(Kind::Read)]));
    assert_eq!(started(&quanta), vec![(2, stage(1, 0), vec![0])]);
    assert_eq!(quanta[0].kind, Kind::Read);
    assert_eq!((run.on(0), run.on(1)), (Some((2, stage(1, 0))), Some((1, stage(0, 1)))));
}

#[test]
fn each_device_runs_its_ready_quantum_of_the_highest_class() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Filler, &[PREFILL, PREFILL]));
    run.arrive(job(2, Class::Batch, &[PREFILL]));
    run.arrive(job(3, Class::Interactive, &[PREFILL]));
    // Device 0 runs the filler that came first. When it ends, the interactive job takes device 0, and the filler's
    // stage 1 has device 1 to itself (engine-abi D1, "Filler on each device").
    assert_eq!(run.on(0), Some((1, stage(0, 0))));
    let quanta = run.complete(0, 3);
    assert_eq!(
        started(&quanta),
        vec![(3, stage(1, 0), vec![0]), (1, stage(0, 1), vec![1])]
    );
    // Then batch before filler on device 0, and the interactive stage 1 on device 1.
    assert!(run.complete(1, 1).is_empty());
    assert_eq!(
        started(&run.complete(0, 3)),
        vec![(2, stage(2, 0), vec![0]), (3, stage(1, 1), vec![1])]
    );
}

#[test]
fn a_restore_copies_on_each_device_before_the_first_stage_of_its_block() {
    let mut run = Run::new();
    let quanta = run.arrive(job(1, Class::Interactive, &[Item::Restore, PREFILL]));
    assert_eq!(
        started(&quanta),
        vec![
            (1, Work::Restore { block: 0, stage: 0 }, vec![0]),
            (1, Work::Restore { block: 0, stage: 1 }, vec![1]),
        ]
    );
    assert!(quanta.iter().all(|quantum| quantum.kind == Kind::Copy));
    // Stage 0 of the block follows the copy on device 0. Stage 1 waits for the copy on device 1.
    assert_eq!(started(&run.complete(0, 1)), vec![(1, stage(1, 0), vec![0])]);
    assert!(run.complete(0, 1).is_empty());
    assert_eq!(started(&run.complete(1, 9)), vec![(1, stage(1, 1), vec![1])]);
}

#[test]
fn a_copy_of_the_whole_state_waits_for_every_device_and_holds_them() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Batch, &[PREFILL, Item::Whole(Kind::Copy)]));
    run.complete(0, 1);
    assert_eq!(run.on(1), Some((1, stage(0, 1))));
    // The copy of job 1 waits for its own block. A job of a lower rank does not take device 0 before it either.
    run.arrive(job(2, Class::Interactive, &[Item::Whole(Kind::Copy)]));
    run.arrive(job(3, Class::Filler, &[PREFILL]));
    assert_eq!(run.on(0), None, "the interactive copy holds device 0 free for itself");
    let quanta = run.complete(1, 4);
    assert_eq!(started(&quanta), vec![(2, Work::Whole, vec![0, 1])]);
    assert_eq!((run.on(0), run.on(1)), (Some((2, Work::Whole)), Some((2, Work::Whole))));
    // A copy ends once, on either device, and frees both.
    let quanta = run.complete(1, 2);
    assert_eq!(started(&quanta), vec![(1, Work::Whole, vec![0, 1])]);
    run.complete(0, 2);
    assert_eq!(run.job(1).state, JobState::Done);
    assert_eq!(run.on(0), Some((3, stage(1, 0))));
}

#[test]
fn each_stage_quantum_has_its_own_bound() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Batch, &[PREFILL]));
    assert!(run.state().past_bound(8).is_empty());
    run.complete(0, 8);
    // The block ran 8 ms so far. Stage 1 started at 8 ms, so at 16 ms it is within its bound.
    assert!(run.state().past_bound(16).is_empty());
    let late = run.state().past_bound(19);
    assert_eq!(late.len(), 1);
    assert_eq!((late[0].0, late[0].1.work), (1, stage(0, 1)));
}

#[test]
fn a_failed_stage_fails_the_job_and_starts_nothing_more_of_it() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Batch, &[PREFILL, PREFILL, PREFILL]));
    run.complete(0, 1);
    // Block 1 fails on device 0 while block 0 runs on device 1.
    assert!(run.complete_with(0, 1, Completion::Failed).is_empty());
    assert_eq!(run.job(1).state, JobState::Failed);
    assert!(
        run.complete(1, 1).is_empty(),
        "the stage that ran goes to its end, and nothing follows"
    );
    assert_eq!(run.job(1).state, JobState::Failed);
    assert!(run.state().idle());
}

#[test]
fn a_cancel_lets_the_running_stages_end_and_drops_the_rest() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Batch, &[PREFILL, PREFILL]));
    run.complete(0, 1);
    run.step(StageEvent::Cancel { now_ms: 2, job: 1 });
    assert_eq!(run.job(1).state, JobState::Running);
    assert!(run.complete(0, 1).is_empty(), "stage 1 of block 1 does not start");
    assert_eq!(run.job(1).state, JobState::Running);
    assert!(run.complete(1, 1).is_empty());
    assert_eq!(run.job(1).state, JobState::Cancelled);
}

#[test]
fn a_job_past_its_deadline_expires_between_quanta() {
    let mut run = Run::new();
    let mut late = job(1, Class::Batch, &[PREFILL]);
    late.deadline_ms = Some(3);
    run.arrive(job(2, Class::Interactive, &[Item::Whole(Kind::Copy)]));
    run.arrive(late);
    run.complete(0, 5);
    assert_eq!(run.job(1).state, JobState::Expired);
}

#[test]
fn an_interactive_stage_quantum_charges_its_job() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Interactive, &[PREFILL]));
    run.arrive(job(2, Class::Interactive, &[Item::Whole(Kind::Copy)]));
    run.complete(0, 10);
    assert_eq!((run.job(1).balance_ms, run.job(2).balance_ms), (-5, 5));
}

#[test]
fn an_event_that_does_not_fit_is_an_error() {
    let state = scheduler();
    let (state, _) = stage_step(
        state,
        StageEvent::Arrive {
            now_ms: 0,
            job: job(1, Class::Batch, &[]),
        },
    )
    .unwrap();
    let again = stage_step(
        state.clone(),
        StageEvent::Arrive {
            now_ms: 0,
            job: job(1, Class::Batch, &[]),
        },
    );
    assert_eq!(again.unwrap_err(), StageError::DuplicateJob(1));
    let idle = stage_step(
        state.clone(),
        StageEvent::Complete {
            now_ms: 0,
            device: 1,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    assert_eq!(idle.unwrap_err(), StageError::NothingRunning(1));
    let unknown = stage_step(state.clone(), StageEvent::Cancel { now_ms: 0, job: 9 });
    assert_eq!(unknown.unwrap_err(), StageError::UnknownJob(9));
    let mut state = state;
    assert!(matches!(
        state.append_items(1, &[PREFILL]),
        Err(StageError::AppendToIdleJob(1, _))
    ));
}

#[test]
fn items_that_a_running_job_adds_run_after_its_blocks() {
    let mut run = Run::new();
    run.arrive(job(1, Class::Interactive, &[Item::Block(Kind::Decode)]));
    run.complete(0, 1);
    let mut state = run.state.take().unwrap();
    state.append_items(1, &[Item::Block(Kind::Decode)]).unwrap();
    run.state = Some(state);
    // The new block starts on device 0 at once: its stage 0 follows stage 0 of the block before.
    let quanta = run.step(StageEvent::Tick { now_ms: 2 });
    assert_eq!(started(&quanta), vec![(1, stage(1, 0), vec![0])]);
}

#[test]
fn a_whole_quantum_waits_for_a_higher_rank_stage_on_another_device() {
    let mut run = Run::new();
    // Job 1 (batch) finishes stage 0 on device 0 and occupies device 1 with stage 1.
    run.arrive(job(1, Class::Batch, &[PREFILL]));
    run.complete(0, 1);
    assert_eq!(run.on(1), Some((1, stage(0, 1))));

    // Job 2 (interactive) runs stage 0 on device 0 while device 1 is busy, then waits for device 1.
    run.arrive(job(2, Class::Interactive, &[PREFILL]));
    assert!(run.complete(0, 1).is_empty());
    assert_eq!((run.on(0), run.on(1)), (None, Some((1, stage(0, 1)))));

    // Job 3 (filler) arrives with a whole-group copy.
    assert!(run.arrive(job(3, Class::Filler, &[Item::Whole(Kind::Copy)])).is_empty());

    // When device 1 finishes job 1, job 2's interactive stage 1 must take device 1 before job 3's filler copy.
    let quanta = run.complete(1, 1);
    assert_eq!(started(&quanta), vec![(2, stage(1, 1), vec![1])]);
    assert_eq!((run.on(0), run.on(1)), (None, Some((2, stage(1, 1)))));

    // Once job 2 finishes on device 1, job 3's whole-group copy starts on both devices.
    let quanta = run.complete(1, 1);
    assert_eq!(started(&quanta), vec![(3, Work::Whole, vec![0, 1])]);
}
