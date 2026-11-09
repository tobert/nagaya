use nagaya_abi::Kind;
use nagaya_sched::{Class, Completion, Engine, Event, Job, JobId, JobState, Quantum, Scheduler, StepError};

/// Step with an event that must fit the state.
fn step(state: Scheduler, event: Event) -> (Scheduler, Option<Quantum>) {
    nagaya_sched::step(state, event).expect("the event fits the state")
}

/// Step with an event that may not fit the state.
fn try_step(state: Scheduler, event: Event) -> Result<(Scheduler, Option<Quantum>), StepError> {
    nagaya_sched::step(state, event)
}

fn job(id: JobId, class: Class, priority: i64, arrival_ms: u64, engine: u32, quanta: &[Kind]) -> Job {
    Job {
        id,
        class,
        priority,
        arrival_ms,
        deadline_ms: None,
        engine,
        quanta: quanta.iter().copied().collect(),
        admitted: true,
        inputs_present: true,
        state: JobState::Ready,
        balance_ms: 0,
        cancel_requested: false,
        checkpoint_interval_ms: None,
    }
}

fn state(jobs: Vec<Job>) -> Scheduler {
    Scheduler {
        jobs,
        engines: Vec::new(),
        running: None,
    }
}

fn quantum(class: Class, kind: Kind, segments: &[JobId]) -> Option<Quantum> {
    Some(Quantum {
        engine: 0,
        class,
        kind,
        segments: segments.to_vec(),
    })
}

#[test]
fn an_interactive_job_runs_before_a_batch_job() {
    let jobs = vec![
        job(1, Class::Batch, 10, 0, 0, &[Kind::Prefill]),
        job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]),
    ];
    let (_, next) = step(state(jobs), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[2]));
}

#[test]
fn a_class_applies_at_the_first_quantum() {
    let mut running_job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill, Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let waiting = {
        let mut job = job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]);
        job.admitted = false;
        job
    };
    let other = job(3, Class::Batch, 0, 2, 0, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![running_job, waiting, other],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Batch,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, next) = step(scheduler, Event::Tick { now_ms: 1 });
    assert_eq!(next, None);
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 2,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
}

#[test]
fn batch_orders_by_priority_then_arrival() {
    let jobs = vec![
        job(1, Class::Batch, 1, 0, 0, &[Kind::Prefill]),
        job(2, Class::Batch, 2, 10, 0, &[Kind::Prefill]),
        job(3, Class::Batch, 2, 5, 0, &[Kind::Prefill]),
    ];
    let (scheduler, next) = step(state(jobs), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[3]));
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 10,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[2]));
}

#[test]
fn batch_keeps_the_flow() {
    let jobs = vec![
        job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill, Kind::Prefill]),
        job(2, Class::Batch, 0, 1, 0, &[Kind::Prefill]),
    ];
    let (scheduler, next) = step(state(jobs), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 10,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
}

#[test]
fn interactive_interleaves_by_owed_seconds() {
    let mut first = job(1, Class::Interactive, 0, 0, 0, &[Kind::Prefill, Kind::Prefill]);
    first.balance_ms = 1000;
    let second = job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]);
    let (scheduler, next) = step(state(vec![first, second]), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[1]));
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 2000,
            duration_ms: 2000,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[2]));
}

#[test]
fn interactive_priority_comes_before_owed_seconds() {
    let mut first = job(1, Class::Interactive, 1, 0, 0, &[Kind::Prefill]);
    first.balance_ms = 0;
    let mut second = job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]);
    second.balance_ms = 30_000;
    let (_, next) = step(state(vec![first, second]), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[1]));
}

#[test]
fn a_filler_job_waits_for_interactive_and_batch() {
    let filler = job(1, Class::Filler, 100, 0, 0, &[Kind::Prefill]);
    let batch = job(2, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    let (_, next) = step(state(vec![filler.clone(), batch]), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[2]));
    let (_, next) = step(state(vec![filler]), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Filler, Kind::Prefill, &[1]));
}

#[test]
fn a_running_quantum_is_not_preempted() {
    let mut running_job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let waiting = job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![running_job, waiting],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Batch,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let arriving = job(3, Class::Interactive, 0, 2, 0, &[Kind::Prefill]);
    let (scheduler, next) = step(
        scheduler,
        Event::Arrive {
            now_ms: 1,
            job: arriving,
        },
    );
    assert_eq!(next, None);
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 10,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[2]));
}

#[test]
fn ready_segments_of_one_class_share_a_pass() {
    let first = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    let second = job(2, Class::Batch, 0, 1, 0, &[Kind::Prefill]);
    let other_engine = job(3, Class::Batch, 0, 2, 1, &[Kind::Prefill]);
    let mut not_ready = job(4, Class::Batch, 0, 3, 0, &[Kind::Prefill]);
    not_ready.admitted = false;
    let mut scheduler = state(vec![first, second, other_engine, not_ready]);
    scheduler.engines = vec![Engine {
        id: 0,
        segments_exact: true,
        max_segments: 16,
        joins_reads: false,
        joins_decode: false,
    }];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1, 2]));
}

#[test]
fn an_engine_with_a_feed_rule_joins_reads_only() {
    let feed = Engine {
        id: 0,
        segments_exact: false,
        max_segments: 16,
        joins_reads: true,
        joins_decode: false,
    };
    let reads = vec![
        job(1, Class::Batch, 0, 0, 0, &[Kind::Read]),
        job(2, Class::Batch, 0, 1, 0, &[Kind::Read]),
    ];
    let mut scheduler = state(reads);
    scheduler.engines = vec![feed];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Read, &[1, 2]));
    // The rows of a prefill are kept, so a prefill has a pass alone (engine-abi D11 rule 6).
    let prefills = vec![
        job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]),
        job(2, Class::Batch, 0, 1, 0, &[Kind::Prefill]),
    ];
    let mut scheduler = state(prefills);
    scheduler.engines = vec![feed];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
}

/// An engine with a feed rule and batched decode joins decode quanta, whose rows the store never keeps (engine-abi
/// D11 rule 6, "LFM3", "Batched passes"). A prefill still has a pass alone.
#[test]
fn an_engine_that_joins_decode_joins_decode_quanta_and_not_prefills() {
    let batched = Engine {
        id: 0,
        segments_exact: false,
        max_segments: 8,
        joins_reads: true,
        joins_decode: true,
    };
    let decodes = || {
        (1..=10)
            .map(|id| job(id, Class::Interactive, 0, id, 0, &[Kind::Decode]))
            .collect::<Vec<_>>()
    };
    let mut scheduler = state(decodes());
    scheduler.engines = vec![batched];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(
        next,
        quantum(Class::Interactive, Kind::Decode, &[1, 2, 3, 4, 5, 6, 7, 8])
    );
    // Without joins_decode, a decode quantum is alone.
    let mut scheduler = state(decodes());
    scheduler.engines = vec![Engine {
        joins_decode: false,
        ..batched
    }];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Interactive, Kind::Decode, &[1]));
    let mut scheduler = state(vec![
        job(1, Class::Interactive, 0, 0, 0, &[Kind::Prefill]),
        job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]),
    ]);
    scheduler.engines = vec![batched];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[1]));
}

/// One job of a joined quantum can fail alone. The others go on.
#[test]
fn one_job_of_a_joined_quantum_fails_and_the_others_go_on() {
    let mut scheduler = state(vec![
        job(1, Class::Interactive, 0, 0, 0, &[Kind::Decode, Kind::Decode]),
        job(2, Class::Interactive, 0, 1, 0, &[Kind::Decode, Kind::Decode]),
    ]);
    scheduler.engines = vec![Engine {
        id: 0,
        segments_exact: false,
        max_segments: 8,
        joins_reads: true,
        joins_decode: true,
    }];
    let (mut scheduler, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Interactive, Kind::Decode, &[1, 2]));
    scheduler.fail_running(2).unwrap();
    assert!(
        scheduler.fail_running(3).is_err(),
        "job 3 is not in the running quantum"
    );
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 5,
            duration_ms: 5,
            outcome: Completion::Done,
        },
    );
    let state_of = |id: JobId| scheduler.jobs.iter().find(|j| j.id == id).unwrap().state;
    assert_eq!(state_of(2), JobState::Failed);
    assert_eq!(next, quantum(Class::Interactive, Kind::Decode, &[1]));
}

#[test]
fn a_job_past_its_deadline_expires() {
    let mut job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    job.deadline_ms = Some(100);
    let (scheduler, next) = step(state(vec![job]), Event::Tick { now_ms: 101 });
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Expired);
}

#[test]
fn a_running_job_expires_after_its_quantum() {
    let mut job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill, Kind::Prefill]);
    job.deadline_ms = Some(100);
    job.state = JobState::Running;
    job.quanta.pop_front();
    let scheduler = Scheduler {
        jobs: vec![job],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Batch,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, next) = step(scheduler, Event::Tick { now_ms: 101 });
    assert_eq!(next, None);
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 101,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Expired);
}

#[test]
fn a_cancelled_job_runs_no_more() {
    let mut running_job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let waiting = job(2, Class::Batch, 0, 1, 0, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![running_job, waiting],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Batch,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, next) = step(scheduler, Event::Cancel { now_ms: 10, job: 2 });
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[1].state, JobState::Cancelled);
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 20,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Done);
}

#[test]
fn cancelling_the_running_job_is_not_failure() {
    let mut running_job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let scheduler = Scheduler {
        jobs: vec![running_job],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Batch,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, next) = step(scheduler, Event::Cancel { now_ms: 10, job: 1 });
    assert_eq!(next, None);
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 20,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Cancelled);
}

#[test]
fn a_failed_job_is_failed() {
    let mut running_job = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let scheduler = Scheduler {
        jobs: vec![running_job],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Batch,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 20,
            duration_ms: 10,
            outcome: Completion::Failed,
        },
    );
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Failed);
}

#[test]
fn the_next_quantum_has_the_jobs_kind() {
    let jobs = vec![job(1, Class::Batch, 0, 0, 0, &[Kind::Decode])];
    let (_, next) = step(state(jobs), Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Decode, &[1]));
}

#[test]
fn owed_balances_stay_zero_sum() {
    let jobs = vec![
        job(1, Class::Interactive, 0, 0, 0, &[Kind::Prefill; 20]),
        job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill; 20]),
        job(3, Class::Interactive, 0, 2, 0, &[Kind::Prefill; 20]),
    ];
    let scheduler = state(jobs);
    let (mut scheduler, mut next) = step(scheduler, Event::Tick { now_ms: 0 });
    for duration in [1, 2, 5, 7, 3, 11, 4, 9, 6, 8] {
        assert!(next.is_some());
        let (next_state, next_quantum) = step(
            scheduler,
            Event::Complete {
                now_ms: 0,
                duration_ms: duration,
                outcome: Completion::Done,
            },
        );
        scheduler = next_state;
        next = next_quantum;
    }
    assert!(scheduler.jobs.iter().all(|job| job.balance_ms.abs() < 30_000));
    let total: i64 = scheduler.jobs.iter().map(|job| job.balance_ms).sum();
    assert_eq!(total, 0);
}

#[test]
fn the_owed_remainder_goes_to_the_earliest() {
    let mut running_job = job(1, Class::Interactive, 1, 1, 0, &[Kind::Prefill, Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let earliest = job(2, Class::Interactive, 0, 0, 0, &[Kind::Prefill]);
    let latest = job(3, Class::Interactive, 0, 2, 0, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![running_job, earliest, latest],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Interactive,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, _) = step(
        scheduler,
        Event::Complete {
            now_ms: 2,
            duration_ms: 2,
            outcome: Completion::Done,
        },
    );
    assert_eq!(scheduler.jobs[0].balance_ms, -1);
    assert_eq!(scheduler.jobs[1].balance_ms, 1);
    assert_eq!(scheduler.jobs[2].balance_ms, 0);
}

/// Three present jobs. Jobs 1 and 2 run in one pass of `duration_ms`.
fn two_segment_pass(duration_ms: u64) -> Scheduler {
    let mut first = job(1, Class::Interactive, 0, 0, 0, &[Kind::Prefill, Kind::Prefill]);
    first.state = JobState::Running;
    first.quanta.pop_front();
    let mut second = job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill, Kind::Prefill]);
    second.state = JobState::Running;
    second.quanta.pop_front();
    let third = job(3, Class::Interactive, 0, 2, 1, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![first, second, third],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Interactive,
            kind: Kind::Prefill,
            segments: vec![1, 2],
        }),
    };
    let (scheduler, _) = step(
        scheduler,
        Event::Complete {
            now_ms: duration_ms,
            duration_ms,
            outcome: Completion::Done,
        },
    );
    scheduler
}

fn balances(scheduler: &Scheduler) -> Vec<i64> {
    scheduler.jobs.iter().map(|job| job.balance_ms).collect()
}

#[test]
fn a_shared_pass_splits_its_time() {
    // Segment charges: job 1 owes 3 ms, job 2 owes 2 ms.
    // Job 1: falls 2, job 2 and job 3 rise 1 each.
    // Job 2: falls 1, job 1 and job 3 rise 0 each (2 / 3 rounds down).
    // The sum is -1. The remainder goes to the earliest job, job 1.
    assert_eq!(balances(&two_segment_pass(5)), vec![-1, 0, 1]);
}

#[test]
fn a_segment_job_rises_by_the_time_of_the_other_segments() {
    // Segment charges: job 1 owes 4 ms, job 2 owes 3 ms.
    // Job 1 falls 2 and job 2 falls 2. Each segment makes the other job and job 3 rise 1.
    // The sum is zero, so no remainder moves.
    assert_eq!(balances(&two_segment_pass(7)), vec![-1, -1, 2]);
}

#[test]
fn two_engines_get_their_own_passes() {
    let jobs = vec![
        job(1, Class::Interactive, 0, 0, 0, &[Kind::Prefill]),
        job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill]),
        job(3, Class::Interactive, 0, 0, 1, &[Kind::Prefill]),
        job(4, Class::Interactive, 0, 1, 1, &[Kind::Prefill]),
    ];
    let mut scheduler = state(jobs);
    scheduler.engines = vec![
        Engine {
            id: 0,
            segments_exact: true,
            max_segments: 16,
            joins_reads: false,
            joins_decode: false,
        },
        Engine {
            id: 1,
            segments_exact: true,
            max_segments: 16,
            joins_reads: false,
            joins_decode: false,
        },
    ];
    let (scheduler, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(
        next,
        Some(Quantum {
            engine: 0,
            class: Class::Interactive,
            kind: Kind::Prefill,
            segments: vec![1, 2],
        })
    );
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 10,
            duration_ms: 10,
            outcome: Completion::Done,
        },
    );
    assert_eq!(
        next,
        Some(Quantum {
            engine: 1,
            class: Class::Interactive,
            kind: Kind::Prefill,
            segments: vec![3, 4],
        })
    );
}

#[test]
fn the_clamp_forgives_the_excess() {
    let mut running_job = job(1, Class::Interactive, 1, 1, 0, &[Kind::Prefill, Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    let mut high = job(2, Class::Interactive, 0, 2, 0, &[Kind::Prefill]);
    high.balance_ms = 30_000;
    let earliest = job(3, Class::Interactive, 0, 0, 0, &[Kind::Prefill]);
    let latest = job(4, Class::Interactive, 0, 3, 0, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![running_job, high, earliest, latest],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Interactive,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, _) = step(
        scheduler,
        Event::Complete {
            now_ms: 4,
            duration_ms: 4,
            outcome: Completion::Done,
        },
    );
    assert_eq!(scheduler.jobs[0].balance_ms, -3);
    assert_eq!(scheduler.jobs[1].balance_ms, 30_000);
    assert_eq!(scheduler.jobs[2].balance_ms, 1);
    assert_eq!(scheduler.jobs[3].balance_ms, 1);
    let total: i64 = scheduler.jobs.iter().map(|job| job.balance_ms).sum();
    assert_eq!(total, 29_999);
}

#[test]
fn the_lower_clamp_forgives_the_excess() {
    let mut running_job = job(1, Class::Interactive, 1, 1, 0, &[Kind::Prefill, Kind::Prefill]);
    running_job.state = JobState::Running;
    running_job.quanta.pop_front();
    running_job.balance_ms = -30_000;
    let earliest = job(2, Class::Interactive, 0, 0, 0, &[Kind::Prefill]);
    let middle = job(3, Class::Interactive, 0, 2, 0, &[Kind::Prefill]);
    let latest = job(4, Class::Interactive, 0, 3, 0, &[Kind::Prefill]);
    let scheduler = Scheduler {
        jobs: vec![running_job, earliest, middle, latest],
        engines: Vec::new(),
        running: Some(Quantum {
            engine: 0,
            class: Class::Interactive,
            kind: Kind::Prefill,
            segments: vec![1],
        }),
    };
    let (scheduler, _) = step(
        scheduler,
        Event::Complete {
            now_ms: 4,
            duration_ms: 4,
            outcome: Completion::Done,
        },
    );
    assert_eq!(scheduler.jobs[0].balance_ms, -30_000);
    assert_eq!(scheduler.jobs[1].balance_ms, 1);
    assert_eq!(scheduler.jobs[2].balance_ms, 1);
    assert_eq!(scheduler.jobs[3].balance_ms, 1);
    let total: i64 = scheduler.jobs.iter().map(|job| job.balance_ms).sum();
    assert_eq!(total, -29_997);
}

fn limited_engine(max_segments: u32) -> Vec<Engine> {
    vec![Engine {
        id: 0,
        segments_exact: true,
        max_segments,
        joins_reads: false,
        joins_decode: false,
    }]
}

#[test]
fn a_pass_holds_no_more_segments_than_the_engine_allows() {
    let jobs = (1..=3)
        .map(|id| job(id, Class::Batch, 0, id, 0, &[Kind::Prefill]))
        .collect();
    let mut scheduler = state(jobs);
    scheduler.engines = limited_engine(2);
    let (scheduler, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1, 2]));
    let (_, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 1,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[3]));
}

#[test]
fn an_engine_with_one_segment_runs_one_job_a_pass() {
    let jobs = (1..=2)
        .map(|id| job(id, Class::Batch, 0, id, 0, &[Kind::Prefill]))
        .collect();
    let mut scheduler = state(jobs);
    scheduler.engines = limited_engine(1);
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
}

#[test]
#[should_panic(expected = "max_segments")]
fn an_engine_that_allows_no_segment_is_a_bug() {
    let mut scheduler = state(vec![job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill])]);
    scheduler.engines = limited_engine(0);
    let _ = step(scheduler, Event::Tick { now_ms: 0 });
}

#[test]
fn a_pass_holds_segments_of_one_kind() {
    let jobs = vec![
        job(1, Class::Batch, 0, 0, 0, &[Kind::Decode]),
        job(2, Class::Batch, 0, 1, 0, &[Kind::Prefill]),
        job(3, Class::Batch, 0, 2, 0, &[Kind::Decode]),
    ];
    let mut scheduler = state(jobs);
    scheduler.engines = limited_engine(16);
    let (scheduler, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Decode, &[1, 3]));
    let (scheduler, _) = step(
        scheduler,
        Event::Complete {
            now_ms: 1,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    // The prefill job still has its own quantum, and it is still ready.
    assert_eq!(scheduler.running, quantum(Class::Batch, Kind::Prefill, &[2]));
    assert!(scheduler.jobs[1].quanta.is_empty());
}

fn error_of(state: Scheduler, event: Event) -> StepError {
    try_step(state, event).expect_err("the event does not fit the state")
}

#[test]
fn admit_of_an_unknown_job_is_an_error() {
    let scheduler = state(vec![job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill])]);
    assert_eq!(
        error_of(scheduler, Event::Admit { now_ms: 0, job: 9 }),
        StepError::UnknownJob(9)
    );
}

#[test]
fn cancel_of_an_unknown_job_is_an_error() {
    let scheduler = state(vec![job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill])]);
    assert_eq!(
        error_of(scheduler, Event::Cancel { now_ms: 0, job: 9 }),
        StepError::UnknownJob(9)
    );
}

#[test]
fn complete_with_nothing_running_is_an_error() {
    let event = Event::Complete {
        now_ms: 0,
        duration_ms: 1,
        outcome: Completion::Done,
    };
    assert_eq!(error_of(state(Vec::new()), event), StepError::NothingRunning);
}

#[test]
fn a_duplicate_arrival_is_an_error() {
    let scheduler = state(vec![job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill])]);
    let again = job(1, Class::Batch, 0, 5, 0, &[Kind::Prefill]);
    assert_eq!(
        error_of(scheduler, Event::Arrive { now_ms: 5, job: again }),
        StepError::DuplicateJob(1)
    );
}

#[test]
fn an_arrival_that_is_not_ready_is_an_error() {
    let mut arriving = job(2, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    arriving.state = JobState::Running;
    let event = Event::Arrive {
        now_ms: 0,
        job: arriving,
    };
    assert_eq!(
        error_of(state(Vec::new()), event),
        StepError::ArrivalNotReady(2, JobState::Running)
    );
}

#[test]
fn an_arrival_and_an_admit_that_fit_are_not_errors() {
    let mut late = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    late.admitted = false;
    let (scheduler, next) = step(state(Vec::new()), Event::Arrive { now_ms: 0, job: late });
    assert_eq!(next, None);
    let (_, next) = step(scheduler, Event::Admit { now_ms: 1, job: 1 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
}

#[test]
fn a_job_expires_at_its_deadline_and_not_before() {
    let mut early = job(1, Class::Batch, 0, 0, 0, &[Kind::Prefill]);
    early.deadline_ms = Some(100);
    let (scheduler, next) = step(state(vec![early.clone()]), Event::Tick { now_ms: 99 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1]));
    assert_eq!(scheduler.jobs[0].state, JobState::Running);
    let (scheduler, next) = step(state(vec![early]), Event::Tick { now_ms: 100 });
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Expired);
}

#[test]
fn a_running_job_can_append_quanta_and_then_runs_them() {
    let (scheduler, next) = step(
        state(vec![job(1, Class::Interactive, 0, 0, 0, &[Kind::Copy])]),
        Event::Tick { now_ms: 0 },
    );
    assert_eq!(next, quantum(Class::Interactive, Kind::Copy, &[1]));
    let mut scheduler = scheduler;
    scheduler.append_quanta(1, &[Kind::Prefill, Kind::Copy]).unwrap();
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 1,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[1]));
    let (scheduler, _) = step(
        scheduler,
        Event::Complete {
            now_ms: 2,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    assert_eq!(scheduler.running.as_ref().map(|q| q.kind), Some(Kind::Copy));
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 3,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, None);
    assert_eq!(scheduler.jobs[0].state, JobState::Done);
}

#[test]
fn appended_quanta_keep_their_job_in_its_class_against_other_jobs() {
    let jobs = vec![
        job(1, Class::Batch, 0, 0, 0, &[Kind::Copy]),
        job(2, Class::Interactive, 0, 1, 0, &[Kind::Prefill, Kind::Prefill]),
    ];
    let mut scheduler = state(jobs);
    scheduler.jobs[1].admitted = false;
    let (mut scheduler, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Copy, &[1]));
    scheduler.append_quanta(1, &[Kind::Prefill]).unwrap();
    let (scheduler, _) = step(scheduler, Event::Admit { now_ms: 1, job: 2 });
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 2,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    // The interactive job goes first, and the batch job keeps its appended quantum.
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[2]));
    assert_eq!(scheduler.jobs[0].quanta.len(), 1);
}

#[test]
fn a_running_job_can_put_a_quantum_before_the_ones_that_it_has() {
    // Spec 6.13: an interval copy holds the state at the end of the quantum that crossed the multiple. Thus the copy
    // runs before the next quantum of the job.
    let (mut scheduler, next) = step(
        state(vec![job(
            1,
            Class::Interactive,
            0,
            0,
            0,
            &[Kind::Prefill, Kind::Prefill],
        )]),
        Event::Tick { now_ms: 0 },
    );
    assert_eq!(next, quantum(Class::Interactive, Kind::Prefill, &[1]));
    scheduler.prepend_quanta(1, &[Kind::Copy]).unwrap();
    let (scheduler, next) = step(
        scheduler,
        Event::Complete {
            now_ms: 1,
            duration_ms: 1,
            outcome: Completion::Done,
        },
    );
    assert_eq!(next, quantum(Class::Interactive, Kind::Copy, &[1]));
    assert_eq!(scheduler.jobs[0].quanta, [Kind::Prefill]);
    let mut idle = state(vec![job(2, Class::Batch, 0, 0, 0, &[Kind::Copy])]);
    assert_eq!(
        idle.prepend_quanta(2, &[Kind::Copy]),
        Err(StepError::AppendToIdleJob(2, JobState::Ready))
    );
}

#[test]
fn appending_to_a_job_that_does_not_run_is_an_error() {
    let mut scheduler = state(vec![job(1, Class::Batch, 0, 0, 0, &[Kind::Copy])]);
    assert_eq!(
        scheduler.append_quanta(1, &[Kind::Prefill]),
        Err(StepError::AppendToIdleJob(1, JobState::Ready))
    );
    assert_eq!(
        scheduler.append_quanta(9, &[Kind::Prefill]),
        Err(StepError::UnknownJob(9))
    );
    assert_eq!(scheduler.jobs[0].quanta.len(), 1, "a refusal changes nothing");
}
