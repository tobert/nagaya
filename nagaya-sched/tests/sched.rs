use nagaya_abi::Kind;
use nagaya_sched::{Class, Completion, Engine, Event, Job, JobId, JobState, Quantum, Scheduler, step};

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
    }];
    let (_, next) = step(scheduler, Event::Tick { now_ms: 0 });
    assert_eq!(next, quantum(Class::Batch, Kind::Prefill, &[1, 2]));
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

#[test]
fn a_shared_pass_splits_its_time() {
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
            now_ms: 5,
            duration_ms: 5,
            outcome: Completion::Done,
        },
    );
    assert_eq!(scheduler.jobs[0].balance_ms, 0);
    assert_eq!(scheduler.jobs[1].balance_ms, -1);
    assert_eq!(scheduler.jobs[2].balance_ms, 1);
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
        },
        Engine {
            id: 1,
            segments_exact: true,
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
