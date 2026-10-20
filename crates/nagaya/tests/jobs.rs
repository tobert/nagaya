//! The jobs of a group run through the scheduler's state machine (spec 5.3, 5.4, 5.9).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya::jobs::{GroupJobs, JobBody, JobSpec, Ran};
use nagaya_abi::{Engine, Kind};
use nagaya_sched::Class;

type Log = Arc<Mutex<Vec<String>>>;

/// A body that writes its name and the kind of each quantum to a log.
struct Script {
    name: &'static str,
    log: Log,
    /// The quanta that the first quantum adds.
    more: Vec<Kind>,
    /// The number of quanta that have run.
    ran: usize,
    /// The quantum, counted from 0, that fails.
    fail_at: Option<usize>,
    dropped: Option<Arc<AtomicUsize>>,
    /// The time that each quantum takes, so that the scheduler has time to charge.
    sleep: Duration,
}

impl JobBody for Script {
    fn run(&mut self, kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        self.log.lock().unwrap().push(format!("{}:{kind:?}", self.name));
        std::thread::sleep(self.sleep);
        let index = self.ran;
        self.ran += 1;
        if self.fail_at == Some(index) {
            return Ran::Failed;
        }
        if index == 0 && !self.more.is_empty() {
            return Ran::More(std::mem::take(&mut self.more));
        }
        Ran::Done
    }
}

impl Drop for Script {
    fn drop(&mut self) {
        if let Some(count) = &self.dropped {
            count.fetch_add(1, Ordering::SeqCst);
        }
    }
}

fn script(name: &'static str, log: &Log, more: &[Kind]) -> Script {
    Script {
        name,
        log: Arc::clone(log),
        more: more.to_vec(),
        ran: 0,
        fail_at: None,
        dropped: None,
        sleep: Duration::ZERO,
    }
}

fn spec(class: Class, priority: i64, first: Kind, body: Script) -> JobSpec {
    JobSpec {
        class,
        priority,
        first,
        body: Box::new(body),
    }
}

fn run_all(jobs: &mut GroupJobs) {
    let mut guard = 0;
    while !jobs.idle() {
        jobs.run_one(&mut []);
        guard += 1;
        assert!(guard < 100, "the jobs never ended");
    }
}

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn lines(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

#[test]
fn a_job_runs_its_first_quantum_and_then_the_quanta_that_it_added() {
    let log = log();
    let mut jobs = GroupJobs::new(0);
    jobs.add(spec(
        Class::Interactive,
        0,
        Kind::Copy,
        script("a", &log, &[Kind::Prefill, Kind::Prefill, Kind::Copy]),
    ));
    run_all(&mut jobs);
    assert_eq!(lines(&log), ["a:Copy", "a:Prefill", "a:Prefill", "a:Copy"]);
}

#[test]
fn an_interactive_job_runs_before_a_batch_job_that_arrived_first() {
    let log = log();
    let mut jobs = GroupJobs::new(0);
    jobs.add(spec(Class::Batch, 0, Kind::Copy, script("b", &log, &[Kind::Prefill])));
    jobs.add(spec(
        Class::Interactive,
        0,
        Kind::Copy,
        script("i", &log, &[Kind::Prefill]),
    ));
    run_all(&mut jobs);
    assert_eq!(lines(&log), ["i:Copy", "i:Prefill", "b:Copy", "b:Prefill"]);
}

#[test]
fn an_interactive_job_that_arrives_between_quanta_runs_before_the_next_batch_quantum() {
    let log = log();
    let mut jobs = GroupJobs::new(0);
    jobs.add(spec(
        Class::Batch,
        0,
        Kind::Copy,
        script("b", &log, &[Kind::Prefill, Kind::Prefill]),
    ));
    assert!(jobs.run_one(&mut []));
    jobs.add(spec(Class::Interactive, 0, Kind::Read, script("i", &log, &[])));
    run_all(&mut jobs);
    assert_eq!(lines(&log), ["b:Copy", "i:Read", "b:Prefill", "b:Prefill"]);
}

#[test]
fn in_the_batch_class_a_higher_priority_goes_first_and_equal_priorities_go_in_arrival_order() {
    let log = log();
    let mut jobs = GroupJobs::new(0);
    jobs.add(spec(
        Class::Batch,
        0,
        Kind::Copy,
        script("first", &log, &[Kind::Prefill]),
    ));
    jobs.add(spec(
        Class::Batch,
        0,
        Kind::Copy,
        script("second", &log, &[Kind::Prefill]),
    ));
    jobs.add(spec(
        Class::Batch,
        5,
        Kind::Copy,
        script("urgent", &log, &[Kind::Prefill]),
    ));
    run_all(&mut jobs);
    // One batch job runs its quanta in flow, with no interleave (spec 5.4).
    assert_eq!(
        lines(&log),
        [
            "urgent:Copy",
            "urgent:Prefill",
            "first:Copy",
            "first:Prefill",
            "second:Copy",
            "second:Prefill"
        ]
    );
}

#[test]
fn interactive_jobs_of_equal_priority_take_turns_by_owed_time() {
    let log = log();
    let mut jobs = GroupJobs::new(0);
    let more = [Kind::Prefill, Kind::Prefill];
    for name in ["a", "b"] {
        let mut body = script(name, &log, &more);
        body.sleep = Duration::from_millis(20);
        jobs.add(spec(Class::Interactive, 0, Kind::Copy, body));
    }
    run_all(&mut jobs);
    // The first quantum of a lowers its balance, so b is owed more and goes next (spec 5.4).
    let order = lines(&log);
    assert_eq!(order.len(), 6);
    assert_eq!(&order[..3], ["a:Copy", "b:Copy", "a:Prefill"], "{order:?}");
}

#[test]
fn a_failed_quantum_ends_its_job_and_the_other_jobs_go_on() {
    let log = log();
    let mut jobs = GroupJobs::new(0);
    let mut failing = script("bad", &log, &[Kind::Prefill, Kind::Prefill]);
    failing.fail_at = Some(1);
    jobs.add(spec(Class::Batch, 1, Kind::Copy, failing));
    jobs.add(spec(
        Class::Batch,
        0,
        Kind::Copy,
        script("good", &log, &[Kind::Prefill]),
    ));
    run_all(&mut jobs);
    assert_eq!(lines(&log), ["bad:Copy", "bad:Prefill", "good:Copy", "good:Prefill"]);
}

#[test]
fn a_job_that_ends_drops_its_body() {
    let log = log();
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut body = script("a", &log, &[Kind::Prefill]);
    body.dropped = Some(Arc::clone(&dropped));
    let mut jobs = GroupJobs::new(0);
    jobs.add(spec(Class::Batch, 0, Kind::Copy, body));
    run_all(&mut jobs);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(jobs.idle());
}

#[test]
fn a_cancel_drops_every_body_and_counts_them() {
    let log = log();
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut jobs = GroupJobs::new(0);
    for name in ["a", "b"] {
        let mut body = script(name, &log, &[Kind::Prefill]);
        body.dropped = Some(Arc::clone(&dropped));
        jobs.add(spec(Class::Batch, 0, Kind::Copy, body));
    }
    assert!(jobs.run_one(&mut []));
    assert_eq!(jobs.cancel_all(), 2);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert!(jobs.idle());
    assert!(!jobs.run_one(&mut []), "nothing is left to run");
    assert_eq!(lines(&log), ["a:Copy"]);
}

#[test]
fn a_group_with_no_job_is_idle_and_runs_nothing() {
    let mut jobs = GroupJobs::new(0);
    assert!(jobs.idle());
    assert!(!jobs.run_one(&mut []));
}
