//! A request reaches an engine through the scheduler on the group thread (spec 3.2, 5.3, 10.3).

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::{OpenedEngine, Process, Quantum, Start, SubmitError};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Kind, Limits, SlotLimit, Tier};
use nagaya_sched::Class;

fn two_slots() -> FakeAlways {
    FakeAlways::with_slots(vec![
        SlotLimit {
            bytes: 64,
            count: 1,
            tokens: 4096,
        },
        SlotLimit {
            bytes: 256,
            count: 1,
            tokens: 4096,
        },
    ])
}

struct Simple;

impl Factory for Simple {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(two_slots().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(two_slots()))
    }
}

fn profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: "one".to_owned(),
            kind: "simple".to_owned(),
            devices: vec![0],
            config: toml::Table::new(),
            family: toml::Table::new(),
            share: 1,
            pin_identity: false,
            residency: Default::default(),
            pinned: false,
        }],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 4096,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 4096,
            },
        ],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn start(directory: &std::path::Path) -> (Process, Vec<OpenedEngine>) {
    let mut registry = Registry::new();
    registry.register("simple", Box::new(Simple));
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1024,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1024,
    };
    let mut process = Start::new(profile(), registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, opened)
}

type Log = Arc<Mutex<Vec<String>>>;

struct Record {
    name: &'static str,
    log: Log,
    more: Vec<Kind>,
    done: Option<Sender<std::thread::ThreadId>>,
}

impl JobBody for Record {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        assert_eq!(engines.len(), 1, "the group holds the engine of the profile");
        self.log.lock().unwrap().push(format!("{}:{kind:?}", self.name));
        if let Some(done) = self.done.take() {
            let _ = done.send(std::thread::current().id());
        }
        if self.more.is_empty() {
            Ran::Done
        } else {
            Ran::More(std::mem::take(&mut self.more))
        }
    }
}

fn job(class: Class, name: &'static str, log: &Log, more: &[Kind]) -> JobSpec {
    JobSpec {
        class,
        priority: 0,
        deadline: None,
        first: Kind::Copy,
        engine: 0,
        body: Box::new(Record {
            name,
            log: Arc::clone(log),
            more: more.to_vec(),
            done: None,
        }),
    }
}

/// Hold the group thread until the test lets go, so that later requests queue behind the gate.
fn gate(process: &Process) -> Sender<()> {
    let (open, closed) = mpsc::channel::<()>();
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    process
        .submit(
            0,
            Quantum::new(move |_| {
                entered_tx.send(()).unwrap();
                let _ = closed.recv_timeout(Duration::from_secs(10));
            }),
        )
        .unwrap();
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the gate opened");
    open
}

fn wait_for(log: &Log, count: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while log.lock().unwrap().len() < count {
        assert!(std::time::Instant::now() < deadline, "log: {:?}", log.lock().unwrap());
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_job_runs_on_the_thread_that_opened_the_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (process, opened) = start(directory.path());
    let log: Log = Arc::default();
    let (done_tx, done_rx) = mpsc::channel();
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine: 0,
                body: Box::new(Record {
                    name: "a",
                    log: Arc::clone(&log),
                    more: Vec::new(),
                    done: Some(done_tx),
                }),
            },
        )
        .unwrap();
    let thread = done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(thread, opened[0].thread);
    assert_ne!(thread, std::thread::current().id());
}

#[test]
fn an_interactive_job_that_queued_behind_a_batch_job_runs_first() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _) = start(directory.path());
    let log: Log = Arc::default();
    let open = gate(&process);
    process
        .submit_job(0, job(Class::Batch, "b", &log, &[Kind::Prefill]))
        .unwrap();
    process
        .submit_job(0, job(Class::Interactive, "i", &log, &[Kind::Prefill]))
        .unwrap();
    open.send(()).unwrap();
    wait_for(&log, 4);
    assert_eq!(*log.lock().unwrap(), ["i:Copy", "i:Prefill", "b:Copy", "b:Prefill"]);
}

#[test]
fn a_stop_cancels_the_jobs_that_have_not_started() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, _) = start(directory.path());
    let log: Log = Arc::default();
    let (result_tx, result_rx) = mpsc::channel::<()>();
    let open = gate(&process);
    struct Holds(#[allow(dead_code)] Sender<()>);
    impl JobBody for Holds {
        fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
            panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
        }

        fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
            panic!("a cancelled job does not run");
        }
    }
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Batch,
                priority: 0,
                deadline: None,
                first: Kind::Copy,
                engine: 0,
                body: Box::new(Holds(result_tx)),
            },
        )
        .unwrap();
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        open.send(()).unwrap();
    });
    let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
    release.join().unwrap();
    assert_eq!(report.parked.cancelled, 1, "{report:?}");
    // The body dropped its sender, so the caller sees the end of the job with no result.
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_secs(5)),
        Err(mpsc::RecvTimeoutError::Disconnected)
    ));
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn a_stopping_process_refuses_a_job() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, _) = start(directory.path());
    let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
    assert!(report.complete, "{report:?}");
    let log: Log = Arc::default();
    assert_eq!(
        process.submit_job(0, job(Class::Batch, "a", &log, &[])),
        Err(SubmitError::Stopping)
    );
}

#[test]
fn a_job_for_a_group_that_does_not_exist_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _) = start(directory.path());
    let log: Log = Arc::default();
    assert_eq!(
        process.submit_job(9, job(Class::Batch, "a", &log, &[])),
        Err(SubmitError::UnknownGroup)
    );
}

#[test]
fn a_stop_cancels_a_job_that_waits_for_its_next_quantum() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, _) = start(directory.path());
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (open, closed) = mpsc::channel::<()>();
    let (result_tx, result_rx) = mpsc::channel::<()>();
    struct TwoQuanta {
        entered: Sender<()>,
        closed: Receiver<()>,
        result: Sender<()>,
        ran: usize,
    }
    impl JobBody for TwoQuanta {
        fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
            panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
        }

        fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
            self.ran += 1;
            assert_eq!(self.ran, 1, "the stop cancels the second quantum");
            self.entered.send(()).unwrap();
            let _ = self.closed.recv_timeout(Duration::from_secs(10));
            let _ = &self.result;
            Ran::More(vec![Kind::Prefill])
        }
    }
    process
        .submit_job(
            0,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Copy,
                engine: 0,
                body: Box::new(TwoQuanta {
                    entered: entered_tx,
                    closed,
                    result: result_tx,
                    ran: 0,
                }),
            },
        )
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        open.send(()).unwrap();
    });
    let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
    release.join().unwrap();
    assert_eq!(report.parked.cancelled, 1, "{report:?}");
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_secs(5)),
        Err(mpsc::RecvTimeoutError::Disconnected)
    ));
}
