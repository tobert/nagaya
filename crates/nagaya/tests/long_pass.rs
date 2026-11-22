//! A long pass with a declared bound (engine-abi D8, spec 7.4). The watchdog arms a deadline for the pass alone: a pass
//! that ends after the timeout and before its deadline is not a stall, and a pass past its deadline stops the process.
//! The timings are short: a timeout of 100 ms and a beat of 10 ms.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::{Event, LongPass, Process, Quantum, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    DeviceId, Engine, Identity, Kind, Limits, OutputsWanted, PassError, PassErrorKind, PassOut, PassStatus, Segment,
    SegmentInputs, State, Tap, Tier,
};
use nagaya_sched::{Class, JobState};

const TIMEOUT: Duration = Duration::from_millis(100);
const BEAT: Duration = Duration::from_millis(10);

/// What the engine of a test declares and does.
#[derive(Clone)]
struct Plan {
    /// `max_long_quantum_ms` of the limits.
    ceiling: Option<u32>,
    /// What `pass_bound_ms` gives.
    bound: Result<Option<u32>, String>,
    /// How long each pass sleeps.
    sleep: Duration,
}

/// The fake engine, with the bound and the time of `plan`. It counts the passes that ran.
struct Long {
    inner: FakeAlways,
    plan: Plan,
    ran: Arc<AtomicUsize>,
}

fn limits_of(plan: &Plan) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.max_quantum_ms = 10;
    limits.max_long_quantum_ms = plan.ceiling.map(|ms| NonZeroU32::new(ms).unwrap());
    // A swap may close the engine, so it shares a group with an engine on demand (spec 3.3a).
    limits.capabilities.push(nagaya_abi::Capability::Close);
    limits
}

impl Engine for Long {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        limits_of(&self.plan)
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.ran.fetch_add(1, Ordering::SeqCst);
        thread::sleep(self.plan.sleep);
        self.inner.pass(segments, taps)
    }

    fn pass_bound_ms(&self, _segments: &[Segment], _taps: &[Tap]) -> Result<Option<NonZeroU32>, String> {
        self.plan
            .bound
            .clone()
            .map(|bound| bound.map(|ms| NonZeroU32::new(ms).unwrap()))
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.inner.slots()
    }
}

struct LongFactory {
    plan: Plan,
    ran: Arc<AtomicUsize>,
}

impl Factory for LongFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits_of(&self.plan))
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Long {
            inner: FakeAlways::new(),
            plan: self.plan.clone(),
            ran: Arc::clone(&self.ran),
        }))
    }
}

fn placement(name: &str, kind: &str, devices: &[DeviceId]) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: devices.to_vec(),
        config: toml::Table::new(),
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        residency: Default::default(),
        pinned: false,
    }
}

fn inventory(ids: &[DeviceId]) -> Inventory {
    Inventory {
        devices: ids
            .iter()
            .map(|id| Device {
                id: *id,
                name: format!("device-{id}"),
                pci_address: format!("0000:0{id}:00.0"),
                memory_bytes: 1 << 20,
                unified: false,
                ignored: false,
            })
            .collect(),
        host_bytes: 1 << 20,
    }
}

fn budgets(ids: &[DeviceId]) -> Vec<BudgetSpec> {
    let mut budgets: Vec<BudgetSpec> = ids
        .iter()
        .map(|id| BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Device(*id))],
            bytes: 1 << 16,
        })
        .collect();
    budgets.push(BudgetSpec {
        tiers: vec![TierSpec::from(Tier::Host)],
        bytes: 1 << 16,
    });
    budgets
}

/// A process with one engine `long` on device 0, in a group of its own. It gives the count of the passes that ran.
fn started(plan: Plan) -> (Process, Arc<AtomicUsize>) {
    let ran = Arc::new(AtomicUsize::new(0));
    let mut registry = Registry::new();
    registry.register(
        "long",
        Box::new(LongFactory {
            plan,
            ran: Arc::clone(&ran),
        }),
    );
    let profile = PlacementProfile {
        engines: vec![placement("long", "long", &[0])],
        budgets: budgets(&[0]),
        store: None,
    };
    let mut process = Start::new(profile, registry, inventory(&[0]))
        .heartbeat(BEAT)
        .timeout(TIMEOUT)
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    (process, ran)
}

fn segment() -> Segment {
    Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(vec![5, 6, 7]),
        positions: vec![0, 1, 2],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![2] },
        control: Vec::new(),
        layer_loop: None,
        cycles: None,
    }
}

/// Run one pass of `segments` on the engine of group 0, as a quantum. The receiver gets the result of the pass.
fn pass(process: &Process, segments: Vec<Segment>) -> Receiver<Result<PassOut, PassError>> {
    let (result, results) = mpsc::channel();
    let quantum = Quantum::new(move |engines| {
        let _ = result.send(engines[0].pass(&segments, &[]));
    });
    process.submit(0, quantum).unwrap();
    results
}

fn stalled(devices: Vec<DeviceId>) -> Event {
    Event::Stalled { group: 0, devices }
}

/// Falsified by a watchdog that keeps the timeout during a long pass: the process stalls at 100 ms.
#[test]
fn a_long_pass_that_ends_after_the_timeout_and_before_its_deadline_is_not_a_stall() {
    // The deadline is the start plus 400 + 10 ms. The pass ends at 250 ms.
    let (process, _) = started(Plan {
        ceiling: Some(1_000),
        bound: Ok(Some(400)),
        sleep: Duration::from_millis(250),
    });
    let results = pass(&process, vec![segment()]);
    let result = results.recv_timeout(Duration::from_secs(2)).expect("the pass ends");
    assert!(result.is_ok(), "{result:?}");
    // The group beats again after the pass, so no stall follows it.
    assert_eq!(process.recv(Duration::from_millis(400)), None);
}

/// Falsified by a deadline that grows with each beat, or by a watchdog that never reads the deadline.
#[test]
fn a_long_pass_past_its_deadline_stops_the_process_and_its_result_is_dropped() {
    // The deadline is the start plus the timeout, 100 ms, because 20 + 10 ms is less. The pass ends at 400 ms.
    let (process, _) = started(Plan {
        ceiling: Some(1_000),
        bound: Ok(Some(20)),
        sleep: Duration::from_millis(400),
    });
    let results = pass(&process, vec![segment()]);
    // Past its bound and before its deadline, the pass shows as overdue.
    thread::sleep(Duration::from_millis(60));
    let shown = process.ports().long_passes().on_device(0).expect("the pass runs");
    assert!(shown.overdue && shown.elapsed_ms > 20, "{shown:?}");
    let event = process
        .recv(Duration::from_secs(2))
        .expect("the watchdog stops the process");
    assert_eq!(event, stalled(vec![0]));
    assert_eq!(event.exit_code(), Some(nagaya::exit::ExitCode::Crash));
    // The pass past its deadline leaves the table, so the route does not show it as running.
    assert_eq!(process.ports().long_passes().on_device(0), None);
    let result = results.recv_timeout(Duration::from_secs(2)).expect("the pass ends");
    let error = result.expect_err("a result past the deadline is not published");
    assert_eq!(error.kind, PassErrorKind::Ran(PassStatus::Timeout));
    assert!(
        error.message.contains("engine `long`") && error.message.contains("send the request again"),
        "{}",
        error.message
    );
    // The engine runs no more passes, and the refusal says why and what to do.
    let error = pass(&process, vec![segment()])
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .expect_err("the engine runs no more passes");
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(
        error.message.contains("engine `long`") && error.message.contains("after the restart"),
        "{}",
        error.message
    );
}

/// Falsified by a group that arms a deadline for a pass with no bound.
#[test]
fn a_pass_with_no_bound_still_stalls_at_the_timeout() {
    let (process, _) = started(Plan {
        ceiling: Some(1_000),
        bound: Ok(None),
        sleep: Duration::from_millis(400),
    });
    let _results = pass(&process, vec![segment()]);
    let event = process
        .recv(Duration::from_secs(2))
        .expect("the watchdog reports the stall");
    assert_eq!(event, stalled(vec![0]));
}

/// A bound with no ceiling, a bound past the ceiling, a bound of a joined pass, and a refusal of the engine each
/// refuse the pass before it runs. Falsified by a group that does not check the bound.
#[test]
fn a_bound_that_breaks_a_rule_refuses_the_pass_before_it_runs() {
    let cases = [
        (None, Ok(Some(400)), vec![segment()], "declares no max_long_quantum_ms"),
        (Some(300), Ok(Some(400)), vec![segment()], "at most max_long_quantum_ms"),
        (Some(1_000), Ok(Some(5)), vec![segment()], "more than max_quantum_ms"),
        (Some(1_000), Ok(Some(400)), vec![segment(), segment()], "one segment"),
        (
            Some(1_000),
            Err("the tower is not ready".to_owned()),
            vec![segment()],
            "the tower is not ready",
        ),
    ];
    for (ceiling, bound, segments, words) in cases {
        let (process, ran) = started(Plan {
            ceiling,
            bound,
            sleep: Duration::ZERO,
        });
        let result = pass(&process, segments)
            .recv_timeout(Duration::from_secs(2))
            .expect("the pass ends");
        let error = result.expect_err("the pass is refused");
        assert_eq!(error.kind, PassErrorKind::Refused, "{words}");
        assert!(error.message.contains(words), "`{}` must name {words}", error.message);
        assert_eq!(ran.load(Ordering::SeqCst), 0, "nothing ran: {words}");
    }
}

/// The table of long passes shows the pass while it runs, and not after it ends.
#[test]
fn the_long_passes_show_a_pass_while_it_runs() {
    let (process, _) = started(Plan {
        ceiling: Some(1_000),
        bound: Ok(Some(400)),
        sleep: Duration::from_millis(200),
    });
    let passes = process.ports().long_passes();
    assert_eq!(passes.on_device(0), None);
    let results = pass(&process, vec![segment()]);
    thread::sleep(Duration::from_millis(100));
    let shown = passes.on_device(0).expect("the pass runs");
    assert_eq!((shown.group, shown.devices.clone(), shown.bound_ms), (0, vec![0], 400));
    assert!(
        shown.elapsed_ms >= 50 && shown.elapsed_ms < 400 && !shown.overdue,
        "{shown:?}"
    );
    results.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
    assert_eq!(passes.on_device(0), None::<LongPass>);
}

/// A body that runs one pass on engine 0 and writes its name to a log.
struct Named {
    name: &'static str,
    log: Arc<Mutex<Vec<&'static str>>>,
}

impl JobBody for Named {
    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        self.log.lock().unwrap().push(self.name);
        match engines[0].pass(&[segment()], &[]) {
            Ok(_) => Ran::Done,
            Err(_) => Ran::Failed,
        }
    }

    fn ended(&mut self, _state: JobState) {}

    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}", state.as_str());
    }
}

fn job(class: Class, name: &'static str, log: &Arc<Mutex<Vec<&'static str>>>) -> JobSpec {
    JobSpec {
        class,
        priority: 0,
        deadline: None,
        first: Kind::Prefill,
        engine: 0,
        body: Box::new(Named {
            name,
            log: Arc::clone(log),
        }),
    }
}

/// A long pass keeps the order of classes: each pass of the engine is long, and a ready interactive job runs before
/// a batch job that came first. Falsified by a scheduler that runs the classes in the order of arrival.
#[test]
fn a_long_pass_does_not_start_before_ready_work_of_a_higher_class() {
    let (process, _) = started(Plan {
        ceiling: Some(1_000),
        bound: Ok(Some(400)),
        sleep: Duration::from_millis(20),
    });
    let log = Arc::new(Mutex::new(Vec::new()));
    // A quantum holds the group, so both jobs are ready when it ends.
    let (go, gate) = mpsc::channel::<()>();
    process
        .submit(
            0,
            Quantum::new(move |_| {
                let _ = gate.recv_timeout(Duration::from_secs(2));
            }),
        )
        .unwrap();
    process.submit_job(0, job(Class::Batch, "batch", &log)).unwrap();
    process
        .submit_job(0, job(Class::Interactive, "interactive", &log))
        .unwrap();
    thread::sleep(Duration::from_millis(30));
    go.send(()).unwrap();
    let ends = std::time::Instant::now() + Duration::from_secs(2);
    while log.lock().unwrap().len() < 2 && std::time::Instant::now() < ends {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(*log.lock().unwrap(), ["interactive", "batch"]);
}

/// An engine of two devices that never loads. It puts devices 0 and 1 in one group, so `long` runs on the device
/// thread of device 0, as an engine of one card does in a group of two cards (spec 3.2).
struct WideFactory;

impl Factory for WideFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        let mut limits = FakeAlways::new().limits();
        let second: Vec<_> = limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.tier == Tier::Device(0))
            .map(|term| {
                let mut term = term.clone();
                term.tier = Tier::Device(1);
                term.name = format!("{}-1", term.name);
                term
            })
            .collect();
        limits.memory.mandatory.extend(second);
        limits.devices = vec![0, 1];
        limits.capabilities.push(nagaya_abi::Capability::Close);
        Ok(limits)
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("the wide engine is on demand, and no test loads it")
    }
}

/// Falsified by a device thread that keeps the timeout during a long pass: the group stalls at 100 ms.
#[test]
fn a_long_pass_on_a_device_thread_is_not_a_stall_and_its_result_reaches_the_job() {
    let ran = Arc::new(AtomicUsize::new(0));
    let mut registry = Registry::new();
    registry.register(
        "long",
        Box::new(LongFactory {
            plan: Plan {
                ceiling: Some(1_000),
                bound: Ok(Some(400)),
                sleep: Duration::from_millis(250),
            },
            ran: Arc::clone(&ran),
        }),
    );
    registry.register("wide", Box::new(WideFactory));
    let mut wide = placement("wide", "wide", &[0, 1]);
    wide.residency = nagaya::profile::Residency::OnDemand;
    let profile = PlacementProfile {
        engines: vec![placement("long", "long", &[0]), wide],
        budgets: budgets(&[0, 1]),
        store: None,
    };
    let mut process = Start::new(profile, registry, inventory(&[0, 1]))
        .heartbeat(BEAT)
        .timeout(TIMEOUT)
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (ended, ends) = mpsc::channel();
    let mut spec = job(Class::Interactive, "long", &log);
    spec.body = Box::new(Ending {
        inner: Named {
            name: "long",
            log: Arc::clone(&log),
        },
        ended,
    });
    process.submit_job(0, spec).unwrap();
    thread::sleep(Duration::from_millis(120));
    let shown = process.ports().long_passes().on_device(0).expect("the pass runs");
    assert_eq!((shown.group, shown.devices, shown.bound_ms), (0, vec![0], 400));
    assert_eq!(process.ports().long_passes().on_device(1), None);
    let state = ends.recv_timeout(Duration::from_secs(2)).expect("the job ends");
    assert_eq!(state, JobState::Done);
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert_eq!(process.recv(Duration::from_millis(400)), None);
}

/// A body that sends the end state of its job.
struct Ending {
    inner: Named,
    ended: mpsc::Sender<JobState>,
}

impl JobBody for Ending {
    fn run(&mut self, kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        self.inner.run(kind, engines)
    }

    fn ended(&mut self, state: JobState) {
        let _ = self.ended.send(state);
    }

    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}", state.as_str());
    }
}

/// A council with the ports of `process`, and with a main loop that answers the active profile `profile`.
fn council(process: &Process, profile: PlacementProfile) -> Arc<nagaya::http::Council> {
    let (calls, asked) = mpsc::channel();
    thread::spawn(move || {
        while let Ok(call) = asked.recv() {
            if let nagaya::reprofile::ProfileCall::Get(reply) = call {
                let _ = reply.send(profile.clone());
            }
        }
    });
    let council = nagaya::http::Council::new(Vec::new(), None).unwrap();
    Arc::new(council.with_loads(process.ports()).with_profile(calls))
}

fn get(council: &Arc<nagaya::http::Council>, path: &str) -> (axum::http::StatusCode, serde_json::Value) {
    use tower::ServiceExt;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = axum::http::Request::builder()
            .method("GET")
            .uri(path)
            .body(axum::body::Body::empty())
            .unwrap();
        let response = nagaya::http::router(Arc::clone(council))
            .oneshot(request)
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    })
}

/// The route `busy` of a device shows its long pass while it runs, and `null` before and after. A device that no
/// engine names is a 404 that names `id`. Falsified by a route that reads another device.
#[test]
fn the_route_busy_shows_the_long_pass_of_a_device() {
    use axum::http::StatusCode;
    let (process, _) = started(Plan {
        ceiling: Some(1_000),
        bound: Ok(Some(400)),
        sleep: Duration::from_millis(250),
    });
    let profile = PlacementProfile {
        engines: vec![placement("long", "long", &[0])],
        budgets: budgets(&[0]),
        store: None,
    };
    let council = council(&process, profile);
    let path = "/nagaya/v1/devices/0/busy";
    assert_eq!(
        get(&council, path),
        (StatusCode::OK, serde_json::json!({"device": 0, "long_pass": null}))
    );
    let results = pass(&process, vec![segment()]);
    thread::sleep(Duration::from_millis(100));
    let (status, body) = get(&council, path);
    assert_eq!(status, StatusCode::OK, "{body}");
    let shown = &body["long_pass"];
    assert_eq!(
        (
            &shown["group"],
            &shown["devices"],
            &shown["bound_ms"],
            &shown["overdue"]
        ),
        (
            &serde_json::json!(0),
            &serde_json::json!([0]),
            &serde_json::json!(400),
            &serde_json::json!(false)
        ),
        "{body}"
    );
    let elapsed = shown["elapsed_ms"].as_u64().unwrap();
    assert!((50..400).contains(&elapsed), "{body}");
    results.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
    assert_eq!(get(&council, path).1["long_pass"], serde_json::Value::Null);
    for path in ["/nagaya/v1/devices/7/busy", "/nagaya/v1/devices/x/busy"] {
        let (status, body) = get(&council, path);
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}: {body}");
        assert_eq!(body["error"]["param"], "id", "{path}");
    }
}
