//! The load routes (spec 3.3b): `POST /nagaya/v1/engines/{name}/load` and `/unload` start a swap on the group
//! thread of the engine. A request never starts a load (spec 3.3).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::http::{Council, router};
use serde_json::{Value, json};
use tower::ServiceExt;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Event, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal, Segment,
    Slots, Tap, Tier,
};

type Log = Arc<Mutex<Vec<String>>>;

/// A fake engine that writes its open and its close to a log, and declares `close`.
struct Counted {
    inner: FakeAlways,
    name: String,
    limits: Limits,
    log: Log,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.log.lock().unwrap().push(format!("close:{}", self.name));
    }
}

impl Engine for Counted {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.inner.seal()
    }
}

fn counted_limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes: 600,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    limits
}

struct CountedFactory(Log);

impl Factory for CountedFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(counted_limits())
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let name = config["name"].as_str().unwrap().to_owned();
        self.0.lock().unwrap().push(format!("open:{name}"));
        Ok(Box::new(Counted {
            inner: FakeAlways::new(),
            name,
            limits: counted_limits(),
            log: Arc::clone(&self.0),
        }))
    }
}

fn engine(name: &str, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("name".to_owned(), toml::Value::String(name.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: "counted".to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
    }
}

/// A resident engine `chat` and an on-demand engine `voice` on one device. With 1000 device bytes they do not fit
/// together, so a load of one unloads the other.
fn start(directory: &std::path::Path) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("counted", Box::new(CountedFactory(Arc::clone(&log))));
    let profile = PlacementProfile {
        engines: vec![
            engine("chat", Residency::Resident),
            engine("voice", Residency::OnDemand),
        ],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1000,
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
    };
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 4096,
            unified: false,
            ignored: false,
        }],
        host_bytes: 4096,
    };
    let mut process = Start::new(profile, registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, log)
}

fn log_of(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// The next `Swapped` event of the process.
fn swapped(process: &Process) -> (Vec<String>, Vec<String>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Some(Event::Swapped { unloaded, loaded, .. }) = process.recv(Duration::from_millis(50)) {
            return (unloaded, loaded.into_iter().map(|engine| engine.name).collect());
        }
    }
    panic!("no swap came in 5 s");
}

fn council(process: &Process) -> Arc<Council> {
    Arc::new(Council::new(Vec::new(), None).unwrap().with_loads(process.ports()))
}

fn post(council: &Arc<Council>, path: &str) -> (StatusCode, Value) {
    send(council, "POST", path)
}

fn send(council: &Arc<Council>, method: &str, path: &str) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path).body(Body::empty()).unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    })
}

#[test]
fn a_load_is_a_202_that_names_the_engines_of_the_swap() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/load");
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body, json!({"engine": "voice", "unload": ["chat"], "load": ["voice"]}));
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice"]);
}

#[test]
fn a_load_of_a_ready_engine_is_a_200_with_its_status() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/chat/load");
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"engine": "chat", "status": "ready"}));
}

#[test]
fn an_unload_is_a_202_and_the_resident_engine_loads_again() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    post(&council, "/nagaya/v1/engines/voice/load");
    swapped(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/unload");
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body, json!({"engine": "voice", "unload": ["voice"], "load": ["chat"]}));
    assert_eq!(swapped(&process), (vec!["voice".to_owned()], vec!["chat".to_owned()]));
}

#[test]
fn an_unload_of_an_unloaded_engine_is_a_200() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/unload");
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"engine": "voice", "status": "unloaded"}));
}

#[test]
fn an_unknown_engine_is_a_404_that_names_name() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    for path in ["/nagaya/v1/engines/nobody/load", "/nagaya/v1/engines/nobody/unload"] {
        let (status, body) = post(&council, path);
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(body["error"]["type"], json!("not_found"), "{body}");
        assert_eq!(body["error"]["param"], json!("name"), "{body}");
        assert!(body["request_id"].is_string(), "{body}");
    }
}

#[test]
fn an_unload_of_a_resident_engine_is_a_409_that_names_name() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/chat/unload");
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["type"], json!("invalid_request"), "{body}");
    assert_eq!(body["error"]["param"], json!("name"), "{body}");
    assert_eq!(log_of(&log), ["open:chat"]);
}

#[test]
fn a_council_with_no_process_has_no_load_routes() {
    let council = Arc::new(Council::new(Vec::new(), None).unwrap());
    let (status, body) = post(&council, "/nagaya/v1/engines/chat/load");
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["param"], Value::Null, "{body}");
}

#[test]
fn a_get_of_a_load_route_is_a_404() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    let (status, _) = send(&council, "GET", "/nagaya/v1/engines/voice/load");
    assert_eq!(status, StatusCode::NOT_FOUND);
}
