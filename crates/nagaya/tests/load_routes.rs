//! The load routes (spec 3.3b): `POST /nagaya/v1/engines/{name}/load` and `/unload` start a swap on the group
//! thread of the engine. A request never starts a load (spec 3.3).

use std::sync::atomic::{AtomicBool, Ordering};
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

/// The factory of the counted engines. Its open check refuses the engine `voice` while `refuse` is true.
struct CountedFactory(Log, Arc<AtomicBool>);

impl Factory for CountedFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(counted_limits())
    }

    fn open_problem(&self, config: &toml::Table, _devices: &[GroupDevice], freed_bytes: u64) -> Option<String> {
        (config["name"].as_str() == Some("voice") && self.1.load(Ordering::SeqCst)).then(|| {
            format!("the shards of the stub take 9 bytes in memory, but the unload frees only {freed_bytes} bytes")
        })
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
        lab: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

/// A resident engine `chat` and an on-demand engine `voice` on one device. With 1000 device bytes they do not fit
/// together, so a load of one must name the other in `unload`.
fn start(directory: &std::path::Path) -> (Process, Log) {
    start_pinned(directory, false)
}

/// As [`start`], with the open check of `voice` that `refuse` switches (spec 3.3b, "The memory check of an open").
fn start_checked(directory: &std::path::Path, refuse: &Arc<AtomicBool>) -> (Process, Log) {
    start_with(directory, false, refuse)
}

/// As [`start`], with `chat` pinned when `pinned` is true (spec 3.3a, "Explicit layout").
fn start_pinned(directory: &std::path::Path, pinned: bool) -> (Process, Log) {
    start_with(directory, pinned, &Arc::default())
}

fn start_with(directory: &std::path::Path, pinned: bool, refuse: &Arc<AtomicBool>) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register(
        "counted",
        Box::new(CountedFactory(Arc::clone(&log), Arc::clone(refuse))),
    );
    let mut chat = engine("chat", Residency::Resident);
    chat.pinned = pinned;
    let profile = PlacementProfile {
        engines: vec![chat, engine("voice", Residency::OnDemand)],
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
    send_body(council, method, path, Body::empty())
}

fn post_json(council: &Arc<Council>, path: &str, body: &Value) -> (StatusCode, Value) {
    send_body(council, "POST", path, Body::from(body.to_string()))
}

fn send_body(council: &Arc<Council>, method: &str, path: &str, body: Body) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path).body(body).unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    })
}

#[test]
fn a_load_with_an_unload_list_is_a_202_that_names_the_engines_of_the_swap() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
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

/// A load that does not fit is a 409 that names the shortfall and the engine in the way, and nothing changes (spec
/// 3.3a, "Explicit layout").
#[test]
fn a_load_that_does_not_fit_is_a_409_that_names_the_engine_in_the_way() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/load");
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["type"], json!("invalid_request"), "{body}");
    assert_eq!(body["error"]["param"], json!("name"), "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("short by 200 bytes"), "{body}");
    assert!(message.contains("`chat`") && message.contains("`unload`"), "{body}");
    assert_eq!(log_of(&log), ["open:chat"]);
}

/// A load that the open check of the engine refuses is a 409 that names the shortfall and the engines in the way. No
/// swap starts, no state changes, and the process lives (spec 3.3b, "The memory check of an open").
#[test]
fn a_load_that_the_open_check_refuses_is_a_409_and_the_process_lives() {
    let directory = tempfile::tempdir().unwrap();
    let refuse = Arc::new(AtomicBool::new(true));
    let (process, log) = start_checked(directory.path(), &refuse);
    let council = council(&process);
    // An empty list frees nothing and leaves `chat` in the way. The ledger would refuse this load too, but the check
    // of the engine comes first, so the message is the shortfall of the engine.
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/load");
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["type"], json!("invalid_request"), "{body}");
    assert_eq!(body["error"]["param"], json!("name"), "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("nothing changed:"), "{body}");
    assert!(message.contains("frees only 0 bytes"), "{body}");
    assert!(message.contains("`chat`") && message.contains("`unload`"), "{body}");
    // A list that names `chat` frees its 600 bytes, and the check still refuses.
    let (status, body) = post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("frees only 600 bytes"), "{body}");
    assert_eq!(log_of(&log), ["open:chat"], "no engine closed or opened");
    let ports = process.ports();
    assert_eq!(ports.engine_state("voice").unwrap().as_str(), "unloaded");
    assert_eq!(ports.engine_state("chat").unwrap().as_str(), "ready");
    // The process lives: the group answers the next request.
    let (status, body) = post(&council, "/nagaya/v1/engines/chat/load");
    assert_eq!(status, StatusCode::OK, "{body}");
    // The same load goes on when the check passes.
    refuse.store(false, Ordering::SeqCst);
    let (status, body) = post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice"]);
}

/// A load whose open check passes goes on as before.
#[test]
fn a_load_that_the_open_check_passes_is_a_202() {
    let directory = tempfile::tempdir().unwrap();
    let refuse = Arc::new(AtomicBool::new(false));
    let (process, log) = start_checked(directory.path(), &refuse);
    let council = council(&process);
    let (status, body) = post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
    assert_eq!(log_of(&log), ["open:chat", "close:chat", "open:voice"]);
}

/// A body that does not decode, or an `unload` that names the engine itself, is a 400 that names `unload`.
#[test]
fn a_load_with_a_bad_unload_list_is_a_400_that_names_unload() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    for body in [
        json!({"unload": "chat"}),
        json!({"drop": ["chat"]}),
        json!({"unload": ["voice"]}),
    ] {
        let (status, answer) = post_json(&council, "/nagaya/v1/engines/voice/load", &body);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} gave {answer}");
        assert_eq!(answer["error"]["param"], json!("unload"), "{answer}");
    }
    assert_eq!(log_of(&log), ["open:chat"]);
}

/// A name that comes twice in `unload` is a 400 that names `unload`, and nothing changes. A repeat once crashed the
/// ledger thread (spec 3.3b).
#[test]
fn a_repeated_name_in_unload_is_a_400_and_the_process_lives() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    let (status, answer) = post_json(
        &council,
        "/nagaya/v1/engines/voice/load",
        &json!({"unload": ["chat", "chat"]}),
    );
    assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    assert_eq!(answer["error"]["param"], json!("unload"), "{answer}");
    assert_eq!(log_of(&log), ["open:chat"]);
    let (status, body) = post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], vec!["voice".to_owned()]));
}

/// A load of a loaded engine checks its `unload` list before the 200 (spec 3.3b).
#[test]
fn a_load_of_a_loaded_engine_checks_its_unload_list() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start(directory.path());
    let council = council(&process);
    for body in [json!({"unload": ["chat"]}), json!({"unload": ["voice", "voice"]})] {
        let (status, answer) = post_json(&council, "/nagaya/v1/engines/chat/load", &body);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} gave {answer}");
        assert_eq!(answer["error"]["param"], json!("unload"), "{answer}");
    }
    let (status, body) = post_json(&council, "/nagaya/v1/engines/chat/load", &json!({}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(log_of(&log), ["open:chat"]);
}

#[test]
fn an_unload_is_a_202_and_no_engine_loads() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
    swapped(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/unload");
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body, json!({"engine": "voice", "unload": ["voice"], "load": []}));
    assert_eq!(swapped(&process), (vec!["voice".to_owned()], Vec::new()));
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
fn an_unload_of_a_pinned_engine_is_a_409_that_names_name() {
    let directory = tempfile::tempdir().unwrap();
    let (process, log) = start_pinned(directory.path(), true);
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/chat/unload");
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["type"], json!("invalid_request"), "{body}");
    assert_eq!(body["error"]["param"], json!("name"), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("pinned"), "{body}");
    // A load that needs the memory of the pin is a 409 that names the pin, and an `unload` of it is refused too.
    let (status, body) = post(&council, "/nagaya/v1/engines/voice/load");
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("pinned engines `chat`"),
        "{body}"
    );
    let (status, body) = post_json(&council, "/nagaya/v1/engines/voice/load", &json!({"unload": ["chat"]}));
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(log_of(&log), ["open:chat"]);
}

/// A resident engine that is not pinned unloads by the route, so a client can make room (spec 3.3b).
#[test]
fn an_unload_of_a_resident_engine_that_is_not_pinned_is_a_202() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let council = council(&process);
    let (status, body) = post(&council, "/nagaya/v1/engines/chat/unload");
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body, json!({"engine": "chat", "unload": ["chat"], "load": []}));
    assert_eq!(swapped(&process), (vec!["chat".to_owned()], Vec::new()));
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
