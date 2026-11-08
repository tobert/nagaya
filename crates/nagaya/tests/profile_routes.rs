//! The profile routes (spec 3.3, "The profile routes"): `GET` and `PUT /nagaya/v1/profile`, through the main loop.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Event, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::reprofile::Changer;
use nagaya::swap::Swap;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal, Segment,
    Slots, Tap, Tier,
};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};
use tower::ServiceExt;

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
        lab: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

/// A resident engine `chat` and an on-demand engine `voice` on one device of `device_bytes`. With 1000 bytes they do not
/// fit together; with 2000 they do.
fn profile(device_bytes: u64) -> PlacementProfile {
    PlacementProfile {
        engines: vec![
            engine("chat", Residency::Resident),
            engine("voice", Residency::OnDemand),
        ],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: device_bytes,
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

fn start(directory: &std::path::Path) -> (Process, Log) {
    let log: Log = Arc::default();
    let mut registry = Registry::new();
    registry.register("counted", Box::new(CountedFactory(Arc::clone(&log))));
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
    let mut process = Start::new(profile(1000), registry, inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, log)
}

/// A process whose main loop ticks on its own thread, and a council with the profile routes. The tick thread ends when
/// the guard drops.
struct Main {
    council: Arc<Council>,
    stop: Arc<AtomicBool>,
    restarts: Arc<AtomicBool>,
    events: std::sync::mpsc::Receiver<(Vec<String>, Vec<String>)>,
    directory: tempfile::TempDir,
}

impl Main {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let (process, _log) = start(directory.path());
        profile(1000).write(directory.path()).unwrap();
        let mut swap = Swap::open(directory.path(), "host", profile(1000)).unwrap();
        let (mut changer, calls) = Changer::new();
        let council = Arc::new(Council::new(Vec::new(), None).unwrap().with_profile(calls));
        let stop = Arc::new(AtomicBool::new(false));
        let restarts = Arc::new(AtomicBool::new(false));
        let (events_tx, events) = std::sync::mpsc::channel();
        let (stop_tick, restarts_tick) = (Arc::clone(&stop), Arc::clone(&restarts));
        std::thread::spawn(move || {
            while !stop_tick.load(Ordering::SeqCst) {
                if changer.tick(&mut swap, &process).unwrap() {
                    restarts_tick.store(true, Ordering::SeqCst);
                }
                if let Some(Event::Swapped { unloaded, loaded, .. }) = process.recv(Duration::from_millis(5)) {
                    let _ = events_tx.send((unloaded, loaded.into_iter().map(|engine| engine.name).collect()));
                }
            }
        });
        Self {
            council,
            stop,
            restarts,
            events,
            directory,
        }
    }

    fn send(&self, method: &str, body: Option<Value>) -> (StatusCode, Value) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let mut request = Request::builder().method(method).uri("/nagaya/v1/profile");
            if body.is_some() {
                request = request.header("content-type", "application/json");
            }
            let body = body.map_or_else(Body::empty, |value| Body::from(value.to_string()));
            let response = router(Arc::clone(&self.council))
                .oneshot(request.body(body).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            (status, serde_json::from_slice(&bytes).unwrap())
        })
    }

    fn swapped(&self) -> (Vec<String>, Vec<String>) {
        self.events.recv_timeout(Duration::from_secs(5)).expect("a swap in 5 s")
    }
}

impl Drop for Main {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

fn as_json(profile: &PlacementProfile) -> Value {
    serde_json::to_value(profile).unwrap()
}

#[test]
fn a_get_gives_the_active_profile() {
    let main = Main::new();
    let (status, body) = main.send("GET", None);
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, as_json(&profile(1000)));
}

#[test]
fn an_equal_profile_is_a_200_that_keeps_each_engine() {
    let main = Main::new();
    let (status, body) = main.send("PUT", Some(as_json(&profile(1000))));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body,
        json!({"actions": [{"engine": "chat", "action": "keep"}, {"engine": "voice", "action": "keep"}],
               "restart": false})
    );
}

#[test]
fn a_change_in_place_is_a_202_and_runs() {
    let main = Main::new();
    let mut new = profile(2000);
    new.engines[1].residency = Residency::Resident;
    let (status, body) = main.send("PUT", Some(as_json(&new)));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(
        body,
        json!({"actions": [{"engine": "chat", "action": "keep"}, {"engine": "voice", "action": "residency"}],
               "restart": false})
    );
    assert_eq!(main.swapped(), (Vec::new(), vec!["voice".to_owned()]));
    assert_eq!(main.send("GET", None).1, as_json(&new));
    assert_eq!(PlacementProfile::read_active(main.directory.path()).unwrap(), new);
}

#[test]
fn a_new_engine_is_a_202_that_restarts() {
    let main = Main::new();
    let mut new = profile(1000);
    let mut paint = new.engines[1].clone();
    paint.name = "paint".to_owned();
    new.engines.push(paint);
    let (status, body) = main.send("PUT", Some(as_json(&new)));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["restart"], json!(true), "{body}");
    assert_eq!(
        body["actions"][2],
        json!({"engine": "paint", "action": "load"}),
        "{body}"
    );
    assert!(main.restarts.load(Ordering::SeqCst));
}

#[test]
fn a_profile_that_does_not_fit_is_a_507_with_the_shortfall() {
    let main = Main::new();
    let mut both = profile(1000);
    both.engines[1].residency = Residency::Resident;
    let (status, body) = main.send("PUT", Some(as_json(&both)));
    assert_eq!(status.as_u16(), 507, "{body}");
    assert_eq!(body["error"]["type"], json!("too_large"), "{body}");
    assert_eq!(body["error"]["param"], json!("profile"), "{body}");
    assert_eq!(
        body["shortfall"],
        json!([{"tiers": ["device(0)"], "bytes": 200}]),
        "{body}"
    );
    assert!(body["request_id"].is_string(), "{body}");
}

#[test]
fn a_body_that_does_not_decode_or_breaks_a_rule_is_a_400_that_names_profile() {
    let main = Main::new();
    let mut unknown = as_json(&profile(1000));
    unknown["engines"][0]["weights"] = json!("x");
    // A pin on an on-demand engine breaks a rule of 3.3a.
    let mut pinned = profile(1000);
    pinned.engines[1].pinned = true;
    for body in [unknown, json!("not a profile"), as_json(&pinned)] {
        let (status, answer) = main.send("PUT", Some(body.clone()));
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} gave {answer}");
        assert_eq!(answer["error"]["type"], json!("invalid_request"), "{answer}");
        assert_eq!(answer["error"]["param"], json!("profile"), "{answer}");
    }
}

/// A lab table on an engine kind that the lab does not read is a 400 before the change, as at the start (docs/lab.md,
/// "The rules of the J-Lens"). Without the check, the node would accept it, restart, and stop with exit 3.
#[test]
fn a_lab_table_on_another_kind_is_a_400_that_names_profile() {
    let main = Main::new();
    let mut lab = as_json(&profile(1000));
    lab["engines"][0]["lab"] = json!({"jlens": "/lenses/a.safetensors"});
    let (status, answer) = main.send("PUT", Some(lab));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    assert_eq!(answer["error"]["param"], json!("profile"), "{answer}");
    assert!(
        answer["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("[engines.lab]"),
        "{answer}"
    );
}

/// The idle time is gone (spec 3.3a, "Explicit layout"). A `PUT` that names it is a 400, and the message says so.
#[test]
fn a_profile_with_an_idle_time_is_a_400_that_says_the_key_is_gone() {
    let main = Main::new();
    let mut idle = as_json(&profile(1000));
    idle["engines"][1]["idle_unload_s"] = json!(600);
    let (status, answer) = main.send("PUT", Some(idle));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    assert_eq!(answer["error"]["param"], json!("profile"), "{answer}");
    let message = answer["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("idle_unload_s") && message.contains("the key is gone"),
        "{answer}"
    );
}

#[test]
fn a_council_with_no_main_loop_has_no_profile_routes() {
    let council = Arc::new(Council::new(Vec::new(), None).unwrap());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let status = runtime.block_on(async {
        let request = Request::builder()
            .method("GET")
            .uri("/nagaya/v1/profile")
            .body(Body::empty())
            .unwrap();
        router(council).oneshot(request).await.unwrap().status()
    });
    assert_eq!(status, StatusCode::NOT_FOUND);
}
