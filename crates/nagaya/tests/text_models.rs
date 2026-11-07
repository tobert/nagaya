//! One process serves several text models (spec 11.7, "Several text models in one process", Proposed): each answers
//! by its name, identity lists each, and the specs, the pins and the snapshots stay one set for the server.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Event, OpenedEngine, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::serving::{Described, FamilyRegistry, OpenedFamily, council_with, describe_on_demand};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Limits, Parts, PassError, PassOut, Seal, Segment, SlotLimit, Slots, Tap,
    Tier,
};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// The served models of the profile: two resident engines and one on demand.
const ALPHA: &str = "toy-alpha";
const BETA: &str = "toy-beta";
const GAMMA: &str = "toy-gamma";

/// The fake engine with the toy family, and the served model name of `model`.
fn fake(model: &str) -> FakeAlways {
    let slots = vec![SlotLimit {
        bytes: 16384,
        count: 3,
        tokens: 4096,
    }];
    FakeAlways::with_slots(slots)
        .with_vocab(gogi_toy::VOCAB)
        .with_family("toy")
        .with_model(model)
}

fn limits() -> Limits {
    let mut limits = fake(ALPHA).limits();
    limits.capabilities.push(Capability::Close);
    limits
}

/// The fake engine with `close` in its limits, as a swap needs.
struct Closing(FakeAlways);

impl Engine for Closing {
    fn identity(&self) -> Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.0.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.0.seal()
    }
}

/// A factory of the toy engine. The config key `model` names its served model.
struct Toy;

fn model_of(config: &toml::Table) -> String {
    config["model"]
        .as_str()
        .expect("each engine of the test names its model")
        .to_owned()
}

impl Factory for Toy {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits())
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Closing(fake(&model_of(config)))))
    }

    fn describe(&self, config: &toml::Table) -> Result<Option<(Identity, Limits)>, PlanError> {
        Ok(Some((fake(&model_of(config)).identity(), limits())))
    }
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register("toy", Box::new(Toy));
    registry
}

fn engine(name: &str, model: &str, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("model".to_owned(), toml::Value::String(model.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: "toy".to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

fn profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![
            engine("alpha", ALPHA, Residency::Resident),
            engine("beta", BETA, Residency::Resident),
            engine("gamma", GAMMA, Residency::OnDemand),
        ],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1 << 30,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 1 << 30,
            },
        ],
        store: Some(StoreSpec {
            nvme: 1 << 24,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn start(directory: &std::path::Path) -> (Process, Vec<OpenedEngine>) {
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1 << 31,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 31,
    };
    let mut process = Start::new(profile(), registry(), inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(5)).unwrap();
    (process, opened)
}

fn families() -> FamilyRegistry {
    let mut families = FamilyRegistry::new();
    families.register("toy", |_, _| {
        Ok(OpenedFamily {
            family: Arc::new(ToyFamily::new()),
            tokenizer_hash: None,
        })
    });
    families
}

/// The council of the process, as `main` makes it: the opened engines and the described engine on demand.
fn council(directory: &std::path::Path, process: &Process, opened: &[OpenedEngine]) -> Result<Arc<Council>, String> {
    let profile = profile();
    let described: Vec<Described> = describe_on_demand(&profile, &registry()).unwrap();
    let council = council_with(
        &families(),
        &profile,
        opened,
        &described,
        &process.ports(),
        directory,
        &[],
        None,
    )?;
    Ok(Arc::new(council.with_loads(process.ports())))
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

/// A context with the name of `model` in it. The fake engines of two models make the same state from the same tokens,
/// and the store keeps one recipe for each content id. Thus each model builds its own text.
fn context(model: &str) -> Value {
    json!({"messages": [{"role": "user", "content": format!("Delete the build directory, {model}."), "snap": true}]})
}

fn generation(model: &str) -> Value {
    json!({"model": model, "context": context(model), "max_tokens": 2})
}

/// Build the context on `model`, and give the head.
fn head(council: &Arc<Council>, model: &str) -> String {
    let (status, built) = send(
        council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": model, "context": context(model)})),
    );
    assert_eq!(status, StatusCode::OK, "{built}");
    assert_eq!(built["model"], json!(model), "{built}");
    built["head"].as_str().unwrap().to_owned()
}

fn swapped(process: &Process) -> Vec<String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Some(Event::Swapped { loaded, .. }) = process.recv(Duration::from_millis(50)) {
            return loaded.into_iter().map(|engine| engine.name).collect();
        }
    }
    panic!("no swap came in 5 s");
}

#[test]
fn a_process_with_two_text_models_serves_each_by_its_name() {
    let directory = tempfile::tempdir().unwrap();
    let (process, opened) = start(directory.path());
    let council = council(directory.path(), &process, &opened).expect("a profile with several text models starts");
    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    let listed: Vec<(&str, &str)> = identity["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| (model["model"].as_str().unwrap(), model["status"].as_str().unwrap()))
        .collect();
    assert_eq!(
        listed,
        vec![(ALPHA, "ready"), (BETA, "ready"), (GAMMA, "unloaded")],
        "{identity}"
    );
    for model in [ALPHA, BETA] {
        let (status, body) = send(&council, "POST", "/council/v2/generations", Some(generation(model)));
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
        assert_eq!(body["model"], json!(model), "{body}");
    }
    // Two served models never share a snapshot (gogi "Served models").
    assert_ne!(head(&council, ALPHA), head(&council, BETA));
}

#[test]
fn the_specs_pins_and_snapshots_of_two_text_models_are_one_set() {
    let directory = tempfile::tempdir().unwrap();
    let (process, opened) = start(directory.path());
    let council = council(directory.path(), &process, &opened).unwrap();
    let (alpha, beta) = (head(&council, ALPHA), head(&council, BETA));
    // A snapshot names the served model of the engine that made it, whichever service holds the routes.
    for (snapshot, model) in [(&alpha, ALPHA), (&beta, BETA)] {
        let (status, body) = send(&council, "GET", &format!("/council/v2/snapshots/{snapshot}"), None);
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
        assert_eq!(body["model"], json!(model), "{body}");
    }
    // A pin names no served model, so one owner pins a snapshot of each.
    let (status, put) = send(
        &council,
        "PUT",
        "/council/v2/pins/kernel-1",
        Some(json!({"snapshots": [alpha, beta], "ttl_s": 600})),
    );
    assert_eq!(status, StatusCode::OK, "{put}");
    assert_eq!(
        put["snapshots"],
        json!([{"id": alpha, "held": true}, {"id": beta, "held": true}]),
        "{put}"
    );
    let (_, pinned) = send(&council, "GET", &format!("/council/v2/snapshots/{beta}"), None);
    assert_eq!(pinned["pins"][0]["owner"], json!("kernel-1"), "{pinned}");
    // A spec that the server holds is read by a decision of each text model.
    let spec = json!({"name": "gate", "instructions": "Judge it.", "input_label": "Case", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "What happens.",
         "criteria": [{"option": "allow", "means": "fine"}, {"option": "ask", "means": "not fine"}]}]});
    let (status, held) = send(&council, "POST", "/council/v2/specs", Some(spec));
    assert_eq!(status, StatusCode::OK, "{held}");
    for model in [ALPHA, BETA] {
        let decision = json!({"model": model, "spec_id": held["spec_id"], "case": "rm -rf build",
                              "contexts": [context(model)]});
        let (status, body) = send(&council, "POST", "/council/v2/decisions", Some(decision));
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
    }
    // One set of files in the store directory.
    let specs: Vec<_> = std::fs::read_dir(directory.path().join("specs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(specs.len(), 1, "{specs:?}");
    let book: Value = serde_json::from_slice(&std::fs::read(directory.path().join("pins.json")).unwrap()).unwrap();
    assert_eq!(
        book["kernel-1"]["snapshots"].as_array().map(Vec::len),
        Some(2),
        "{book}"
    );
}

#[test]
fn a_text_model_that_is_not_ready_is_a_503_that_names_it_and_loads_by_its_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (process, opened) = start(directory.path());
    let council = council(directory.path(), &process, &opened).unwrap();
    let (status, body) = send(&council, "POST", "/council/v2/generations", Some(generation(GAMMA)));
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(
        (
            &body["error"]["type"],
            &body["error"]["model"],
            &body["error"]["status"]
        ),
        (&json!("model_unavailable"), &json!(GAMMA), &json!("unloaded")),
        "{body}"
    );
    // The other text models answer while one is unloaded.
    let (status, body) = send(&council, "POST", "/council/v2/generations", Some(generation(BETA)));
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = send(&council, "POST", "/nagaya/v1/engines/gamma/load", Some(json!({})));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(swapped(&process), vec!["gamma".to_owned()]);
    let (status, body) = send(&council, "POST", "/council/v2/generations", Some(generation(GAMMA)));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], json!(GAMMA), "{body}");
    let gamma = head(&council, GAMMA);
    let (status, body) = send(&council, "GET", &format!("/council/v2/snapshots/{gamma}"), None);
    assert_eq!((status, &body["model"]), (StatusCode::OK, &json!(GAMMA)), "{body}");
}
