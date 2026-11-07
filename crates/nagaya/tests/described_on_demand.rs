//! A text engine on demand is a served model before it loads (spec 11.1): its factory describes the identity and the
//! limits of its open (`Factory::describe`), and the council serves it from them. A request gets a 503 until a load.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Event, Process, Start};
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

/// The fake engine with the toy family's vocabulary and slots that hold its contexts, as the council tests have.
fn fake() -> FakeAlways {
    let slots = vec![SlotLimit {
        bytes: 16384,
        count: 3,
        tokens: 4096,
    }];
    FakeAlways::with_slots(slots)
        .with_vocab(gogi_toy::VOCAB)
        .with_family("toy")
}

fn limits() -> Limits {
    let mut limits = fake().limits();
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

/// A factory that describes its engine when `describes` is true.
struct Toy {
    describes: bool,
}

impl Factory for Toy {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Closing(fake())))
    }

    fn describe(&self, _config: &toml::Table) -> Result<Option<(Identity, Limits)>, PlanError> {
        Ok(self.describes.then(|| (fake().identity(), limits())))
    }
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register("described", Box::new(Toy { describes: true }));
    registry.register("plain", Box::new(Toy { describes: false }));
    registry
}

fn engine(name: &str, kind: &str, residency: Residency) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: vec![0],
        config: toml::Table::new(),
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    PlacementProfile {
        engines,
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

#[test]
fn only_an_engine_on_demand_that_its_factory_describes_is_described() {
    let profile = profile(vec![
        engine("resident", "described", Residency::Resident),
        engine("chat", "described", Residency::OnDemand),
        engine("other", "plain", Residency::OnDemand),
    ]);
    let described = describe_on_demand(&profile, &registry()).unwrap();
    let names: Vec<&str> = described.iter().map(|engine| engine.name.as_str()).collect();
    assert_eq!(names, vec!["chat"]);
    assert_eq!(described[0].identity, fake().identity());
}

fn start(directory: &std::path::Path, profile: PlacementProfile) -> Process {
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
    let mut process = Start::new(profile, registry(), inventory)
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    process
}

fn send(council: &Arc<Council>, path: &str, body: Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
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
fn a_described_engine_is_a_served_model_that_answers_after_a_load() {
    let directory = tempfile::tempdir().unwrap();
    let profile = profile(vec![engine("chat", "described", Residency::OnDemand)]);
    let described: Vec<Described> = describe_on_demand(&profile, &registry()).unwrap();
    let process = start(directory.path(), profile.clone());
    let mut families = FamilyRegistry::new();
    families.register("toy", |_, _| {
        Ok(OpenedFamily {
            family: Arc::new(ToyFamily::new()),
            tokenizer_hash: None,
        })
    });
    let council = Arc::new(
        council_with(
            &families,
            &profile,
            &[],
            &described,
            &process.ports(),
            directory.path(),
            &[],
            None,
        )
        .unwrap()
        .with_loads(process.ports()),
    );
    let generation = json!({
        "model": fake().identity().model,
        "context": {"thinking": true, "messages": [{"role": "user", "content": "hi"}]},
        "max_tokens": 1,
    });
    let (status, body) = send(&council, "/council/v2/generations", generation.clone());
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], json!("model_unavailable"), "{body}");

    let (status, body) = send(&council, "/nagaya/v1/engines/chat/load", json!({}));
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(swapped(&process), vec!["chat".to_owned()]);
    let (status, body) = send(&council, "/council/v2/generations", generation);
    assert_eq!(status, StatusCode::OK, "{body}");
}
