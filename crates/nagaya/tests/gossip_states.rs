//! The states of a swap in gossip (spec 2.3, "The state of an engine in gossip"): the main loop writes them through
//! the state feed.

mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::gossip::{Gossip, GossipConfig};
use nagaya::inventory::{Device, Inventory};
use nagaya::node::Node;
use nagaya::notify::EngineState;
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::state_feed::StateFeed;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal, Segment,
    Slots, Tap, Tier,
};
use support::wait_for_view;

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

fn state_in(node: &Node, engine: &str) -> String {
    let value = node.gossip().view().live[0].keys[&format!("engine:{engine}")].clone();
    serde_json::from_str::<serde_json::Value>(&value).unwrap()["state"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn a_load_writes_each_state_of_its_swap_to_gossip() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _log) = start(directory.path());
    let mut config = GossipConfig::new("solo", "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    let mut node = Node::new(Gossip::start(config).unwrap(), "127.0.0.1:8090".parse().unwrap());
    let names = vec!["chat".to_owned(), "voice".to_owned()];
    // The start writes its states at once (spec 10.1).
    node.engines_loading(&names).unwrap();
    node.engines_state(&["voice".to_owned()], EngineState::Unloaded)
        .unwrap();
    node.engines_state(&["chat".to_owned()], EngineState::Ready).unwrap();
    let ports = process.ports();
    let mut feed = StateFeed::new();
    process.load_with("voice", &["chat".to_owned()]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        node.publish_states(&mut feed, &ports, &names, Instant::now()).unwrap();
        if node.published_state("voice") == Some(EngineState::Ready)
            && node.published_state("chat") == Some(EngineState::Unloaded)
        {
            break;
        }
        assert!(Instant::now() < deadline, "the states of the swap did not reach gossip");
        std::thread::sleep(Duration::from_millis(20));
    }
    wait_for_view(node.gossip(), "the swapped states in gossip", |view| {
        let keys = &view.live[0].keys;
        keys["engine:voice"].contains("\"ready\"") && keys["engine:chat"].contains("\"unloaded\"")
    });
    assert_eq!(state_in(&node, "voice"), "ready");
    node.shutdown();
}
