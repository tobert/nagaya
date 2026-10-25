//! The states of engines through a swap in place (spec 3.3a, 9.4, 10.1): an on-demand engine starts `unloaded`, a swap
//! records the new state of each engine, and the main loop hears of each swap.

use std::sync::mpsc;
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::lifecycle::{self, Options};
use nagaya::process::Event;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Kind, Lifetime, Limits, MandatoryTerm, Tier};
use nagaya_sched::Class;
use nagaya_telemetry::testing::Capture;

/// The fake with `close` and 600 bytes of weights on device 0.
struct Closing;

fn limits() -> Limits {
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

struct Fake(FakeAlways);

impl Engine for Fake {
    fn identity(&self) -> nagaya_abi::Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        limits()
    }

    fn pass(
        &mut self,
        segments: &[nagaya_abi::Segment],
        taps: &[nagaya_abi::Tap],
    ) -> Result<nagaya_abi::PassOut, nagaya_abi::PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.0.coverage()
    }
}

impl Factory for Closing {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Fake(FakeAlways::new())))
    }
}

fn engine(name: &str, residency: Residency) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: "closing".to_owned(),
        devices: vec![0],
        config: toml::Table::new(),
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
    }
}

fn profile() -> PlacementProfile {
    PlacementProfile {
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
                bytes: 1 << 20,
            },
        ],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
        }),
    }
}

fn start(directory: &std::path::Path) -> lifecycle::Ready {
    let mut registry = Registry::new();
    registry.register("closing", Box::new(Closing));
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1 << 20,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 20,
    };
    lifecycle::start_with(
        profile(),
        registry,
        inventory,
        directory,
        None,
        Options::new(Duration::from_secs(5)),
    )
    .unwrap()
}

/// The value of `nagaya.engine.state` for one engine and one state.
fn state(capture: &Capture, engine: &str, state: &str) -> Option<f64> {
    capture
        .metric("nagaya.engine.state")
        .into_iter()
        .find(|point| point.attributes["engine"] == engine && point.attributes["state"] == state)
        .map(|point| point.value)
}

struct Done(mpsc::Sender<()>);

impl JobBody for Done {
    fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = self.0.send(());
        Ran::Done
    }
}

fn job(engine: usize) -> (JobSpec, mpsc::Receiver<()>) {
    let (done, finished) = mpsc::channel();
    let spec = JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first: Kind::Read,
        engine,
        body: Box::new(Done(done)),
    };
    (spec, finished)
}

#[test]
fn an_on_demand_engine_starts_unloaded() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start(directory.path());
    assert_eq!(state(&capture, "chat", "ready"), Some(1.0));
    assert_eq!(state(&capture, "voice", "unloaded"), Some(1.0));
    assert_eq!(state(&capture, "voice", "loading"), Some(0.0));
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn a_swap_records_the_new_state_of_each_engine() {
    let capture = Capture::start(Vec::new());
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start(directory.path());
    let (spec, finished) = job(1);
    ready.process.submit_job(0, spec).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(state(&capture, "chat", "unloaded"), Some(1.0));
    assert_eq!(state(&capture, "voice", "ready"), Some(1.0));
    let (spec, finished) = job(0);
    ready.process.submit_job(0, spec).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(state(&capture, "chat", "ready"), Some(1.0));
    assert_eq!(state(&capture, "voice", "unloaded"), Some(1.0));
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}

#[test]
fn the_wait_gives_each_swap_to_the_main_loop_and_goes_on() {
    let directory = tempfile::tempdir().unwrap();
    let mut ready = start(directory.path());
    let (spec, finished) = job(1);
    ready.process.submit_job(0, spec).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    let mut events = Vec::new();
    let seen = std::cell::Cell::new(0);
    // The wait ends at the first event, or after 2 s with none, so a lost event fails the test below.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let code = lifecycle::wait_with_events(
        &ready.process,
        &SignalFlag::new(),
        None,
        Duration::from_millis(5),
        &mut || (seen.get() > 0 || std::time::Instant::now() > deadline).then_some(ExitCode::Clean),
        &mut |event| {
            seen.set(seen.get() + 1);
            events.push(event);
        },
    );
    assert_eq!(code, ExitCode::Clean);
    assert!(
        matches!(&events[..], [Event::Swapped { unloaded, loaded, .. }]
            if unloaded == &["chat".to_owned()] && loaded.len() == 1 && loaded[0].name == "voice"),
        "{events:?}"
    );
    // The main loop keeps the list of loaded engines, so the stop parks the engines that are loaded.
    lifecycle::apply_swap(&mut ready.opened, &events[0]);
    let names: Vec<&str> = ready.opened.iter().map(|engine| engine.name.as_str()).collect();
    assert_eq!(names, ["voice"]);
    ready.process.stop(Duration::from_secs(2), &SignalFlag::new());
}
