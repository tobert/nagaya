//! Helpers for the tests of the council service: a fake engine with a context term, a profile, and a started process.

#![allow(dead_code)]

use std::path::Path;
use std::time::Duration;

use nagaya::admission::{Factory, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{OpenedEngine, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Coverage, Engine, Identity, Kind, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassOut, Seal, Segment,
    SlotLimit, Slots, Tap, Tier,
};

/// The name of the engine in the profile.
pub const ENGINE: &str = "toy";

/// A fake engine that declares a mandatory term for each context.
pub struct WithContextTerm {
    inner: FakeAlways,
    bytes: u64,
}

impl Engine for WithContextTerm {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        let mut limits = self.inner.limits();
        if self.bytes > 0 {
            limits.memory.mandatory.push(MandatoryTerm {
                name: "context".to_owned(),
                tier: Tier::Host,
                bytes: self.bytes,
                lifetime: Lifetime::Context,
                kinds: vec![Kind::Prefill, Kind::Read],
            });
        }
        limits
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

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }
}

/// The factory of [`WithContextTerm`] engines.
pub struct TermFactory {
    pub bytes: u64,
    pub slots: Vec<SlotLimit>,
    pub vocab: Option<usize>,
}

impl TermFactory {
    fn engine(&self) -> WithContextTerm {
        let mut inner = FakeAlways::with_slots(self.slots.clone());
        if let Some(vocab) = self.vocab {
            inner = inner.with_vocab(vocab);
        }
        WithContextTerm {
            inner,
            bytes: self.bytes,
        }
    }
}

impl Factory for TermFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.engine().limits())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(self.engine()))
    }
}

/// Slots that are big enough for the toy contexts of the tests.
pub fn roomy_slots(count: u32) -> Vec<SlotLimit> {
    vec![SlotLimit { bytes: 16384, count }]
}

pub fn profile(host_budget: u64, nvme: u64, host_store: Option<u64>) -> PlacementProfile {
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: ENGINE.to_owned(),
            kind: "term".to_owned(),
            devices: vec![0],
            config: toml::Table::new(),
            share: 1,
            pin_identity: false,
        }],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 4096,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: host_budget,
            },
        ],
        store: Some(StoreSpec {
            nvme,
            host: host_store,
            interval_copies: 0,
        }),
    }
}

pub fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            memory_bytes: 1024,
            unified: false,
        }],
        host_bytes: 1 << 20,
    }
}

/// Start a process with one engine and wait until it opens.
pub fn start(
    directory: &Path,
    factory: TermFactory,
    host_budget: u64,
    host_store: Option<u64>,
) -> (Process, Vec<OpenedEngine>) {
    let mut registry = Registry::new();
    registry.register("term", Box::new(factory));
    let mut process = Start::new(profile(host_budget, 1 << 24, host_store), registry, inventory())
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(10))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(10)).unwrap();
    (process, opened)
}
