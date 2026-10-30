//! The ledger plans each engine when it takes a profile, and a swap uses those plans (spec 3.3, 3.3a). A plan can
//! take a second on a real checkpoint, and the group thread waits for each swap check. A plan at each swap check then
//! stalls the other card of the group.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::OwnedLedger;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Lifetime, Limits, MandatoryTerm, Tier};

/// A factory that counts the calls of `plan`. The config key `bytes` adds a weights term of that size on device 0.
struct Counting(Arc<AtomicUsize>);

impl Counting {
    fn limits(config: &toml::Table) -> Limits {
        let mut limits = FakeAlways::new().limits();
        limits.devices = vec![0];
        if !limits.capabilities.contains(&Capability::Close) {
            limits.capabilities.push(Capability::Close);
        }
        if let Some(bytes) = config.get("bytes").and_then(|value| value.as_integer()) {
            limits.memory.mandatory.push(MandatoryTerm {
                name: "weights".to_owned(),
                tier: Tier::Device(0),
                bytes: bytes as u64,
                lifetime: Lifetime::Engine,
                kinds: Vec::new(),
            });
        }
        limits
    }
}

impl Factory for Counting {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Self::limits(config))
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        unreachable!("the ledger opens no engine")
    }
}

fn engine(name: &str, residency: Residency) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: "counting".to_owned(),
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
    let bytes = 1 << 40;
    PlacementProfile {
        engines,
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes,
            },
        ],
        store: None,
    }
}

fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1 << 40,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 40,
    }
}

#[test]
fn a_swap_uses_the_plans_of_the_admission_and_a_new_profile_plans_again() {
    let plans = Arc::new(AtomicUsize::new(0));
    let mut registry = Registry::new();
    registry.register("counting", Box::new(Counting(Arc::clone(&plans))));
    let inventory = inventory();
    let first = profile(vec![engine("a", Residency::Resident), engine("b", Residency::OnDemand)]);
    let mut ledger = OwnedLedger::new();
    ledger.admit(&first, &registry, &inventory).expect("the profile fits");
    let admitted = plans.load(Ordering::SeqCst);
    assert!(admitted >= 2, "the admission planned {admitted} times for two engines");

    let (b, none) = (vec!["b".to_owned()], Vec::<String>::new());
    ledger.check_swap(&b, &none).expect("the swap fits");
    ledger.swap(&b, &none).expect("the swap fits");
    ledger.check_swap(&none, &b).expect("the swap back fits");
    ledger.swap(&none, &b).expect("the swap back fits");
    assert_eq!(
        plans.load(Ordering::SeqCst),
        admitted,
        "a swap planned the engines again"
    );

    // Engine `c` of the new profile does not fit its budget, so a swap that loads it must see its new plan.
    let mut c = engine("c", Residency::OnDemand);
    c.config.insert("bytes".to_owned(), toml::Value::Integer(1 << 41));
    let second = profile(vec![
        engine("a", Residency::Resident),
        engine("b", Residency::OnDemand),
        c,
    ]);
    ledger
        .reprofile(&second, &registry, &inventory)
        .expect("the new profile fits");
    let changed = plans.load(Ordering::SeqCst);
    assert!(
        changed >= admitted + 3,
        "the new profile planned {} times for three engines",
        changed - admitted
    );
    let c = vec!["c".to_owned()];
    assert!(
        ledger.check_swap(&c, &none).is_err(),
        "a swap after the new profile used the plans of the old profile"
    );
    assert_eq!(
        plans.load(Ordering::SeqCst),
        changed,
        "a swap after the new profile planned again"
    );
}
