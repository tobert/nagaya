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

/// A factory that counts the calls of `plan`. The config key `bytes` gives one weights term of that size on device 0.
struct Counting(Arc<AtomicUsize>);

impl Counting {
    fn limits(config: &toml::Table) -> Limits {
        let mut limits = FakeAlways::new().limits();
        limits.devices = vec![0];
        if !limits.capabilities.contains(&Capability::Close) {
            limits.capabilities.push(Capability::Close);
        }
        if let Some(bytes) = config.get("bytes").and_then(|value| value.as_integer()) {
            // The weights are then the only term, so a test can count the bytes of each engine.
            limits.memory.mandatory.clear();
            limits.memory.elastic.clear();
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
        lab: toml::Table::new(),
        share: 1,
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
    ledger.plan_swap(&b, &none).expect("the swap fits");
    ledger.swap(&b, &none).expect("the swap fits");
    ledger.plan_swap(&none, &b).expect("the swap back fits");
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

/// An engine of `bytes` on device 0.
fn sized(name: &str, bytes: i64, residency: Residency) -> EnginePlacement {
    let mut placement = engine(name, residency);
    placement.config.insert("bytes".to_owned(), toml::Value::Integer(bytes));
    placement
}

/// Two pending swaps can end in either order, so each must fit while the other has not unloaded (spec 3.3a). In 1000
/// bytes, a (400) and b (400) are loaded. The swap of d (100) for b fits. The swap of c (700) for a fits after both,
/// with c and d (800), and not when it ends first, with b and c (1100), so its plan is refused.
#[test]
fn a_pending_swap_must_fit_in_any_order() {
    let mut registry = Registry::new();
    registry.register("counting", Box::new(Counting(Arc::new(AtomicUsize::new(0)))));
    let mut profile = profile(vec![
        sized("a", 400, Residency::Resident),
        sized("b", 400, Residency::Resident),
        sized("c", 700, Residency::OnDemand),
        sized("d", 100, Residency::OnDemand),
    ]);
    profile.budgets[0].bytes = 1000;
    let mut ledger = OwnedLedger::new();
    ledger.admit(&profile, &registry, &inventory()).expect("a and b fit");
    let names = |list: &[&str]| list.iter().map(|name| (*name).to_owned()).collect::<Vec<_>>();
    ledger.plan_swap(&names(&["d"]), &names(&["b"])).expect("a and d fit");
    assert!(
        ledger.plan_swap(&names(&["c"]), &names(&["a"])).is_err(),
        "the swap of c can end before the swap of d, with b still loaded"
    );
    ledger
        .swap(&names(&["d"]), &names(&["b"]))
        .expect("the swap of d is charged");
    ledger
        .plan_swap(&names(&["c"]), &names(&["a"]))
        .expect("after the swap of d, c fits beside d");
    ledger
        .swap(&names(&["c"]), &names(&["a"]))
        .expect("the swap of c is charged");
}

/// A swap is checked beside each set of other pending swaps, each one whole, so no check counts a load without its
/// unload (spec 3.3a). In 1000 bytes, a (500) and b (400) are loaded, and the swap of c (550) for a waits. An unload of
/// b fits in each order, and its check once counted a and c together (1050).
#[test]
fn a_pending_swap_counts_each_other_swap_whole() {
    let mut registry = Registry::new();
    registry.register("counting", Box::new(Counting(Arc::new(AtomicUsize::new(0)))));
    let mut profile = profile(vec![
        sized("a", 500, Residency::Resident),
        sized("b", 400, Residency::Resident),
        sized("c", 550, Residency::OnDemand),
        sized("d", 100, Residency::OnDemand),
    ]);
    profile.budgets[0].bytes = 1000;
    let mut ledger = OwnedLedger::new();
    ledger.admit(&profile, &registry, &inventory()).expect("a and b fit");
    let names = |list: &[&str]| list.iter().map(|name| (*name).to_owned()).collect::<Vec<_>>();
    ledger.plan_swap(&names(&["c"]), &names(&["a"])).expect("b and c fit");
    ledger
        .plan_swap(&[], &names(&["b"]))
        .expect("an unload of b fits in each order");
    ledger.swap(&[], &names(&["b"])).expect("the unload of b is charged");
    ledger
        .swap(&names(&["c"]), &names(&["a"]))
        .expect("the swap of c is charged");
    // Now c (550) and b (400) are loaded, and the swap of a (500) for c waits. A swap of d (100) for b fits in each
    // order: beside c (650), or beside a (600). With a and c both counted, it would be 1150.
    ledger.plan_swap(&names(&["b"]), &[]).expect("b fits beside c");
    ledger.swap(&names(&["b"]), &[]).expect("the load of b is charged");
    ledger
        .plan_swap(&names(&["a"]), &names(&["c"]))
        .expect("a fits beside b");
    ledger
        .plan_swap(&names(&["d"]), &names(&["b"]))
        .expect("d fits beside c, and beside a");
}
