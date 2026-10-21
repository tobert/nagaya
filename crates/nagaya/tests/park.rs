//! The park writes every state to the store, and a restore reads every part back (spec 10.3).

use std::time::Duration;

use nagaya::admission::{Factory, OpenError, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::park::{StoreBudget, identity_digest, open_store};
use nagaya::process::Start;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{DeviceId, Engine, Limits, OutputsWanted, PartKind, Segment, SegmentInputs, SlotLimit, State, Tier};
use nagaya_ledger::Admission;
use nagaya_store::{
    EngineFacts, Put, Reason, Restored, Snapshot, Store, StoreError, Tensor, Tier as StoreTier, recipe_not_sealed,
    recipe_root, snapshot_from_engine,
};

struct SeedFactory {
    limits: Limits,
}

impl Factory for SeedFactory {
    fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
        Ok(self.limits.clone())
    }

    fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(seeded()))
    }
}

fn segment(state: State, tokens: &[u32]) -> Segment {
    Segment {
        state,
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (0..tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    }
}

/// A fake with two slots. The park writes every slot, so the test keeps the count at two.
fn two_slots() -> FakeAlways {
    FakeAlways::with_slots(vec![
        SlotLimit { bytes: 64, count: 1 },
        SlotLimit { bytes: 256, count: 1 },
    ])
}

/// A fake with three tokens in slot 0 and five tokens in slot 1. The row coverage differs.
fn seeded() -> FakeAlways {
    let mut engine = two_slots();
    engine.pass(&[segment(State::Slot(0), &[1, 2, 3])], &[]).unwrap();
    engine.pass(&[segment(State::Slot(1), &[4, 5, 6, 7, 8])], &[]).unwrap();
    engine
}

fn limits_on(devices: &[DeviceId]) -> Limits {
    let mut limits = two_slots().limits();
    limits.devices = devices.to_vec();
    limits
}

fn engine(name: &str, kind: &str, devices: &[DeviceId]) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: devices.to_vec(),
        config: toml::Table::new(),
        share: 1,
        pin_identity: false,
    }
}

fn profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![engine("seed", "seed", &[0])],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 4096,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 4096,
            },
        ],
        store: Some(StoreSpec {
            nvme: 65536,
            host: None,
            interval_copies: 0,
        }),
    }
}

fn inventory(ids: &[DeviceId]) -> Inventory {
    Inventory {
        devices: ids
            .iter()
            .map(|id| Device {
                id: *id,
                name: format!("device-{id}"),
                memory_bytes: 1024,
                unified: false,
            })
            .collect(),
        host_bytes: 1024,
    }
}

fn expected() -> Vec<(State, Snapshot)> {
    let mut source = seeded();
    let slot0 = snapshot_from_engine(&mut source, State::Slot(0), &[(1, 0, 3), (2, 0, 3)]).unwrap();
    let mut source = seeded();
    let slot1 = snapshot_from_engine(&mut source, State::Slot(1), &[(1, 0, 5), (2, 1, 4)]).unwrap();
    vec![(State::Slot(0), slot0), (State::Slot(1), slot1)]
}

#[test]
fn a_stop_writes_every_state_and_a_restore_reads_every_part() {
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(
        "seed",
        Box::new(SeedFactory {
            limits: limits_on(&[0]),
        }),
    );
    let mut process = Start::new(profile(), registry, inventory(&[0]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert!(report.parked.failures.is_empty(), "{report:?}");
    assert_eq!(report.parked.states, 2, "both slots are parked");
    assert_eq!(report.parked.parts, 6, "three stored parts in each slot");
    assert_eq!(report.parked.bytes, 152, "64 bytes in slot 0 and 88 in slot 1");

    let mut store = Store::open(directory.path()).unwrap();
    assert_eq!(store.entries().len(), 2, "the two states have different bytes");
    let identity = identity_digest(&seeded().identity());
    for (state, snapshot) in expected() {
        let id = snapshot.content_id().unwrap();
        assert!(store.entries().contains_key(&id), "the store holds {id:?}");
        // Spec 6.1: each put names its engine and carries the engine's class and chain limits.
        let entry = &store.entries()[&id];
        let limits = seeded().limits();
        assert_eq!(entry.engine, "seed");
        assert_eq!(
            entry.determinism,
            nagaya_store::DeterminismClass::from(limits.determinism)
        );
        assert_eq!(entry.chain.max_depth, limits.chain.max_depth);
        assert_eq!(entry.chain.delta_fraction, limits.chain.delta_fraction);
        let mut target = FakeAlways::new();
        let results = store.restore(&id, &mut target, state, &identity, 20).unwrap();
        assert_eq!(results, Restored::Done);
        for tensor in &snapshot.tensors {
            let index = target
                .limits()
                .parts
                .iter()
                .position(|part| part.name == tensor.name)
                .unwrap();
            let mut bytes = vec![0u8; tensor.data.len()];
            match tensor.kind {
                PartKind::Fixed => target.parts().unwrap().part_read(state, index, &mut bytes).unwrap(),
                PartKind::Cells | PartKind::Ring => {
                    let (start, end) = tensor.coverage.unwrap();
                    let from = start;
                    target
                        .parts()
                        .unwrap()
                        .cells_read(state, index, from, end - start, &mut bytes)
                        .unwrap();
                }
                PartKind::Derived => continue,
            }
            assert_eq!(bytes, tensor.data, "part {}", tensor.name);
        }
    }
}

#[test]
fn the_store_host_budget_is_the_unassigned_host_bytes() {
    let admission = Admission {
        engines: Vec::new(),
        unassigned: vec![(vec![Tier::Device(0), Tier::Host], 4096), (vec![Tier::Host], 128)],
    };
    assert_eq!(
        StoreBudget::host_from_admission(&admission),
        128,
        "a shared pool is not a store budget"
    );
}

#[test]
fn the_store_budget_takes_the_profile_values() {
    let admission = Admission {
        engines: Vec::new(),
        unassigned: vec![(vec![Tier::Host], 128)],
    };
    let named = StoreSpec {
        nvme: 4096,
        host: Some(1024),
        interval_copies: 5,
    };
    assert_eq!(
        StoreBudget::from_spec(&named, &admission),
        StoreBudget {
            host: 1024,
            nvme: 4096,
            interval_copies: 5
        }
    );
    let unnamed = StoreSpec {
        nvme: 4096,
        host: None,
        interval_copies: 0,
    };
    assert_eq!(
        StoreBudget::from_spec(&unnamed, &admission),
        StoreBudget {
            host: 128,
            nvme: 4096,
            interval_copies: 0
        }
    );
}

fn copy_snapshot(seed: u8) -> Snapshot {
    Snapshot {
        tensors: vec![Tensor {
            name: "state".to_owned(),
            kind: PartKind::Fixed,
            dtype: nagaya_abi::Dtype::U8,
            shape: vec![4],
            data: vec![seed; 4],
            coverage: None,
            block_format: None,
        }],
        parent: None,
    }
}

fn put_copy(
    store: &mut Store,
    sealed: nagaya_store::RecipeId,
    seed: u8,
) -> Result<nagaya_store::ContentId, StoreError> {
    let engine = FakeAlways::new();
    store.put(Put {
        snapshot: &copy_snapshot(seed),
        recipe: recipe_not_sealed(&sealed),
        previous_recipe: None,
        chains_from: None,
        engine: EngineFacts::of("fake", &engine.limits()),
        identity: [1u8; 32],
        owner: 1,
        tier: StoreTier::Host,
        reason: Reason::Interval,
        sealed_recipe: Some(sealed),
        provenance: None,
        now_ms: 10,
    })
}

#[test]
fn the_opened_store_holds_the_profile_cap_of_interval_copies() {
    let sealed = recipe_root(&[1u8; 32], b"", &[], b"");
    let other = recipe_root(&[2u8; 32], b"", &[], b"");
    let dir = tempfile::tempdir().unwrap();
    let budget = |interval_copies| StoreBudget {
        host: 1 << 20,
        nvme: 1 << 20,
        interval_copies,
    };
    let mut none = open_store(dir.path(), budget(0)).unwrap();
    assert!(
        put_copy(&mut none, sealed, 1).is_err(),
        "a cap of 0 refuses interval copies"
    );
    drop(none);
    let mut one = open_store(dir.path(), budget(1)).unwrap();
    let first = put_copy(&mut one, sealed, 1).unwrap();
    let second = put_copy(&mut one, other, 2).unwrap();
    assert!(!one.entries().contains_key(&first), "a cap of 1 keeps one copy");
    assert!(one.entries().contains_key(&second));
}

/// A fake with equal coverage in both slots and different bytes.
fn equal_coverage(second: &[u32]) -> FakeAlways {
    let mut engine = two_slots();
    engine.pass(&[segment(State::Slot(0), &[1, 2, 3])], &[]).unwrap();
    engine.pass(&[segment(State::Slot(1), second)], &[]).unwrap();
    engine
}

fn stop_with(engine: fn() -> FakeAlways) -> (nagaya::process::StopReport, tempfile::TempDir) {
    struct Factory2(fn() -> FakeAlways);
    impl Factory for Factory2 {
        fn plan(&self, _config: &toml::Table) -> Result<Limits, PlanError> {
            Ok(limits_on(&[0]))
        }
        fn open(&self, _config: &toml::Table) -> Result<Box<dyn Engine>, OpenError> {
            Ok(Box::new((self.0)()))
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("seed", Box::new(Factory2(engine)));
    let mut process = Start::new(profile(), registry, inventory(&[0]))
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(2))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(2)).unwrap();
    let report = process.stop(Duration::from_secs(2), &SignalFlag::new());
    (report, directory)
}

#[test]
fn a_stop_with_equal_coverage_and_different_bytes_stores_both_states() {
    // Review finding 2: one recipe for every slot tripped the tripwire on equal coverage.
    let (report, directory) = stop_with(|| equal_coverage(&[4, 5, 6]));
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert!(report.parked.failures.is_empty(), "{report:?}");
    assert_eq!(report.parked.states, 2);
    let store = Store::open(directory.path()).unwrap();
    assert_eq!(store.entries().len(), 2, "different bytes give two entries");
    assert!(store.refused().is_empty(), "no recipe is refused");
    for entry in store.entries().values() {
        assert_eq!(entry.state, nagaya_store::EntryState::Live, "no entry is quarantined");
    }
    let recipes: std::collections::BTreeSet<_> = store.entries().values().map(|entry| entry.recipe).collect();
    assert_eq!(recipes.len(), 2, "different bytes give different recipes");
}

#[test]
fn a_stop_with_equal_bytes_in_two_slots_stores_one_entry() {
    let (report, directory) = stop_with(|| equal_coverage(&[1, 2, 3]));
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.states, 2, "both states are parked");
    let store = Store::open(directory.path()).unwrap();
    assert_eq!(store.entries().len(), 1, "equal bytes give one entry");
    assert!(store.refused().is_empty());
}

#[test]
fn the_park_recipe_has_a_fixed_test_vector() {
    // The vector was computed outside the code. See the commit message.
    let identity = [7u8; 32];
    let content = nagaya_store::ContentId::from_hex(&"ab".repeat(32)).unwrap();
    let recipe = nagaya::park::park_recipe(&identity, &content);
    assert_eq!(
        recipe.hex(),
        "5dfd383a409537f237e3b81e34a85e7e33964978f7574ecf5e446848cfcd863a"
    );
}

/// The put of a park of `copy_snapshot(seed)`, as the store thread makes it.
fn park_put(snapshot: &Snapshot) -> Put<'_> {
    let identity = [1u8; 32];
    Put {
        snapshot,
        recipe: nagaya::park::park_recipe(&identity, &snapshot.content_id().unwrap()),
        previous_recipe: None,
        chains_from: None,
        engine: EngineFacts::of("fake", &FakeAlways::new().limits()),
        identity,
        owner: nagaya::park::PARK_OWNER,
        tier: StoreTier::Nvme,
        reason: Reason::Shutdown,
        sealed_recipe: None,
        provenance: None,
        now_ms: 20,
    }
}

#[test]
fn a_park_of_the_bytes_of_an_interval_copy_replaces_the_copy_with_a_park_entry() {
    // Spec 6.13: an interval copy never goes to NVMe. The park removes it and puts the state under the park recipe.
    let dir = tempfile::tempdir().unwrap();
    let mut store = open_store(
        dir.path(),
        StoreBudget {
            host: 1 << 20,
            nvme: 1 << 20,
            interval_copies: 4,
        },
    )
    .unwrap();
    let sealed = recipe_root(&[1u8; 32], b"", &[], b"");
    let copy = put_copy(&mut store, sealed, 3).unwrap();
    let snapshot = copy_snapshot(3);
    let put = park_put(&snapshot);
    let recipe = put.recipe;
    let parked = nagaya::park::store_park(&mut store, put, copy).unwrap();
    assert_eq!(parked, copy, "the same bytes give the same content id");
    let entry = &store.entries()[&copy];
    assert_eq!(
        (entry.tier, entry.reason, entry.recipe),
        (StoreTier::Nvme, Reason::Shutdown, recipe)
    );
}

#[test]
fn a_park_of_the_bytes_of_a_held_snapshot_parks_that_entry_with_its_recipe() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let snapshot = copy_snapshot(4);
    let built = recipe_root(&[1u8; 32], b"geometry", &[], b"");
    let mut first = park_put(&snapshot);
    (first.recipe, first.tier, first.reason, first.owner) = (built, StoreTier::Host, Reason::PrefillDone, 1);
    let id = store.put(first).unwrap();
    let parked = nagaya::park::store_park(&mut store, park_put(&snapshot), id).unwrap();
    assert_eq!(parked, id);
    let entry = &store.entries()[&id];
    assert_eq!(
        (entry.tier, entry.reason, entry.recipe, entry.owner),
        (StoreTier::Nvme, Reason::PrefillDone, built, 1),
        "the park changes only the tier"
    );
    assert_eq!(store.entries().len(), 1);
}
