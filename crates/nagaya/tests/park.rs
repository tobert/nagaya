//! The park writes every state to the store, and a restore reads every part back (spec 10.3).

use std::time::Duration;

use nagaya::admission::{Factory, OpenError, PlanError, Registry};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory};
use nagaya::park::identity_digest;
use nagaya::process::Start;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{DeviceId, Engine, Limits, OutputsWanted, PartKind, Segment, SegmentInputs, State, Tier, WriteResult};
use nagaya_store::{Snapshot, Store, snapshot_from_engine};

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

/// A fake with three tokens in slot 0 and five tokens in slot 1. The row coverage differs.
fn seeded() -> FakeAlways {
    let mut engine = FakeAlways::new();
    engine.pass(&[segment(State::Slot(0), &[1, 2, 3])], &[]).unwrap();
    engine.pass(&[segment(State::Slot(1), &[4, 5, 6, 7, 8])], &[]).unwrap();
    engine
}

fn limits_on(devices: &[DeviceId]) -> Limits {
    let mut limits = FakeAlways::new().limits();
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
        let mut target = FakeAlways::new();
        let results = store.restore(&id, &mut target, state, &identity, 20).unwrap();
        assert!(results.iter().all(|result| *result == WriteResult::Ok), "{results:?}");
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
                    let from = if tensor.kind == PartKind::Ring { 0 } else { start };
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
