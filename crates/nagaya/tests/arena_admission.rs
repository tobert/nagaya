//! Profile admission counts the arena fds of the profile against the ceiling of the fd store (spec 10.2).

use nagaya::admission::{
    AdmitError, FAKE_ARENA_KIND, Factory, GroupDevice, OpenError, PlanError, Registry, admit, admit_with_fd_ceiling,
};
use nagaya::inventory::{Device, Inventory};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya_abi::fake::{FAKE_ARENA_PART, FakeAlways};
use nagaya_abi::{ArenaPart, Engine, Limits, Tier};
use nagaya_ledger::Ledger;

fn engine(name: &str, kind: &str) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: vec![0],
        config: toml::Table::new(),
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency: Default::default(),
        pinned: false,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    let count = engines.len() as u64;
    PlacementProfile {
        engines,
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 4096 * count,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 256 * count,
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
            memory_bytes: 1 << 20,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 20,
    }
}

fn fake_part() -> Vec<ArenaPart> {
    vec![ArenaPart {
        device: 0,
        part: FAKE_ARENA_PART.to_owned(),
    }]
}

#[test]
fn a_profile_with_more_arena_fds_than_the_ceiling_is_refused_and_the_ledger_keeps_its_state() {
    let two = profile(vec![engine("one", FAKE_ARENA_KIND), engine("two", FAKE_ARENA_KIND)]);
    let mut ledger = Ledger::new();
    let before = ledger.unassigned();
    let error = admit_with_fd_ceiling(&two, &Registry::with_fake(), &inventory(), &mut ledger, Some(1)).unwrap_err();
    match &error {
        AdmitError::ArenaFds { needed, ceiling } => assert_eq!((*needed, *ceiling), (2, 1)),
        other => panic!("the refusal is {other:?}"),
    }
    assert!(error.to_string().contains("FileDescriptorStoreMax"), "{error}");
    assert_eq!(ledger.unassigned(), before, "a refusal changes the ledger");
    // The same profile fits the ledger, so the ceiling alone refused it.
    admit_with_fd_ceiling(&two, &Registry::with_fake(), &inventory(), &mut ledger, Some(2)).unwrap();
    assert_ne!(ledger.unassigned(), before);
}

#[test]
fn a_profile_within_the_ceiling_carries_the_arena_parts_of_each_engine() {
    let two = profile(vec![engine("one", FAKE_ARENA_KIND), engine("two", "fake")]);
    let admitted =
        admit_with_fd_ceiling(&two, &Registry::with_fake(), &inventory(), &mut Ledger::new(), Some(1)).unwrap();
    assert_eq!(admitted.engines[0].arena_parts, fake_part());
    assert!(
        admitted.engines[1].arena_parts.is_empty(),
        "a kind with no arena has a part"
    );
}

#[test]
fn with_no_fd_store_the_arena_fds_are_not_counted() {
    let two = profile(vec![engine("one", FAKE_ARENA_KIND), engine("two", FAKE_ARENA_KIND)]);
    let admitted = admit(&two, &Registry::with_fake(), &inventory(), &mut Ledger::new()).unwrap();
    assert_eq!(admitted.engines[1].arena_parts, fake_part());
}

struct BadPartName;

impl Factory for BadPartName {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(FakeAlways::new()))
    }

    fn arena_parts(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
        Ok(vec![ArenaPart {
            device: 0,
            part: "w:0".to_owned(),
        }])
    }
}

#[test]
fn an_arena_part_whose_fd_name_systemd_refuses_refuses_the_profile() {
    let mut registry = Registry::new();
    registry.register("bad", Box::new(BadPartName));
    let error = admit_with_fd_ceiling(
        &profile(vec![engine("one", "bad")]),
        &registry,
        &inventory(),
        &mut Ledger::new(),
        None,
    )
    .unwrap_err();
    assert!(matches!(error, AdmitError::Plan { .. }), "{error:?}");
    assert!(error.to_string().contains("w:0"), "{error}");
}

struct PartOnAnotherDevice;

impl Factory for PartOnAnotherDevice {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(FakeAlways::new()))
    }

    fn arena_parts(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
        Ok(vec![ArenaPart {
            device: 1,
            part: "w".to_owned(),
        }])
    }
}

#[test]
fn an_arena_part_on_a_device_that_the_group_lacks_refuses_the_profile() {
    let mut registry = Registry::new();
    registry.register("far", Box::new(PartOnAnotherDevice));
    let error = admit_with_fd_ceiling(
        &profile(vec![engine("one", "far")]),
        &registry,
        &inventory(),
        &mut Ledger::new(),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("device 1"), "{error}");
}
