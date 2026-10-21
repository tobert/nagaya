use nagaya::admission::AdmittedEngine;
use nagaya::inventory::{Device, Inventory};
use nagaya::placement::{DeviceError, check_devices, device_groups};
use nagaya::profile::{EnginePlacement, PlacementProfile};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{DeviceId, Engine, Limits};

fn engine(name: &str, devices: &[DeviceId]) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: "fake".to_owned(),
        devices: devices.to_vec(),
        config: toml::Table::new(),
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    PlacementProfile {
        engines,
        budgets: Vec::new(),
        store: None,
    }
}

fn limits(devices: &[DeviceId]) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.devices = devices.to_vec();
    limits
}

fn planned(name: &str, devices: &[DeviceId]) -> AdmittedEngine {
    AdmittedEngine {
        name: name.to_owned(),
        limits: limits(devices),
    }
}

fn inventory(ids: &[DeviceId]) -> Inventory {
    Inventory {
        devices: ids
            .iter()
            .map(|id| Device {
                id: *id,
                name: format!("device-{id}"),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 1024,
                unified: false,
            })
            .collect(),
        host_bytes: 1024,
    }
}

#[test]
fn group_indexes_for_each_profile_device_pass() {
    let profile = profile(vec![engine("a", &[0])]);
    assert_eq!(check_devices(&profile, &[planned("a", &[0])], &inventory(&[0])), Ok(()));
}

/// The engine declares group indexes, so a profile on machine devices 1 and 2 takes a plan of [0, 1] (spec 2.4).
#[test]
fn a_plan_of_group_indexes_passes_on_other_machine_devices() {
    let profile = profile(vec![engine("a", &[1, 2])]);
    assert_eq!(
        check_devices(&profile, &[planned("a", &[0, 1])], &inventory(&[0, 1, 2])),
        Ok(())
    );
}

#[test]
fn a_device_that_the_inventory_lacks_is_refused() {
    let profile = profile(vec![engine("a", &[7])]);
    let error = check_devices(&profile, &[planned("a", &[0])], &inventory(&[0])).unwrap_err();
    assert_eq!(
        error,
        DeviceError::Unknown {
            engine: "a".to_owned(),
            device: 7,
        }
    );
    let message = error.to_string();
    assert!(message.contains('a'), "{message}");
    assert!(message.contains('7'), "{message}");
}

#[test]
fn a_plan_with_fewer_devices_than_the_profile_is_refused() {
    let profile = profile(vec![engine("a", &[0, 1])]);
    let error = check_devices(&profile, &[planned("a", &[0])], &inventory(&[0, 1])).unwrap_err();
    assert_eq!(
        error,
        DeviceError::GroupIndexes {
            engine: "a".to_owned(),
            planned: vec![0],
            count: 2,
        }
    );
    let message = error.to_string();
    assert!(message.contains("`a`"), "{message}");
    assert!(message.contains("[0]"), "{message}");
    assert!(message.contains("0 to 1"), "{message}");
}

#[test]
fn a_plan_with_more_devices_than_the_profile_is_refused() {
    let profile = profile(vec![engine("a", &[0])]);
    let error = check_devices(&profile, &[planned("a", &[0, 1])], &inventory(&[0, 1])).unwrap_err();
    assert_eq!(
        error,
        DeviceError::GroupIndexes {
            engine: "a".to_owned(),
            planned: vec![0, 1],
            count: 1,
        }
    );
}

/// A plan of machine ids, or of indexes out of order, is not a list of group indexes.
#[test]
fn a_plan_that_is_not_the_group_indexes_in_order_is_refused() {
    let profile = profile(vec![engine("a", &[1, 2])]);
    for plan in [vec![1, 2], vec![1, 0]] {
        let error = check_devices(&profile, &[planned("a", &plan)], &inventory(&[0, 1, 2])).unwrap_err();
        assert_eq!(
            error,
            DeviceError::GroupIndexes {
                engine: "a".to_owned(),
                planned: plan.clone(),
                count: 2,
            }
        );
    }
}

#[test]
fn two_devices_get_two_groups() {
    let profile = profile(vec![engine("a", &[0]), engine("b", &[1])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].devices, vec![0]);
    assert_eq!(groups[0].engines, vec![0]);
    assert_eq!(groups[1].devices, vec![1]);
    assert_eq!(groups[1].engines, vec![1]);
}

#[test]
fn two_engines_on_one_device_share_a_group() {
    let profile = profile(vec![engine("a", &[0]), engine("b", &[0])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].devices, vec![0]);
    assert_eq!(groups[0].engines, vec![0, 1]);
}

#[test]
fn two_engines_on_one_device_set_share_a_group() {
    let profile = profile(vec![engine("a", &[0, 1]), engine("b", &[1, 0])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].devices, vec![0, 1]);
    assert_eq!(groups[0].engines, vec![0, 1]);
}

#[test]
fn engines_whose_device_sets_overlap_share_a_group() {
    let profile = profile(vec![engine("a", &[0, 1]), engine("b", &[1])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].devices, vec![0, 1]);
    assert_eq!(groups[0].engines, vec![0, 1]);
}

#[test]
fn an_engine_that_bridges_two_groups_merges_them() {
    // Engines a and b share no device. Engine c uses a device of each, so all three share one group.
    let profile = profile(vec![engine("a", &[0]), engine("b", &[1]), engine("c", &[0, 1, 2])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].devices, vec![0, 1, 2]);
    assert_eq!(groups[0].engines, vec![0, 1, 2]);
}

#[test]
fn a_chain_of_overlaps_makes_one_group() {
    let profile = profile(vec![engine("a", &[0, 1]), engine("b", &[2, 3]), engine("c", &[1, 2])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].devices, vec![0, 1, 2, 3]);
    assert_eq!(groups[0].engines, vec![0, 1, 2]);
}

#[test]
fn disjoint_device_sets_stay_apart_beside_an_overlap() {
    let profile = profile(vec![engine("a", &[0]), engine("b", &[1]), engine("c", &[1, 2])]);
    let groups = device_groups(&profile);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].devices, vec![0]);
    assert_eq!(groups[0].engines, vec![0]);
    assert_eq!(groups[1].devices, vec![1, 2]);
    assert_eq!(groups[1].engines, vec![1, 2]);
}
