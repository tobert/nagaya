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
                memory_bytes: 1024,
                unified: false,
            })
            .collect(),
        host_bytes: 1024,
    }
}

#[test]
fn the_matching_devices_pass() {
    let profile = profile(vec![engine("a", &[0])]);
    assert_eq!(check_devices(&profile, &[planned("a", &[0])], &inventory(&[0])), Ok(()));
}

#[test]
fn a_device_that_the_inventory_lacks_is_refused() {
    let profile = profile(vec![engine("a", &[7])]);
    let error = check_devices(&profile, &[planned("a", &[7])], &inventory(&[0])).unwrap_err();
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
fn a_profile_device_that_the_plan_lacks_is_refused() {
    let profile = profile(vec![engine("a", &[0, 1])]);
    let error = check_devices(&profile, &[planned("a", &[0])], &inventory(&[0, 1])).unwrap_err();
    assert_eq!(
        error,
        DeviceError::NotPlanned {
            engine: "a".to_owned(),
            device: 1,
        }
    );
    let message = error.to_string();
    assert!(message.contains('a'), "{message}");
    assert!(message.contains('1'), "{message}");
}

#[test]
fn a_plan_device_that_the_profile_lacks_is_refused() {
    let profile = profile(vec![engine("a", &[0])]);
    let error = check_devices(&profile, &[planned("a", &[0, 1])], &inventory(&[0, 1])).unwrap_err();
    assert_eq!(
        error,
        DeviceError::NotInProfile {
            engine: "a".to_owned(),
            device: 1,
        }
    );
    let message = error.to_string();
    assert!(message.contains('a'), "{message}");
    assert!(message.contains('1'), "{message}");
}

#[test]
fn a_device_order_mismatch_is_refused() {
    let profile = profile(vec![engine("a", &[1, 0])]);
    let error = check_devices(&profile, &[planned("a", &[0, 1])], &inventory(&[0, 1])).unwrap_err();
    assert_eq!(
        error,
        DeviceError::Order {
            engine: "a".to_owned(),
            device: 1,
        }
    );
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
