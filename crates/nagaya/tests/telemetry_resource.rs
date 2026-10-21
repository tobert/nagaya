//! The resource attributes of the telemetry (spec 9.1).

use nagaya::inventory::{Device, Inventory};
use nagaya::telemetry::{host_name, resource};
use nagaya_telemetry::testing::Capture;

fn inventory() -> Inventory {
    Inventory {
        devices: vec![
            Device {
                id: 0,
                name: "gfx1151".to_owned(),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 1000,
                unified: true,
            },
            Device {
                id: 1,
                name: "cpu".to_owned(),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 24,
                unified: false,
            },
        ],
        host_bytes: 5000,
    }
}

#[test]
fn the_resource_names_the_host_and_the_hardware_of_the_inventory() {
    let capture = Capture::start(resource(Some("gfx1151host"), Some(&inventory())));
    nagaya_telemetry::instruments().store_lookup("any", nagaya_telemetry::LookupOutcome::Miss);
    let attributes = capture.resource();
    let get = |key: &str| attributes.get(key).map(String::as_str);
    assert_eq!(get("service.name"), Some("nagaya"));
    assert_eq!(get("host.name"), Some("gfx1151host"));
    assert_eq!(get("nagaya.device.count"), Some("2"));
    assert_eq!(get("nagaya.device.names"), Some("gfx1151,cpu"));
    assert_eq!(get("nagaya.device.memory_bytes"), Some("1024"));
    assert_eq!(get("nagaya.host.memory_bytes"), Some("5000"));
}

#[test]
fn a_missing_host_name_or_inventory_leaves_the_attribute_out() {
    assert!(resource(None, None).is_empty());
    let only_host = resource(Some("gfx1151host"), None);
    assert_eq!(only_host.len(), 1);
}

#[test]
fn the_host_name_is_read_from_the_kernel() {
    let name = host_name().expect("this machine has a host name");
    assert!(!name.is_empty());
    assert!(!name.contains('\n'));
}
