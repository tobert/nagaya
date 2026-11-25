use nagaya::inventory::{Device, FakeProbe, Inventory, Probe, ProbeError};

fn strix_halo_inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "gfx1151".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 128 << 30,
            unified: true,
            ignored: false,
        }],
        host_bytes: 128 << 30,
    }
}

#[test]
fn the_fake_probe_gives_the_inventory() {
    let inventory = strix_halo_inventory();
    assert_eq!(FakeProbe::new(inventory.clone()).probe().unwrap(), inventory);
}

#[test]
fn a_device_is_found_by_its_id() {
    let inventory = strix_halo_inventory();
    assert_eq!(inventory.device(0).unwrap().name, "gfx1151");
    assert_eq!(inventory.device(0).unwrap().memory_bytes, 128 << 30);
    assert!(inventory.device(1).is_none());
}

#[test]
fn a_failed_probe_is_an_error() {
    let error = FakeProbe::failing("no device was found").probe().unwrap_err();
    assert_eq!(
        error,
        ProbeError {
            message: "no device was found".to_owned()
        }
    );
    assert!(error.to_string().contains("no device was found"), "{error}");
}

fn b70_pair_inventory() -> Inventory {
    let device = |id, address: &str| Device {
        id,
        name: format!("device-{id}"),
        pci_address: address.to_owned(),
        memory_bytes: 32 << 30,
        unified: false,
        ignored: false,
    };
    Inventory {
        devices: vec![
            device(0, "0000:03:00.0"),
            device(1, "0000:07:00.0"),
            device(2, "0000:80:00.0"),
        ],
        host_bytes: 128 << 30,
    }
}

#[test]
fn a_device_is_not_ignored_by_default() {
    assert!(strix_halo_inventory().devices.iter().all(|device| !device.ignored));
}

#[test]
fn an_ignored_device_keeps_its_id_and_the_others_are_not_marked() {
    let mut inventory = b70_pair_inventory();
    inventory.ignore(&["0000:80:00.0".to_owned()]).unwrap();
    assert_eq!(
        inventory
            .devices
            .iter()
            .map(|device| (device.id, device.ignored))
            .collect::<Vec<_>>(),
        vec![(0, false), (1, false), (2, true)]
    );
    assert_eq!(inventory.device(2).unwrap().pci_address, "0000:80:00.0");
}

#[test]
fn an_ignored_address_in_capital_letters_matches() {
    let mut inventory = b70_pair_inventory();
    inventory.devices[1].pci_address = "0000:c2:00.0".to_owned();
    inventory.ignore(&["0000:C2:00.0".to_owned()]).unwrap();
    assert!(inventory.devices[1].ignored);
}

#[test]
fn an_ignored_address_that_the_probe_did_not_find_refuses_the_start_and_names_it() {
    let mut inventory = b70_pair_inventory();
    let error = inventory
        .ignore(&["0000:80:00.0".to_owned(), "0000:99:00.0".to_owned()])
        .unwrap_err();
    assert!(error.to_string().contains("0000:99:00.0"), "{error}");
    assert_eq!(error.exit_code().code(), 3);
}
