use nagaya::inventory::{Device, FakeProbe, Inventory, Probe, ProbeError};

fn gfx1151host_inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "gfx1151".to_owned(),
            memory_bytes: 128 << 30,
            unified: true,
        }],
        host_bytes: 128 << 30,
    }
}

#[test]
fn the_fake_probe_gives_the_inventory() {
    let inventory = gfx1151host_inventory();
    assert_eq!(FakeProbe::new(inventory.clone()).probe().unwrap(), inventory);
}

#[test]
fn a_device_is_found_by_its_id() {
    let inventory = gfx1151host_inventory();
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
