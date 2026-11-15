//! The devices of a card test of the process. `NAGAYA_TEST_CARDS` names the cards by PCI address
//! (`nagaya_mk_bmg::test_cards`), and a profile names a device by its id in the inventory.
#![allow(dead_code)]

use nagaya::inventory::Probe;
use nagaya::probe::FileProbe;

/// The device ids of the first `count` cards of `NAGAYA_TEST_CARDS`, in the order of the variable.
pub fn devices(count: usize) -> Vec<u32> {
    let inventory = Probe::probe(&FileProbe::new("/")).expect("the inventory of this machine");
    nagaya_mk_bmg::test_cards::cards(count)
        .iter()
        .map(|address| {
            let device = inventory
                .devices
                .iter()
                .find(|device| device.pci_address.eq_ignore_ascii_case(address));
            let device = device.unwrap_or_else(|| {
                panic!(
                    "{} names {address}, and the inventory has no device at that address",
                    nagaya_mk_bmg::test_cards::VARIABLE
                )
            });
            device.id
        })
        .collect()
}

/// The device id of the one card of the test.
pub fn device() -> u32 {
    devices(1)[0]
}

/// A test of Gemma 4 names the devices 0 and 1 in its profile. It runs only when `NAGAYA_TEST_CARDS` gives it both.
pub fn need_devices_0_and_1() {
    let mut ids = devices(2);
    ids.sort_unstable();
    assert_eq!(
        ids,
        [0, 1],
        "the test names the devices 0 and 1, and {} names the devices {ids:?}",
        nagaya_mk_bmg::test_cards::VARIABLE
    );
}
