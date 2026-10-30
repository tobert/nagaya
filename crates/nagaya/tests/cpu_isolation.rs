//! The CPU masks of the threads (spec 3.2a). The test reads the mask of each thread with `sched_getaffinity`.
//!
//! This file holds one test. The test starts the control runtime and the process threads in this process, and it
//! reads every thread of this process by name. Another test in the file would add threads.

mod task_support;

use std::mem::size_of;
use std::time::Duration;

use nagaya::admission::Registry;
use nagaya::cpu_sets::{self, MIN_ONLINE_CPUS};
use nagaya::gossip::{Gossip, GossipConfig};
use nagaya::inventory::{Device, Inventory};
use nagaya::lifecycle::{self, Options};
use nagaya::probe::FileProbe;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya_abi::Tier;
use task_support::threads_named;

/// The thread names of the nagaya (spec 3.2). The kernel keeps 15 bytes of a name.
const CONTROL: &str = "nagaya-control";
const LEDGER: &str = "nagaya-ledger";
const WATCHDOG: &str = "nagaya-watchdog";
const GROUP: &str = "nagaya-group-0";
const STORE: &str = "nagaya-store";

/// The CPUs that the thread with this id may run on, from `sched_getaffinity`.
fn mask_of(tid: i32) -> Vec<u32> {
    // SAFETY: an all-zero `cpu_set_t` is an empty mask, and the call fills it.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `mask` is writable and has the size that the call is given.
    let result = unsafe { libc::sched_getaffinity(tid, size_of::<libc::cpu_set_t>(), &mut mask) };
    assert_eq!(result, 0, "sched_getaffinity failed for thread {tid}");
    (0..libc::CPU_SETSIZE as usize)
        // SAFETY: each index is below the mask size.
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .map(|cpu| cpu as u32)
        .collect()
}

/// The masks of the threads of this process that have the name, in thread-id order.
fn masks_named(name: &str) -> Vec<Vec<u32>> {
    threads_named(name).into_iter().map(mask_of).collect()
}

fn profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: "fake-one".to_owned(),
            kind: "fake".to_owned(),
            devices: vec![0],
            config: toml::Table::new(),
            family: toml::Table::new(),
            share: 1,
            pin_identity: false,
            residency: Default::default(),
            pinned: false,
        }],
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
            objects: None,
        }),
    }
}

#[test]
fn the_control_threads_do_not_run_on_the_group_cores() {
    let sets = match cpu_sets::for_machine(&FileProbe::new("/")) {
        Ok(sets) => sets,
        Err(error) => {
            eprintln!("skipped: this machine has fewer than {MIN_ONLINE_CPUS} online CPUs: {error}");
            return;
        }
    };
    let directory = tempfile::tempdir().unwrap();
    let mut gossip_config = GossipConfig::new("isolation", "127.0.0.1:0".parse().unwrap());
    gossip_config.housekeeping = Some(sets.housekeeping.clone());
    let gossip = Gossip::start(gossip_config).unwrap();
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "fake-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1024,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1024,
    };
    // The start goes through `lifecycle`, as the binary does.
    let options = Options::new(Duration::from_secs(5)).cpu_sets(sets.clone());
    let ready = lifecycle::start_with(
        profile(),
        Registry::with_fake(),
        inventory,
        directory.path(),
        None,
        options,
    )
    .unwrap();

    let control = masks_named(CONTROL);
    assert!(control.len() >= 2, "the control runtime has two workers: {control:?}");
    for (name, masks) in [(LEDGER, masks_named(LEDGER)), (WATCHDOG, masks_named(WATCHDOG))] {
        assert_eq!(masks.len(), 1, "one {name} thread");
        for mask in masks.iter().chain(control.iter()) {
            assert_eq!(
                mask, &sets.housekeeping,
                "{name} and the control runtime run on the housekeeping set"
            );
        }
    }
    let group = masks_named(GROUP);
    assert_eq!(group.len(), 1, "one group thread");
    assert_eq!(group[0], sets.pool, "the group thread runs on the pool set");
    let store = masks_named(STORE);
    assert_eq!(store.len(), 1, "one store thread");
    assert_eq!(store[0], sets.pool, "the store thread runs on the pool set");
    for mask in control
        .iter()
        .chain(masks_named(LEDGER).iter())
        .chain(masks_named(WATCHDOG).iter())
    {
        assert!(
            mask.iter().all(|cpu| !group[0].contains(cpu)),
            "a control thread uses a group core: {mask:?} against {:?}",
            group[0]
        );
    }
    drop(ready);
    gossip.shutdown();
}
