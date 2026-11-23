//! The patch of a task of a ring (milestone 37 step 10): a pass patches the count of items and the addresses of a task
//! and keeps the plan. The plan is the waits and their targets, so the targets of a later task follow the new count.
//! The test reads the plan and runs no kernel, but the ring needs a device.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_patch -- --test-threads=1`

#![cfg(feature = "cuda")]

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_NOP};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

const BUFFER: u64 = 0x1000;

fn desc(items: u32, a: u64) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode: OP_NOP,
        items,
        a,
        ..Default::default()
    }
}

/// Task 0 writes the buffer, task 1 reads it, and task 2 is independent.
fn tasks(items: u32) -> Vec<Task> {
    vec![
        Task::with_items(desc(items, 1)).writes(BUFFER, 64),
        Task::with_items(desc(4, 2)).reads(BUFFER, 64),
        Task::with_items(desc(3, 3)),
    ]
}

fn open() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU")
}

#[test]
fn a_patched_count_of_items_moves_the_target_of_the_task_that_waits() {
    let dev = open();
    let mut ring = Ring::new(&dev, 1, false).unwrap();
    ring.set(&tasks(10));
    let wait = |ring: &Ring<'_>| {
        let plan = ring.planned(1);
        (plan[ring.ring_index(0)].items, plan[ring.ring_index(1)].waits.clone())
    };
    let (items, waits) = wait(&ring);
    assert_eq!((items, waits.len(), waits[0].target), (10, 1, 10));
    ring.patch(0, Task::with_items(desc(25, 0x77)).writes(BUFFER, 64));
    let (items, waits) = wait(&ring);
    assert_eq!((items, waits[0].target), (25, 25));
    assert_eq!(waits[0].event as usize, ring.ring_index(0));
    // The descriptor of the patch is the one that the launch uploads: its address changed with it.
    let plan = ring.planned(1);
    assert_eq!(plan.len(), 3, "a patch adds and removes no task");
    ring.free().unwrap();
}

#[test]
#[should_panic(expected = "changes its ranges")]
fn a_patch_that_changes_a_range_panics() {
    let dev = open();
    let mut ring = Ring::new(&dev, 1, false).unwrap();
    ring.set(&tasks(10));
    ring.patch(0, Task::with_items(desc(10, 1)).writes(BUFFER + 64, 64));
}

#[test]
#[should_panic(expected = "changes its ranges")]
fn a_patch_that_changes_the_kind_of_count_panics() {
    let dev = open();
    let mut ring = Ring::new(&dev, 1, false).unwrap();
    ring.set(&tasks(10));
    ring.patch(0, Task::per_block(desc(10, 1)).writes(BUFFER, 64));
}

#[test]
#[should_panic(expected = "no task number 3")]
fn a_patch_of_a_task_that_is_not_in_the_list_panics() {
    let dev = open();
    let mut ring = Ring::new(&dev, 1, false).unwrap();
    ring.set(&tasks(10));
    ring.patch(3, Task::with_items(desc(1, 1)));
}
