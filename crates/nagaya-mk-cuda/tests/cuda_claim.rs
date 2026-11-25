//! GPU gate of the claim of the later items. A task with more items than blocks gives item
//! b to block b, and the later items to the blocks by claim. `OP_SELFTEST_CLAIM` counts each item that runs and writes
//! the block that ran it. Every item must run exactly once, and the event of the task must reach its item count.
//! Some items spin, so that their blocks are slow and the other blocks take more items.
//!
//! The ring holds the same tasks twice, so one launch runs the claim words of 2 * TASKS tasks. A launch zeroes the claim
//! words, so each of the launches that follow gives each item once again.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_claim -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_SELFTEST_CLAIM};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

const GRIDS: [u32; 4] = [48, 24, 7, 1];
/// The tasks of one copy of the ring. The ring holds two copies.
const TASKS: usize = 40;
const LAUNCHES: usize = 4;
/// Every 5th item spins for 40 us. A spin is a slow block.
const SPIN_PERIOD: u32 = 5;
const SPIN_NS: u32 = 40_000;

/// The items of task `t`: more than the blocks of the largest grid for most tasks, and fewer for some.
fn items_of(t: usize) -> usize {
    match t % 5 {
        0 => 30,
        1 => 49,
        2 => 97,
        3 => 150,
        _ => 211,
    }
}

fn run_with_watchdog(ring: &Ring<'_>, grid: u32, threads: u32, shared: u32) {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..3000 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                if done.load(Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_claim: the ring did not finish within 30 s at grid {grid}; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run_grid(grid, threads, shared).unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

#[test]
fn the_claim_gives_each_item_once_in_each_launch() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let counts_of = |tasks: usize| (0..tasks).map(items_of).sum::<usize>();
    let all_tasks = 2 * TASKS;
    let words: usize = (0..all_tasks).map(|t| items_of(t % TASKS)).sum();
    let counts = dev.alloc_device(words * 4).unwrap();
    let blocks = dev.alloc_device(words * 4).unwrap();
    // The word of task `t` in the arrays: the items of the tasks before it.
    let mut offsets = Vec::with_capacity(all_tasks);
    let mut at = 0usize;
    for t in 0..all_tasks {
        offsets.push(at);
        at += items_of(t % TASKS);
    }
    assert_eq!(at, words);
    assert_eq!(counts_of(TASKS) * 2, words);
    let descs: Vec<MkTaskDescriptor> = (0..all_tasks)
        .map(|t| MkTaskDescriptor {
            opcode: OP_SELFTEST_CLAIM,
            items: items_of(t % TASKS) as u32,
            c: counts + (offsets[t] * 4) as u64,
            e: blocks + (offsets[t] * 4) as u64,
            n: SPIN_NS,
            k: SPIN_PERIOD,
            ..Default::default()
        })
        .collect();
    let mut ring = Ring::new(&dev, 1, false).unwrap();
    let tasks: Vec<Task> = descs
        .iter()
        .enumerate()
        .map(|(i, desc)| {
            let task = Task::with_items(*desc);
            if i == 0 { task } else { task.after(i - 1) }
        })
        .collect();
    ring.set(&tasks);
    ring.zero_device().unwrap();
    let shapes = [
        (DEFAULT_BLOCK_THREADS, dev.max_dynamic_shared_bytes().unwrap()),
        (256u32, 0u32),
    ];
    for (threads, shared) in shapes {
        for grid in GRIDS {
            let mut moved = 0usize;
            for launch in 0..LAUNCHES {
                dev.memset_zero(counts, words * 4).unwrap();
                dev.memset_zero(blocks, words * 4).unwrap();
                run_with_watchdog(&ring, grid, threads, shared);
                let mut got = vec![0u32; words];
                let mut who = vec![0u32; words];
                unsafe {
                    dev.copy_dtoh(got.as_mut_ptr().cast::<c_void>(), counts, words * 4)
                        .unwrap();
                    dev.copy_dtoh(who.as_mut_ptr().cast::<c_void>(), blocks, words * 4)
                        .unwrap();
                }
                let events = ring.read_scoreboard().unwrap();
                for t in 0..all_tasks {
                    let items = items_of(t % TASKS);
                    for i in 0..items {
                        let w = offsets[t] + i;
                        assert_eq!(
                            got[w], 1,
                            "task {t}, item {i} ran {} times at grid {grid}, {threads} threads, launch {launch}",
                            got[w]
                        );
                        assert!(who[w] < grid, "task {t}, item {i} ran in block {} of {grid}", who[w]);
                        if i < grid as usize {
                            assert_eq!(who[w] as usize, i, "task {t}: item {i} is the first item of block {i}");
                        } else if who[w] as usize != i % grid as usize {
                            moved += 1;
                        }
                    }
                    // The task signals the event with its index in the ring. Its count is the items of the task.
                    let event = events[ring.ring_index(t)] as usize;
                    assert_eq!(
                        event, items,
                        "task {t}: the blocks added {event} to the event, not {items}, at grid {grid}, launch {launch}"
                    );
                }
            }
            if grid >= 7 {
                assert!(
                    moved > 0,
                    "at grid {grid} no later item went to a block other than the static one: the claim is not dynamic"
                );
            }
            eprintln!("cuda_claim: grid {grid}, {threads} threads: {moved} later items ran off the static block");
        }
    }
    ring.free().unwrap();
    for ptr in [counts, blocks] {
        dev.free_device(ptr).unwrap();
    }
}
