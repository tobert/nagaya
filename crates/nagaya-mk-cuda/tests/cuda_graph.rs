//! GPU gate of the dependency graph (Milestone 37 step 7, substep 3): a ring with chains and branches of copy tasks
//! gives exact results, and the profile shows that tasks with no dependency overlap.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_graph -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::graph::signals;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_COPY, OP_SELFTEST_FILL, PROFILE_WORDS, selftest_value};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::{INKLING2_PROFILE_CUBIN, test_cards};

/// The slow block of a fill. It is block 0, because a copy reads the words that block 0 writes. The other blocks
/// pass the fill at once.
const SLOW_BLOCK: u32 = 0;
/// The delay of the fill of chain A in ns. It is the slowest, so the JOIN waits for chain A.
const DELAY_NS: u32 = 1_200_000;
/// The delay of the fill of chain C in ns.
const DELAY_C_NS: u32 = 400_000;
/// Floats in each buffer of a copy: `COPY_ITEMS` items of `ITEM_FLOATS` floats.
const COPY_ITEMS: u32 = 16;
const ITEM_FLOATS: u32 = 256;
const FLOATS: usize = (COPY_ITEMS * ITEM_FLOATS) as usize;
const BYTES: usize = FLOATS * 4;

/// Run the ring in a thread with a watchdog, so that a hang fails the test.
fn run_with_watchdog(ring: &Ring<'_>) {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..300 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if done.load(Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_graph: the ring did not finish within 30 s; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run().unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

fn copy(dst: CUdeviceptr, src: CUdeviceptr) -> Task {
    Task::with_items(MkTaskDescriptor {
        opcode: OP_COPY,
        items: COPY_ITEMS,
        a: dst,
        b: src,
        n: ITEM_FLOATS,
        ..Default::default()
    })
    .reads(src, BYTES)
    .writes(dst, BYTES)
}

fn download(dev: &CudaDevice, ptr: CUdeviceptr) -> Vec<u32> {
    let mut out = vec![0u32; FLOATS];
    unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), ptr, BYTES) }.unwrap();
    out
}

#[test]
fn chains_and_branches_of_copy_tasks_are_exact_and_independent_tasks_overlap() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU");
    let mut ring = Ring::new(&dev, 1024, true).unwrap();
    let blocks = ring.grid_blocks() as usize;
    assert!(blocks > SLOW_BLOCK as usize);
    let alloc = || {
        let ptr = dev.alloc_device(BYTES.max(blocks * FLOATS * 4)).unwrap();
        dev.memset_zero(ptr, BYTES).unwrap();
        ptr
    };
    // The fill of chain A writes `FLOATS` words in each block, so its buffer has one slice for each block.
    let fill_buf = alloc();
    let fill_c = alloc();
    let [a2, a3, b0, b1, b2, b3, c1, out] = [(); 8].map(|()| alloc());
    let b_pattern: Vec<u32> = (0..FLOATS as u32).map(|i| 0x3f80_0000 + i * 7).collect();
    unsafe { dev.copy_htod(b0, b_pattern.as_ptr().cast::<c_void>(), BYTES) }.unwrap();

    // Chain A starts with a slow fill. Chains B and C do not depend on it. The ring lists the chains in turn, so a
    // block that is past the fill runs the first task of chain B while the slow block is still in the fill.
    let fill = Task::per_block(MkTaskDescriptor {
        opcode: OP_SELFTEST_FILL,
        a: fill_buf,
        n: FLOATS as u32,
        k: DELAY_NS,
        kt: SLOW_BLOCK,
        group_items: 5,
        ..Default::default()
    })
    .writes(fill_buf, blocks * FLOATS * 4);
    let fill_of_c = Task::per_block(MkTaskDescriptor {
        opcode: OP_SELFTEST_FILL,
        a: fill_c,
        n: FLOATS as u32,
        k: DELAY_C_NS,
        kt: SLOW_BLOCK,
        group_items: 9,
        ..Default::default()
    })
    .writes(fill_c, blocks * FLOATS * 4);
    // A copy of chain A or C reads the words of block 0 of the fill.
    let tasks = vec![
        fill,               // 0: A1
        copy(b1, b0),       // 1: B1
        fill_of_c,          // 2: C1, a slow fill
        copy(a2, fill_buf), // 3: A2
        copy(b2, b1),       // 4: B2
        copy(a3, a2),       // 5: A3
        copy(b3, b2),       // 6: B3
        copy(c1, fill_c),   // 7: C2
        // The end of the three chains: a copy of the end of chain A that also reads the ends of B and C.
        copy(out, a3).reads(b3, BYTES).reads(c1, BYTES), // 8: the end
    ];
    ring.set(&tasks);

    // Three producers (A3, B3 and C2) feed the last task, so the builder inserts one JOIN before it.
    let planned = ring.planned(ring.grid_blocks());
    assert_eq!(planned.iter().filter(|p| p.origin.is_none()).count(), 1, "{planned:?}");
    assert_eq!(planned.len(), tasks.len() + 1);
    let join_at = planned.iter().position(|p| p.origin.is_none()).unwrap();
    assert_eq!(join_at, tasks.len() - 1, "the JOIN stands before the last task");

    dev.memset_zero(out, BYTES).unwrap();
    run_with_watchdog(&ring);

    // The bits: B and C copy their sources, and A copies the first words of block 0 of the fill.
    assert_eq!(download(&dev, b3), b_pattern);
    let c_want: Vec<u32> = (0..FLOATS as u32).map(|i| selftest_value(9, 0, i)).collect();
    assert_eq!(download(&dev, c1), c_want);
    let a_want: Vec<u32> = (0..FLOATS as u32).map(|i| selftest_value(5, 0, i)).collect();
    assert_eq!(download(&dev, a3), a_want);
    assert_eq!(
        download(&dev, out),
        a_want,
        "the last task ran before the end of chain A"
    );

    // Each event holds the items of its task: 16 for a copy, one for each block of a fill, and 1 for the JOIN.
    let scoreboard = ring.read_scoreboard().unwrap();
    assert_eq!(scoreboard.len(), planned.len() + 1);
    for (index, task) in planned.iter().enumerate() {
        let want = match task.origin.map(|i| &tasks[i]) {
            None => 1,
            Some(t) if t.per_block => signals(blocks as u32),
            Some(t) => t.desc.items,
        };
        assert_eq!(scoreboard[index], want, "event {index}: {task:?}");
    }

    // The profile: the first task of chain B leaves its wait in most blocks before the slow block ends the fill.
    let profile = ring.read_profile().unwrap();
    let slot = |task: usize, block: usize| {
        let base = (ring.ring_index(task) * blocks + block) * PROFILE_WORDS;
        (profile[base], profile[base + 1], profile[base + 2])
    };
    let fill_end = (0..blocks).map(|b| slot(0, b).2).max().unwrap();
    // Only the blocks that have an item of B1 leave a wait.
    let with_item = 0..COPY_ITEMS as usize;
    let early: Vec<usize> = with_item.clone().filter(|&b| slot(1, b).1 < fill_end).collect();
    let overlap_ns = fill_end - with_item.map(|b| slot(1, b).1).min().unwrap();
    eprintln!(
        "cuda_graph: {} of {COPY_ITEMS} blocks passed the wait of task B1 before the fill ended; the first passed {overlap_ns} ns before",
        early.len()
    );
    assert!(
        early.len() >= COPY_ITEMS as usize - 1,
        "only {} of {COPY_ITEMS} blocks passed the wait of B1 before the fill ended: the tasks do not overlap",
        early.len()
    );
    assert!(overlap_ns > u64::from(DELAY_NS) / 2, "the overlap is {overlap_ns} ns");
    // Each wait of the plan holds: no block leaves the wait of a task before the last block ends each producer.
    let slot_of = |task: usize, block: usize| {
        let base = (task * blocks + block) * PROFILE_WORDS;
        (profile[base], profile[base + 1], profile[base + 2])
    };
    for (index, task) in planned.iter().enumerate() {
        for wait in &task.waits {
            let producer = wait.event as usize;
            let last_end = (0..blocks).map(|b| slot_of(producer, b).2).max().unwrap();
            // A block with no item of the task does not wait.
            for b in 0..(task.items.max(1) as usize).min(blocks) {
                let after_wait = slot_of(index, b).1;
                assert!(
                    after_wait >= last_end,
                    "block {b} left the wait of task {index} {} ns before task {producer} ended",
                    last_end.saturating_sub(after_wait)
                );
            }
        }
    }

    ring.free().unwrap();
    for ptr in [fill_buf, fill_c, a2, a3, b0, b1, b2, b3, c1, out] {
        dev.free_device(ptr).unwrap();
    }
}
