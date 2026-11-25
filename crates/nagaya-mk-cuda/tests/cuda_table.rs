//! GPU gate of the task table: `d` of the descriptor points to a table of 16 words.
//! The producer warp copies it to shared memory with the descriptor. `OP_SELFTEST_TABLE` writes the words that its
//! item sees. Each task has its own table, and the tasks do not wait on one another, so the producer runs ahead of
//! the consumers: a table that the consumers read too early, or a table of another task, shows.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_table -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, MkTaskTable, OP_SELFTEST_TABLE, TABLE_POINTERS};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

const GRIDS: [u32; 4] = [48, 24, 7, 1];
/// The tasks of the ring. A task has no wait, so the producer fetches up to two tasks ahead.
const TASKS: usize = 24;
/// The items of a task: more than the blocks of the smallest grids, and fewer than the largest.
const ITEMS: usize = 40;
const RUNS: usize = 20;

/// A word of the table of task `task`: it is never 0, and no two words of the test are equal.
fn word(task: usize, slot: usize) -> u64 {
    let x = ((task as u64 + 1) << 32) | (slot as u64 + 1);
    x.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
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
            eprintln!("cuda_table: the ring did not finish within 30 s at grid {grid}; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run_grid(grid, threads, shared).unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

fn upload<T: Copy>(dev: &CudaDevice, data: &[T]) -> CUdeviceptr {
    let bytes = std::mem::size_of_val(data);
    let ptr = dev.alloc_device(bytes).unwrap();
    unsafe { dev.copy_htod(ptr, data.as_ptr().cast::<c_void>(), bytes) }.unwrap();
    ptr
}

#[test]
fn the_producer_copy_of_the_task_table_gives_each_item_the_table_of_its_task() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    // One array of tables. The address of each table is a multiple of 128.
    let tables: Vec<MkTaskTable> = (0..TASKS)
        .map(|t| MkTaskTable {
            ptr: std::array::from_fn(|s| word(t, s)),
        })
        .collect();
    let table_ptr = upload(&dev, &tables);
    let words = TASKS * ITEMS * TABLE_POINTERS;
    let out = dev.alloc_device(words * 8).unwrap();
    let mut ring = Ring::new(&dev, 1, false).unwrap();
    let tasks: Vec<Task> = (0..TASKS)
        .map(|t| {
            let c = out + (t * ITEMS * TABLE_POINTERS * 8) as u64;
            Task::with_items(MkTaskDescriptor {
                opcode: OP_SELFTEST_TABLE,
                items: ITEMS as u32,
                c,
                d: table_ptr + (t * std::mem::size_of::<MkTaskTable>()) as u64,
                ..Default::default()
            })
            .writes(c, ITEMS * TABLE_POINTERS * 8)
        })
        .collect();
    ring.set(&tasks);
    ring.zero_device().unwrap();
    // The producer shape, and the flat shape of 256 threads, where every thread reads the table from global memory.
    let shapes = [
        (DEFAULT_BLOCK_THREADS, dev.max_dynamic_shared_bytes().unwrap()),
        (256u32, 0u32),
    ];
    for (threads, shared) in shapes {
        for grid in GRIDS {
            for run in 0..RUNS {
                dev.memset_zero(out, words * 8).unwrap();
                run_with_watchdog(&ring, grid, threads, shared);
                let mut got = vec![0u64; words];
                unsafe { dev.copy_dtoh(got.as_mut_ptr().cast::<c_void>(), out, words * 8) }.unwrap();
                for t in 0..TASKS {
                    for item in 0..ITEMS {
                        for s in 0..TABLE_POINTERS {
                            let at = (t * ITEMS + item) * TABLE_POINTERS + s;
                            assert_eq!(
                                got[at],
                                word(t, s),
                                "task {t}, item {item}, slot {s} at grid {grid}, {threads} threads, run {run}"
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!("cuda_table: {TASKS} tasks of {ITEMS} items read their own table at {GRIDS:?} blocks, in both shapes");
    ring.free().unwrap();
    for ptr in [table_ptr, out] {
        dev.free_device(ptr).unwrap();
    }
}
