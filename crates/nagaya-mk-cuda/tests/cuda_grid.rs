//! GPU gate of the work items and the static assignment (Milestone 37 step 7, substep 4): the bits of a ring do not
//! depend on the grid width, and each event counts the items of its task.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_grid -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::graph::signals;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_COPY, OP_SELFTEST_CHECK, OP_SELFTEST_FILL};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

/// The grid widths of the gate. 48 is every SM of the GB10, and 7 does not divide any count of items.
const GRIDS: [u32; 4] = [48, 24, 7, 1];
/// Floats in each buffer.
const FLOATS: usize = 100_000;
const BYTES: usize = FLOATS * 4;

/// Run the ring in a thread with a watchdog, so that a hang fails the test.
fn run_with_watchdog(ring: &Ring<'_>, grid: u32) {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..300 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if done.load(Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_grid: the ring did not finish within 30 s at grid {grid}; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run_grid(grid, 256, 0).unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

/// A copy of `items` items of `per_item` floats each. The task covers `items * per_item` floats.
fn copy(dst: CUdeviceptr, src: CUdeviceptr, items: u32, per_item: u32) -> Task {
    let bytes = items as usize * per_item as usize * 4;
    Task::with_items(MkTaskDescriptor {
        opcode: OP_COPY,
        items,
        a: dst,
        b: src,
        n: per_item,
        ..Default::default()
    })
    .reads(src, bytes)
    .writes(dst, bytes)
}

fn download(dev: &CudaDevice, ptr: CUdeviceptr, floats: usize) -> Vec<u32> {
    let mut out = vec![0u32; floats];
    unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), ptr, floats * 4) }.unwrap();
    out
}

fn pattern(seed: u32) -> Vec<u32> {
    (0..FLOATS as u32)
        .map(|i| seed.wrapping_mul(0x9E37_79B1) ^ i.wrapping_mul(0x85EB_CA6B))
        .collect()
}

#[test]
fn the_bits_of_a_ring_do_not_depend_on_the_grid_and_each_event_counts_its_items() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let mut ring = Ring::new(&dev, 1024, false).unwrap();
    assert!(ring.grid_blocks() >= GRIDS[0]);
    let alloc = |data: Option<&[u32]>| {
        let ptr = dev.alloc_device(BYTES).unwrap();
        match data {
            Some(words) => unsafe { dev.copy_htod(ptr, words.as_ptr().cast::<c_void>(), BYTES) }.unwrap(),
            None => dev.memset_zero(ptr, BYTES).unwrap(),
        }
        ptr
    };
    let (src_a, src_b) = (pattern(1), pattern(2));
    let (sa, sb) = (alloc(Some(&src_a)), alloc(Some(&src_b)));
    let [x, y, z, w, out] = [(); 5].map(|()| alloc(None));
    // A per-block fill and check give the count of wrong words of each block.
    let words_per_block = 512usize;
    let fill_bytes = GRIDS[0] as usize * words_per_block * 4;
    let fill_buf = dev.alloc_device(fill_bytes).unwrap();
    let wrong = dev.alloc_device(GRIDS[0] as usize * 4).unwrap();

    // The items of two tasks cut the same buffer in different places: 100 items of 1,000 floats, then 125 of 800.
    // A dependent task reads words that several items of its producer wrote.
    let tasks = vec![
        copy(x, sa, 100, 1000),         // 0: x = a, in 100 items
        copy(y, x, 125, 800),           // 1: y = x, in 125 items
        copy(z, sb, 37, 2700).after(0), // 2: z = b (the first 99,900 floats), 37 items; it follows task 0
        copy(w, y, 3, 33_333),          // 3: w = y (the first 99,999 floats), 3 items, fewer than the grid
        Task::per_block(MkTaskDescriptor {
            opcode: OP_SELFTEST_FILL,
            a: fill_buf,
            n: words_per_block as u32,
            group_items: 3,
            ..Default::default()
        })
        .writes(fill_buf, fill_bytes), // 4: one item in each block
        Task::per_block(MkTaskDescriptor {
            opcode: OP_SELFTEST_CHECK,
            a: fill_buf,
            d: wrong,
            n: words_per_block as u32,
            group_items: 3,
            ..Default::default()
        })
        .reads(fill_buf, fill_bytes)
        .writes(wrong, GRIDS[0] as usize * 4), // 5: one item in each block
        // The end: out = w, after the three chains. z and the check have no reader, so the end reads them too.
        copy(out, w, 3, 33_333)
            .reads(z, BYTES)
            .reads(wrong, GRIDS[0] as usize * 4), // 6
    ];
    ring.set(&tasks);

    let mut results: Vec<(u32, Vec<Vec<u32>>)> = Vec::new();
    for &grid in &GRIDS {
        for ptr in [x, y, z, w, out] {
            dev.memset_zero(ptr, BYTES).unwrap();
        }
        dev.memset_zero(wrong, GRIDS[0] as usize * 4).unwrap();
        dev.memset_zero(fill_buf, fill_bytes).unwrap();
        run_with_watchdog(&ring, grid);

        // Each event counts the items of its task, and a per-block task has one item for each block.
        let planned = ring.planned(grid);
        let scoreboard = ring.read_scoreboard().unwrap();
        for (index, task) in planned.iter().enumerate() {
            let want = match task.origin.map(|i| &tasks[i]) {
                Some(t) if t.per_block => grid,
                Some(t) => t.desc.items,
                None => 1,
            };
            assert_eq!(scoreboard[index], want, "grid {grid}: event {index}");
            assert_eq!(want, signals(task.items), "grid {grid}: event {index} against the plan");
        }
        // The stats: each block walked every task.
        let stats = ring.read_stats().unwrap();
        assert_eq!(stats.len(), grid as usize);
        assert!(
            stats.iter().all(|s| s.tasks_run as usize == planned.len()),
            "grid {grid}: {stats:?}"
        );

        let counts = download(&dev, wrong, grid as usize);
        assert!(counts.iter().all(|&c| c == 0), "grid {grid}: wrong words {counts:?}");
        results.push((grid, [x, y, z, w, out].map(|p| download(&dev, p, FLOATS)).to_vec()));
    }

    // The values: x and y copy a, z copies b up to 99,900 floats, w copies y up to 99,999, out copies w.
    let (first_grid, first) = &results[0];
    assert_eq!(first[0], src_a, "x");
    assert_eq!(first[1], src_a, "y");
    assert_eq!(&first[2][..99_900], &src_b[..99_900], "z");
    assert!(first[2][99_900..].iter().all(|&v| v == 0), "z writes only its items");
    assert_eq!(&first[3][..99_999], &src_a[..99_999], "w");
    assert_eq!(&first[4][..99_999], &src_a[..99_999], "out");
    for (grid, got) in &results[1..] {
        for (name, (a, b)) in ["x", "y", "z", "w", "out"].iter().zip(first.iter().zip(got)) {
            assert_eq!(a, b, "{name} differs between the grids {first_grid} and {grid}");
        }
    }
    eprintln!("cuda_grid: the bits of 5 buffers are equal at the grids {GRIDS:?}");

    ring.free().unwrap();
    for ptr in [sa, sb, x, y, z, w, out, fill_buf, wrong] {
        dev.free_device(ptr).unwrap();
    }
}

#[test]
fn a_grid_outside_the_range_of_the_ring_is_refused() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let mut ring = Ring::new(&dev, 1024, false).unwrap();
    ring.set(&[Task::per_block(MkTaskDescriptor::default())]);
    let capacity = ring.grid_blocks();
    for grid in [0, capacity + 1] {
        let error = unsafe { ring.run_grid(grid, 256, 0) }.expect_err("the grid is refused");
        assert!(error.contains(&format!("a grid of {grid} blocks")), "{error}");
        assert!(error.contains(&format!("from 1 to {capacity}")), "{error}");
    }
    unsafe { ring.run_grid(1, 256, 0).unwrap() };
    unsafe { ring.run_grid(capacity, 256, 0).unwrap() };
}
