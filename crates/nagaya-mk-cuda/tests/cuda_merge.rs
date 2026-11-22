//! GPU gate of the merge groups (Milestone 37 step 7, substep 5): `OP_REDUCE_PROBE` splits the sum of a vector over
//! several items. The item that arrives last in its group sums the partials in index order. The bits do not depend
//! on the grid width or on which block merges, and they equal a host sum in the same order.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_merge -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_COPY, OP_REDUCE_PROBE};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

/// The grid widths of the gate. 7 and 24 divide no group size below.
const GRIDS: [u32; 4] = [48, 24, 7, 1];
/// The runs at each grid. The merging block can change from run to run, and the bits must not.
const RUNS: usize = 20;
/// The lanes of the fixed tree of an item. They are the workers of the interpreter, whatever the grid is.
const WORKERS: usize = 256;

/// A shape: `groups` vectors of `n` floats, each one cut into slices of `kt` floats.
struct Shape {
    groups: usize,
    n: usize,
    kt: usize,
}

impl Shape {
    fn slices(&self) -> usize {
        self.n.div_ceil(self.kt)
    }
}

/// The shapes. The groups have 11, 16 and 66 items: no group size is a multiple of a grid width except 16 and 48.
const SHAPES: [Shape; 3] = [
    Shape {
        groups: 5,
        n: 10_000,
        kt: 950,
    },
    Shape {
        groups: 3,
        n: 7_777,
        kt: 500,
    },
    Shape {
        groups: 8,
        n: 100_000,
        kt: 1536,
    },
];

/// Run the ring in a thread with a watchdog, so that a hang fails the test.
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
            eprintln!("cuda_merge: the ring did not finish within 30 s at grid {grid}; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run_grid(grid, threads, shared).unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

/// A float that makes the order of a sum matter: a 11-bit integer times a power of two from 2^-6 to 2^6.
fn value(index: u64) -> f32 {
    let h = (index as u32).wrapping_mul(0x9E37_79B1) ^ ((index >> 32) as u32).wrapping_mul(0x85EB_CA6B);
    let h = h ^ (h >> 15);
    let h = h.wrapping_mul(0x2C1B_3C6D);
    let mantissa = ((h >> 4) % 2001) as f32 - 1000.0;
    mantissa * 2f32.powi(((h >> 20) % 13) as i32 - 6)
}

/// The sum that the op defines: each item sums its slice in `WORKERS` lanes and then in a tree of halves, and the
/// merge adds the partials in index order.
fn host_sum(shape: &Shape, group: usize, input: &[f32]) -> f32 {
    let x = &input[group * shape.n..(group + 1) * shape.n];
    let mut partials = Vec::new();
    for slice in 0..shape.slices() {
        let (begin, end) = (slice * shape.kt, ((slice + 1) * shape.kt).min(shape.n));
        let mut lanes = [0f32; WORKERS];
        for (lane, acc) in lanes.iter_mut().enumerate() {
            for i in (begin + lane..end).step_by(WORKERS) {
                *acc += x[i];
            }
        }
        let mut half = WORKERS / 2;
        while half > 0 {
            for lane in 0..half {
                lanes[lane] += lanes[lane + half];
            }
            half /= 2;
        }
        partials.push(lanes[0]);
    }
    partials[1..].iter().fold(partials[0], |sum, p| sum + p)
}

fn download_f32(dev: &CudaDevice, ptr: CUdeviceptr, count: usize) -> Vec<f32> {
    let mut out = vec![0f32; count];
    unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), ptr, count * 4) }.unwrap();
    out
}

#[test]
fn the_merge_of_k_slices_gives_the_bits_of_the_fixed_order_at_each_grid() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    for shape in &SHAPES {
        let items = shape.groups * shape.slices();
        let input: Vec<f32> = (0..(shape.groups * shape.n) as u64).map(value).collect();
        let want: Vec<f32> = (0..shape.groups).map(|g| host_sum(shape, g, &input)).collect();
        // A sum in plain left-to-right order differs from the fixed order, so the test can tell the two apart.
        let left_to_right: Vec<f32> = (0..shape.groups)
            .map(|g| input[g * shape.n..(g + 1) * shape.n].iter().sum())
            .collect();
        assert_ne!(
            left_to_right.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the input does not make the order of the sum matter"
        );

        let input_bytes = input.len() * 4;
        let src = dev.alloc_device(input_bytes).unwrap();
        unsafe { dev.copy_htod(src, input.as_ptr().cast::<c_void>(), input_bytes) }.unwrap();
        let out = dev.alloc_device(shape.groups * 4).unwrap();
        let copy = dev.alloc_device(shape.groups * 4).unwrap();
        // The scratch holds a float for each item, then a counter for each group.
        let mut ring = Ring::new(&dev, items + shape.groups, false).unwrap();
        let reduce = Task::with_items(MkTaskDescriptor {
            opcode: OP_REDUCE_PROBE,
            items: items as u32,
            a: src,
            c: out,
            f: ring.scratch(),
            n: shape.n as u32,
            kt: shape.kt as u32,
            group_items: shape.slices() as u32,
            ..Default::default()
        })
        .reads(src, input_bytes)
        .writes(out, shape.groups * 4);
        // A task that reads the sums: the signal of the blocks must publish what the merging block wrote.
        let after = Task::with_items(MkTaskDescriptor {
            opcode: OP_COPY,
            items: 1,
            a: copy,
            b: out,
            n: shape.groups as u32,
            ..Default::default()
        })
        .reads(out, shape.groups * 4)
        .writes(copy, shape.groups * 4);
        ring.set(&[reduce, after]);

        // The flat shape of the probes, and the producer shape with the most dynamic shared memory.
        let shapes = [
            (256u32, 0u32),
            (DEFAULT_BLOCK_THREADS, dev.max_dynamic_shared_bytes().unwrap()),
        ];
        for ((threads, shared), &grid) in shapes.into_iter().flat_map(|s| GRIDS.iter().map(move |g| (s, g))) {
            for run in 0..RUNS {
                dev.memset_zero(out, shape.groups * 4).unwrap();
                dev.memset_zero(copy, shape.groups * 4).unwrap();
                // The counters of the groups start at 0.
                ring.zero_device().unwrap();
                run_with_watchdog(&ring, grid, threads, shared);
                for (name, ptr) in [("the merged sums", out), ("the copy of the sums", copy)] {
                    let got = download_f32(&dev, ptr, shape.groups);
                    for (g, (got, want)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{name}: group {g} of {} items at grid {grid}, {threads} threads, run {run}: got {got:e}, wanted {want:e}",
                            shape.slices()
                        );
                    }
                }
            }
        }
        eprintln!(
            "cuda_merge: {} groups of {} items: the bits equal the host at the grids {GRIDS:?}, in both shapes, {RUNS} runs each",
            shape.groups,
            shape.slices()
        );
        ring.free().unwrap();
        for ptr in [src, out, copy] {
            dev.free_device(ptr).unwrap();
        }
    }
}

/// The rule of the merge counters: the item that merges resets the counter of its group. Then a ring needs a host
/// zero only once, at the open, and the next launch finds every counter at 0.
#[test]
fn a_ring_runs_again_with_no_host_zero_in_between() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let shape = &SHAPES[0];
    let items = shape.groups * shape.slices();
    let input: Vec<f32> = (0..(shape.groups * shape.n) as u64).map(value).collect();
    let want: Vec<f32> = (0..shape.groups).map(|g| host_sum(shape, g, &input)).collect();
    let input_bytes = input.len() * 4;
    let src = dev.alloc_device(input_bytes).unwrap();
    unsafe { dev.copy_htod(src, input.as_ptr().cast::<c_void>(), input_bytes) }.unwrap();
    let out = dev.alloc_device(shape.groups * 4).unwrap();
    let mut ring = Ring::new(&dev, items + shape.groups, false).unwrap();
    ring.set(&[Task::with_items(MkTaskDescriptor {
        opcode: OP_REDUCE_PROBE,
        items: items as u32,
        a: src,
        c: out,
        f: ring.scratch(),
        n: shape.n as u32,
        kt: shape.kt as u32,
        group_items: shape.slices() as u32,
        ..Default::default()
    })
    .reads(src, input_bytes)
    .writes(out, shape.groups * 4)]);
    // The only zero of the counters.
    ring.zero_device().unwrap();
    for (run, grid) in [48u32, 24, 7, 1, 48, 24, 7, 1, 48, 5].into_iter().enumerate() {
        dev.memset_zero(out, shape.groups * 4).unwrap();
        run_with_watchdog(
            &ring,
            grid,
            DEFAULT_BLOCK_THREADS,
            dev.max_dynamic_shared_bytes().unwrap(),
        );
        let got = download_f32(&dev, out, shape.groups);
        for (g, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "group {g} at grid {grid}, run {run}: got {got:e}, wanted {want:e}"
            );
        }
        // The scratch holds a float for each item, then a counter for each group. Each counter is back at 0.
        let mut scratch = vec![0u32; items + shape.groups];
        unsafe { dev.copy_dtoh(scratch.as_mut_ptr().cast::<c_void>(), ring.scratch(), scratch.len() * 4) }.unwrap();
        assert_eq!(
            &scratch[items..],
            vec![0u32; shape.groups],
            "the counters after run {run} at grid {grid}"
        );
    }
    ring.free().unwrap();
    for ptr in [src, out] {
        dev.free_device(ptr).unwrap();
    }
}
