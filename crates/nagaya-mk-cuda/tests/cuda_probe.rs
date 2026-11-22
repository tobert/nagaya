//! GPU probes (Milestone 37 step 7, substep 1): measure the facts of the GB10 that the design of the interpreter
//! assumes, and print them as tables. Each probe op returns a checksum or a result tile, and the test compares it
//! with the host, so a wrong result fails the test. Read the tables with `--nocapture`.
//!
//! Run with one thread and a free GPU PCI address:
//! `NAGAYA_TEST_CARDS=0000:0a:00.0 cargo test -p nagaya-mk-cuda --features cuda --test cuda_probe -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::time::Instant;

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice, REPORTED_ATTRIBUTES};
use nagaya_mk_cuda::protocol::{
    CONSUMER_THREADS, MkTaskDescriptor, OP_NOP, OP_PROBE_L2, OP_PROBE_MMA_RATE, OP_PROBE_MMA_TILE, OP_PROBE_READ,
    OP_PROBE_TIMER, PROBE_ITEM_BYTES, PROBE_MMA_BF16, PROBE_MMA_CHAINS, PROBE_MMA_MXFP8, PROBE_MMA_NVFP4,
    PROBE_READ_NC, PROBE_READ_PLAIN, PROBE_READ_PREFETCH_BULK, PROBE_READ_PREFETCH_LINE, PROBE_READ_SMEM_BULK,
    PRODUCER_BLOCK_THREADS, PROFILE_WORDS, probe_smem_bytes,
};
use nagaya_mk_cuda::ring::{Ring, chained};
use nagaya_mk_cuda::{INKLING2_CUBIN, INKLING2_PROFILE_CUBIN, test_cards};

fn open() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU with the profile cubin")
}

fn upload<T: Copy>(dev: &CudaDevice, data: &[T]) -> CUdeviceptr {
    let bytes = std::mem::size_of_val(data);
    let ptr = dev.alloc_device(bytes).expect("allocate");
    unsafe { dev.copy_htod(ptr, data.as_ptr().cast::<c_void>(), bytes) }.expect("upload");
    ptr
}

fn download<T: Copy + Default>(dev: &CudaDevice, ptr: CUdeviceptr, count: usize) -> Vec<T> {
    let mut out = vec![T::default(); count];
    unsafe {
        dev.copy_dtoh(
            out.as_mut_ptr().cast::<c_void>(),
            ptr,
            std::mem::size_of_val(out.as_slice()),
        )
    }
    .expect("download");
    out
}

/// Nanoseconds from the first block that passes the wait of task `task` to the last block that ends it.
fn task_ns(profile: &[u64], blocks: usize, task: usize) -> u64 {
    let slots = &profile[task * blocks * PROFILE_WORDS..(task + 1) * blocks * PROFILE_WORDS];
    let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
    let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
    assert!(end > begin, "task {task} has no time: begin {begin}, end {end}");
    end - begin
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// The value of word `index` of the pattern in the probe buffers. It differs from word to word.
fn pattern(index: u64) -> u32 {
    let x = (index as u32).wrapping_mul(0x9E37_79B1);
    x ^ (x >> 15) ^ ((index >> 32) as u32)
}

fn gb_per_s(bytes: u64, ns: u64) -> f64 {
    bytes as f64 / ns as f64
}

// ---------------------------------------------------------------------------------------------------------------
// 1. Device attributes
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn probe_device_attributes() {
    let dev = open();
    eprintln!(
        "probe attributes: driver {}, SMs {}",
        dev.driver_version(),
        dev.sm_count()
    );
    let mut values = std::collections::HashMap::new();
    for (name, code) in REPORTED_ATTRIBUTES {
        let value = dev.attribute(*code).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("probe attributes: {name:<40} {value}");
        values.insert(*name, value);
    }
    assert_eq!(values["MULTIPROCESSOR_COUNT"] as u32, dev.sm_count());
    assert_eq!(values["WARP_SIZE"], 32);
    assert_eq!(values["COOPERATIVE_LAUNCH"], 1);
    assert!(values["MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"] >= values["MAX_SHARED_MEMORY_PER_BLOCK"]);
    assert!(values["MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"] >= 48 * 1024);
}

#[test]
fn a_launch_with_more_shared_memory_than_the_gpu_allows_is_refused() {
    let dev = open();
    let optin = dev.max_dynamic_shared_bytes().unwrap();
    eprintln!("probe attributes: most dynamic shared memory of a block of the kernel {optin}");
    let mut ring = Ring::new(&dev, 1024, false).unwrap();
    ring.set(&chained(&[MkTaskDescriptor {
        opcode: OP_NOP,
        ..Default::default()
    }]));
    let error = unsafe { ring.run_shaped(256, optin + 1) }.expect_err("too much shared memory is an error");
    assert!(
        error.contains(&(optin + 1).to_string()) && error.contains(&optin.to_string()),
        "{error}"
    );
    assert!(error.contains("Ask for"), "{error}");
    // The largest size that the GPU allows works.
    unsafe { ring.run_shaped(256, optin).unwrap() };
    ring.free().unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// 2. Peak read bandwidth
// ---------------------------------------------------------------------------------------------------------------

const READ_BYTES: usize = 1 << 30;
const READ_TASKS: usize = 5;

struct ReadCase {
    name: String,
    variant: u16,
    param: u32,
    shared_bytes: u32,
}

#[test]
fn probe_read_bandwidth() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    let items = READ_BYTES / PROBE_ITEM_BYTES;
    // The buffer, and the sum of the words of each item.
    let words = READ_BYTES / 4;
    let host: Vec<u32> = (0..words as u64).map(pattern).collect();
    let item_words = PROBE_ITEM_BYTES / 4;
    let item_sums: Vec<u32> = host
        .chunks(item_words)
        .map(|c| c.iter().fold(0u32, |s, w| s.wrapping_add(*w)))
        .collect();
    let expected: Vec<u32> = (0..blocks)
        .map(|b| {
            (b..items)
                .step_by(blocks)
                .fold(0u32, |s, i| s.wrapping_add(item_sums[i]))
        })
        .collect();
    let buffer = upload(&dev, &host);
    drop(host);

    let optin = dev.max_dynamic_shared_bytes().unwrap();
    let mut cases = vec![
        ReadCase {
            name: "plain ld.global.v4".into(),
            variant: PROBE_READ_PLAIN,
            param: 0,
            shared_bytes: 0,
        },
        ReadCase {
            name: "ld.global.nc.L1::no_allocate.v4".into(),
            variant: PROBE_READ_NC,
            param: 0,
            shared_bytes: 0,
        },
    ];
    for distance in [1, 2, 4] {
        cases.push(ReadCase {
            name: format!("nc + prefetch.global.L2, {distance} ahead"),
            variant: PROBE_READ_PREFETCH_LINE,
            param: distance,
            shared_bytes: 0,
        });
        cases.push(ReadCase {
            name: format!("nc + cp.async.bulk.prefetch.L2, {distance} ahead"),
            variant: PROBE_READ_PREFETCH_BULK,
            param: distance,
            shared_bytes: 0,
        });
    }
    for stages in [2usize, 4, 6] {
        let shared = probe_smem_bytes(stages) as u32;
        if shared <= optin {
            cases.push(ReadCase {
                name: format!("cp.async.bulk into shared, {stages} stages of 16 KB"),
                variant: PROBE_READ_SMEM_BULK,
                param: stages as u32,
                shared_bytes: shared,
            });
        }
    }

    eprintln!("probe read: 1 GiB buffer, items of 32 KB, block b reads items b, b+G, ...; G = {blocks} blocks");
    eprintln!(
        "probe read: GB/s is 1e9 bytes per second; timer = median of %globaltimer spans of {} tasks, wall = host",
        READ_TASKS - 1
    );
    for case in &cases {
        for threads in [256u32, 288] {
            let checks = dev.alloc_device(READ_TASKS * blocks * 4).unwrap();
            dev.memset_zero(checks, READ_TASKS * blocks * 4).unwrap();
            let mut ring = Ring::new(&dev, 1024, true).unwrap();
            ring.zero_device().unwrap();
            let tasks: Vec<MkTaskDescriptor> = (0..READ_TASKS)
                .map(|t| MkTaskDescriptor {
                    opcode: OP_PROBE_READ,
                    a: buffer,
                    c: checks + (t * blocks * 4) as u64,
                    n: items as u32,
                    k: u32::from(case.variant),
                    kt: case.param,
                    ..Default::default()
                })
                .collect();
            ring.set(&chained(&tasks));
            let start = Instant::now();
            unsafe { ring.run_shaped(threads, case.shared_bytes).unwrap() };
            let wall = start.elapsed().as_nanos() as u64;
            let profile = ring.read_profile().unwrap();
            let spans: Vec<u64> = (1..READ_TASKS).map(|t| task_ns(&profile, blocks, t)).collect();
            let got: Vec<u32> = download(&dev, checks, READ_TASKS * blocks);
            for t in 0..READ_TASKS {
                assert_eq!(
                    &got[t * blocks..(t + 1) * blocks],
                    &expected[..],
                    "{} with {threads} threads: the checksums of task {t} differ from the host",
                    case.name
                );
            }
            eprintln!(
                "probe read: {:<52} {threads:>3} threads: timer median {:>6.1} GB/s, best {:>6.1} GB/s, wall {:>6.1} GB/s",
                case.name,
                gb_per_s(READ_BYTES as u64, median(spans.clone())),
                gb_per_s(READ_BYTES as u64, *spans.iter().min().unwrap()),
                gb_per_s((READ_BYTES * READ_TASKS) as u64, wall),
            );
            ring.free().unwrap();
            dev.free_device(checks).unwrap();
        }
    }
    dev.free_device(buffer).unwrap();
}

/// The producer shape reads at the peak (Milestone 37 step 7, substep 6). The gate is 85% of the best read rate of the
/// flat shape, which is the measured peak. The producer warp does not read. In the last case it prefetches the
/// first item of each block into L2 before the consumers start.
#[test]
fn probe_read_bandwidth_in_the_producer_shape() {
    const GATE: f64 = 0.85;
    let dev = open();
    let blocks = dev.sm_count() as usize;
    let items = READ_BYTES / PROBE_ITEM_BYTES;
    let host: Vec<u32> = (0..(READ_BYTES / 4) as u64).map(pattern).collect();
    let item_sums: Vec<u32> = host
        .chunks(PROBE_ITEM_BYTES / 4)
        .map(|c| c.iter().fold(0u32, |s, w| s.wrapping_add(*w)))
        .collect();
    let expected: Vec<u32> = (0..blocks)
        .map(|b| {
            (b..items)
                .step_by(blocks)
                .fold(0u32, |s, i| s.wrapping_add(item_sums[i]))
        })
        .collect();
    let buffer = upload(&dev, &host);
    drop(host);
    let max_shared = dev.max_dynamic_shared_bytes().unwrap();

    // (name, threads, shared bytes, prefetch the first item of each block)
    let cases: [(&str, u32, u32, bool); 5] = [
        ("flat, 256 threads, nc", 256, 0, false),
        ("flat, 128 threads, nc", 128, 0, false),
        ("producer shape, no shared memory", PRODUCER_BLOCK_THREADS, 0, false),
        (
            "producer shape, most shared memory",
            PRODUCER_BLOCK_THREADS,
            max_shared,
            false,
        ),
        (
            "producer shape, most shared memory, first item prefetched",
            PRODUCER_BLOCK_THREADS,
            max_shared,
            true,
        ),
    ];
    let mut rates: Vec<(f64, bool)> = Vec::new();
    for (name, threads, shared, prefetch) in cases {
        let checks = dev.alloc_device(READ_TASKS * blocks * 4).unwrap();
        dev.memset_zero(checks, READ_TASKS * blocks * 4).unwrap();
        let mut ring = Ring::new(&dev, 1024, true).unwrap();
        ring.zero_device().unwrap();
        let tasks: Vec<MkTaskDescriptor> = (0..READ_TASKS)
            .map(|t| MkTaskDescriptor {
                opcode: OP_PROBE_READ,
                a: buffer,
                b: buffer,
                c: checks + (t * blocks * 4) as u64,
                n: items as u32,
                k: u32::from(PROBE_READ_NC),
                item_bytes: if prefetch { PROBE_ITEM_BYTES as u32 } else { 0 },
                ..Default::default()
            })
            .collect();
        ring.set(&chained(&tasks));
        unsafe { ring.run_shaped(threads, shared).unwrap() };
        let profile = ring.read_profile().unwrap();
        let spans: Vec<u64> = (1..READ_TASKS).map(|t| task_ns(&profile, blocks, t)).collect();
        let got: Vec<u32> = download(&dev, checks, READ_TASKS * blocks);
        for t in 0..READ_TASKS {
            assert_eq!(
                &got[t * blocks..(t + 1) * blocks],
                &expected[..],
                "{name}: the checksums of task {t}"
            );
        }
        let rate = gb_per_s(READ_BYTES as u64, median(spans));
        eprintln!("probe read, producer shape: {name:<58} {rate:>6.1} GB/s");
        rates.push((rate, threads == PRODUCER_BLOCK_THREADS));
        ring.free().unwrap();
        dev.free_device(checks).unwrap();
    }
    let peak = rates.iter().filter(|r| !r.1).map(|r| r.0).fold(0.0, f64::max);
    for &(rate, producer) in rates.iter().filter(|r| r.1) {
        eprintln!(
            "probe read, producer shape: {:.1}% of the flat peak {peak:.1} GB/s",
            rate / peak * 100.0
        );
        assert!(
            producer && rate >= GATE * peak,
            "the producer shape reads {rate:.1} GB/s, under {:.0}% of the peak {peak:.1} GB/s",
            GATE * 100.0
        );
    }
    dev.free_device(buffer).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// 3. The knee of L2
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn probe_l2_knee() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    const MAX_FOOTPRINT: usize = 64 << 20;
    let host: Vec<u32> = (0..(MAX_FOOTPRINT / 4) as u64).map(pattern).collect();
    let buffer = upload(&dev, &host);
    let l2 = dev.attribute(38).unwrap();
    eprintln!(
        "probe l2: L2 size reported {} MiB; each block re-reads its own slice; about 8 GiB read in each task",
        l2 >> 20
    );
    for threads in [128u32, 256] {
        let mut footprint = 1usize << 20;
        while footprint <= MAX_FOOTPRINT {
            // The slice of a block is a multiple of 16 bytes. The footprint is the slices together.
            let slice = (footprint / blocks) & !15;
            let effective = slice * blocks;
            let passes = ((8usize << 30) / effective).max(2) as u32;
            let expected: Vec<u32> = (0..blocks)
                .map(|b| {
                    let words = &host[b * slice / 4..(b + 1) * slice / 4];
                    words.iter().fold(0u32, |s, w| s.wrapping_add(*w)).wrapping_mul(passes)
                })
                .collect();
            let checks = dev.alloc_device(2 * blocks * 4).unwrap();
            dev.memset_zero(checks, 2 * blocks * 4).unwrap();
            let mut ring = Ring::new(&dev, 1024, true).unwrap();
            ring.zero_device().unwrap();
            let tasks: Vec<MkTaskDescriptor> = (0..2)
                .map(|t| MkTaskDescriptor {
                    opcode: OP_PROBE_L2,
                    a: buffer,
                    c: checks + (t * blocks * 4) as u64,
                    n: slice as u32,
                    k: passes,
                    ..Default::default()
                })
                .collect();
            ring.set(&chained(&tasks));
            unsafe { ring.run_shaped(threads, 0).unwrap() };
            let profile = ring.read_profile().unwrap();
            let got: Vec<u32> = download(&dev, checks, 2 * blocks);
            for t in 0..2 {
                assert_eq!(
                    &got[t * blocks..(t + 1) * blocks],
                    &expected[..],
                    "L2 checksums of task {t}, footprint {effective}"
                );
            }
            let ns = task_ns(&profile, blocks, 1);
            eprintln!(
                "probe l2: {threads:>3} threads, footprint {:>8.3} MiB ({passes:>5} passes): {:>7.1} GB/s",
                effective as f64 / (1 << 20) as f64,
                gb_per_s(effective as u64 * u64::from(passes), ns)
            );
            ring.free().unwrap();
            dev.free_device(checks).unwrap();
            footprint *= 2;
        }
    }
    dev.free_device(buffer).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// 4. The cost of a task boundary
// ---------------------------------------------------------------------------------------------------------------

/// The cost of a task boundary, in the flat shape and in the producer shape. A strict chain of no-op tasks has each
/// task wait for the whole task before it, so the period of the chain is the cost of one boundary.
#[test]
fn probe_task_boundary() {
    const TASKS: usize = 1000;
    let shapes = [
        ("flat, 256 threads", 256u32),
        ("producer shape, 288 threads", PRODUCER_BLOCK_THREADS),
    ];
    let nops = vec![
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        };
        TASKS
    ];
    let dev = open();
    let blocks = dev.sm_count() as usize;
    for (name, threads) in shapes {
        let mut ring = Ring::new(&dev, 1024, true).unwrap();
        ring.zero_device().unwrap();
        ring.set(&chained(&nops));
        let mut walls = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            unsafe { ring.run_shaped(threads, 0).unwrap() };
            walls.push(start.elapsed().as_nanos() as u64);
        }
        // The period of the chain: the time from the end of the last block of task k - 1 to that of task k.
        let profile = ring.read_profile().unwrap();
        let ends: Vec<u64> = (0..TASKS)
            .map(|t| {
                profile[t * blocks * PROFILE_WORDS..(t + 1) * blocks * PROFILE_WORDS]
                    .chunks(PROFILE_WORDS)
                    .map(|s| s[2])
                    .max()
                    .unwrap()
            })
            .collect();
        let periods: Vec<u64> = ends.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = periods.iter().sum::<u64>() as f64 / periods.len() as f64;
        let stats = ring.read_stats().unwrap();
        let tasks_run: Vec<u32> = stats.iter().map(|s| s.tasks_run).collect();
        assert!(
            tasks_run.iter().all(|&n| n as usize == TASKS),
            "{name}: each block runs {TASKS} tasks: {tasks_run:?}"
        );
        let mean_wait = stats.iter().map(|s| s.wait_ns).sum::<u64>() as f64 / blocks as f64 / TASKS as f64;
        eprintln!("probe boundary, {name}: ring of {TASKS} no-op tasks, strict chain, {blocks} blocks");
        eprintln!(
            "probe boundary, {name}: profile period, mean {mean:.0} ns, median {} ns, min {} ns, max {} ns",
            median(periods.clone()),
            periods.iter().min().unwrap(),
            periods.iter().max().unwrap()
        );
        eprintln!("probe boundary, {name}: mean wait of a block per task {mean_wait:.0} ns (stats)");
        eprintln!(
            "probe boundary, {name}: host wall time of the whole run, min of 5: {} us (includes the upload and the launch)",
            walls.iter().min().unwrap() / 1000
        );
        assert!(mean > 0.0 && mean < 1.0e6, "a boundary of {mean} ns is not credible");
        ring.free().unwrap();
    }
    drop(dev);

    // The cubin with no profile writes no timestamps, so the host wall time of two rings gives the boundary that an
    // engine sees: the difference of the two times, over the difference of the task counts.
    let plain =
        CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU with the plain cubin");
    for (name, threads) in shapes {
        let wall_min = |tasks: usize| -> u64 {
            let mut ring = Ring::new(&plain, 1024, false).unwrap();
            ring.zero_device().unwrap();
            ring.set(&chained(&vec![
                MkTaskDescriptor {
                    opcode: OP_NOP,
                    ..Default::default()
                };
                tasks
            ]));
            let best = (0..20)
                .map(|_| {
                    let start = Instant::now();
                    unsafe { ring.run_shaped(threads, 0).unwrap() };
                    start.elapsed().as_nanos() as u64
                })
                .min()
                .unwrap();
            ring.free().unwrap();
            best
        };
        let (long, short) = (wall_min(TASKS), wall_min(10));
        let boundary = (long - short) as f64 / (TASKS - 10) as f64;
        eprintln!(
            "probe boundary, {name}: cubin with no profile: wall {} us for {TASKS} tasks, {} us for 10 tasks (min of 20): {boundary:.0} ns for each boundary",
            long / 1000,
            short / 1000
        );
        assert!(
            boundary > 0.0 && boundary < 1.0e6,
            "a boundary of {boundary} ns is not credible"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 5. The resolution of %globaltimer
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn probe_globaltimer_resolution() {
    let dev = open();
    const READS: u32 = 1_000_000;
    let out = dev.alloc_device(32).unwrap();
    dev.memset_zero(out, 32).unwrap();
    let mut ring = Ring::new(&dev, 1024, false).unwrap();
    ring.zero_device().unwrap();
    ring.set(&chained(&[MkTaskDescriptor {
        opcode: OP_PROBE_TIMER,
        c: out,
        n: READS,
        ..Default::default()
    }]));
    unsafe { ring.run().unwrap() };
    let words: Vec<u64> = download(&dev, out, 4);
    eprintln!(
        "probe timer: {} back-to-back reads of %globaltimer: smallest nonzero step {} ns, {} steps in {} ns, mean {:.1} ns for each step",
        words[3],
        words[0],
        words[1],
        words[2],
        words[2] as f64 / words[1].max(1) as f64
    );
    assert_eq!(words[3], u64::from(READS));
    assert!(words[0] > 0 && words[1] > 0, "the timer never changed: {words:?}");
    ring.free().unwrap();
    dev.free_device(out).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// 6. MMA tiles
// ---------------------------------------------------------------------------------------------------------------

/// A small deterministic generator, so that a failure repeats.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

fn e2m1(code: u8) -> f64 {
    const MAGNITUDE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = MAGNITUDE[(code & 7) as usize];
    if code & 8 != 0 { -value } else { value }
}

fn ue4m3(code: u8) -> f64 {
    let (exponent, mantissa) = (i32::from(code >> 3) & 15, f64::from(code & 7));
    if exponent == 0 {
        mantissa / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    }
}

fn e4m3(code: u8) -> f64 {
    let value = ue4m3(code & 0x7f);
    if code & 0x80 != 0 { -value } else { value }
}

fn ue8m0(code: u8) -> f64 {
    2f64.powi(i32::from(code) - 127)
}

fn bf16_bits(value: f32) -> u16 {
    assert_eq!(value.to_bits() & 0xffff, 0, "{value} is not exact in bf16");
    (value.to_bits() >> 16) as u16
}

fn bf16_value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// The operands of one tile, in the layout of `kernels/probe_ops.cuh`, and the exact result on the host.
struct Tile {
    a: Vec<u8>,
    b: Vec<u8>,
    scales: Vec<u8>,
    expected: Vec<f64>,
    /// The sum of the absolute values of the terms of each result, to scale an error.
    magnitude: Vec<f64>,
}

/// `wide` picks values over a wide range: the result is then not exact in `f32`.
fn make_tile(kind: u16, seed: u32, wide: bool) -> Tile {
    let mut rng = Rng(seed);
    let mut a = vec![0u8; 16 * 32];
    let mut b = vec![0u8; 8 * 32];
    let mut scales = vec![0u8; 24 * 4];
    let (k, scale_bytes) = match kind {
        PROBE_MMA_BF16 => (16, 0),
        PROBE_MMA_NVFP4 => (64, 4),
        _ => (32, 1),
    };
    let fill_scales = |scales: &mut [u8], rng: &mut Rng| {
        for byte in scales.iter_mut().take(24 * scale_bytes) {
            *byte = match (kind, wide) {
                (PROBE_MMA_NVFP4, false) => 0x30 + (rng.next() % 0x10) as u8,
                (PROBE_MMA_NVFP4, true) => 0x10 + (rng.next() % 0x60) as u8,
                (_, false) => 120 + (rng.next() % 14) as u8,
                (_, true) => 100 + (rng.next() % 50) as u8,
            };
        }
    };
    match kind {
        PROBE_MMA_BF16 => {
            // Multiples of 1/4 from -2 to 2: exact in bf16, and the sums are exact in f32. The wide form uses
            // multiples of 2^-12 up to 2^8 in A, which is still exact in bf16 and has sums that need more bits.
            for chunk in a.chunks_mut(2).chain(b.chunks_mut(2)) {
                let v = if wide {
                    ((rng.next() % 513) as f32 - 256.0) * 2f32.powi(4 - (rng.next() % 12) as i32)
                } else {
                    ((rng.next() % 17) as f32 - 8.0) / 4.0
                };
                let bits = bf16_bits(v);
                chunk.copy_from_slice(&bits.to_le_bytes());
            }
        }
        PROBE_MMA_NVFP4 => {
            for byte in a.iter_mut().chain(b.iter_mut()) {
                *byte = (rng.next() % 256) as u8;
            }
            fill_scales(&mut scales, &mut rng);
        }
        _ => {
            for byte in a.iter_mut().chain(b.iter_mut()) {
                let magnitude = if wide {
                    rng.next() % 0x78
                } else {
                    0x38 + rng.next() % 0x20
                } as u8;
                *byte = magnitude | (((rng.next() & 1) as u8) << 7);
            }
            fill_scales(&mut scales, &mut rng);
        }
    }
    let element = |bytes: &[u8], row: usize, index: usize| -> f64 {
        match kind {
            PROBE_MMA_BF16 => bf16_value(u16::from_le_bytes([
                bytes[row * 32 + index * 2],
                bytes[row * 32 + index * 2 + 1],
            ])),
            PROBE_MMA_NVFP4 => {
                let byte = bytes[row * 32 + index / 2];
                e2m1(if index.is_multiple_of(2) { byte & 15 } else { byte >> 4 })
            }
            _ => e4m3(bytes[row * 32 + index]),
        }
    };
    let scale = |row: usize, index: usize| -> f64 {
        match kind {
            PROBE_MMA_BF16 => 1.0,
            PROBE_MMA_NVFP4 => ue4m3(scales[row * 4 + index / 16]),
            _ => ue8m0(scales[row]),
        }
    };
    let mut expected = vec![0.0; 128];
    let mut magnitude = vec![0.0; 128];
    for m in 0..16 {
        for n in 0..8 {
            for index in 0..k {
                let term = element(&a, m, index) * scale(m, index) * element(&b, n, index) * scale(16 + n, index);
                expected[m * 8 + n] += term;
                magnitude[m * 8 + n] += term.abs();
            }
        }
    }
    Tile {
        a,
        b,
        scales,
        expected,
        magnitude,
    }
}

fn run_tile(dev: &CudaDevice, kind: u16, tile: &Tile) -> Vec<f32> {
    let (a, b, s) = (upload(dev, &tile.a), upload(dev, &tile.b), upload(dev, &tile.scales));
    let out = dev.alloc_device(128 * 4).unwrap();
    dev.memset_zero(out, 128 * 4).unwrap();
    let mut ring = Ring::new(dev, 1024, false).unwrap();
    ring.zero_device().unwrap();
    ring.set(&chained(&[MkTaskDescriptor {
        opcode: OP_PROBE_MMA_TILE,
        a,
        b,
        c: out,
        e: s,
        k: u32::from(kind),
        ..Default::default()
    }]));
    unsafe { ring.run().unwrap() };
    let result = download(dev, out, 128);
    ring.free().unwrap();
    for ptr in [a, b, s, out] {
        dev.free_device(ptr).unwrap();
    }
    result
}

const KINDS: [(u16, &str); 3] = [
    (PROBE_MMA_BF16, "bf16 m16n8k16"),
    (PROBE_MMA_NVFP4, "NVFP4 m16n8k64"),
    (PROBE_MMA_MXFP8, "MXFP8 m16n8k32"),
];

#[test]
fn probe_mma_tiles_are_exact_on_designed_inputs() {
    let dev = open();
    for (kind, name) in KINDS {
        let mut worst = 0.0f64;
        for seed in 1..=20 {
            let tile = make_tile(kind, seed, false);
            let got = run_tile(&dev, kind, &tile);
            for (i, (&g, &want)) in got.iter().zip(&tile.expected).enumerate() {
                let diff = (f64::from(g) - want).abs();
                worst = worst.max(diff);
                assert_eq!(
                    f64::from(g),
                    want,
                    "{name}, seed {seed}, result {i}: the GPU gives {g}, the host sum is {want}"
                );
            }
        }
        eprintln!("probe mma exact: {name}: 20 tiles of designed inputs, 2560 results, max difference {worst}");
    }
}

#[test]
fn probe_mma_tiles_on_wide_ranges_report_the_error() {
    let dev = open();
    for (kind, name) in KINDS {
        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        let mut inexact = 0;
        for seed in 1..=20 {
            let tile = make_tile(kind, 100 + seed, true);
            let got = run_tile(&dev, kind, &tile);
            for i in 0..128 {
                let diff = (f64::from(got[i]) - tile.expected[i]).abs();
                worst_abs = worst_abs.max(diff);
                if tile.magnitude[i] > 0.0 {
                    worst_rel = worst_rel.max(diff / tile.magnitude[i]);
                }
                inexact += usize::from(diff != 0.0);
            }
        }
        eprintln!(
            "probe mma wide: {name}: 20 tiles, {inexact} of 2560 results differ from the exact sum, max difference {worst_abs}, max difference over the sum of absolute terms {worst_rel:.3e}"
        );
        // An f32 sum of at most 64 terms is within 64 ulps of the sum of the absolute terms.
        assert!(
            worst_rel < 64.0 * 2f64.powi(-23),
            "{name}: the relative error {worst_rel} is larger than f32 explains"
        );
    }
}

#[test]
fn probe_mma_rate() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    const REPS: u32 = 20_000;
    eprintln!(
        "probe mma rate: {blocks} blocks, {PROBE_MMA_CHAINS} chains of {REPS} MMAs in each warp; time from %globaltimer"
    );
    for (kind, name) in KINDS {
        let (a_byte, scale_a, k, flops): (Vec<u8>, Vec<u8>, usize, usize) = match kind {
            PROBE_MMA_BF16 => ([0x80u8, 0x3f].repeat(16 * 16), vec![0; 96], 16, 16 * 8 * 16 * 2),
            PROBE_MMA_NVFP4 => (vec![0x22; 512], vec![0x38; 96], 64, 16 * 8 * 64 * 2),
            _ => (vec![0x38; 512], vec![127; 96], 32, 16 * 8 * 32 * 2),
        };
        let b_byte = a_byte[..256].to_vec();
        let (a, b, s) = (upload(&dev, &a_byte), upload(&dev, &b_byte), upload(&dev, &scale_a));
        for threads in [256u32, 288] {
            // A block of 288 threads has a producer warp, and only the consumers run MMAs.
            let workers = if threads == PRODUCER_BLOCK_THREADS {
                CONSUMER_THREADS as usize
            } else {
                threads as usize
            };
            let warps = workers / 32;
            let words = blocks * workers * PROBE_MMA_CHAINS * 4;
            let out = dev.alloc_device(2 * words * 4).unwrap();
            dev.memset_zero(out, 2 * words * 4).unwrap();
            let mut ring = Ring::new(&dev, 1024, true).unwrap();
            ring.zero_device().unwrap();
            let tasks: Vec<MkTaskDescriptor> = (0..2)
                .map(|t| MkTaskDescriptor {
                    opcode: OP_PROBE_MMA_RATE,
                    a,
                    b,
                    c: out + (t * words * 4) as u64,
                    e: s,
                    n: REPS,
                    k: u32::from(kind),
                    ..Default::default()
                })
                .collect();
            ring.set(&chained(&tasks));
            unsafe { ring.run_shaped(threads, 0).unwrap() };
            let profile = ring.read_profile().unwrap();
            let got: Vec<f32> = download(&dev, out, 2 * words);
            // The tile is all ones, so every result is K, and each accumulator holds REPS times K.
            let want = (REPS as usize * k) as f32;
            assert!(
                got.iter().all(|&v| v == want),
                "{name}, {threads} threads: an accumulator is not {want}"
            );
            let ns = task_ns(&profile, blocks, 1);
            let mmas_per_sm = (warps * PROBE_MMA_CHAINS) as f64 * f64::from(REPS);
            eprintln!(
                "probe mma rate: {name:<15} {threads:>3} threads: {:>6.2} ms, {:>7.2} MMA/us for each SM, {:>7.2} TFLOP/s for the GPU, one MMA each {:>5.2} ns for each SM",
                ns as f64 / 1e6,
                mmas_per_sm / ns as f64 * 1e3,
                mmas_per_sm * blocks as f64 * flops as f64 / ns as f64 / 1e3,
                ns as f64 / mmas_per_sm,
            );
            ring.free().unwrap();
            dev.free_device(out).unwrap();
        }
        for ptr in [a, b, s] {
            dev.free_device(ptr).unwrap();
        }
    }
}
