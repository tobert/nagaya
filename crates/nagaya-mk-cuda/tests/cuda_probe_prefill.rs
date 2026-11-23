//! GPU probes of the prefill form of the dense matmul (milestone 37 step 9, lane F). They measure what the design
//! assumed and had not measured on the GB10: the rate of the MMA with few chains in a warp, the rate of shared memory,
//! the ring of bulk copies when its barriers are reused, and the rate of L2 under bulk copies from all the SMs. Each
//! op returns a checksum, and the test compares it with the host. Read the tables with `--nocapture`.
//!
//! `NAGAYA_TEST_CARDS=000f:01:00.0 cargo test -p nagaya-mk-cuda --features cuda --test cuda_probe_prefill -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{
    CONSUMER_THREADS, MkTaskDescriptor, OP_PROBE_MMA_RATE, OP_PROBE_READ, OP_PROBE_SMEM, PROBE_ITEM_BYTES,
    PROBE_MMA_BF16, PROBE_MMA_CHAINS, PROBE_READ_SMEM_BULK, PROBE_READ_SMEM_L2, PROBE_SMEM_LD_V2, PROBE_SMEM_LD_V4,
    PROBE_SMEM_MMA_FED, PROBE_SMEM_MMA_REGS, PROBE_SMEM_ST_V2, PROBE_SMEM_ST_V4, PRODUCER_BLOCK_THREADS, PROFILE_WORDS,
    probe_smem_bytes,
};
use nagaya_mk_cuda::ring::{Ring, chained};
use nagaya_mk_cuda::{INKLING2_PROBE_PROFILE_CUBIN, test_cards};

const CORE_CLOCK_GHZ: f64 = 2.418;

fn open() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROBE_PROFILE_CUBIN)
        .expect("open the GPU with the profile cubin")
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

fn pattern(index: u64) -> u32 {
    let x = (index as u32).wrapping_mul(0x9E37_79B1);
    x ^ (x >> 15) ^ ((index >> 32) as u32)
}

fn workers(threads: u32) -> usize {
    if threads == PRODUCER_BLOCK_THREADS {
        CONSUMER_THREADS as usize
    } else {
        threads as usize
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (a) The MMA rate with few chains
// ---------------------------------------------------------------------------------------------------------------

/// The prefill form keeps 4 chains in a warp. This test runs 8 warps (and 4 warps) with 4 chains (`OP_PROBE_SMEM`,
/// the variant with the fragments in registers) and with 8 chains (`OP_PROBE_MMA_RATE`), and prints the rates. The
/// chain count of `OP_PROBE_MMA_RATE` is fixed: an instance with fewer chains makes ptxas spill in the kernel.
#[test]
fn probe_mma_four_chains_against_eight() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    let max_shared = dev.max_dynamic_shared_bytes().unwrap();
    const REPS: u32 = 20_000;
    let a_byte = [0x80u8, 0x3f].repeat(16 * 16);
    let b_byte = a_byte[..256].to_vec();
    let (a, b, s) = (upload(&dev, &a_byte), upload(&dev, &b_byte), upload(&dev, &[0u8; 96]));
    eprintln!("probe mma chains: bf16 m16n8k16, {blocks} blocks; MMA/ns is for one SM; in flight = warps x chains");
    for threads in [128u32, 256, 288] {
        let w = workers(threads);
        let warps = w / 32;
        // Eight chains.
        let words = blocks * w * PROBE_MMA_CHAINS * 4;
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
                k: u32::from(PROBE_MMA_BF16),
                ..Default::default()
            })
            .collect();
        ring.set(&chained(&tasks));
        unsafe { ring.run_shaped(threads, 0).unwrap() };
        let profile = ring.read_profile().unwrap();
        let got: Vec<f32> = download(&dev, out, 2 * words);
        let want = (REPS as usize * 16) as f32;
        assert!(
            got.iter().all(|&v| v == want),
            "{threads} threads, 8 chains: an accumulator is wrong"
        );
        let ns = task_ns(&profile, blocks, 1);
        let eight = (warps * 8) as f64 * f64::from(REPS) / ns as f64;
        eprintln!(
            "probe mma chains: {threads:>3} threads ({warps} warps) x 8 chains = {:>2} in flight: {eight:>5.3} MMA/ns for each SM, {:>6.1} TFLOP/s for the GPU",
            warps * 8,
            eight * blocks as f64 * 4096.0 / 1e3
        );
        ring.free().unwrap();
        dev.free_device(out).unwrap();
        // Four chains, three terms: 12 MMAs for each step.
        let (iterations, steps) = (40u32, 400u32);
        let out = dev.alloc_device(2 * blocks * 4).unwrap();
        dev.memset_zero(out, 2 * blocks * 4).unwrap();
        let mut ring = Ring::new(&dev, 1024, true).unwrap();
        ring.zero_device().unwrap();
        let tasks: Vec<MkTaskDescriptor> = (0..2)
            .map(|t| MkTaskDescriptor {
                opcode: OP_PROBE_SMEM,
                c: out + (t * blocks * 4) as u64,
                n: iterations,
                k: u32::from(PROBE_SMEM_MMA_REGS),
                kt: steps,
                ..Default::default()
            })
            .collect();
        ring.set(&chained(&tasks));
        unsafe { ring.run_shaped(threads, max_shared).unwrap() };
        let profile = ring.read_profile().unwrap();
        let got: Vec<u32> = download(&dev, out, 2 * blocks);
        let want = smem_expected(PROBE_SMEM_MMA_REGS, iterations, steps, w);
        assert!(
            got.iter().all(|&v| v == want),
            "{threads} threads, 4 chains: a checksum is wrong"
        );
        let ns = task_ns(&profile, blocks, 1);
        let four = (warps * 12) as f64 * f64::from(iterations * steps) / ns as f64;
        eprintln!(
            "probe mma chains: {threads:>3} threads ({warps} warps) x 4 chains = {:>2} in flight: {four:>5.3} MMA/ns for each SM, {:>6.1} TFLOP/s for the GPU = {:.1}% of 8 chains",
            warps * 4,
            four * blocks as f64 * 4096.0 / 1e3,
            four / eight * 100.0
        );
        ring.free().unwrap();
        dev.free_device(out).unwrap();
    }
    for ptr in [a, b, s] {
        dev.free_device(ptr).unwrap();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (b) Shared memory
// ---------------------------------------------------------------------------------------------------------------

const SMEM_WORD: u32 = 0x3F80_3F80;

/// The checksum of one block of `OP_PROBE_SMEM`. A load variant sums the words that it reads, a store variant sums the
/// first 32 KB of the area, and an MMA variant sums the f32 bits of the 16 accumulators of each thread.
fn smem_expected(variant: u16, iterations: u32, steps: u32, workers: usize) -> u32 {
    let words_per_load = |v4: bool| if v4 { 4u32 } else { 2 };
    match variant {
        PROBE_SMEM_LD_V4 => (iterations * workers as u32 * 8 * words_per_load(true)).wrapping_mul(SMEM_WORD),
        PROBE_SMEM_LD_V2 => (iterations * workers as u32 * 8 * words_per_load(false)).wrapping_mul(SMEM_WORD),
        PROBE_SMEM_ST_V4 | PROBE_SMEM_ST_V2 => (32768u32 / 4).wrapping_mul(SMEM_WORD),
        _ => {
            // Each accumulator word holds 48 for each step: 3 terms, and each MMA adds 16.
            let total = 16.0 * 48.0 * (iterations * steps) as f32;
            assert!(total < 16_777_216.0, "the sum must stay exact in f32");
            total.to_bits().wrapping_mul(workers as u32)
        }
    }
}

#[test]
fn probe_smem_rates() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    let max_shared = dev.max_dynamic_shared_bytes().unwrap();
    eprintln!(
        "probe smem: {blocks} blocks, dynamic shared memory {max_shared} B; rates are for one SM, B/clk at {CORE_CLOCK_GHZ} GHz"
    );
    // (name, variant, iterations, steps, bytes of one iteration for each worker)
    let cases: [(&str, u16, u32, u32, f64); 6] = [
        ("ld.shared.v4 ", PROBE_SMEM_LD_V4, 20_000, 0, 8.0 * 16.0),
        ("ld.shared.v2 ", PROBE_SMEM_LD_V2, 20_000, 0, 8.0 * 8.0),
        ("st.shared.v4 ", PROBE_SMEM_ST_V4, 20_000, 0, 8.0 * 16.0),
        ("st.shared.v2 ", PROBE_SMEM_ST_V2, 20_000, 0, 8.0 * 8.0),
        // 2.5 KB of fragments for each warp for each step; 12 MMAs for each warp for each step.
        (
            "mma, 4 chains, fragments from shared memory",
            PROBE_SMEM_MMA_FED,
            40,
            400,
            0.0,
        ),
        (
            "mma, 4 chains, fragments in registers      ",
            PROBE_SMEM_MMA_REGS,
            40,
            400,
            0.0,
        ),
    ];
    let mut mma_rates = Vec::new();
    for threads in [128u32, 256, 288] {
        let w = workers(threads);
        for &(name, variant, iterations, steps, bytes) in &cases {
            let out = dev.alloc_device(2 * blocks * 4).unwrap();
            dev.memset_zero(out, 2 * blocks * 4).unwrap();
            let mut ring = Ring::new(&dev, 1024, true).unwrap();
            ring.zero_device().unwrap();
            let tasks: Vec<MkTaskDescriptor> = (0..2)
                .map(|t| MkTaskDescriptor {
                    opcode: OP_PROBE_SMEM,
                    c: out + (t * blocks * 4) as u64,
                    n: iterations,
                    k: u32::from(variant),
                    kt: steps,
                    ..Default::default()
                })
                .collect();
            ring.set(&chained(&tasks));
            unsafe { ring.run_shaped(threads, max_shared).unwrap() };
            let profile = ring.read_profile().unwrap();
            let got: Vec<u32> = download(&dev, out, 2 * blocks);
            let want = smem_expected(variant, iterations, steps, w);
            for (i, v) in got.iter().enumerate() {
                assert_eq!(
                    *v,
                    want,
                    "{name}, {threads} threads: checksum of task {}, block {}",
                    i / blocks,
                    i % blocks
                );
            }
            let ns = task_ns(&profile, blocks, 1) as f64;
            if steps == 0 {
                let moved = f64::from(iterations) * w as f64 * bytes;
                eprintln!(
                    "probe smem: {threads:>3} threads, {name}: {:>6.2} ms, {:>6.1} B/ns = {:>5.1} B/clk",
                    ns / 1e6,
                    moved / ns,
                    moved / ns / CORE_CLOCK_GHZ
                );
            } else {
                let mmas = (w / 32) as f64 * 12.0 * f64::from(steps) * f64::from(iterations);
                let fragments = (w / 32) as f64 * 2560.0 * f64::from(steps) * f64::from(iterations);
                eprintln!(
                    "probe smem: {threads:>3} threads, {name}: {:>6.2} ms, {:>5.3} MMA/ns, fragments {:>6.1} B/ns = {:>5.1} B/clk",
                    ns / 1e6,
                    mmas / ns,
                    if variant == PROBE_SMEM_MMA_FED {
                        fragments / ns
                    } else {
                        0.0
                    },
                    if variant == PROBE_SMEM_MMA_FED {
                        fragments / ns / CORE_CLOCK_GHZ
                    } else {
                        0.0
                    },
                );
                mma_rates.push((threads, variant, mmas / ns));
            }
            ring.free().unwrap();
            dev.free_device(out).unwrap();
        }
    }
    for threads in [128u32, 256, 288] {
        let rate = |v: u16| mma_rates.iter().find(|r| r.0 == threads && r.1 == v).unwrap().2;
        eprintln!(
            "probe smem: {threads} threads: the MMAs fed from shared memory run at {:.1}% of the rate with registers",
            rate(PROBE_SMEM_MMA_FED) / rate(PROBE_SMEM_MMA_REGS) * 100.0
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (c) The ring of bulk copies, its barriers reused, and the area given to another op
// ---------------------------------------------------------------------------------------------------------------

const RING_BYTES: usize = 128 << 20;

/// The wrapping sum of the words of each item.
fn item_sums(host: &[u32]) -> Vec<u32> {
    host.chunks(PROBE_ITEM_BYTES / 4)
        .map(|c| c.iter().fold(0u32, |s, w| s.wrapping_add(*w)))
        .collect()
}

#[test]
fn bulk_ring_barriers_survive_reuse_by_other_ops() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    let max_shared = dev.max_dynamic_shared_bytes().unwrap();
    let items = RING_BYTES / PROBE_ITEM_BYTES;
    let host: Vec<u32> = (0..(RING_BYTES / 4) as u64).map(pattern).collect();
    let sums = item_sums(&host);
    let expected: Vec<u32> = (0..blocks)
        .map(|b| (b..items).step_by(blocks).fold(0u32, |s, i| s.wrapping_add(sums[i])))
        .collect();
    let buffer = upload(&dev, &host);
    drop(host);
    // A task of the bulk ring with `stages` stages, then a task that writes the whole shared memory (the barriers
    // too), and so on. The stage counts change, so a barrier is at another place each time. The ring of 6 stages
    // uses most of the shared memory.
    let stage_counts = [3u32, 6, 2, 4, 6, 1, 5, 6];
    let mut tasks: Vec<MkTaskDescriptor> = Vec::new();
    let checks = dev.alloc_device(2 * stage_counts.len() * blocks * 4).unwrap();
    dev.memset_zero(checks, 2 * stage_counts.len() * blocks * 4).unwrap();
    for (i, &stages) in stage_counts.iter().enumerate() {
        assert!(probe_smem_bytes(stages as usize) as u32 <= max_shared);
        tasks.push(MkTaskDescriptor {
            opcode: OP_PROBE_READ,
            a: buffer,
            c: checks + (2 * i * blocks * 4) as u64,
            n: items as u32,
            k: u32::from(PROBE_READ_SMEM_BULK),
            kt: stages,
            ..Default::default()
        });
        tasks.push(MkTaskDescriptor {
            opcode: OP_PROBE_SMEM,
            c: checks + ((2 * i + 1) * blocks * 4) as u64,
            n: 50,
            k: u32::from(PROBE_SMEM_ST_V4),
            ..Default::default()
        });
    }
    let mut ring = Ring::new(&dev, 1024, true).unwrap();
    ring.zero_device().unwrap();
    ring.set(&chained(&tasks));
    unsafe { ring.run_shaped(PRODUCER_BLOCK_THREADS, max_shared).unwrap() };
    let got: Vec<u32> = download(&dev, checks, 2 * stage_counts.len() * blocks);
    let area = smem_expected(PROBE_SMEM_ST_V4, 50, 0, 256);
    for (i, &stages) in stage_counts.iter().enumerate() {
        assert_eq!(
            &got[2 * i * blocks..(2 * i + 1) * blocks],
            &expected[..],
            "the bulk ring of {stages} stages, task {i}"
        );
        assert!(
            got[(2 * i + 1) * blocks..(2 * i + 2) * blocks]
                .iter()
                .all(|&v| v == area),
            "the store task after the ring {i}"
        );
    }
    eprintln!(
        "probe bulk ring: {} tasks (rings of {stage_counts:?} stages, each followed by a task that overwrites all the shared memory), all checksums right",
        tasks.len()
    );
    ring.free().unwrap();
    dev.free_device(checks).unwrap();
    dev.free_device(buffer).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// (d) L2 under bulk copies from all the SMs
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn probe_l2_under_bulk_copies() {
    let dev = open();
    let blocks = dev.sm_count() as usize;
    let max_shared = dev.max_dynamic_shared_bytes().unwrap();
    const MAX_FOOTPRINT: usize = 64 << 20;
    let host: Vec<u32> = (0..(MAX_FOOTPRINT / 4) as u64).map(pattern).collect();
    let sums = item_sums(&host);
    let buffer = upload(&dev, &host);
    drop(host);
    eprintln!(
        "probe l2 bulk: all {blocks} blocks read the same footprint through bulk copies into 16 KB stages; GB/s is the sum over the blocks"
    );
    for stages in [3u32, 6] {
        for rotate in [0u32, 1] {
            for footprint in [1usize << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20, 64 << 20] {
                let items = footprint / PROBE_ITEM_BYTES;
                let passes = ((8usize << 30) / (blocks * footprint)).max(2) as u32;
                let expected = sums[..items]
                    .iter()
                    .fold(0u32, |s, w| s.wrapping_add(*w))
                    .wrapping_mul(passes);
                let checks = dev.alloc_device(2 * blocks * 4).unwrap();
                dev.memset_zero(checks, 2 * blocks * 4).unwrap();
                let mut ring = Ring::new(&dev, 1024, true).unwrap();
                ring.zero_device().unwrap();
                let tasks: Vec<MkTaskDescriptor> = (0..2)
                    .map(|t| MkTaskDescriptor {
                        opcode: OP_PROBE_READ,
                        a: buffer,
                        c: checks + (t * blocks * 4) as u64,
                        n: items as u32,
                        k: u32::from(PROBE_READ_SMEM_L2),
                        kt: stages,
                        group_items: passes,
                        rows: rotate,
                        ..Default::default()
                    })
                    .collect();
                ring.set(&chained(&tasks));
                unsafe { ring.run_shaped(PRODUCER_BLOCK_THREADS, max_shared).unwrap() };
                let profile = ring.read_profile().unwrap();
                let got: Vec<u32> = download(&dev, checks, 2 * blocks);
                assert!(
                    got.iter().all(|&v| v == expected),
                    "stages {stages}, footprint {footprint}: a checksum is wrong"
                );
                let ns = task_ns(&profile, blocks, 1);
                let bytes = (blocks * footprint) as f64 * f64::from(passes);
                eprintln!(
                    "probe l2 bulk: {stages} stages, {}, footprint {:>5.1} MiB ({passes:>4} passes): {:>7.1} GB/s",
                    if rotate == 1 { "rotated start" } else { "same start   " },
                    footprint as f64 / (1 << 20) as f64,
                    bytes / ns as f64
                );
                ring.free().unwrap();
                dev.free_device(checks).unwrap();
            }
        }
    }
    dev.free_device(buffer).unwrap();
}
