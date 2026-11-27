//! GPU report: the floor of the time of the draft head (docs/gb10-research.md, D1, measure M2). It times the existing
//! `OP_MATMUL_BF16` on the shapes that the head reads, with random `bf16` weights, 8 rows and no norm, and sums them as one
//! block of the head would: 3 layers of qkv, o, gate_up and down, the base logits of the unembedding, the matvec of the Markov
//! context, the 3 reads of the fold table (a matmul of N = 200,064 and K = 256 would read as fast as the base matmul per byte),
//! and the append `kv_ctx` of the accepted rows. The down matmul has K = 9,216, which the table of K slices does not hold
//! yet, so it runs as two matmuls of K = 4,608 over the same bytes. Ops that are not matmuls (norms, attention, the
//! argmax) are not in the sum: the report is a floor for the head, not its cost.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test --release -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_head_cost
//!  -- --ignored --test-threads=1 --nocapture`
//!
//! A report test: nothing is asserted but that each call ran, so no mutation applies.

#![cfg(feature = "cuda")]

mod common;

use common::gpu::{open_profile, run_with_watchdog, upload};
use nagaya_mk_cuda::protocol::PROFILE_WORDS;
use nagaya_mk_cuda::repack::{MatmulShape, repack_bf16};
use nagaya_mk_cuda::ring::Ring;
use nagaya_mk_cuda::ring::Task;

const ROWS: usize = 8;
const COPIES: usize = 3;

/// `bf16` values of 1 and -1 and small steps: finite, and no NaN pattern.
fn weights(n: usize, k: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9u32;
    (0..n * k)
        .flat_map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((state >> 16) % 2048) as f32 / 1024.0 - 1.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// The median time in ns of one matmul `W[n, k]` at `ROWS` rows, from the profile of tasks that each read their own copy.
fn time_matmul(n: usize, k: usize) -> (u64, usize) {
    let dev = open_profile();
    let blocks = dev.sm_count() as usize;
    let shape = MatmulShape::new(n, k).unwrap();
    let packed_bytes = repack_bf16(&weights(n, k), &shape).unwrap();
    let copies: Vec<u64> = (0..COPIES).map(|_| upload(&dev, &packed_bytes)).collect();
    let x: Vec<f32> = (0..ROWS * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let x_dev = upload(&dev, &x);
    let y = dev.alloc_device(ROWS * n * 4).unwrap();
    let mut ring = Ring::new(&dev, shape.scratch_floats(ROWS), true).unwrap();
    ring.zero_device().unwrap();
    let tasks: Vec<Task> = (0..COPIES * 2)
        .map(|i| {
            let desc = shape
                .descriptor(ROWS, x_dev, copies[i % COPIES], y, ring.scratch())
                .unwrap();
            Task::with_items(desc)
                .reads(x_dev, ROWS * k * 4)
                .reads(copies[i % COPIES], shape.packed_bytes())
                .writes(y, ROWS * n * 4)
                .writes(ring.scratch(), shape.scratch_floats(ROWS) * 4)
        })
        .collect();
    ring.set(&tasks);
    let shared = dev.max_dynamic_shared_bytes().unwrap();
    run_with_watchdog(&ring, blocks as u32, 288, shared);
    let profile = ring.read_profile().unwrap();
    let mut spans: Vec<u64> = (1..tasks.len())
        .map(|t| {
            let at = ring.ring_index(t);
            let slots = &profile[at * blocks * PROFILE_WORDS..(at + 1) * blocks * PROFILE_WORDS];
            let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
            let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
            end - begin
        })
        .collect();
    spans.sort_unstable();
    let bytes = shape.packed_bytes();
    for p in copies {
        dev.free_device(p).unwrap();
    }
    dev.free_device(x_dev).unwrap();
    dev.free_device(y).unwrap();
    ring.free().unwrap();
    (spans[spans.len() / 2], bytes)
}

#[test]
#[ignore = "a report on the card"]
fn the_matmul_floor_of_one_block_of_the_head() {
    // (name, N, K, times in one block of K=3 drafts)
    let list: [(&str, usize, usize, f64); 7] = [
        ("qkv (q 3072 + k 1024 + v 1024)", 5120, 2304, 3.0),
        ("o_proj", 2304, 3072, 3.0),
        ("gate_up", 18432, 2304, 3.0),
        ("down as 2 x K=4608", 2304, 4608, 6.0),
        ("base logits (unembed, 200,064 rows)", 200_064, 2304, 1.0),
        ("Markov context matvec", 768, 4608, 1.0),
        ("append kv_ctx (3 layers)", 2048, 2304, 3.0),
    ];
    let mut total_ns = 0.0;
    let mut total_bytes = 0.0;
    let mut append_ns = 0.0;
    let mut base_rate = 0.0;
    for (name, n, k, times) in list {
        let (ns, bytes) = time_matmul(n, k);
        let gbps = bytes as f64 / ns as f64;
        eprintln!(
            "{name:<40} {n:>7} x {k:<5} {:>8.1} us {:>7.1} MB {gbps:>6.1} GB/s  x{times}",
            ns as f64 / 1e3,
            bytes as f64 / 1e6
        );
        if name.starts_with("append") {
            append_ns = ns as f64 * times;
        } else {
            total_ns += ns as f64 * times;
            total_bytes += bytes as f64 * times;
        }
        if name.starts_with("base") {
            base_rate = gbps;
        }
    }
    // The fold table: 201,024 rows of 256 `bf16` values, read once for each of the K = 3 Markov steps, at the rate of the base matmul.
    let fold = 3.0 * 103e6 / base_rate;
    eprintln!(
        "fold table, 3 reads of 103 MB at {base_rate:.1} GB/s: {:.1} us",
        fold / 1e3
    );
    let head_ns = total_ns + fold;
    let step_ms = 26.6;
    eprintln!(
        "matmul floor of a block at K=3: {:.2} ms ({:.2} GB read, {:.1} GB/s) = {:.3} decode steps of {step_ms} ms; with 90 small tasks of 2 us: {:.2} ms = {:.3} steps; append of the accepted rows (kv_ctx, 3 layers): {:.0} us",
        head_ns / 1e6,
        (total_bytes + 3.0 * 103e6) / 1e9,
        (total_bytes + 3.0 * 103e6) / head_ns,
        head_ns / 1e6 / step_ms,
        (head_ns + 90.0 * 2000.0) / 1e6,
        (head_ns + 90.0 * 2000.0) / 1e6 / step_ms,
        append_ns / 1e3
    );
}
