//! GPU gate of the down op `OP_EXPERT_NVFP4` and `OP_EXPERT_MXFP8` with `FLAG_DOWN` (milestone 37 step 9, wave B lane M):
//! the matmul of `w2` on the quantized activation rows, the sum of the 8 slots of a token in slot order, the short
//! convolution `mlp_conv` with its state, and the residual `x = r + conv(f)`. The weights are the real `w2` of Inkling 2
//! Mini, layer 0 (NVFP4) and layer 20 (MXFP8), for 8 experts.
//!
//! The gates:
//! - The slot rows are within the bound of the host dot products of `quant.rs` (4e-7 of the sum of the absolute terms).
//! - The finisher is exact: from the slot rows of the device, the sum in slot order, `ops::short_conv` and the residual
//!   give the output bit for bit, and the conv state after the call is the state of `ops::conv_push`.
//! - A call of 40 tokens equals two calls of 20 tokens that carry the state.
//! - The bits of a token do not depend on the count of tokens, the grid, the order of the pairs, or an output that is the
//!   residual.
//! - The row factor of each output row (`EXPERT_TABLE_OUT_RF`) equals a host model of its sum order, bit for bit, in
//!   every run: at 1, 7, 16 and 40 tokens and at each grid. It is also within 16 epsilons of `ops::rms_norm`'s factor.
//! - The counters are 0 after a run.
//! - The bits of the standard case are pinned by a hash.
//! - Speed: the time of 8 experts with 1 and 16 rows.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder of Mini> cargo test -p nagaya-mk-inkling2-sm121
//!  --features cuda --test cuda_down -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;
mod moe_support;

use common::gpu::{GRIDS, download, open, run_with_watchdog, upload};
use moe_support::expert::{
    Bank, Matrix, N, QuantRow, bank_pairs, bits, host_dot, quant_rows_bytes, quantize_row, upload_bank,
};
use moe_support::{Checkpoint, Rng, activation_row, model_dir};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::experts::{
    EXPERT_TABLE_CONV_STATE, EXPERT_TABLE_CONV_WEIGHT, EXPERT_TABLE_OUT, EXPERT_TABLE_OUT_RF, EXPERT_TABLE_RESIDUAL,
    ExpertFormat, QuantRowsLayout, down_scratch_words, pair_list_bytes,
};
use nagaya_mk_cuda::protocol::{
    EXPERT_TABLE_AMAX, ExpertPair, FLAG_DOWN, FLAG_PAIRS, MkTaskDescriptor, MkTaskTable, OP_EXPERT_MXFP8,
    OP_EXPERT_NVFP4, OP_PROBE_READ, PROBE_ITEM_BYTES, PROBE_READ_NC, PROFILE_WORDS,
};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_inkling2_bmg::host::ops;

const K: usize = 1152;

/// The tiles of a pair that one item covers (a pair has 36).
const TPI: usize = 6;
const BOUND: f64 = 4.0e-7;
const PEAK_GBPS: f64 = 252.0;

fn matrices() -> Option<Vec<Matrix>> {
    let dir = model_dir("cuda_down")?;
    let mut checkpoint = Checkpoint::open(&dir);
    Some(vec![
        checkpoint.matrix(0, "w2", ExpertFormat::Nvfp4, K),
        checkpoint.matrix(20, "w2", ExpertFormat::Mxfp8, K),
    ])
}

/// The inputs of a call: the quantized activation rows (`8 * tokens`), the residual, the weights and the state of the
/// convolution. The same generator gives the same rows for token `t` whatever the count of tokens, so a smaller call is a
/// prefix of a larger one.
struct Inputs {
    act: Vec<QuantRow>,
    residual: Vec<f32>,
    conv_weight: Vec<f32>,
    state: Vec<f32>,
}

fn inputs(m: &Matrix, tokens: usize) -> Inputs {
    let mut rng = Rng(0xD0D0_0000 + m.k as u64);
    // Rows of a token: the generator is advanced for 40 tokens always, so that the prefix does not depend on the count.
    let all_act: Vec<Vec<f32>> = (0..40 * 8)
        .map(|i| activation_row(&mut rng, K, [0.5f32, 0.02, 8.0][i % 3]))
        .collect();
    let residual: Vec<f32> = (0..40 * N).map(|_| rng.unit() * 3.0).collect();
    let conv_weight: Vec<f32> = (0..N * 4).map(|_| rng.unit() * 0.6).collect();
    let state: Vec<f32> = (0..3 * N).map(|_| rng.unit() * 2.0).collect();
    Inputs {
        act: all_act[..tokens * 8]
            .iter()
            .map(|row| quantize_row(m.format, row))
            .collect(),
        residual: residual[..tokens * N].to_vec(),
        conv_weight,
        state,
    }
}

/// The inputs of the tokens `from` to `to - 1` of the long call, with the state `state`.
fn inputs_slice(m: &Matrix, from: usize, to: usize, state: Vec<f32>) -> Inputs {
    let long = inputs(m, 40);
    // The activation rows are generated again, and the rows of the slice are quantized.
    let mut rng = Rng(0xD0D0_0000 + m.k as u64);
    let all_act: Vec<Vec<f32>> = (0..40 * 8)
        .map(|i| activation_row(&mut rng, K, [0.5f32, 0.02, 8.0][i % 3]))
        .collect();
    Inputs {
        act: all_act[from * 8..to * 8]
            .iter()
            .map(|row| quantize_row(m.format, row))
            .collect(),
        residual: long.residual[from * N..to * N].to_vec(),
        conv_weight: long.conv_weight,
        state,
    }
}

struct DownRun {
    slots: Vec<f32>,
    out: Vec<f32>,
    state: Vec<f32>,
    /// The row factor of each output row.
    rf: Vec<f32>,
    /// The counters of the scratch: one for each column tile, then the counter of the row finisher.
    counters: Vec<u32>,
}

/// The sum of the squares of one half tile (32 columns) in the order of the kernel: a butterfly of shuffles with the
/// offsets 16, 8, 4, 2 and 1. Every lane ends with the same bits, and the host takes lane 0.
fn half_tile_sum(values: &[f32]) -> f32 {
    let mut lane: Vec<f32> = values.iter().map(|v| v * v).collect();
    for offset in [16usize, 8, 4, 2, 1] {
        lane = (0..32).map(|l| lane[l] + lane[l ^ offset]).collect();
    }
    lane[0]
}

/// The host model of the row factors of the output rows, in the exact order of the row finisher of the down op: for
/// each column tile of 64 columns, the two half tiles (`half_tile_sum`), added as `half 0 + half 1`; then the 36 tile
/// sums added one after the other in tile order from the tile 0; then `1 / sqrt(sum / N + EPS)`.
fn host_rf(out: &[f32], tokens: usize) -> Vec<f32> {
    (0..tokens)
        .map(|t| {
            let row = &out[t * N..(t + 1) * N];
            let mut total = 0f32;
            for tile in 0..N / 64 {
                let c = &row[tile * 64..(tile + 1) * 64];
                let sum = half_tile_sum(&c[..32]) + half_tile_sum(&c[32..]);
                total = if tile == 0 { sum } else { total + sum };
            }
            1.0 / (total / N as f32 + ops::EPS).sqrt()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn down_task(
    m: &Matrix,
    bank: &Bank,
    x: CUdeviceptr,
    tokens: usize,
    slots: CUdeviceptr,
    table: CUdeviceptr,
    pairs: CUdeviceptr,
    counters: CUdeviceptr,
    bound: usize,
    tpi: usize,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode: match m.format {
            ExpertFormat::Nvfp4 => OP_EXPERT_NVFP4,
            ExpertFormat::Mxfp8 => OP_EXPERT_MXFP8,
        },
        flags: FLAG_PAIRS | FLAG_DOWN,
        items: (bound * N / 64 / tpi) as u32,
        rows: (8 * tokens) as u32,
        a: x,
        b: bank.ptr,
        c: slots,
        d: table,
        e: pairs,
        f: counters,
        n: N as u32,
        k: K as u32,
        kt: tokens as u32,
        pos0: tpi as u32,
        p0: 1.0,
        p1: m.denominator,
        ..Default::default()
    }
}

/// Run the down op. `in_place` makes the output the residual. With `repeat` runs, the state and the output go back to
/// their start values before each run, and the counters stay as the op left them.
#[allow(clippy::too_many_arguments)]
fn run_down(
    dev: &CudaDevice,
    m: &Matrix,
    bank: &Bank,
    inp: &Inputs,
    tokens: usize,
    pairs: &[ExpertPair],
    bound: usize,
    grid: u32,
    repeat: usize,
    in_place: bool,
    tpi: usize,
) -> DownRun {
    let layout = QuantRowsLayout::new(m.format, 8 * tokens, K).unwrap();
    let x_bytes = quant_rows_bytes(&layout, &inp.act);
    let x = upload(dev, &x_bytes);
    let pair_ptr = upload(dev, &pair_list_bytes(pairs, bound).unwrap());
    let slots = upload(dev, &vec![0xFFu8; 8 * tokens * N * 4]);
    let residual = upload(dev, &inp.residual);
    let out = if in_place {
        residual
    } else {
        upload(dev, &vec![0xFFu8; tokens * N * 4])
    };
    let weight = upload(dev, &inp.conv_weight);
    let state = upload(dev, &inp.state);
    let counters = upload(dev, &vec![0u32; down_scratch_words(N, tokens)]);
    let rf = upload(dev, &vec![0xFFu8; tokens * 4]);
    let mut table = MkTaskTable::default();
    table.ptr[EXPERT_TABLE_AMAX] = bank.amax;
    table.ptr[EXPERT_TABLE_RESIDUAL] = residual;
    table.ptr[EXPERT_TABLE_CONV_WEIGHT] = weight;
    table.ptr[EXPERT_TABLE_CONV_STATE] = state;
    table.ptr[EXPERT_TABLE_OUT] = out;
    table.ptr[EXPERT_TABLE_OUT_RF] = rf;
    let table_ptr = upload(dev, &[table]);
    let mut ring = Ring::new(dev, 1, false).unwrap();
    ring.set(&[Task::with_items(down_task(
        m, bank, x, tokens, slots, table_ptr, pair_ptr, counters, bound, tpi,
    ))
    .reads(x, x_bytes.len())
    .reads(bank.ptr, bank.bytes)
    .writes(slots, 8 * tokens * N * 4)
    .writes(out, tokens * N * 4)
    .writes(state, 3 * N * 4)
    .writes(rf, tokens * 4)
    .writes(counters, down_scratch_words(N, tokens) * 4)]);
    for _ in 0..repeat {
        // Back to the start values.
        unsafe {
            dev.copy_htod(state, inp.state.as_ptr().cast(), inp.state.len() * 4)
                .unwrap();
            dev.copy_htod(residual, inp.residual.as_ptr().cast(), inp.residual.len() * 4)
                .unwrap();
        }
        ring.zero_device().unwrap();
        run_with_watchdog(
            &ring,
            grid,
            DEFAULT_BLOCK_THREADS,
            dev.max_dynamic_shared_bytes().unwrap(),
        );
    }
    let result = DownRun {
        slots: download(dev, slots, 8 * tokens * N),
        out: download(dev, out, tokens * N),
        state: download(dev, state, 3 * N),
        rf: download(dev, rf, tokens),
        counters: download(dev, counters, N / 64 + 1),
    };
    // The factors are the bits of the host model of the sum order, in every run.
    assert_eq!(
        bits(&result.rf),
        bits(&host_rf(&result.out, tokens)),
        "{}: the row factors differ from the host model of the sum order at {tokens} tokens, grid {grid}",
        m.name
    );
    // And they are the factor of `ops::rms_norm` within 16 epsilons.
    for (t, got) in result.rf.iter().enumerate() {
        let row = &result.out[t * N..(t + 1) * N];
        let want = 1.0 / (ops::dot(row, row) / N as f32 + ops::EPS).sqrt();
        let e = f64::from((got - want).abs()) / f64::from(want) / f64::from(f32::EPSILON);
        assert!(
            e <= 16.0,
            "{}: row {t}: the factor is {got:e}, the host form {want:e}",
            m.name
        );
    }
    ring.free().unwrap();
    let mut frees = vec![x, pair_ptr, slots, residual, weight, state, counters, rf, table_ptr];
    if !in_place {
        frees.push(out);
    }
    for ptr in frees {
        dev.free_device(ptr).unwrap();
    }
    result
}

/// The host composition from the slot rows `slots` (`8 * tokens` rows of `N`): the sum in slot order, `ops::short_conv`,
/// the residual, and `ops::conv_push`. It returns the output and the final state.
fn host_finish(slots: &[f32], inp: &Inputs, tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = inp.state.clone();
    let mut out = vec![0f32; tokens * N];
    for t in 0..tokens {
        let mut f = vec![0f32; N];
        for s in 0..8 {
            for (a, b) in f.iter_mut().zip(&slots[(t * 8 + s) * N..(t * 8 + s + 1) * N]) {
                *a += *b;
            }
        }
        let mut conv = vec![0f32; N];
        ops::short_conv(&state, &f, &inp.conv_weight, &mut conv);
        ops::conv_push(&mut state, &f);
        for c in 0..N {
            out[t * N + c] = inp.residual[t * N + c] + conv[c];
        }
    }
    (out, state)
}

fn pairs_of(tokens: usize) -> Vec<ExpertPair> {
    bank_pairs(tokens, |_, _| 1.0)
}

#[test]
fn the_slot_rows_are_within_the_bound_and_the_finisher_is_exact() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let tokens = 5;
    for m in &matrices {
        let inp = inputs(m, tokens);
        let bank = upload_bank(&dev, m, false);
        let pairs = pairs_of(tokens);
        let run = run_down(&dev, m, &bank, &inp, tokens, &pairs, pairs.len(), 48, 1, false, TPI);
        assert!(
            run.counters.iter().all(|c| *c == 0),
            "{}: a counter was not reset",
            m.name
        );
        // 1. The slot rows against the host dot products.
        let mut worst = 0.0f64;
        let mut inexact = 0usize;
        for t in 0..tokens {
            for e in 0..8 {
                let slot = t * 8 + e;
                for col in 0..N {
                    let (want, terms) = host_dot(m, e, col, &inp.act[slot]);
                    let got = run.slots[slot * N + col];
                    assert!(got.is_finite(), "{}: slot {slot} column {col} was not written", m.name);
                    let error = (f64::from(got) - f64::from(want)).abs();
                    let ratio = if terms > 0.0 { error / terms } else { error };
                    worst = worst.max(ratio);
                    inexact += usize::from(error != 0.0);
                    assert!(
                        ratio <= BOUND,
                        "{}: slot {slot} column {col}: gpu {got:e}, host {want:e}; the error is {ratio:e} of the terms; the bound is {BOUND:e}",
                        m.name
                    );
                }
            }
        }
        eprintln!(
            "cuda_down: {}: {} slot outputs within {BOUND:e} of the terms; the worst is {worst:e}; {inexact} differ from the host dot in a bit",
            m.name,
            tokens * 8 * N
        );
        // 2. The finisher is exact on the slot rows of the device.
        let (want_out, want_state) = host_finish(&run.slots, &inp, tokens);
        assert_eq!(
            bits(&run.out),
            bits(&want_out),
            "{}: the output of the finisher",
            m.name
        );
        assert_eq!(
            bits(&run.state),
            bits(&want_state),
            "{}: the conv state after the call",
            m.name
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

/// The inputs of rows with a tiny output: activations of 1e-12 and a state of 0, so the output is the residual. The
/// residual of the token `t` is the constant `t * 1e-5`, so `mean(x^2)` runs from 0 to 3e-5 and the epsilon of the
/// factor (1e-6) is a part of each sum. With the rows of the other tests (`mean(x^2)` about 10) the epsilon is below
/// the ulp of the sum, and a change of it does not show.
fn tiny_inputs(m: &Matrix, tokens: usize) -> Inputs {
    let mut rng = Rng(0xE0E0_0000 + m.k as u64);
    Inputs {
        act: (0..tokens * 8)
            .map(|_| quantize_row(m.format, &activation_row(&mut rng, K, 1.0e-12)))
            .collect(),
        residual: (0..tokens * N).map(|i| (i / N) as f32 * 1.0e-5).collect(),
        conv_weight: (0..N * 4).map(|_| rng.unit() * 0.6).collect(),
        state: vec![0.0; 3 * N],
    }
}

/// `run_down` checks the factors against the host model of the sum order in each run. This test uses 512 tokens, so
/// that a change of the epsilon by one ulp moves the bits of some of the rows. The row 0 is `1 / sqrt(EPS)`.
#[test]
fn a_tiny_output_gives_the_factor_of_the_epsilon_and_a_change_of_it_shows() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let want = 1.0 / ops::EPS.sqrt();
    for m in &matrices {
        let bank = upload_bank(&dev, m, false);
        let tokens = 512;
        let inp = tiny_inputs(m, tokens);
        let pairs = pairs_of(tokens);
        let run = run_down(&dev, m, &bank, &inp, tokens, &pairs, pairs.len(), 48, 1, false, TPI);
        let biggest = run.out.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(biggest < 6.0e-3, "{}: the output is not tiny: {biggest:e}", m.name);
        assert_eq!(
            run.rf[0].to_bits(),
            want.to_bits(),
            "{}: row 0: {:e}, want {want:e}",
            m.name,
            run.rf[0]
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

#[test]
fn a_call_of_40_tokens_equals_two_calls_of_20_with_the_state_carried() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let bank = upload_bank(&dev, m, false);
        let long = inputs(m, 40);
        let pairs = pairs_of(40);
        let whole = run_down(&dev, m, &bank, &long, 40, &pairs, pairs.len(), 48, 1, false, TPI);
        let first = inputs_slice(m, 0, 20, long.state.clone());
        let p20 = pairs_of(20);
        let a = run_down(&dev, m, &bank, &first, 20, &p20, p20.len(), 24, 1, false, TPI);
        let second = inputs_slice(m, 20, 40, a.state.clone());
        let b = run_down(&dev, m, &bank, &second, 20, &p20, p20.len(), 7, 1, false, TPI);
        let joined: Vec<f32> = a.out.iter().chain(&b.out).copied().collect();
        assert_eq!(
            bits(&joined),
            bits(&whole.out),
            "{}: two calls of 20 tokens and one of 40",
            m.name
        );
        assert_eq!(
            bits(&b.state),
            bits(&whole.state),
            "{}: the state after two calls and after one",
            m.name
        );
        let (want, _) = host_finish(&whole.slots, &long, 40);
        assert_eq!(
            bits(&whole.out),
            bits(&want),
            "{}: the call of 40 tokens against the host composition",
            m.name
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

#[test]
fn the_bits_do_not_depend_on_the_count_of_tokens_the_grid_the_order_of_the_pairs_or_an_output_in_place() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let bank = upload_bank(&dev, m, false);
        let tokens = 16;
        let inp = inputs(m, tokens);
        let pairs = pairs_of(tokens);
        let reference = run_down(&dev, m, &bank, &inp, tokens, &pairs, pairs.len(), 48, 1, false, TPI);
        for grid in GRIDS {
            // Two runs in one ring with no zeroing of the counters: the first leaves them at 0.
            let got = run_down(&dev, m, &bank, &inp, tokens, &pairs, pairs.len(), grid, 2, false, TPI);
            assert_eq!(
                bits(&got.slots),
                bits(&reference.slots),
                "{}: the slots at grid {grid}",
                m.name
            );
            assert_eq!(
                bits(&got.out),
                bits(&reference.out),
                "{}: the output at grid {grid}",
                m.name
            );
            assert_eq!(
                bits(&got.rf),
                bits(&reference.rf),
                "{}: the row factors at grid {grid}",
                m.name
            );
            assert_eq!(
                bits(&got.state),
                bits(&reference.state),
                "{}: the state at grid {grid}",
                m.name
            );
            assert!(
                got.counters.iter().all(|c| *c == 0),
                "{}: a counter was not reset",
                m.name
            );
        }
        // The cut of an item: 1, 3 and 36 tiles for an item.
        for tpi in [1usize, 3, 36] {
            let got = run_down(&dev, m, &bank, &inp, tokens, &pairs, pairs.len(), 7, 1, false, tpi);
            assert_eq!(
                bits(&got.out),
                bits(&reference.out),
                "{}: the output with {tpi} tiles for an item",
                m.name
            );
            assert_eq!(
                bits(&got.state),
                bits(&reference.state),
                "{}: the state with {tpi} tiles for an item",
                m.name
            );
        }
        // The count of tokens: the slots and the output of the first tokens of a larger call equal the smaller call.
        for count in [1usize, 7] {
            let small = inputs(m, count);
            let p = pairs_of(count);
            let got = run_down(&dev, m, &bank, &small, count, &p, p.len(), 7, 1, false, TPI);
            assert_eq!(
                bits(&got.slots),
                bits(&reference.slots[..count * 8 * N]),
                "{}: the slots of {count} tokens",
                m.name
            );
            assert_eq!(
                bits(&got.out),
                bits(&reference.out[..count * N]),
                "{}: the output of {count} tokens",
                m.name
            );
            assert_eq!(
                bits(&got.rf),
                bits(&reference.rf[..count]),
                "{}: the row factors of {count} tokens",
                m.name
            );
        }
        // The order of the pairs, with room for 5 pairs that are not live, and an output that is the residual.
        let mut reversed: Vec<ExpertPair> = pairs.iter().rev().copied().collect();
        for p in &mut reversed {
            let n = p.count as usize;
            p.x_row[..n].reverse();
            p.y_row[..n].reverse();
            p.weight[..n].reverse();
        }
        let got = run_down(
            &dev,
            m,
            &bank,
            &inp,
            tokens,
            &reversed,
            reversed.len() + 5,
            24,
            1,
            true,
            TPI,
        );
        assert_eq!(
            bits(&got.slots),
            bits(&reference.slots),
            "{}: the slots with the pairs reversed",
            m.name
        );
        assert_eq!(
            bits(&got.out),
            bits(&reference.out),
            "{}: the output in place, with the pairs reversed",
            m.name
        );
        assert_eq!(
            bits(&got.rf),
            bits(&reference.rf),
            "{}: the row factors in place, with the pairs reversed",
            m.name
        );
        eprintln!(
            "cuda_down: {}: the bits equal at the grids {GRIDS:?}, at 1, 7 and 16 tokens, with the pairs reversed, and in place",
            m.name
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

/// The hashes of the standard case (16 tokens, grid 48): the slot rows, the output and the state. They are part of the
/// engine identity.
const PINNED_HASHES: [&str; 2] = [
    "1d6e538df387f80c2d1cf164cc79839c592ab584707a133f44e3f08811e1c04d",
    "af99dea24278e73fa193ccb1f6235f8ec969dd0967e4bc0c8b5c139391bfabe1",
];

#[test]
fn the_bits_of_the_standard_case_are_pinned() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let print_only = std::env::var("CUDA_MOE_PRINT_HASHES").is_ok();
    for (index, m) in matrices.iter().enumerate() {
        let bank = upload_bank(&dev, m, false);
        let inp = inputs(m, 16);
        let pairs = pairs_of(16);
        let run = run_down(&dev, m, &bank, &inp, 16, &pairs, pairs.len(), 48, 1, false, TPI);
        let mut hasher = blake3::Hasher::new();
        for v in run.slots.iter().chain(&run.out).chain(&run.state) {
            hasher.update(&v.to_le_bytes());
        }
        let hash = hasher.finalize().to_hex().to_string();
        if print_only {
            eprintln!("cuda_down pinned: {index} {} {hash}", m.name);
        } else {
            assert_eq!(
                hash, PINNED_HASHES[index],
                "{}: the bits of the standard case changed",
                m.name
            );
        }
        dev.free_device(bank.ptr).unwrap();
    }
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[test]
fn speed_of_the_down_op_for_8_experts_with_1_and_16_rows() {
    let Some(matrices) = matrices() else { return };
    let dev = moe_support::open_profile_or_env();
    let blocks = dev.sm_count() as usize;
    let flush_bytes = 1usize << 30;
    let flush = dev.alloc_device(flush_bytes).unwrap();
    let checks = dev.alloc_device(blocks * 4).unwrap();
    for (m, tokens) in matrices.iter().flat_map(|m| [1usize, 16].map(move |t| (m, t))) {
        let bank = upload_bank(&dev, m, false);
        let inp = inputs(m, tokens);
        let pairs = pairs_of(tokens);
        let layout = QuantRowsLayout::new(m.format, 8 * tokens, K).unwrap();
        let x = upload(&dev, &quant_rows_bytes(&layout, &inp.act));
        let pair_ptr = upload(&dev, &pair_list_bytes(&pairs, pairs.len()).unwrap());
        let slots = upload(&dev, &vec![0u8; 8 * tokens * N * 4]);
        let residual = upload(&dev, &inp.residual);
        let out = upload(&dev, &vec![0u8; tokens * N * 4]);
        let rf = upload(&dev, &vec![0u8; tokens * 4]);
        let weight = upload(&dev, &inp.conv_weight);
        let state = upload(&dev, &inp.state);
        let counters = upload(&dev, &vec![0u32; down_scratch_words(N, tokens)]);
        let mut table = MkTaskTable::default();
        table.ptr[EXPERT_TABLE_AMAX] = bank.amax;
        table.ptr[EXPERT_TABLE_RESIDUAL] = residual;
        table.ptr[EXPERT_TABLE_CONV_WEIGHT] = weight;
        table.ptr[EXPERT_TABLE_CONV_STATE] = state;
        table.ptr[EXPERT_TABLE_OUT] = out;
        table.ptr[EXPERT_TABLE_OUT_RF] = rf;
        let table_ptr = upload(&dev, &[table]);
        let mut ring = Ring::new(&dev, 1, true).unwrap();
        let read = Task::per_block(MkTaskDescriptor {
            opcode: OP_PROBE_READ,
            a: flush,
            c: checks,
            n: (flush_bytes / PROBE_ITEM_BYTES) as u32,
            k: u32::from(PROBE_READ_NC),
            ..Default::default()
        });
        let down = Task::with_items(down_task(
            m,
            &bank,
            x,
            tokens,
            slots,
            table_ptr,
            pair_ptr,
            counters,
            pairs.len(),
            TPI,
        ))
        .after(0);
        ring.set(&[read, down]);
        let index = ring.ring_index(1);
        let mut spans = Vec::new();
        for _ in 0..9 {
            ring.zero_device().unwrap();
            run_with_watchdog(
                &ring,
                blocks as u32,
                DEFAULT_BLOCK_THREADS,
                dev.max_dynamic_shared_bytes().unwrap(),
            );
            let profile = ring.read_profile().unwrap();
            let s = &profile[index * blocks * PROFILE_WORDS..(index + 1) * blocks * PROFILE_WORDS];
            let begin = s.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
            let end = s.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
            spans.push(end - begin);
        }
        let ns = median(spans.clone());
        let weight_bytes = 8 * (N / 64) * m.format.item_bytes(K);
        let gbps = weight_bytes as f64 / ns as f64;
        eprintln!(
            "cuda_down speed: {:<14} {:?}: 8 experts x {tokens:>2} rows: {:>7.1} us (best {:>7.1}), {:>6.2} MB, {:>6.1} GB/s, {:>5.1}% of {PEAK_GBPS} GB/s (the budget is 50 us NVFP4, 91 us MXFP8)",
            m.name,
            m.format,
            ns as f64 / 1e3,
            *spans.iter().min().unwrap() as f64 / 1e3,
            weight_bytes as f64 / 1e6,
            gbps,
            gbps / PEAK_GBPS * 100.0
        );
        ring.free().unwrap();
        for ptr in [
            x, pair_ptr, slots, residual, out, rf, weight, state, counters, table_ptr, bank.ptr, bank.amax,
        ] {
            if ptr != 0 {
                dev.free_device(ptr).unwrap();
            }
        }
    }
    dev.free_device(flush).unwrap();
    dev.free_device(checks).unwrap();
}
