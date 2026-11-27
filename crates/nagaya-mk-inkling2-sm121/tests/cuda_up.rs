//! GPU gate of the up op `OP_EXPERT_NVFP4` and `OP_EXPERT_MXFP8` with `FLAG_UP`: the
//! matmul of `w13` with the gate and the up in the rows `g` and `g + 8` of the MMA, the softcapped SwiGLU times the route
//! weight, and the finisher that quantizes each activation row. The weights are the real `w13` of Inkling 2 Mini, layer 0
//! (NVFP4) and layer 20 (MXFP8), for 8 experts.
//!
//! The gates:
//! - The activation in `f32` is within a bound of the host activation, which comes from the host dot products of `quant.rs`
//!   on the same quantized rows and from `ops::swiglu`. The device `tanhf` and `expf` differ from the host by an ulp or two.
//! - The quantized rows equal `quant.rs` applied to the activation rows of the device, byte for byte. This is the gate of
//!   the finisher. The count of bytes that differ from `quant.rs` of the host activation is printed: a code flips where the
//!   value is near a threshold.
//! - The bits of a row do not depend on the count of rows, on the grid, or on the order of the pairs.
//! - The counters are 0 after a run, and a second run without zeroing gives the same bits.
//! - The bits of the standard case are pinned by a hash.
//! - Speed: the time of 8 experts with 1 row, as a decode step has them.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder of Mini> cargo test -p nagaya-mk-inkling2-sm121
//!  --features cuda --test cuda_up -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;
mod moe_support;

use common::gpu::{GRIDS, download, open, run_with_watchdog, upload};
use moe_support::expert::{
    Bank, EXPERTS, Matrix, N, QuantRow, bank_pairs, bits, expert_index, host_dot, quant_rows_bytes, quantize_row,
    uneven_pairs, upload_bank,
};
use moe_support::{Checkpoint, Rng, activation_row, model_dir};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::experts::{EXPERT_TABLE_INDEX, EXPERT_TABLE_QACT, ExpertFormat, QuantRowsLayout, pair_list_bytes};
use nagaya_mk_cuda::protocol::{
    EXPERT_TABLE_AMAX, ExpertPair, FLAG_BLOCK_FORM, FLAG_PAIRS, FLAG_UP, MkTaskDescriptor, MkTaskTable,
    OP_EXPERT_MXFP8, OP_EXPERT_NVFP4, OP_PROBE_READ, PROBE_ITEM_BYTES, PROBE_READ_NC, PROFILE_WORDS,
};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_inkling2_bmg::host::ops;
use nagaya_mk_inkling2_sm121::quant::{quantize_mxfp8_ue8m0, quantize_nvfp4_fouroversix_mae};

const INNER: usize = N / 2;

/// The tiles of a pair that one item covers. A pair has `N / 64 = 36` tiles. With 6 tiles for an item, a decode step of 8
/// pairs has 48 items: one for each block, and one arrival at a merge counter for each block.
const TPI: usize = 6;
const PEAK_GBPS: f64 = 252.0;

/// The bound of the MMA of the gate and of the up, as a share of the sum of their absolute terms (the bound of the expert
/// gate of wave A).
const MMA_BOUND: f64 = 4.0e-7;

fn matrices() -> Option<Vec<Matrix>> {
    let dir = model_dir("cuda_up")?;
    let mut checkpoint = Checkpoint::open(&dir);
    Some(vec![
        checkpoint.matrix(0, "w13", ExpertFormat::Nvfp4, 2304),
        checkpoint.matrix(20, "w13", ExpertFormat::Mxfp8, 2304),
    ])
}

/// The input rows of the gate: `count` quantized rows of the hidden size, with three magnitudes.
fn input_rows(m: &Matrix, count: usize) -> Vec<QuantRow> {
    let mut rng = Rng(0x0F0F_0000 + m.k as u64);
    (0..count)
        .map(|i| quantize_row(m.format, &activation_row(&mut rng, m.k, [1.0f32, 0.05, 20.0][i % 3])))
        .collect()
}

/// The route weight of a token and a bank expert.
fn weight(token: usize, expert: usize) -> f32 {
    0.25 + 0.1 * ((token * 5 + expert * 3) % 17) as f32
}

struct UpRun {
    /// The activation rows: `8 * tokens` rows of `INNER` floats.
    act: Vec<f32>,
    /// The quantized rows: the layout of `8 * tokens` rows.
    quant: Vec<u8>,
    layout: QuantRowsLayout,
    counters: Vec<u32>,
}

/// The descriptor of the up op.
#[allow(clippy::too_many_arguments)]
fn up_task(
    m: &Matrix,
    bank: &Bank,
    x: CUdeviceptr,
    x_rows: usize,
    tokens: usize,
    act: CUdeviceptr,
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
        flags: FLAG_PAIRS | FLAG_UP,
        items: (bound * N / 64 / tpi) as u32,
        rows: x_rows as u32,
        a: x,
        b: bank.ptr,
        c: act,
        d: table,
        e: pairs,
        f: counters,
        n: N as u32,
        k: m.k as u32,
        kt: tokens as u32,
        group_items: (N / 64 / tpi) as u32,
        pos0: tpi as u32,
        p0: 1.0,
        p1: m.denominator,
        ..Default::default()
    }
}

/// Run the up op over `pairs` at `grid`, `repeat` times with no zeroing of the counters between the runs.
#[allow(clippy::too_many_arguments)]
fn run_up(
    dev: &CudaDevice,
    m: &Matrix,
    bank: &Bank,
    rows: &[QuantRow],
    tokens: usize,
    pairs: &[ExpertPair],
    bound: usize,
    grid: u32,
    repeat: usize,
    tpi: usize,
) -> UpRun {
    run_up_form(dev, m, bank, rows, tokens, pairs, bound, grid, repeat, tpi, false)
}

/// Like `run_up`; with `block` the op is in the block form (`FLAG_BLOCK_FORM`, the expert index in the table).
#[allow(clippy::too_many_arguments)]
fn run_up_form(
    dev: &CudaDevice,
    m: &Matrix,
    bank: &Bank,
    rows: &[QuantRow],
    tokens: usize,
    pairs: &[ExpertPair],
    bound: usize,
    grid: u32,
    repeat: usize,
    tpi: usize,
    block: bool,
) -> UpRun {
    let x_layout = QuantRowsLayout::new(m.format, rows.len(), m.k).unwrap();
    let layout = QuantRowsLayout::new(m.format, 8 * tokens, INNER).unwrap();
    let x_bytes = quant_rows_bytes(&x_layout, rows);
    let x = upload(dev, &x_bytes);
    let pair_ptr = upload(dev, &pair_list_bytes(pairs, bound).unwrap());
    let act = upload(dev, &vec![0xFFu8; 8 * tokens * INNER * 4]);
    let qact = upload(dev, &vec![0xABu8; layout.bytes()]);
    let counters = upload(dev, &vec![0u32; bound]);
    let mut table = MkTaskTable::default();
    table.ptr[EXPERT_TABLE_AMAX] = bank.amax;
    table.ptr[EXPERT_TABLE_QACT] = qact;
    let index = upload(dev, &expert_index(pairs));
    if block {
        table.ptr[EXPERT_TABLE_INDEX] = index;
    }
    let table_ptr = upload(dev, &[table]);
    let mut ring = Ring::new(dev, 1, false).unwrap();
    let mut task = up_task(
        m,
        bank,
        x,
        rows.len(),
        tokens,
        act,
        table_ptr,
        pair_ptr,
        counters,
        bound,
        tpi,
    );
    if block {
        task.flags |= FLAG_BLOCK_FORM;
        task.items = (128 * N / 64 / tpi) as u32;
    }
    ring.set(&[Task::with_items(task)
        .reads(x, x_bytes.len())
        .reads(bank.ptr, bank.bytes)
        .writes(act, 8 * tokens * INNER * 4)
        .writes(qact, layout.bytes())
        .writes(counters, bound * 4)]);
    for _ in 0..repeat {
        ring.zero_device().unwrap();
        run_with_watchdog(
            &ring,
            grid,
            DEFAULT_BLOCK_THREADS,
            dev.max_dynamic_shared_bytes().unwrap(),
        );
    }
    let out = UpRun {
        act: download(dev, act, 8 * tokens * INNER),
        quant: download(dev, qact, layout.bytes()),
        layout,
        counters: download(dev, counters, bound),
    };
    ring.free().unwrap();
    for ptr in [x, pair_ptr, act, qact, counters, table_ptr, index] {
        dev.free_device(ptr).unwrap();
    }
    out
}

/// The bytes of the quantized row `row` of a run: the codes, the scales, and the amax.
fn quant_row_bytes(run: &UpRun, format: ExpertFormat, row: usize) -> Vec<u8> {
    let l = &run.layout;
    let mut out = run.quant[l.codes_offset(row)..][..format.code_bytes(INNER)].to_vec();
    out.extend_from_slice(&run.quant[l.scales_offset(row)..][..format.scale_bytes(INNER)]);
    if format == ExpertFormat::Nvfp4 {
        out.extend_from_slice(&run.quant[l.amax_offset(row)..][..4]);
    }
    out
}

/// The bytes that `quant.rs` gives for a row of floats, in the order of [`quant_row_bytes`].
fn host_quant_bytes(format: ExpertFormat, row: &[f32]) -> Vec<u8> {
    match format {
        ExpertFormat::Nvfp4 => {
            let q = quantize_nvfp4_fouroversix_mae(row).unwrap();
            [q.codes, q.scales, q.amax.to_le_bytes().to_vec()].concat()
        }
        ExpertFormat::Mxfp8 => {
            let q = quantize_mxfp8_ue8m0(row).unwrap();
            [q.codes, q.scales].concat()
        }
    }
}

/// The host activation of a token and a bank expert: the host dot products of `quant.rs` for the gate and the up, the
/// SwiGLU of `ops`, and the route weight. The second vector holds the error that the device may have in each value: the
/// error of the MMA of the gate and of the up (`MMA_BOUND` of the sum of their terms) through the derivatives of the
/// SwiGLU, with a margin of 4, and a share of the value for the `tanhf` and the `expf` of the device.
fn host_activation(m: &Matrix, expert: usize, row: &QuantRow, w: f32) -> (Vec<f32>, Vec<f64>) {
    let (mut gate, mut up) = (vec![0f32; INNER], vec![0f32; INNER]);
    let mut allowed = vec![0f64; INNER];
    for i in 0..INNER {
        let (g, g_terms) = host_dot(m, expert, 2 * i, row);
        let (u, u_terms) = host_dot(m, expert, 2 * i + 1, row);
        gate[i] = g;
        up[i] = u;
        let (gd, ud) = (f64::from(g), f64::from(u));
        let sigmoid = 1.0 / (1.0 + (-gd).exp());
        let softcap4 = 4.0 * (gd / 4.0).tanh();
        let softcap25 = 25.0 * (ud / 25.0).tanh();
        // The derivatives are at most these numbers: the softcap has a slope of at most 1, and the sigmoid of at most 1/4.
        let d_gate = (sigmoid + softcap4.abs() * 0.25) * softcap25.abs();
        let d_up = softcap4.abs() * sigmoid;
        let value = softcap4 * sigmoid * softcap25;
        allowed[i] = f64::from(w) * (4.0 * MMA_BOUND * (d_gate * g_terms + d_up * u_terms) + 1.0e-5 * value.abs());
    }
    let mut act = vec![0f32; INNER];
    ops::swiglu(&gate, &up, &mut act);
    (act.iter().map(|a| a * w).collect(), allowed)
}

fn standard_pairs(tokens: usize) -> Vec<ExpertPair> {
    bank_pairs(tokens, weight)
}

#[test]
fn the_activation_is_within_the_bound_of_the_host_and_the_quantized_rows_are_the_bytes_of_quant_rs() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let tokens = 7;
    for m in &matrices {
        let rows = input_rows(m, tokens);
        let bank = upload_bank(&dev, m, true);
        let pairs = standard_pairs(tokens);
        let run = run_up(&dev, m, &bank, &rows, tokens, &pairs, pairs.len(), 48, 1, TPI);
        assert!(
            run.counters.iter().all(|c| *c == 0),
            "{}: a counter was not reset",
            m.name
        );
        let (mut worst, mut worst_at, mut worst_relative) = (0.0f64, (0, 0, 0), 0.0f64);
        let (mut flips, mut total_bytes) = (0usize, 0usize);
        for (t, row) in rows.iter().enumerate() {
            for e in 0..8 {
                let slot = t * 8 + e;
                let (want, allowed) = host_activation(m, e, row, weight(t, e));
                let got = &run.act[slot * INNER..(slot + 1) * INNER];
                for i in 0..INNER {
                    assert!(got[i].is_finite(), "{}: slot {slot} column {i} was not written", m.name);
                    let error = (f64::from(got[i]) - f64::from(want[i])).abs();
                    let ratio = error / allowed[i].max(1e-30);
                    if ratio > worst {
                        worst = ratio;
                        worst_at = (t, e, i);
                    }
                    worst_relative = worst_relative.max(error / f64::from(want[i]).abs().max(1e-30));
                    assert!(
                        ratio <= 1.0,
                        "{}: token {t} expert {e} column {i}: gpu {:e}, host {:e}; the error {error:e} is above the allowed {:e}",
                        m.name,
                        got[i],
                        want[i],
                        allowed[i]
                    );
                }
                // The finisher: the bytes of quant.rs on the activation of the device, exactly.
                let device_bytes = quant_row_bytes(&run, m.format, slot);
                assert_eq!(
                    device_bytes,
                    host_quant_bytes(m.format, got),
                    "{}: token {t} expert {e}: the quantized row differs from quant.rs of the activation of the device",
                    m.name
                );
                // And against the host activation: a flip where a value is near a threshold.
                let host_bytes = host_quant_bytes(m.format, &want);
                flips += device_bytes.iter().zip(&host_bytes).filter(|(a, b)| a != b).count();
                total_bytes += device_bytes.len();
            }
        }
        eprintln!(
            "cuda_up: {}: {} activation rows: the worst error is {worst:e} of the allowed error (token {}, expert {}, column {}), and {worst_relative:e} of the value; the quantized rows equal quant.rs of the device activation; {flips} of {total_bytes} bytes differ from quant.rs of the host activation",
            m.name,
            tokens * 8,
            worst_at.0,
            worst_at.1,
            worst_at.2
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

#[test]
fn the_bits_do_not_depend_on_the_count_of_rows_the_grid_or_the_order_of_the_pairs_and_a_rerun_gives_the_same_bits() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let rows = input_rows(m, 40);
        let bank = upload_bank(&dev, m, true);
        let tokens = 16;
        let pairs = standard_pairs(tokens);
        let reference = run_up(&dev, m, &bank, &rows[..tokens], tokens, &pairs, pairs.len(), 48, 1, TPI);
        // The grid, and a second run with no zeroing of the counters.
        for grid in GRIDS {
            let got = run_up(
                &dev,
                m,
                &bank,
                &rows[..tokens],
                tokens,
                &pairs,
                pairs.len(),
                grid,
                2,
                TPI,
            );
            assert_eq!(
                bits(&got.act),
                bits(&reference.act),
                "{}: the activation at grid {grid}",
                m.name
            );
            assert_eq!(
                got.quant, reference.quant,
                "{}: the quantized rows at grid {grid}",
                m.name
            );
            assert!(
                got.counters.iter().all(|c| *c == 0),
                "{}: a counter was not reset",
                m.name
            );
        }
        // The cut of an item: 1, 2, 4, 9 and 36 tiles for an item, with a bound above the live pairs.
        for tpi in [1usize, 2, 4, 9, 36] {
            for grid in [48u32, 7] {
                let got = run_up(
                    &dev,
                    m,
                    &bank,
                    &rows[..tokens],
                    tokens,
                    &pairs,
                    pairs.len() + 3,
                    grid,
                    1,
                    tpi,
                );
                assert_eq!(
                    bits(&got.act),
                    bits(&reference.act),
                    "{}: the activation with {tpi} tiles for an item at grid {grid}",
                    m.name
                );
                assert_eq!(
                    got.quant, reference.quant,
                    "{}: the quantized rows with {tpi} tiles for an item at grid {grid}",
                    m.name
                );
            }
        }
        // The count of rows: 1, 7 and 40 tokens. Token 0..min(count, 16) must equal the rows of the reference.
        for count in [1usize, 7, 40] {
            let p = standard_pairs(count);
            let got = run_up(&dev, m, &bank, &rows[..count], count, &p, p.len(), 7, 1, TPI);
            for t in 0..count.min(tokens) {
                for e in 0..8 {
                    let (a, b) = (t * 8 + e, t * 8 + e);
                    assert_eq!(
                        bits(&got.act[a * INNER..(a + 1) * INNER]),
                        bits(&reference.act[b * INNER..(b + 1) * INNER]),
                        "{}: the activation of token {t} expert {e} with {count} tokens",
                        m.name
                    );
                    assert_eq!(
                        quant_row_bytes(&got, m.format, a),
                        quant_row_bytes(&reference, m.format, b),
                        "{}: the quantized row of token {t} expert {e} with {count} tokens",
                        m.name
                    );
                }
            }
        }
        // The order of the pairs, and the order of the rows inside a pair: the same slots, so the same bits.
        let mut reversed: Vec<ExpertPair> = pairs.iter().rev().copied().collect();
        for p in &mut reversed {
            let n = p.count as usize;
            p.x_row[..n].reverse();
            p.y_row[..n].reverse();
            p.weight[..n].reverse();
        }
        let got = run_up(
            &dev,
            m,
            &bank,
            &rows[..tokens],
            tokens,
            &reversed,
            reversed.len() + 5,
            24,
            1,
            TPI,
        );
        assert_eq!(
            bits(&got.act),
            bits(&reference.act),
            "{}: the activation with the pairs reversed",
            m.name
        );
        assert_eq!(
            got.quant, reference.quant,
            "{}: the quantized rows with the pairs reversed",
            m.name
        );
        eprintln!(
            "cuda_up: {}: the bits equal at the grids {GRIDS:?}, at 1, 7, 16 and 40 tokens, and with the pairs reversed",
            m.name
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

/// The cuts of an item that the pass may pick: each divisor of the 36 tiles of a pair, and the cut of the pass.
const CUTS: [usize; 9] = [1, 2, 3, 4, 6, 9, 12, 18, 36];

#[test]
fn the_bits_of_each_cut_of_an_item_are_equal_at_1_7_and_16_tokens_and_the_grids_48_and_7() {
    use nagaya_mk_inkling2_sm121::pass::desc::up_tiles_per_item;
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let rows = input_rows(m, 16);
        let bank = upload_bank(&dev, m, true);
        for tokens in [1usize, 7, 16] {
            assert!(
                CUTS.contains(&up_tiles_per_item(tokens)),
                "the cut of the pass is not in the list"
            );
            let pairs = standard_pairs(tokens);
            let reference = run_up(&dev, m, &bank, &rows[..tokens], tokens, &pairs, pairs.len(), 48, 1, TPI);
            for tpi in CUTS {
                for grid in [48u32, 7] {
                    let got = run_up(
                        &dev,
                        m,
                        &bank,
                        &rows[..tokens],
                        tokens,
                        &pairs,
                        pairs.len(),
                        grid,
                        1,
                        tpi,
                    );
                    assert_eq!(
                        bits(&got.act),
                        bits(&reference.act),
                        "{}: the activation of {tokens} tokens with {tpi} tiles for an item at grid {grid}",
                        m.name
                    );
                    assert_eq!(
                        got.quant, reference.quant,
                        "{}: the quantized rows of {tokens} tokens with {tpi} tiles for an item at grid {grid}",
                        m.name
                    );
                }
            }
        }
        dev.free_device(bank.ptr).unwrap();
    }
    eprintln!("cuda_up: the bits equal at the cuts {CUTS:?}, at 1, 7 and 16 tokens, and at the grids 48 and 7");
}

/// The hashes of the standard case (16 tokens, grid 48): the activation rows, then the quantized rows. They are part of the
/// engine identity: they pin the repack of `w13`, the epilogue, and the finisher.
const PINNED_HASHES: [&str; 2] = [
    "478ba73f6a7d06ae9e895610a7fa887baae13c24a2ed30595dddd6b876c33654",
    "a87c2f2f4a7d6f56dbaf3a5bf3acd759e76c71a72bd0ddafab37434150229a5f",
];

#[test]
fn the_bits_of_the_standard_case_are_pinned() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let print_only = std::env::var("CUDA_MOE_PRINT_HASHES").is_ok();
    for (index, m) in matrices.iter().enumerate() {
        let rows = input_rows(m, 16);
        let bank = upload_bank(&dev, m, true);
        let pairs = standard_pairs(16);
        let run = run_up(&dev, m, &bank, &rows, 16, &pairs, pairs.len(), 48, 1, TPI);
        let mut hasher = blake3::Hasher::new();
        for v in &run.act {
            hasher.update(&v.to_le_bytes());
        }
        hasher.update(&run.quant);
        let hash = hasher.finalize().to_hex().to_string();
        if print_only {
            eprintln!("cuda_up pinned: {index} {} {hash}", m.name);
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
fn speed_of_the_up_op_for_8_experts_with_1_row() {
    let Some(matrices) = matrices() else { return };
    let dev = moe_support::open_profile_or_env();
    let blocks = dev.sm_count() as usize;
    let flush_bytes = 1usize << 30;
    let flush = dev.alloc_device(flush_bytes).unwrap();
    let checks = dev.alloc_device(blocks * 4).unwrap();
    for m in &matrices {
        let rows = input_rows(m, 1);
        let bank = upload_bank(&dev, m, true);
        let pairs = bank_pairs(1, weight);
        let x_layout = QuantRowsLayout::new(m.format, 1, m.k).unwrap();
        let layout = QuantRowsLayout::new(m.format, 8, INNER).unwrap();
        let x = upload(&dev, &quant_rows_bytes(&x_layout, &rows));
        let pair_ptr = upload(&dev, &pair_list_bytes(&pairs, pairs.len()).unwrap());
        let act = upload(&dev, &vec![0u8; 8 * INNER * 4]);
        let qact = upload(&dev, &vec![0u8; layout.bytes()]);
        let counters = upload(&dev, &vec![0u32; pairs.len()]);
        let mut table = MkTaskTable::default();
        table.ptr[EXPERT_TABLE_AMAX] = bank.amax;
        table.ptr[EXPERT_TABLE_QACT] = qact;
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
        let up = Task::with_items(up_task(
            m,
            &bank,
            x,
            1,
            1,
            act,
            table_ptr,
            pair_ptr,
            counters,
            pairs.len(),
            TPI,
        ))
        .after(0);
        ring.set(&[read, up]);
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
            let slots = &profile[index * blocks * PROFILE_WORDS..(index + 1) * blocks * PROFILE_WORDS];
            let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
            let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
            spans.push(end - begin);
        }
        let ns = median(spans.clone());
        let weight_bytes = 8 * (N / 64) * m.format.item_bytes(m.k);
        let gbps = weight_bytes as f64 / ns as f64;
        eprintln!(
            "cuda_up speed: {:<14} {:?}: 8 experts x 1 row: {:>7.1} us (best {:>7.1}), {:>6.2} MB, {:>6.1} GB/s, {:>5.1}% of {PEAK_GBPS} GB/s (the budget is 100 us NVFP4, 182 us MXFP8)",
            m.name,
            m.format,
            ns as f64 / 1e3,
            *spans.iter().min().unwrap() as f64 / 1e3,
            weight_bytes as f64 / 1e6,
            gbps,
            gbps / PEAK_GBPS * 100.0
        );
        if m.format == ExpertFormat::Nvfp4 {
            assert!(
                gbps >= 0.8 * PEAK_GBPS,
                "the up op of NVFP4 for 8 experts with 1 row reads {gbps:.1} GB/s, under 80% of {PEAK_GBPS} GB/s; read the profile, and do not lower the bar"
            );
        }
        ring.free().unwrap();
        for ptr in [x, pair_ptr, act, qact, counters, table_ptr, bank.ptr, bank.amax] {
            if ptr != 0 {
                dev.free_device(ptr).unwrap();
            }
        }
    }
    dev.free_device(flush).unwrap();
    dev.free_device(checks).unwrap();
}

// The experts of the gate are the same as the wave A gate: the checkpoint experts at the start, the middle and the end.
const _: [usize; 8] = EXPERTS;

/// The route of the block tests: token `t` goes to expert `e` unless `(7 t + 13 e) % 5 == 0`, expert 3 takes no row, and
/// expert 5 takes every row. Expert row counts are then uneven, so a block meets the ends of 8, 16 and 32 rows.
fn uneven(tokens: usize) -> Vec<ExpertPair> {
    uneven_pairs(tokens, |t, e| e != 3 && (e == 5 || (7 * t + 13 * e) % 5 != 0), weight)
}

#[test]
fn the_block_form_gives_the_bits_of_the_decode_form() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let bank = upload_bank(&dev, m, true);
        for tokens in [17usize, 33, 64, 128, 512] {
            let rows = input_rows(m, tokens);
            for pairs in [standard_pairs(tokens), uneven(tokens)] {
                let bound = nagaya_mk_cuda::protocol::expert_pair_bound(tokens).max(pairs.len());
                let want = run_up(&dev, m, &bank, &rows, tokens, &pairs, bound, 48, 1, TPI);
                for (grid, tpi) in [(48u32, 3usize), (24, 3), (7, 6), (1, 9), (48, 1)] {
                    let got = run_up_form(&dev, m, &bank, &rows, tokens, &pairs, bound, grid, 2, tpi, true);
                    assert!(
                        got.counters.iter().all(|c| *c == 0),
                        "{}: a counter was left at {tokens} rows",
                        m.name
                    );
                    assert_eq!(
                        bits(&got.act),
                        bits(&want.act),
                        "{}: act differs at {tokens} rows, grid {grid}, tpi {tpi}",
                        m.name
                    );
                    assert!(
                        got.quant == want.quant,
                        "{}: the quantized rows differ at {tokens} rows, grid {grid}, tpi {tpi}",
                        m.name
                    );
                }
            }
        }
    }
}
