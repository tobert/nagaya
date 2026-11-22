//! GPU gate of the expert ops `OP_EXPERT_NVFP4` and `OP_EXPERT_MXFP8` (milestone 37 step 9, wave A), with the real
//! weights of Inkling 2 Mini: layer 0 (NVFP4) and layer 20 (MXFP8), the gate and up matrix `w13` and the down matrix
//! `w2`, for 8 experts.
//!
//! The gates:
//! - Accuracy: each output is within a relative bound of the dot product of `quant.rs`. The bound is a share of the sum
//!   of the absolute terms.
//! - The bits of a row do not depend on the grid, on the count of rows of its pair, or on its place in the pair.
//! - The quantizer and the expert op run in one ring, and give the bits of the host quantizer and the same bits.
//! - Speed: the time of 8 experts with 1 row, as a decode step has them, in GB/s of the weights and the scales.
//!
//! The tests skip with one line when `INKLING2_MODEL_DIR` is absent. `NAGAYA_REQUIRE_MODELS=1` makes an absence fail.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder of Mini> cargo test -p nagaya-mk-inkling2-sm121
//!  --features cuda --test cuda_experts -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use common::gpu::{GRIDS, download, open, open_profile, run_with_watchdog, upload};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::experts::{ExpertFormat, QuantRowsLayout, repack_expert};
use nagaya_mk_cuda::protocol::{
    EXPERT_MAX_ROWS, ExpertPair, MkTaskDescriptor, OP_EXPERT_MXFP8, OP_EXPERT_NVFP4, OP_PROBE_READ, OP_QUANT_MXFP8,
    OP_QUANT_NVFP4, PROBE_ITEM_BYTES, PROBE_READ_NC, PROFILE_WORDS,
};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::safetensors::SafeTensorsHeader;
use nagaya_mk_inkling2_sm121::quant::{
    E2M1, Mxfp8Row, Nvfp4Row, decode_e4m3, decode_ue8m0, dot_mxfp8_block_scaled, dot_nvfp4_block_scaled,
    quantize_mxfp8_ue8m0, quantize_nvfp4_fouroversix_mae,
};

/// The experts of the gate. They are at the start, the middle and the end of the 128.
const EXPERTS: [usize; 8] = [0, 17, 31, 42, 64, 85, 99, 127];

/// The checkpoint rows of one expert matrix.
const N: usize = 2304;

/// The relative bound of the accuracy gate: the error of an output over the sum of the absolute terms of its dot
/// product, in the units of the result. The probe measured at most 7.6e-8 (NVFP4) and 1.42e-7 (MXFP8) for one MMA.
/// An output sums 36 MMAs (NVFP4, K 2304) and the tensor cores do not round as the host does. The worst error of this
/// gate is printed by the test, and the bound is a few times that number.
const BOUND: f64 = 4.0e-7;

/// Peak read bandwidth that the probes measured (docs/engine-abi.md "The facts of the GB10"), in GB/s.
const PEAK_GBPS: f64 = 252.0;

/// The weights of a matrix: what the gate needs of one matrix of the layer.
struct Matrix {
    format: ExpertFormat,
    k: usize,
    /// The codes and scales of each expert of `EXPERTS`, as the checkpoint has them.
    codes: Vec<Vec<u8>>,
    scales: Vec<Vec<u8>>,
    /// NVFP4 only: the amax of each expert, and the denominator of the matrix.
    amax: Vec<f32>,
    denominator: f32,
    name: String,
}

fn model_dir() -> Option<PathBuf> {
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(dir) if PathBuf::from(&dir).join("model-00001-of-00001.safetensors").is_file() => Some(dir.into()),
        _ => {
            let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
            assert!(
                !require,
                "NAGAYA_REQUIRE_MODELS=1 and INKLING2_MODEL_DIR holds no model-00001-of-00001.safetensors; set the variable to the folder of Inkling 2 Mini"
            );
            eprintln!("cuda_experts: skipped: INKLING2_MODEL_DIR holds no model of Inkling 2 Mini");
            None
        }
    }
}

struct Checkpoint {
    header: SafeTensorsHeader,
    file: File,
}

impl Checkpoint {
    fn open(dir: &std::path::Path) -> Self {
        let path = dir.join("model-00001-of-00001.safetensors");
        let header = SafeTensorsHeader::read(&path).unwrap_or_else(|e| panic!("{e}"));
        let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        Self { header, file }
    }

    /// `len` bytes of the tensor `name`, from `offset` bytes into it.
    fn range(&mut self, name: &str, offset: usize, len: usize) -> Vec<u8> {
        let meta = self
            .header
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"));
        assert!(
            offset + len <= meta.nbytes as usize,
            "{name}: the range passes the end of the tensor"
        );
        self.file
            .seek(SeekFrom::Start(
                self.header.data_start + meta.data_offset + offset as u64,
            ))
            .unwrap();
        let mut bytes = vec![0u8; len];
        self.file
            .read_exact(&mut bytes)
            .unwrap_or_else(|e| panic!("cannot read {name}: {e}"));
        bytes
    }

    fn f32s(&mut self, name: &str, offset: usize, count: usize) -> Vec<f32> {
        self.range(name, offset * 4, count * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// One matrix of the experts of `layer`: `w13` or `w2`.
    fn matrix(&mut self, layer: usize, which: &str, format: ExpertFormat, k: usize) -> Matrix {
        let prefix = format!("model.llm.layers.{layer}.mlp.experts.{which}");
        let (code_row, scale_row) = (format.code_bytes(k), format.scale_bytes(k));
        let mut m = Matrix {
            format,
            k,
            codes: Vec::new(),
            scales: Vec::new(),
            amax: Vec::new(),
            denominator: 1.0,
            name: format!("layer {layer} {which}"),
        };
        for &e in &EXPERTS {
            m.codes
                .push(self.range(&format!("{prefix}_weight"), e * N * code_row, N * code_row));
            m.scales
                .push(self.range(&format!("{prefix}_weight_scale"), e * N * scale_row, N * scale_row));
            if format == ExpertFormat::Nvfp4 {
                m.amax.push(self.f32s(&format!("{prefix}_weight_amax"), e, 1)[0]);
            }
        }
        if format == ExpertFormat::Nvfp4 {
            m.denominator = self.f32s(&format!("{prefix}_weight_outer_scale_denominator"), 0, 1)[0];
        }
        m
    }
}

/// The four matrices of the gate.
fn matrices() -> Option<Vec<Matrix>> {
    let dir = model_dir()?;
    let mut checkpoint = Checkpoint::open(&dir);
    Some(vec![
        checkpoint.matrix(0, "w13", ExpertFormat::Nvfp4, 2304),
        checkpoint.matrix(0, "w2", ExpertFormat::Nvfp4, 1152),
        checkpoint.matrix(20, "w13", ExpertFormat::Mxfp8, 2304),
        checkpoint.matrix(20, "w2", ExpertFormat::Mxfp8, 1152),
    ])
}

/// A generator of rows that is the same in each run: xorshift.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// An activation row: a sum of 3 uniform values, with a few large outliers, like a normalized hidden row.
fn activation_row(rng: &mut Rng, k: usize, magnitude: f32) -> Vec<f32> {
    let mut row: Vec<f32> = (0..k)
        .map(|_| (rng.unit() + rng.unit() + rng.unit()) * magnitude)
        .collect();
    for _ in 0..3 {
        let i = rng.next() as usize % k;
        row[i] *= 25.0;
    }
    row
}

enum QuantRow {
    Nvfp4(Nvfp4Row),
    Mxfp8(Mxfp8Row),
}

fn quantize_row(format: ExpertFormat, row: &[f32]) -> QuantRow {
    match format {
        ExpertFormat::Nvfp4 => QuantRow::Nvfp4(quantize_nvfp4_fouroversix_mae(row).unwrap()),
        ExpertFormat::Mxfp8 => QuantRow::Mxfp8(quantize_mxfp8_ue8m0(row).unwrap()),
    }
}

/// The bytes of a buffer of quantized rows, in the layout of the quantizer op.
fn quant_rows_bytes(layout: &QuantRowsLayout, rows: &[QuantRow]) -> Vec<u8> {
    let mut bytes = vec![0u8; layout.bytes()];
    for (r, row) in rows.iter().enumerate() {
        let (codes, scales) = match row {
            QuantRow::Nvfp4(q) => (&q.codes, &q.scales),
            QuantRow::Mxfp8(q) => (&q.codes, &q.scales),
        };
        bytes[layout.codes_offset(r)..][..codes.len()].copy_from_slice(codes);
        bytes[layout.scales_offset(r)..][..scales.len()].copy_from_slice(scales);
        if let QuantRow::Nvfp4(q) = row {
            bytes[layout.amax_offset(r)..][..4].copy_from_slice(&q.amax.to_le_bytes());
        }
    }
    bytes
}

/// A pair for the op: the bank index of the expert, and the rows with the slots that they write.
fn pair(expert: usize, x_rows: &[usize], first_slot: usize) -> ExpertPair {
    assert!(!x_rows.is_empty() && x_rows.len() <= EXPERT_MAX_ROWS);
    let mut p = ExpertPair {
        expert: expert as u32,
        count: x_rows.len() as u32,
        ..Default::default()
    };
    for (i, &r) in x_rows.iter().enumerate() {
        p.x_row[i] = r as u16;
        p.y_row[i] = (first_slot + i) as u16;
    }
    p
}

/// The bank on the device, and what the op needs of the matrix.
struct Bank {
    ptr: CUdeviceptr,
    amax: CUdeviceptr,
    bytes: usize,
}

fn upload_bank(dev: &CudaDevice, m: &Matrix) -> Bank {
    let mut bytes = Vec::new();
    for (codes, scales) in m.codes.iter().zip(&m.scales) {
        bytes.extend(repack_expert(m.format, N, m.k, codes, scales).unwrap());
    }
    let amax = if m.format == ExpertFormat::Nvfp4 {
        upload(dev, &m.amax)
    } else {
        0
    };
    Bank {
        ptr: upload(dev, &bytes),
        amax,
        bytes: bytes.len(),
    }
}

/// The descriptor of an expert op over `pairs`.
#[allow(clippy::too_many_arguments)]
fn expert_task(
    m: &Matrix,
    bank: &Bank,
    x: CUdeviceptr,
    x_rows: usize,
    y: CUdeviceptr,
    pairs: CUdeviceptr,
    count: usize,
    expert_scale: f32,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode: match m.format {
            ExpertFormat::Nvfp4 => OP_EXPERT_NVFP4,
            ExpertFormat::Mxfp8 => OP_EXPERT_MXFP8,
        },
        items: (count * N / 64) as u32,
        rows: x_rows as u32,
        a: x,
        b: bank.ptr,
        c: y,
        d: bank.amax,
        f: pairs,
        n: N as u32,
        k: m.k as u32,
        p0: expert_scale,
        p1: m.denominator,
        ..Default::default()
    }
}

/// Run the expert op over `pairs` at `grid`, and return the `slots` rows of `N` floats that it wrote. The slots start
/// as a NaN, so that a slot that the op does not write shows.
#[allow(clippy::too_many_arguments)]
fn run_experts(
    dev: &CudaDevice,
    m: &Matrix,
    bank: &Bank,
    x_bytes: &[u8],
    x_rows: usize,
    pairs: &[ExpertPair],
    slots: usize,
    grid: u32,
) -> Vec<f32> {
    run_experts_scaled(dev, m, bank, x_bytes, x_rows, pairs, slots, grid, 1.0)
}

/// [`run_experts`] with an expert scale that is not 1.0.
#[allow(clippy::too_many_arguments)]
fn run_experts_scaled(
    dev: &CudaDevice,
    m: &Matrix,
    bank: &Bank,
    x_bytes: &[u8],
    x_rows: usize,
    pairs: &[ExpertPair],
    slots: usize,
    grid: u32,
    expert_scale: f32,
) -> Vec<f32> {
    let x = upload(dev, x_bytes);
    let pair_ptr = upload(dev, pairs);
    let y = upload(dev, &vec![0xFFu8; slots * N * 4]);
    let mut ring = Ring::new(dev, 1, false).unwrap();
    ring.set(&[
        Task::with_items(expert_task(m, bank, x, x_rows, y, pair_ptr, pairs.len(), expert_scale))
            .reads(x, x_bytes.len())
            .reads(bank.ptr, bank.bytes)
            .writes(y, slots * N * 4),
    ]);
    ring.zero_device().unwrap();
    run_with_watchdog(
        &ring,
        grid,
        DEFAULT_BLOCK_THREADS,
        dev.max_dynamic_shared_bytes().unwrap(),
    );
    let out = download::<f32>(dev, y, slots * N);
    ring.free().unwrap();
    for ptr in [x, pair_ptr, y] {
        dev.free_device(ptr).unwrap();
    }
    out
}

/// The nibble `i` of a packed row.
fn nibble(codes: &[u8], i: usize) -> usize {
    usize::from(if i.is_multiple_of(2) {
        codes[i / 2] & 0xF
    } else {
        codes[i / 2] >> 4
    })
}

/// The host dot product of `quant.rs`, and the sum of the absolute terms in the units of the result.
fn host_dot(m: &Matrix, e: usize, col: usize, row: &QuantRow) -> (f32, f64) {
    let k = m.k;
    let (code_row, scale_row) = (m.format.code_bytes(k), m.format.scale_bytes(k));
    let w_codes = &m.codes[e][col * code_row..(col + 1) * code_row];
    let w_scales = &m.scales[e][col * scale_row..(col + 1) * scale_row];
    match row {
        QuantRow::Nvfp4(q) => {
            let value = dot_nvfp4_block_scaled(q, w_codes, w_scales, m.amax[e], m.denominator, 1.0).unwrap();
            let mut total = 0.0f64;
            for b in 0..scale_row {
                let scale = f64::from(decode_e4m3(q.scales[b]).unwrap()) * f64::from(decode_e4m3(w_scales[b]).unwrap());
                let block: f64 = (b * 16..(b + 1) * 16)
                    .map(|i| f64::from(E2M1[nibble(&q.codes, i)] * E2M1[nibble(w_codes, i)]).abs())
                    .sum();
                total += block * scale.abs();
            }
            let row_scale = f64::from(q.amax) * f64::from(m.amax[e]) / (1536.0 * f64::from(m.denominator));
            (value, total * row_scale.abs())
        }
        QuantRow::Mxfp8(q) => {
            let value = dot_mxfp8_block_scaled(q, w_codes, w_scales, 1.0).unwrap();
            let mut total = 0.0f64;
            for b in 0..scale_row {
                let scale =
                    f64::from(decode_ue8m0(q.scales[b]).unwrap()) * f64::from(decode_ue8m0(w_scales[b]).unwrap());
                let block: f64 = (b * 32..(b + 1) * 32)
                    .map(|i| {
                        (f64::from(decode_e4m3(q.codes[i]).unwrap()) * f64::from(decode_e4m3(w_codes[i]).unwrap()))
                            .abs()
                    })
                    .sum();
                total += block * scale;
            }
            (value, total)
        }
    }
}

/// The rows and the pairs of the accuracy gate: 16 quantized rows of 3 magnitudes, and 8 pairs with 1 to 16 rows that
/// share the rows in several ways.
fn standard_case(m: &Matrix) -> (QuantRowsLayout, Vec<QuantRow>, Vec<ExpertPair>, usize) {
    let mut rng = Rng(0x5EED_0000 + m.k as u64);
    let rows: Vec<QuantRow> = (0..16)
        .map(|i| {
            let magnitude = [1.0f32, 0.05, 20.0][i % 3];
            quantize_row(m.format, &activation_row(&mut rng, m.k, magnitude))
        })
        .collect();
    let layout = QuantRowsLayout::new(m.format, 16, m.k).unwrap();
    let counts = [1usize, 2, 3, 5, 8, 9, 13, 16];
    let mut pairs = Vec::new();
    let mut slot = 0;
    for (p, &count) in counts.iter().enumerate() {
        let x_rows: Vec<usize> = (0..count).map(|j| (3 * p + j) % 16).collect();
        pairs.push(pair(p, &x_rows, slot));
        slot += count;
    }
    (layout, rows, pairs, slot)
}

/// Check the outputs of the standard case of `m` against the host dot products.
fn check_accuracy(dev: &CudaDevice, m: &Matrix) {
    let (layout, rows, pairs, slots) = standard_case(m);
    let bank = upload_bank(dev, m);
    let x_bytes = quant_rows_bytes(&layout, &rows);
    let y = run_experts(dev, m, &bank, &x_bytes, 16, &pairs, slots, 48);
    let mut worst = (0.0f64, 0usize, 0usize, 0usize);
    for p in &pairs {
        for i in 0..p.count as usize {
            let slot = p.y_row[i] as usize;
            for col in 0..N {
                let (want, terms) = host_dot(m, p.expert as usize, col, &rows[p.x_row[i] as usize]);
                let got = y[slot * N + col];
                assert!(
                    got.is_finite(),
                    "{}: slot {slot} column {col} was not written or is not finite",
                    m.name
                );
                let error = (f64::from(got) - f64::from(want)).abs();
                let ratio = if terms > 0.0 { error / terms } else { error };
                if ratio > worst.0 {
                    worst = (ratio, p.expert as usize, slot, col);
                }
                // The sums of NVFP4 on this data are exact in f32 (the scales of a row span a few binades), so the
                // MMA and the host agree to the last bit. Any change of the epilogue breaks the equality.
                assert!(
                    m.format != ExpertFormat::Nvfp4 || error == 0.0,
                    "{}: expert {} slot {slot} column {col}: gpu {got:e}, host {want:e}; the NVFP4 sums of this data are exact, so the bits must be equal",
                    m.name,
                    p.expert
                );
                assert!(
                    ratio <= BOUND,
                    "{}: expert {} slot {slot} column {col}: gpu {got:e}, host {want:e}, error {error:e} is {ratio:e} of the terms {terms:e}; the bound is {BOUND:e}",
                    m.name,
                    p.expert
                );
            }
        }
    }
    eprintln!(
        "cuda_experts: {}: {} outputs within {BOUND:e} of the sum of the terms; the worst is {:e} (expert {}, slot {}, column {})",
        m.name,
        slots * N,
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    dev.free_device(bank.ptr).unwrap();
}

#[test]
fn the_expert_ops_are_within_the_bound_of_the_host_dot_products() {
    let Some(mut matrices) = matrices() else { return };
    let dev = open();
    for m in &mut matrices {
        check_accuracy(&dev, m);
        if m.format == ExpertFormat::Nvfp4 {
            // Denominators that are not powers of two, so that the order of the divisions of the epilogue shows. A
            // single value can hide a change, because two roundings can agree.
            let real = m.denominator;
            for denominator in [3.7f32, 11.3, 0.77, 5.1] {
                m.denominator = denominator;
                check_accuracy(&dev, m);
            }
            m.denominator = real;
        }
    }
}

#[test]
fn the_bits_do_not_depend_on_the_grid_the_count_of_rows_or_the_place_in_the_pair() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let (layout, rows, pairs, slots) = standard_case(m);
        let bank = upload_bank(&dev, m);
        let x_bytes = quant_rows_bytes(&layout, &rows);
        let reference = run_experts(&dev, m, &bank, &x_bytes, 16, &pairs, slots, 48);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        // 1. The grid.
        for grid in GRIDS {
            let got = run_experts(&dev, m, &bank, &x_bytes, 16, &pairs, slots, grid);
            assert_eq!(bits(&got), bits(&reference), "{}: the bits at grid {grid}", m.name);
        }
        // 2. The count of rows: each row of each pair, alone in a pair of one row.
        let singles: Vec<(ExpertPair, usize, usize)> = pairs
            .iter()
            .flat_map(|p| (0..p.count as usize).map(move |i| (*p, i, p.y_row[i] as usize)))
            .collect();
        let alone_pairs: Vec<ExpertPair> = singles
            .iter()
            .enumerate()
            .map(|(s, (p, i, _))| pair(p.expert as usize, &[p.x_row[*i] as usize], s))
            .collect();
        let alone = run_experts(&dev, m, &bank, &x_bytes, 16, &alone_pairs, alone_pairs.len(), 48);
        for (s, (_, _, slot)) in singles.iter().enumerate() {
            assert_eq!(
                bits(&alone[s * N..(s + 1) * N]),
                bits(&reference[slot * N..(slot + 1) * N]),
                "{}: slot {slot} alone and in its pair",
                m.name
            );
        }
        // 3. The place in the pair: the rows of each pair in the reverse order, and on the second n8 tile.
        let reversed: Vec<ExpertPair> = pairs
            .iter()
            .map(|p| {
                let mut x: Vec<usize> = (0..p.count as usize).map(|i| p.x_row[i] as usize).collect();
                x.reverse();
                pair(p.expert as usize, &x, p.y_row[0] as usize)
            })
            .collect();
        let rev = run_experts(&dev, m, &bank, &x_bytes, 16, &reversed, slots, 7);
        for (p, r) in pairs.iter().zip(&reversed) {
            for i in 0..p.count as usize {
                let (a, b) = (p.y_row[i] as usize, r.y_row[p.count as usize - 1 - i] as usize);
                assert_eq!(
                    bits(&rev[b * N..(b + 1) * N]),
                    bits(&reference[a * N..(a + 1) * N]),
                    "{}: expert {} row {i} in the reverse order",
                    m.name,
                    p.expert
                );
            }
        }
        eprintln!(
            "cuda_experts: {}: the bits equal at the grids {GRIDS:?}, with {} rows alone, and with the rows reversed",
            m.name,
            singles.len()
        );
        dev.free_device(bank.ptr).unwrap();
    }
}

#[test]
fn the_device_quantizer_and_the_expert_op_in_one_ring_give_the_bits_of_the_host_quantizer() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in &matrices {
        let (layout, rows, pairs, slots) = standard_case(m);
        let bank = upload_bank(&dev, m);
        let host_bytes = quant_rows_bytes(&layout, &rows);
        let reference = run_experts(&dev, m, &bank, &host_bytes, 16, &pairs, slots, 48);
        // The input rows in f32, quantized by the op in the ring before the expert op reads them.
        let mut rng = Rng(0x5EED_0000 + m.k as u64);
        let input: Vec<f32> = (0..16)
            .flat_map(|i| activation_row(&mut rng, m.k, [1.0f32, 0.05, 20.0][i % 3]))
            .collect();
        let (a, x, pair_ptr) = (
            upload(&dev, &input),
            upload(&dev, &vec![0u8; layout.bytes()]),
            upload(&dev, &pairs),
        );
        let y = upload(&dev, &vec![0xFFu8; slots * N * 4]);
        let mut ring = Ring::new(&dev, 1, false).unwrap();
        let quant = Task::with_items(MkTaskDescriptor {
            opcode: match m.format {
                ExpertFormat::Nvfp4 => OP_QUANT_NVFP4,
                ExpertFormat::Mxfp8 => OP_QUANT_MXFP8,
            },
            items: 16,
            rows: 16,
            a,
            c: x,
            k: m.k as u32,
            ..Default::default()
        })
        .reads(a, input.len() * 4)
        .writes(x, layout.bytes());
        let experts = Task::with_items(expert_task(m, &bank, x, 16, y, pair_ptr, pairs.len(), 1.0))
            .reads(x, layout.bytes())
            .reads(bank.ptr, bank.bytes)
            .writes(y, slots * N * 4);
        ring.set(&[quant, experts]);
        for grid in GRIDS {
            dev.memset_zero(y, slots * N * 4).unwrap();
            ring.zero_device().unwrap();
            run_with_watchdog(
                &ring,
                grid,
                DEFAULT_BLOCK_THREADS,
                dev.max_dynamic_shared_bytes().unwrap(),
            );
            let got: Vec<f32> = download(&dev, y, slots * N);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{}: the quantizer and the expert op in one ring at grid {grid}",
                m.name
            );
        }
        eprintln!(
            "cuda_experts: {}: quantizer then expert op in one ring: the bits equal the host quantizer at {GRIDS:?}",
            m.name
        );
        ring.free().unwrap();
        for ptr in [a, x, pair_ptr, y, bank.ptr] {
            dev.free_device(ptr).unwrap();
        }
    }
}

#[test]
fn the_expert_scale_multiplies_the_result_and_the_bits_are_pinned() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    // The bits of the standard case, as a BLAKE3 hash of the words of the output at the grid 48. The hash pins the cut
    // of K, the order of the sums and the epilogue: they are part of the engine identity, and a change of one of them
    // changes the hash. Change the numbers here only with that change of identity.
    let print_only = std::env::var("CUDA_EXPERTS_PRINT_HASHES").is_ok();
    let want_hashes = PINNED_HASHES;
    for (index, m) in matrices.iter().enumerate() {
        let (layout, rows, pairs, slots) = standard_case(m);
        let bank = upload_bank(&dev, m);
        let x_bytes = quant_rows_bytes(&layout, &rows);
        let one = run_experts(&dev, m, &bank, &x_bytes, 16, &pairs, slots, 48);
        // A scale of 0.5 is a power of two, so it moves each result by an exact factor.
        let half = run_experts_scaled(&dev, m, &bank, &x_bytes, 16, &pairs, slots, 48, 0.5);
        for (i, (a, b)) in one.iter().zip(&half).enumerate() {
            assert_eq!(
                (a * 0.5).to_bits(),
                b.to_bits(),
                "{}: output {i}: the expert scale of 0.5 gives {b:e}, and half of the result is {:e}",
                m.name,
                a * 0.5
            );
        }
        let mut bytes = Vec::new();
        for v in &one {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let hash = blake3::hash(&bytes).to_hex().to_string();
        if print_only {
            eprintln!("cuda_experts pinned: {index} {} {hash}", m.name);
        } else {
            assert_eq!(
                hash, want_hashes[index],
                "{}: the bits of the standard case changed",
                m.name
            );
        }
        dev.free_device(bank.ptr).unwrap();
    }
}

/// The hashes of the output of the standard case of each matrix, in the order of `matrices`.
const PINNED_HASHES: [&str; 4] = [
    "c23198c27e0c2ce0177b525865625435cf5a9be1d62a036973404a4fd57b2a4a",
    "e6e6e5e492e6e4cfda874efabc657e851685e48748699f1f29f8752a86a10bf4",
    "9cfc355160146501310bd31ac89765903ef718d1058f45b8d889305242cf374c",
    "2458719067278fce60a98f19f5babd1a9a86e46e248b4d2eff2eb15b5650694c",
];

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// The nanoseconds from the first block that passes the wait of the task `task` to the last block that ends it.
fn task_ns(profile: &[u64], blocks: usize, task: usize) -> u64 {
    let slots = &profile[task * blocks * PROFILE_WORDS..(task + 1) * blocks * PROFILE_WORDS];
    let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
    let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
    assert!(end > begin, "task {task} has no time");
    end - begin
}

#[test]
fn speed_of_8_experts_with_1_row() {
    let Some(matrices) = matrices() else { return };
    let dev = open_profile();
    let blocks = dev.sm_count() as usize;
    // A task that reads 1 GiB before the expert op, so that the weights come from memory and not from L2.
    let flush_bytes = 1usize << 30;
    let flush = dev.alloc_device(flush_bytes).unwrap();
    let checks = dev.alloc_device(blocks * 4).unwrap();
    let mut table = Vec::new();
    for m in &matrices {
        let (layout, rows, _, _) = standard_case(m);
        let bank = upload_bank(&dev, m);
        let x_bytes = quant_rows_bytes(&layout, &rows);
        // The decode step of one layer: each of 8 experts takes the one row.
        let pairs: Vec<ExpertPair> = (0..8).map(|e| pair(e, &[0], e)).collect();
        let x = upload(&dev, &x_bytes);
        let pair_ptr = upload(&dev, &pairs);
        let y = upload(&dev, &vec![0u8; 8 * N * 4]);
        let mut ring = Ring::new(&dev, 1, true).unwrap();
        let read = Task::per_block(MkTaskDescriptor {
            opcode: OP_PROBE_READ,
            a: flush,
            c: checks,
            n: (flush_bytes / PROBE_ITEM_BYTES) as u32,
            k: u32::from(PROBE_READ_NC),
            ..Default::default()
        });
        let experts = Task::with_items(expert_task(m, &bank, x, 16, y, pair_ptr, pairs.len(), 1.0)).after(0);
        ring.set(&[read, experts]);
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
            spans.push(task_ns(&ring.read_profile().unwrap(), blocks, index));
        }
        let ns = median(spans.clone());
        let weight_bytes = 8 * (N / 64) * m.format.item_bytes(m.k);
        let gbps = weight_bytes as f64 / ns as f64;
        table.push(format!(
            "{:<18} {:?}: 8 experts x 1 row: {:>7.1} us (best {:>7.1}), {:>6.2} MB of weights and scales, {:>6.1} GB/s, {:>5.1}% of {PEAK_GBPS} GB/s",
            m.name,
            m.format,
            ns as f64 / 1e3,
            *spans.iter().min().unwrap() as f64 / 1e3,
            weight_bytes as f64 / 1e6,
            gbps,
            gbps / PEAK_GBPS * 100.0
        ));
        if m.name == "layer 0 w13" {
            assert!(
                gbps >= 0.7 * PEAK_GBPS,
                "the up matrix of NVFP4 for 8 experts with 1 row reads {gbps:.1} GB/s, under 70% of {PEAK_GBPS} GB/s; read the profile, and do not lower the bar"
            );
        }
        ring.free().unwrap();
        for ptr in [x, pair_ptr, y, bank.ptr, bank.amax] {
            if ptr != 0 {
                dev.free_device(ptr).unwrap();
            }
        }
    }
    for line in &table {
        eprintln!("cuda_experts speed: {line}");
    }
    dev.free_device(flush).unwrap();
    dev.free_device(checks).unwrap();
}
