//! GPU gate of the device quantizers `OP_QUANT_NVFP4` and `OP_QUANT_MXFP8` (milestone 37 step 9, wave A). The bytes
//! equal the bytes of the vLLM kernels on the 154 rows of the fixture, and the bytes of `quant.rs` on random rows of
//! several magnitudes, at the grids 48, 24, 7 and 1.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_quant -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use common::Fixture;
use common::gpu::{GRIDS, download, open, run_with_watchdog, upload};
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::experts::{ExpertFormat, QuantRowsLayout};
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_QUANT_MXFP8, OP_QUANT_NVFP4};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_inkling2_sm121::quant::{quantize_mxfp8_ue8m0, quantize_nvfp4_fouroversix_mae};

/// The bytes that a quantizer gave: for each row, the codes, the scales and (NVFP4) the amax bits.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Quantized {
    codes: Vec<Vec<u8>>,
    scales: Vec<Vec<u8>>,
    amax: Vec<u32>,
}

/// Quantize the rows on the GPU at the grid `grid`.
fn quantize_on_gpu(dev: &CudaDevice, format: ExpertFormat, rows: &[Vec<f32>], grid: u32) -> Quantized {
    let k = rows[0].len();
    let layout = QuantRowsLayout::new(format, rows.len(), k).unwrap();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let input = upload(dev, &flat);
    // The output starts with a byte that no quantizer gives for all of a row, so that a byte left unwritten shows.
    let out = upload(dev, &vec![0xA5u8; layout.bytes()]);
    let mut ring = Ring::new(dev, 1, false).unwrap();
    let opcode = match format {
        ExpertFormat::Nvfp4 => OP_QUANT_NVFP4,
        ExpertFormat::Mxfp8 => OP_QUANT_MXFP8,
    };
    ring.set(&[Task::with_items(MkTaskDescriptor {
        opcode,
        items: rows.len() as u32,
        rows: rows.len() as u32,
        a: input,
        c: out,
        k: k as u32,
        ..Default::default()
    })
    .reads(input, flat.len() * 4)
    .writes(out, layout.bytes())]);
    ring.zero_device().unwrap();
    run_with_watchdog(
        &ring,
        grid,
        DEFAULT_BLOCK_THREADS,
        dev.max_dynamic_shared_bytes().unwrap(),
    );
    let bytes: Vec<u8> = download(dev, out, layout.bytes());
    let mut got = Quantized {
        codes: Vec::new(),
        scales: Vec::new(),
        amax: Vec::new(),
    };
    for r in 0..rows.len() {
        let codes = layout.codes_offset(r);
        got.codes.push(bytes[codes..codes + format.code_bytes(k)].to_vec());
        let scales = layout.scales_offset(r);
        got.scales.push(bytes[scales..scales + format.scale_bytes(k)].to_vec());
        if format == ExpertFormat::Nvfp4 {
            let at = layout.amax_offset(r);
            got.amax.push(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
        }
    }
    ring.free().unwrap();
    dev.free_device(input).unwrap();
    dev.free_device(out).unwrap();
    got
}

/// The bytes that `quant.rs` gives for the rows.
fn quantize_on_host(format: ExpertFormat, rows: &[Vec<f32>]) -> Quantized {
    let mut want = Quantized {
        codes: Vec::new(),
        scales: Vec::new(),
        amax: Vec::new(),
    };
    for row in rows {
        match format {
            ExpertFormat::Nvfp4 => {
                let q = quantize_nvfp4_fouroversix_mae(row).unwrap();
                want.codes.push(q.codes);
                want.scales.push(q.scales);
                want.amax.push(q.amax.to_bits());
            }
            ExpertFormat::Mxfp8 => {
                let q = quantize_mxfp8_ue8m0(row).unwrap();
                want.codes.push(q.codes);
                want.scales.push(q.scales);
            }
        }
    }
    want
}

/// Panic with the first difference: the row, the block and the bytes.
#[track_caller]
fn assert_same(what: &str, rows: &[Vec<f32>], format: ExpertFormat, got: &Quantized, want: &Quantized) {
    let block = format.scale_block();
    for r in 0..rows.len() {
        let code_block = format.code_bytes(block);
        if let Some(i) = (0..got.codes[r].len()).find(|&i| got.codes[r][i] != want.codes[r][i]) {
            let b = i / code_block;
            panic!(
                "{what}: row {r}, block {b}: codes differ at byte {i}: gpu 0x{:02x}, want 0x{:02x}; the block was {:?}, \
                 the scales are gpu 0x{:02x}, want 0x{:02x}",
                got.codes[r][i],
                want.codes[r][i],
                &rows[r][b * block..(b + 1) * block],
                got.scales[r][b],
                want.scales[r][b]
            );
        }
        if let Some(b) = (0..got.scales[r].len()).find(|&b| got.scales[r][b] != want.scales[r][b]) {
            panic!(
                "{what}: row {r}, block {b}: scales differ: gpu 0x{:02x}, want 0x{:02x}; the block was {:?}",
                got.scales[r][b],
                want.scales[r][b],
                &rows[r][b * block..(b + 1) * block]
            );
        }
    }
    assert_eq!(got.amax.len(), want.amax.len(), "{what}: the count of amax words");
    for r in 0..got.amax.len() {
        assert_eq!(
            got.amax[r],
            want.amax[r],
            "{what}: row {r}: amax bits (gpu {:e}, want {:e})",
            f32::from_bits(got.amax[r]),
            f32::from_bits(want.amax[r])
        );
    }
}

/// The bytes of the shipped kernel for a set of the fixture, in the form of `Quantized`.
fn fixture_bytes(f: &Fixture, set: &str, format: ExpertFormat) -> Quantized {
    let (values, vshape) = f.raw(&format!("{set}.values"));
    let (scales, sshape) = f.raw(&format!("{set}.scales"));
    let rows = vshape[0];
    let mut q = Quantized {
        codes: (0..rows)
            .map(|r| values[r * vshape[1]..(r + 1) * vshape[1]].to_vec())
            .collect(),
        scales: (0..rows)
            .map(|r| scales[r * sshape[1]..(r + 1) * sshape[1]].to_vec())
            .collect(),
        amax: Vec::new(),
    };
    if format == ExpertFormat::Nvfp4 {
        let (amax, _) = f.raw(&format!("{set}.amax"));
        q.amax = (0..rows)
            .map(|r| u32::from_le_bytes(amax[4 * r..4 * r + 4].try_into().unwrap()))
            .collect();
    }
    q
}

#[test]
fn the_device_quantizers_give_the_bytes_of_the_vllm_kernels_on_the_rows_of_the_fixture() {
    let dev = open();
    let f = Fixture::load();
    let sets = [
        ("nvfp4_k2304", ExpertFormat::Nvfp4),
        ("nvfp4_k1152", ExpertFormat::Nvfp4),
        ("mxfp8_k2304", ExpertFormat::Mxfp8),
    ];
    let mut total = 0;
    for (set, format) in sets {
        let rows = f.rows_f32(&format!("{set}.x"));
        let want = fixture_bytes(&f, set, format);
        for grid in GRIDS {
            let got = quantize_on_gpu(&dev, format, &rows, grid);
            assert_same(&format!("{set} at grid {grid}"), &rows, format, &got, &want);
        }
        total += rows.len();
        eprintln!(
            "cuda_quant: {set}: {} rows give the bytes of the vLLM kernel at the grids {GRIDS:?}",
            rows.len()
        );
    }
    assert_eq!(total, 154, "the fixture has 154 rows");
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

    /// A float from -1 to 1.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One row of `k` values. The class picks the shape. The magnitudes go from 1e-40 (a denormal) to 1e30.
fn random_row(rng: &mut Rng, k: usize, class: usize) -> Vec<f32> {
    let scale = [1.0f32, 1e-3, 1e3, 1e-20, 1e30, 1e-40, 0.37, 448.0][class % 8];
    let mut row: Vec<f32> = (0..k).map(|_| rng.unit() * scale).collect();
    match class % 12 {
        // A heavy tail: a few values far above the rest.
        3 => {
            for _ in 0..4 {
                let i = rng.below(k as u64) as usize;
                row[i] *= 1000.0;
            }
        }
        // Blocks that are all zero beside blocks that are not.
        5 => {
            for b in 0..k / 32 {
                if b % 3 == 0 {
                    row[b * 32..(b + 1) * 32].fill(0.0);
                }
            }
        }
        // Values on the grid of the codes, so that the ties of the rounding come up: the ties of e2m1 are 0.25, 0.75, 1.25.
        7 => {
            for v in &mut row {
                *v = (rng.below(81) as f32 - 40.0) * 0.125 * scale;
            }
        }
        // One value in a zero row.
        9 => {
            row.fill(0.0);
            let i = rng.below(k as u64) as usize;
            row[i] = rng.unit() * scale;
        }
        // The zero row.
        10 => row.fill(0.0),
        // Blocks with a value at full scale: 448 times a power of two, or just under it. The largest code of `e4m3` is
        // 448, so the clamp and the carry of the scale byte decide these blocks.
        11 => {
            for b in 0..k / 32 {
                let power = 2f32.powi(rng.below(41) as i32 - 20);
                let top = [1.0f32, 1.0, 0.999, 0.99, 0.985, 0.98][rng.below(6) as usize];
                for v in &mut row[b * 32..(b + 1) * 32] {
                    *v = rng.unit() * 448.0 * power;
                }
                row[b * 32 + rng.below(32) as usize] = 448.0 * power * top;
            }
        }
        _ => {}
    }
    row
}

#[test]
fn the_device_quantizers_give_the_bytes_of_quant_rs_on_random_rows_at_each_grid() {
    let dev = open();
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    for (format, k, count) in [
        (ExpertFormat::Nvfp4, 2304usize, 400usize),
        (ExpertFormat::Nvfp4, 1152, 400),
        (ExpertFormat::Nvfp4, 128, 300),
        (ExpertFormat::Mxfp8, 2304, 400),
        (ExpertFormat::Mxfp8, 1152, 400),
        (ExpertFormat::Mxfp8, 128, 300),
    ] {
        let rows: Vec<Vec<f32>> = (0..count).map(|i| random_row(&mut rng, k, i)).collect();
        let want = quantize_on_host(format, &rows);
        for grid in GRIDS {
            let got = quantize_on_gpu(&dev, format, &rows, grid);
            assert_same(&format!("{format:?} k {k} at grid {grid}"), &rows, format, &got, &want);
        }
        eprintln!(
            "cuda_quant: {format:?} k {k}: {count} random rows give the bytes of quant.rs at the grids {GRIDS:?}"
        );
    }
}

#[test]
fn a_row_of_a_buffer_has_the_same_bytes_in_a_buffer_of_one_row() {
    // The quantizer of a row does not depend on its neighbors or on the capacity of the buffer.
    let dev = open();
    let mut rng = Rng(77);
    for format in [ExpertFormat::Nvfp4, ExpertFormat::Mxfp8] {
        let rows: Vec<Vec<f32>> = (0..9).map(|i| random_row(&mut rng, 1152, i)).collect();
        let together = quantize_on_gpu(&dev, format, &rows, 48);
        for (r, row) in rows.iter().enumerate() {
            let alone = quantize_on_gpu(&dev, format, std::slice::from_ref(row), 7);
            assert_eq!(alone.codes[0], together.codes[r], "{format:?} row {r}: codes");
            assert_eq!(alone.scales[0], together.scales[r], "{format:?} row {r}: scales");
            if format == ExpertFormat::Nvfp4 {
                assert_eq!(alone.amax[0], together.amax[r], "{format:?} row {r}: amax");
            }
        }
    }
}
