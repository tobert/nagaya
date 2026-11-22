//! GPU gates of the row ops (milestone 37 step 9, wave B lane R): the row finisher, EMBED, MIX, the message gates and
//! MIX_OUT. The oracle of each op is the host form of the B70 crate (`nagaya_mk_inkling2_bmg::host::ops`).
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_rows -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::sync::OnceLock;

use common::gpu::{self, GRIDS};
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, MkTaskTable, OP_EMBED, OP_ROWS_PROBE};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_inkling2_bmg::host::forward::Model;
use nagaya_mk_inkling2_bmg::host::ops::{self, HIDDEN};

const MODEL_VAR: &str = "INKLING2_MODEL_DIR";

/// The host model of Inkling 2 Mini, mapped one time. `None` means that the model is absent and the test skips.
fn model() -> Option<&'static Model> {
    static CELL: OnceLock<Option<Model>> = OnceLock::new();
    CELL.get_or_init(|| {
        let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
        let Some(dir) = std::env::var_os(MODEL_VAR).map(PathBuf::from) else {
            assert!(!require, "{MODEL_VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
            eprintln!("cuda_rows: skipped, {MODEL_VAR} is not set");
            return None;
        };
        let path = dir.join("model-00001-of-00001.safetensors");
        if !path.exists() {
            assert!(
                !require,
                "{} is absent and NAGAYA_REQUIRE_MODELS is set",
                path.display()
            );
            eprintln!("cuda_rows: skipped, {} is absent", path.display());
            return None;
        }
        Some(Model::open(&path).unwrap_or_else(|e| panic!("{e}")))
    })
    .as_ref()
}

/// Upload a task table, 128-byte aligned.
fn upload_table(dev: &CudaDevice, slots: &[u64]) -> u64 {
    let mut table = MkTaskTable::default();
    table.ptr[..slots.len()].copy_from_slice(slots);
    gpu::upload(dev, &[table])
}

/// Run one task at `grid` blocks, with a zeroed ring.
fn run_task(dev: &CudaDevice, desc: MkTaskDescriptor, scratch_floats: usize, grid: u32) {
    let shared = dev.max_dynamic_shared_bytes().unwrap();
    let mut ring = Ring::new(dev, scratch_floats, false).unwrap();
    let mut desc = desc;
    desc.f = ring.scratch();
    ring.set(&[Task::with_items(desc)]);
    ring.zero_device().unwrap();
    gpu::run_with_watchdog(&ring, grid, DEFAULT_BLOCK_THREADS, shared);
    ring.free().unwrap();
}

/// The sum of the squares of one tile, in the order of the kernel: 256 lanes, a butterfly in each warp, and the
/// 8 warps as a tree.
fn block_sum(lanes: &[f32; 256]) -> f32 {
    let mut warp = [0f32; 8];
    for (w, out) in warp.iter_mut().enumerate() {
        let mut v: [f32; 32] = std::array::from_fn(|i| lanes[w * 32 + i]);
        for offset in [16, 8, 4, 2, 1] {
            v = std::array::from_fn(|i| v[i] + v[i ^ offset]);
        }
        *out = v[0];
    }
    ((warp[0] + warp[1]) + (warp[2] + warp[3])) + ((warp[4] + warp[5]) + (warp[6] + warp[7]))
}

/// A value of a row: a random sign, and an exponent from 2^-5 to 2^2, so that the order of a sum matters.
fn value(index: u64) -> f32 {
    let h = (index as u32).wrapping_mul(0x9E37_79B1) ^ ((index >> 32) as u32).wrapping_mul(0x85EB_CA6B);
    let h = (h ^ (h >> 15)).wrapping_mul(0x2C1B_3C6D);
    let m = ((h >> 4) % 2001) as f32 - 1000.0;
    m * 2f32.powi(((h >> 20) % 8) as i32 - 5)
}

/// What the probe computes, with the order of the kernel: the sums, the factors and the normed rows.
fn probe_host(x: &[f32], rows: usize, tiles: usize, width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let stride = tiles * width;
    let (mut sums, mut factors) = (Vec::new(), Vec::new());
    let mut normed = vec![0f32; rows * stride];
    for r in 0..rows {
        let row = &x[r * stride..(r + 1) * stride];
        let mut total = 0f32;
        for tile in 0..tiles {
            let mut lanes = [0f32; 256];
            for j in 0..width {
                let v = row[tile * width + j];
                lanes[j % 256] += v * v;
            }
            let partial = block_sum(&lanes);
            total = if tile == 0 { partial } else { total + partial };
        }
        let rf = 1.0 / (total / stride as f32 + 1e-6).sqrt();
        for (o, v) in normed[r * stride..(r + 1) * stride].iter_mut().zip(row) {
            *o = v * rf;
        }
        sums.push(total);
        factors.push(rf);
    }
    (sums, factors, normed)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_row_finisher_gives_the_bits_of_the_fixed_order_at_each_grid_and_each_count_of_rows() {
    let dev = gpu::open();
    let shared = dev.max_dynamic_shared_bytes().unwrap();
    // 64 columns per tile is the width of a gate tile, and 300 needs a second pass of the lanes.
    for (tiles, width) in [(36usize, 64usize), (11, 300)] {
        // The rows of 16: the rows of a smaller count are the first rows, and their bits must not change.
        let full: Vec<f32> = (0..(16 * tiles * width) as u64).map(value).collect();
        let mut first: Option<(Vec<u32>, Vec<u32>, Vec<u32>)> = None;
        for rows in [16usize, 7, 1] {
            let x = &full[..rows * tiles * width];
            let (sums, factors, normed) = probe_host(x, rows, tiles, width);
            let a = gpu::upload(&dev, x);
            let c = gpu::upload(&dev, &vec![0f32; 2 * rows]);
            let e = gpu::upload(&dev, &vec![0f32; x.len()]);
            let mut ring = Ring::new(&dev, 4 + tiles * rows, false).unwrap();
            let f = ring.scratch();
            let desc = MkTaskDescriptor {
                opcode: OP_ROWS_PROBE,
                items: tiles as u32,
                rows: rows as u32,
                a,
                c,
                e,
                f,
                n: width as u32,
                group_items: tiles as u32,
                ..Default::default()
            };
            ring.set(&[Task::with_items(desc)]);
            ring.zero_device().unwrap();
            let mut ran = 0u32;
            for grid in GRIDS {
                // Two runs with no zero between them: the finisher resets its counter.
                for run in 0..2 {
                    gpu::run_with_watchdog(&ring, grid, DEFAULT_BLOCK_THREADS, shared);
                    // One item did the work of the whole row, in each run.
                    let counts: Vec<u32> = gpu::download(&dev, f, 2);
                    assert_eq!(counts[0], 0, "the counter is not 0 after the run at grid {grid}");
                    ran += 1;
                    assert_eq!(counts[1], ran, "{ran} runs, and the finisher ran {} times", counts[1]);
                    let got: Vec<f32> = gpu::download(&dev, c, 2 * rows);
                    assert_eq!(
                        bits(&got[..rows]),
                        bits(&sums),
                        "tiles {tiles}, width {width}, rows {rows}, grid {grid}, run {run}: the sums differ from the host order"
                    );
                    assert_eq!(bits(&got[rows..]), bits(&factors), "the factors differ at grid {grid}");
                    let out: Vec<f32> = gpu::download(&dev, e, x.len());
                    assert_eq!(bits(&out), bits(&normed), "the normed rows differ at grid {grid}");
                }
            }
            let got: Vec<f32> = gpu::download(&dev, c, 2 * rows);
            let key = (
                bits(&got[..1]),
                bits(&got[rows..rows + 1]),
                bits(&host_row0(&dev, e, tiles * width)),
            );
            match &first {
                None => first = Some(key),
                Some(want) => assert_eq!(&key, want, "row 0 at {rows} rows differs from row 0 at 16 rows"),
            }
            ring.free().unwrap();
            for ptr in [a, c, e] {
                dev.free_device(ptr).unwrap();
            }
        }
    }
    eprintln!("cuda_rows: the row finisher gives the bits of the host order at {GRIDS:?} blocks and 16, 7 and 1 rows");
}

fn host_row0(dev: &nagaya_mk_cuda::driver::CudaDevice, ptr: u64, width: usize) -> Vec<f32> {
    gpu::download(dev, ptr, width)
}

/// The tokens of a gate: the first and the last of the vocabulary, and others spread over it.
fn token_ids(rows: usize) -> Vec<u32> {
    (0..rows as u64)
        .map(|i| match i {
            0 => 0,
            1 => 201_023,
            _ => ((i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) % 201_024) as u32,
        })
        .collect()
}

#[test]
fn embed_gives_the_normed_rows_of_the_host_form_and_bits_that_do_not_depend_on_rows_or_grid() {
    let Some(model) = model() else { return };
    let dev = gpu::open();
    let w = &model.weights;
    let table_bytes = w.checkpoint.of(&w.embed);
    let embed = gpu::upload(&dev, table_bytes);
    let norm = gpu::upload(&dev, &w.embed_norm);
    let mut worst = 0f64;
    let mut worst_rf = 0f64;
    let mut reference: Option<Vec<u32>> = None;
    for rows in [64usize, 16, 7, 1] {
        let ids = token_ids(rows);
        let want = model.embed(&ids);
        let d_ids = gpu::upload(&dev, &ids);
        let x = gpu::upload(&dev, &vec![0f32; rows * HIDDEN]);
        let rf = gpu::upload(&dev, &vec![0f32; rows]);
        let plane0 = gpu::upload(&dev, &vec![0f32; rows * HIDDEN]);
        let plane0_rf = gpu::upload(&dev, &vec![0f32; rows]);
        let table = upload_table(&dev, &[norm, plane0, plane0_rf]);
        for grid in GRIDS {
            let desc = MkTaskDescriptor {
                opcode: OP_EMBED,
                items: rows as u32,
                rows: rows as u32,
                a: d_ids,
                b: embed,
                c: x,
                d: table,
                e: rf,
                n: 201_024,
                ..Default::default()
            };
            run_task(&dev, desc, 16, grid);
            let got: Vec<f32> = gpu::download(&dev, x, rows * HIDDEN);
            let copy: Vec<f32> = gpu::download(&dev, plane0, rows * HIDDEN);
            assert_eq!(
                bits(&got),
                bits(&copy),
                "B[0] differs from x at {rows} rows, grid {grid}"
            );
            let factors: Vec<f32> = gpu::download(&dev, rf, rows);
            let plane_factors: Vec<f32> = gpu::download(&dev, plane0_rf, rows);
            assert_eq!(bits(&factors), bits(&plane_factors), "the factors of B[0] differ");
            for r in 0..rows {
                // The factor is the factor of the row that the op wrote.
                let row = &want[r * HIDDEN..(r + 1) * HIDDEN];
                let rf_want = 1.0 / (ops::dot(row, row) / HIDDEN as f32 + ops::EPS).sqrt();
                let error = f64::from((factors[r] - rf_want).abs()) / f64::from(rf_want) / f64::from(f32::EPSILON);
                worst_rf = worst_rf.max(error);
                assert!(
                    error <= 8.0,
                    "row {r} at {rows} rows: the factor is {:e}, the host gives {rf_want:e}",
                    factors[r]
                );
            }
            for r in 0..rows {
                // The factor is the factor of the row x: the norm of the raw row, as the host form gives it.
                let row = &got[r * HIDDEN..(r + 1) * HIDDEN];
                for (i, (g, h)) in row.iter().zip(&want[r * HIDDEN..(r + 1) * HIDDEN]).enumerate() {
                    let error = f64::from((g - h).abs()) / f64::from(h.abs()).max(1e-30) / f64::from(f32::EPSILON);
                    worst = worst.max(error);
                    assert!(
                        error <= 8.0,
                        "row {r} value {i} at {rows} rows: GPU {g:e}, host {h:e}, {error:.1} epsilons"
                    );
                }
            }
            let key = bits(&got[..HIDDEN]);
            match &reference {
                None => reference = Some(key),
                Some(want_bits) => assert_eq!(&key, want_bits, "row 0 differs at {rows} rows, grid {grid}"),
            }
        }
        for ptr in [d_ids, x, rf, plane0, plane0_rf, table] {
            dev.free_device(ptr).unwrap();
        }
    }
    eprintln!(
        "cuda_rows embed: worst relative error against the host form {worst:.2} epsilons for a value and {worst_rf:.2} for \
         a factor (bound 8)"
    );
}
