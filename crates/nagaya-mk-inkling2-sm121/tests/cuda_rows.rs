//! GPU gates of the row ops: the row finisher, EMBED, MIX, the message gates and
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
use nagaya_mk_cuda::protocol::{
    FLAG_NORM, MkTaskDescriptor, MkTaskTable, OP_EMBED, OP_GATE_DOWN, OP_GATE_UP, OP_MIX, OP_MIX_OUT, OP_ROWS_PROBE,
    PROFILE_WORDS,
};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::rows::{GATE_UP_ITEM_BYTES, GATE_UP_TILES, gate_down_shape, gate_down_weights, repack_gate_up};
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
    run_tasks(dev, &[desc], scratch_floats, grid);
}

/// Run tasks one after the other at `grid` blocks, with a zeroed ring. The `f` of a descriptor is the offset in floats
/// from the start of the ring scratch, so each task has its own part.
fn run_tasks(dev: &CudaDevice, descs: &[MkTaskDescriptor], scratch_floats: usize, grid: u32) {
    let shared = dev.max_dynamic_shared_bytes().unwrap();
    let mut ring = Ring::new(dev, scratch_floats, false).unwrap();
    let scratch = ring.scratch();
    let list: Vec<Task> = descs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut d = *d;
            d.f += scratch;
            let task = Task::with_items(d);
            if i == 0 { task } else { task.after(i - 1) }
        })
        .collect();
    ring.set(&list);
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
    let dev = gpu::open_probe();
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

/// The time of a task, in microseconds: `tasks` copies of `desc` run one after the other, and each time is from the
/// first block that passes the wait to the last block that ends. The result is the median of the copies after the
/// first. The task writes the same results each time, so the copies are safe.
fn task_us(dev: &CudaDevice, desc: MkTaskDescriptor, scratch_floats: usize, tasks: usize) -> f64 {
    median_us(task_spans(dev, &vec![desc; tasks], scratch_floats))
}

fn median_us(mut spans: Vec<f64>) -> f64 {
    spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spans[spans.len() / 2]
}

/// The same for a list of descriptors that run one after the other: the times in us from the second one on.
/// The caller gives each one its own scratch `f`, or 0 for the start of the ring scratch.
fn task_spans(dev: &CudaDevice, descs: &[MkTaskDescriptor], scratch_floats: usize) -> Vec<f64> {
    let blocks = dev.sm_count() as usize;
    let shared = dev.max_dynamic_shared_bytes().unwrap();
    let mut ring = Ring::new(dev, scratch_floats, true).unwrap();
    let scratch = ring.scratch();
    let list: Vec<Task> = descs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut d = *d;
            d.f += scratch;
            let task = Task::with_items(d);
            if i == 0 { task } else { task.after(i - 1) }
        })
        .collect();
    ring.set(&list);
    ring.zero_device().unwrap();
    gpu::run_with_watchdog(&ring, blocks as u32, DEFAULT_BLOCK_THREADS, shared);
    let profile = ring.read_profile().unwrap();
    let spans: Vec<f64> = (1..descs.len())
        .map(|t| {
            let at = ring.ring_index(t);
            let slots = &profile[at * blocks * PROFILE_WORDS..(at + 1) * blocks * PROFILE_WORDS];
            // A block with no item does not wait, so only the blocks that have an item count.
            let used = (descs[t].items as usize).min(blocks);
            let begin = slots.chunks(PROFILE_WORDS).take(used).map(|s| s[1]).min().unwrap();
            let end = slots.chunks(PROFILE_WORDS).take(used).map(|s| s[2]).max().unwrap();
            (end - begin) as f64 / 1e3
        })
        .collect();
    ring.free().unwrap();
    spans
}

/// One mix case: the stack of the attention mix, and the message mix when `message` is set.
struct MixCase {
    name: String,
    queries: Vec<Vec<f32>>,
    depth: usize,
    messages: Option<[Vec<f32>; 4]>,
}

/// A stack row: the normed embedding rows of other tokens, scaled, so that the rows differ and have the scale of
/// the residual stream.
fn stack_rows(model: &Model, stack: usize, rows: usize, seed: u32) -> Vec<f32> {
    let ids: Vec<u32> = token_ids(rows)
        .iter()
        .map(|id| (id + seed * 977 + stack as u32 * 131) % 201_024)
        .collect();
    let scale = 0.6 + 0.5 * stack as f32;
    model.embed(&ids).iter().map(|v| v * scale).collect()
}

fn row_factor(row: &[f32]) -> f32 {
    1.0 / (ops::dot(row, row) / HIDDEN as f32 + ops::EPS).sqrt()
}

/// What the mix gives on the GPU, for `rows` rows: the attention outputs and factors, then the message outputs and
/// factors. The stack rows are `stacks` (the last row of the stack is the partial row).
struct MixOut {
    a: Vec<f32>,
    a_rf: Vec<f32>,
    m: Vec<f32>,
    m_rf: Vec<f32>,
}

/// The device buffers of a mix task: the descriptor, the four outputs, and every pointer to free.
struct MixSetup {
    desc: MkTaskDescriptor,
    a: u64,
    a_rf: u64,
    m: u64,
    m_rf: u64,
    owned: Vec<u64>,
}

fn mix_setup(
    dev: &CudaDevice,
    opcode: u16,
    case: &MixCase,
    stacks: &[Vec<f32>],
    msg: &[Vec<f32>],
    rows: usize,
) -> MixSetup {
    let q = case.queries.len();
    let mq = usize::from(case.messages.is_some()) * 4;
    let rf_of = |plane: &[f32]| -> Vec<f32> {
        (0..rows)
            .map(|r| row_factor(&plane[r * HIDDEN..(r + 1) * HIDDEN]))
            .collect()
    };
    let mut slots = vec![0u64; 16];
    let mut owned = Vec::new();
    for (s, plane) in stacks.iter().enumerate() {
        let p = gpu::upload(dev, &plane[..rows * HIDDEN]);
        let rf = gpu::upload(dev, &rf_of(&plane[..rows * HIDDEN]));
        slots[s] = p;
        slots[5 + s] = rf;
        owned.extend([p, rf]);
    }
    for (s, plane) in msg.iter().enumerate() {
        let p = gpu::upload(dev, &plane[..rows * HIDDEN]);
        let rf = gpu::upload(dev, &rf_of(&plane[..rows * HIDDEN]));
        slots[10 + s] = p;
        slots[12 + s] = rf;
        owned.extend([p, rf]);
    }
    let flat: Vec<f32> = case.queries.iter().flatten().copied().collect();
    slots[14] = gpu::upload(dev, &flat);
    let mflat: Vec<f32> = case.messages.iter().flatten().flatten().copied().collect();
    slots[15] = if mflat.is_empty() { 0 } else { gpu::upload(dev, &mflat) };
    owned.extend([slots[14], slots[15]]);
    let table = upload_table(dev, &slots);
    let a = gpu::upload(dev, &vec![0f32; (q * rows * HIDDEN).max(4)]);
    let a_rf = gpu::upload(dev, &vec![0f32; (q * rows).max(4)]);
    let m = gpu::upload(dev, &vec![0f32; (mq * rows * HIDDEN).max(4)]);
    let m_rf = gpu::upload(dev, &vec![0f32; (mq * rows).max(4)]);
    let desc = MkTaskDescriptor {
        opcode,
        items: (rows * (q + mq)) as u32,
        rows: rows as u32,
        a,
        b: a_rf,
        c: m,
        d: table,
        e: m_rf,
        n: q as u32,
        k: case.depth as u32,
        kt: mq as u32,
        group_items: if mq > 0 { 2 } else { 0 },
        p0: (HIDDEN as f32).powf(-0.5),
        ..Default::default()
    };
    owned.push(table);
    MixSetup {
        desc,
        a,
        a_rf,
        m,
        m_rf,
        owned,
    }
}

fn run_mix(
    dev: &CudaDevice,
    opcode: u16,
    case: &MixCase,
    stacks: &[Vec<f32>],
    msg: &[Vec<f32>],
    rows: usize,
    grid: u32,
) -> MixOut {
    let setup = mix_setup(dev, opcode, case, stacks, msg, rows);
    let q = case.queries.len();
    let mq = usize::from(case.messages.is_some()) * 4;
    run_task(dev, setup.desc, 16, grid);
    let out = MixOut {
        a: gpu::download(dev, setup.a, q * rows * HIDDEN),
        a_rf: gpu::download(dev, setup.a_rf, q * rows),
        m: gpu::download(dev, setup.m, mq * rows * HIDDEN),
        m_rf: gpu::download(dev, setup.m_rf, mq * rows),
    };
    for ptr in setup
        .owned
        .into_iter()
        .filter(|p| *p != 0)
        .chain([setup.a, setup.a_rf, setup.m, setup.m_rf])
    {
        dev.free_device(ptr).unwrap();
    }
    out
}

fn mix_cases(model: &Model) -> Vec<MixCase> {
    let w = &model.weights;
    let cycle = |layer: usize| w.message_queries[layer - 6].clone();
    vec![
        MixCase {
            name: "layer 6, sliding, 1 query over 2 rows".into(),
            queries: w.layers[6].attn_queries.clone(),
            depth: 2,
            messages: None,
        },
        MixCase {
            name: "layer 17, full, 3 queries over 4 rows, with messages".into(),
            queries: w.layers[17].attn_queries.clone(),
            depth: 4,
            messages: Some(cycle(17)),
        },
        MixCase {
            name: "layer 23, full, 3 queries over 5 rows".into(),
            queries: w.layers[23].attn_queries.clone(),
            depth: 5,
            messages: None,
        },
        MixCase {
            name: "layer 9, sliding, 1 query over 3 rows, with messages".into(),
            queries: w.layers[9].attn_queries.clone(),
            depth: 3,
            messages: Some(cycle(9)),
        },
    ]
}

/// The error of `got` against `want` for a row, in epsilons of the largest magnitude of the stacked normed rows.
fn mix_error(got: &[f32], want: &[f32], scale: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .zip(scale)
        .map(|((g, w), s)| f64::from((g - w).abs()) / f64::from(*s) / f64::from(f32::EPSILON))
        .fold(0.0, f64::max)
}

/// Check one mix output against the host form `ops::attention_residual`; return the worst error in epsilons.
fn check_mix(case: &MixCase, stacks: &[Vec<f32>], msg: &[Vec<f32>], rows: usize, got: &MixOut) -> (f64, f64) {
    let (mut worst, mut worst_rf) = (0f64, 0f64);
    let take = |plane: &[f32], r: usize| plane[r * HIDDEN..(r + 1) * HIDDEN].to_vec();
    let mut check = |planes: &[Vec<f32>], queries: &[Vec<f32>], out: &[f32], out_rf: &[f32], r: usize| {
        let rows_r: Vec<Vec<f32>> = planes.iter().map(|p| take(p, r)).collect();
        let blocks: Vec<&[f32]> = rows_r[..rows_r.len() - 1].iter().map(|v| v.as_slice()).collect();
        let q: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
        let mut want = vec![Vec::new(); q.len()];
        ops::attention_residual(&blocks, rows_r.last().unwrap(), &q, &mut want);
        // The scale of the error: the largest normed value of the stack at each place.
        let normed: Vec<Vec<f32>> = rows_r
            .iter()
            .map(|v| v.iter().map(|x| (x * row_factor(v)).abs()).collect())
            .collect();
        let scale: Vec<f32> = (0..HIDDEN)
            .map(|i| normed.iter().map(|n| n[i]).fold(1e-20, f32::max))
            .collect();
        for (qi, w) in want.iter().enumerate() {
            let g = &out[(qi * rows + r) * HIDDEN..(qi * rows + r + 1) * HIDDEN];
            worst = worst.max(mix_error(g, w, &scale));
            let rf_want = row_factor(w);
            let e = f64::from((out_rf[qi * rows + r] - rf_want).abs()) / f64::from(rf_want) / f64::from(f32::EPSILON);
            worst_rf = worst_rf.max(e);
        }
    };
    for r in 0..rows {
        check(stacks, &case.queries, &got.a, &got.a_rf, r);
        if let Some(m) = &case.messages {
            check(msg, m, &got.m, &got.m_rf, r);
        }
    }
    (worst, worst_rf)
}

#[test]
fn mix_gives_the_rows_of_the_host_form_and_bits_that_do_not_depend_on_rows_or_grid() {
    let Some(model) = model() else { return };
    let dev = gpu::open();
    let mut worst_all = (0f64, 0f64);
    for case in mix_cases(model) {
        let stacks: Vec<Vec<f32>> = (0..case.depth).map(|s| stack_rows(model, s, 16, 1)).collect();
        let msg: Vec<Vec<f32>> = (0..2).map(|s| stack_rows(model, 7 + s, 16, 2)).collect();
        let mut first: Option<Vec<u32>> = None;
        for rows in [16usize, 7, 1] {
            for grid in GRIDS {
                let got = run_mix(&dev, OP_MIX, &case, &stacks, &msg, rows, grid);
                let (worst, worst_rf) = check_mix(&case, &stacks, &msg, rows, &got);
                worst_all = (worst_all.0.max(worst), worst_all.1.max(worst_rf));
                assert!(
                    worst <= MIX_BOUND_EPS,
                    "{}: {rows} rows, grid {grid}: error {worst:.1} epsilons",
                    case.name
                );
                assert!(
                    worst_rf <= MIX_BOUND_EPS,
                    "{}: {rows} rows, grid {grid}: factor error {worst_rf:.1}",
                    case.name
                );
                // Row 0 of query 0, row 0 of the last message query, and the factors: the same at each count of rows.
                let mut key = bits(&got.a[..HIDDEN]);
                key.extend(bits(&got.a_rf[..1]));
                if case.messages.is_some() {
                    let last = 3 * rows;
                    key.extend(bits(&got.m[last * HIDDEN..(last + 1) * HIDDEN]));
                    key.extend(bits(&got.m_rf[last..last + 1]));
                }
                match &first {
                    None => first = Some(key),
                    Some(want) => assert_eq!(&key, want, "{}: row 0 differs at {rows} rows, grid {grid}", case.name),
                }
            }
        }
    }
    eprintln!(
        "cuda_rows mix: worst error against the host form {:.2} epsilons of the largest normed value, and {:.2} for a factor (bound {MIX_BOUND_EPS})",
        worst_all.0, worst_all.1
    );
}

/// The bound of a mix, in epsilons of the largest normed value of the stack at the place. The gate prints the worst
/// error that it measures.
const MIX_BOUND_EPS: f64 = 8.0;

#[test]
fn the_mix_at_one_row_takes_this_many_microseconds() {
    let Some(model) = model() else { return };
    let dev = gpu::open_profile();
    for case in mix_cases(model) {
        for rows in [1usize, 16] {
            let stacks: Vec<Vec<f32>> = (0..case.depth).map(|s| stack_rows(model, s, 16, 1)).collect();
            let msg: Vec<Vec<f32>> = (0..2).map(|s| stack_rows(model, 7 + s, 16, 2)).collect();
            // Build the descriptor the same way as the gate, and time a chain of copies.
            let setup = mix_setup(&dev, OP_MIX, &case, &stacks, &msg, rows);
            let us = task_us(&dev, setup.desc, 16, 40);
            let items = setup.desc.items;
            eprintln!(
                "cuda_rows mix time: {} at {rows} rows: {items} items, {us:.2} us",
                case.name
            );
        }
    }
}

/// The bytes of `bf16` for values that are exact in `bf16` (a checkpoint matrix that the host form widened).
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The device side of the three gates of one layer: the weights of both ops, and the rows.
struct GateRig {
    down: u64,
    up: u64,
    a: [u64; 3],
    m: [u64; 3],
    a_rf: [u64; 3],
    m_rf: [u64; 3],
    g: u64,
    u: [u64; 3],
    u_rf: [u64; 3],
    table_down: u64,
    table_up: u64,
    h_e: u64,
    norm: u64,
}

/// The floats of scratch for the two ops of the gates at `rows` rows: the part of GATE_DOWN, then the part of GATE_UP.
fn down_scratch(rows: usize) -> usize {
    16 + 72 * rows * 16
}
fn up_scratch(rows: usize) -> usize {
    4 + 108 * rows
}

/// The bits of one gate: `g`, `u` and the row factors of `u`.
type GateBits = (Vec<u32>, Vec<u32>, Vec<u32>);

/// The counts of rows of the gate gates, the most first: it is the reference of the bits.
const GATE_ROWS: [usize; 6] = [512, 100, 17, 16, 7, 1];
/// The rows that the accuracy check visits: the first tile, every 37th row, and the last row.
fn checked(row: usize, rows: usize) -> bool {
    row < 16 || row.is_multiple_of(37) || row + 1 == rows
}

impl GateRig {
    fn new(dev: &CudaDevice, model: &Model, layer: usize, a: &[Vec<f32>; 3], m: &[Vec<f32>; 3], rows: usize) -> Self {
        let gates = &model.weights.layers[layer].gates;
        let down_bytes: Vec<Vec<u8>> = gates[..3].iter().map(|g| bf16_bytes(&g.down)).collect();
        let down = gate_down_weights(&down_bytes.iter().map(|v| v.as_slice()).collect::<Vec<_>>()).unwrap();
        let up: Vec<u8> = gates[..3]
            .iter()
            .flat_map(|g| repack_gate_up(&bf16_bytes(&g.forget), &bf16_bytes(&g.gain)).unwrap())
            .collect();
        let rf_of = |plane: &[f32]| -> Vec<f32> {
            (0..rows)
                .map(|r| row_factor(&plane[r * HIDDEN..(r + 1) * HIDDEN]))
                .collect()
        };
        let each = |f: &dyn Fn(usize) -> u64| [f(0), f(1), f(2)];
        let rig = Self {
            down: gpu::upload(dev, &down),
            up: gpu::upload(dev, &up),
            a: each(&|i| gpu::upload(dev, &a[i][..rows * HIDDEN])),
            m: each(&|i| gpu::upload(dev, &m[i][..rows * HIDDEN])),
            a_rf: each(&|i| gpu::upload(dev, &rf_of(&a[i]))),
            m_rf: each(&|i| gpu::upload(dev, &rf_of(&m[i]))),
            g: gpu::upload(dev, &vec![0f32; 3 * rows * 64]),
            u: each(&|_| gpu::upload(dev, &vec![0f32; rows * HIDDEN])),
            u_rf: each(&|_| gpu::upload(dev, &vec![0f32; rows])),
            table_down: 0,
            table_up: 0,
            h_e: gpu::upload(dev, &vec![0f32; rows * HIDDEN]),
            norm: gpu::upload(dev, &model.weights.layers[layer].mlp_norm),
        };
        let slots_down: Vec<u64> = [rig.a, rig.m, rig.a_rf, rig.m_rf].concat();
        let slots_up: Vec<u64> = [rig.a, rig.m, rig.u, rig.u_rf]
            .concat()
            .into_iter()
            .chain([rig.norm])
            .collect();
        Self {
            table_down: upload_table(dev, &slots_down),
            table_up: upload_table(dev, &slots_up),
            ..rig
        }
    }

    fn descriptors(&self, rows: usize) -> [MkTaskDescriptor; 2] {
        let shape = gate_down_shape();
        [
            MkTaskDescriptor {
                opcode: OP_GATE_DOWN,
                items: 72,
                rows: rows as u32,
                b: self.down,
                c: self.g,
                d: self.table_down,
                n: 3,
                k: shape.k as u32,
                kt: shape.kt as u32,
                group_items: shape.slices() as u32,
                item_bytes: shape.item_bytes() as u32,
                ..Default::default()
            },
            MkTaskDescriptor {
                opcode: OP_GATE_UP,
                items: 108,
                rows: rows as u32,
                a: self.g,
                b: self.up,
                d: self.table_up,
                n: 3,
                k: 64,
                item_bytes: GATE_UP_ITEM_BYTES as u32,
                f: down_scratch(rows) as u64 * 4,
                ..Default::default()
            },
        ]
    }

    fn free(self, dev: &CudaDevice) {
        let all = [
            self.down,
            self.up,
            self.g,
            self.table_down,
            self.table_up,
            self.h_e,
            self.norm,
        ]
        .into_iter()
        .chain(self.a)
        .chain(self.m)
        .chain(self.a_rf)
        .chain(self.m_rf)
        .chain(self.u)
        .chain(self.u_rf);
        for ptr in all {
            dev.free_device(ptr).unwrap();
        }
    }
}

/// The rows of a gate gate: `a_i` and `m_i` of each of the 3 gates, with the scale of the residual stream.
fn gate_rows(model: &Model, seed: u32) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let a = std::array::from_fn(|i| stack_rows(model, i, 512, seed));
    let m = std::array::from_fn(|i| stack_rows(model, 3 + i, 512, seed + 5));
    (a, m)
}

/// The bound of a gate result, in epsilons of the sum of the absolute terms of the result of the down matmul.
const GATE_DOWN_BOUND_EPS: f64 = 4.0;
/// The bound of `u` against the host form, in units of the first-order budget of the error of `u`. The host sums
/// in f32 with 16 lanes, so it has an error of its own of about the same size.
const GATE_UP_HOST_BOUND: f64 = 0.25;
/// The bound of `u` against f64, in units of the same budget. The budget is worst-case: it adds the terms of each sum
/// with the sign that hurts. The gate measured 0.012 at most.
const GATE_UP_BOUND: f64 = 0.1;
/// The bound of the row factor of `u`, in epsilons.
const GATE_RF_BOUND_EPS: f64 = 16.0;

#[test]
fn the_gates_give_the_rows_of_the_host_form_and_bits_that_do_not_depend_on_rows_or_grid() {
    let Some(model) = model() else { return };
    let dev = gpu::open();
    let (mut worst_g, mut worst_u, mut worst_u64, mut worst_rf) = (0f64, 0f64, 0f64, 0f64);
    for layer in [6usize, 12, 17] {
        let (a, m) = gate_rows(model, layer as u32);
        // The bits of the run at 512 rows and grid 48: g, u and the factors of each gate.
        let mut reference: Option<Vec<GateBits>> = None;
        for rows in GATE_ROWS {
            let rig = GateRig::new(&dev, model, layer, &a, &m, rows);
            for grid in GRIDS {
                run_tasks(
                    &dev,
                    &rig.descriptors(rows),
                    down_scratch(rows) + up_scratch(rows),
                    grid,
                );
                let g: Vec<f32> = gpu::download(&dev, rig.g, 3 * rows * 64);
                let mut key = Vec::new();
                for gate in 0..3 {
                    let w = &model.weights.layers[layer].gates[gate];
                    let u: Vec<f32> = gpu::download(&dev, rig.u[gate], rows * HIDDEN);
                    let u_rf: Vec<f32> = gpu::download(&dev, rig.u_rf[gate], rows);
                    for r in 0..rows {
                        // The bits do not depend on the grid, so the other grids are checked against the first.
                        if grid != GRIDS[0] || !checked(r, rows) {
                            continue;
                        }
                        let (h, mm) = (
                            &a[gate][r * HIDDEN..(r + 1) * HIDDEN],
                            &m[gate][r * HIDDEN..(r + 1) * HIDDEN],
                        );
                        // The down output against an f64 sum of the exact products.
                        let mut both: Vec<f32> = h.iter().map(|v| v * row_factor(h)).collect();
                        both.extend(mm.iter().map(|v| v * row_factor(mm)));
                        let mut g64 = [0f64; 64];
                        let mut mags = [0f64; 64];
                        for (k, g64k) in g64.iter_mut().enumerate() {
                            let terms: Vec<f64> = (0..2 * HIDDEN)
                                .map(|j| f64::from(w.down[k * 2 * HIDDEN + j]) * f64::from(both[j]))
                                .collect();
                            let (sum, mag) = (terms.iter().sum::<f64>(), terms.iter().map(|t| t.abs()).sum::<f64>());
                            let got = f64::from(g[(gate * rows + r) * 64 + k]);
                            let e = (got - sum).abs() / mag / f64::from(f32::EPSILON);
                            worst_g = worst_g.max(e);
                            assert!(
                                e <= GATE_DOWN_BOUND_EPS,
                                "layer {layer} gate {gate} row {r} k {k} at {rows} rows, grid {grid}: down error {e:.2} epsilons"
                            );
                            *g64k = sum;
                            mags[k] = mag;
                        }
                        // The gate against an f64 evaluation of the same formula, and against the host form.
                        let mut want = vec![0f32; HIDDEN];
                        ops::message_gate(h, mm, &w.down, &w.forget, &w.gain, &mut want);
                        let got_u = &u[r * HIDDEN..(r + 1) * HIDDEN];
                        for i in 0..HIDDEN {
                            // The first-order bound of the error of `u`. The error of the down output is within
                            // GATE_DOWN_BOUND_EPS epsilons of its terms, it enters the sum of the forget (or gain)
                            // row, and the sum adds its own error of the same size. tanh has a slope of at most 1.
                            let reach = |m: &[f32]| {
                                let sums = |f: &dyn Fn(usize) -> f64| {
                                    (0..64).map(|k| f64::from(m[i * 64 + k]).abs() * f(k)).sum::<f64>()
                                };
                                GATE_DOWN_BOUND_EPS * (sums(&|k| mags[k]) + sums(&|k| g64[k].abs()))
                            };
                            let dot64 = |m: &[f32]| (0..64).map(|k| f64::from(m[i * 64 + k]) * g64[k]).sum::<f64>();
                            let exact = (1.0 - dot64(&w.forget).tanh()) * f64::from(h[i])
                                + dot64(&w.gain).tanh() * f64::from(mm[i]);
                            let budget = f64::from(f32::EPSILON)
                                * (f64::from(h[i].abs()) * reach(&w.forget)
                                    + f64::from(mm[i].abs()) * reach(&w.gain)
                                    + 8.0 * f64::from(h[i].abs() + mm[i].abs()));
                            let r64 = (f64::from(got_u[i]) - exact).abs() / budget;
                            let r_host = f64::from((got_u[i] - want[i]).abs()) / budget;
                            worst_u64 = worst_u64.max(r64);
                            worst_u = worst_u.max(r_host);
                            assert!(
                                r64 <= 1.0,
                                "layer {layer} gate {gate} row {r} channel {i} at {rows} rows, grid {grid}: u {:e}, f64 {exact:e}, {r64:.2} of the budget",
                                got_u[i]
                            );
                            assert!(
                                r_host <= GATE_UP_HOST_BOUND,
                                "layer {layer} gate {gate} row {r} channel {i} at {rows} rows, grid {grid}: u {:e}, host {:e}, {r_host:.2} of the budget",
                                got_u[i],
                                want[i]
                            );
                        }
                        let rf_want = row_factor(&want);
                        let e = f64::from((u_rf[r] - rf_want).abs()) / f64::from(rf_want) / f64::from(f32::EPSILON);
                        worst_rf = worst_rf.max(e);
                        assert!(
                            e <= GATE_RF_BOUND_EPS,
                            "layer {layer} gate {gate} row {r}: factor error {e:.2}"
                        );
                    }
                    key.push((
                        bits(&g[gate * rows * 64..(gate + 1) * rows * 64]),
                        bits(&u),
                        bits(&u_rf),
                    ));
                }
                match &reference {
                    None => reference = Some(key),
                    Some(want) => {
                        for (gate, (got, full)) in key.iter().zip(want).enumerate() {
                            // The rows of a smaller call equal the first rows of the call at 512 rows.
                            assert_eq!(
                                got.0[..],
                                full.0[..rows * 64],
                                "layer {layer} gate {gate}: g differs at {rows} rows, grid {grid}"
                            );
                            assert_eq!(
                                got.1[..],
                                full.1[..rows * HIDDEN],
                                "layer {layer} gate {gate}: u differs at {rows} rows, grid {grid}"
                            );
                            assert_eq!(
                                got.2[..],
                                full.2[..rows],
                                "layer {layer} gate {gate}: the factors differ at {rows} rows, grid {grid}"
                            );
                        }
                    }
                }
            }
            rig.free(&dev);
        }
    }
    eprintln!(
        "cuda_rows gates: worst error against the host form: down {worst_g:.2} epsilons of the sum of |terms| (bound {GATE_DOWN_BOUND_EPS}), \
         u {worst_u64:.3} of the first-order budget against f64 (bound {GATE_UP_BOUND}) and {worst_u:.3} against the host form (bound {GATE_UP_HOST_BOUND}), factor {worst_rf:.2} epsilons"
    );
}

/// The 4th gate, the MLP one: one gate, with FLAG_NORM, writes `h_e = mlp_norm * (u * rf)`, the input of the router.
#[test]
fn the_mlp_gate_writes_h_e_as_the_host_form_does() {
    let Some(model) = model() else { return };
    let dev = gpu::open();
    let layer = 12usize;
    let w = &model.weights.layers[layer];
    let gate = &w.gates[3];
    let (a, m) = gate_rows(model, 40);
    let mut reference: Option<Vec<u32>> = None;
    let mut worst = 0f64;
    for rows in GATE_ROWS {
        let rf_of = |p: &[f32]| -> Vec<f32> {
            (0..rows)
                .map(|r| row_factor(&p[r * HIDDEN..(r + 1) * HIDDEN]))
                .collect()
        };
        let down = gate_down_weights(&[&bf16_bytes(&gate.down)]).unwrap();
        let up = repack_gate_up(&bf16_bytes(&gate.forget), &bf16_bytes(&gate.gain)).unwrap();
        assert_eq!(up.len(), GATE_UP_TILES * GATE_UP_ITEM_BYTES);
        let (d_down, d_up) = (gpu::upload(&dev, &down), gpu::upload(&dev, &up));
        let (d_a, d_m) = (
            gpu::upload(&dev, &a[0][..rows * HIDDEN]),
            gpu::upload(&dev, &m[0][..rows * HIDDEN]),
        );
        let (d_arf, d_mrf) = (gpu::upload(&dev, &rf_of(&a[0])), gpu::upload(&dev, &rf_of(&m[0])));
        let g = gpu::upload(&dev, &vec![0f32; rows * 64]);
        let (u, u_rf) = (
            gpu::upload(&dev, &vec![0f32; rows * HIDDEN]),
            gpu::upload(&dev, &vec![0f32; rows]),
        );
        let h_e = gpu::upload(&dev, &vec![0f32; rows * HIDDEN]);
        let norm = gpu::upload(&dev, &w.mlp_norm);
        let table_down = upload_table(&dev, &[d_a, 0, 0, d_m, 0, 0, d_arf, 0, 0, d_mrf]);
        let table_up = upload_table(&dev, &[d_a, 0, 0, d_m, 0, 0, u, 0, 0, u_rf, 0, 0, norm]);
        let shape = gate_down_shape();
        let descs = [
            MkTaskDescriptor {
                opcode: OP_GATE_DOWN,
                items: 24,
                rows: rows as u32,
                b: d_down,
                c: g,
                d: table_down,
                n: 1,
                k: shape.k as u32,
                kt: shape.kt as u32,
                group_items: shape.slices() as u32,
                item_bytes: shape.item_bytes() as u32,
                ..Default::default()
            },
            MkTaskDescriptor {
                opcode: OP_GATE_UP,
                flags: FLAG_NORM,
                items: 36,
                rows: rows as u32,
                a: g,
                b: d_up,
                c: h_e,
                d: table_up,
                n: 1,
                k: 64,
                item_bytes: GATE_UP_ITEM_BYTES as u32,
                f: down_scratch(rows) as u64 * 4,
                ..Default::default()
            },
        ];
        for grid in GRIDS {
            run_tasks(&dev, &descs, down_scratch(rows) + up_scratch(rows), grid);
            let got: Vec<f32> = gpu::download(&dev, h_e, rows * HIDDEN);
            if grid == GRIDS[0] {
                for r in (0..rows).filter(|r| checked(*r, rows)) {
                    let mut want_u = vec![0f32; HIDDEN];
                    let (h, mm) = (&a[0][r * HIDDEN..(r + 1) * HIDDEN], &m[0][r * HIDDEN..(r + 1) * HIDDEN]);
                    ops::message_gate(h, mm, &gate.down, &gate.forget, &gate.gain, &mut want_u);
                    let mut want = vec![0f32; HIDDEN];
                    ops::rms_norm(&want_u, &w.mlp_norm, &mut want);
                    for i in 0..HIDDEN {
                        let scale = (w.mlp_norm[i] * row_factor(&want_u) * (h[i].abs() + mm[i].abs())).abs();
                        let e = f64::from((got[r * HIDDEN + i] - want[i]).abs())
                            / f64::from(scale)
                            / f64::from(f32::EPSILON);
                        worst = worst.max(e);
                        assert!(
                            e <= H_E_BOUND_EPS,
                            "row {r} channel {i} at {rows} rows: h_e {:e}, host {:e}, {e:.1} epsilons",
                            got[r * HIDDEN + i],
                            want[i]
                        );
                    }
                }
            }
            let key = bits(&got);
            match &reference {
                None => reference = Some(key),
                Some(want_bits) => assert_eq!(
                    key[..],
                    want_bits[..rows * HIDDEN],
                    "h_e differs at {rows} rows, grid {grid}"
                ),
            }
        }
    }
    eprintln!("cuda_rows gates: h_e worst error against the host form {worst:.2} epsilons (bound {H_E_BOUND_EPS})");
}

/// The bound of `h_e` against the host form, in epsilons of `w * rf * (|h| + |m|)`, the scale of the error of `u`
/// (`u` is a difference, so its error does not follow its own size).
const H_E_BOUND_EPS: f64 = 48.0;

/// The speed of the gates. Each task of the chain reads its own copy of the weights, 24 copies in all (about 42 MB for
/// each op, more than the L2 cache), so the weights come from DRAM, as they do in a run.
#[test]
fn the_gates_take_this_many_microseconds_and_this_many_gb_per_second() {
    let Some(model) = model() else { return };
    let dev = gpu::open_profile();
    const COPIES: usize = 24;
    const PEAK_GBPS: f64 = 252.0;
    let (a, m) = gate_rows(model, 6);
    for rows in [1usize, 16, 512] {
        let rigs: Vec<GateRig> = (0..COPIES)
            .map(|_| GateRig::new(&dev, model, 12, &a, &m, rows))
            .collect();
        let descs: Vec<MkTaskDescriptor> = rigs.iter().flat_map(|r| r.descriptors(rows)).collect();
        let spans = task_spans(&dev, &descs, down_scratch(rows) + up_scratch(rows));
        // The task `i` of the list is the span `i - 1`: the down tasks are the odd spans, and the up tasks the even.
        let down = median_us(spans.iter().skip(1).step_by(2).copied().collect());
        let up = median_us(spans.iter().step_by(2).copied().collect());
        let down_mb = (gate_down_shape().packed_bytes() * 3) as f64 / 1e6;
        let up_mb = (GATE_UP_TILES * GATE_UP_ITEM_BYTES * 3) as f64 / 1e6;
        eprintln!(
            "cuda_rows gates time: {rows:>2} rows: GATE_DOWN {down:.2} us ({down_mb:.2} MB, {:.0} GB/s, {:.0}% of the peak), GATE_UP {up:.2} us ({up_mb:.2} MB, {:.0} GB/s, {:.0}% of the peak); the budget is about 7.4 us each",
            down_mb * 1e3 / down,
            down_mb * 1e3 / down / PEAK_GBPS * 100.0,
            up_mb * 1e3 / up,
            up_mb * 1e3 / up / PEAK_GBPS * 100.0
        );
        for rig in rigs {
            rig.free(&dev);
        }
    }
}

/// The case of the output residual: the 4 planes of the blocks and the row of the last run, one query.
fn mix_out_case(model: &Model) -> MixCase {
    MixCase {
        name: "the output residual over 4 planes and x".into(),
        queries: vec![model.weights.output_query.clone()],
        depth: 5,
        messages: None,
    }
}

#[test]
fn mix_out_gives_the_output_residual_of_the_host_form_and_bits_that_do_not_depend_on_rows_or_grid() {
    let Some(model) = model() else { return };
    let dev = gpu::open();
    let case = mix_out_case(model);
    let stacks: Vec<Vec<f32>> = (0..5).map(|s| stack_rows(model, s, 16, 9)).collect();
    let (mut worst, mut worst_rf) = (0f64, 0f64);
    let mut first: Option<Vec<u32>> = None;
    for rows in [16usize, 7, 1] {
        for grid in GRIDS {
            let got = run_mix(&dev, OP_MIX_OUT, &case, &stacks, &[], rows, grid);
            let (e, e_rf) = check_mix(&case, &stacks, &[], rows, &got);
            (worst, worst_rf) = (worst.max(e), worst_rf.max(e_rf));
            assert!(
                e <= MIX_BOUND_EPS && e_rf <= MIX_BOUND_EPS,
                "{rows} rows, grid {grid}: {e:.2} and {e_rf:.2} epsilons"
            );
            let mut key = bits(&got.a[..HIDDEN]);
            key.extend(bits(&got.a_rf[..1]));
            match &first {
                None => first = Some(key),
                Some(want) => assert_eq!(&key, want, "row 0 differs at {rows} rows, grid {grid}"),
            }
        }
    }
    eprintln!(
        "cuda_rows mix_out: worst error against the host form {worst:.2} epsilons of the largest normed value, and {worst_rf:.2} for a factor (bound {MIX_BOUND_EPS})"
    );
}

#[test]
fn the_output_residual_at_one_row_takes_this_many_microseconds() {
    let Some(model) = model() else { return };
    let dev = gpu::open_profile();
    let case = mix_out_case(model);
    let stacks: Vec<Vec<f32>> = (0..5).map(|s| stack_rows(model, s, 16, 9)).collect();
    for rows in [1usize, 16] {
        let setup = mix_setup(&dev, OP_MIX_OUT, &case, &stacks, &[], rows);
        let us = task_us(&dev, setup.desc, 16, 40);
        eprintln!(
            "cuda_rows mix_out time: {rows} rows: {} items, {us:.2} us",
            setup.desc.items
        );
    }
}
