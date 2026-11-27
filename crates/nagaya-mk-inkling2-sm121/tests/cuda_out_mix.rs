//! GPU gates of `OP_OUT_MIX`: the output matmul `wo`, the short convolution of the
//! attention, the residual and the mix of the MLP input, on real weights of Inkling 2 Mini, against the host forms of the
//! B70 crate (`ops::*`, the order of `Model::layer` in `host/forward.rs`).
//!
//! The layers: 0 (sliding, the first layer of a block, so `r` has no `x`), 5 (full, a layer inside a block), 6 (sliding,
//! the first layer of the second block), 11 (full, with two block planes) and 18 (sliding, the first layer of the fourth
//! block, a stack of 5 rows).
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_out_mix -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;
mod dense_support;

use common::gpu::{GRIDS, download, open, open_profile, run_with_watchdog, upload};
use dense_support::*;
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::outmix::{
    self, K, OutMixBuffers, OutMixShape, OutMixTable, WIDTH, out_floats, out_offsets, scratch_floats,
};
use nagaya_mk_cuda::protocol::PROFILE_WORDS;
use nagaya_mk_cuda::repack::repack_bf16;
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_inkling2_bmg::host::ops;

/// The weights of one layer that the task reads.
struct Layer {
    wo: Vec<u8>,
    conv: Vec<u8>,
    norm: Vec<f32>,
    query: Vec<f32>,
}

fn load_layer(ck: &Checkpoint, l: usize) -> Layer {
    let name = |n: &str| format!("model.llm.layers.{l}.{n}");
    Layer {
        wo: ck.bytes(&name("attn.wo_ud.weight"), "BF16", &[WIDTH, K]),
        conv: ck.bytes(&name("attn_sconv.weight"), "BF16", &[WIDTH, 1, 4]),
        norm: ck.bf16(&name("mlp_norm.weight"), &[WIDTH]),
        query: ck
            .bytes(&format!("model.llm.attn_res.mlp_queries.{l}"), "F32", &[WIDTH])
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    }
}

/// The case of a layer: its place in the stack.
#[derive(Clone, Copy)]
struct Case {
    layer: usize,
    /// The planes of the stack before `r`.
    planes: usize,
    accumulate: bool,
}

/// The five layers that the gate runs. A layer of index `l` with `l % 6 == 0` starts a block: the input `x` becomes a plane
/// and `r` has no `x`.
const CASES: [Case; 5] = [
    Case {
        layer: 0,
        planes: 1,
        accumulate: false,
    },
    Case {
        layer: 5,
        planes: 1,
        accumulate: true,
    },
    Case {
        layer: 6,
        planes: 2,
        accumulate: false,
    },
    Case {
        layer: 11,
        planes: 2,
        accumulate: true,
    },
    Case {
        layer: 18,
        planes: 4,
        accumulate: false,
    },
];

/// The inputs of a block.
struct Inputs {
    attn: Vec<f32>,
    x: Vec<f32>,
    planes: Vec<Vec<f32>>,
    state: Vec<f32>,
}

fn make_inputs(case: &Case, rows: usize, seed: u64) -> Inputs {
    let mut rng = Rng::new(seed);
    // The attention output has values of the order of 1, as the gate of a real layer gives.
    let attn: Vec<f32> = (0..rows * K).map(|_| rng.value(1.0)).collect();
    Inputs {
        attn,
        x: make_rows(rows, WIDTH, seed + 1),
        planes: (0..case.planes)
            .map(|p| make_rows(rows, WIDTH, seed + 10 + p as u64))
            .collect(),
        state: (0..3 * WIDTH).map(|_| rng.value(2.0)).collect(),
    }
}

/// What the device gives for a block.
#[derive(Default, Clone)]
struct Out {
    r: Vec<f32>,
    e: Vec<f32>,
    h: Vec<f32>,
    rf: Vec<f32>,
    state: Vec<f32>,
}

/// The device side of one case.
struct Rig<'d> {
    dev: &'d CudaDevice,
    case: Case,
    max_rows: usize,
    packed: CUdeviceptr,
    attn: CUdeviceptr,
    x: CUdeviceptr,
    r: CUdeviceptr,
    state: CUdeviceptr,
    out: CUdeviceptr,
    table: CUdeviceptr,
    ring: Ring<'d>,
    shared: u32,
    owned: Vec<CUdeviceptr>,
}

impl<'d> Rig<'d> {
    fn new(dev: &'d CudaDevice, layer: &'d Layer, case: Case, inputs: &Inputs, max_rows: usize, profile: bool) -> Self {
        let owned = std::cell::RefCell::new(Vec::new());
        let up = |p: CUdeviceptr| {
            owned.borrow_mut().push(p);
            p
        };
        let packed = up(upload(dev, &repack_bf16(&layer.wo, &outmix::shape()).unwrap()));
        let attn = up(dev.alloc_device(max_rows * K * 4).unwrap());
        let x = up(upload(dev, &inputs.x));
        let r = up(dev.alloc_device(max_rows * WIDTH * 4).unwrap());
        let state = up(upload(dev, &inputs.state));
        let out = up(dev.alloc_device(out_floats(max_rows) * 4).unwrap());
        let r_rf = up(dev.alloc_device(max_rows * 4 + 16).unwrap());
        let planes: Vec<(u64, u64)> = inputs
            .planes
            .iter()
            .map(|p| {
                let rf: Vec<f32> = p.chunks(WIDTH).map(row_factor).collect();
                (up(upload(dev, p)), up(upload(dev, &rf)))
            })
            .collect();
        let table = OutMixTable {
            planes,
            r,
            r_rf,
            conv_weight: up(upload(dev, &layer.conv)),
            conv_state: state,
            norm_weight: up(upload(dev, &layer.norm)),
            out,
            query: up(upload(dev, &layer.query)),
        }
        .table()
        .unwrap();
        let table = up(upload(dev, &[table]));
        let ring = Ring::new(dev, scratch_floats(max_rows), profile).unwrap();
        ring.zero_device().unwrap();
        Self {
            dev,
            case,
            max_rows,
            packed,
            attn,
            x,
            r,
            state,
            out,
            table,
            ring,
            shared: dev.max_dynamic_shared_bytes().unwrap(),
            owned: owned.into_inner(),
        }
    }

    fn task(&self, rows: usize) -> Task {
        self.task_spread(rows, rows >= outmix::SPREAD_MIN_ROWS)
    }

    /// The task with the finisher in the item that arrives last (`spread` false) or in items after the matmul.
    fn task_spread(&self, rows: usize, spread: bool) -> Task {
        let shape = OutMixShape::new(rows, self.case.planes + 1, self.case.accumulate, true)
            .unwrap()
            .with_spread(spread);
        let desc = shape
            .descriptor(&OutMixBuffers {
                attn: self.attn,
                packed: self.packed,
                r: self.r,
                x: if self.case.accumulate { self.x } else { 0 },
                scratch: self.ring.scratch(),
                table: self.table,
            })
            .unwrap();
        let mut task = Task::with_items(desc)
            .reads(self.attn, rows * K * 4)
            .reads(self.packed, outmix::shape().packed_bytes())
            .writes(self.r, rows * WIDTH * 4)
            .writes(self.out, out_floats(rows) * 4)
            .writes(self.ring.scratch(), scratch_floats(rows) * 4);
        if self.case.accumulate {
            task = task.reads(self.x, rows * WIDTH * 4);
        }
        task
    }

    /// One launch of a block of `rows` rows, with the state and the rows of `inputs`.
    fn run(&mut self, inputs: &Inputs, rows: usize, grid: u32) -> Out {
        let spread = rows >= outmix::SPREAD_MIN_ROWS;
        self.run_spread(inputs, rows, grid, spread)
    }

    /// `run` with the choice of the form of the finisher.
    fn run_spread(&mut self, inputs: &Inputs, rows: usize, grid: u32, spread: bool) -> Out {
        let dev = self.dev;
        unsafe {
            dev.copy_htod(self.state, inputs.state.as_ptr().cast(), inputs.state.len() * 4)
                .unwrap();
            dev.copy_htod(self.attn, inputs.attn.as_ptr().cast(), rows * K * 4)
                .unwrap();
            dev.copy_htod(self.x, inputs.x.as_ptr().cast(), rows * WIDTH * 4)
                .unwrap();
        }
        dev.memset_zero(self.r, self.max_rows * WIDTH * 4).unwrap();
        dev.memset_zero(self.out, out_floats(self.max_rows) * 4).unwrap();
        let task = self.task_spread(rows, spread);
        self.ring.set(&[task]);
        run_with_watchdog(&self.ring, grid, 288, self.shared);
        assert!(self.ring.read_scoreboard().is_ok());
        // The outputs sit at the offsets of `out_offsets(max_rows)`: `rows` is the count of the launch.
        let all: Vec<f32> = download(dev, self.out, out_floats(rows));
        let (e_at, h_at, rf_at) = out_offsets(rows);
        Out {
            r: download(dev, self.r, rows * WIDTH),
            e: all[e_at..h_at].to_vec(),
            h: all[h_at..rf_at].to_vec(),
            rf: all[rf_at..].to_vec(),
            state: download(dev, self.state, 3 * WIDTH),
        }
    }

    fn free(self) {
        for p in &self.owned {
            self.dev.free_device(*p).unwrap();
        }
        self.ring.free().unwrap();
    }
}

fn row(v: &[f32], t: usize) -> &[f32] {
    &v[t * WIDTH..(t + 1) * WIDTH]
}

/// The error of `got` against `want`, in epsilons of `f32`, relative to the largest value of each vector of `width`.
fn scaled_error(got: &[f32], want: &[f32], width: usize) -> f64 {
    let mut worst = 0.0f64;
    for (g, w) in got.chunks(width).zip(want.chunks(width)) {
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        for (a, b) in g.iter().zip(w) {
            worst = worst.max(f64::from((a - b).abs() / scale) / f64::from(f32::EPSILON));
        }
    }
    worst
}

fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    widen(bytes)
}

/// The host form of `r` for each row, and the conv state after the block: `wo`, then `ops::short_conv` with the three
/// rows before the row, then the residual with `x`.
fn host_r(layer: &Layer, case: &Case, inputs: &Inputs, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let conv = widen_bf16(&layer.conv);
    let raw: Vec<Vec<f32>> = (0..rows)
        .map(|t| {
            let mut o = vec![0f32; WIDTH];
            ops::matvec_bf16(&layer.wo, &inputs.attn[t * K..(t + 1) * K], &mut o);
            o
        })
        .collect();
    // The row at the index `j` of the block, or of the state when `j` is negative.
    let at = |j: isize| -> Vec<f32> {
        if j >= 0 {
            raw[j as usize].clone()
        } else {
            row(&inputs.state, (3 + j) as usize).to_vec()
        }
    };
    let mut r = Vec::new();
    for (t, raw_t) in raw.iter().enumerate().take(rows) {
        let state: Vec<f32> = (-3..0).flat_map(|d| at(t as isize + d)).collect();
        let mut c = vec![0f32; WIDTH];
        ops::short_conv(&state, raw_t, &conv, &mut c);
        if case.accumulate {
            c.iter_mut().zip(row(&inputs.x, t)).for_each(|(c, x)| *c += x);
        }
        r.extend(c);
    }
    let after: Vec<f32> = (0..3).flat_map(|j| at(rows as isize - 3 + j)).collect();
    (r, after)
}

/// The host form of the mix, its row factor and `h_e`, for each row of `r`.
fn host_mix(layer: &Layer, inputs: &Inputs, r: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut e, mut rf, mut h) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..rows {
        let stack: Vec<&[f32]> = inputs.planes.iter().map(|p| row(p, t)).collect();
        let mut mixed = vec![Vec::new()];
        ops::attention_residual(&stack, row(r, t), &[&layer.query], &mut mixed);
        rf.push(row_factor(&mixed[0]));
        let mut n = vec![0f32; WIDTH];
        ops::rms_norm(&mixed[0], &layer.norm, &mut n);
        e.extend(mixed.pop().unwrap());
        h.extend(n);
    }
    (e, rf, h)
}

/// The bound of the error of `r` against the host, in the unit of `scaled_error`. The matmul rounds as the MMA of the
/// projection does, and the convolution and the residual add a few operations. The gate measured 9.0 at worst, over the five
/// layers at 1 and 16 rows, and the bound is about twice that.
const R_BOUND_EPS: f64 = 16.0;
/// The bound of the mix, `e`, `rf` and `h_e`, against the host form of the same `r`. It has no matmul: the same
/// operations in a fixed order. The gate measured 2.9 at worst.
const MIX_BOUND_EPS: f64 = 8.0;

/// The bound of the matmul alone, as a share of the sum of the absolute terms of a column.
/// The brief gives about 4e-7. The gate measured 3.7e-8 at worst over the five layers at 1 and 16 rows.
const MATMUL_BOUND: f64 = 1e-7;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_outputs_equal_the_host_forms() {
    let Some(ck) = checkpoint("cuda_out_mix") else { return };
    let dev = open();
    for case in CASES {
        let layer = load_layer(ck, case.layer);
        for rows in [1usize, 16] {
            let inputs = make_inputs(&case, rows, 100 + case.layer as u64);
            let mut rig = Rig::new(&dev, &layer, case, &inputs, rows, false);
            let got = rig.run(&inputs, rows, 48);
            let (r, after) = host_r(&layer, &case, &inputs, rows);
            let label = format!("layer {} ({} planes), {rows} rows", case.layer, case.planes);
            assert!(
                got.r.iter().chain(&got.e).chain(&got.h).all(|v| v.is_finite()),
                "{label}: not finite"
            );
            let r_err = scaled_error(&got.r, &r, WIDTH);
            let state_err = scaled_error(&got.state, &after, WIDTH);
            // The mix of the device `r`: it isolates the finisher from the matmul.
            let (e, rf, h) = host_mix(&layer, &inputs, &got.r, rows);
            let e_err = scaled_error(&got.e, &e, WIDTH);
            let h_err = scaled_error(&got.h, &h, WIDTH);
            let rf_err = scaled_error(&got.rf, &rf, rows);
            eprintln!(
                "cuda_out_mix {label}: r {r_err:.2} state {state_err:.2} e {e_err:.2} h_e {h_err:.2} rf {rf_err:.2} epsilons"
            );
            assert!(r_err <= R_BOUND_EPS, "{label}: r error {r_err} over {R_BOUND_EPS}");
            assert!(
                state_err <= R_BOUND_EPS,
                "{label}: state error {state_err} over {R_BOUND_EPS}"
            );
            for (name, error) in [("e", e_err), ("h_e", h_err), ("rf", rf_err)] {
                assert!(
                    error <= MIX_BOUND_EPS,
                    "{label}: {name} error {error} over {MIX_BOUND_EPS}"
                );
            }
            // The last row of the state is the raw sum `o` of the last row of the block: the matmul alone, with no
            // convolution. Its error against the host is bounded by the share of the sum of the absolute terms.
            let wo = widen_bf16(&layer.wo);
            let last = &inputs.attn[(rows - 1) * K..rows * K];
            let mut worst_share = 0.0f64;
            for (c, got) in row(&got.state, 2).iter().enumerate() {
                let w = &wo[c * K..(c + 1) * K];
                let (mut sum, mut terms) = (0.0f64, 0.0f64);
                for (a, b) in w.iter().zip(last) {
                    sum += f64::from(*a) * f64::from(*b);
                    terms += (f64::from(*a) * f64::from(*b)).abs();
                }
                worst_share = worst_share.max((f64::from(*got) - sum).abs() / terms);
            }
            eprintln!("cuda_out_mix {label}: the matmul error is {worst_share:.2e} of the sum of |terms|");
            assert!(
                worst_share <= MATMUL_BOUND,
                "{label}: matmul error {worst_share} of the sum of |terms|"
            );
            // At one row the last row of the state is the raw sum of that row, so the convolution and the residual
            // have an exact host form from the device values: the bits of `r` must equal it. The bound of `r` does
            // not catch a divisor that is one ulp off.
            if rows == 1 {
                let mut c = vec![0f32; WIDTH];
                ops::short_conv(&inputs.state, row(&got.state, 2), &widen_bf16(&layer.conv), &mut c);
                if case.accumulate {
                    c.iter_mut().zip(&inputs.x).for_each(|(c, x)| *c += x);
                }
                assert_eq!(
                    bits(&got.r),
                    bits(&c),
                    "{label}: r is not the exact convolution of the raw sums"
                );
            }
            // `h_e` is `w * (e * rf)` in f32 with two multiplies, so the host form of the device `e` is exact. This catches a
            // norm weight that is one ulp off, which the bound of the mix does not.
            let w = &layer.norm;
            for t in 0..rows {
                for i in 0..WIDTH {
                    let want = w[i] * (got.e[t * WIDTH + i] * got.rf[t]);
                    assert_eq!(
                        got.h[t * WIDTH + i].to_bits(),
                        want.to_bits(),
                        "{label}: h_e at row {t}, value {i}"
                    );
                }
            }
            // The old rows of the state that stay in the state are exact.
            for j in 0..3usize.saturating_sub(rows) {
                assert_eq!(
                    bits(row(&got.state, j)),
                    bits(row(&inputs.state, j + rows)),
                    "{label}: the old state row {j} moved"
                );
            }
            rig.free();
        }
    }
}

#[test]
fn the_bits_do_not_depend_on_the_rows_or_the_grid() {
    let Some(ck) = checkpoint("cuda_out_mix") else { return };
    let dev = open();
    for case in [CASES[1], CASES[4]] {
        let layer = load_layer(ck, case.layer);
        let max = 512;
        let inputs = make_inputs(&case, max, 300 + case.layer as u64);
        let mut rig = Rig::new(&dev, &layer, case, &inputs, max, false);
        let full = rig.run(&inputs, max, 48);
        // Row `t` reads only the rows up to `t`: the first `rows` rows of a block of 512 equal a block of `rows`.
        for rows in [1usize, 7, 16, 17, 100] {
            let part = rig.run(&inputs, rows, 48);
            assert_eq!(
                bits(&part.r),
                bits(&full.r[..rows * WIDTH]),
                "layer {}: r of {rows} rows",
                case.layer
            );
            assert_eq!(
                bits(&part.e),
                bits(&full.e[..rows * WIDTH]),
                "layer {}: e of {rows} rows",
                case.layer
            );
            assert_eq!(
                bits(&part.h),
                bits(&full.h[..rows * WIDTH]),
                "layer {}: h_e of {rows} rows",
                case.layer
            );
            assert_eq!(
                bits(&part.rf),
                bits(&full.rf[..rows]),
                "layer {}: rf of {rows} rows",
                case.layer
            );
        }
        for grid in &GRIDS[1..] {
            let part = rig.run(&inputs, max, *grid);
            assert_eq!(bits(&part.r), bits(&full.r), "layer {}: r at grid {grid}", case.layer);
            assert_eq!(bits(&part.e), bits(&full.e), "layer {}: e at grid {grid}", case.layer);
            assert_eq!(bits(&part.h), bits(&full.h), "layer {}: h_e at grid {grid}", case.layer);
            assert_eq!(
                bits(&part.rf),
                bits(&full.rf),
                "layer {}: rf at grid {grid}",
                case.layer
            );
            assert_eq!(
                bits(&part.state),
                bits(&full.state),
                "layer {}: state at grid {grid}",
                case.layer
            );
        }
        eprintln!(
            "cuda_out_mix layer {}: the bits are equal at 1, 7, 16, 17, 100 and 512 rows and at grids {GRIDS:?}",
            case.layer
        );
        rig.free();
    }
}

#[test]
fn the_spread_finisher_has_the_bits_of_the_serial_finisher() {
    let Some(ck) = checkpoint("cuda_out_mix") else { return };
    let dev = open();
    for case in CASES {
        let layer = load_layer(ck, case.layer);
        let max = 512;
        let inputs = make_inputs(&case, max, 700 + case.layer as u64);
        let mut rig = Rig::new(&dev, &layer, case, &inputs, max, false);
        for rows in [1usize, 2, 3, 4, 5, 7, 16, 17, 100, 512] {
            // The serial form is the finisher of the code before the spread: the item that arrives last runs each row.
            let serial = rig.run_spread(&inputs, rows, 48, false);
            for grid in GRIDS {
                let spread = rig.run_spread(&inputs, rows, grid, true);
                let label = format!("layer {}, {rows} rows, grid {grid}", case.layer);
                assert_eq!(bits(&spread.r), bits(&serial.r), "{label}: r");
                assert_eq!(bits(&spread.e), bits(&serial.e), "{label}: e");
                assert_eq!(bits(&spread.h), bits(&serial.h), "{label}: h_e");
                assert_eq!(bits(&spread.rf), bits(&serial.rf), "{label}: rf");
                assert_eq!(bits(&spread.state), bits(&serial.state), "{label}: state");
            }
        }
        eprintln!(
            "cuda_out_mix layer {}: the spread finisher equals the serial one, bit for bit, at 10 row counts and grids {GRIDS:?}",
            case.layer
        );
        rig.free();
    }
}

#[test]
fn a_task_that_runs_twice_leaves_its_counters_at_zero() {
    // The counters of the tiles and of the finisher items start at 0 and the op sets them back. Two tasks in one ring
    // reuse them: the second would trap or hang if the first left a count.
    let Some(ck) = checkpoint("cuda_out_mix") else { return };
    let dev = open();
    let case = CASES[1];
    let layer = load_layer(ck, case.layer);
    let rows = 37;
    let inputs = make_inputs(&case, rows, 900);
    let mut rig = Rig::new(&dev, &layer, case, &inputs, rows, false);
    let once = rig.run(&inputs, rows, 48);
    let list = [rig.task(rows), rig.task(rows)];
    unsafe {
        dev.copy_htod(rig.state, inputs.state.as_ptr().cast(), inputs.state.len() * 4)
            .unwrap();
    }
    rig.ring.set(&list);
    run_with_watchdog(&rig.ring, 48, 288, rig.shared);
    assert!(rig.ring.read_scoreboard().is_ok());
    let counters: Vec<u32> = download(&dev, rig.ring.scratch(), outmix::COUNTERS);
    assert!(
        counters.iter().all(|c| *c == 0),
        "a counter of the task is not 0 after two runs: {counters:?}"
    );
    // The second task reads the state that the first left, so its rows differ from the first run's. The check is the
    // counters, and that both tasks ended with finite rows.
    let out: Vec<f32> = download(&dev, rig.out, out_floats(rows));
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(once.r.iter().all(|v| v.is_finite()));
    rig.free();
}

/// The rows `from..to` of a block, with the conv state `state`.
fn slice_inputs(all: &Inputs, from: usize, to: usize, state: Vec<f32>) -> Inputs {
    let cut = |v: &[f32], width: usize| v[from * width..to * width].to_vec();
    Inputs {
        attn: cut(&all.attn, K),
        x: cut(&all.x, WIDTH),
        planes: all.planes.iter().map(|p| cut(p, WIDTH)).collect(),
        state,
    }
}

#[test]
fn the_conv_state_carries_from_one_block_to_the_next() {
    let Some(ck) = checkpoint("cuda_out_mix") else { return };
    let dev = open();
    for case in [CASES[1], CASES[4]] {
        let layer = load_layer(ck, case.layer);
        for (first, second) in [(5usize, 11usize), (1, 16), (2, 30), (17, 1)] {
            let total = first + second;
            let all = make_inputs(&case, total, 500 + case.layer as u64);
            let mut joined = Rig::new(&dev, &layer, case, &all, total, false);
            let whole = joined.run(&all, total, 48);
            let one = slice_inputs(&all, 0, first, all.state.clone());
            let mut rig_one = Rig::new(&dev, &layer, case, &one, first, false);
            let got_one = rig_one.run(&one, first, 48);
            // The state after block 1 is the state before block 2.
            let two = slice_inputs(&all, first, total, got_one.state.clone());
            let mut rig_two = Rig::new(&dev, &layer, case, &two, second, false);
            let got_two = rig_two.run(&two, second, 48);
            let label = format!("layer {}, blocks of {first} and {second} rows", case.layer);
            assert_eq!(
                bits(&got_one.r),
                bits(&whole.r[..first * WIDTH]),
                "{label}: r of block 1"
            );
            assert_eq!(
                bits(&got_two.r),
                bits(&whole.r[first * WIDTH..]),
                "{label}: r of block 2"
            );
            assert_eq!(
                bits(&got_two.e),
                bits(&whole.e[first * WIDTH..]),
                "{label}: e of block 2"
            );
            assert_eq!(
                bits(&got_two.h),
                bits(&whole.h[first * WIDTH..]),
                "{label}: h_e of block 2"
            );
            assert_eq!(bits(&got_two.rf), bits(&whole.rf[first..]), "{label}: rf of block 2");
            assert_eq!(
                bits(&got_two.state),
                bits(&whole.state),
                "{label}: the state after block 2"
            );
            eprintln!("cuda_out_mix {label}: the two blocks equal the joined block, bit for bit");
            joined.free();
            rig_one.free();
            rig_two.free();
        }
    }
}

/// The task at one row must read `wo` at this share of the peak (the brief of the lane).
const OUT_MIX_GATE: f64 = 0.80;
/// The time of the design for the task at one row, in microseconds (`wo`: 14.16 MB).
const DESIGN_US: f64 = 61.0;

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// Nanoseconds from the first block that passes the wait of ring task `task` to the last block that ends it.
fn task_ns(profile: &[u64], blocks: usize, task: usize) -> u64 {
    let slots = &profile[task * blocks * PROFILE_WORDS..(task + 1) * blocks * PROFILE_WORDS];
    let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
    let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
    end - begin
}

#[test]
fn the_speed_of_the_task() {
    let Some(ck) = checkpoint("cuda_out_mix") else { return };
    let dev = open_profile();
    let blocks = dev.sm_count() as usize;
    let mut worst_at_one = f64::MAX;
    for case in [CASES[0], CASES[1]] {
        let layer = load_layer(ck, case.layer);
        let bytes = outmix::shape().packed_bytes();
        for rows in [1usize, 16, 512] {
            let inputs = make_inputs(&case, rows, 400);
            let mut rig = Rig::new(&dev, &layer, case, &inputs, rows, true);
            let tasks = 6;
            // The serial finisher (the form before the spread) for the comparison, then the form that `task` gives.
            if rows >= 2 {
                let list: Vec<Task> = (0..tasks).map(|_| rig.task_spread(rows, false)).collect();
                rig.ring.set(&list);
                run_with_watchdog(&rig.ring, blocks as u32, 288, rig.shared);
                let profile = rig.ring.read_profile().unwrap();
                let spans: Vec<u64> = (1..tasks)
                    .map(|t| task_ns(&profile, blocks, rig.ring.ring_index(t)))
                    .collect();
                eprintln!(
                    "cuda_out_mix speed: layer {} {rows:>3} rows, serial finisher: {:>8.1} us (best {:>8.1} us)",
                    case.layer,
                    median(spans.clone()) as f64 / 1e3,
                    *spans.iter().min().unwrap() as f64 / 1e3
                );
            }
            let list: Vec<Task> = (0..tasks).map(|_| rig.task(rows)).collect();
            rig.ring.set(&list);
            run_with_watchdog(&rig.ring, blocks as u32, 288, rig.shared);
            let profile = rig.ring.read_profile().unwrap();
            let spans: Vec<u64> = (1..tasks)
                .map(|t| task_ns(&profile, blocks, rig.ring.ring_index(t)))
                .collect();
            let (typical, best) = (median(spans.clone()), *spans.iter().min().unwrap());
            let gbps = bytes as f64 / typical as f64;
            eprintln!(
                "cuda_out_mix speed: layer {} {rows:>3} rows: {:>8.1} us (best {:>8.1} us), wo {:.2} MB, {:.1} GB/s ({:.1}% of {PEAK_GBPS}); design {DESIGN_US} us at one row",
                case.layer,
                typical as f64 / 1e3,
                best as f64 / 1e3,
                bytes as f64 / 1e6,
                gbps,
                gbps / PEAK_GBPS * 100.0
            );
            if rows == 1 {
                worst_at_one = worst_at_one.min(gbps);
            }
            rig.free();
        }
    }
    assert!(
        worst_at_one >= OUT_MIX_GATE * PEAK_GBPS,
        "OUT_MIX at one row reads {worst_at_one:.1} GB/s, under {:.0}% of {PEAK_GBPS} GB/s",
        OUT_MIX_GATE * 100.0
    );
}
