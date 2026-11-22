//! GPU gates of `OP_PROJ` (milestone 37 step 9, wave B lane D): the joined projection of the attention input, on real
//! weights of Inkling 2 Mini, against the host forms of the B70 crate (`ops::*`, the order of `host/forward.rs`).
//!
//! The layers: 0 (sliding, the first layer, no message), 6 (sliding, a layer of the loop) and 5 (full: the dynamic
//! convolutions, the log scale and three inputs).
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_proj -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;
mod dense_support;

use common::gpu::{GRIDS, download, open, open_profile, run_with_watchdog, upload};
use dense_support::*;
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::proj::{ProjLayout, SINKS, Segment, join};
use nagaya_mk_cuda::protocol::{MkTaskTable, PROFILE_WORDS};
use nagaya_mk_cuda::repack::repack_bf16;
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_inkling2_bmg::host::ops::{self, HEAD_DIM};

const Q: usize = 3072;
const REL: usize = 384;
const KV: usize = 1024;

// The slots of the task table of `OP_PROJ` (`src/protocol.rs`, `PROJ_TABLE_*`).
use nagaya_mk_cuda::protocol::{
    PROJ_TABLE_K_CONV, PROJ_TABLE_K_OUT, PROJ_TABLE_K_STATE, PROJ_TABLE_NORM_K, PROJ_TABLE_NORM_Q, PROJ_TABLE_NORM_V,
    PROJ_TABLE_Q_SCALE, PROJ_TABLE_REL_SCALE, PROJ_TABLE_RF_K, PROJ_TABLE_RF_V, PROJ_TABLE_V_CONV, PROJ_TABLE_V_OUT,
    PROJ_TABLE_V_STATE, PROJ_TABLE_X_K, PROJ_TABLE_X_V,
};

/// The weights of one layer, as the checkpoint stores them (`bf16` bytes) and widened.
struct Layer {
    layout: ProjLayout,
    wq: Vec<u8>,
    wr: Vec<u8>,
    sink: Vec<u8>,
    gate: Vec<u8>,
    wk: Vec<u8>,
    wv: Vec<u8>,
    /// The `weight_proj` of the key and of the value, a full layer only.
    taps_k: Vec<u8>,
    taps_v: Vec<u8>,
    /// The convolution weights: `[4][1024]` (a full layer) or `[1024][1][4]` (a sliding layer).
    conv_k: Vec<u8>,
    conv_v: Vec<u8>,
    /// The three norm weights (q, k, v). A sliding layer has one, three times.
    norms: [Vec<f32>; 3],
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
}

fn load_layer(ck: &Checkpoint, l: usize) -> Layer {
    let full = matches!(l, 5 | 11 | 17 | 23);
    let name = |n: &str| format!("model.llm.layers.{l}.{n}");
    let bytes = |n: &str, shape: &[usize]| ck.bytes(&name(n), "BF16", shape);
    let norms = if full {
        ["q_input_norm", "k_input_norm", "v_input_norm"].map(|n| ck.bf16(&name(&format!("{n}.weight")), &[HIDDEN]))
    } else {
        let n = ck.bf16(&name("attn_norm.weight"), &[HIDDEN]);
        [n.clone(), n.clone(), n]
    };
    let conv = |which: &str| {
        if full {
            bytes(&format!("attn.{which}_sconv.static_w"), &[4, KV])
        } else {
            bytes(&format!("attn.{which}_sconv.weight"), &[KV, 1, 4])
        }
    };
    let taps = |which: &str| {
        if full {
            bytes(&format!("attn.{which}_sconv.weight_proj"), &[32, HIDDEN])
        } else {
            Vec::new()
        }
    };
    Layer {
        layout: ProjLayout { full },
        wq: bytes("attn.wq_du.weight", &[Q, HIDDEN]),
        wr: bytes("attn.wr_du.weight", &[REL, HIDDEN]),
        sink: bytes("attn.attn_sink_proj.weight", &[SINKS, HIDDEN]),
        gate: bytes("attn.gate_attn.weight", &[Q, HIDDEN]),
        wk: bytes("attn.wk_dv.weight", &[KV, HIDDEN]),
        wv: bytes("attn.wv_dv.weight", &[KV, HIDDEN]),
        taps_k: taps("k"),
        taps_v: taps("v"),
        conv_k: conv("k"),
        conv_v: conv("v"),
        norms,
        q_scale: ck.bf16(&name("attn.q_l2_clip.scale"), &[24, HEAD_DIM]),
        rel_scale: ck.bf16(&name("attn.rel_l2_clip.scale"), &[24, 16]),
    }
}

impl Layer {
    fn full(&self) -> bool {
        self.layout.full
    }

    /// The joined matrix `J`, as the checkpoint rows.
    fn joined(&self) -> Vec<u8> {
        let mut parts: Vec<(Segment, &[u8])> = vec![
            (Segment::Q, &self.wq),
            (Segment::Rel, &self.wr),
            (Segment::Sink, &self.sink),
            (Segment::Gate, &self.gate),
            (Segment::K, &self.wk),
            (Segment::V, &self.wv),
        ];
        if self.full() {
            parts.push((Segment::TapsK, &self.taps_k));
            parts.push((Segment::TapsV, &self.taps_v));
        }
        join(&self.layout, &parts).unwrap()
    }
}

/// The inputs of a block: three input rows (the same row three times on a sliding layer), and the state of the two
/// convolutions before the block.
struct Inputs {
    x: [Vec<f32>; 3],
    rf: [Vec<f32>; 3],
    /// `[3][1024]` for the key and for the value, oldest row first.
    state: [Vec<f32>; 2],
}

fn make_inputs(layer: &Layer, rows: usize, seed: u64) -> Inputs {
    let first = make_rows(rows, HIDDEN, seed);
    let x = if layer.full() {
        [
            first,
            make_rows(rows, HIDDEN, seed + 1),
            make_rows(rows, HIDDEN, seed + 2),
        ]
    } else {
        [first.clone(), first.clone(), first]
    };
    let rf = [0, 1, 2].map(|i| x[i].chunks(HIDDEN).map(row_factor).collect());
    let mut rng = Rng::new(seed + 3);
    let state = [0, 1].map(|_| (0..3 * KV).map(|_| rng.value(4.0)).collect());
    Inputs { x, rf, state }
}

/// What the GPU or the host gives for a block.
#[derive(Default, Clone)]
struct Block {
    q: Vec<f32>,
    rel: Vec<f32>,
    sink: Vec<f32>,
    gate: Vec<f32>,
    /// The key and the value rows of the block, `[rows][1024]`.
    k: Vec<f32>,
    v: Vec<f32>,
    /// The conv states after the block.
    k_state: Vec<f32>,
    v_state: Vec<f32>,
}

impl Block {
    fn parts(&self) -> [(&'static str, &Vec<f32>); 8] {
        [
            ("q", &self.q),
            ("rel", &self.rel),
            ("sink", &self.sink),
            ("gate", &self.gate),
            ("k", &self.k),
            ("v", &self.v),
            ("k_state", &self.k_state),
            ("v_state", &self.v_state),
        ]
    }

    /// The first `rows` rows of each output of a block of more rows. The states are not a prefix: they are left out.
    fn prefix(&self, rows: usize, total: usize) -> Vec<Vec<u32>> {
        let cut = |v: &Vec<f32>| bits(&v[..v.len() / total * rows]);
        vec![
            cut(&self.q),
            cut(&self.rel),
            cut(&self.sink),
            cut(&self.gate),
            cut(&self.k),
            cut(&self.v),
        ]
    }
}

/// The normed row `r` of input `which` (0 q, 1 k, 2 v).
fn normed(layer: &Layer, inputs: &Inputs, which: usize, r: usize) -> Vec<f32> {
    let mut out = vec![0f32; HIDDEN];
    ops::rms_norm(
        &inputs.x[which][r * HIDDEN..(r + 1) * HIDDEN],
        &layer.norms[which],
        &mut out,
    );
    out
}

fn linear(w: &[u8], x: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    ops::matvec_bf16(w, x, &mut out);
    out
}

/// The host form of the rows `list` of q, rel, sink and gate (`host/forward.rs` `attention`), for the block at `pos0`.
fn host_rows(layer: &Layer, inputs: &Inputs, list: &[usize], pos0: usize) -> Block {
    let mut b = Block::default();
    for &r in list {
        let xq = normed(layer, inputs, 0, r);
        let mut q = linear(&layer.wq, &xq, Q);
        let mut rel = linear(&layer.wr, &xq, REL);
        ops::scaled_l2_clip(&mut q, &layer.q_scale, HEAD_DIM, ops::QK_CLIP);
        ops::scaled_l2_clip(&mut rel, &layer.rel_scale, ops::D_REL, ops::REL_CLIP);
        if layer.full() {
            let s = ops::log_scale(pos0 + r);
            q.iter_mut().chain(rel.iter_mut()).for_each(|v| *v *= s);
        }
        b.q.extend(q);
        b.rel.extend(rel);
        b.sink.extend(linear(&layer.sink, &xq, SINKS));
        b.gate.extend(linear(&layer.gate, &xq, Q).into_iter().map(ops::sigmoid));
    }
    b
}

/// The raw row `r` of the key (`key`) or the value, and its taps on a full layer: the projection of the normed input.
fn raw_row(layer: &Layer, inputs: &Inputs, key: bool, r: usize) -> (Vec<f32>, Vec<f32>) {
    let (which, w, taps_w) = if key {
        (1, &layer.wk, &layer.taps_k)
    } else {
        (2, &layer.wv, &layer.taps_v)
    };
    let input = normed(layer, inputs, which, r);
    let mut taps = vec![0f32; 32];
    if layer.full() {
        ops::matvec(&widen(taps_w), &input, &mut taps);
    }
    (linear(w, &input, KV), taps)
}

/// The host form of the rows `list` of the key or the value after the convolution and the clip (`host/forward.rs`
/// `kv`), and the conv state after the block of `rows` rows. Row `t` takes the three rows before it from the block,
/// or from the state before the block.
fn host_kv(layer: &Layer, inputs: &Inputs, key: bool, list: &[usize], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let (conv_w, state0) = if key {
        (&layer.conv_k, &inputs.state[0])
    } else {
        (&layer.conv_v, &inputs.state[1])
    };
    let static_w = widen(conv_w);
    // The row at the index `j` of the block, or of the state when `j` is negative.
    let raw = |j: isize| -> Vec<f32> {
        if j >= 0 {
            raw_row(layer, inputs, key, j as usize).0
        } else {
            state0[(3 + j) as usize * KV..(4 + j) as usize * KV].to_vec()
        }
    };
    let mut out = Vec::new();
    for &t in list {
        let (xp, taps) = raw_row(layer, inputs, key, t);
        let state: Vec<f32> = (-3..0).flat_map(|d| raw(t as isize + d)).collect();
        let mut o = vec![0f32; KV];
        if layer.full() {
            ops::dynamic_conv(&state, &xp, &static_w, &taps, &mut o);
        } else {
            ops::short_conv(&state, &xp, &static_w, &mut o);
        }
        for head in o.as_chunks_mut::<HEAD_DIM>().0 {
            if key {
                ops::l2_clip(head, ops::QK_CLIP);
            } else {
                ops::rms_norm_unweighted(head);
            }
        }
        out.extend(o);
    }
    let after: Vec<f32> = (0..3).flat_map(|j| raw(rows as isize - 3 + j)).collect();
    (out, after)
}

/// The host form of a whole block of `rows` rows at the position `pos0`.
fn host_block(layer: &Layer, inputs: &Inputs, rows: usize, pos0: usize) -> Block {
    let all: Vec<usize> = (0..rows).collect();
    let mut b = host_rows(layer, inputs, &all, pos0);
    (b.k, b.k_state) = host_kv(layer, inputs, true, &all, rows);
    (b.v, b.v_state) = host_kv(layer, inputs, false, &all, rows);
    b
}

/// The device side of one layer.
struct Rig<'d> {
    dev: &'d CudaDevice,
    layer: &'d Layer,
    max_rows: usize,
    packed: CUdeviceptr,
    table: CUdeviceptr,
    x: [CUdeviceptr; 3],
    rf: [CUdeviceptr; 3],
    out: CUdeviceptr,
    k_slot: CUdeviceptr,
    v_slot: CUdeviceptr,
    k_state: CUdeviceptr,
    v_state: CUdeviceptr,
    slot_rows: usize,
    ring: Ring<'d>,
    shared: u32,
    /// Every buffer of the rig, so that `free` gives them all back.
    owned: Vec<CUdeviceptr>,
}

impl<'d> Rig<'d> {
    fn new(
        dev: &'d CudaDevice,
        layer: &'d Layer,
        inputs: &Inputs,
        max_rows: usize,
        slot_rows: usize,
        profile: bool,
    ) -> Self {
        let shape = layer.layout.shape();
        let owned = std::cell::RefCell::new(Vec::new());
        let up = |p: CUdeviceptr| {
            owned.borrow_mut().push(p);
            p
        };
        let packed = up(upload(dev, &repack_bf16(&layer.joined(), &shape).unwrap()));
        let bf = |v: &[f32]| {
            up(upload(
                dev,
                &v.iter().map(|x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>(),
            ))
        };
        let x = [0, 1, 2].map(|i| up(upload(dev, &inputs.x[i])));
        let rf = [0, 1, 2].map(|i| up(upload(dev, &inputs.rf[i])));
        let (k_state, v_state) = (up(upload(dev, &inputs.state[0])), up(upload(dev, &inputs.state[1])));
        let k_slot = up(dev.alloc_device(slot_rows * KV * 4).unwrap());
        let v_slot = up(dev.alloc_device(slot_rows * KV * 4).unwrap());
        let mut table = MkTaskTable::default();
        // A sliding layer shares the norm weight between q, k and v. A full layer has three.
        let norm_ptrs = [bf(&layer.norms[0]), bf(&layer.norms[1]), bf(&layer.norms[2])];
        table.ptr[PROJ_TABLE_NORM_Q] = norm_ptrs[0];
        table.ptr[PROJ_TABLE_NORM_K] = norm_ptrs[1];
        table.ptr[PROJ_TABLE_NORM_V] = norm_ptrs[2];
        table.ptr[PROJ_TABLE_X_K] = x[1];
        table.ptr[PROJ_TABLE_X_V] = x[2];
        table.ptr[PROJ_TABLE_RF_K] = rf[1];
        table.ptr[PROJ_TABLE_RF_V] = rf[2];
        table.ptr[PROJ_TABLE_Q_SCALE] = bf(&layer.q_scale);
        table.ptr[PROJ_TABLE_REL_SCALE] = bf(&layer.rel_scale);
        table.ptr[PROJ_TABLE_K_CONV] = up(upload(dev, &layer.conv_k));
        table.ptr[PROJ_TABLE_V_CONV] = up(upload(dev, &layer.conv_v));
        table.ptr[PROJ_TABLE_K_STATE] = k_state;
        table.ptr[PROJ_TABLE_V_STATE] = v_state;
        table.ptr[PROJ_TABLE_K_OUT] = k_slot;
        table.ptr[PROJ_TABLE_V_OUT] = v_slot;
        let ring = Ring::new(dev, layer.layout.scratch_floats(max_rows), profile).unwrap();
        ring.zero_device().unwrap();
        Self {
            dev,
            layer,
            max_rows,
            packed,
            table: up(upload(dev, &[table])),
            x,
            rf,
            out: up(dev.alloc_device(ProjLayout::output_floats(max_rows) * 4).unwrap()),
            k_slot,
            v_slot,
            k_state,
            v_state,
            slot_rows,
            ring,
            shared: dev.max_dynamic_shared_bytes().unwrap(),
            owned: owned.into_inner(),
        }
    }

    fn task(&self, rows: usize, pos0: u32, ring_rows: u32) -> Task {
        let layout = &self.layer.layout;
        let desc = layout
            .descriptor(
                rows,
                pos0,
                ring_rows,
                self.x[0],
                self.packed,
                self.out,
                self.table,
                self.rf[0],
                self.ring.scratch(),
            )
            .unwrap();
        Task::with_items(desc)
            .reads(self.x[0], rows * HIDDEN * 4)
            .reads(self.packed, layout.shape().packed_bytes())
            .writes(self.out, ProjLayout::output_floats(rows) * 4)
            .writes(self.k_slot, self.slot_rows * KV * 4)
            .writes(self.v_slot, self.slot_rows * KV * 4)
            .writes(self.ring.scratch(), layout.scratch_floats(rows) * 4)
    }

    /// One launch of a block of `rows` rows, with the state of `inputs` before it.
    fn run(&mut self, inputs: &Inputs, rows: usize, pos0: u32, ring_rows: u32, grid: u32) -> Block {
        for (ptr, state) in [(self.k_state, &inputs.state[0]), (self.v_state, &inputs.state[1])] {
            unsafe { self.dev.copy_htod(ptr, state.as_ptr().cast(), state.len() * 4) }.unwrap();
        }
        self.dev
            .memset_zero(self.out, ProjLayout::output_floats(self.max_rows) * 4)
            .unwrap();
        self.dev.memset_zero(self.k_slot, self.slot_rows * KV * 4).unwrap();
        self.dev.memset_zero(self.v_slot, self.slot_rows * KV * 4).unwrap();
        let task = self.task(rows, pos0, ring_rows);
        self.ring.set(&[task]);
        run_with_watchdog(&self.ring, grid, 288, self.shared);
        let all: Vec<f32> = download(self.dev, self.out, ProjLayout::output_floats(rows));
        let (q, rest) = all.split_at(rows * Q);
        let (rel, rest) = rest.split_at(rows * REL);
        let (sink, gate) = rest.split_at(rows * SINKS);
        // The rows of the slot, in the order of the block: the row `t` is at `(pos0 + t) % ring`.
        let slot = |ptr: CUdeviceptr| -> Vec<f32> {
            let all: Vec<f32> = download(self.dev, ptr, self.slot_rows * KV);
            (0..rows)
                .flat_map(|t| {
                    let at = if ring_rows != 0 {
                        (pos0 as usize + t) % ring_rows as usize
                    } else {
                        pos0 as usize + t
                    };
                    all[at * KV..(at + 1) * KV].to_vec()
                })
                .collect()
        };
        Block {
            q: q.to_vec(),
            rel: rel.to_vec(),
            sink: sink.to_vec(),
            gate: gate.to_vec(),
            k: slot(self.k_slot),
            v: slot(self.v_slot),
            k_state: download(self.dev, self.k_state, 3 * KV),
            v_state: download(self.dev, self.v_state, 3 * KV),
        }
    }

    fn free(self) {
        for p in &self.owned {
            self.dev.free_device(*p).unwrap();
        }
        self.ring.free().unwrap();
    }
}

/// The error of `got` against `want`, in epsilons of `f32`, relative to the largest value of the vector that holds it
/// (`width` values): an element of a clipped row, a head or a row is judged by the scale of its own vector.
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

/// The widths of the vectors of each output: the unit in which `scaled_error` takes its scale.
fn widths() -> [(&'static str, usize); 8] {
    [
        ("q", HEAD_DIM),
        ("rel", 16),
        ("sink", SINKS),
        ("gate", Q),
        ("k", HEAD_DIM),
        ("v", HEAD_DIM),
        ("k_state", KV),
        ("v_state", KV),
    ]
}

/// The bound of the error of each output of the block, in the unit of `scaled_error` (epsilons of `f32` relative to
/// the largest value of the vector). The gate measured the worst values that `the_outputs_equal_the_host_forms`
/// prints; the bound is about twice the largest of them.
const PROJ_BOUND_EPS: f64 = 64.0;

fn compare(label: &str, got: &Block, want: &Block) -> f64 {
    let mut worst_all = 0.0f64;
    for ((name, g), (_, w)) in got.parts().into_iter().zip(want.parts()) {
        let width = widths().iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(
            g.len(),
            w.len(),
            "{label}: {name} has {} values, the host has {}",
            g.len(),
            w.len()
        );
        assert!(
            g.iter().all(|v| v.is_finite()),
            "{label}: {name} has a value that is not finite"
        );
        let error = scaled_error(g, w, width);
        assert!(
            error <= PROJ_BOUND_EPS,
            "{label}: {name}: the worst error is {error:.1} epsilons of the largest value of its vector; the bound is {PROJ_BOUND_EPS}"
        );
        eprintln!("    {label}: {name:<8} worst error {error:>7.2} epsilons of the largest value of its vector");
        worst_all = worst_all.max(error);
    }
    worst_all
}

const LAYERS: [usize; 3] = [0, 6, 5];

#[test]
fn the_outputs_equal_the_host_forms() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open();
    for l in LAYERS {
        let layer = load_layer(ck, l);
        // The full layer sits past position 8,191, where the log scale is not 1. The sliding layer writes to a ring of 3,072
        // rows, and its block crosses the end of the ring.
        let (pos0, ring_rows, slot_rows) = if layer.full() {
            (20_000u32, 0u32, 20_100usize)
        } else {
            (3_060, 3_072, 3_072)
        };
        for rows in [1usize, 16] {
            let inputs = make_inputs(&layer, rows, 100 + l as u64);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, slot_rows, false);
            let got = rig.run(&inputs, rows, pos0, ring_rows, 48);
            let want = host_block(&layer, &inputs, rows, pos0 as usize);
            eprintln!(
                "cuda_proj layer {l} ({}), {rows} rows at position {pos0}:",
                if layer.full() { "full" } else { "sliding" }
            );
            compare(&format!("layer {l} rows {rows}"), &got, &want);
            rig.free();
        }
    }
}

/// The rows of a block of 512 that the host checks: the first rows (they read the state), the rows after a tile
/// border, the middle, and the last rows.
const BLOCK_CHECK_ROWS: [usize; 14] = [0, 1, 2, 3, 4, 15, 16, 17, 255, 256, 257, 509, 510, 511];

#[test]
fn a_block_of_512_rows_convolves_across_its_own_rows() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open();
    for l in [5usize, 6] {
        let layer = load_layer(ck, l);
        let rows = 512;
        let pos0 = if layer.full() { 9_000u32 } else { 100 };
        let inputs = make_inputs(&layer, rows, 200 + l as u64);
        let mut rig = Rig::new(&dev, &layer, &inputs, rows, pos0 as usize + rows, false);
        let got = rig.run(&inputs, rows, pos0, 0, 48);
        // The host forms of the chosen rows. The block is long, so the host checks the rows that matter.
        let pick = |v: &[f32], width: usize| -> Vec<f32> {
            BLOCK_CHECK_ROWS
                .iter()
                .flat_map(|&r| v[r * width..(r + 1) * width].to_vec())
                .collect()
        };
        let mut want = host_rows(&layer, &inputs, &BLOCK_CHECK_ROWS, pos0 as usize);
        let (k, k_state) = host_kv(&layer, &inputs, true, &BLOCK_CHECK_ROWS, rows);
        let (v, v_state) = host_kv(&layer, &inputs, false, &BLOCK_CHECK_ROWS, rows);
        (want.k, want.k_state, want.v, want.v_state) = (k, k_state, v, v_state);
        let have = Block {
            q: pick(&got.q, Q),
            rel: pick(&got.rel, REL),
            sink: pick(&got.sink, SINKS),
            gate: pick(&got.gate, Q),
            k: pick(&got.k, KV),
            v: pick(&got.v, KV),
            k_state: got.k_state.clone(),
            v_state: got.v_state.clone(),
        };
        eprintln!(
            "cuda_proj layer {l}: a block of {rows} rows at position {pos0}, {} rows checked:",
            BLOCK_CHECK_ROWS.len()
        );
        compare(&format!("layer {l} block"), &have, &want);
        rig.free();
    }
}

#[test]
fn the_bits_do_not_depend_on_the_rows_or_the_grid() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open();
    for l in [5usize, 6] {
        let layer = load_layer(ck, l);
        let max = 512;
        let pos0 = if layer.full() { 9_000u32 } else { 100 };
        let inputs = make_inputs(&layer, max, 300 + l as u64);
        let mut rig = Rig::new(&dev, &layer, &inputs, max, pos0 as usize + max, false);
        let full = rig.run(&inputs, max, pos0, 0, 48);
        assert!(rig.ring.read_scoreboard().is_ok());
        // Row `t` reads only the rows up to `t`, so the first `rows` rows of a block of 512 equal a block of `rows`.
        for rows in [1usize, 7, 16, 17, 100] {
            let part = rig.run(&inputs, rows, pos0, 0, 48);
            assert_eq!(
                part.prefix(rows, rows),
                full.prefix(rows, max),
                "layer {l}: a block of {rows} rows differs from the first {rows} rows of 512"
            );
        }
        for grid in &GRIDS[1..] {
            let part = rig.run(&inputs, max, pos0, 0, *grid);
            assert_eq!(
                part.prefix(max, max),
                full.prefix(max, max),
                "layer {l}: grid {grid} differs from grid 48"
            );
            assert_eq!(
                bits(&part.k_state),
                bits(&full.k_state),
                "layer {l}: the key state differs at grid {grid}"
            );
            assert_eq!(
                bits(&part.v_state),
                bits(&full.v_state),
                "layer {l}: the value state differs at grid {grid}"
            );
        }
        eprintln!(
            "cuda_proj layer {l}: the bits of all outputs are equal at 1, 7, 16, 17, 100 and 512 rows and at grids {GRIDS:?}"
        );
        rig.free();
    }
}

/// The projection at one row must read at this share of the peak (the brief of the lane).
const PROJ_GATE: f64 = 0.80;

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
fn the_speed_of_the_projection() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open_profile();
    let blocks = dev.sm_count() as usize;
    let mut worst_at_one = f64::MAX;
    for l in [0usize, 6, 5] {
        let layer = load_layer(ck, l);
        let bytes = layer.layout.shape().packed_bytes();
        for rows in [1usize, 16, 512] {
            let inputs = make_inputs(&layer, rows, 400);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, 20_600, true);
            let tasks = 6;
            let list: Vec<Task> = (0..tasks).map(|_| rig.task(rows, 100, 0)).collect();
            rig.ring.set(&list);
            run_with_watchdog(&rig.ring, blocks as u32, 288, rig.shared);
            let profile = rig.ring.read_profile().unwrap();
            let spans: Vec<u64> = (1..tasks)
                .map(|t| task_ns(&profile, blocks, rig.ring.ring_index(t)))
                .collect();
            let (typical, best) = (median(spans.clone()), *spans.iter().min().unwrap());
            let gbps = bytes as f64 / typical as f64;
            let flops = 2.0 * rows as f64 * layer.layout.rows() as f64 * HIDDEN as f64;
            eprintln!(
                "cuda_proj speed: layer {l} ({}) J {} rows, {rows:>3} rows: {:>8.1} us (best {:>8.1} us), J {:.2} MB, {:.1} GB/s ({:.1}% of {PEAK_GBPS}), {:.1} TFLOP/s of f32-equivalent work",
                if layer.full() { "full" } else { "sliding" },
                layer.layout.rows(),
                typical as f64 / 1e3,
                best as f64 / 1e3,
                bytes as f64 / 1e6,
                gbps,
                gbps / PEAK_GBPS * 100.0,
                flops / typical as f64 / 1e3
            );
            if rows == 1 {
                worst_at_one = worst_at_one.min(gbps);
            }
            rig.free();
        }
    }
    assert!(
        worst_at_one >= PROJ_GATE * PEAK_GBPS,
        "the projection at one row reads {worst_at_one:.1} GB/s, under {:.0}% of {PEAK_GBPS} GB/s",
        PROJ_GATE * 100.0
    );
}
