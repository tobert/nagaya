//! GPU gates of `OP_PROJ`: the joined projection of the attention input, on real
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
use nagaya_mk_cuda::attn::{CacheFormat, KV_HEADS, plane_bytes, row_of, row_offset};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::proj::{ProjLayout, SINKS, Segment, join};
use nagaya_mk_cuda::protocol::{MkTaskTable, PROFILE_WORDS};
use nagaya_mk_cuda::repack::repack_bf16;
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_inkling2_bmg::host::ops::{self, HEAD_DIM};
use nagaya_mk_inkling2_sm121::cache::{RunCache, pack_row, unpack_row};

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
    /// The relative projection of the attention, `bf16`, `[16][1024]`.
    rel_proj: Vec<u8>,
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
        rel_proj: bytes("attn.rel_logits_proj.proj", &[16, 1024]),
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
    /// The bytes of the two planes of the cache, as the device holds them after the block.
    k_plane: Vec<u8>,
    v_plane: Vec<u8>,
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
        // The gate stays raw: the attention applies the sigmoid (`OP_ATTN`).
        b.gate.extend(linear(&layer.gate, &xq, Q));
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
    plane_rows: usize,
    format: CacheFormat,
    ring: Ring<'d>,
    shared: u32,
    /// Every buffer of the rig, so that `free` gives them all back.
    owned: Vec<CUdeviceptr>,
    /// Run the block form (`FLAG_BLOCK_FORM`) instead of the decode form.
    block: bool,
}

impl<'d> Rig<'d> {
    fn new(
        dev: &'d CudaDevice,
        layer: &'d Layer,
        inputs: &Inputs,
        max_rows: usize,
        plane_rows: usize,
        format: CacheFormat,
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
        let k_slot = up(dev.alloc_device(plane_bytes(plane_rows, format)).unwrap());
        let v_slot = up(dev.alloc_device(plane_bytes(plane_rows, format)).unwrap());
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
            plane_rows,
            format,
            ring,
            shared: dev.max_dynamic_shared_bytes().unwrap(),
            owned: owned.into_inner(),
            block: false,
        }
    }

    fn task(&self, rows: usize, pos0: u32) -> Task {
        let layout = &self.layer.layout;
        let build = if self.block {
            ProjLayout::descriptor_block
        } else {
            ProjLayout::descriptor
        };
        let desc = build(
            layout,
            rows,
            pos0,
            self.plane_rows as u32,
            self.format,
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
            .writes(self.k_slot, plane_bytes(self.plane_rows, self.format))
            .writes(self.v_slot, plane_bytes(self.plane_rows, self.format))
            .writes(self.ring.scratch(), layout.scratch_floats(rows) * 4)
    }

    /// One launch of a block of `rows` rows, with the state of `inputs` before it.
    fn run(&mut self, inputs: &Inputs, rows: usize, pos0: u32, grid: u32) -> Block {
        self.reset(inputs);
        let task = self.task(rows, pos0);
        self.ring.set(&[task]);
        run_with_watchdog(&self.ring, grid, 288, self.shared);
        self.collect(rows, pos0)
    }

    /// Put the state of `inputs` in place, and clear the output and the two planes.
    fn reset(&self, inputs: &Inputs) {
        for (ptr, state) in [(self.k_state, &inputs.state[0]), (self.v_state, &inputs.state[1])] {
            unsafe { self.dev.copy_htod(ptr, state.as_ptr().cast(), state.len() * 4) }.unwrap();
        }
        self.dev
            .memset_zero(self.out, ProjLayout::output_floats(self.max_rows) * 4)
            .unwrap();
        let plane = plane_bytes(self.plane_rows, self.format);
        self.dev.memset_zero(self.k_slot, plane).unwrap();
        self.dev.memset_zero(self.v_slot, plane).unwrap();
    }

    /// The outputs and the planes after a launch of `rows` rows at `pos0`.
    fn collect(&self, rows: usize, pos0: u32) -> Block {
        let plane = plane_bytes(self.plane_rows, self.format);
        let all: Vec<f32> = download(self.dev, self.out, ProjLayout::output_floats(rows));
        let (q, rest) = all.split_at(rows * Q);
        let (rel, rest) = rest.split_at(rows * REL);
        let (sink, gate) = rest.split_at(rows * SINKS);
        // The rows of the cache, in the order of the block: the 8 heads of a position, one after the other.
        let (k_plane, v_plane): (Vec<u8>, Vec<u8>) = (
            download(self.dev, self.k_slot, plane),
            download(self.dev, self.v_slot, plane),
        );
        let window = if self.layer.full() { 0 } else { 1 };
        let unpack = |bytes: &[u8]| -> Vec<f32> {
            let n = self.format.row_bytes();
            (0..rows)
                .flat_map(|t| {
                    let row = row_of(pos0 as usize + t, window, self.plane_rows);
                    (0..KV_HEADS).flat_map(move |h| {
                        let at = row_offset(h, row, self.plane_rows, self.format);
                        unpack_row(self.format, &bytes[at..at + n]).unwrap()
                    })
                })
                .collect()
        };
        Block {
            q: q.to_vec(),
            rel: rel.to_vec(),
            sink: sink.to_vec(),
            gate: gate.to_vec(),
            k: unpack(&k_plane),
            v: unpack(&v_plane),
            k_state: download(self.dev, self.k_state, 3 * KV),
            v_state: download(self.dev, self.v_state, 3 * KV),
            k_plane,
            v_plane,
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
        // The full layer sits past position 8,191, where the log scale is not 1. The sliding layer writes to a ring of 1,536
        // rows, and its block crosses the end of the ring.
        let (pos0, plane_rows) = if layer.full() {
            (20_000u32, 20_032usize)
        } else {
            (3_060, 1_536)
        };
        for rows in [1usize, 16] {
            let inputs = make_inputs(&layer, rows, 100 + l as u64);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::F32, false);
            let got = rig.run(&inputs, rows, pos0, 48);
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
        let mut rig = Rig::new(
            &dev,
            &layer,
            &inputs,
            rows,
            if layer.full() { 9_536 } else { 1_536 },
            CacheFormat::F32,
            false,
        );
        let got = rig.run(&inputs, rows, pos0, 48);
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
            ..Default::default()
        };
        eprintln!(
            "cuda_proj layer {l}: a block of {rows} rows at position {pos0}, {} rows checked:",
            BLOCK_CHECK_ROWS.len()
        );
        compare(&format!("layer {l} block"), &have, &want);
        rig.free();
    }
}

/// The counts of rows of the gate of the block form.
const FORM_ROWS: [usize; 7] = [1, 7, 16, 17, 100, 511, 512];

#[test]
fn the_block_form_equals_the_decode_form_and_does_not_depend_on_the_grid() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open();
    // Layer 6 is a full layer, and layer 5 a sliding one. Both formats of the cache run.
    for l in [5usize, 6] {
        let layer = load_layer(ck, l);
        let max = 512;
        let pos0 = if layer.full() { 9_000u32 } else { 100 };
        let inputs = make_inputs(&layer, max, 500 + l as u64);
        for format in [CacheFormat::F32, CacheFormat::E4m3] {
            let plane_rows = if layer.full() { 9_536 } else { 1_536 };
            let mut decode = Rig::new(&dev, &layer, &inputs, max, plane_rows, format, false);
            let mut block = Rig::new(&dev, &layer, &inputs, max, plane_rows, format, false);
            block.block = true;
            for rows in FORM_ROWS {
                let want = decode.run(&inputs, rows, pos0, 48);
                let got = block.run(&inputs, rows, pos0, 48);
                assert_eq!(
                    got.prefix(rows, rows),
                    want.prefix(rows, rows),
                    "layer {l} {format:?}: {rows} rows: an output of the block form differs from the decode form"
                );
                assert!(
                    got.k_plane == want.k_plane && got.v_plane == want.v_plane,
                    "layer {l} {format:?}: {rows} rows: a plane of the cache differs"
                );
                assert_eq!(
                    bits(&got.k_state),
                    bits(&want.k_state),
                    "layer {l} {format:?}: {rows} rows: the key state differs"
                );
                assert_eq!(
                    bits(&got.v_state),
                    bits(&want.v_state),
                    "layer {l} {format:?}: {rows} rows: the value state differs"
                );
            }
            // The grids: the block form at 512 rows gives the same bits at each width.
            let full = block.run(&inputs, max, pos0, 48);
            for grid in &GRIDS[1..] {
                let part = block.run(&inputs, max, pos0, *grid);
                assert_eq!(
                    part.prefix(max, max),
                    full.prefix(max, max),
                    "layer {l} {format:?}: grid {grid} differs from grid 48"
                );
                assert!(
                    part.k_plane == full.k_plane && part.v_plane == full.v_plane,
                    "layer {l} {format:?}: a plane of the cache differs at grid {grid}"
                );
                assert!(
                    bits(&part.k_state) == bits(&full.k_state) && bits(&part.v_state) == bits(&full.v_state),
                    "layer {l} {format:?}: a conv state differs at grid {grid}"
                );
            }
            eprintln!(
                "cuda_proj block: layer {l} {format:?}: outputs, cache planes and states equal the decode form at rows {FORM_ROWS:?}, and equal at grids {GRIDS:?}"
            );
            decode.free();
            block.free();
        }
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
        let mut rig = Rig::new(
            &dev,
            &layer,
            &inputs,
            max,
            if layer.full() { 9_536 } else { 1_536 },
            CacheFormat::F32,
            false,
        );
        let full = rig.run(&inputs, max, pos0, 48);
        assert!(rig.ring.read_scoreboard().is_ok());
        // Row `t` reads only the rows up to `t`, so the first `rows` rows of a block of 512 equal a block of `rows`.
        for rows in [1usize, 7, 16, 17, 100] {
            let part = rig.run(&inputs, rows, pos0, 48);
            assert_eq!(
                part.prefix(rows, rows),
                full.prefix(rows, max),
                "layer {l}: a block of {rows} rows differs from the first {rows} rows of 512"
            );
        }
        for grid in &GRIDS[1..] {
            let part = rig.run(&inputs, max, pos0, *grid);
            assert!(
                part.k_plane == full.k_plane && part.v_plane == full.v_plane,
                "layer {l}: a plane of the cache differs at grid {grid}"
            );
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
        for (rows, block) in [
            (1usize, false),
            (16, false),
            (17, false),
            (17, true),
            (33, false),
            (33, true),
            (64, false),
            (64, true),
            (128, false),
            (128, true),
            (512, false),
            (512, true),
        ] {
            let inputs = make_inputs(&layer, rows, 400);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, 1_536, CacheFormat::F32, true);
            rig.block = block;
            let tasks = 6;
            let list: Vec<Task> = (0..tasks).map(|_| rig.task(rows, 100)).collect();
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
                "cuda_proj speed: layer {l} ({}) J {} rows, {rows:>3} rows {}: {:>8.1} us (best {:>8.1} us), J {:.2} MB, {:.1} GB/s ({:.1}% of {PEAK_GBPS}), {:.1} TFLOP/s of f32-equivalent work",
                if layer.full() { "full" } else { "sliding" },
                layer.layout.rows(),
                if block { "block " } else { "decode" },
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

// ---------------------------------------------------------------------------------------------------------------
// The cache that the projection writes, and the attention that reads it.
// ---------------------------------------------------------------------------------------------------------------

/// The window of the sliding layers of Inkling 2 Mini.
const WINDOW: usize = 1024;

fn window_of(layer: &Layer) -> usize {
    if layer.full() { 0 } else { WINDOW }
}

/// The rows of the cache that the projection of the block writes, as the host cache image of `cache.rs` holds them:
/// built from the f32 rows `block.k` and `block.v` of the block (the 8 heads of a position, one after the other).
fn image_of(layer: &Layer, rig: &Rig, block: &Block, rows: usize, pos0: usize) -> RunCache {
    let mut image = RunCache::new(rig.format, window_of(layer), rig.plane_rows);
    for t in 0..rows {
        for h in 0..KV_HEADS {
            let at = t * KV + h * HEAD_DIM;
            image
                .write(pos0 + t, h, &block.k[at..at + HEAD_DIM], &block.v[at..at + HEAD_DIM])
                .unwrap();
        }
    }
    image
}

#[test]
fn the_cache_bytes_are_the_host_image_in_both_formats() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open();
    for l in LAYERS {
        let layer = load_layer(ck, l);
        let (pos0, plane_rows) = if layer.full() {
            (20_000u32, 20_032usize)
        } else {
            (3_060, 1_536)
        };
        for rows in [1usize, 16] {
            let inputs = make_inputs(&layer, rows, 500 + l as u64);
            // The f32 rows of the device, and the whole f32 plane: every byte that nothing wrote stays zero.
            let mut f32_rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::F32, false);
            let f32_block = f32_rig.run(&inputs, rows, pos0, 48);
            let image = image_of(&layer, &f32_rig, &f32_block, rows, pos0 as usize);
            assert!(
                f32_block.k_plane == image.keys && f32_block.v_plane == image.values,
                "layer {l}, {rows} rows: the f32 cache is not the host image of its own rows"
            );
            // The e4m3 cache: the packed bytes of the f32 rows of the same block, at the grids that divide no count.
            let mut e_rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::E4m3, false);
            for grid in [48, 7] {
                let e_block = e_rig.run(&inputs, rows, pos0, grid);
                let packed = image_of(&layer, &e_rig, &f32_block, rows, pos0 as usize);
                assert!(
                    e_block.k_plane == packed.keys && e_block.v_plane == packed.values,
                    "layer {l}, {rows} rows, grid {grid}: the e4m3 cache is not the pack of the f32 rows"
                );
            }
            eprintln!("cuda_proj cache: layer {l}, {rows} rows: the f32 and e4m3 planes equal the host image");
            f32_rig.free();
            e_rig.free();
        }
    }
}

/// The distinct rows of the history that the cache holds before the block: position `j` holds the pool row `j % POOL`.
const POOL: usize = 256;

struct Rng2(u64);

impl Rng2 {
    fn normal(&mut self) -> f32 {
        let mut sum = 0.0f32;
        for _ in 0..12 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            sum += (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        }
        sum - 6.0
    }
}

/// The history pool as the cache holds it (`format`): the rows after the pack and the unpack, key and value of each of
/// the 8 heads of each pool position.
struct History {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    packed_keys: Vec<Vec<u8>>,
    packed_values: Vec<Vec<u8>>,
}

fn make_history(format: CacheFormat) -> History {
    let mut rng = Rng2(0x1234_5678_9abc_def1);
    let n = format.row_bytes();
    let mut h = History {
        keys: vec![],
        values: vec![],
        packed_keys: vec![],
        packed_values: vec![],
    };
    for _ in 0..POOL * KV_HEADS {
        let mut k: Vec<f32> = (0..HEAD_DIM).map(|_| rng.normal()).collect();
        let mut v: Vec<f32> = (0..HEAD_DIM).map(|_| rng.normal()).collect();
        ops::l2_clip(&mut k, ops::QK_CLIP);
        ops::rms_norm_unweighted(&mut v);
        for (row, vals, packed) in [
            (k, &mut h.keys, &mut h.packed_keys),
            (v, &mut h.values, &mut h.packed_values),
        ] {
            let mut bytes = vec![0u8; n];
            pack_row(format, &row, &mut bytes).unwrap();
            vals.push(unpack_row(format, &bytes).unwrap());
            packed.push(bytes);
        }
    }
    h
}

/// The cache image of the positions before `pos0` (the last `plane_rows` of them in a ring).
fn history_image(h: &History, layer: &Layer, format: CacheFormat, plane_rows: usize, pos0: usize) -> RunCache {
    let mut image = RunCache::new(format, window_of(layer), plane_rows);
    let n = format.row_bytes();
    for j in pos0.saturating_sub(plane_rows)..pos0 {
        for kvh in 0..KV_HEADS {
            let at = row_offset(kvh, row_of(j, window_of(layer), plane_rows), plane_rows, format);
            image.keys[at..at + n].copy_from_slice(&h.packed_keys[(j % POOL) * KV_HEADS + kvh]);
            image.values[at..at + n].copy_from_slice(&h.packed_values[(j % POOL) * KV_HEADS + kvh]);
        }
    }
    image
}

/// What the attention reads: the projection outputs of the block, in f32, and the rows of the block that enter the
/// cache (`k` and `v`: `[rows][8][128]`, as the cache holds them).
struct AttnInputs<'a> {
    q: &'a [f32],
    rel: &'a [f32],
    sink: &'a [f32],
    gate: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
}

/// The host form of the attention of every row of the block: `ops::attend` over the history and the rows of the block
/// up to the row, with the relative bias of the row, times the sigmoid of the gate. `[rows][3072]`.
fn host_attention(layer: &Layer, h: &History, i: &AttnInputs, rows: usize, pos0: usize) -> Vec<f32> {
    let proj = widen(&layer.rel_proj);
    let window = window_of(layer);
    let mut out = vec![0f32; rows * Q];
    for t in 0..rows {
        let pos = pos0 + t;
        let mut profiles = vec![0f32; 24 * 1024];
        ops::relative_logits(&i.rel[t * REL..(t + 1) * REL], &proj, &mut profiles);
        let lowest = nagaya_mk_cuda::attn::lowest_key(pos, window);
        let heads: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..24)
                .map(|head| {
                    let (profiles, proj_row) = (&profiles, &i);
                    scope.spawn(move || {
                        let kvh = head / 3;
                        let row_of = |j: usize, keys: bool| -> &[f32] {
                            if j >= pos0 {
                                let at = (j - pos0) * KV + kvh * HEAD_DIM;
                                if keys {
                                    &proj_row.k[at..at + HEAD_DIM]
                                } else {
                                    &proj_row.v[at..at + HEAD_DIM]
                                }
                            } else if keys {
                                &h.keys[(j % POOL) * KV_HEADS + kvh]
                            } else {
                                &h.values[(j % POOL) * KV_HEADS + kvh]
                            }
                        };
                        let ks: Vec<&[f32]> = (lowest..=pos).map(|j| row_of(j, true)).collect();
                        let vs: Vec<&[f32]> = (lowest..=pos).map(|j| row_of(j, false)).collect();
                        let profile = &profiles[head * 1024..(head + 1) * 1024];
                        let bias: Vec<f32> = (lowest..=pos)
                            .map(|j| ops::relative_bias(profile, (pos - j) as i64))
                            .collect();
                        let q = &proj_row.q[t * Q + head * HEAD_DIM..t * Q + (head + 1) * HEAD_DIM];
                        let mut o = vec![0f32; HEAD_DIM];
                        ops::attend(q, &ks, &vs, &bias, proj_row.sink[t * SINKS + head], &mut o);
                        let g = &proj_row.gate[t * Q + head * HEAD_DIM..t * Q + (head + 1) * HEAD_DIM];
                        o.iter_mut().zip(g).for_each(|(o, g)| *o *= ops::sigmoid(*g));
                        o
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        out[t * Q..(t + 1) * Q].copy_from_slice(&heads.concat());
    }
    out
}

impl Rig<'_> {
    /// The task list of PROJ then ATTN in one ring, over the planes that hold `history`: the output of the attention
    /// `[rows][3072]`, and the projection outputs.
    fn chain(&mut self, inputs: &Inputs, rows: usize, pos0: u32, grid: u32, history: &RunCache) -> (Vec<f32>, Block) {
        use nagaya_mk_cuda::attn::{AttnBuffers, AttnShape, table};
        self.reset(inputs);
        let dev = self.dev;
        unsafe {
            dev.copy_htod(self.k_slot, history.keys.as_ptr().cast(), history.keys.len())
                .unwrap();
            dev.copy_htod(self.v_slot, history.values.as_ptr().cast(), history.values.len())
                .unwrap();
        }
        let window = window_of(self.layer);
        let shape = AttnShape::new(rows, pos0 as usize, window, self.plane_rows, self.format).unwrap();
        let proj_floats = self.layer.layout.scratch_floats(rows).next_multiple_of(4);
        let mut ring = Ring::new(dev, proj_floats + shape.scratch_floats().max(4), false).unwrap();
        ring.zero_device().unwrap();
        let rel = self.out + (rows * Q * 4) as u64;
        let sinks = rel + (rows * REL * 4) as u64;
        let gate = sinks + (rows * SINKS * 4) as u64;
        let proj = upload(dev, &self.layer.rel_proj);
        let attn_table = upload(dev, &[table(rel, sinks, gate, proj)]);
        let attn_out = dev.alloc_device(rows * Q * 4).unwrap();
        dev.memset_zero(attn_out, rows * Q * 4).unwrap();
        let plane = plane_bytes(self.plane_rows, self.format);
        let proj_desc = self
            .layer
            .layout
            .descriptor(
                rows,
                pos0,
                self.plane_rows as u32,
                self.format,
                self.x[0],
                self.packed,
                self.out,
                self.table,
                self.rf[0],
                ring.scratch(),
            )
            .unwrap();
        let attn_desc = shape.descriptor(&AttnBuffers {
            q: self.out,
            k: self.k_slot,
            v: self.v_slot,
            out: attn_out,
            scratch: ring.scratch() + (proj_floats * 4) as u64,
            table: attn_table,
        });
        let proj_task = Task::with_items(proj_desc)
            .reads(self.x[0], rows * HIDDEN * 4)
            .reads(self.packed, self.layer.layout.shape().packed_bytes())
            .writes(self.out, ProjLayout::output_floats(rows) * 4)
            .writes(self.k_slot, plane)
            .writes(self.v_slot, plane)
            .writes(ring.scratch(), proj_floats * 4);
        let attn_task = Task::with_items(attn_desc)
            .reads(self.out, ProjLayout::output_floats(rows) * 4)
            .reads(self.k_slot, plane)
            .reads(self.v_slot, plane)
            .writes(attn_out, rows * Q * 4)
            .writes(
                ring.scratch() + (proj_floats * 4) as u64,
                shape.scratch_floats().max(4) * 4,
            );
        ring.set(&[proj_task, attn_task]);
        run_with_watchdog(&ring, grid, 288, self.shared);
        assert!(ring.read_scoreboard().is_ok());
        let result: Vec<f32> = download(dev, attn_out, rows * Q);
        let block = self.collect(rows, pos0);
        ring.free().unwrap();
        dev.free_device(proj).unwrap();
        dev.free_device(attn_table).unwrap();
        dev.free_device(attn_out).unwrap();
        (result, block)
    }
}

/// The error of the attention output, relative to the largest value of its row, in the worst row.
fn attn_error(got: &[f32], want: &[f32]) -> f64 {
    let mut worst = 0.0f64;
    for (g, w) in got.chunks(Q).zip(want.chunks(Q)) {
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        for (a, b) in g.iter().zip(w) {
            worst = worst.max(f64::from((a - b).abs() / scale));
        }
    }
    worst
}

/// The bound of the attention output after the projection, in the unit of `attn_error`. The attention alone measured
/// 4e-6 against the host form; the projection adds its own error in q, rel, k and v (bound `PROJ_BOUND_EPS` epsilons).
const CHAIN_BOUND: f64 = 6e-6;

#[test]
fn the_projection_then_the_attention_equal_the_host_forms() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = open();
    let histories = [make_history(CacheFormat::F32), make_history(CacheFormat::E4m3)];
    for l in [0usize, 5] {
        let layer = load_layer(ck, l);
        // The sliding layer is past two wraps of its ring of 1,536 rows. The full layer is past position 8,191.
        let (pos0, plane_rows) = if layer.full() {
            (9_000u32, 9_536usize)
        } else {
            (3_060, 1_536)
        };
        for rows in [1usize, 16] {
            let inputs = make_inputs(&layer, rows, 600 + l as u64);
            let host = host_block(&layer, &inputs, rows, pos0 as usize);
            // f32 cache: the host form from the host rows of the projection.
            let image = history_image(&histories[0], &layer, CacheFormat::F32, plane_rows, pos0 as usize);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::F32, false);
            let (got, block) = rig.chain(&inputs, rows, pos0, 48, &image);
            let flat = |v: &[f32]| v.to_vec();
            let want = host_attention(
                &layer,
                &histories[0],
                &AttnInputs {
                    q: &flat(&host.q),
                    rel: &flat(&host.rel),
                    sink: &flat(&host.sink),
                    gate: &flat(&host.gate),
                    k: &flat(&host.k),
                    v: &flat(&host.v),
                },
                rows,
                pos0 as usize,
            );
            let error = attn_error(&got, &want);
            eprintln!("cuda_proj chain: layer {l}, {rows} rows, f32 cache: worst error {error:.2e} of the row maximum");
            assert!(
                error <= CHAIN_BOUND,
                "layer {l}, {rows} rows, f32: error {error:e} over {CHAIN_BOUND:e}"
            );
            // The bits of the chain do not depend on the grid.
            for grid in [24, 7, 1] {
                let (again, _) = rig.chain(&inputs, rows, pos0, grid, &image);
                assert_eq!(bits(&again), bits(&got), "layer {l}, {rows} rows: grid {grid} differs");
            }
            // e4m3 cache: the oracle takes the device rows of the projection, packed, so that the error of the
            // projection does not move a code.
            let image = history_image(&histories[1], &layer, CacheFormat::E4m3, plane_rows, pos0 as usize);
            let mut e_rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::E4m3, false);
            let (e_got, e_block) = e_rig.chain(&inputs, rows, pos0, 48, &image);
            let want = host_attention(
                &layer,
                &histories[1],
                &AttnInputs {
                    q: &block.q,
                    rel: &block.rel,
                    sink: &block.sink,
                    gate: &block.gate,
                    k: &e_block.k,
                    v: &e_block.v,
                },
                rows,
                pos0 as usize,
            );
            let error = attn_error(&e_got, &want);
            eprintln!(
                "cuda_proj chain: layer {l}, {rows} rows, e4m3 cache: worst error {error:.2e} of the row maximum"
            );
            assert!(error <= 4e-6, "layer {l}, {rows} rows, e4m3: error {error:e} over 4e-6");
            rig.free();
            e_rig.free();
        }
    }
}

/// The bf16 cache (cubin of the format `Bf16`): the planes are the host image of the rows that the projection computed,
/// at the grids that divide no count. The f32 rows come from the f32 cubin, so the test opens each device in turn.
#[test]
fn the_bf16_cache_bytes_are_the_host_image() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    for l in LAYERS {
        let layer = load_layer(ck, l);
        let (pos0, plane_rows) = if layer.full() {
            (20_000u32, 20_032usize)
        } else {
            (3_060, 1_536)
        };
        for rows in [1usize, 16] {
            let inputs = make_inputs(&layer, rows, 500 + l as u64);
            let f32_block = {
                let dev = common::gpu::open_cache(CacheFormat::F32);
                let mut rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::F32, false);
                let block = rig.run(&inputs, rows, pos0, 48);
                rig.free();
                block
            };
            let dev = common::gpu::open_cache(CacheFormat::Bf16);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::Bf16, false);
            let packed = image_of(&layer, &rig, &f32_block, rows, pos0 as usize);
            assert_eq!(
                packed.keys.len(),
                8 * plane_rows * 256,
                "a bf16 plane has 256 bytes for each row"
            );
            for grid in [48, 7] {
                let block = rig.run(&inputs, rows, pos0, grid);
                assert!(
                    block.k_plane == packed.keys && block.v_plane == packed.values,
                    "layer {l}, {rows} rows, grid {grid}: the bf16 cache is not the pack of the f32 rows"
                );
            }
            eprintln!("cuda_proj cache: layer {l}, {rows} rows: the bf16 planes equal the host image");
            rig.free();
        }
    }
}

/// The projection then the attention with a bf16 cache: the host form takes the device rows of the projection, rounded to
/// bf16, so that the error of the projection does not move a code.
#[test]
fn the_projection_then_the_attention_equal_the_host_forms_with_a_bf16_cache() {
    let Some(ck) = checkpoint("cuda_proj") else { return };
    let dev = common::gpu::open_cache(CacheFormat::Bf16);
    let history = make_history(CacheFormat::Bf16);
    for l in [0usize, 5] {
        let layer = load_layer(ck, l);
        let (pos0, plane_rows) = if layer.full() {
            (9_000u32, 9_536usize)
        } else {
            (3_060, 1_536)
        };
        for rows in [1usize, 16] {
            let inputs = make_inputs(&layer, rows, 600 + l as u64);
            let image = history_image(&history, &layer, CacheFormat::Bf16, plane_rows, pos0 as usize);
            let mut rig = Rig::new(&dev, &layer, &inputs, rows, plane_rows, CacheFormat::Bf16, false);
            let (got, block) = rig.chain(&inputs, rows, pos0, 48, &image);
            let want = host_attention(
                &layer,
                &history,
                &AttnInputs {
                    q: &block.q,
                    rel: &block.rel,
                    sink: &block.sink,
                    gate: &block.gate,
                    k: &block.k,
                    v: &block.v,
                },
                rows,
                pos0 as usize,
            );
            let error = attn_error(&got, &want);
            eprintln!(
                "cuda_proj chain: layer {l}, {rows} rows, bf16 cache: worst error {error:.2e} of the row maximum"
            );
            assert!(error <= 4e-6, "layer {l}, {rows} rows, bf16: error {error:e} over 4e-6");
            for grid in [24, 7, 1] {
                let (again, _) = rig.chain(&inputs, rows, pos0, grid, &image);
                assert_eq!(bits(&again), bits(&got), "layer {l}, {rows} rows: grid {grid} differs");
            }
            rig.free();
        }
    }
}
