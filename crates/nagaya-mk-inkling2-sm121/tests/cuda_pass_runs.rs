//! GPU gate of the runs of the pass (milestone 37 step 10, commits 5 to 8). Each run is built by `pass::run::run_tasks`
//! and runs in three stages with the inputs of the tape of the quantized reference:
//!
//! A. MIX (not layer 0), PROJ, ATTN, OUT_MIX, QUANT and ROUTE, from the rows `runNN.out` of the run before. The gate
//!    compares the attention output, `r` and `h_e` with the tape by a measured bound, the picks exactly, and it counts
//!    the flips of the quantized `h_e` against `x_buf`. Then the test writes the bytes of the tape over `Xq`, and the
//!    pair list that the tape implies over `Pairs`.
//! B. UP. The gate counts the flips of the quantized activation against `down_buf`, and writes the bytes of the tape
//!    over `Actq`.
//! C. DOWN, with `r` of the tape. The gate compares the sum of the slots with `moe`, and the row after the run with `out`.
//!
//! The bits of every stage must not depend on the grid (48, 24, 7, 1), and a row of a run of 1 row must equal the same
//! row of the run of 5 rows. The rows of the stack and of `Xw` that the run reads come from the tape, with their row
//! factors computed on the host. A run with messages is not covered yet.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_runs -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;
mod pass_support;

use std::path::PathBuf;

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::experts::{QuantRowsLayout, pair_list_bytes};
use nagaya_mk_cuda::protocol::expert_pair_bound;
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_inkling2_sm121::pass::arena::Arena;
use nagaya_mk_inkling2_sm121::pass::desc;
use nagaya_mk_inkling2_sm121::pass::dev::{Checkpoint, Dev};
use nagaya_mk_inkling2_sm121::pass::layout::{Buf, RunPlan, ScratchLayout, TaskKind, runs};
use nagaya_mk_inkling2_sm121::pass::run::{RunEnv, StepShape, TableSet, run_tasks};
use nagaya_mk_inkling2_sm121::pass::slot::Slot;
use nagaya_mk_inkling2_sm121::pass::weights::{DeviceWeights, expert_format};
use nagaya_mk_inkling2_sm121::quant::{Mxfp8Row, Nvfp4Row, dot_mxfp8_block_scaled, dot_nvfp4_block_scaled};
use nagaya_mk_inkling2_sm121::spec::{self, HIDDEN, INTERMEDIATE};
use pass_support::*;

const CYCLES: usize = 3;
const TAPE_ROWS: usize = 5;
const SEQ: usize = 256;

/// The bounds of the gates: the worst error over the rms of the tape, about 4 times the worst of the measurement of the runs
/// 0 to 17 (attn 7.2e-5, r 4.8e-5, h_e 9.6e-6). The tape holds the CPU reference, which rounds each weight of an expert to
/// bf16 (`expand_nvfp4_row`), while the card multiplies the exact block-scaled codes. So `moe` and `out` differ from
/// the tape by the rounding of the weights (up to 2.6e-2 and 4.2e-2). The slots of DOWN are gated by the exact
/// block-scaled dot of the host: bit for bit on NVFP4 layers, and within 1.0e-6 on MXFP8 layers, where the host sums in
/// sequence and the card sums with the MMA (`EXACT_BOUND`).
const ATTN_BOUND: f64 = 3.0e-4;
const R_BOUND: f64 = 2.0e-4;
const HE_BOUND: f64 = 4.0e-5;
const MOE_BOUND: f64 = 5.0e-2;
const OUT_BOUND: f64 = 8.0e-2;
const EXACT_BOUND: f64 = 4.0e-6;
/// The flips of the quantized `h_e` against the tape (measured 1 of 11,520 codes) and of the activation after the up
/// matmul (measured 702 of 46,080 codes: the weights of the tape are rounded to bf16).
const X_FLIP_FRACTION: f64 = 5.0e-3;
const ACT_FLIP_FRACTION: f64 = 5.0e-2;

fn model_dir() -> Option<PathBuf> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(dir) if PathBuf::from(&dir).join(spec::WEIGHTS_FILE).exists() => Some(dir.into()),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_runs: skipped, INKLING2_MODEL_DIR has no checkpoint");
            None
        }
    }
}

fn watchdog(ring: &Ring<'_>, grid: u32, dev: &CudaDevice) {
    common::gpu::run_with_watchdog(
        ring,
        grid,
        DEFAULT_BLOCK_THREADS,
        dev.max_dynamic_shared_bytes().unwrap(),
    );
}

/// A ring of tasks that shares the scratch of the test.
fn ring_of<'d>(dev: &'d CudaDevice, tasks: &[Task]) -> Ring<'d> {
    let mut ring = Ring::new(dev, 1, false).unwrap();
    ring.set(tasks);
    ring
}

/// The index of the run in the tape that gives the rows of `buf` for the run `plan`.
fn tape_run(buf: Buf, plan: &RunPlan) -> usize {
    match buf {
        Buf::B1 => 5,
        Buf::B2(p) => {
            // The block output after layer 11 of the cycle whose parity is `p`, as the plan reads it.
            let c = if plan.cycle % 2 == usize::from(p) {
                plan.cycle
            } else {
                plan.cycle - 1
            };
            6 + 12 * c + 5
        }
        Buf::B3 => 6 + 12 * (CYCLES - 1) + 11,
        Buf::Xw => {
            if plan.layer == 6 {
                5
            } else {
                plan.run - 1
            }
        }
        other => panic!("no tape run gives the rows of {other:?}"),
    }
}

/// The part of a snapshot that two runs compare.
type Snap = Vec<(&'static str, Vec<u8>)>;

struct Measured {
    attn: Gap,
    r: Gap,
    h_e: Gap,
    route_mismatch_rows: usize,
    pairs_equal: bool,
    weight_rel: f64,
    x_flips: Flips,
    act_flips: Flips,
    moe: Gap,
    out: Gap,
    /// The slots of DOWN against the exact block-scaled dot of the host, on the tape rows of the quantized activation.
    exact: Gap,
    /// The slots of the tape reference against the same exact dots.
    tape_exact: Gap,
}

struct Rig<'d> {
    dev: &'d CudaDevice,
    weights: &'d DeviceWeights<'d>,
    tape: Tape,
    ck: Checkpoint,
    ids: Vec<u32>,
    plans: Vec<RunPlan>,
}

impl<'d> Rig<'d> {
    fn rows_of(&mut self, run: usize, key: &str, rows: usize, width: usize) -> Vec<f32> {
        let all = self.tape.f32s(&format!("run{run:02}.{key}"));
        assert_eq!(all.len(), TAPE_ROWS * width);
        all[..rows * width].to_vec()
    }

    /// The slots of the first rows of DOWN against the exact dot of the block-scaled products, for 32 columns. The second
    /// result compares the sum of the exact slots with the `moe` of the tape.
    fn exact_slots(&mut self, run: usize, rows: usize, slots: &[f32], tape_moe: &[f32]) -> (Gap, Gap) {
        let plan = self.plans[run].clone();
        let format = expert_format(plan.layer);
        let l = plan.layer;
        let layout = QuantRowsLayout::new(format, 8 * TAPE_ROWS, INTERMEDIATE).unwrap();
        let buf = self.tape.bytes(&format!("run{run:02}.down_buf"));
        let ids = self.tape.i32s(&format!("run{run:02}.route_ids"));
        let bank = &self.weights.layers[l].experts;
        let (code_dtype, scale_dtype) = match format {
            nagaya_mk_cuda::experts::ExpertFormat::Nvfp4 => ("U8", "F8_E4M3"),
            nagaya_mk_cuda::experts::ExpertFormat::Mxfp8 => ("F8_E4M3", "U8"),
        };
        let (code_row, scale_row) = (format.code_bytes(INTERMEDIATE), format.scale_bytes(INTERMEDIATE));
        let columns: Vec<usize> = (0..16).chain(2000..2016).collect();
        let (mut got, mut want, mut tape_sum, mut tape_ref) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in 0..rows.min(2) {
            let mut exact_sum = vec![0.0f32; columns.len()];
            for j in 0..8 {
                let row = t * 8 + j;
                let e = ids[t * 8 + j] as usize;
                let codes = buf[layout.codes_offset(row)..][..code_row].to_vec();
                let scales = buf[layout.scales_offset(row)..][..scale_row].to_vec();
                let name = format!("model.llm.layers.{l}.mlp.experts.w2_weight");
                let (amax_w, _) = (
                    if format == nagaya_mk_cuda::experts::ExpertFormat::Nvfp4 {
                        self.ck.f32s(&format!("{name}_amax"), &[128]).unwrap()[e]
                    } else {
                        0.0
                    },
                    0,
                );
                for (ci, &col) in columns.iter().enumerate() {
                    let wc = self
                        .ck
                        .range(
                            &name,
                            code_dtype,
                            &[128, HIDDEN, code_row],
                            (e * HIDDEN + col) * code_row,
                            code_row,
                        )
                        .unwrap();
                    let ws = self
                        .ck
                        .range(
                            &format!("{name}_scale"),
                            scale_dtype,
                            &[128, HIDDEN, scale_row],
                            (e * HIDDEN + col) * scale_row,
                            scale_row,
                        )
                        .unwrap();
                    let v = match format {
                        nagaya_mk_cuda::experts::ExpertFormat::Nvfp4 => {
                            let amax = f32::from_le_bytes(buf[layout.amax_offset(row)..][..4].try_into().unwrap());
                            let act = Nvfp4Row {
                                codes: codes.clone(),
                                scales: scales.clone(),
                                amax,
                            };
                            dot_nvfp4_block_scaled(&act, &wc, &ws, amax_w, bank.down_denominator, 1.0).unwrap()
                        }
                        nagaya_mk_cuda::experts::ExpertFormat::Mxfp8 => {
                            let act = Mxfp8Row {
                                codes: codes.clone(),
                                scales: scales.clone(),
                            };
                            dot_mxfp8_block_scaled(&act, &wc, &ws, 1.0).unwrap()
                        }
                    };
                    got.push(slots[row * HIDDEN + col]);
                    want.push(v);
                    exact_sum[ci] += v;
                }
            }
            for (ci, &col) in columns.iter().enumerate() {
                tape_sum.push(exact_sum[ci]);
                tape_ref.push(tape_moe[t * HIDDEN + col]);
            }
        }
        (gap(&got, &want), gap(&tape_ref, &tape_sum))
    }

    /// One run at `rows` rows. `grids` are the widths; the accuracy comes from the first.
    fn exec(&mut self, run: usize, rows: usize, grids: &[u32]) -> (Measured, Vec<Snap>) {
        let dev = self.dev;
        let plan = self.plans[run].clone();
        let format = expert_format(plan.layer);
        let arena = Arena::new(dev, rows, 1).unwrap();
        let mut slot = Slot::new(dev, SEQ, spec::DEFAULT_EXTRA_RUNS, CacheFormat::F32).unwrap();
        let scratch_layout = ScratchLayout::new(rows, SEQ);
        let scratch = Dev::alloc(dev, scratch_layout.total_floats() * 4).unwrap();
        let table_dev = Dev::alloc(dev, 128 * 64).unwrap();
        let mut tables = TableSet::new(table_dev.ptr(), 64);
        let shape = StepShape { rows, pos0: 0 };
        let tasks = {
            let env = RunEnv {
                weights: &self.weights.layers[plan.layer],
                slot: &slot.runs[plan.state],
                arena: &arena,
                scratch: scratch.ptr(),
                scratch_layout: &scratch_layout,
                format: CacheFormat::F32,
            };
            run_tasks(&plan, &shape, &env, &mut tables).unwrap_or_else(|e| panic!("{e}"))
        };
        // The EMBED task, with its own table.
        let ids = Dev::upload(
            dev,
            &self.ids[..rows]
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let embed_table = tables
            .add(desc::embed_table(self.weights.embed_norm.ptr(), arena.addr(Buf::B0), arena.rf(Buf::B0)).unwrap())
            .unwrap();
        let mut embed_desc = desc::embed(
            rows,
            ids.ptr(),
            self.weights.embed.ptr(),
            arena.addr(Buf::B0),
            arena.rf(Buf::B0),
        )
        .unwrap();
        embed_desc.d = embed_table;
        let embed_task = Task::with_items(embed_desc)
            .writes(arena.addr(Buf::B0), rows * HIDDEN * 4)
            .writes(arena.rf(Buf::B0), rows * 4);
        let host_tables: Vec<u8> = tables
            .host()
            .iter()
            .flat_map(|t| t.ptr.iter().flat_map(|p| p.to_le_bytes()))
            .collect();
        table_dev.write(0, &host_tables).unwrap();
        let stage = |from: TaskKind, to: TaskKind| -> Vec<Task> {
            let a = tasks.iter().position(|(k, _)| *k == from).unwrap();
            let b = tasks.iter().position(|(k, _)| *k == to).unwrap();
            tasks[a..=b].iter().map(|(_, t)| t.clone()).collect()
        };
        // The order that the ranges give: each task waits, directly or by a chain, on the tasks that write what it reads.
        {
            let all: Vec<Task> = tasks.iter().map(|(_, t)| t.clone()).collect();
            let ring = ring_of(dev, &all);
            let planned = ring.planned(1);
            let at = |kind: TaskKind| ring.ring_index(tasks.iter().position(|(k, _)| *k == kind).unwrap());
            let waits_on = |later: TaskKind, earlier: TaskKind| {
                let target = at(earlier);
                let mut stack = vec![at(later)];
                let mut seen = vec![false; planned.len()];
                while let Some(t) = stack.pop() {
                    for w in &planned[t].waits {
                        let e = w.event as usize;
                        if e == target {
                            return true;
                        }
                        if !seen[e] {
                            seen[e] = true;
                            stack.push(e);
                        }
                    }
                }
                false
            };
            let needs = [
                (TaskKind::Attn, TaskKind::Proj),
                (TaskKind::OutMix, TaskKind::Attn),
                (TaskKind::Quant, TaskKind::OutMix),
                (TaskKind::Route, TaskKind::OutMix),
                (TaskKind::Up, TaskKind::Quant),
                (TaskKind::Up, TaskKind::Route),
                (TaskKind::Down, TaskKind::Up),
                (TaskKind::Down, TaskKind::Route),
                (TaskKind::Down, TaskKind::OutMix),
            ];
            for (later, earlier) in needs {
                assert!(
                    waits_on(later, earlier),
                    "run {run}: {later:?} does not wait on {earlier:?}"
                );
            }
            if plan.layer > 0 {
                assert!(
                    waits_on(TaskKind::Proj, TaskKind::Mix),
                    "run {run}: Proj does not wait on Mix"
                );
            }
        }
        let first = if plan.layer == 0 { TaskKind::Proj } else { TaskKind::Mix };
        let ring_embed = ring_of(dev, &[embed_task]);
        let ring_a = ring_of(dev, &stage(first, TaskKind::Route));
        let ring_b = ring_of(dev, &stage(TaskKind::Up, TaskKind::Up));
        let ring_c = ring_of(dev, &stage(TaskKind::Down, TaskKind::Down));

        // The inputs from the tape: the stack rows, the input row, and the quantized rows of the stages B and C.
        let tape_rows = rows;
        let inputs: Vec<(Buf, Vec<f32>)> = {
            let mut v = Vec::new();
            for buf in plan.reads() {
                if buf == Buf::B0 {
                    continue;
                }
                let src = tape_run(buf, &plan);
                v.push((buf, self.rows_of(src, "out", tape_rows, HIDDEN)));
            }
            v
        };
        let tape_r = self.rows_of(run, "r", rows, HIDDEN);
        let tape_h = self.rows_of(run, "h_e", rows, HIDDEN);
        let tape_attn = self.rows_of(run, "attn", rows, 3072);
        let tape_moe = self.rows_of(run, "moe", rows, HIDDEN);
        let tape_out = self.rows_of(run, "out", rows, HIDDEN);
        let all_ids = self.tape.i32s(&format!("run{run:02}.route_ids"));
        let all_w = self.tape.f32s(&format!("run{run:02}.route_weights"));
        let (route_ids, route_w): (Vec<u32>, Vec<f32>) = (
            all_ids[..rows * 8].iter().map(|x| *x as u32).collect(),
            all_w[..rows * 8].to_vec(),
        );
        let from_x = QuantRowsLayout::new(format, TAPE_ROWS, HIDDEN).unwrap();
        let to_x = QuantRowsLayout::new(format, rows, HIDDEN).unwrap();
        let tape_xq = convert_quant(&self.tape.bytes(&format!("run{run:02}.x_buf")), &from_x, &to_x, rows);
        let from_a = QuantRowsLayout::new(format, 8 * TAPE_ROWS, INTERMEDIATE).unwrap();
        let to_a = QuantRowsLayout::new(format, 8 * rows, INTERMEDIATE).unwrap();
        let tape_actq = convert_quant(
            &self.tape.bytes(&format!("run{run:02}.down_buf")),
            &from_a,
            &to_a,
            8 * rows,
        );
        let expect_pairs = expected_pairs(&route_ids, &route_w);
        let pair_capacity = expert_pair_bound(rows);
        let tape_pairs = pair_list_bytes(&expect_pairs, pair_capacity).unwrap();

        let write = |buf: Buf, data: &[u8]| arena.buffer.write(arena.layout.offset(buf), data).unwrap();
        let read = |buf: Buf, len: usize| arena.buffer.read(arena.layout.offset(buf), len).unwrap();
        let (e_off, h_off, _) = arena.layout.eout_parts();
        let _ = e_off;
        let mut snaps: Vec<Snap> = Vec::new();
        let mut measured = None;
        for (gi, &grid) in grids.iter().enumerate() {
            slot.zero().unwrap();
            arena.buffer.zero().unwrap();
            ids.write(
                0,
                &self.ids[..rows]
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
            scratch.zero().unwrap();
            ring_embed.zero_device().unwrap();
            watchdog(&ring_embed, grid, dev);
            for (buf, data) in &inputs {
                write(*buf, &f32_bytes(data));
                let rf: Vec<f32> = data.chunks(HIDDEN).map(row_factor).collect();
                arena.buffer.write(arena.layout.rf(*buf), &f32_bytes(&rf)).unwrap();
            }
            // Stage A.
            ring_a.zero_device().unwrap();
            watchdog(&ring_a, grid, dev);
            let attn = bytes_f32(&read(Buf::AttnOut, rows * 3072 * 4));
            let r = bytes_f32(&read(Buf::R, rows * HIDDEN * 4));
            let h_e = bytes_f32(
                &arena
                    .buffer
                    .read(arena.layout.offset(Buf::EOut) + h_off, rows * HIDDEN * 4)
                    .unwrap(),
            );
            let picks_raw = read(Buf::Picks, rows * 8 * 8);
            let picks: Vec<u32> = picks_raw[..rows * 32]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            let xq = read(Buf::Xq, to_x.bytes());
            let pairs_raw = read(
                Buf::Pairs,
                16 + pair_capacity * std::mem::size_of::<nagaya_mk_cuda::protocol::ExpertPair>(),
            );
            let mut snap_a: Snap = vec![
                ("attn", f32_bytes(&attn)),
                ("r", f32_bytes(&r)),
                ("h_e", f32_bytes(&h_e)),
                ("picks", picks_raw.clone()),
                ("xq", xq.clone()),
                ("pairs", pairs_raw.clone()),
            ];
            let route_bad = (0..rows)
                .filter(|t| picks[t * 8..(t + 1) * 8] != route_ids[t * 8..(t + 1) * 8])
                .count();
            let device_pairs = parse_pairs(&pairs_raw);
            let (pairs_equal, weight_rel) = pairs_close(&device_pairs, &expect_pairs);
            let pairs_equal = route_bad == 0 && pairs_equal;
            let x_flips = count_flips(&xq, &tape_xq, &to_x, rows);
            write(Buf::Xq, &tape_xq);
            write(Buf::Pairs, &tape_pairs);
            // Stage B.
            ring_b.zero_device().unwrap();
            watchdog(&ring_b, grid, dev);
            let actq = read(Buf::Actq, to_a.bytes());
            let act = read(Buf::Act, 8 * rows * INTERMEDIATE * 4);
            let act_flips = count_flips(&actq, &tape_actq, &to_a, 8 * rows);
            snap_a.push(("act", act));
            snap_a.push(("actq", actq));
            write(Buf::Actq, &tape_actq);
            // Stage C.
            write(Buf::R, &f32_bytes(&tape_r));
            ring_c.zero_device().unwrap();
            watchdog(&ring_c, grid, dev);
            let slots = bytes_f32(&read(Buf::Slots, 8 * rows * HIDDEN * 4));
            let out = bytes_f32(&read(plan.x_out, rows * HIDDEN * 4));
            let moe: Vec<f32> = (0..rows * HIDDEN)
                .map(|i| {
                    let (t, c) = (i / HIDDEN, i % HIDDEN);
                    (0..8).fold(0.0f32, |s, j| s + slots[(t * 8 + j) * HIDDEN + c])
                })
                .collect();
            let exact = if gi == 0 {
                self.exact_slots(run, rows, &slots, &tape_moe)
            } else {
                (gap(&[0.0], &[0.0]), gap(&[0.0], &[0.0]))
            };
            snap_a.push(("slots", f32_bytes(&slots)));
            snap_a.push(("out", f32_bytes(&out)));
            if gi == 0 {
                measured = Some(Measured {
                    attn: gap(&attn, &tape_attn),
                    r: gap(&r, &tape_r),
                    h_e: gap(&h_e, &tape_h),
                    route_mismatch_rows: route_bad,
                    pairs_equal,
                    weight_rel,
                    x_flips,
                    act_flips,
                    moe: gap(&moe, &tape_moe),
                    out: gap(&out, &tape_out),
                    exact: exact.0,
                    tape_exact: exact.1,
                });
            }
            snaps.push(snap_a);
        }
        (measured.unwrap(), snaps)
    }
}

fn compare_snaps(what: &str, a: &Snap, b: &Snap) {
    for ((name, x), (_, y)) in a.iter().zip(b) {
        assert!(x == y, "{what}: the bytes of {name} differ");
    }
}

#[test]
fn the_runs_match_the_tape_stage_by_stage() {
    let Some(dir) = model_dir() else { return };
    let Some(tape) = tape("cuda_pass_runs") else { return };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut rig = Rig {
        dev: &dev,
        weights: &weights,
        ck: Checkpoint::open(&dir).unwrap(),
        ids: Vec::new(),
        plans: runs(CYCLES),
        tape,
    };
    rig.ids = rig.tape.ids();
    assert_eq!(rig.ids.len(), TAPE_ROWS);
    let wanted: Vec<usize> = std::env::var("PASS_RUNS")
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|_| (0..18).chain(42..48).collect());
    for run in wanted {
        let grids = [48u32, 24, 7, 1];
        let (m, snaps) = rig.exec(run, TAPE_ROWS, &grids);
        eprintln!(
            "run {run:02} layer {:2}: attn {:.2e} ({:.2e} abs) r {:.2e} h_e {:.2e} | route rows wrong {} pairs equal {} (weights {:.1e}) | \
             x flips {:?} | act flips {:?} | moe {:.2e} out {:.2e} ({:.2e} abs) | DOWN slots vs exact dots {:.2e} ({:.2e} abs), tape moe vs exact {:.2e}",
            rig.plans[run].layer,
            m.attn.relative(),
            m.attn.max_abs,
            m.r.relative(),
            m.h_e.relative(),
            m.route_mismatch_rows,
            m.pairs_equal,
            m.weight_rel,
            m.x_flips,
            m.act_flips,
            m.moe.relative(),
            m.out.relative(),
            m.out.max_abs,
            m.exact.relative(),
            m.exact.max_abs,
            m.tape_exact.relative(),
        );
        assert!(m.attn.relative() < ATTN_BOUND, "run {run}: attn {:?}", m.attn);
        assert!(m.r.relative() < R_BOUND, "run {run}: r {:?}", m.r);
        assert!(m.h_e.relative() < HE_BOUND, "run {run}: h_e {:?}", m.h_e);
        assert_eq!(m.route_mismatch_rows, 0, "run {run}: the picks differ from the tape");
        assert!(m.pairs_equal, "run {run}: the pair list is not the list of the rules");
        assert!(
            m.exact.relative() < EXACT_BOUND,
            "run {run}: the slots of DOWN differ from the exact dots: {:?}",
            m.exact
        );
        assert!(
            (m.x_flips.codes as f64) < X_FLIP_FRACTION * m.x_flips.total_codes as f64,
            "run {run}: x flips {:?}",
            m.x_flips
        );
        assert!(
            (m.act_flips.codes as f64) < ACT_FLIP_FRACTION * m.act_flips.total_codes as f64,
            "run {run}: act flips {:?}",
            m.act_flips
        );
        assert!(m.moe.relative() < MOE_BOUND, "run {run}: moe {:?}", m.moe);
        assert!(m.out.relative() < OUT_BOUND, "run {run}: out {:?}", m.out);
        for (g, s) in grids.iter().zip(&snaps).skip(1) {
            compare_snaps(&format!("run {run}: grid {g} against grid 48"), &snaps[0], s);
        }
        // A run of 1 row: its row 0 has the bits of the row 0 of the run of 5 rows.
        let (_, one) = rig.exec(run, 1, &[48]);
        let five = &snaps[0];
        let one = &one[0];
        let prefix = |snap: &Snap, name: &str, bytes: usize| -> Vec<u8> {
            snap.iter().find(|(n, _)| *n == name).unwrap().1[..bytes].to_vec()
        };
        for (name, bytes) in [
            ("attn", 3072 * 4),
            ("r", HIDDEN * 4),
            ("act", 8 * INTERMEDIATE * 4),
            ("slots", 8 * HIDDEN * 4),
            ("out", HIDDEN * 4),
        ] {
            assert!(
                prefix(five, name, bytes) == prefix(one, name, bytes),
                "run {run}: row 0 of {name} differs between 1 and 5 rows"
            );
        }
    }
}
