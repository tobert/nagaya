//! The tasks of one run of the pass (`cuda` feature): the ops of a layer in order, each with its descriptor, its task
//! table and the ranges that it reads and writes. The ranges give the waits (`nagaya_mk_cuda::graph`).
//!
//! This builder covers the runs with no messages. A task writes only its own regions of the scratch.

use nagaya_mk_cuda::attn::{AttnBuffers, AttnShape, CacheFormat, table as attn_table};
use nagaya_mk_cuda::experts::{QuantRowsLayout, route_pair_list_bytes, route_picks_bytes};
use nagaya_mk_cuda::outmix::{self, OutMixBuffers, OutMixShape, OutMixTable};
use nagaya_mk_cuda::proj::ProjLayout;
use nagaya_mk_cuda::protocol::{MkTaskTable, expert_pair_bound};
use nagaya_mk_cuda::ring::Task;

use super::arena::Arena;
use super::desc::{self, ExpertAddrs, GateInputs, MixShape, MixTable, ProjTable, RouteAddrs};
use super::layout::{Buf, RunPlan, ScratchKind, ScratchLayout, TaskKind};
use super::slot::RunSlot;
use super::weights::{LayerWeights, expert_format};
use crate::spec::{HIDDEN, INTERMEDIATE};

/// The task tables of a ring, in one device array. A task has the table at `base + 128 * index`.
pub struct TableSet {
    base: u64,
    capacity: usize,
    tables: Vec<MkTaskTable>,
}

impl TableSet {
    /// A set over the device array at `base`, which has room for `capacity` tables.
    pub fn new(base: u64, capacity: usize) -> Self {
        Self {
            base,
            capacity,
            tables: Vec::new(),
        }
    }

    /// Add a table. The result is its device address.
    pub fn add(&mut self, table: MkTaskTable) -> Result<u64, String> {
        if self.tables.len() >= self.capacity {
            return Err(format!(
                "the ring has room for {} task tables. Make the array of tables larger",
                self.capacity
            ));
        }
        self.tables.push(table);
        Ok(self.base + 128 * (self.tables.len() as u64 - 1))
    }

    /// The tables, in the order of their addresses.
    pub fn host(&self) -> &[MkTaskTable] {
        &self.tables
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

/// The rows of a step and the position of its first row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepShape {
    pub rows: usize,
    pub pos0: usize,
}

/// What a run reads and writes outside its own tables.
pub struct RunEnv<'a, 'd> {
    pub weights: &'a LayerWeights<'d>,
    pub slot: &'a RunSlot<'d>,
    pub arena: &'a Arena<'d>,
    /// The device address of the scratch of the ring, and the offsets of its regions.
    pub scratch: u64,
    pub scratch_layout: &'a ScratchLayout,
    pub format: CacheFormat,
    /// The device address of the message queries of the layer (`f32`, `[4][2304]`), or 0 for a layer with none.
    pub message_queries: u64,
}

impl RunEnv<'_, '_> {
    fn region(&self, kind: ScratchKind) -> (u64, usize) {
        (
            self.scratch + self.scratch_layout.offset(kind) as u64,
            self.scratch_layout.bytes(kind),
        )
    }

    /// The address and the bytes of a buffer, and of its row factors.
    fn buf(&self, buf: Buf) -> (u64, usize) {
        (self.arena.addr(buf), self.arena.layout.place(buf).bytes)
    }

    fn rf(&self, buf: Buf, rows: usize) -> (u64, usize) {
        let per = if matches!(buf, Buf::A | Buf::U) { 3 } else { 1 };
        (self.arena.rf(buf), per * rows * 4)
    }
}

/// The scratch bytes that a descriptor names with `floats` floats must fit in the region.
fn fits(kind: ScratchKind, floats: usize, region: usize) -> Result<(), String> {
    if floats * 4 > region {
        return Err(format!(
            "the op of {kind:?} needs {} bytes of scratch, and its region has {region}. Make the scratch layout for more rows",
            floats * 4
        ));
    }
    Ok(())
}

fn add_reads(mut task: Task, spans: &[(u64, usize)]) -> Task {
    for (at, bytes) in spans {
        task = task.reads(*at, *bytes);
    }
    task
}

fn add_writes(mut task: Task, spans: &[(u64, usize)]) -> Task {
    for (at, bytes) in spans {
        task = task.writes(*at, *bytes);
    }
    task
}

/// A block of more rows than this runs in the block form of PROJ (`FLAG_BLOCK_FORM`, for 17 to 512 rows).
pub const BLOCK_FORM_ROWS_ABOVE: usize = 16;

/// The tasks of the run `plan`, in order, for a step of `shape`.
pub fn run_tasks(
    plan: &RunPlan,
    shape: &StepShape,
    env: &RunEnv<'_, '_>,
    tables: &mut TableSet,
) -> Result<Vec<(TaskKind, Task)>, String> {
    if plan.messages && env.message_queries == 0 {
        return Err(format!(
            "the run {} has messages, and the environment has no message queries. Give the queries of the layer",
            plan.run
        ));
    }
    let rows = shape.rows;
    let w = env.weights;
    let a = env.arena;
    let mut out: Vec<(TaskKind, Task)> = Vec::new();
    let row_bytes = rows * HIDDEN * 4;

    // The row that PROJ reads: the input row on layer 0, else the first output of the attention mix.
    let u_row = |i: usize| a.addr(Buf::U) + (i * row_bytes) as u64;
    let u_rf = |i: usize| a.rf(Buf::U) + (i * rows * 4) as u64;
    let (proj_x, proj_rf, proj_inputs): ([u64; 3], [u64; 3], Vec<(u64, usize)>) = if plan.messages {
        // The three gated inputs: the query, the key and the value.
        (
            std::array::from_fn(u_row),
            std::array::from_fn(u_rf),
            vec![env.buf(Buf::U), env.rf(Buf::U, rows)],
        )
    } else if plan.layer == 0 {
        let x = a.addr(plan.x_in);
        let rf = a.rf(plan.x_in);
        ([x; 3], [rf; 3], vec![env.buf(plan.x_in), env.rf(plan.x_in, rows)])
    } else {
        let queries = if plan.full { 3 } else { 1 };
        let x: [u64; 3] = std::array::from_fn(|i| a.addr(Buf::A) + (i.min(queries - 1) * row_bytes) as u64);
        let rf: [u64; 3] = std::array::from_fn(|i| a.rf(Buf::A) + (i.min(queries - 1) * rows * 4) as u64);
        (x, rf, vec![env.buf(Buf::A), env.rf(Buf::A, rows)])
    };

    // MIX: the attention residual over the stack of the block outputs and the input row.
    if plan.layer > 0 {
        let queries = if plan.full { 3 } else { 1 };
        let mix_shape = MixShape {
            rows,
            queries,
            depth: plan.depth_attn(),
            message_queries: if plan.messages { 4 } else { 0 },
        };
        let stack: Vec<(u64, u64)> = plan.stack_attn.iter().map(|b| (a.addr(*b), a.rf(*b))).collect();
        let query = w
            .attn_queries
            .as_ref()
            .ok_or_else(|| format!("layer {} has no attention queries. Load the weights again", plan.layer))?
            .ptr();
        let table = tables.add(
            MixTable {
                stack,
                message: plan.p.iter().chain(&plan.y).map(|b| (a.addr(*b), a.rf(*b))).collect(),
                queries: query,
                message_queries: env.message_queries,
            }
            .build()?,
        )?;
        let (m, m_rf) = if plan.messages {
            (a.addr(Buf::M), a.rf(Buf::M))
        } else {
            (0, 0)
        };
        let d = desc::mix(&mix_shape, a.addr(Buf::A), a.rf(Buf::A), m, m_rf, table)?;
        let mut reads: Vec<(u64, usize)> = plan.stack_attn.iter().map(|b| env.buf(*b)).collect();
        reads.extend(plan.stack_attn.iter().map(|b| env.rf(*b, rows)));
        for b in plan.p.iter().chain(&plan.y) {
            reads.push(env.buf(*b));
            reads.push(env.rf(*b, rows));
        }
        let task = add_reads(Task::with_items(d), &reads);
        let mut writes = vec![env.buf(Buf::A), env.rf(Buf::A, rows)];
        if plan.messages {
            writes.extend([env.buf(Buf::M), env.rf(Buf::M, rows)]);
        }
        out.push((TaskKind::Mix, add_writes(task, &writes)));
    }

    // The three message gates of the attention inputs: a gate blends the row of its query with its message.
    if plan.messages {
        let queries = if plan.full { 3 } else { 1 };
        let inputs: Vec<GateInputs> = (0..3)
            .map(|i| GateInputs {
                a: a.addr(Buf::A) + (i.min(queries - 1) * row_bytes) as u64,
                m: a.addr(Buf::M) + (i * row_bytes) as u64,
                a_rf: a.rf(Buf::A) + (i.min(queries - 1) * rows * 4) as u64,
                m_rf: a.rf(Buf::M) + (i * rows * 4) as u64,
            })
            .collect();
        let gate_down = w
            .gate_down
            .as_ref()
            .ok_or_else(|| format!("layer {} has no gates", plan.layer))?;
        let gate_up = w
            .gate_up
            .as_ref()
            .ok_or_else(|| format!("layer {} has no gates", plan.layer))?;
        let g = a.addr(Buf::G);
        let g_bytes = a.layout.place(Buf::G).bytes;
        {
            let (scratch, region) = env.region(ScratchKind::GateDown3);
            fits(ScratchKind::GateDown3, 16 + 3 * 24 * rows * 16, region)?;
            let table = tables.add(desc::gate_down_table(&inputs)?)?;
            let d = desc::gate_down(rows, 3, gate_down.ptr(), g, scratch, table)?;
            let task = add_reads(
                Task::with_items(d),
                &[
                    env.buf(Buf::A),
                    env.rf(Buf::A, rows),
                    env.buf(Buf::M),
                    env.rf(Buf::M, rows),
                ],
            );
            out.push((
                TaskKind::GateDown3,
                add_writes(task, &[(g, g_bytes), (scratch, region)]),
            ));
        }
        {
            let (scratch, region) = env.region(ScratchKind::GateUp3);
            fits(ScratchKind::GateUp3, 4 + 3 * 36 * rows, region)?;
            let outputs: Vec<(u64, u64)> = (0..3).map(|i| (u_row(i), u_rf(i))).collect();
            let table = tables.add(desc::gate_up_table(&inputs, &outputs, None)?)?;
            let d = desc::gate_up(rows, 3, g, gate_up.ptr(), None, scratch, table)?;
            let task = add_reads(
                Task::with_items(d),
                &[
                    (g, g_bytes),
                    env.buf(Buf::A),
                    env.buf(Buf::M),
                    env.rf(Buf::A, rows),
                    env.rf(Buf::M, rows),
                ],
            );
            out.push((
                TaskKind::GateUp3,
                add_writes(task, &[env.buf(Buf::U), env.rf(Buf::U, rows), (scratch, region)]),
            ));
        }
    }

    // PROJ.
    {
        let (scratch, region) = env.region(ScratchKind::Proj);
        let layout = desc::proj_layout(plan.full);
        fits(ScratchKind::Proj, layout.scratch_floats(rows), region)?;
        let norm = |i: usize| w.input_norms[i.min(w.input_norms.len() - 1)].ptr();
        let table = tables.add(
            ProjTable {
                norms: [norm(0), norm(1), norm(2)],
                x: [proj_x[1], proj_x[2]],
                rf: [proj_rf[1], proj_rf[2]],
                q_scale: w.q_scale.ptr(),
                rel_scale: w.rel_scale.ptr(),
                conv: [w.k_conv.ptr(), w.v_conv.ptr()],
                state: [env.slot.conv_ptr(0), env.slot.conv_ptr(1)],
                planes: [env.slot.k_plane.ptr(), env.slot.v_plane.ptr()],
            }
            .build()?,
        )?;
        // A prefill block of 17 to 512 rows runs in the block form: the bits of each row are those of the decode form.
        let build = if rows > BLOCK_FORM_ROWS_ABOVE {
            fits(ScratchKind::Proj, layout.block_scratch_floats(rows), region)?;
            ProjLayout::descriptor_block
        } else {
            ProjLayout::descriptor
        };
        let d = build(
            &layout,
            rows,
            shape.pos0 as u32,
            env.slot.plane_rows as u32,
            env.format,
            proj_x[0],
            w.j.ptr(),
            a.addr(Buf::ProjOut),
            table,
            proj_rf[0],
            scratch,
        )?;
        let task = add_reads(Task::with_items(d), &proj_inputs);
        let writes = [
            env.buf(Buf::ProjOut),
            (env.slot.k_plane.ptr(), env.slot.k_plane.bytes()),
            (env.slot.v_plane.ptr(), env.slot.v_plane.bytes()),
            (env.slot.conv_ptr(0), 3 * 1024 * 4),
            (env.slot.conv_ptr(1), 3 * 1024 * 4),
            (scratch, region),
        ];
        out.push((TaskKind::Proj, add_writes(task, &writes)));
    }

    // ATTN.
    {
        let (scratch, region) = env.region(ScratchKind::Attn);
        let attn_shape = AttnShape::new(rows, shape.pos0, env.slot.window, env.slot.plane_rows, env.format)?;
        fits(
            ScratchKind::Attn,
            attn_shape.scratch_floats(),
            region.max(attn_shape.scratch_floats() * 4),
        )?;
        if attn_shape.scratch_floats() * 4 > region {
            return Err(format!(
                "the attention of {rows} rows at position {} needs {} bytes of scratch, and its region has {region}. \
                 Make the scratch layout for a longer slot",
                shape.pos0,
                attn_shape.scratch_floats() * 4
            ));
        }
        let q = a.addr(Buf::ProjOut);
        let rel = q + (rows * 3072 * 4) as u64;
        let sinks = q + (rows * 3456 * 4) as u64;
        let gate = q + (rows * 3480 * 4) as u64;
        let table = tables.add(attn_table(rel, sinks, gate, w.rel_proj.ptr()))?;
        let d = attn_shape.descriptor(&AttnBuffers {
            q,
            k: env.slot.k_plane.ptr(),
            v: env.slot.v_plane.ptr(),
            out: a.addr(Buf::AttnOut),
            scratch,
            table,
        });
        let task = add_reads(
            Task::with_items(d),
            &[
                env.buf(Buf::ProjOut),
                (env.slot.k_plane.ptr(), env.slot.k_plane.bytes()),
                (env.slot.v_plane.ptr(), env.slot.v_plane.bytes()),
            ],
        );
        let writes = [env.buf(Buf::AttnOut), (scratch, region)];
        out.push((TaskKind::Attn, add_writes(task, &writes)));
    }

    // OUT_MIX.
    {
        let (scratch, region) = env.region(ScratchKind::OutMix);
        let depth = plan.depth_mlp();
        let om = OutMixShape::new(rows, depth, !plan.first_of_block, !plan.messages)?;
        fits(ScratchKind::OutMix, outmix::scratch_floats(rows), region)?;
        let planes: Vec<(u64, u64)> = plan.stack_mlp[..depth - 1]
            .iter()
            .map(|b| (a.addr(*b), a.rf(*b)))
            .collect();
        let table = tables.add(
            OutMixTable {
                planes,
                r_rf: a.rf(Buf::R),
                r: a.addr(Buf::R),
                conv_weight: w.attn_conv.ptr(),
                conv_state: env.slot.conv_ptr(2),
                norm_weight: if plan.messages { 0 } else { w.mlp_norm.ptr() },
                out: a.addr(Buf::EOut),
                query: w.mlp_query.ptr(),
            }
            .table()?,
        )?;
        let x = if plan.first_of_block { 0 } else { a.addr(plan.x_in) };
        let d = om.descriptor(&OutMixBuffers {
            attn: a.addr(Buf::AttnOut),
            packed: w.wo.ptr(),
            r: a.addr(Buf::R),
            x,
            scratch,
            table,
        })?;
        let mut reads = vec![env.buf(Buf::AttnOut)];
        for b in &plan.stack_mlp[..depth - 1] {
            reads.push(env.buf(*b));
            reads.push(env.rf(*b, rows));
        }
        if !plan.first_of_block {
            reads.push(env.buf(plan.x_in));
        }
        let task = add_reads(Task::with_items(d), &reads);
        let writes = [
            env.buf(Buf::R),
            env.rf(Buf::R, rows),
            env.buf(Buf::EOut),
            (env.slot.conv_ptr(2), 3 * HIDDEN * 4),
            (scratch, region),
        ];
        out.push((TaskKind::OutMix, add_writes(task, &writes)));
    }

    let (_, h_off, rf_off) = a.layout.eout_parts();
    let format = expert_format(plan.layer);
    let h_e = a.addr(Buf::EOut) + h_off as u64;
    let h_e_span = (h_e, row_bytes);
    // The fourth gate blends the row `e` of the mlp mix with its message, and writes `h_e`, the input of the experts.
    if plan.messages {
        let e = a.addr(Buf::EOut);
        let inputs = [GateInputs {
            a: e,
            m: a.addr(Buf::M) + (3 * row_bytes) as u64,
            a_rf: a.addr(Buf::EOut) + rf_off as u64,
            m_rf: a.rf(Buf::M) + (3 * rows * 4) as u64,
        }];
        let gate_down = w
            .gate_down
            .as_ref()
            .ok_or_else(|| format!("layer {} has no gates", plan.layer))?;
        let gate_up = w
            .gate_up
            .as_ref()
            .ok_or_else(|| format!("layer {} has no gates", plan.layer))?;
        let down_per_gate = nagaya_mk_cuda::rows::gate_down_shape().packed_bytes() as u64;
        let up_per_gate = (nagaya_mk_cuda::rows::GATE_UP_TILES * nagaya_mk_cuda::rows::GATE_UP_ITEM_BYTES) as u64;
        let g4 = a.addr(Buf::G) + (3 * rows * 64 * 4) as u64;
        let g4_bytes = rows * 64 * 4;
        let e_spans = [
            (e, row_bytes),
            (a.addr(Buf::EOut) + rf_off as u64, rows * 4),
            (a.addr(Buf::M) + (3 * row_bytes) as u64, row_bytes),
            (a.rf(Buf::M) + (3 * rows * 4) as u64, rows * 4),
        ];
        {
            let (scratch, region) = env.region(ScratchKind::GateDownE);
            fits(ScratchKind::GateDownE, 16 + 24 * rows * 16, region)?;
            let table = tables.add(desc::gate_down_table(&inputs)?)?;
            let d = desc::gate_down(rows, 1, gate_down.ptr() + 3 * down_per_gate, g4, scratch, table)?;
            out.push((
                TaskKind::GateDownE,
                add_writes(
                    add_reads(Task::with_items(d), &e_spans),
                    &[(g4, g4_bytes), (scratch, region)],
                ),
            ));
        }
        {
            let (scratch, region) = env.region(ScratchKind::GateUpE);
            fits(ScratchKind::GateUpE, 4 + 36 * rows, region)?;
            let table = tables.add(desc::gate_up_table(
                &inputs,
                &[(u_row(0), u_rf(0))],
                Some(w.mlp_norm.ptr()),
            )?)?;
            let d = desc::gate_up(rows, 1, g4, gate_up.ptr() + 3 * up_per_gate, Some(h_e), scratch, table)?;
            let mut reads = vec![(g4, g4_bytes)];
            reads.extend(e_spans);
            out.push((
                TaskKind::GateUpE,
                add_writes(
                    add_reads(Task::with_items(d), &reads),
                    &[env.buf(Buf::U), env.rf(Buf::U, rows), h_e_span, (scratch, region)],
                ),
            ));
        }
    }
    // QUANT of h_e, and ROUTE of h_e, in parallel: they read the same rows and write other buffers.
    let xq_layout = QuantRowsLayout::new(format, rows, HIDDEN)?;
    out.push((
        TaskKind::Quant,
        add_writes(
            add_reads(
                Task::with_items(desc::quant(format, rows, HIDDEN, h_e, a.addr(Buf::Xq))?),
                &[h_e_span],
            ),
            &[(a.addr(Buf::Xq), xq_layout.bytes())],
        ),
    ));
    {
        let (scratch, region) = env.region(ScratchKind::Route);
        fits(
            ScratchKind::Route,
            nagaya_mk_cuda::experts::route_scratch_words(rows),
            region,
        )?;
        let table = tables.add(desc::route_table(w.router_bias.ptr(), None)?)?;
        let d = desc::route(
            rows,
            &RouteAddrs {
                h: h_e,
                router: w.router.ptr(),
                picks: a.addr(Buf::Picks),
                pairs: a.addr(Buf::Pairs),
                scratch,
                table,
            },
            false,
        )?;
        out.push((
            TaskKind::Route,
            add_writes(
                add_reads(Task::with_items(d), &[h_e_span]),
                &[
                    (a.addr(Buf::Picks), route_picks_bytes(rows)),
                    (a.addr(Buf::Pairs), route_pair_list_bytes(rows)),
                    (scratch, region),
                ],
            ),
        ));
    }

    // UP: the gate and up matmul, the activation, and its quantization.
    let actq_layout = QuantRowsLayout::new(format, 8 * rows, INTERMEDIATE)?;
    let bank = &w.experts;
    {
        let (counters, region) = env.region(ScratchKind::Up);
        let table = tables.add(desc::up_table(
            bank.up_amax.as_ref().map(|d| d.ptr()),
            a.addr(Buf::Actq),
        )?)?;
        let d = desc::up(
            format,
            rows,
            rows,
            &ExpertAddrs {
                input: a.addr(Buf::Xq),
                bank: bank.up.ptr(),
                out: a.addr(Buf::Act),
                table,
                pairs: a.addr(Buf::Pairs),
                counters,
            },
            bank.up_denominator,
        )?;
        fits(ScratchKind::Up, expert_pair_bound(rows), region)?;
        out.push((
            TaskKind::Up,
            add_writes(
                add_reads(
                    Task::with_items(d),
                    &[
                        (a.addr(Buf::Xq), xq_layout.bytes()),
                        (a.addr(Buf::Pairs), route_pair_list_bytes(rows)),
                    ],
                ),
                &[
                    env.buf(Buf::Act),
                    (a.addr(Buf::Actq), actq_layout.bytes()),
                    (counters, region),
                ],
            ),
        ));
    }

    // DOWN: the down matmul, the slot sum, the conv of the mlp and the residual.
    {
        let (counters, region) = env.region(ScratchKind::Down);
        let table = tables.add(desc::down_table(
            bank.down_amax.as_ref().map(|d| d.ptr()),
            a.addr(Buf::R),
            w.mlp_conv.ptr(),
            env.slot.conv_ptr(3),
            a.addr(plan.x_out),
            a.rf(plan.x_out),
        )?)?;
        fits(
            ScratchKind::Down,
            nagaya_mk_cuda::experts::down_scratch_words(HIDDEN, rows),
            region,
        )?;
        let d = desc::down(
            format,
            rows,
            &ExpertAddrs {
                input: a.addr(Buf::Actq),
                bank: bank.down.ptr(),
                out: a.addr(Buf::Slots),
                table,
                pairs: a.addr(Buf::Pairs),
                counters,
            },
            bank.down_denominator,
        )?;
        out.push((
            TaskKind::Down,
            add_writes(
                add_reads(
                    Task::with_items(d),
                    &[
                        (a.addr(Buf::Actq), actq_layout.bytes()),
                        (a.addr(Buf::Pairs), route_pair_list_bytes(rows)),
                        env.buf(Buf::R),
                        (env.slot.conv_ptr(3), 3 * HIDDEN * 4),
                    ],
                ),
                &[
                    env.buf(Buf::Slots),
                    env.buf(plan.x_out),
                    env.rf(plan.x_out, rows),
                    (env.slot.conv_ptr(3), 3 * HIDDEN * 4),
                    (counters, region),
                ],
            ),
        ));
    }
    debug_assert_eq!(out.iter().map(|(k, _)| *k).collect::<Vec<_>>(), plan.task_kinds());
    Ok(out)
}
