//! A pass of the model on the device: the embedding, the runs, the output residual and the head (`cuda` feature).
//!
//! One launch of one ring runs the whole pass: 1 + the tasks of the runs + 2 tasks. The DOWN op of each run writes the
//! row factors of the row that it wrote, so no task waits for the host. The waits come from the ranges that each task
//! declares (`nagaya_mk_cuda::graph`).
//!
//! One thread owns a pass. A launch that fails poisons the slot, and the pass refuses the next call until the slot is zeroed.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nagaya_mk_cuda::attn::{AttnShape, CacheFormat};
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::ring::{Ring, RingList, Task};

use super::arena::Arena;
use super::desc::{self, MixTable};
use super::dev::Dev;
use super::layout::{Buf, LOGIT_COLUMNS, RunPlan, ScratchKind, ScratchLayout, TaskKind, runs};
use super::run::{BLOCK_FORM_ROWS_ABOVE, RunEnv, StepShape, TableSet, run_tasks};
use super::slot::Slot;
use super::weights::DeviceWeights;
use crate::spec::{CYCLE_LAYERS, CYCLE_START, HIDDEN, LOGITS, MAX_LOGIT_ROWS};

/// The tables that a pass can hold: the embedding, 12 for each run of 5 cycles at most, and the output residual and head.
const TABLES: usize = 1024;

/// The time of a launch.
#[derive(Clone, Debug)]
pub struct Launch {
    pub what: String,
    pub wall: Duration,
}

/// What the pass knows of a task: its kind, its run, and the bytes of weights that it reads for one row.
#[derive(Clone, Debug)]
pub struct OpTag {
    pub name: &'static str,
    pub run: Option<usize>,
    pub layer: Option<usize>,
    /// The bytes of weights that the task reads for a call of 1 row. The cache of the attention is not counted.
    pub weight_bytes: usize,
}

/// The time of one task in the last call that the pass profiled: from the first wait that ended to the last end of an item.
#[derive(Clone, Debug)]
pub struct OpSpan {
    pub tag: OpTag,
    pub items: u32,
    pub ns: u64,
}

/// The times of the blocks of one task: `[t_start, t_after_wait, t_end]` in nanoseconds of the device timer, for each block.
#[derive(Clone, Debug)]
pub struct TaskTimes {
    pub tag: OpTag,
    pub items: u32,
    pub blocks: Vec<[u64; 3]>,
}

fn kind_name(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Mix => "MIX",
        TaskKind::GateDown3 => "GATE_DOWN",
        TaskKind::GateUp3 => "GATE_UP",
        TaskKind::Proj => "PROJ",
        TaskKind::Attn => "ATTN",
        TaskKind::OutMix => "OUT_MIX",
        TaskKind::GateDownE => "GATE_DOWN_e",
        TaskKind::GateUpE => "GATE_UP_e",
        TaskKind::Quant => "QUANT",
        TaskKind::Route => "ROUTE",
        TaskKind::Up => "UP",
        TaskKind::Down => "DOWN",
    }
}

/// The bytes of weights that a task reads for a call of 1 row (8 experts, each read once).
fn weight_bytes(kind: TaskKind, plan: &RunPlan) -> usize {
    let nvfp4 = super::weights::expert_format(plan.layer) == nagaya_mk_cuda::experts::ExpertFormat::Nvfp4;
    // The bytes of one expert of 2,304 rows and `k` values: the codes and the scales.
    let expert = |k: usize| {
        let (codes, scales) = if nvfp4 { (k / 2, k / 16) } else { (k, k / 32) };
        HIDDEN * (codes + scales)
    };
    match kind {
        TaskKind::Mix | TaskKind::Attn | TaskKind::Quant => 0,
        TaskKind::GateDown3 => 3 * 64 * 2 * HIDDEN * 2,
        TaskKind::GateUp3 => 3 * 2 * HIDDEN * 64 * 2,
        TaskKind::GateDownE => 64 * 2 * HIDDEN * 2,
        TaskKind::GateUpE => 2 * HIDDEN * 64 * 2,
        TaskKind::Proj => super::desc::proj_layout(plan.full).rows() * HIDDEN * 2,
        TaskKind::OutMix => HIDDEN * 3072 * 2,
        TaskKind::Route => 128 * HIDDEN * 2,
        TaskKind::Up => 8 * expert(HIDDEN),
        TaskKind::Down => 8 * expert(crate::spec::INTERMEDIATE),
    }
}

/// What makes two calls share a ring: the arena and the tasks differ only in the fields that [`Pass::patch`] sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ShapeKey {
    rows: usize,
    cycles: usize,
    logit_rows: usize,
    /// The block form of PROJ (more than `BLOCK_FORM_ROWS_ABOVE` rows) has other descriptors and other scratch.
    block: bool,
}

/// A field of a task that moves between calls of one shape.
#[derive(Clone, Copy, Debug)]
struct Patch {
    /// The number of the task in the list of the ring.
    input: usize,
    kind: TaskKind,
    /// The index of the run in `plans`.
    plan: usize,
}

/// A ring list of one shape, with what the pass needs to launch it again.
struct Built {
    /// The list, while another shape holds the ring. It is empty while this shape holds the ring.
    list: RingList,
    /// The task tables, as bytes for the device array.
    tables: Vec<u8>,
    tags: Vec<OpTag>,
    proj_block_form: usize,
    patches: Vec<Patch>,
}

/// The host time of the last call, by part. The kernel time is in `Launch::wall` less `prep`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostTime {
    /// The whole call, from the checks to the logits.
    pub call: Duration,
    /// The build of the ring: zero when the call found the ring of its shape.
    pub build: Duration,
    /// The patches of the tasks that move between calls.
    pub patch: Duration,
    /// The descriptors and their copy to the device, inside the launch.
    pub prep: Duration,
}

pub struct Pass<'d> {
    dev: &'d CudaDevice,
    weights: &'d DeviceWeights<'d>,
    pub slot: Slot<'d>,
    cycles: usize,
    max_rows: usize,
    scratch_layout: ScratchLayout,
    ring: Ring<'d>,
    tables: Dev<'d>,
    arenas: HashMap<(usize, usize), Arena<'d>>,
    plans: Vec<RunPlan>,
    /// The width of the launches. 48 is the width of the GPU.
    pub grid: u32,
    /// The launches of the last call, in order.
    pub launches: Vec<Launch>,
    /// The count of PROJ tasks of the last call that ran in the block form (a block of more than 16 rows).
    pub proj_block_form: usize,
    tags: Vec<OpTag>,
    profiled: bool,
    /// The lists of the shapes that the pass has run. The list of the shape in `loaded` is in the ring.
    built: HashMap<ShapeKey, Built>,
    loaded: Option<ShapeKey>,
    /// The host time of the last call.
    pub host: HostTime,
}

impl<'d> Pass<'d> {
    /// A pass over `weights` for sequences of up to `seq` positions, blocks of up to `max_rows` rows and `cycles` cycles.
    pub fn new(
        dev: &'d CudaDevice,
        weights: &'d DeviceWeights<'d>,
        seq: usize,
        max_rows: usize,
        cycles: usize,
    ) -> Result<Self, String> {
        Self::with_profile(dev, weights, seq, max_rows, cycles, false)
    }

    /// Like [`Pass::new`]. With `profile` the ring records the time of each task, and the device must be open with the
    /// profile cubin.
    pub fn with_profile(
        dev: &'d CudaDevice,
        weights: &'d DeviceWeights<'d>,
        seq: usize,
        max_rows: usize,
        cycles: usize,
        profile: bool,
    ) -> Result<Self, String> {
        Self::with_options(dev, weights, seq, max_rows, cycles, profile, CacheFormat::F32)
    }

    /// Like [`Pass::new`], with the cache in `format`. The default format of the pass is `CacheFormat::F32`.
    pub fn with_cache_format(
        dev: &'d CudaDevice,
        weights: &'d DeviceWeights<'d>,
        seq: usize,
        max_rows: usize,
        cycles: usize,
        format: CacheFormat,
    ) -> Result<Self, String> {
        Self::with_options(dev, weights, seq, max_rows, cycles, false, format)
    }

    fn with_options(
        dev: &'d CudaDevice,
        weights: &'d DeviceWeights<'d>,
        seq: usize,
        max_rows: usize,
        cycles: usize,
        profile: bool,
        format: CacheFormat,
    ) -> Result<Self, String> {
        let plans = runs(cycles);
        let scratch_layout = ScratchLayout::new(max_rows, seq);
        let ring = Ring::new(dev, scratch_layout.total_floats(), profile)?;
        ring.zero_device()?;
        let extra = (cycles - 1) * CYCLE_LAYERS;
        Ok(Self {
            dev,
            weights,
            slot: Slot::new(dev, seq, extra, format)?,
            cycles,
            max_rows,
            scratch_layout,
            ring,
            tables: Dev::alloc(dev, 128 * TABLES)?,
            arenas: HashMap::new(),
            plans,
            grid: dev.sm_count(),
            launches: Vec::new(),
            proj_block_form: 0,
            tags: Vec::new(),
            profiled: profile,
            built: HashMap::new(),
            loaded: None,
            host: HostTime::default(),
        })
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    fn launch(&mut self, what: &str) -> Result<(), String> {
        let started = Instant::now();
        let result = unsafe {
            self.ring.run_grid(
                self.grid,
                nagaya_mk_cuda::ring::DEFAULT_BLOCK_THREADS,
                self.dev.max_dynamic_shared_bytes()?,
            )
        };
        if let Err(e) = result {
            self.slot.poisoned = true;
            return Err(format!(
                "the launch of {what} failed: {e}. The slot state may be wrong; zero the slot before the next call"
            ));
        }
        self.launches.push(Launch {
            what: what.to_owned(),
            wall: started.elapsed(),
        });
        Ok(())
    }

    /// Drop the rings of all shapes. The next call builds its ring again.
    pub fn forget_rings(&mut self) {
        drop(self.ring.take_list());
        self.built.clear();
        self.loaded = None;
    }

    /// Run the rows `ids` at the next positions of the slot, and return the logits of the last `logit_rows` rows,
    /// `LOGITS` values each. The position of the slot moves by the count of rows.
    pub fn run(&mut self, ids: &[u32], logit_rows: usize) -> Result<Vec<f32>, String> {
        let call = Instant::now();
        let rows = ids.len();
        if self.slot.poisoned {
            return Err("the slot is poisoned by a failed launch. Zero the slot before the next call".to_owned());
        }
        if rows == 0 || rows > self.max_rows {
            return Err(format!(
                "a call takes from 1 to {} rows, and this one has {rows}",
                self.max_rows
            ));
        }
        if logit_rows == 0 || logit_rows > rows.min(MAX_LOGIT_ROWS) {
            return Err(format!(
                "a call of {rows} rows gives logits for 1 to {} rows, and this one asks for {logit_rows}",
                rows.min(MAX_LOGIT_ROWS)
            ));
        }
        let pos0 = self.slot.position;
        if pos0 + rows > self.slot.seq {
            return Err(format!(
                "the slot holds {} positions, and this call ends at position {}. Use a slot of more positions",
                self.slot.seq,
                pos0 + rows
            ));
        }
        if let Some(bad) = ids.iter().find(|i| **i as usize >= crate::spec::VOCAB) {
            return Err(format!(
                "the id {bad} is not in the vocabulary of {} entries",
                crate::spec::VOCAB
            ));
        }
        self.launches.clear();
        self.host = HostTime::default();
        if !self.arenas.contains_key(&(rows, logit_rows)) {
            let arena = Arena::new(self.dev, rows, logit_rows)?;
            self.arenas.insert((rows, logit_rows), arena);
        }
        // The arena leaves the map for the call, so that the pass can borrow itself for each launch.
        let arena = self
            .arenas
            .remove(&(rows, logit_rows))
            .expect("the arena was just made");
        let result = self.run_with(&arena, ids, logit_rows, pos0);
        self.arenas.insert((rows, logit_rows), arena);
        if result.is_ok() {
            self.slot.position += rows;
        }
        if let Some(launch) = self.launches.last() {
            self.host.prep = self.ring.last_prep();
            self.host.call = call.elapsed().saturating_sub(launch.wall) + self.host.prep;
        }
        result
    }

    fn run_with(&mut self, arena: &Arena<'d>, ids: &[u32], logit_rows: usize, pos0: usize) -> Result<Vec<f32>, String> {
        let rows = ids.len();
        arena
            .ids
            .write(0, &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>())?;
        let key = ShapeKey {
            rows,
            cycles: self.cycles,
            logit_rows,
            block: rows > BLOCK_FORM_ROWS_ABOVE,
        };
        let started = Instant::now();
        if self.loaded == Some(key) {
            self.patch(key, rows, pos0)?;
            self.host.patch = started.elapsed();
        } else {
            self.activate(key, arena, pos0)?;
            self.host.build = started.elapsed();
        }
        self.launch("the pass")?;
        let raw = arena
            .buffer
            .read(arena.layout.offset(Buf::Logits), logit_rows * LOGIT_COLUMNS * 4)?;
        let all: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok((0..logit_rows)
            .flat_map(|r| all[r * LOGIT_COLUMNS..r * LOGIT_COLUMNS + LOGITS].to_vec())
            .collect())
    }

    /// Put the list of the shape `key` in the ring: the list that an earlier call built, or a new one. The ring then
    /// holds this shape, and the device holds its task tables.
    fn activate(&mut self, key: ShapeKey, arena: &Arena<'d>, pos0: usize) -> Result<(), String> {
        if let Some(old) = self.loaded.take() {
            let list = self.ring.take_list();
            self.built.get_mut(&old).expect("the loaded shape is in the map").list = list;
        }
        let parked = self.built.contains_key(&key);
        if parked {
            let built = self.built.get_mut(&key).expect("the shape was just found");
            self.ring.put_list(std::mem::take(&mut built.list));
        } else {
            let (built, tasks) = self.build(arena, key.rows, key.logit_rows, pos0)?;
            self.ring.set(&tasks);
            self.built.insert(key, built);
        }
        let built = &self.built[&key];
        self.tables.write(0, &built.tables)?;
        self.tags = built.tags.clone();
        self.proj_block_form = built.proj_block_form;
        self.loaded = Some(key);
        if parked {
            self.patch(key, key.rows, pos0)?;
        }
        Ok(())
    }

    /// Set the fields of the loaded shape that move between calls: the position of the first row in PROJ and ATTN, and
    /// the chunks of ATTN with the count of its items. The ranges of each task stay as they were, so the ring plans
    /// nothing (`Ring::patch`). A task that writes the cache or the scratch declares the whole plane or region.
    fn patch(&mut self, key: ShapeKey, rows: usize, pos0: usize) -> Result<(), String> {
        let built = &self.built[&key];
        for p in &built.patches {
            let mut task = self.ring.task(p.input).clone();
            match p.kind {
                TaskKind::Proj => task.desc.pos0 = pos0 as u32,
                TaskKind::Attn => {
                    let run = &self.slot.runs[self.plans[p.plan].state];
                    let shape = AttnShape::new(rows, pos0, run.window, run.plane_rows, self.slot.format)?;
                    let region = self.scratch_layout.bytes(ScratchKind::Attn);
                    if shape.scratch_floats() * 4 > region {
                        return Err(format!(
                            "the attention of {rows} rows at position {pos0} needs {} bytes of scratch, and its region has {region}. \
                             Make the scratch layout for a longer slot",
                            shape.scratch_floats() * 4
                        ));
                    }
                    task.desc.items = shape.items() as u32;
                    task.desc.group_items = shape.chunks() as u32;
                    task.desc.pos0 = pos0 as u32;
                }
                other => unreachable!("the task kind {other:?} has no field that moves"),
            }
            self.ring.patch(p.input, task);
        }
        Ok(())
    }

    /// Build the tasks of a pass for `rows` rows at `pos0`, with their task tables.
    fn build(
        &self,
        arena: &Arena<'d>,
        rows: usize,
        logit_rows: usize,
        pos0: usize,
    ) -> Result<(Built, Vec<Task>), String> {
        let w = self.weights;
        let mut tables = TableSet::new(self.tables.ptr(), TABLES);
        let mut list: Vec<Task> = Vec::new();
        let mut tags: Vec<OpTag> = Vec::new();
        let mut patches: Vec<Patch> = Vec::new();
        // The embedding.
        {
            let table = tables.add(desc::embed_table(
                w.embed_norm.ptr(),
                arena.addr(Buf::B0),
                arena.rf(Buf::B0),
            )?)?;
            let mut d = desc::embed(
                rows,
                arena.ids.ptr(),
                w.embed.ptr(),
                arena.addr(Buf::B0),
                arena.rf(Buf::B0),
            )?;
            d.d = table;
            list.push(
                Task::with_items(d)
                    .writes(arena.addr(Buf::B0), rows * HIDDEN * 4)
                    .writes(arena.rf(Buf::B0), rows * 4),
            );
            tags.push(OpTag {
                name: "EMBED",
                run: None,
                layer: None,
                weight_bytes: HIDDEN * 2,
            });
        }
        let shape = StepShape { rows, pos0 };
        for (plan_index, plan) in self.plans.iter().enumerate() {
            let env = RunEnv {
                weights: &w.layers[plan.layer],
                slot: &self.slot.runs[plan.state],
                arena,
                scratch: self.ring.scratch(),
                scratch_layout: &self.scratch_layout,
                format: self.slot.format,
                message_queries: if plan.messages {
                    w.message_queries[plan.layer - CYCLE_START].ptr()
                } else {
                    0
                },
            };
            let tasks = run_tasks(plan, &shape, &env, &mut tables)?;
            debug_assert_eq!(
                tasks.iter().map(|(k, _)| *k).collect::<Vec<TaskKind>>(),
                plan.task_kinds()
            );
            for (kind, task) in tasks {
                if matches!(kind, TaskKind::Proj | TaskKind::Attn) {
                    patches.push(Patch {
                        input: list.len(),
                        kind,
                        plan: plan_index,
                    });
                }
                tags.push(OpTag {
                    name: kind_name(kind),
                    run: Some(plan.run),
                    layer: Some(plan.layer),
                    weight_bytes: weight_bytes(kind, plan),
                });
                list.push(task);
            }
        }
        // The output residual and the head.
        let last = self.plans.last().expect("a pass has runs");
        let r0 = rows - logit_rows;
        let stack: Vec<Buf> = {
            let mut s = last.stack_attn.clone();
            // The stack of the output residual: the four block outputs, then the row after the last run.
            s.pop();
            s.push(last.x_out);
            s
        };
        let stack_rows: Vec<(u64, u64)> = stack.iter().map(|b| (arena.addr(*b), arena.rf(*b))).collect();
        let mix_table = tables.add(
            MixTable {
                stack: stack_rows,
                message: Vec::new(),
                queries: w.output_query.ptr(),
                message_queries: 0,
            }
            .build()?,
        )?;
        let f = arena.addr(Buf::F);
        let f_rf = arena.rf(Buf::F);
        let mix_desc = desc::mix_out(rows, stack.len(), f, f_rf, mix_table)?;
        let mut mix_task = Task::with_items(mix_desc);
        for b in &stack {
            mix_task = mix_task
                .reads(arena.addr(*b), rows * HIDDEN * 4)
                .reads(arena.rf(*b), rows * 4);
        }
        list.push(mix_task.writes(f, rows * HIDDEN * 4).writes(f_rf, rows * 4));
        tags.push(OpTag {
            name: "MIX_OUT",
            run: None,
            layer: None,
            weight_bytes: 0,
        });
        let head_table = tables.add(desc::head_table(w.norm_bf16.ptr())?)?;
        let scratch = self.ring.scratch() + self.scratch_layout.offset(ScratchKind::Head) as u64;
        let mut head_desc = desc::head(
            logit_rows,
            f + (r0 * HIDDEN * 4) as u64,
            f_rf + (r0 * 4) as u64,
            w.head.ptr(),
            arena.addr(Buf::Logits),
            scratch,
        )?;
        head_desc.d = head_table;
        list.push(
            Task::with_items(head_desc)
                .reads(f + (r0 * HIDDEN * 4) as u64, logit_rows * HIDDEN * 4)
                .reads(f_rf + (r0 * 4) as u64, logit_rows * 4)
                .writes(arena.addr(Buf::Logits), logit_rows * LOGIT_COLUMNS * 4)
                .writes(scratch, self.scratch_layout.bytes(ScratchKind::Head)),
        );
        tags.push(OpTag {
            name: "HEAD",
            run: None,
            layer: None,
            weight_bytes: LOGIT_COLUMNS * HIDDEN * 2,
        });
        let proj_block_form = list
            .iter()
            .filter(|t| {
                t.desc.opcode == nagaya_mk_cuda::protocol::OP_PROJ
                    && t.desc.flags & nagaya_mk_cuda::protocol::FLAG_BLOCK_FORM != 0
            })
            .count();
        let host: Vec<u8> = tables
            .host()
            .iter()
            .flat_map(|t| t.ptr.iter().flat_map(|p| p.to_le_bytes()))
            .collect();
        let built = Built {
            list: RingList::default(),
            tables: host,
            tags,
            proj_block_form,
            patches,
        };
        Ok((built, list))
    }

    /// The times of each block of each task of the last profiled call. A block that had no item of the task is left out.
    pub fn block_times(&self) -> Result<Vec<TaskTimes>, String> {
        if !self.profiled {
            return Err("the pass has no profile. Make it with Pass::with_profile".to_owned());
        }
        let profile = self.ring.read_profile()?;
        let blocks = self.ring.last_grid() as usize;
        let mut out = Vec::new();
        for (index, planned) in self.ring.planned(self.ring.last_grid()).iter().enumerate() {
            let Some(origin) = planned.origin else { continue };
            let used = (planned.items as usize).min(blocks);
            let slots = &profile[index * blocks * 3..(index + 1) * blocks * 3];
            let times = slots.chunks(3).take(used).map(|s| [s[0], s[1], s[2]]).collect();
            out.push(TaskTimes {
                tag: self.tags[origin].clone(),
                items: planned.items,
                blocks: times,
            });
        }
        Ok(out)
    }

    /// The time of each task in the last call, when the pass was made with a profile. Only the blocks that had an item count.
    pub fn spans(&self) -> Result<Vec<OpSpan>, String> {
        if !self.profiled {
            return Err("the pass has no profile. Make it with Pass::with_profile".to_owned());
        }
        let profile = self.ring.read_profile()?;
        let blocks = self.ring.last_grid() as usize;
        let mut out = Vec::new();
        for (index, planned) in self.ring.planned(self.ring.last_grid()).iter().enumerate() {
            let Some(origin) = planned.origin else { continue };
            let used = (planned.items as usize).min(blocks);
            if used == 0 {
                continue;
            }
            let slots = &profile[index * blocks * 3..(index + 1) * blocks * 3];
            let begin = slots.chunks(3).take(used).map(|s| s[1]).min().expect("a block");
            let end = slots.chunks(3).take(used).map(|s| s[2]).max().expect("a block");
            out.push(OpSpan {
                tag: self.tags[origin].clone(),
                items: planned.items,
                ns: end.saturating_sub(begin),
            });
        }
        Ok(out)
    }
}
