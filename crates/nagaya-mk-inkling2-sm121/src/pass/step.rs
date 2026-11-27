//! A pass of the model on the device: the embedding, the runs, the output residual and the head (`cuda` feature).
//!
//! One launch of one ring runs the whole pass: 1 + the tasks of the runs + 2 tasks. The DOWN op of each run writes the
//! row factors of the row that it wrote, so no task waits for the host. The waits come from the ranges that each task
//! declares (`nagaya_mk_cuda::graph`).
//!
//! One thread owns a pass. A pass holds several slots. A call names its slot, its count of cycles and the rows that give
//! logits. A launch that fails poisons the slot of the call, and the pass refuses the next call on that slot until the
//! slot is cleared. The pass allocates nothing in a call: the arena, the scratch and the slots exist from the open.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nagaya_mk_cuda::attn::{AttnShape, CacheFormat};
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::ring::{Ring, RingList, Task};

use super::arena::Arena;
use super::desc::{self, MixTable};
use super::dev::Dev;
use super::layout::{Buf, LOGIT_COLUMNS, MAX_CYCLES, RunPlan, ScratchKind, ScratchLayout, TaskKind, runs};
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
        TaskKind::OutMatmul => "OUT_MATMUL",
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
fn weight_bytes(kind: TaskKind, plan: &RunPlan, rows: usize) -> usize {
    let nvfp4 = super::weights::expert_format(plan.layer) == nagaya_mk_cuda::experts::ExpertFormat::Nvfp4;
    // The bytes of one expert of 2,304 rows and `k` values: the codes and the scales.
    let expert = |k: usize| {
        let (codes, scales) = if nvfp4 { (k / 2, k / 16) } else { (k, k / 32) };
        HIDDEN * (codes + scales)
    };
    match kind {
        TaskKind::Mix | TaskKind::Attn | TaskKind::Quant => 0,
        TaskKind::OutMatmul => HIDDEN * 3072 * 2,
        TaskKind::GateDown3 => 3 * 64 * 2 * HIDDEN * 2,
        TaskKind::GateUp3 => 3 * 2 * HIDDEN * 64 * 2,
        TaskKind::GateDownE => 64 * 2 * HIDDEN * 2,
        TaskKind::GateUpE => 2 * HIDDEN * 64 * 2,
        TaskKind::Proj => super::desc::proj_layout(plan.full).rows() * HIDDEN * 2,
        // The split form of a big block has its matmul in `OutMatmul`: this task reads no weights.
        TaskKind::OutMix if rows >= nagaya_mk_cuda::outmix::SPLIT_MIN_ROWS => 0,
        TaskKind::OutMix => HIDDEN * 3072 * 2,
        TaskKind::Route => 128 * HIDDEN * 2,
        TaskKind::Up => 8 * expert(HIDDEN),
        TaskKind::Down => 8 * expert(crate::spec::INTERMEDIATE),
    }
}

/// The lists that the pass keeps. A list holds a few hundred tasks, so the bound costs little host memory. A call that
/// needs a list that the bound dropped builds it again.
pub const MAX_BUILT: usize = 64;

/// What makes two calls share a ring: the arena and the tasks differ only in the fields that [`Pass::patch`] sets.
/// The tasks hold the addresses of the planes of their slot, so the slot is part of the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ShapeKey {
    slot: usize,
    rows: usize,
    cycles: usize,
    /// The first row that gives logits, and the count of rows from there (0 for no logits).
    logit_first: usize,
    logit_span: usize,
    /// The block form of PROJ (more than `BLOCK_FORM_ROWS_ABOVE` rows) has other descriptors and other scratch.
    block: bool,
}

/// The first row and the count of rows of the span that holds `logit_rows`, a list of rows below `rows`.
fn span_of(logit_rows: &[usize], rows: usize) -> Result<(usize, usize), String> {
    let Some(first) = logit_rows.first() else {
        return Ok((0, 0));
    };
    if let Some(bad) = logit_rows.iter().find(|r| **r >= rows) {
        return Err(format!(
            "a call of {rows} rows has no row {bad} for logits. Name rows below {rows}"
        ));
    }
    if let Some(pair) = logit_rows.windows(2).find(|w| w[0] >= w[1]) {
        return Err(format!(
            "the rows for logits must rise one by one, and {} comes before {}. Sort the rows and drop a repeat",
            pair[0], pair[1]
        ));
    }
    let last = logit_rows[logit_rows.len() - 1];
    let span = last - first + 1;
    if span > MAX_LOGIT_ROWS {
        return Err(format!(
            "the rows for logits lie in {span} rows, from {first} to {last}. They must lie inside {MAX_LOGIT_ROWS} rows"
        ));
    }
    Ok((*first, span))
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
    /// The byte offset of the logits in the arena (0 when the call gives none).
    logits_at: usize,
    /// The count of the call that last used this list.
    used: u64,
}

/// What a pass holds, from the open.
#[derive(Clone, Debug)]
pub struct PassConfig {
    /// The positions of each slot. The slots differ in length; the scratch serves the longest.
    pub seqs: Vec<usize>,
    /// The most rows of a call.
    pub max_rows: usize,
    /// The most cycles of a call. Each slot keeps the reserve of the runs of this count.
    pub max_cycles: usize,
    /// Record the time of each task. The device must be open with the profile cubin.
    pub profile: bool,
    pub format: CacheFormat,
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
    pub slots: Vec<Slot<'d>>,
    max_cycles: usize,
    max_rows: usize,
    scratch_layout: ScratchLayout,
    ring: Ring<'d>,
    tables: Dev<'d>,
    arena: Arena<'d>,
    /// The runs for 1 to `max_cycles` cycles: `plans[c - 1]` for `c` cycles.
    plans: Vec<Vec<RunPlan>>,
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
    /// The count of calls, for the order of use of the lists.
    clock: u64,
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
        Self::open(
            dev,
            weights,
            &PassConfig {
                seqs: vec![seq],
                max_rows,
                max_cycles: cycles,
                profile,
                format,
            },
        )
    }

    /// A pass with one slot for each length in `config.seqs`.
    pub fn open(dev: &'d CudaDevice, weights: &'d DeviceWeights<'d>, config: &PassConfig) -> Result<Self, String> {
        let PassConfig {
            seqs,
            max_rows,
            max_cycles: cycles,
            profile,
            format,
        } = config.clone();
        if seqs.is_empty() || seqs.contains(&0) {
            return Err("a pass needs at least 1 slot, and each slot needs at least 1 position".to_owned());
        }
        if !(1..=MAX_CYCLES).contains(&cycles) {
            return Err(format!(
                "a pass runs from 1 to {MAX_CYCLES} cycles, and the call names {cycles}. Name a count in that range"
            ));
        }
        if !(1..=nagaya_mk_cuda::attn::MAX_ROWS).contains(&max_rows) {
            return Err(format!(
                "a pass takes from 1 to {} rows in a call, and the open names {max_rows}",
                nagaya_mk_cuda::attn::MAX_ROWS
            ));
        }
        let plans: Vec<Vec<RunPlan>> = (1..=cycles).map(runs).collect();
        let longest = *seqs.iter().max().expect("a slot");
        let scratch_layout = ScratchLayout::new(max_rows, longest);
        let ring = Ring::new(dev, scratch_layout.total_floats(), profile)?;
        ring.zero_device()?;
        let extra = (cycles - 1) * CYCLE_LAYERS;
        let slots = seqs
            .iter()
            .map(|seq| Slot::new(dev, *seq, extra, format))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            dev,
            weights,
            slots,
            max_cycles: cycles,
            max_rows,
            scratch_layout,
            ring,
            tables: Dev::alloc(dev, 128 * TABLES)?,
            arena: Arena::with_capacity(dev, max_rows, MAX_LOGIT_ROWS)?,
            plans,
            grid: dev.sm_count(),
            launches: Vec::new(),
            proj_block_form: 0,
            tags: Vec::new(),
            profiled: profile,
            built: HashMap::new(),
            loaded: None,
            clock: 0,
            host: HostTime::default(),
        })
    }

    /// The most cycles of a call, and the count of the calls of [`Pass::run`].
    pub fn cycles(&self) -> usize {
        self.max_cycles
    }

    fn launch(&mut self, what: &str, slot: usize) -> Result<(), String> {
        let started = Instant::now();
        let result = unsafe {
            self.ring.run_grid(
                self.grid,
                nagaya_mk_cuda::ring::DEFAULT_BLOCK_THREADS,
                self.dev.max_dynamic_shared_bytes()?,
            )
        };
        if let Err(e) = result {
            self.slots[slot].poisoned = true;
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

    /// Run the rows `ids` on slot 0 at its default count of cycles, and return the logits of the last `logit_rows` rows,
    /// `LOGITS` values each. The position of the slot moves by the count of rows.
    pub fn run(&mut self, ids: &[u32], logit_rows: usize) -> Result<Vec<f32>, String> {
        let rows = ids.len();
        if logit_rows == 0 || logit_rows > rows.min(MAX_LOGIT_ROWS) {
            return Err(format!(
                "a call of {rows} rows gives logits for 1 to {} rows, and this one asks for {logit_rows}",
                rows.min(MAX_LOGIT_ROWS)
            ));
        }
        let wanted: Vec<usize> = (rows - logit_rows..rows).collect();
        self.run_slot(0, self.max_cycles, ids, &wanted)
    }

    /// Run the rows `ids` at the next positions of slot `slot`, with `cycles` cycles (1 to the most of the open).
    ///
    /// `logit_rows` names the rows that give logits: each one is below the count of rows, they rise one by one, and they
    /// lie inside a span of at most `MAX_LOGIT_ROWS` rows. An empty list runs no MIX_OUT and no HEAD, and returns no
    /// logits. The result holds `LOGITS` values for each named row, in the order of the names. The position of the slot
    /// moves by the count of rows. A refused call changes nothing.
    pub fn run_slot(
        &mut self,
        slot: usize,
        cycles: usize,
        ids: &[u32],
        logit_rows: &[usize],
    ) -> Result<Vec<f32>, String> {
        let call = Instant::now();
        let rows = ids.len();
        let Some(state) = self.slots.get(slot) else {
            return Err(format!(
                "the pass has {} slots, and the call names slot {slot}. Name a slot below {}",
                self.slots.len(),
                self.slots.len()
            ));
        };
        if state.poisoned {
            return Err(format!(
                "slot {slot} is poisoned by a failed launch. Clear the slot before the next call"
            ));
        }
        if !(1..=self.max_cycles).contains(&cycles) {
            return Err(format!(
                "a call runs from 1 to {} cycles, and this one names {cycles}",
                self.max_cycles
            ));
        }
        if rows == 0 || rows > self.max_rows {
            return Err(format!(
                "a call takes from 1 to {} rows, and this one has {rows}",
                self.max_rows
            ));
        }
        let (logit_first, logit_span) = span_of(logit_rows, rows)?;
        let pos0 = state.position;
        if pos0 + rows > state.seq {
            return Err(format!(
                "slot {slot} holds {} positions, and this call ends at position {}. Use a slot of more positions",
                state.seq,
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
        let result = self.run_with(slot, cycles, ids, logit_first, logit_span, pos0);
        if result.is_ok() {
            self.slots[slot].position += rows;
        }
        if let Some(launch) = self.launches.last() {
            self.host.prep = self.ring.last_prep();
            self.host.call = call.elapsed().saturating_sub(launch.wall) + self.host.prep;
        }
        let all = result?;
        let mut out = Vec::with_capacity(logit_rows.len() * LOGITS);
        for r in logit_rows {
            let at = (r - logit_first) * LOGIT_COLUMNS;
            out.extend_from_slice(&all[at..at + LOGITS]);
        }
        Ok(out)
    }

    /// Run one call, and return the logits of the span, `LOGIT_COLUMNS` values for each row of it.
    fn run_with(
        &mut self,
        slot: usize,
        cycles: usize,
        ids: &[u32],
        logit_first: usize,
        logit_span: usize,
        pos0: usize,
    ) -> Result<Vec<f32>, String> {
        let rows = ids.len();
        self.arena
            .ids
            .write(0, &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>())?;
        let key = ShapeKey {
            slot,
            rows,
            cycles,
            logit_first,
            logit_span,
            block: rows > BLOCK_FORM_ROWS_ABOVE,
        };
        let started = Instant::now();
        if self.loaded == Some(key) {
            self.patch(key, rows, pos0)?;
            self.host.patch = started.elapsed();
        } else {
            self.activate(key, pos0)?;
            self.host.build = started.elapsed();
        }
        self.clock += 1;
        self.built.get_mut(&key).expect("the shape is loaded").used = self.clock;
        self.launch("the pass", slot)?;
        if logit_span == 0 {
            return Ok(Vec::new());
        }
        let at = self.built[&key].logits_at;
        let raw = self.arena.buffer.read(at, logit_span * LOGIT_COLUMNS * 4)?;
        Ok(raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    /// Put the list of the shape `key` in the ring: the list that an earlier call built, or a new one. The ring then
    /// holds this shape, and the device holds its task tables. The pass keeps at most [`MAX_BUILT`] lists.
    fn activate(&mut self, key: ShapeKey, pos0: usize) -> Result<(), String> {
        if let Some(old) = self.loaded.take() {
            let list = self.ring.take_list();
            self.built.get_mut(&old).expect("the loaded shape is in the map").list = list;
        }
        let parked = self.built.contains_key(&key);
        if parked {
            let built = self.built.get_mut(&key).expect("the shape was just found");
            self.ring.put_list(std::mem::take(&mut built.list));
        } else {
            let arena = self.arena.view(key.rows, key.logit_span)?;
            let (built, tasks) = self.build(&arena, &key, pos0)?;
            if self.built.len() >= MAX_BUILT {
                let oldest = self
                    .built
                    .iter()
                    .min_by_key(|(_, b)| b.used)
                    .map(|(k, _)| *k)
                    .expect("the map is full");
                self.built.remove(&oldest);
            }
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
        let slot = &self.slots[key.slot];
        for p in &built.patches {
            let mut task = self.ring.task(p.input).clone();
            match p.kind {
                TaskKind::Proj => task.desc.pos0 = pos0 as u32,
                TaskKind::Attn => {
                    let run = &slot.runs[self.plans[key.cycles - 1][p.plan].state];
                    let shape = AttnShape::new(rows, pos0, run.window, run.plane_rows, slot.format)?;
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
    fn build(&self, arena: &Arena<'d>, key: &ShapeKey, pos0: usize) -> Result<(Built, Vec<Task>), String> {
        let rows = key.rows;
        let plans = &self.plans[key.cycles - 1];
        let slot = &self.slots[key.slot];
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
        for (plan_index, plan) in plans.iter().enumerate() {
            let env = RunEnv {
                weights: &w.layers[plan.layer],
                slot: &slot.runs[plan.state],
                arena,
                scratch: self.ring.scratch(),
                scratch_layout: &self.scratch_layout,
                format: slot.format,
                message_queries: if plan.messages {
                    w.message_queries[plan.layer - CYCLE_START].ptr()
                } else {
                    0
                },
            };
            let tasks = run_tasks(plan, &shape, &env, &mut tables)?;
            // The order of the list orders the tasks (a task waits on the tasks before it), so a list that differs from the
            // plan's, in the place of the matmul task of OUT_MIX too, is an error and not a wrong result.
            let kinds = tasks.iter().map(|(k, _)| *k).collect::<Vec<TaskKind>>();
            if kinds != plan.task_kinds_for(rows) {
                return Err(format!(
                    "the tasks built for run {} ({rows} rows) are {kinds:?}, and the plan lists {:?}. Fix `run_tasks`",
                    plan.run,
                    plan.task_kinds_for(rows)
                ));
            }
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
                    weight_bytes: weight_bytes(kind, plan, rows),
                });
                list.push(task);
            }
        }
        // The output residual and the head. A call that asks for no logits ends after the last run.
        if key.logit_span > 0 {
            let last = plans.last().expect("a pass has runs");
            let r0 = key.logit_first;
            let logit_rows = key.logit_span;
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
        }
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
            logits_at: if key.logit_span > 0 {
                arena.layout.offset(Buf::Logits)
            } else {
                0
            },
            used: 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_logit_rows_make_an_empty_span() {
        assert_eq!(span_of(&[], 5).unwrap(), (0, 0));
    }

    #[test]
    fn a_set_inside_16_rows_gives_its_first_row_and_span() {
        assert_eq!(span_of(&[3], 9).unwrap(), (3, 1));
        assert_eq!(span_of(&[1, 4, 16], 17).unwrap(), (1, 16));
        assert_eq!(span_of(&[0, 15], 16).unwrap(), (0, 16));
    }

    #[test]
    fn a_wide_unsorted_repeated_or_outside_set_is_refused_with_a_message() {
        assert!(span_of(&[0, 16], 17).unwrap_err().contains("lie in 17 rows"));
        assert!(span_of(&[2, 2], 5).unwrap_err().contains("rise one by one"));
        assert!(span_of(&[3, 1], 5).unwrap_err().contains("rise one by one"));
        assert!(span_of(&[5], 5).unwrap_err().contains("has no row 5"));
    }
}
