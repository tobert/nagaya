//! A pass of the model on the device: the embedding, the runs, the output residual and the head (`cuda` feature).
//!
//! The pass launches the ring once for the embedding, once for each run, and once for the output residual and the head.
//! Between two runs the host computes the row factors of the row that the run wrote, because the DOWN op does not give
//! them yet (the rule "the producer gives the row factor" needs the second-level finisher of lane M2). When that op
//! lands, the pass becomes one ring, and only `Pass::run` changes.
//!
//! One thread owns a pass. A launch that fails poisons the slot, and the pass refuses the next call until the slot is zeroed.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::ring::{Ring, Task};

use super::arena::Arena;
use super::desc::{self, MixTable};
use super::dev::Dev;
use super::layout::{Buf, LOGIT_COLUMNS, RunPlan, ScratchKind, ScratchLayout, TaskKind, runs};
use super::run::{RunEnv, StepShape, TableSet, run_tasks};
use super::slot::Slot;
use super::weights::DeviceWeights;
use crate::spec::{CYCLE_LAYERS, CYCLE_START, HIDDEN, LOGITS, MAX_LOGIT_ROWS};

/// The tables that a ring can hold: the tasks of a run, or of the embedding, or of the head. A run has at most 12.
const TABLES: usize = 64;

/// The time of a launch.
#[derive(Clone, Debug)]
pub struct Launch {
    pub what: String,
    pub wall: Duration,
}

/// The rows of the stream to the row factor, as the host form computes it: `1 / sqrt(mean(x^2) + eps)`.
pub fn row_factor(x: &[f32]) -> f32 {
    let sum: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    1.0 / ((sum / x.len() as f64) as f32 + 1e-6).sqrt()
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
        let plans = runs(cycles);
        let scratch_layout = ScratchLayout::new(max_rows, seq);
        let ring = Ring::new(dev, scratch_layout.total_floats(), false)?;
        ring.zero_device()?;
        let extra = (cycles - 1) * CYCLE_LAYERS;
        Ok(Self {
            dev,
            weights,
            slot: Slot::new(dev, seq, extra, CacheFormat::F32)?,
            cycles,
            max_rows,
            scratch_layout,
            ring,
            tables: Dev::alloc(dev, 128 * TABLES)?,
            arenas: HashMap::new(),
            plans,
            grid: dev.sm_count(),
            launches: Vec::new(),
        })
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    fn launch(&mut self, what: &str, tasks: &[Task], tables: &TableSet) -> Result<(), String> {
        let host: Vec<u8> = tables
            .host()
            .iter()
            .flat_map(|t| t.ptr.iter().flat_map(|p| p.to_le_bytes()))
            .collect();
        self.tables.write(0, &host)?;
        self.ring.set(tasks);
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

    /// Run the rows `ids` at the next positions of the slot, and return the logits of the last `logit_rows` rows,
    /// `LOGITS` values each. The position of the slot moves by the count of rows.
    pub fn run(&mut self, ids: &[u32], logit_rows: usize) -> Result<Vec<f32>, String> {
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
        self.zero_counter_heads()?;
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
        result
    }

    fn run_with(&mut self, arena: &Arena<'d>, ids: &[u32], logit_rows: usize, pos0: usize) -> Result<Vec<f32>, String> {
        let rows = ids.len();
        let w = self.weights;
        arena
            .ids
            .write(0, &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>())?;
        // The embedding.
        {
            let mut tables = TableSet::new(self.tables.ptr(), TABLES);
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
            let task = Task::with_items(d)
                .writes(arena.addr(Buf::B0), rows * HIDDEN * 4)
                .writes(arena.rf(Buf::B0), rows * 4);
            self.launch("embed", &[task], &tables)?;
        }
        let shape = StepShape { rows, pos0 };
        let plans = self.plans.clone();
        for plan in &plans {
            let mut tables = TableSet::new(self.tables.ptr(), TABLES);
            let tasks = {
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
                run_tasks(plan, &shape, &env, &mut tables)?
            };
            let list: Vec<Task> = tasks.iter().map(|(_, t)| t.clone()).collect();
            debug_assert_eq!(
                tasks.iter().map(|(k, _)| *k).collect::<Vec<TaskKind>>(),
                plan.task_kinds()
            );
            self.launch(&format!("run {} (layer {})", plan.run, plan.layer), &list, &tables)?;
            self.set_row_factors(arena, plan.x_out, rows)?;
        }
        // The output residual and the head.
        let last = self.plans.last().expect("a pass has runs").clone();
        let r0 = rows - logit_rows;
        let mut tables = TableSet::new(self.tables.ptr(), TABLES);
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
        let mix_task = mix_task.writes(f, rows * HIDDEN * 4).writes(f_rf, rows * 4);
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
        let head_task = Task::with_items(head_desc)
            .reads(f + (r0 * HIDDEN * 4) as u64, logit_rows * HIDDEN * 4)
            .reads(f_rf + (r0 * 4) as u64, logit_rows * 4)
            .writes(arena.addr(Buf::Logits), logit_rows * LOGIT_COLUMNS * 4)
            .writes(scratch, self.scratch_layout.bytes(ScratchKind::Head));
        self.launch("output residual and head", &[mix_task, head_task], &tables)?;
        let raw = arena
            .buffer
            .read(arena.layout.offset(Buf::Logits), logit_rows * LOGIT_COLUMNS * 4)?;
        let all: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok((0..logit_rows)
            .flat_map(|r| all[r * LOGIT_COLUMNS..r * LOGIT_COLUMNS + LOGITS].to_vec())
            .collect())
    }

    /// Zero the counters of the scratch of the attention and of the router before a call.
    ///
    /// The ring is zero once, and each merger resets its counter, so a counter should stay zero. These two ops break
    /// the rule across calls of different row counts: the counters of the attention are `8 * tiles` words, the
    /// partials follow them (`ops_attn.cuh` `counter_ptr` and `partials_ptr`, and `AttnShape::scratch_floats`), and the
    /// counters of the router are `9 * tiles + 1` words, with the logits after them (`route_scratch_words`). A call
    /// with fewer tiles writes partials or logits where a call with more tiles has counters. Then the second call
    /// starts with a counter that is not zero, merges too early, and gives other bits. The heads are small, so the
    /// pass zeroes them for the most tiles that a call can have. A fix of the two layouts (counters at a place that
    /// does not depend on the rows) makes this call unneeded.
    fn zero_counter_heads(&self) -> Result<(), String> {
        let attn = self.ring.scratch() + self.scratch_layout.offset(ScratchKind::Attn) as u64;
        let attn_bytes = (4 * 8 * 3 + 16).min(self.scratch_layout.bytes(ScratchKind::Attn));
        self.dev.memset_zero(attn, attn_bytes)?;
        let route = self.ring.scratch() + self.scratch_layout.offset(ScratchKind::Route) as u64;
        let tiles = self.max_rows.div_ceil(16);
        let route_bytes = 4 * (tiles * 9 + 1).next_multiple_of(4);
        self.dev
            .memset_zero(route, route_bytes.min(self.scratch_layout.bytes(ScratchKind::Route)))
    }

    /// Compute the row factors of the rows of `buf` on the host and write them.
    fn set_row_factors(&self, arena: &Arena<'d>, buf: Buf, rows: usize) -> Result<(), String> {
        let bytes = arena.buffer.read(arena.layout.offset(buf), rows * HIDDEN * 4)?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let rf: Vec<u8> = values
            .chunks(HIDDEN)
            .flat_map(|row| row_factor(row).to_le_bytes())
            .collect();
        arena.buffer.write(arena.layout.rf(buf), &rf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_row_factor_is_the_inverse_root_of_the_mean_square_plus_epsilon() {
        let ones = vec![1.0f32; 2304];
        assert_eq!(row_factor(&ones), (1.0f64 / (1.0f64 + 1e-6).sqrt()) as f32);
        let twos = vec![-2.0f32; 2304];
        assert_eq!(row_factor(&twos), (1.0f64 / (4.0f64 + 1e-6).sqrt()) as f32);
        assert!(row_factor(&vec![0.0f32; 2304]).is_finite());
    }
}
