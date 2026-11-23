//! A persistent task ring on one CUDA GPU, with its scoreboard, stats, scratch, and optional profile buffer.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::driver::{CUdeviceptr, CudaDevice};
use crate::graph::{self, Node, Planned, Span};
use crate::protocol::{
    BlockStats, CLAIM_BASE, MAX_TASKS_PER_CARD, MkTaskDescriptor, NO_EVENT, OP_HALT, PRODUCER_BLOCK_THREADS,
    PROFILE_WORDS, RING_DEV_BYTES, SCOREBOARD_WORDS,
};

/// Default thread count per block for the persistent interpreter: the producer shape, with a producer warp and eight
/// consumer warps.
pub const DEFAULT_BLOCK_THREADS: u32 = PRODUCER_BLOCK_THREADS;

/// A task of the ring: its descriptor, and what the host builder needs to order it (see [`crate::graph`]).
///
/// The builder sets the waits, the signal and, for a task that runs once in each block, the count of items. Any
/// value that the descriptor holds in those fields is replaced.
#[derive(Clone, Debug)]
pub struct Task {
    pub desc: MkTaskDescriptor,
    pub reads: Vec<Span>,
    pub writes: Vec<Span>,
    /// Indices of earlier tasks that this task follows, whatever the ranges say.
    pub after: Vec<usize>,
    /// The task runs one item in each block, so its count of items is the grid width of the launch.
    pub per_block: bool,
}

impl Task {
    /// A task with no declared ranges that runs one item in each block.
    pub fn per_block(desc: MkTaskDescriptor) -> Self {
        Self {
            desc,
            reads: Vec::new(),
            writes: Vec::new(),
            after: Vec::new(),
            per_block: true,
        }
    }

    /// A task with the count of items that `desc.items` gives.
    pub fn with_items(desc: MkTaskDescriptor) -> Self {
        Self {
            per_block: false,
            ..Self::per_block(desc)
        }
    }

    /// Declare that the task reads `bytes` bytes at the device address `at`.
    pub fn reads(mut self, at: CUdeviceptr, bytes: usize) -> Self {
        self.reads.push(Span::new(at, bytes as u64));
        self
    }

    /// Declare that the task writes `bytes` bytes at the device address `at`.
    pub fn writes(mut self, at: CUdeviceptr, bytes: usize) -> Self {
        self.writes.push(Span::new(at, bytes as u64));
        self
    }

    /// Declare that the task follows the task with index `index` in the list.
    pub fn after(mut self, index: usize) -> Self {
        self.after.push(index);
        self
    }
}

/// Tasks that run one after the other, each one after the whole task before it. A timing probe needs this order.
pub fn chained(descs: &[MkTaskDescriptor]) -> Vec<Task> {
    descs
        .iter()
        .enumerate()
        .map(|(i, desc)| {
            let task = Task::per_block(*desc);
            if i == 0 { task } else { task.after(i - 1) }
        })
        .collect()
}

/// The list of a ring: its tasks and their plan. A caller can park the list of one shape of launch and load it again
/// with no new plan ([`Ring::take_list`], [`Ring::put_list`]).
#[derive(Default)]
pub struct RingList {
    tasks: Vec<Task>,
    plan: Vec<Planned>,
    len: usize,
}

/// One task ring allocated on a [`CudaDevice`]. The ring borrows the device, so the device outlives it.
///
/// `Drop` frees each buffer and prints each failure to stderr. Use [`Ring::free`] to read the failures.
pub struct Ring<'d> {
    dev: &'d CudaDevice,
    host_tasks: *mut MkTaskDescriptor,
    dev_tasks: CUdeviceptr,
    scoreboard: CUdeviceptr,
    dev_stats: CUdeviceptr,
    scratch: CUdeviceptr,
    scratch_floats: usize,
    profile: CUdeviceptr,
    /// The most blocks that a launch can use: the buffers of the stats and the profile have room for them.
    grid_blocks: u32,
    /// The width of the last launch. The stats and the profile of that launch have this many blocks.
    last_grid: AtomicU32,
    tasks: Vec<Task>,
    plan: Vec<Planned>,
    len: usize,
    /// The nanoseconds that the last launch spent on the host before it started the kernel: the descriptors and their copy.
    last_prep_ns: AtomicU64,
}

impl<'d> Ring<'d> {
    /// Allocate a ring on `dev` with `scratch_floats` floats of scratch and optional `%globaltimer` profile buffer.
    ///
    /// If an allocation fails, `Drop` frees the buffers that the ring already holds.
    pub fn new(dev: &'d CudaDevice, scratch_floats: usize, with_profile: bool) -> Result<Self, String> {
        let grid_blocks = dev.sm_count();
        let scratch_floats = scratch_floats.max(1);
        let scratch_bytes = scratch_floats.checked_mul(4).ok_or_else(|| {
            format!("a scratch of {scratch_floats} floats overflows the byte count. Ask for fewer floats")
        })?;
        let mut ring = Self {
            dev,
            host_tasks: std::ptr::null_mut(),
            dev_tasks: 0,
            scoreboard: 0,
            dev_stats: 0,
            scratch: 0,
            scratch_floats,
            profile: 0,
            grid_blocks,
            last_grid: AtomicU32::new(grid_blocks),
            tasks: Vec::new(),
            plan: Vec::new(),
            len: 0,
            last_prep_ns: AtomicU64::new(0),
        };
        ring.host_tasks = dev.alloc_host_pinned(RING_DEV_BYTES)?.cast::<MkTaskDescriptor>();
        ring.dev_tasks = dev.alloc_device(RING_DEV_BYTES)?;
        ring.scoreboard = dev.alloc_device(SCOREBOARD_WORDS * 4)?;
        ring.dev_stats = dev.alloc_device(grid_blocks as usize * std::mem::size_of::<BlockStats>())?;
        ring.scratch = dev.alloc_device(scratch_bytes)?;
        if with_profile {
            let bytes = MAX_TASKS_PER_CARD * grid_blocks as usize * PROFILE_WORDS * 8;
            ring.profile = dev.alloc_device(bytes)?;
        }
        Ok(ring)
    }

    /// Load `tasks` into the ring. The launch plans the dependencies, because the count of items of a per-block task
    /// is the width of the launch.
    ///
    /// # Panics
    /// The ring holds fewer than `MAX_TASKS_PER_CARD` tasks. The plan panics for a wait that does not point backward.
    pub fn set(&mut self, tasks: &[Task]) {
        assert!(
            tasks.len() < MAX_TASKS_PER_CARD,
            "a ring holds at most {} tasks, and this one has {}",
            MAX_TASKS_PER_CARD - 1,
            tasks.len()
        );
        self.tasks = tasks.to_vec();
        // Plan once now with a grid of 1, so that a bad graph panics here and not at the launch.
        self.plan = self.plan_for(1);
        assert!(
            self.plan.len() < MAX_TASKS_PER_CARD,
            "the graph needs {} tasks with its JOIN tasks, and a ring holds at most {}. Fuse some tasks",
            self.plan.len(),
            MAX_TASKS_PER_CARD - 1
        );
        self.len = self.plan.len();
    }

    /// Take the list out of the ring and leave it empty. A ring with an empty list cannot launch.
    pub fn take_list(&mut self) -> RingList {
        RingList {
            tasks: std::mem::take(&mut self.tasks),
            plan: std::mem::take(&mut self.plan),
            len: std::mem::take(&mut self.len),
        }
    }

    /// Load a list that [`Ring::take_list`] returned. The ring has the plan of that list, and it plans nothing.
    pub fn put_list(&mut self, list: RingList) {
        self.tasks = list.tasks;
        self.plan = list.plan;
        self.len = list.len;
    }

    /// The task number `input` of the list of [`Ring::set`].
    ///
    /// # Panics
    /// The number is not in the list.
    pub fn task(&self, input: usize) -> &Task {
        &self.tasks[input]
    }

    /// The time that the last launch spent on the host before the kernel started.
    pub fn last_prep(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.last_prep_ns.load(Ordering::SeqCst))
    }

    /// Replace the task number `input` of the list of [`Ring::set`] with `task`, which has the same ranges, the same
    /// order and the same kind of count of items. The descriptor may differ in any field, and so the count of items
    /// and the addresses of a later pass can change with no new plan. The waits of the tasks after it follow the new
    /// count.
    ///
    /// # Panics
    /// The number is not in the list, or the task has other ranges, another `after` list or another `per_block` than
    /// the task that it replaces: the plan of the waits would change, and the ring needs a new [`Ring::set`].
    pub fn patch(&mut self, input: usize, task: Task) {
        assert!(
            input < self.tasks.len(),
            "the ring holds {} tasks, and there is no task number {input} to patch",
            self.tasks.len()
        );
        let old = &self.tasks[input];
        assert!(
            old.reads == task.reads
                && old.writes == task.writes
                && old.after == task.after
                && old.per_block == task.per_block,
            "the patch of task {input} changes its ranges, its order or its kind of count. Call set again with the new list"
        );
        let index = self.ring_index(input);
        self.plan[index].items = task.desc.items;
        self.tasks[input] = task;
    }

    fn plan_for(&self, grid: u32) -> Vec<Planned> {
        let nodes: Vec<Node> = self
            .tasks
            .iter()
            .map(|t| Node {
                items: if t.per_block { grid } else { t.desc.items },
                reads: t.reads.clone(),
                writes: t.writes.clone(),
                after: t.after.clone(),
            })
            .collect();
        graph::plan(&nodes)
    }

    /// The tasks of the ring as planned for a launch of `grid` blocks, with the JOIN tasks that the builder inserted.
    /// A planned task has the index of its input in `origin`.
    pub fn planned(&self, grid: u32) -> Vec<Planned> {
        let items: Vec<u32> = (0..self.plan.len()).map(|index| self.items_at(index, grid)).collect();
        let mut plan = self.plan.clone();
        for (task, &count) in plan.iter_mut().zip(&items) {
            task.items = count;
            for wait in &mut task.waits {
                wait.target = graph::signals(items[wait.event as usize]);
            }
        }
        plan
    }

    /// The count of items of the planned task `index` in a launch of `grid` blocks.
    fn items_at(&self, index: usize, grid: u32) -> u32 {
        match self.plan[index].origin.map(|i| &self.tasks[i]) {
            Some(task) if task.per_block => grid,
            _ => self.plan[index].items,
        }
    }

    /// The index in the ring of the task that was number `input` in the list of [`Ring::set`].
    pub fn ring_index(&self, input: usize) -> usize {
        self.plan
            .iter()
            .position(|p| p.origin == Some(input))
            .unwrap_or_else(|| panic!("the ring holds no task number {input}"))
    }

    /// The descriptors of a launch of `grid` blocks: the planned tasks, then `OP_HALT`.
    fn descriptors(&self, grid: u32) -> Vec<MkTaskDescriptor> {
        let plan = self.planned(grid);
        graph::assert_waits_backward(&plan);
        let mut out: Vec<MkTaskDescriptor> = Vec::with_capacity(plan.len() + 1);
        for (index, planned) in plan.iter().enumerate() {
            let mut desc = planned.origin.map(|i| self.tasks[i].desc).unwrap_or_default();
            desc.items = planned.items;
            desc.signal = index as u32;
            (desc.wait0_event, desc.wait0_target) =
                planned.waits.first().map_or((NO_EVENT, 0), |w| (w.event, w.target));
            (desc.wait1_event, desc.wait1_target) = planned.waits.get(1).map_or((NO_EVENT, 0), |w| (w.event, w.target));
            out.push(desc);
        }
        out.push(MkTaskDescriptor {
            opcode: OP_HALT,
            ..Default::default()
        });
        out
    }

    /// Zero the scoreboard (the events and the claim words) and scratch buffers.
    pub fn zero_device(&self) -> Result<(), String> {
        let dev = self.dev;
        dev.memset_zero(self.scoreboard, SCOREBOARD_WORDS * 4)?;
        dev.memset_zero(self.scratch, self.scratch_floats * 4)
    }

    /// Upload the descriptors, zero the scoreboard, and run the ring with one block for each SM, in the producer shape,
    /// with the most dynamic shared memory that a block can have.
    ///
    /// # Safety
    /// Every device pointer referenced inside the descriptors must be valid for its op.
    pub unsafe fn run(&self) -> Result<(), String> {
        unsafe { self.run_width(self.grid_blocks) }
    }

    /// Like [`Ring::run`], with a grid of `grid` blocks.
    ///
    /// # Safety
    /// The same rules as [`Ring::run`].
    pub unsafe fn run_width(&self, grid: u32) -> Result<(), String> {
        let shared = self.dev.max_dynamic_shared_bytes()?;
        unsafe { self.run_grid(grid, DEFAULT_BLOCK_THREADS, shared) }
    }

    /// Like [`Ring::run`], with `block_threads` threads and `shared_bytes` of dynamic shared memory in each block.
    ///
    /// # Safety
    /// The same rules as [`Ring::run`].
    pub unsafe fn run_shaped(&self, block_threads: u32, shared_bytes: u32) -> Result<(), String> {
        unsafe { self.run_grid(self.grid_blocks, block_threads, shared_bytes) }
    }

    /// Like [`Ring::run_shaped`], with a grid of `grid` blocks. The grid is from 1 to [`Ring::grid_blocks`].
    ///
    /// The bits of an op do not depend on the grid. The driver refuses a grid that is not all resident.
    ///
    /// # Safety
    /// The same rules as [`Ring::run`].
    pub unsafe fn run_grid(&self, grid: u32, block_threads: u32, shared_bytes: u32) -> Result<(), String> {
        if grid == 0 || grid > self.grid_blocks {
            return Err(format!(
                "a grid of {grid} blocks is not from 1 to {}, the width that this ring has room for. Use a grid in that range",
                self.grid_blocks
            ));
        }
        let dev = self.dev;
        let prep = std::time::Instant::now();
        let descs = self.descriptors(grid);
        let count = descs.len();
        // SAFETY: `host_tasks` holds `MAX_TASKS_PER_CARD` descriptors, and `descs` has fewer.
        unsafe { std::ptr::copy_nonoverlapping(descs.as_ptr(), self.host_tasks, count) };
        // The events of the tasks, and the claim words of the tasks, in one memset (the words between them are not used).
        dev.memset_zero(self.scoreboard, (CLAIM_BASE + count) * 4)?;
        self.last_grid.store(grid, Ordering::SeqCst);
        unsafe {
            dev.copy_htod(
                self.dev_tasks,
                self.host_tasks.cast::<c_void>(),
                count * std::mem::size_of::<MkTaskDescriptor>(),
            )?;
            self.last_prep_ns
                .store(prep.elapsed().as_nanos() as u64, Ordering::SeqCst);
            dev.launch_megakernel_shared(
                grid,
                block_threads,
                shared_bytes,
                self.dev_tasks,
                count as u32,
                self.scoreboard,
                0,
                self.dev_stats,
                self.scratch,
                self.profile,
            )
        }
    }

    /// Read back per-block execution statistics from the last run.
    pub fn read_stats(&self) -> Result<Vec<BlockStats>, String> {
        let dev = self.dev;
        let mut out = vec![BlockStats::default(); self.last_grid.load(Ordering::SeqCst) as usize];
        let bytes = out.len() * std::mem::size_of::<BlockStats>();
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.dev_stats, bytes)? };
        Ok(out)
    }

    /// Read back the scoreboard after the last run: the counter of each task in the ring, then one more event that
    /// no task signals. A done task has a counter that equals the target of a wait on it.
    pub fn read_scoreboard(&self) -> Result<Vec<u32>, String> {
        let dev = self.dev;
        let mut out = vec![0u32; self.len + 1];
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.scoreboard, out.len() * 4)? };
        Ok(out)
    }

    /// Read back `%globaltimer` timestamps (`[start, after_wait, end]` per task per block) from the last profiled run.
    pub fn read_profile(&self) -> Result<Vec<u64>, String> {
        let dev = self.dev;
        if self.profile == 0 {
            return Err("ring was allocated without a profile buffer".to_owned());
        }
        let count = self.len + 1;
        let words = count * self.last_grid.load(Ordering::SeqCst) as usize * PROFILE_WORDS;
        let mut out = vec![0u64; words];
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.profile, words * 8)? };
        Ok(out)
    }

    /// The most blocks that a launch of this ring can use. `run` and `run_shaped` launch this many.
    pub fn grid_blocks(&self) -> u32 {
        self.grid_blocks
    }

    /// The width of the last launch. The stats and the profile that the reads return have this many blocks.
    pub fn last_grid(&self) -> u32 {
        self.last_grid.load(Ordering::SeqCst)
    }

    /// Device pointer to the ring's scratch buffer.
    pub fn scratch(&self) -> CUdeviceptr {
        self.scratch
    }

    /// Free all device and pinned host buffers, and report every failure. A failure does not stop the other frees.
    pub fn free(mut self) -> Result<(), String> {
        self.release()
    }

    fn release(&mut self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();
        let host = std::mem::replace(&mut self.host_tasks, std::ptr::null_mut());
        // SAFETY: `host` is null or the pinned block that `new` allocated, and the field is null now.
        if let Err(e) = unsafe { self.dev.free_host_pinned(host.cast::<c_void>()) } {
            errors.push(e);
        }
        for buffer in [
            &mut self.dev_tasks,
            &mut self.scoreboard,
            &mut self.dev_stats,
            &mut self.scratch,
            &mut self.profile,
        ] {
            let ptr = std::mem::take(buffer);
            if let Err(e) = self.dev.free_device(ptr) {
                errors.push(e);
            }
        }
        self.len = 0;
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl Drop for Ring<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            eprintln!("nagaya-mk-cuda: the ring did not free all of its buffers: {error}");
        }
    }
}

impl std::fmt::Debug for Ring<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ring")
            .field("grid_blocks", &self.grid_blocks)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}
