//! A persistent task ring on one CUDA GPU, with its scoreboard, stats, scratch, and optional profile buffer.

use std::ffi::c_void;

use crate::driver::{CUdeviceptr, CudaDevice};
use crate::protocol::{
    BlockStats, MAX_TASKS_PER_CARD, MkTaskDescriptor, NO_EVENT, NUM_SCOREBOARD_EVENTS, OP_HALT, PROFILE_WORDS,
    RING_DEV_BYTES,
};

/// Default thread count per block for the persistent interpreter.
pub const DEFAULT_BLOCK_THREADS: u32 = 256;

/// One task ring allocated on a [`CudaDevice`].
pub struct Ring {
    host_tasks: *mut MkTaskDescriptor,
    dev_tasks: CUdeviceptr,
    scoreboard: CUdeviceptr,
    dev_stats: CUdeviceptr,
    scratch: CUdeviceptr,
    scratch_floats: usize,
    profile: CUdeviceptr,
    grid_blocks: u32,
    len: usize,
}

unsafe impl Send for Ring {}

impl Ring {
    /// Allocate a ring on `dev` with `scratch_floats` floats of scratch and optional `%globaltimer` profile buffer.
    pub fn new(dev: &CudaDevice, scratch_floats: usize, with_profile: bool) -> Result<Self, String> {
        let grid_blocks = dev.sm_count();
        let host_tasks = dev.alloc_host_pinned(RING_DEV_BYTES)?.cast::<MkTaskDescriptor>();
        let dev_tasks = dev.alloc_device(RING_DEV_BYTES)?;
        let scoreboard = dev.alloc_device(NUM_SCOREBOARD_EVENTS * 4)?;
        let dev_stats = dev.alloc_device(grid_blocks as usize * std::mem::size_of::<BlockStats>())?;
        let scratch_floats = scratch_floats.max(1);
        let scratch = dev.alloc_device(scratch_floats * 4)?;
        let profile = if with_profile {
            let bytes = MAX_TASKS_PER_CARD * grid_blocks as usize * PROFILE_WORDS * 8;
            dev.alloc_device(bytes)?
        } else {
            0
        };
        Ok(Self {
            host_tasks,
            dev_tasks,
            scoreboard,
            dev_stats,
            scratch,
            scratch_floats,
            profile,
            grid_blocks,
            len: 0,
        })
    }

    /// Load `tasks` into the ring, chaining each task's wait and signal events and appending `OP_HALT`.
    pub fn set(&mut self, tasks: &[MkTaskDescriptor]) {
        assert!(
            tasks.len() < MAX_TASKS_PER_CARD,
            "a ring holds at most {} tasks, and this one has {}",
            MAX_TASKS_PER_CARD - 1,
            tasks.len()
        );
        for (i, task) in tasks.iter().enumerate() {
            let mut task = *task;
            task.wait_event = if i == 0 { NO_EVENT } else { i as u32 };
            task.signal_event = i as u32 + 1;
            unsafe { *self.host_tasks.add(i) = task };
        }
        let halt = MkTaskDescriptor {
            opcode: OP_HALT,
            wait_event: if tasks.is_empty() { NO_EVENT } else { tasks.len() as u32 },
            signal_event: NO_EVENT,
            ..Default::default()
        };
        unsafe { *self.host_tasks.add(tasks.len()) = halt };
        self.len = tasks.len();
    }

    /// Zero the scoreboard and scratch buffers on `dev`.
    pub fn zero_device(&self, dev: &CudaDevice) -> Result<(), String> {
        dev.memset_zero(self.scoreboard, NUM_SCOREBOARD_EVENTS * 4)?;
        dev.memset_zero(self.scratch, self.scratch_floats * 4)
    }

    /// Upload the active descriptors, zero the scoreboard, and run the ring on `dev`.
    ///
    /// # Safety
    /// Every device pointer referenced inside the descriptors must be valid for its op.
    pub unsafe fn run(&self, dev: &CudaDevice) -> Result<(), String> {
        let count = self.len + 1;
        dev.memset_zero(self.scoreboard, (count + 1) * 4)?;
        unsafe {
            dev.copy_htod(self.dev_tasks, self.host_tasks.cast::<c_void>(), count * 64)?;
            dev.launch_megakernel(
                self.grid_blocks,
                DEFAULT_BLOCK_THREADS,
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
    pub fn read_stats(&self, dev: &CudaDevice) -> Result<Vec<BlockStats>, String> {
        let mut out = vec![BlockStats::default(); self.grid_blocks as usize];
        let bytes = out.len() * std::mem::size_of::<BlockStats>();
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.dev_stats, bytes)? };
        Ok(out)
    }

    /// Device pointer to the ring's scratch buffer.
    pub fn scratch(&self) -> CUdeviceptr {
        self.scratch
    }

    /// Free all device and pinned host buffers held by this ring.
    pub fn free(&mut self, dev: &CudaDevice) -> Result<(), String> {
        unsafe { dev.free_host_pinned(self.host_tasks.cast::<c_void>())? };
        self.host_tasks = std::ptr::null_mut();
        dev.free_device(self.dev_tasks)?;
        self.dev_tasks = 0;
        dev.free_device(self.scoreboard)?;
        self.scoreboard = 0;
        dev.free_device(self.dev_stats)?;
        self.dev_stats = 0;
        dev.free_device(self.scratch)?;
        self.scratch = 0;
        dev.free_device(self.profile)?;
        self.profile = 0;
        self.len = 0;
        Ok(())
    }
}
