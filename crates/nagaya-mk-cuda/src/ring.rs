//! A persistent task ring on one CUDA GPU, with its scoreboard, stats, scratch, and optional profile buffer.

use std::ffi::c_void;

use crate::driver::{CUdeviceptr, CudaDevice};
use crate::protocol::{
    BlockStats, MAX_TASKS_PER_CARD, MkTaskDescriptor, NO_EVENT, NUM_SCOREBOARD_EVENTS, OP_HALT, PROFILE_WORDS,
    RING_DEV_BYTES,
};

/// Default thread count per block for the persistent interpreter.
pub const DEFAULT_BLOCK_THREADS: u32 = 256;

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
    grid_blocks: u32,
    len: usize,
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
            len: 0,
        };
        ring.host_tasks = dev.alloc_host_pinned(RING_DEV_BYTES)?.cast::<MkTaskDescriptor>();
        ring.dev_tasks = dev.alloc_device(RING_DEV_BYTES)?;
        ring.scoreboard = dev.alloc_device(NUM_SCOREBOARD_EVENTS * 4)?;
        ring.dev_stats = dev.alloc_device(grid_blocks as usize * std::mem::size_of::<BlockStats>())?;
        ring.scratch = dev.alloc_device(scratch_bytes)?;
        if with_profile {
            let bytes = MAX_TASKS_PER_CARD * grid_blocks as usize * PROFILE_WORDS * 8;
            ring.profile = dev.alloc_device(bytes)?;
        }
        Ok(ring)
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

    /// Zero the scoreboard and scratch buffers.
    pub fn zero_device(&self) -> Result<(), String> {
        let dev = self.dev;
        dev.memset_zero(self.scoreboard, NUM_SCOREBOARD_EVENTS * 4)?;
        dev.memset_zero(self.scratch, self.scratch_floats * 4)
    }

    /// Upload the active descriptors, zero the scoreboard, and run the ring.
    ///
    /// # Safety
    /// Every device pointer referenced inside the descriptors must be valid for its op.
    pub unsafe fn run(&self) -> Result<(), String> {
        unsafe { self.run_shaped(DEFAULT_BLOCK_THREADS, 0) }
    }

    /// Like [`Ring::run`], with `block_threads` threads and `shared_bytes` of dynamic shared memory in each block.
    ///
    /// # Safety
    /// The same rules as [`Ring::run`].
    pub unsafe fn run_shaped(&self, block_threads: u32, shared_bytes: u32) -> Result<(), String> {
        let dev = self.dev;
        let count = self.len + 1;
        dev.memset_zero(self.scoreboard, (count + 1) * 4)?;
        unsafe {
            dev.copy_htod(self.dev_tasks, self.host_tasks.cast::<c_void>(), count * 64)?;
            dev.launch_megakernel_shared(
                self.grid_blocks,
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
        let mut out = vec![BlockStats::default(); self.grid_blocks as usize];
        let bytes = out.len() * std::mem::size_of::<BlockStats>();
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.dev_stats, bytes)? };
        Ok(out)
    }

    /// Read back the scoreboard after the last run: one counter for each event from 0 to `len + 1`.
    pub fn read_scoreboard(&self) -> Result<Vec<u32>, String> {
        let dev = self.dev;
        let mut out = vec![0u32; self.len + 2];
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
        let words = count * self.grid_blocks as usize * PROFILE_WORDS;
        let mut out = vec![0u64; words];
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.profile, words * 8)? };
        Ok(out)
    }

    /// Number of persistent thread blocks launched for this ring.
    pub fn grid_blocks(&self) -> u32 {
        self.grid_blocks
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
