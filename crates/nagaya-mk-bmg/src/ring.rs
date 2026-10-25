//! A ring of tasks on one card, with its scoreboard, its statistics and the scratch of its ops.
//!
//! Task `i` waits on event `i` and signals event `i + 1`, so the tasks run in order and each one sees what the one
//! before it wrote (protocol `MkTaskDescriptor`). The ring ends with `OP_HALT`. The speech engine builds its passes on
//! it, and the op tests run single ops on it.

use std::ffi::c_void;

use crate::level_zero::B70Card;
use crate::protocol::{MAX_TASKS_PER_CARD, MkTaskDescriptor, NO_EVENT, NUM_SCOREBOARD_EVENTS, OP_HALT, XeCoreStats};

/// The ring of one card.
pub struct Ring {
    tasks: *mut MkTaskDescriptor,
    scoreboard: *mut u32,
    stats: *mut XeCoreStats,
    scratch: *mut f32,
    len: usize,
}

// The pointers are USM of one card, and one thread at a time runs the ring.
unsafe impl Send for Ring {}

impl Ring {
    /// Allocate a ring on `card` with `scratch_floats` floats of scratch for the ops that need it (one-token attention
    /// keeps its chunks there).
    pub fn new(card: &B70Card, scratch_floats: usize) -> Self {
        Self {
            tasks: card.alloc_host_usm::<MkTaskDescriptor>(MAX_TASKS_PER_CARD),
            scoreboard: card.alloc_device_usm::<u32>(NUM_SCOREBOARD_EVENTS),
            stats: card.alloc_host_usm::<XeCoreStats>(32),
            scratch: card.alloc_device_usm::<f32>(scratch_floats.max(1)),
            len: 0,
        }
    }

    /// Set the tasks of the ring, in order, and chain their events. The ring holds at most `MAX_TASKS_PER_CARD - 1`
    /// tasks, because it ends with `OP_HALT`.
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
            // SAFETY: the ring holds MAX_TASKS_PER_CARD descriptors of host USM.
            unsafe { *self.tasks.add(i) = task };
        }
        let halt = MkTaskDescriptor {
            opcode: OP_HALT,
            ..Default::default()
        };
        // SAFETY: as above; `tasks.len()` is below MAX_TASKS_PER_CARD.
        unsafe { *self.tasks.add(tasks.len()) = halt };
        self.len = tasks.len();
    }

    /// Run the tasks of the ring on `card`, and wait for their end.
    ///
    /// # Safety
    ///
    /// Each pointer in the tasks must point at USM of `card` that is large enough for its op.
    pub unsafe fn run(&self, card: &B70Card) {
        unsafe {
            card.launch_megakernel(
                self.tasks as *const c_void,
                self.len as u32 + 1,
                self.scoreboard as *mut c_void,
                0,
                self.len + 1,
                std::ptr::null(),
                self.stats as *const c_void,
                self.scratch as *mut c_void,
            )
        };
    }

    /// The scratch of the ring, for an op whose task names it.
    pub fn scratch(&self) -> *mut f32 {
        self.scratch
    }

    /// Free the buffers of the ring on `card`. The ring must not run again.
    pub fn free(&mut self, card: &B70Card) {
        card.free(self.tasks as *mut c_void);
        card.free(self.scoreboard as *mut c_void);
        card.free(self.stats as *mut c_void);
        card.free(self.scratch as *mut c_void);
        self.len = 0;
    }
}
