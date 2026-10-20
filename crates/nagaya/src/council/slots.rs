//! The slots of one engine: which are in use, and which fit a prompt (spec 11.3c).

use std::sync::{Arc, Mutex};

use nagaya_abi::{CoverageSpan, Engine, Limits, PartKind};

/// Which slots of an engine are in use. A build holds one slot from its lookup to its end.
#[derive(Debug)]
pub struct SlotPool {
    busy: Vec<bool>,
}

impl SlotPool {
    /// Make a pool for the slots in the limits of an engine.
    pub fn new(limits: &Limits) -> Self {
        let count: u32 = limits.slots.iter().map(|slot| slot.count).sum();
        Self {
            busy: vec![false; count as usize],
        }
    }

    /// The number of slots that nobody uses.
    pub fn free(&self) -> usize {
        self.busy.iter().filter(|busy| !**busy).count()
    }

    /// The number of slots.
    pub fn len(&self) -> usize {
        self.busy.len()
    }

    /// True when the engine has no slot.
    pub fn is_empty(&self) -> bool {
        self.busy.is_empty()
    }
}

/// The hold of a build on one slot. The slot is free again when the lease ends, also when a stop cancels the build.
#[derive(Debug)]
pub struct SlotLease {
    pool: Arc<Mutex<SlotPool>>,
    slot: u32,
}

impl SlotLease {
    /// The slot.
    pub fn slot(&self) -> u32 {
        self.slot
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().expect("the slot pool lock is poisoned");
        assert!(pool.busy[self.slot as usize], "a lease ended on a slot that was free");
        pool.busy[self.slot as usize] = false;
    }
}

/// Why no slot is free for a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoSlot {
    /// No slot of the engine holds the prompt, even when all are free.
    TooLarge,
    /// Slots that hold the prompt exist, and all are in use.
    Busy,
}

/// True when the slot holds a context of `tokens` tokens: each cells part must fit (spec 6.8).
///
/// An engine without the coverage capability holds every context in every slot.
pub fn fits(engine: &mut dyn Engine, limits: &Limits, slot: u32, tokens: usize) -> bool {
    let Some(coverage) = engine.coverage() else {
        return true;
    };
    limits.parts.iter().enumerate().all(|(part, limit)| {
        limit.kind != PartKind::Cells
            || coverage
                .fits(
                    &CoverageSpan {
                        part,
                        start: 0,
                        end: tokens as u64,
                    },
                    slot,
                )
                .is_ok()
    })
}

/// The lowest slot that holds the prompt, in use or not. A dry run reads only, so it need not hold the slot.
pub fn lowest_fitting(engine: &mut dyn Engine, limits: &Limits, pool: &SlotPool, tokens: usize) -> Option<u32> {
    (0..pool.len() as u32).find(|slot| fits(engine, limits, *slot, tokens))
}

/// Take the lowest free slot that holds the prompt.
pub fn take(
    pool: &Arc<Mutex<SlotPool>>,
    engine: &mut dyn Engine,
    limits: &Limits,
    tokens: usize,
) -> Result<SlotLease, NoSlot> {
    let mut held = pool.lock().expect("the slot pool lock is poisoned");
    let mut fitting = false;
    for slot in 0..held.len() as u32 {
        if !fits(engine, limits, slot, tokens) {
            continue;
        }
        fitting = true;
        if !held.busy[slot as usize] {
            held.busy[slot as usize] = true;
            return Ok(SlotLease {
                pool: Arc::clone(pool),
                slot,
            });
        }
    }
    Err(if fitting { NoSlot::Busy } else { NoSlot::TooLarge })
}
