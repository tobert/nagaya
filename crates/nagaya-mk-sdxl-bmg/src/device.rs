//! The device memory that the engine owns on its card.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use nagaya_mk_bmg::level_zero::B70Card;

use crate::plan::STAGING_BYTES;
use crate::weights::Upload;

/// The device memory of the engine on one card. Each allocation stays until `free_all`, so a drop of the engine frees
/// all of it (engine-abi "`close` in Rust").
pub struct Device {
    /// Each allocation with its size, in the order of the allocations.
    allocs: Vec<(*mut c_void, usize)>,
    staging: *mut u8,
    /// The device bytes that the engine allocated.
    pub bytes: usize,
    /// The time in the allocations, the zero fills and the copies, for the log of an open.
    spent: Spent,
}

/// The time of the device calls of an upload.
#[derive(Default)]
struct Spent {
    allocs: usize,
    alloc: Duration,
    fill: Duration,
    copies: usize,
    copy: Duration,
}

impl Device {
    pub fn new(card: &B70Card) -> Self {
        Self {
            allocs: Vec::new(),
            staging: card.alloc_host_usm::<u8>(STAGING_BYTES),
            bytes: 0,
            spent: Spent::default(),
        }
    }

    /// Device memory of `bytes` bytes, filled with zeros.
    pub fn zeros(&mut self, card: &B70Card, bytes: usize) -> u64 {
        let bytes = crate::model::rounded(bytes);
        let at = Instant::now();
        let ptr = card.alloc_device_usm::<u8>(bytes) as *mut c_void;
        let filled = Instant::now();
        card.fill_zero_sync(ptr, bytes);
        self.spent.allocs += 1;
        self.spent.alloc += filled - at;
        self.spent.fill += filled.elapsed();
        self.allocs.push((ptr, bytes));
        self.bytes += bytes;
        ptr as u64
    }

    /// The number of allocations so far. A later `release_to` frees the ones after this mark.
    pub fn mark(&self) -> usize {
        self.allocs.len()
    }

    /// Free the allocations after `mark`, in reverse order. The memory of the others stays.
    pub fn release_to(&mut self, card: &B70Card, mark: usize) {
        while self.allocs.len() > mark {
            let (ptr, bytes) = self.allocs.pop().expect("an allocation after the mark");
            card.free(ptr);
            self.bytes -= bytes;
        }
    }

    /// Device memory that holds `data`.
    pub fn upload(&mut self, card: &B70Card, data: &[u8]) -> u64 {
        let at = self.zeros(card, data.len());
        self.write(card, at, data);
        at
    }

    /// The time of the allocations, fills and copies so far, for the log of an open.
    pub fn times(&self) -> String {
        let Spent {
            allocs,
            alloc,
            fill,
            copies,
            copy,
        } = &self.spent;
        format!(
            "{allocs} allocations in {} ms, their fills in {} ms, {copies} copies in {} ms",
            alloc.as_millis(),
            fill.as_millis(),
            copy.as_millis()
        )
    }

    /// Device memory that holds `data`, with the time of its copy in the log of an open.
    fn upload_timed(&mut self, card: &B70Card, data: &[u8]) -> u64 {
        let at = self.zeros(card, data.len());
        let started = Instant::now();
        self.write(card, at, data);
        self.spent.copies += data.len().div_ceil(STAGING_BYTES);
        self.spent.copy += started.elapsed();
        at
    }

    /// Copy `data` to the device at `at`.
    pub fn write(&self, card: &B70Card, at: u64, data: &[u8]) {
        for (i, chunk) in data.chunks(STAGING_BYTES).enumerate() {
            // SAFETY: the staging buffer holds STAGING_BYTES bytes, and a chunk is at most that long.
            unsafe { std::ptr::copy_nonoverlapping(chunk.as_ptr(), self.staging, chunk.len()) };
            card.memcpy_sync(
                (at as usize + i * STAGING_BYTES) as *mut c_void,
                self.staging as *const c_void,
                chunk.len(),
            );
        }
    }

    /// Copy `len` bytes from the device at `at`.
    pub fn read(&self, card: &B70Card, at: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        for (i, chunk) in out.chunks_mut(STAGING_BYTES).enumerate() {
            card.memcpy_sync(
                self.staging as *mut c_void,
                (at as usize + i * STAGING_BYTES) as *const c_void,
                chunk.len(),
            );
            // SAFETY: as in `write`.
            unsafe { std::ptr::copy_nonoverlapping(self.staging, chunk.as_mut_ptr(), chunk.len()) };
        }
        out
    }

    /// Free every allocation and the staging buffer. The engine runs no ring after this.
    pub fn free_all(&mut self, card: &B70Card) {
        for (ptr, _) in self.allocs.drain(..) {
            card.free(ptr);
        }
        if !self.staging.is_null() {
            card.free(self.staging as *mut c_void);
            self.staging = std::ptr::null_mut();
        }
        self.bytes = 0;
    }
}

/// The device memory of the engine with its card, for an upload.
pub struct OnCard<'a> {
    pub device: &'a mut Device,
    pub card: &'a B70Card,
}

impl Upload for OnCard<'_> {
    fn upload(&mut self, data: &[u8]) -> u64 {
        self.device.upload_timed(self.card, data)
    }

    fn writes(&self) -> bool {
        true
    }
}
