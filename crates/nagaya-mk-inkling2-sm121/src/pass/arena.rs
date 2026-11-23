//! The activation arena of a pass: one device buffer that holds every buffer of a step (`cuda` feature).

use nagaya_mk_cuda::driver::CudaDevice;

use super::dev::Dev;
use super::layout::{Buf, StepLayout};

/// The buffers of a step on the device, for blocks of up to `max_rows` rows.
pub struct Arena<'d> {
    pub layout: StepLayout,
    pub buffer: Dev<'d>,
    /// The ids of the tokens of a step (`u32`).
    pub ids: Dev<'d>,
}

impl<'d> Arena<'d> {
    /// Allocate and zero the arena for `rows` rows and `logit_rows` rows of logits.
    pub fn new(dev: &'d CudaDevice, rows: usize, logit_rows: usize) -> Result<Self, String> {
        let layout = StepLayout::new(rows, logit_rows);
        let buffer = Dev::alloc(dev, layout.total_bytes())?;
        buffer.zero()?;
        let ids = Dev::alloc(dev, rows * 4)?;
        Ok(Self { layout, buffer, ids })
    }

    /// The device address of `buf`.
    pub fn addr(&self, buf: Buf) -> u64 {
        self.buffer.ptr() + self.layout.offset(buf) as u64
    }

    /// The device address of the row factors of `buf`.
    pub fn rf(&self, buf: Buf) -> u64 {
        self.buffer.ptr() + self.layout.rf(buf) as u64
    }

    /// The device address of row `row` of the buffer `buf` that holds rows of `width` floats.
    pub fn row(&self, buf: Buf, row: usize, width: usize) -> u64 {
        self.addr(buf) + (row * width * 4) as u64
    }
}
