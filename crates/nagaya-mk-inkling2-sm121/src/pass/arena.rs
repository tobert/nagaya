//! The activation arena of a pass: one device buffer that holds every buffer of a step (`cuda` feature).

use std::rc::Rc;

use nagaya_mk_cuda::driver::CudaDevice;

use super::dev::Dev;
use super::layout::{Buf, StepLayout};

/// The buffers of a step on the device, for blocks of up to `max_rows` rows.
pub struct Arena<'d> {
    pub layout: StepLayout,
    pub buffer: Rc<Dev<'d>>,
    /// The ids of the tokens of a step (`u32`).
    pub ids: Rc<Dev<'d>>,
}

impl<'d> Arena<'d> {
    /// Allocate and zero the arena for `rows` rows and `logit_rows` rows of logits.
    pub fn new(dev: &'d CudaDevice, rows: usize, logit_rows: usize) -> Result<Self, String> {
        let layout = StepLayout::new(rows, logit_rows);
        let buffer = Dev::alloc(dev, layout.total_bytes())?;
        buffer.zero()?;
        let ids = Dev::alloc(dev, rows * 4)?;
        Ok(Self {
            layout,
            buffer: Rc::new(buffer),
            ids: Rc::new(ids),
        })
    }

    /// Allocate and zero one arena that serves every step of up to `max_rows` rows with up to `max_logit_rows` rows of
    /// logits. The pass never allocates in a call, so it takes its one arena at open and uses [`Arena::view`].
    pub fn with_capacity(dev: &'d CudaDevice, max_rows: usize, max_logit_rows: usize) -> Result<Self, String> {
        let bytes = (1..=max_rows)
            .map(|rows| StepLayout::new(rows, rows.min(max_logit_rows)).total_bytes())
            .max()
            .ok_or_else(|| "an arena needs at least 1 row".to_owned())?;
        let layout = StepLayout::new(max_rows, max_rows.min(max_logit_rows));
        let buffer = Dev::alloc(dev, bytes)?;
        buffer.zero()?;
        let ids = Dev::alloc(dev, max_rows * 4)?;
        Ok(Self {
            layout,
            buffer: Rc::new(buffer),
            ids: Rc::new(ids),
        })
    }

    /// The layout of a step of `rows` rows with `logit_rows` rows of logits, over the same memory. The views of two
    /// shapes overlap, so a kernel reads no byte that its own step did not write.
    pub fn view(&self, rows: usize, logit_rows: usize) -> Result<Self, String> {
        if rows == 0 || rows * 4 > self.ids.bytes() || logit_rows > rows {
            return Err(format!(
                "the arena cannot hold a step of {rows} rows with {logit_rows} rows of logits. Open the pass for more rows"
            ));
        }
        let layout = StepLayout::new(rows, logit_rows);
        if layout.total_bytes() > self.buffer.bytes() {
            return Err(format!(
                "a step of {rows} rows with {logit_rows} rows of logits needs {} bytes, and the arena has {}. \
                 Open the pass for more rows or more rows of logits",
                layout.total_bytes(),
                self.buffer.bytes()
            ));
        }
        Ok(Self {
            layout,
            buffer: Rc::clone(&self.buffer),
            ids: Rc::clone(&self.ids),
        })
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
