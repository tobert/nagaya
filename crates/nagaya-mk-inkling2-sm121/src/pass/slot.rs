//! The state of one slot on the device: the planes of the cache and the conv states of each run (`cuda` feature).

use nagaya_mk_cuda::attn::{CacheFormat, plane_bytes};
use nagaya_mk_cuda::driver::CudaDevice;

use super::dev::Dev;
use crate::spec::{CONV_ROW, CONV_ROWS, KV_ROW, LAYERS, WINDOW, is_full, plane_rows, state_layer};

/// The values of the 4 conv states of a run, in the order of their streams: k, v, attn, mlp. Each holds 3 rows.
pub const CONV_STREAM_WIDTHS: [usize; 4] = [KV_ROW, KV_ROW, 2304, 2304];

/// The byte offset of the conv state of `stream` (0 k, 1 v, 2 attn, 3 mlp) in the conv buffer of a run.
pub fn conv_offset(stream: usize) -> usize {
    CONV_STREAM_WIDTHS[..stream].iter().sum::<usize>() * CONV_ROWS * 4
}

/// The bytes of the conv buffer of a run: 3 rows of each stream, `f32`.
pub const CONV_BYTES: usize = CONV_ROWS * CONV_ROW * 4;

/// The state of one run of a slot.
pub struct RunSlot<'d> {
    pub state: usize,
    pub layer: usize,
    /// The rows of a plane, and the window (0 for a full run).
    pub plane_rows: usize,
    pub window: usize,
    pub k_plane: Dev<'d>,
    pub v_plane: Dev<'d>,
    /// The conv states of the four streams, one after the other (`conv_offset`).
    pub conv: Dev<'d>,
}

impl RunSlot<'_> {
    /// The device address of the conv state of `stream`.
    pub fn conv_ptr(&self, stream: usize) -> u64 {
        self.conv.ptr() + conv_offset(stream) as u64
    }
}

/// A slot: the state of every run, for sequences of up to `seq` positions.
pub struct Slot<'d> {
    pub seq: usize,
    pub format: CacheFormat,
    /// The next position to write.
    pub position: usize,
    /// A failed launch can leave the state wrong. A poisoned slot refuses a new pass until it is zeroed.
    pub poisoned: bool,
    pub runs: Vec<RunSlot<'d>>,
}

impl<'d> Slot<'d> {
    /// A zeroed slot of `seq` positions with `extra` runs in the reserve of the cycles after the first.
    pub fn new(dev: &'d CudaDevice, seq: usize, extra: usize, format: CacheFormat) -> Result<Self, String> {
        let mut runs = Vec::with_capacity(LAYERS + extra);
        for state in 0..LAYERS + extra {
            let layer = state_layer(state);
            let rows = plane_rows(seq, state);
            let bytes = plane_bytes(rows, format);
            let run = RunSlot {
                state,
                layer,
                plane_rows: rows,
                window: if is_full(layer) { 0 } else { WINDOW },
                k_plane: Dev::alloc(dev, bytes)?,
                v_plane: Dev::alloc(dev, bytes)?,
                conv: Dev::alloc(dev, CONV_BYTES)?,
            };
            run.k_plane.zero()?;
            run.v_plane.zero()?;
            run.conv.zero()?;
            runs.push(run);
        }
        Ok(Self {
            seq,
            format,
            position: 0,
            poisoned: false,
            runs,
        })
    }

    /// Set the state of every run to zero and the position to 0. This clears the poison.
    pub fn zero(&mut self) -> Result<(), String> {
        for run in &self.runs {
            run.k_plane.zero()?;
            run.v_plane.zero()?;
            run.conv.zero()?;
        }
        self.position = 0;
        self.poisoned = false;
        Ok(())
    }

    /// The bytes of device memory of the slot.
    pub fn bytes(&self) -> usize {
        self.runs
            .iter()
            .map(|r| r.k_plane.bytes() + r.v_plane.bytes() + r.conv.bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conv_buffer_has_the_four_streams_in_order() {
        assert_eq!(conv_offset(0), 0);
        assert_eq!(conv_offset(1), 3 * 1024 * 4);
        assert_eq!(conv_offset(2), 3 * 2048 * 4);
        assert_eq!(conv_offset(3), (3 * 2048 + 3 * 2304) * 4);
        assert_eq!(conv_offset(3) + 3 * 2304 * 4, CONV_BYTES);
        assert_eq!(CONV_BYTES, 79_872);
    }
}
