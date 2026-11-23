//! The state of one slot on the device: the planes of the cache and the conv states of each run (`cuda` feature).

use nagaya_mk_cuda::attn::{CacheFormat, plane_bytes, row_of};
use nagaya_mk_cuda::driver::CudaDevice;

use super::dev::Dev;
use crate::spec::{CONV_ROW, CONV_ROWS, KV_HEADS, KV_ROW, LAYERS, WINDOW, is_full, plane_rows, state_layer};

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

    /// Zero the conv states and the position, and clear the poison. The planes stay as they are: a pass reads only the rows
    /// that it or a restore wrote (the position bounds each read), so a stale row is never read. A zero of the planes
    /// would write the whole slot, which is tens of gigabytes at a long context.
    pub fn clear(&mut self) -> Result<(), String> {
        for run in &self.runs {
            run.conv.zero()?;
        }
        self.position = 0;
        self.poisoned = false;
        Ok(())
    }

    fn run_of(&self, state: usize) -> Result<&RunSlot<'d>, String> {
        self.runs.get(state).ok_or_else(|| {
            format!(
                "the slot has {} states, and the call names state {state}. Name a state below {}",
                self.runs.len(),
                self.runs.len()
            )
        })
    }

    /// The rows of positions `from` to `from + n` as runs: `(first row, count, index of the first position)`. A ring
    /// gives two runs when the range wraps.
    fn row_runs(run: &RunSlot<'_>, from: usize, n: usize) -> Result<Vec<(usize, usize, usize)>, String> {
        if run.window != 0 && n > run.plane_rows {
            return Err(format!(
                "a sliding state holds the last {} positions, and the call names {n}. Name the last {} positions at most",
                run.plane_rows, run.plane_rows
            ));
        }
        let mut out: Vec<(usize, usize, usize)> = Vec::new();
        let mut at = 0;
        while at < n {
            let row = row_of(from + at, run.window, run.plane_rows);
            let count = (n - at).min(run.plane_rows - row);
            out.push((row, count, at));
            at += count;
        }
        Ok(out)
    }

    /// The bytes of one row of values of 8 heads in this format.
    fn cell_bytes(&self) -> usize {
        KV_HEADS * self.format.row_bytes()
    }

    fn check_cells(&self, state: usize, from: usize, n: usize, limit: usize, what: &str) -> Result<(), String> {
        self.run_of(state)?;
        if from + n > limit {
            return Err(format!(
                "the range of positions {from} to {} passes the {what} of {limit}. Name positions below {limit}",
                from + n
            ));
        }
        Ok(())
    }

    /// Read the key rows (`value` false) or the value rows of `state` for positions `from` to `from + n`, as
    /// position-major `[n][8][128]` in the stored format. The device holds the planes head-major. The range ends at the
    /// position of the slot or before. A sliding state keeps the last `plane_rows` positions only.
    pub fn read_cells(&self, state: usize, value: bool, from: usize, n: usize) -> Result<Vec<u8>, String> {
        self.check_cells(state, from, n, self.position, "position of the slot")?;
        let run = self.run_of(state)?;
        if run.window != 0 && n > 0 && self.position - from > run.plane_rows {
            return Err(format!(
                "position {from} is out of the ring of {} rows that ends at position {}. Read the later positions",
                run.plane_rows, self.position
            ));
        }
        let plane = if value { &run.v_plane } else { &run.k_plane };
        let row_bytes = self.format.row_bytes();
        let mut out = vec![0u8; n * self.cell_bytes()];
        for (row, count, first) in Self::row_runs(run, from, n)? {
            for head in 0..KV_HEADS {
                let offset = nagaya_mk_cuda::attn::row_offset(head, row, run.plane_rows, self.format);
                let bytes = plane.read(offset, count * row_bytes)?;
                for j in 0..count {
                    let to = ((first + j) * KV_HEADS + head) * row_bytes;
                    out[to..to + row_bytes].copy_from_slice(&bytes[j * row_bytes..(j + 1) * row_bytes]);
                }
            }
        }
        Ok(out)
    }

    /// Write the rows of positions `from` to `from + n` of `state`, from position-major `[n][8][128]` in the stored
    /// format. The slot position does not move: call [`Slot::set_position`] after the last write. A value must be finite
    /// (and not an e4m3 NaN), because the model never makes another.
    pub fn write_cells(&self, state: usize, value: bool, from: usize, n: usize, bytes: &[u8]) -> Result<(), String> {
        self.check_cells(state, from, n, self.seq, "capacity of the slot")?;
        let run = self.run_of(state)?;
        if bytes.len() != n * self.cell_bytes() {
            return Err(format!(
                "{n} positions of this cache hold {} bytes, and the call gives {}. Give {} bytes",
                n * self.cell_bytes(),
                bytes.len(),
                n * self.cell_bytes()
            ));
        }
        if let Some(at) = first_unholdable(bytes, self.format) {
            return Err(format!(
                "the value at byte {at} is not finite, and the cache holds finite values only. Write the rows of a real state"
            ));
        }
        let plane = if value { &run.v_plane } else { &run.k_plane };
        let row_bytes = self.format.row_bytes();
        for (row, count, first) in Self::row_runs(run, from, n)? {
            for head in 0..KV_HEADS {
                let mut head_bytes = Vec::with_capacity(count * row_bytes);
                for j in 0..count {
                    let at = ((first + j) * KV_HEADS + head) * row_bytes;
                    head_bytes.extend_from_slice(&bytes[at..at + row_bytes]);
                }
                let offset = nagaya_mk_cuda::attn::row_offset(head, row, run.plane_rows, self.format);
                plane.write(offset, &head_bytes)?;
            }
        }
        Ok(())
    }

    /// Read the conv state of `state` as `[3][6656]` `f32`, oldest row first. A row holds the four streams in the order
    /// of [`CONV_STREAM_WIDTHS`]. The device holds one block for each stream.
    pub fn read_conv(&self, state: usize) -> Result<Vec<u8>, String> {
        let run = self.run_of(state)?;
        let device = run.conv.read(0, CONV_BYTES)?;
        let mut out = vec![0u8; CONV_BYTES];
        let mut from = 0;
        let mut column = 0;
        for width in CONV_STREAM_WIDTHS {
            for row in 0..CONV_ROWS {
                let src = from + row * width * 4;
                let dst = (row * CONV_ROW + column) * 4;
                out[dst..dst + width * 4].copy_from_slice(&device[src..src + width * 4]);
            }
            from += CONV_ROWS * width * 4;
            column += width;
        }
        Ok(out)
    }

    /// Write the conv state of `state` from `[3][6656]` `f32`. A value must be finite.
    pub fn write_conv(&self, state: usize, bytes: &[u8]) -> Result<(), String> {
        let run = self.run_of(state)?;
        if bytes.len() != CONV_BYTES {
            return Err(format!(
                "a conv state holds {CONV_BYTES} bytes, and the call gives {}. Give {CONV_BYTES} bytes",
                bytes.len()
            ));
        }
        if let Some(at) = first_unholdable(bytes, CacheFormat::F32) {
            return Err(format!(
                "the conv value at byte {at} is not finite. Write the conv state of a real run"
            ));
        }
        let mut device = vec![0u8; CONV_BYTES];
        let mut to = 0;
        let mut column = 0;
        for width in CONV_STREAM_WIDTHS {
            for row in 0..CONV_ROWS {
                let src = (row * CONV_ROW + column) * 4;
                let dst = to + row * width * 4;
                device[dst..dst + width * 4].copy_from_slice(&bytes[src..src + width * 4]);
            }
            to += CONV_ROWS * width * 4;
            column += width;
        }
        run.conv.write(0, &device)
    }

    /// Set the position of the slot: the count of positions that a restore wrote. This is the position of the next row.
    pub fn set_position(&mut self, position: usize) -> Result<(), String> {
        if position > self.seq {
            return Err(format!(
                "the slot holds {} positions, and the call names {position}. Name at most {}",
                self.seq, self.seq
            ));
        }
        self.position = position;
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

/// The byte offset of the first value that the cache cannot hold: a NaN or an infinity, or an e4m3 NaN.
fn first_unholdable(bytes: &[u8], format: CacheFormat) -> Option<usize> {
    match format {
        CacheFormat::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .position(|b| !f32::from_le_bytes(*b).is_finite())
            .map(|i| i * 4),
        CacheFormat::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .position(|b| u16::from_le_bytes(*b) & 0x7F80 == 0x7F80)
            .map(|i| i * 2),
        CacheFormat::E4m3 => bytes.iter().position(|b| b & 0x7F == 0x7F),
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
