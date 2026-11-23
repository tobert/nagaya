//! The layout of the cache and the shape of `OP_ATTN` (milestone 37 step 9 wave C, step 11 "The layout of the cache").
//!
//! The cache of one run holds two planes, one for the keys and one for the values. Each plane holds, for each KV head, a
//! block of `plane_rows` rows. A row is the 128 channels of one position and one KV head, in one format. The rows of a
//! head are contiguous, so the attention reads them as a stream. A scale of a row never covers more than one position.
//!
//! - A full run keeps position `j` in row `j`. Its `plane_rows` is the capacity of the slot, a multiple of 64.
//! - A sliding run keeps position `j` in row `j % plane_rows`. The projection writes a new row at this place, so the
//!   ring needs no spill copy. The window and the rows of the longest block must not share a row, so `plane_rows` is at
//!   least `window + max_block - 1` (see [`sliding_ring_rows`]).
//!
//! The kernel is `kernels/ops_attn.cuh`. Its fields are the fields of [`AttnShape::descriptor`].

use crate::protocol::{
    ATTN_CHUNK_KEYS, ATTN_PARTIAL_ROW_FLOATS, ATTN_SPLIT_ROWS, FLAG_F32, MkTaskDescriptor, MkTaskTable, OP_ATTN,
};

/// The query heads of a layer, the KV heads, and the width of a head.
pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
/// The query heads that share one KV head.
pub const GROUP: usize = HEADS / KV_HEADS;
/// The most rows of one task: the most rows of a prefill block.
pub const MAX_ROWS: usize = 512;
/// The width of the relative state of a head, and the distances that the relative bias covers.
pub const REL_STATES: usize = 16;
pub const REL_EXTENT: usize = 1024;

/// The format of the values of the cache.
///
/// An e4m3 cache has a unit scale: a value is the e4m3 number itself. The model clips the L2 norm of each query and key
/// and normalizes each value, so the numbers stay in the range of the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheFormat {
    /// 4 bytes for each value. The cache of the oracle.
    F32,
    /// 1 byte for each value, with a unit scale.
    E4m3,
}

impl CacheFormat {
    /// The bytes of one row of 128 values, scale included.
    pub fn row_bytes(self) -> usize {
        match self {
            CacheFormat::F32 => HEAD_DIM * 4,
            CacheFormat::E4m3 => HEAD_DIM,
        }
    }

    /// The flags of the task for this format.
    pub fn flags(self) -> u16 {
        match self {
            CacheFormat::F32 => FLAG_F32,
            CacheFormat::E4m3 => 0,
        }
    }
}

/// The rows of the ring of a sliding run: the least multiple of `ATTN_CHUNK_KEYS` that is at least
/// `window + max_block - 1`. A chunk never crosses the end of the ring, and a block of `max_block` rows never writes a
/// row that the window of its first row still reads.
pub const fn sliding_ring_rows(window: usize, max_block: usize) -> usize {
    (window + max_block - 1).next_multiple_of(ATTN_CHUNK_KEYS)
}

/// The row of position `position` in a plane of `plane_rows` rows. `window` is 0 for a full run.
pub fn row_of(position: usize, window: usize, plane_rows: usize) -> usize {
    if window == 0 { position } else { position % plane_rows }
}

/// The bytes of one plane: `KV_HEADS` blocks of `plane_rows` rows.
pub fn plane_bytes(plane_rows: usize, format: CacheFormat) -> usize {
    KV_HEADS * plane_rows * format.row_bytes()
}

/// The byte offset in a plane of row `row` of KV head `kvh`.
pub fn row_offset(kvh: usize, row: usize, plane_rows: usize, format: CacheFormat) -> usize {
    (kvh * plane_rows + row) * format.row_bytes()
}

/// The first position that a query at `position` reads: 0 for a full run, and `position + 1 - window` for a sliding run.
pub fn lowest_key(position: usize, window: usize) -> usize {
    if window == 0 {
        0
    } else {
        (position + 1).saturating_sub(window)
    }
}

/// The device addresses of one `OP_ATTN` task: the queries, the key plane, the value plane, the output, the scratch and
/// the task table (`ATTN_TABLE_*` gives its slots).
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnBuffers {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub out: u64,
    pub scratch: u64,
    pub table: u64,
}

/// The shape of one `OP_ATTN` task: the rows, the place of the rows, the run and the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub rows: usize,
    pub pos0: usize,
    /// 0 for a full run, or the window of a sliding run.
    pub window: usize,
    pub plane_rows: usize,
    pub format: CacheFormat,
}

impl AttnShape {
    /// Check the shape against the rules of the kernel. The error names the rule and what to change.
    pub fn new(
        rows: usize,
        pos0: usize,
        window: usize,
        plane_rows: usize,
        format: CacheFormat,
    ) -> Result<Self, String> {
        if rows == 0 || rows > MAX_ROWS {
            return Err(format!(
                "OP_ATTN takes 1 to {MAX_ROWS} rows and got {rows}. Cut the block into smaller blocks"
            ));
        }
        if plane_rows == 0 || !plane_rows.is_multiple_of(64) {
            return Err(format!(
                "a plane of {plane_rows} rows is not a multiple of 64. Round the rows of the plane up"
            ));
        }
        if window == 0 && pos0 + rows > plane_rows {
            return Err(format!(
                "a full run holds {plane_rows} positions, and the rows end at position {}. Grow the slot",
                pos0 + rows
            ));
        }
        if window != 0 && pos0 + rows > plane_rows && plane_rows < window + rows - 1 {
            return Err(format!(
                "a ring of {plane_rows} rows is too short for a window of {window} and {rows} rows: it needs {}. \
                 Use sliding_ring_rows",
                window + rows - 1
            ));
        }
        if window != 0 && !plane_rows.is_multiple_of(ATTN_CHUNK_KEYS) && pos0 + rows > plane_rows {
            return Err(format!(
                "a ring of {plane_rows} rows is not a multiple of the chunk of {ATTN_CHUNK_KEYS} keys. Use sliding_ring_rows"
            ));
        }
        Ok(Self {
            rows,
            pos0,
            window,
            plane_rows,
            format,
        })
    }

    /// True when the task runs in the split mode: one item for each chunk, with a merge group.
    pub fn split(&self) -> bool {
        self.rows <= ATTN_SPLIT_ROWS
    }

    /// The tiles of 16 rows. A row is a pair of a position and a query head, `GROUP` heads for each position.
    pub fn tiles(&self) -> usize {
        (GROUP * self.rows).div_ceil(16)
    }

    /// The live rows of tile `tile`.
    pub fn tile_rows(&self, tile: usize) -> usize {
        (GROUP * self.rows - 16 * tile).min(16)
    }

    /// The index of the first chunk that a row of the task reads.
    pub fn first_chunk(&self) -> usize {
        lowest_key(self.pos0, self.window) / ATTN_CHUNK_KEYS
    }

    /// The count of chunks that the task visits.
    pub fn chunks(&self) -> usize {
        (self.pos0 + self.rows - 1) / ATTN_CHUNK_KEYS - self.first_chunk() + 1
    }

    /// The count of work items: a chunk, a KV head and a tile in the split mode, and a KV head and a tile otherwise.
    pub fn items(&self) -> usize {
        (if self.split() { self.chunks() } else { 1 }) * KV_HEADS * self.tiles()
    }

    /// The count of merge groups: a KV head and a tile. Only the split mode uses them.
    pub fn groups(&self) -> usize {
        KV_HEADS * self.tiles()
    }

    /// The floats of the scratch: one counter for each group, padded to 4 floats, then one partial for each chunk of
    /// each group. A partial has 16 rows of `ATTN_PARTIAL_ROW_FLOATS`. The whole mode needs no scratch.
    pub fn scratch_floats(&self) -> usize {
        if self.split() {
            self.groups().next_multiple_of(4) + self.groups() * self.chunks() * 16 * ATTN_PARTIAL_ROW_FLOATS
        } else {
            0
        }
    }

    /// The bytes of the keys that the task reads from the cache: the chunks, for each KV head, in one plane.
    /// A full run at one row reads every position from 0 to `pos0`.
    pub fn key_bytes(&self) -> usize {
        let last = self.pos0 + self.rows - 1;
        let positions = last + 1 - lowest_key(self.pos0, self.window);
        positions * KV_HEADS * self.format.row_bytes()
    }

    /// The descriptor of the task. The score scale is `1 / sqrt(128)` as an `f32` (see [`score_scale`]).
    pub fn descriptor(&self, buffers: &AttnBuffers) -> MkTaskDescriptor {
        MkTaskDescriptor {
            opcode: OP_ATTN,
            flags: self.format.flags(),
            items: self.items() as u32,
            rows: self.rows as u32,
            a: buffers.q,
            b: buffers.k,
            c: buffers.out,
            d: buffers.table,
            e: buffers.v,
            f: buffers.scratch,
            n: self.window as u32,
            k: self.plane_rows as u32,
            kt: ATTN_CHUNK_KEYS as u32,
            group_items: self.chunks() as u32,
            scale_bytes: self.format.row_bytes() as u32,
            pos0: self.pos0 as u32,
            p0: score_scale(),
            ..Default::default()
        }
    }
}

/// The task table of `OP_ATTN`: the relative states, the sinks, the gate and the relative projection.
pub fn table(rel: u64, sinks: u64, gate: u64, proj: u64) -> MkTaskTable {
    use crate::protocol::{ATTN_TABLE_GATE, ATTN_TABLE_PROJ, ATTN_TABLE_REL, ATTN_TABLE_SINKS};
    let mut table = MkTaskTable::default();
    table.ptr[ATTN_TABLE_REL] = rel;
    table.ptr[ATTN_TABLE_SINKS] = sinks;
    table.ptr[ATTN_TABLE_GATE] = gate;
    table.ptr[ATTN_TABLE_PROJ] = proj;
    table
}

/// The scale of a score: `1 / sqrt(128)` as the host form computes it.
pub fn score_scale() -> f32 {
    (HEAD_DIM as f32).powf(-0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_of_a_sliding_run_fits_the_window_and_the_longest_block_without_a_spill() {
        // 1,024 + 512 - 1 = 1,535 positions, rounded up to the chunk: 1,536 rows, not the 3,072 of a ring with a spill.
        assert_eq!(sliding_ring_rows(1024, 512), 1536);
        assert_eq!(sliding_ring_rows(1024, 1), 1024);
        assert_eq!(sliding_ring_rows(1024, 257), 1280);
    }

    #[test]
    fn a_row_of_a_ring_is_the_position_modulo_the_rows() {
        assert_eq!(row_of(5, 1024, 1536), 5);
        assert_eq!(row_of(1536, 1024, 1536), 0);
        assert_eq!(row_of(4000, 1024, 1536), 4000 % 1536);
        assert_eq!(row_of(4000, 0, 8192), 4000);
    }

    #[test]
    fn the_rows_of_a_head_are_contiguous() {
        let f = CacheFormat::F32;
        assert_eq!(row_offset(0, 1, 64, f) - row_offset(0, 0, 64, f), 512);
        assert_eq!(row_offset(1, 0, 64, f), 64 * 512);
        assert_eq!(plane_bytes(64, CacheFormat::E4m3), 8 * 64 * 128);
    }

    #[test]
    fn the_chunks_of_a_decode_step_sit_on_absolute_positions() {
        let full = AttnShape::new(1, 65_535, 0, 65_536, CacheFormat::F32).unwrap();
        assert_eq!((full.chunks(), full.first_chunk(), full.items()), (256, 0, 256 * 8));
        // A sliding run at position 5,000 reads positions 3,977 to 5,000: chunks 15 to 19.
        let slide = AttnShape::new(1, 5000, 1024, 1536, CacheFormat::F32).unwrap();
        assert_eq!((slide.first_chunk(), slide.chunks()), (15, 5));
        // The rows of the first position of the block decide the first chunk, so a block reads one more chunk.
        let block = AttnShape::new(512, 8192, 1024, 1536, CacheFormat::F32).unwrap();
        assert!(!block.split());
        assert_eq!(block.items(), KV_HEADS * block.tiles());
        assert_eq!(block.tiles(), 96);
    }

    #[test]
    fn a_shape_that_the_kernel_does_not_cover_is_an_error() {
        let bad = [
            AttnShape::new(0, 0, 0, 64, CacheFormat::F32),
            AttnShape::new(513, 0, 0, 1024, CacheFormat::F32),
            AttnShape::new(1, 64, 0, 64, CacheFormat::F32),
            AttnShape::new(1, 0, 0, 100, CacheFormat::F32),
            AttnShape::new(512, 4096, 1024, 1024, CacheFormat::F32),
        ];
        for result in bad {
            assert!(result.is_err(), "{result:?}");
        }
    }

    #[test]
    fn the_scratch_holds_the_counters_then_one_partial_for_each_chunk_of_a_group() {
        let s = AttnShape::new(1, 1023, 0, 1024, CacheFormat::F32).unwrap();
        assert_eq!((s.groups(), s.chunks()), (8, 4));
        assert_eq!(s.scratch_floats(), 8 + 8 * 4 * 16 * ATTN_PARTIAL_ROW_FLOATS);
        let whole = AttnShape::new(17, 0, 0, 1024, CacheFormat::F32).unwrap();
        assert_eq!(whole.scratch_floats(), 0);
    }

    #[test]
    fn the_descriptor_names_the_shape() {
        let s = AttnShape::new(3, 700, 1024, 1536, CacheFormat::E4m3).unwrap();
        let b = AttnBuffers {
            q: 16,
            k: 32,
            v: 48,
            out: 64,
            scratch: 80,
            table: 128,
        };
        let d = s.descriptor(&b);
        assert_eq!((d.opcode, d.flags, d.rows, d.pos0), (OP_ATTN, 0, 3, 700));
        assert_eq!((d.n, d.k, d.kt), (1024, 1536, 256));
        assert_eq!(
            (d.items as usize, d.group_items as usize, d.scale_bytes),
            (s.items(), s.chunks(), 128)
        );
        let f = AttnShape::new(3, 700, 1024, 1536, CacheFormat::F32).unwrap();
        assert_eq!(f.descriptor(&b).flags, FLAG_F32);
    }
}
