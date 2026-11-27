//! The output of the attention and the mix of the MLP input (`OP_OUT_MIX`).
//!
//! One task does the matmul `wo` (2,304 columns, K = 3,072), the short convolution of the attention with its conv state,
//! the residual, and, for each row, the mix of the MLP input and its norm. The fields and the order of the steps are in
//! `kernels/ops_out_mix.cuh`. The host form is the part of `Model::layer` in `nagaya-mk-inkling2-bmg` between the
//! attention and the experts.
//!
//! The quantization of `h_e` for the experts is not part of this op. A row of `h_e` is complete only after the finisher of
//! the task, so a quantizer task (`OP_QUANT_NVFP4`) or the prologue of `ROUTE` does it.

use crate::protocol::{FLAG_ACCUMULATE, FLAG_BLOCK_FORM, FLAG_NORM, MkTaskDescriptor, MkTaskTable, OP_OUT_MIX};
use crate::protocol::{
    OUT_MIX_TABLE_CONV, OUT_MIX_TABLE_NORM, OUT_MIX_TABLE_OUT, OUT_MIX_TABLE_QUERY, OUT_MIX_TABLE_STACK_RF,
    OUT_MIX_TABLE_STATE,
};
use crate::repack::{COLUMN_TILE, MatmulShape};

/// The width of the residual stream, and the values in a row of the attention output.
pub const WIDTH: usize = 2304;
pub const K: usize = 3072;
/// The tiles of 16 columns of `wo`. The item that arrives last over the tiles runs the finisher.
pub const TILES: usize = WIDTH / COLUMN_TILE;
/// The words of scratch before the partials: one counter for each tile, the counter of the finisher, and padding.
pub const COUNTERS: usize = 160;
/// The most rows of the stack: the rows of the blocks, at most 4, and the partial row `r`.
pub const MAX_DEPTH: usize = 5;
/// The most rows of one task.
pub const MAX_ROWS: usize = 512;
/// The rows that one finisher item runs.
pub const FINISH_ROWS: usize = crate::protocol::OUT_MIX_FINISH_ROWS;
/// Proposed: a task of fewer rows keeps the finisher in the item that arrives last over the tiles. At one row the
/// serial finisher costs one row, and a finisher item would add the claim and the poll of one more item.
pub const SPREAD_MIN_ROWS: usize = 2;

/// The finisher items after the 576 items of the matmul, for a task of `rows` rows that spreads the finisher: item
/// `576 + j` runs the rows `j * FINISH_ROWS` up to `(j + 1) * FINISH_ROWS`.
/// Proposed: a task of this many rows or more runs its matmul in an earlier block-form task (`FLAG_BLOCK_FORM` on both
/// tasks, see `OutMixShape::split`). Measured on the GB10 (layers 0 and 5, release): the split form takes 105 us from 17 to
/// 32 rows. The fused op takes 101 to 104 us at 17 to 24 rows (it wins by 1 to 3 us), 120 us at 28 and 32 rows, 143 us at 33
/// rows, and 79 us at 16 rows. 25 to 27 rows were not measured. Fewer rows keep the fused op.
pub const SPLIT_MIN_ROWS: usize = 28;

/// The floats from the start of the scratch to the raw sums `[row][2304]` that the split form shares between its two
/// tasks (the place of the raw sums in the fused op): the counters and the partials of the 576 items.
pub fn raw_offset_floats(rows: usize) -> usize {
    COUNTERS + shape().items() * rows * COLUMN_TILE
}

pub fn finisher_items(rows: usize) -> usize {
    rows.div_ceil(FINISH_ROWS)
}

/// The cut of `wo` into work items: 144 tiles with 4 K slices of 768.
pub fn shape() -> MatmulShape {
    MatmulShape::new(WIDTH, K).expect("wo fills its tiles")
}

/// The floats of scratch that a task of `rows` rows needs: the counters, the partials of the 576 items, and the raw sums.
/// It is the number that the kernel checks against `scale_bytes`.
pub fn scratch_floats(rows: usize) -> usize {
    COUNTERS + shape().items() * rows * COLUMN_TILE + WIDTH * rows
}

/// The floats of the output at the slot `OUT_MIX_TABLE_OUT`: `e` for each row, then `h_e` for each row, then the row
/// factor of each `e`.
pub fn out_floats(rows: usize) -> usize {
    2 * rows * WIDTH + rows
}

/// The float offsets in the output of `e`, `h_e` and the row factors of `e`.
pub fn out_offsets(rows: usize) -> (usize, usize, usize) {
    (0, rows * WIDTH, 2 * rows * WIDTH)
}

/// The addresses of one task, on the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutMixBuffers {
    /// The attention output, `[rows][3072]` f32.
    pub attn: u64,
    /// The repacked `wo`.
    pub packed: u64,
    /// `r`, `[rows][2304]` f32: the task writes it. It is the last row of the stack.
    pub r: u64,
    /// `x`, the input of the layer, `[rows][2304]` f32. It is 0 on the first layer of a block, where `r` has no `x`.
    pub x: u64,
    /// The scratch, `scratch_floats(rows)` floats, the counters at 0.
    pub scratch: u64,
    /// The task table.
    pub table: u64,
}

/// The shape of one task: the rows, the depth of the stack and the flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutMixShape {
    pub rows: usize,
    /// The rows of the stack of the mix: the planes of the blocks, then `r`. It is 2 to 5.
    pub depth: usize,
    /// `r = x + conv(o)`, or `r = conv(o)` on the first layer of a block.
    pub accumulate: bool,
    /// Write `h_e = mlp_norm * (e * rf)`. It is false when a message gate follows.
    pub norm: bool,
    /// Run the finisher in items after the matmul, over all the blocks, and not in the one item that arrives last. The
    /// bits are the same. `new` sets it for `SPREAD_MIN_ROWS` rows or more.
    pub spread: bool,
    /// The split form: a block-form `OP_MATMUL_BF16` task writes the raw sums of `wo` (`matmul_descriptor`), and this
    /// task has only the convolution, the state and the finisher items (`FLAG_BLOCK_FORM` on `OP_OUT_MIX` means this).
    /// The bits are those of the fused op. It needs the spread finisher. `new` leaves it off: the pass chooses it with
    /// `with_split(rows >= SPLIT_MIN_ROWS)`.
    pub split: bool,
}

impl OutMixShape {
    /// Check the shape against the rules of the kernel. The error names the rule and what to change.
    pub fn new(rows: usize, depth: usize, accumulate: bool, norm: bool) -> Result<Self, String> {
        if rows == 0 || rows > MAX_ROWS {
            return Err(format!(
                "OP_OUT_MIX takes 1 to {MAX_ROWS} rows and got {rows}. Cut the block into smaller blocks"
            ));
        }
        if !(2..=MAX_DEPTH).contains(&depth) {
            return Err(format!(
                "the stack of OP_OUT_MIX has {depth} rows. It needs 2 to {MAX_DEPTH}: 1 to {} planes, then `r`",
                MAX_DEPTH - 1
            ));
        }
        Ok(Self {
            rows,
            depth,
            accumulate,
            norm,
            spread: rows >= SPREAD_MIN_ROWS,
            split: false,
        })
    }

    /// The same task with the finisher spread over the blocks or not (a test of the bits, a measure of the two forms).
    pub fn with_spread(mut self, spread: bool) -> Self {
        self.spread = spread;
        self
    }

    /// The same task in the split form or not. The split form spreads the finisher.
    pub fn with_split(mut self, split: bool) -> Self {
        self.split = split;
        self.spread |= split;
        self
    }

    /// The items of the task: the 576 of the matmul (the 144 tiles of the convolution in the split form) and, if the
    /// finisher spreads, its items.
    pub fn items(&self) -> usize {
        let first = if self.split { TILES } else { shape().items() };
        first
            + if self.spread || self.split {
                finisher_items(self.rows)
            } else {
                0
            }
    }

    /// The descriptor of the block-form matmul task of the split form: `wo` on the attention output, to the raw sums in
    /// the scratch of this task. It is an `OP_MATMUL_BF16` task in the block form, and the builder sets its waits.
    pub fn matmul_descriptor(&self, buffers: &OutMixBuffers) -> Result<MkTaskDescriptor, String> {
        if !self.split {
            return Err("only the split form of OP_OUT_MIX has a matmul task. Call with_split(true)".to_string());
        }
        shape().descriptor_block(
            self.rows,
            buffers.attn,
            buffers.packed,
            buffers.scratch + 4 * raw_offset_floats(self.rows) as u64,
        )
    }

    /// The descriptor of the task. `scale` is the scale of a score, `2304^-0.5`.
    pub fn descriptor(&self, buffers: &OutMixBuffers) -> Result<MkTaskDescriptor, String> {
        if self.accumulate != (buffers.x != 0) {
            return Err(
                "OP_OUT_MIX adds `x` exactly when FLAG_ACCUMULATE is set. Give `x`, or clear the flag".to_string(),
            );
        }
        let mut desc = shape().descriptor(self.rows, buffers.attn, buffers.packed, buffers.r, buffers.scratch)?;
        desc.opcode = OP_OUT_MIX;
        desc.items = u32::try_from(self.items()).expect("the items of OP_OUT_MIX fit in 32 bits");
        desc.flags = if self.accumulate { FLAG_ACCUMULATE } else { 0 } | if self.norm { FLAG_NORM } else { 0 };
        if self.split {
            // No matmul in this task: the raw sums are in the scratch, and the 144 tiles run the convolution.
            desc.flags |= FLAG_BLOCK_FORM;
            desc.a = buffers.scratch + 4 * raw_offset_floats(self.rows) as u64;
            desc.b = 0;
            desc.item_bytes = 0;
        }
        desc.d = buffers.table;
        desc.e = buffers.x;
        desc.layer = self.depth as u32;
        desc.p0 = score_scale();
        desc.scale_bytes = u32::try_from(scratch_floats(self.rows))
            .map_err(|_| "the scratch of OP_OUT_MIX does not fit in 32 bits".to_string())?;
        Ok(desc)
    }
}

/// The scale of a score of the mix: `2304^-0.5` as an `f32`, as `ops::attention_residual` computes it.
pub fn score_scale() -> f32 {
    (WIDTH as f32).powf(-0.5)
}

/// The addresses that the table of the task holds.
#[derive(Clone, Debug, Default)]
pub struct OutMixTable {
    /// The planes of the stack, f32 `[rows][2304]`, and the row factor of each (`[rows]`), then the row factors that
    /// the task writes for `r`. `planes.len() + 1` is the depth.
    pub planes: Vec<(u64, u64)>,
    /// The row factors of `r`: `rows` floats. The task writes them.
    pub r_rf: u64,
    /// `r`: the same address as `OutMixBuffers::r`.
    pub r: u64,
    /// The weight of the convolution (bf16, `[2304][4]`), the conv state (f32, `[3][2304]`), the norm weight of `h_e`
    /// (f32, 2304, or 0 without the norm), the outputs (`out_floats`) and the query of the mix (f32, 2304).
    pub conv_weight: u64,
    pub conv_state: u64,
    pub norm_weight: u64,
    pub out: u64,
    pub query: u64,
}

impl OutMixTable {
    pub fn table(&self) -> Result<MkTaskTable, String> {
        let depth = self.planes.len() + 1;
        if !(2..=MAX_DEPTH).contains(&depth) {
            return Err(format!(
                "the stack of OP_OUT_MIX has {depth} rows. It needs 1 to {} planes",
                MAX_DEPTH - 1
            ));
        }
        let mut table = MkTaskTable::default();
        for (s, (plane, rf)) in self.planes.iter().enumerate() {
            table.ptr[s] = *plane;
            table.ptr[OUT_MIX_TABLE_STACK_RF + s] = *rf;
        }
        table.ptr[depth - 1] = self.r;
        table.ptr[OUT_MIX_TABLE_STACK_RF + depth - 1] = self.r_rf;
        table.ptr[OUT_MIX_TABLE_CONV] = self.conv_weight;
        table.ptr[OUT_MIX_TABLE_STATE] = self.conv_state;
        table.ptr[OUT_MIX_TABLE_NORM] = self.norm_weight;
        table.ptr[OUT_MIX_TABLE_OUT] = self.out;
        table.ptr[OUT_MIX_TABLE_QUERY] = self.query;
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cut_is_144_tiles_with_four_slices() {
        let s = shape();
        assert_eq!((s.tiles(), s.slices(), s.items(), s.kt), (TILES, 4, 576, 768));
        assert_eq!(s.packed_bytes(), 2304 * 3072 * 2);
    }

    #[test]
    fn the_finisher_items_cover_every_row_once() {
        for rows in 1..=MAX_ROWS {
            let n = finisher_items(rows);
            let covered: usize = (0..n)
                .map(|j| ((j + 1) * FINISH_ROWS).min(rows) - j * FINISH_ROWS)
                .sum();
            assert_eq!(covered, rows, "{rows} rows");
            assert!(
                n * FINISH_ROWS >= rows && (n - 1) * FINISH_ROWS < rows,
                "{rows} rows: the last item is not empty"
            );
        }
        assert_eq!(
            (
                finisher_items(1),
                finisher_items(2),
                finisher_items(3),
                finisher_items(512)
            ),
            (1, 1, 2, 256)
        );
    }

    #[test]
    fn a_task_spreads_the_finisher_from_two_rows_and_keeps_the_scratch() {
        let at = |rows: usize, depth: usize| OutMixShape::new(rows, depth, false, true).unwrap();
        assert_eq!(at(1, 2).items(), 576);
        assert_eq!(at(2, 2).items(), 577);
        assert_eq!(at(16, 3).items(), 576 + 8);
        assert_eq!(at(512, 5).items(), 576 + 256);
        assert_eq!(at(512, 5).with_spread(false).items(), 576);
        assert_eq!(at(1, 2).with_spread(true).items(), 577);
        let d = at(512, 5).descriptor(&OutMixBuffers::default()).unwrap();
        assert_eq!(d.items, 832);
        // The partials and the raw sums are those of the 576 items of the matmul.
        assert_eq!(d.scale_bytes as usize, scratch_floats(512));
    }

    #[test]
    fn the_split_form_has_a_block_form_matmul_task_and_144_tiles_of_convolution() {
        let at = |rows: usize| {
            OutMixShape::new(rows, 3, false, true)
                .unwrap()
                .with_split(rows >= SPLIT_MIN_ROWS)
        };
        assert!(!at(SPLIT_MIN_ROWS - 1).split && at(SPLIT_MIN_ROWS).split);
        assert!(
            !OutMixShape::new(512, 3, false, true).unwrap().split,
            "new keeps the fused op"
        );
        assert_eq!(at(512).items(), 144 + 256);
        assert_eq!(at(64).items(), 144 + 32);
        assert_eq!(at(1).with_split(true).items(), 144 + 1);
        assert!(at(1).with_split(true).spread);
        let b = OutMixBuffers {
            attn: 0x1000,
            packed: 0x2000,
            r: 0x3000,
            x: 0,
            scratch: 0x40000,
            table: 0x6000,
        };
        let s = at(512);
        let d = s.descriptor(&b).unwrap();
        let raw = 0x40000 + 4 * (160 + 576 * 512 * 16) as u64;
        assert_eq!(
            (d.opcode, d.items, d.a, d.b, d.item_bytes),
            (OP_OUT_MIX, 400, raw, 0, 0)
        );
        assert_eq!(d.flags, FLAG_BLOCK_FORM | FLAG_NORM);
        assert_eq!(d.scale_bytes as usize, scratch_floats(512));
        let m = s.matmul_descriptor(&b).unwrap();
        assert_eq!(
            (m.opcode, m.a, m.b, m.c),
            (crate::protocol::OP_MATMUL_BF16, 0x1000, 0x2000, raw)
        );
        assert_eq!(
            (m.flags, m.items, m.n, m.k, m.rows),
            (FLAG_BLOCK_FORM, 18 * 16, 2304, 3072, 512)
        );
        // The raw sums sit inside the scratch that the fused op needs.
        assert!(raw_offset_floats(512) + 2304 * 512 <= scratch_floats(512));
        assert!(at(8).matmul_descriptor(&b).unwrap_err().contains("with_split"));
    }

    #[test]
    fn the_scratch_holds_the_counters_the_partials_and_the_raw_sums() {
        assert_eq!(scratch_floats(1), 160 + 576 * 16 + 2304);
        assert_eq!(scratch_floats(16), 160 + 576 * 16 * 16 + 2304 * 16);
        assert_eq!(out_floats(3), 2 * 3 * 2304 + 3);
        assert_eq!(out_offsets(3), (0, 3 * 2304, 2 * 3 * 2304));
    }

    #[test]
    fn a_shape_that_the_kernel_does_not_cover_is_an_error_that_says_what_to_change() {
        assert!(OutMixShape::new(0, 2, false, true).unwrap_err().contains("1 to 512"));
        assert!(OutMixShape::new(513, 2, false, true).is_err());
        assert!(OutMixShape::new(1, 1, false, true).unwrap_err().contains("2 to 5"));
        assert!(OutMixShape::new(1, 6, false, true).is_err());
        let s = OutMixShape::new(1, 2, true, false).unwrap();
        let none = OutMixBuffers::default();
        assert!(s.descriptor(&none).unwrap_err().contains("FLAG_ACCUMULATE"));
    }

    #[test]
    fn the_descriptor_carries_the_shape_the_flags_and_the_scratch_size() {
        let s = OutMixShape::new(16, 3, true, true).unwrap();
        let b = OutMixBuffers {
            attn: 0x1000,
            packed: 0x2000,
            r: 0x3000,
            x: 0x4000,
            scratch: 0x5000,
            table: 0x6000,
        };
        let d = s.descriptor(&b).unwrap();
        assert_eq!(
            (d.opcode, d.items, d.n, d.k, d.kt, d.group_items, d.layer),
            (OP_OUT_MIX, 576 + 8, 2304, 3072, 768, 4, 3)
        );
        assert_eq!(d.flags, FLAG_ACCUMULATE | FLAG_NORM);
        assert_eq!(
            (d.a, d.b, d.c, d.d, d.e, d.f),
            (0x1000, 0x2000, 0x3000, 0x6000, 0x4000, 0x5000)
        );
        assert_eq!(d.scale_bytes as usize, scratch_floats(16));
        assert_eq!(d.p0, score_scale());
    }

    #[test]
    fn the_table_puts_r_in_the_last_slot_of_the_stack() {
        let t = OutMixTable {
            planes: vec![(0x10, 0x20), (0x30, 0x40)],
            r: 0x50,
            r_rf: 0x60,
            conv_weight: 1,
            conv_state: 2,
            norm_weight: 3,
            out: 4,
            query: 5,
        }
        .table()
        .unwrap();
        assert_eq!(&t.ptr[..3], &[0x10, 0x30, 0x50]);
        assert_eq!(&t.ptr[5..8], &[0x20, 0x40, 0x60]);
        assert_eq!([t.ptr[10], t.ptr[11], t.ptr[12], t.ptr[13], t.ptr[14]], [1, 2, 3, 4, 5]);
        assert!(OutMixTable::default().table().is_err());
    }
}
