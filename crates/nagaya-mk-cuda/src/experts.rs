//! The layout of the expert weights in the bank and of the quantized rows, for the ops `OP_EXPERT_NVFP4` and
//! `OP_EXPERT_MXFP8` (`kernels/experts.cuh`). It is pure logic: it needs no GPU.
//!
//! A checkpoint holds the weight of an expert as rows: `n` rows of `k` values, with one scale for each 16 values
//! (NVFP4) or each 32 values (MXFP8). The op wants the bytes in the order that the lanes of a warp load them. The
//! repack moves bytes and changes no value.
//!
//! The weight is operand A of the MMA, so 16 rows make one tile. A group is 128 values of K: 2 MMAs of `m16n8k64` for
//! NVFP4, and 4 MMAs of `m16n8k32` for MXFP8. The record of a group and a tile of 16 rows holds:
//! - the fragments of A for each MMA: 512 bytes, with 16 bytes for each lane. For the lane `4g + t`, the 4 words are
//!   word `t` of the row `g`, word `t` of the row `g + 8`, word `t + 4` of the row `g`, and word `t + 4` of the row
//!   `g + 8`. A word holds 4 bytes of the row of one MMA, which is a step of K of 32 bytes.
//! - the scales in the order of the scale fragment: the chunk `2g + u` of the lanes `4g + u`, for `u` of 0 and 1,
//!   holds the scale bytes of the row `g + 8u` for the whole group (8 bytes for NVFP4, 4 bytes for MXFP8).
//!
//! An item is a column tile of 64 rows. It holds 4 tiles of 16 rows. Each tile has its records of all groups in
//! order, so that a warp reads one stream. The bank holds the items of expert 0, then expert 1, and so on. An expert
//! has `n / 64` items in the order of the tile.

use crate::protocol::{
    EXPERT_K_GROUP, EXPERT_MXFP8_RECORD_BYTES, EXPERT_NVFP4_RECORD_BYTES, EXPERT_PAIR_HEADER_BYTES, EXPERT_TILE_ROWS,
    ExpertPair, ExpertPairHeader, TOP_K, expert_pair_bound,
};

/// The bytes of a pair list for the device: the header with the live count `pairs.len()`, the live pairs, and zero pairs
/// up to `capacity` pairs. The task needs room for `capacity` pairs, because its count of items is the bound
/// `capacity * tiles`. An error says that the live pairs do not fit.
pub fn pair_list_bytes(pairs: &[ExpertPair], capacity: usize) -> Result<Vec<u8>, String> {
    if pairs.len() > capacity {
        return Err(format!(
            "the list has {} live pairs, and the bound of the task is {capacity}. Raise the bound, or route fewer pairs",
            pairs.len()
        ));
    }
    let header = ExpertPairHeader {
        pairs: pairs.len() as u32,
        // The word 0 of `reserved` is the count of experts that have a pair (the arrivals of the block form of the down op).
        reserved: [
            pairs
                .iter()
                .map(|p| p.expert)
                .collect::<std::collections::BTreeSet<_>>()
                .len() as u32,
            0,
            0,
        ],
    };
    let mut out = Vec::with_capacity(EXPERT_PAIR_HEADER_BYTES + capacity * std::mem::size_of::<ExpertPair>());
    out.extend_from_slice(&header.pairs.to_le_bytes());
    out.extend(header.reserved.iter().flat_map(|word| word.to_le_bytes()));
    for pair in pairs
        .iter()
        .copied()
        .chain(std::iter::repeat(ExpertPair::default()))
        .take(capacity)
    {
        out.extend_from_slice(&pair.expert.to_le_bytes());
        out.extend_from_slice(&pair.count.to_le_bytes());
        out.extend(pair.x_row.iter().flat_map(|row| row.to_le_bytes()));
        out.extend(pair.y_row.iter().flat_map(|row| row.to_le_bytes()));
        out.extend(pair.weight.iter().flat_map(|weight| weight.to_le_bytes()));
    }
    debug_assert_eq!(
        out.len(),
        EXPERT_PAIR_HEADER_BYTES + capacity * std::mem::size_of::<ExpertPair>()
    );
    Ok(out)
}

/// The slot of the pointer to the bias of the router (`f32`, 128 of them) in the task table of `OP_ROUTE`. The slots of
/// the table are the same in `kernels/ops_moe.cuh`.
pub const ROUTE_TABLE_BIAS: usize = 0;

/// The slot of the pointer to the expert index in the table of `OP_ROUTE`: 128 `u32`, one for each expert, with
/// `first_pair | pairs << 16`. A table with 0 here gets no index. The block form of the expert ops reads the same
/// words through [`EXPERT_TABLE_INDEX`].
pub const ROUTE_TABLE_INDEX: usize = 1;

/// The slot of the expert index in the table of an up or a down op in the block form (`FLAG_BLOCK_FORM`).
pub const EXPERT_TABLE_INDEX: usize = 7;

/// The slot of the pointer to the buffer of quantized activation rows in the table of an up op (`FLAG_UP`). The slots of
/// the table are the same in `kernels/experts.cuh`. The slot 0 is `EXPERT_TABLE_AMAX`.
pub const EXPERT_TABLE_QACT: usize = 1;

/// The slots of the table of a down op (`FLAG_DOWN`): the residual, the weights of `mlp_conv`, the state of the
/// convolution, and the output.
pub const EXPERT_TABLE_RESIDUAL: usize = 2;
pub const EXPERT_TABLE_CONV_WEIGHT: usize = 3;
pub const EXPERT_TABLE_CONV_STATE: usize = 4;
pub const EXPERT_TABLE_OUT: usize = 5;

/// The slot of the row factors of the output rows in the table of a down op: `kt` floats, one for each row of the
/// output `x` (the slot [`EXPERT_TABLE_OUT`]). Each is `1 / sqrt(mean(x^2) + 1e-6)`, summed in a fixed order.
pub const EXPERT_TABLE_OUT_RF: usize = 6;

/// The words of the scratch (`f`) of a down op with `n` outputs and `tokens` tokens: one counter for each column tile
/// of 64, the counter of the row finisher, rounded up to a multiple of 4 words; then 2 floats for each token of each
/// tile (the sums of squares of the two halves of the tile). The counters start at 0 and stay 0.
pub fn down_scratch_words(n: usize, tokens: usize) -> usize {
    let tiles = n / 64;
    (tiles + 1).next_multiple_of(4) + tiles * tokens * 2
}

/// The rows that one row tile of `OP_ROUTE` takes.
pub const ROUTE_ROW_TILE: usize = 16;

/// The items of one row tile of `OP_ROUTE`: 8 column tiles of 16 experts, and 3 K slices of 768 values.
pub const ROUTE_ITEMS_PER_ROW_TILE: usize = 24;

/// The most rows that one `OP_ROUTE` takes (`MAX_PLAN_TOKENS`).
pub const ROUTE_MAX_ROWS: usize = 1024;

/// The count of items of `OP_ROUTE` for `rows` rows.
pub fn route_items(rows: usize) -> usize {
    rows.div_ceil(ROUTE_ROW_TILE) * ROUTE_ITEMS_PER_ROW_TILE
}

/// The words of the counters at the start of the scratch of `OP_ROUTE`: a counter for each row tile and column tile,
/// one for each row tile, and one for the plan, for the most rows (`ROUTE_MAX_ROWS`), rounded to 4 words. The logits
/// start after them, at a place that does not depend on the rows.
pub const ROUTE_COUNTER_WORDS: usize = (ROUTE_MAX_ROWS.div_ceil(ROUTE_ROW_TILE) * 9 + 1).next_multiple_of(4);

/// The words of the scratch (`f`) of `OP_ROUTE` for `rows` rows: the counters (`ROUTE_COUNTER_WORDS`), then the logits
/// (128 for each row), then the partials (256 for each item), then the expert index (128 words). The counters start at
/// 0 and stay 0.
pub fn route_scratch_words(rows: usize) -> usize {
    ROUTE_COUNTER_WORDS + rows * 128 + route_items(rows) * ROUTE_ROW_TILE * 16 + ROUTE_INDEX_WORDS
}

/// The words of the expert index at the end of the scratch of `OP_ROUTE` (one for each expert).
pub const ROUTE_INDEX_WORDS: usize = 128;

/// The word of the expert index in the scratch of `OP_ROUTE` for `rows` rows: the last [`ROUTE_INDEX_WORDS`] words.
pub fn route_index_word(rows: usize) -> usize {
    route_scratch_words(rows) - ROUTE_INDEX_WORDS
}

/// The bytes of the pair list of `OP_ROUTE` for `rows` rows: the header and room for the bound of pairs.
pub fn route_pair_list_bytes(rows: usize) -> usize {
    EXPERT_PAIR_HEADER_BYTES + expert_pair_bound(rows) * std::mem::size_of::<ExpertPair>()
}

/// The bytes of the picks of `OP_ROUTE` (`c`): `rows * 8` expert ids, then `rows * 8` route weights.
pub fn route_picks_bytes(rows: usize) -> usize {
    rows * TOP_K * 8
}

/// The two formats of the experts of Inkling 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    /// `e2m1` values with an `e4m3` scale for each 16 values.
    Nvfp4,
    /// `e4m3` values with a `ue8m0` scale for each 32 values.
    Mxfp8,
}

/// The bytes of the codes of one fragment of A: 16 rows by 32 bytes.
const TILE_BYTES: usize = 512;

/// The bytes of one step of K in a row of codes: 64 `e2m1` values or 32 `e4m3` values.
const STEP_BYTES: usize = 32;

impl ExpertFormat {
    /// The MMAs in one group of K.
    pub fn steps(self) -> usize {
        match self {
            Self::Nvfp4 => 2,
            Self::Mxfp8 => 4,
        }
    }

    /// The values that share one scale.
    pub fn scale_block(self) -> usize {
        match self {
            Self::Nvfp4 => 16,
            Self::Mxfp8 => 32,
        }
    }

    /// The bytes of the codes of a row of `k` values.
    pub fn code_bytes(self, k: usize) -> usize {
        match self {
            Self::Nvfp4 => k / 2,
            Self::Mxfp8 => k,
        }
    }

    /// The bytes of the scales of a row of `k` values.
    pub fn scale_bytes(self, k: usize) -> usize {
        k / self.scale_block()
    }

    /// The bytes of one record: the fragments of A of the group, then the scales.
    pub fn record_bytes(self) -> usize {
        match self {
            Self::Nvfp4 => EXPERT_NVFP4_RECORD_BYTES,
            Self::Mxfp8 => EXPERT_MXFP8_RECORD_BYTES,
        }
    }

    /// The scale bytes of one row for one group.
    fn group_scale_bytes(self) -> usize {
        EXPERT_K_GROUP / self.scale_block()
    }

    /// The bytes of an item of `k` values in a row: the column tile of 64 rows, with the scales.
    pub fn item_bytes(self, k: usize) -> usize {
        (EXPERT_TILE_ROWS / 16) * (k / EXPERT_K_GROUP) * self.record_bytes()
    }
}

/// The error for a shape that the layout cannot hold.
fn check_shape(n: usize, k: usize) -> Result<(), String> {
    if n == 0 || !n.is_multiple_of(EXPERT_TILE_ROWS) {
        return Err(format!(
            "an expert of {n} rows is not a whole number of column tiles of {EXPERT_TILE_ROWS} rows; \
             give a row count that is a multiple of {EXPERT_TILE_ROWS}"
        ));
    }
    if k == 0 || !k.is_multiple_of(EXPERT_K_GROUP) {
        return Err(format!(
            "an expert row of {k} values is not a whole number of groups of {EXPERT_K_GROUP} values; \
             give a K that is a multiple of {EXPERT_K_GROUP}"
        ));
    }
    Ok(())
}

fn check_len(what: &str, got: usize, want: usize) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!(
            "{what} holds {got} bytes, and the shape needs {want}; check the shape of the tensor"
        ))
    }
}

/// The place of a byte in the bank for the loops of the repack. `f` is called with the offset of each piece in the
/// checkpoint (a range of the codes or of the scales) and its offset in the item. The two directions share it.
fn for_each_piece(format: ExpertFormat, k: usize, mut piece: impl FnMut(Source, usize, usize, usize)) {
    let steps = format.steps();
    let groups = k / EXPERT_K_GROUP;
    let code_row = format.code_bytes(k);
    let scale_row = format.scale_bytes(k);
    let spb = format.group_scale_bytes();
    for mt in 0..EXPERT_TILE_ROWS / 16 {
        for g in 0..groups {
            let record = (mt * groups + g) * format.record_bytes();
            for j in 0..steps {
                let step = g * steps + j;
                for lane in 0..32 {
                    let (row_g, word_t) = (lane / 4, lane % 4);
                    for reg in 0..4 {
                        // The registers a0 to a3: the row g, then g + 8, for the word t, then the word t + 4.
                        let row = mt * 16 + row_g + 8 * (reg % 2);
                        let word = word_t + 4 * (reg / 2);
                        let from = row * code_row + step * STEP_BYTES + word * 4;
                        piece(Source::Codes, from, record + j * TILE_BYTES + lane * 16 + reg * 4, 4);
                    }
                }
            }
            for chunk in 0..16 {
                let row = mt * 16 + chunk / 2 + 8 * (chunk % 2);
                let from = row * scale_row + g * spb;
                piece(Source::Scales, from, record + steps * TILE_BYTES + chunk * spb, spb);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Source {
    Codes,
    Scales,
}

/// Repack the weight of one expert from the order of the checkpoint to the order of the bank.
///
/// `codes` holds `n` rows of the codes, and `scales` holds `n` rows of the scales, both as the checkpoint has them. The
/// result holds the `n / 64` items of the expert, one after the other. The repack changes no value: `unpack_expert`
/// gives the checkpoint bytes back.
pub fn repack_expert(format: ExpertFormat, n: usize, k: usize, codes: &[u8], scales: &[u8]) -> Result<Vec<u8>, String> {
    check_shape(n, k)?;
    check_len("the codes", codes.len(), n * format.code_bytes(k))?;
    check_len("the scales", scales.len(), n * format.scale_bytes(k))?;
    let item = format.item_bytes(k);
    let mut out = vec![0u8; item * (n / EXPERT_TILE_ROWS)];
    for tile in 0..n / EXPERT_TILE_ROWS {
        let code_base = tile * EXPERT_TILE_ROWS * format.code_bytes(k);
        let scale_base = tile * EXPERT_TILE_ROWS * format.scale_bytes(k);
        let dest = &mut out[tile * item..(tile + 1) * item];
        for_each_piece(format, k, |source, from, to, len| {
            let src = match source {
                Source::Codes => &codes[code_base + from..code_base + from + len],
                Source::Scales => &scales[scale_base + from..scale_base + from + len],
            };
            dest[to..to + len].copy_from_slice(src);
        });
    }
    Ok(out)
}

/// The checkpoint row of the row `p` of the up matrix after [`repack_expert_up`]. The checkpoint interleaves: row `2 i` is
/// the gate `i` and row `2 i + 1` is the up `i` (`gate_up_source_row` of the B70 crate). A tile of 16 rows of the MMA
/// holds 8 activation columns: its row `g` is the gate and its row `g + 8` is the up of one column, so that one lane holds
/// both values.
pub fn up_source_row(p: usize) -> usize {
    let (tile, row) = (p / 16, p % 16);
    2 * (8 * tile + row % 8) + row / 8
}

/// Repack the up matrix `w13` of one expert for `FLAG_UP`: [`repack_expert`] of the rows in the order of [`up_source_row`].
/// `n` is twice the count of activation columns. The repack changes no value, and the engine identity includes it.
pub fn repack_expert_up(
    format: ExpertFormat,
    n: usize,
    k: usize,
    codes: &[u8],
    scales: &[u8],
) -> Result<Vec<u8>, String> {
    check_shape(n, k)?;
    check_len("the codes", codes.len(), n * format.code_bytes(k))?;
    check_len("the scales", scales.len(), n * format.scale_bytes(k))?;
    let (code_row, scale_row) = (format.code_bytes(k), format.scale_bytes(k));
    let mut moved_codes = Vec::with_capacity(codes.len());
    let mut moved_scales = Vec::with_capacity(scales.len());
    for p in 0..n {
        let from = up_source_row(p);
        moved_codes.extend_from_slice(&codes[from * code_row..(from + 1) * code_row]);
        moved_scales.extend_from_slice(&scales[from * scale_row..(from + 1) * scale_row]);
    }
    repack_expert(format, n, k, &moved_codes, &moved_scales)
}

/// The inverse of [`repack_expert`]: the codes and the scales in the order of the checkpoint.
pub fn unpack_expert(format: ExpertFormat, n: usize, k: usize, packed: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    check_shape(n, k)?;
    let item = format.item_bytes(k);
    check_len("the packed expert", packed.len(), item * (n / EXPERT_TILE_ROWS))?;
    let mut codes = vec![0u8; n * format.code_bytes(k)];
    let mut scales = vec![0u8; n * format.scale_bytes(k)];
    for tile in 0..n / EXPERT_TILE_ROWS {
        let code_base = tile * EXPERT_TILE_ROWS * format.code_bytes(k);
        let scale_base = tile * EXPERT_TILE_ROWS * format.scale_bytes(k);
        let src = &packed[tile * item..(tile + 1) * item];
        for_each_piece(format, k, |source, from, to, len| {
            let dest = match source {
                Source::Codes => &mut codes[code_base + from..code_base + from + len],
                Source::Scales => &mut scales[scale_base + from..scale_base + from + len],
            };
            dest.copy_from_slice(&src[to..to + len]);
        });
    }
    Ok((codes, scales))
}

/// The layout of a buffer of quantized rows, as `kernels/quant.cuh` writes it and the expert ops read it. `rows` is
/// the capacity, and `k` is the length of a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantRowsLayout {
    pub format: ExpertFormat,
    pub rows: usize,
    pub k: usize,
}

impl QuantRowsLayout {
    /// A layout, or the reason that it cannot be one.
    pub fn new(format: ExpertFormat, rows: usize, k: usize) -> Result<Self, String> {
        if rows == 0 || rows > u16::MAX as usize {
            return Err(format!(
                "a buffer of {rows} quantized rows is not from 1 to {}",
                u16::MAX
            ));
        }
        check_shape(EXPERT_TILE_ROWS, k)?;
        Ok(Self { format, rows, k })
    }

    /// The offset of the codes of row `row`.
    pub fn codes_offset(&self, row: usize) -> usize {
        row * self.format.code_bytes(self.k)
    }

    /// The offset of the scales of row `row`.
    pub fn scales_offset(&self, row: usize) -> usize {
        self.rows * self.format.code_bytes(self.k) + row * self.format.scale_bytes(self.k)
    }

    /// The offset of the amax of row `row`. Only NVFP4 has it.
    pub fn amax_offset(&self, row: usize) -> usize {
        assert_eq!(self.format, ExpertFormat::Nvfp4, "only NVFP4 rows have an amax");
        self.rows * (self.format.code_bytes(self.k) + self.format.scale_bytes(self.k)) + row * 4
    }

    /// The bytes of the whole buffer.
    pub fn bytes(&self) -> usize {
        let rows = self.rows * (self.format.code_bytes(self.k) + self.format.scale_bytes(self.k));
        match self.format {
            ExpertFormat::Nvfp4 => rows + self.rows * 4,
            ExpertFormat::Mxfp8 => rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pair_list_has_the_header_the_live_pairs_and_zero_pairs_up_to_the_bound() {
        let mut live = ExpertPair {
            expert: 7,
            count: 2,
            ..Default::default()
        };
        live.x_row[1] = 5;
        live.y_row[1] = 9;
        live.weight[1] = 0.25;
        let bytes = pair_list_bytes(&[live], 3).unwrap();
        let pair = std::mem::size_of::<ExpertPair>();
        assert_eq!(bytes.len(), EXPERT_PAIR_HEADER_BYTES + 3 * pair);
        assert_eq!(&bytes[..4], &1u32.to_le_bytes());
        // The word 1 is the count of experts that have a pair (the block form of the down op); the rest is 0.
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert!(bytes[8..EXPERT_PAIR_HEADER_BYTES].iter().all(|b| *b == 0));
        let first = &bytes[EXPERT_PAIR_HEADER_BYTES..][..pair];
        assert_eq!(&first[..4], &7u32.to_le_bytes());
        assert_eq!(&first[4..8], &2u32.to_le_bytes());
        assert_eq!(&first[8 + 2..8 + 4], &5u16.to_le_bytes());
        assert_eq!(&first[40 + 2..40 + 4], &9u16.to_le_bytes());
        assert_eq!(&first[72 + 4..72 + 8], &0.25f32.to_le_bytes());
        assert!(bytes[EXPERT_PAIR_HEADER_BYTES + pair..].iter().all(|b| *b == 0));
        let error = pair_list_bytes(&[live, live], 1).unwrap_err();
        assert!(error.contains("2 live pairs") && error.contains("bound"), "{error}");
    }

    /// A byte that tells its place: no two nearby places hold the same byte.
    fn pattern(index: usize, salt: usize) -> u8 {
        let x =
            (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        (x >> 33) as u8
    }

    fn make(format: ExpertFormat, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let codes = (0..n * format.code_bytes(k)).map(|i| pattern(i, 1)).collect();
        let scales = (0..n * format.scale_bytes(k)).map(|i| pattern(i, 2)).collect();
        (codes, scales)
    }

    #[test]
    fn the_records_and_items_have_the_sizes_of_the_design() {
        assert_eq!(ExpertFormat::Nvfp4.record_bytes(), 1152);
        assert_eq!(ExpertFormat::Mxfp8.record_bytes(), 2112);
        // The up op of NVFP4: 4 tiles of 18 records. The down op: 4 tiles of 9 records.
        assert_eq!(ExpertFormat::Nvfp4.item_bytes(2304), 82_944);
        assert_eq!(ExpertFormat::Nvfp4.item_bytes(1152), 41_472);
        assert_eq!(ExpertFormat::Mxfp8.item_bytes(2304), 152_064);
        assert_eq!(ExpertFormat::Mxfp8.item_bytes(1152), 76_032);
        // An item holds the codes and the scales of 64 rows, and nothing else.
        for (format, k) in [(ExpertFormat::Nvfp4, 2304), (ExpertFormat::Mxfp8, 1152)] {
            assert_eq!(
                format.item_bytes(k),
                64 * (format.code_bytes(k) + format.scale_bytes(k))
            );
        }
    }

    #[test]
    fn a_repack_changes_no_byte_and_its_inverse_gives_the_checkpoint_back() {
        for (format, n, k) in [
            (ExpertFormat::Nvfp4, 192, 384),
            (ExpertFormat::Mxfp8, 128, 256),
            (ExpertFormat::Nvfp4, 64, 128),
            (ExpertFormat::Mxfp8, 64, 1152),
        ] {
            let (codes, scales) = make(format, n, k);
            let packed = repack_expert(format, n, k, &codes, &scales).unwrap();
            assert_eq!(packed.len(), n * (format.code_bytes(k) + format.scale_bytes(k)));
            let (codes_back, scales_back) = unpack_expert(format, n, k, &packed).unwrap();
            assert_eq!(codes_back, codes, "{format:?} {n}x{k}: the codes");
            assert_eq!(scales_back, scales, "{format:?} {n}x{k}: the scales");
            // The bytes are the same multiset: the repack is a permutation.
            let (mut a, mut b) = (packed.clone(), [codes, scales].concat());
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(
                a, b,
                "{format:?} {n}x{k}: a repack moved a byte that is not a byte of the checkpoint"
            );
        }
    }

    /// The bytes of the lane `lane` in the fragment `step` of the group `g` of the tile `mt` in the item `tile`, found
    /// from the layout in the words of the module: a second route to the same bytes as `for_each_piece`.
    #[test]
    fn a_lane_finds_its_fragment_in_the_order_of_the_mma() {
        let (n, k) = (128, 256);
        for format in [ExpertFormat::Nvfp4, ExpertFormat::Mxfp8] {
            let (codes, scales) = make(format, n, k);
            let packed = repack_expert(format, n, k, &codes, &scales).unwrap();
            let item = format.item_bytes(k);
            let groups = k / 128;
            let (tile, mt, g, step) = (1usize, 2usize, 1usize, 1usize);
            let record = &packed[tile * item + (mt * groups + g) * format.record_bytes()..];
            let code_row = format.code_bytes(k);
            let scale_row = format.scale_bytes(k);
            for lane in [0usize, 5, 13, 30, 31] {
                let (gl, t) = (lane / 4, lane % 4);
                let fragment = &record[step * 512 + lane * 16..step * 512 + lane * 16 + 16];
                let word = |row: usize, w: usize| {
                    let at = (tile * 64 + mt * 16 + row) * code_row + (g * format.steps() + step) * 32 + w * 4;
                    codes[at..at + 4].to_vec()
                };
                assert_eq!(&fragment[0..4], &word(gl, t)[..], "{format:?} lane {lane}: a0");
                assert_eq!(&fragment[4..8], &word(gl + 8, t)[..], "{format:?} lane {lane}: a1");
                assert_eq!(&fragment[8..12], &word(gl, t + 4)[..], "{format:?} lane {lane}: a2");
                assert_eq!(
                    &fragment[12..16],
                    &word(gl + 8, t + 4)[..],
                    "{format:?} lane {lane}: a3"
                );
            }
            // The scales: the lane 4g + u holds the row g + 8u, for the whole group.
            let spb = format.group_scale_bytes();
            for (gl, u) in [(0usize, 0usize), (3, 1), (7, 0), (7, 1)] {
                let at = format.steps() * 512 + (2 * gl + u) * spb;
                let row = tile * 64 + mt * 16 + gl + 8 * u;
                assert_eq!(
                    &record[at..at + spb],
                    &scales[row * scale_row + g * spb..row * scale_row + (g + 1) * spb],
                    "{format:?}: the scales of the row {gl} + 8 * {u}"
                );
            }
        }
    }

    #[test]
    fn the_up_repack_puts_the_gate_and_the_up_of_a_column_in_rows_g_and_g_plus_8() {
        // Each checkpoint row is used once.
        let mut seen = vec![false; 2304];
        for p in 0..2304 {
            let from = up_source_row(p);
            assert!(!seen[from], "the row {from} is used twice");
            seen[from] = true;
        }
        // The first tile: the rows 0 to 7 are the gates 0 to 7 (checkpoint rows 0, 2, ..., 14), the rows 8 to 15 the ups.
        for g in 0..8 {
            assert_eq!(up_source_row(g), 2 * g);
            assert_eq!(up_source_row(g + 8), 2 * g + 1);
        }
        // The second tile starts at the activation column 8, and the second item at the column 32.
        assert_eq!(up_source_row(16), 16);
        assert_eq!(up_source_row(64), 64);
        // The bytes: the repack of the permuted rows equals the unpack of the result with the permutation undone.
        for format in [ExpertFormat::Nvfp4, ExpertFormat::Mxfp8] {
            let (n, k) = (128, 256);
            let (codes, scales) = make(format, n, k);
            let packed = repack_expert_up(format, n, k, &codes, &scales).unwrap();
            let (moved_codes, moved_scales) = unpack_expert(format, n, k, &packed).unwrap();
            let (code_row, scale_row) = (format.code_bytes(k), format.scale_bytes(k));
            for p in 0..n {
                let from = up_source_row(p);
                assert_eq!(
                    &moved_codes[p * code_row..(p + 1) * code_row],
                    &codes[from * code_row..(from + 1) * code_row]
                );
                assert_eq!(
                    &moved_scales[p * scale_row..(p + 1) * scale_row],
                    &scales[from * scale_row..(from + 1) * scale_row]
                );
            }
        }
    }

    #[test]
    fn a_shape_that_the_layout_cannot_hold_is_an_error_with_the_reason() {
        let (codes, scales) = make(ExpertFormat::Nvfp4, 64, 128);
        let error = repack_expert(ExpertFormat::Nvfp4, 70, 128, &codes, &scales).unwrap_err();
        assert!(error.contains("70") && error.contains("64"), "{error}");
        let error = repack_expert(ExpertFormat::Nvfp4, 64, 144, &codes, &scales).unwrap_err();
        assert!(error.contains("144") && error.contains("128"), "{error}");
        let error = repack_expert(ExpertFormat::Nvfp4, 64, 128, &codes[1..], &scales).unwrap_err();
        assert!(error.contains("the codes") && error.contains("holds"), "{error}");
        let error = unpack_expert(ExpertFormat::Mxfp8, 64, 128, &codes).unwrap_err();
        assert!(error.contains("packed expert"), "{error}");
    }

    #[test]
    fn the_quantized_rows_have_the_layout_of_the_quantizer() {
        let nv = QuantRowsLayout::new(ExpertFormat::Nvfp4, 16, 2304).unwrap();
        assert_eq!(nv.codes_offset(3), 3 * 1152);
        assert_eq!(nv.scales_offset(0), 16 * 1152);
        assert_eq!(nv.scales_offset(2), 16 * 1152 + 2 * 144);
        assert_eq!(nv.amax_offset(0), 16 * (1152 + 144));
        assert_eq!(nv.bytes(), 16 * (1152 + 144 + 4));
        let mx = QuantRowsLayout::new(ExpertFormat::Mxfp8, 8, 1152).unwrap();
        assert_eq!(mx.scales_offset(1), 8 * 1152 + 36);
        assert_eq!(mx.bytes(), 8 * (1152 + 36));
        assert!(QuantRowsLayout::new(ExpertFormat::Nvfp4, 0, 128).is_err());
        assert!(QuantRowsLayout::new(ExpertFormat::Nvfp4, 4, 100).is_err());
        // The scales of NVFP4 and of MXFP8 start on a 4-byte boundary, as the loads of the op need.
        assert_eq!(nv.scales_offset(0) % 4, 0);
        assert_eq!(nv.amax_offset(0) % 4, 0);
    }
}
