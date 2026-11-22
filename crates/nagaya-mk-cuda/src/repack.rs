//! The repack of a dense `bf16` weight into the order that `OP_MATMUL_BF16` reads (milestone 37, design rule 2).
//!
//! The checkpoint stores `W[N, K]` row-major. The matmul cuts the matrix into work items: an item is a tile of
//! `COLUMN_TILE` output columns and a K slice of `kt` values. The repack makes each item one contiguous span of
//! `item_bytes` bytes, and it orders the span so that each lane of a warp loads one 16-byte vector for each MMA.
//! A repack moves values and changes none of them: [`unpack_bf16`] gives back `W` bit for bit.
//!
//! # The layout
//!
//! - Item `i` is the tile `i / slices` and the slice `i % slices`. It starts at `i * item_bytes`.
//! - Inside an item, step `s` is the K values `s * 16` to `s * 16 + 15` of the slice. It takes 512 bytes: one vector
//!   for each lane, in the order of the lanes.
//! - Let `g` be the lane divided by 4 and `t` the lane modulo 4. The vector of a lane holds four 32-bit words. Each
//!   word holds two consecutive K values (`bf16`, the lower K in the low half):
//!   `a0` = `W[g][2t..]`, `a1` = `W[g + 8][2t..]`, `a2` = `W[g][8 + 2t..]` and `a3` = `W[g + 8][8 + 2t..]`. The rows
//!   are the output columns of the tile. This is the A fragment of `mma.sync.m16n8k16` (`docs/engine-abi.md`,
//!   "The facts of the GB10").
//! - The eight consumer warps of a block each run `kt / 128` consecutive steps of an item.

use crate::protocol::MkTaskDescriptor;
#[cfg(doc)]
use crate::protocol::OP_MATMUL_BF16;

/// Output columns in a tile: the rows of the MMA operand A.
pub const COLUMN_TILE: usize = 16;

/// K values in one MMA step.
pub const K_STEP: usize = 16;

/// Consumer warps that split the K slice of an item. The warps of a block join in a fixed tree.
pub const K_WARPS: usize = 8;

/// The tokens of a tile of columns that one pass takes at most: two MMAs of 8 columns.
pub const MAX_ROWS: usize = 16;

/// Bytes of one step of an item: 32 lanes of 16 bytes.
pub const STEP_BYTES: usize = 512;

/// The K slice for each K that the Inkling 2 matmuls use. A K that is not in the table is an error: add its slice
/// here and to the engine identity, because the slice is part of the order of a sum.
pub const KT_TABLE: [(usize, usize); 3] = [(2304, 768), (3072, 768), (1152, 1152)];

/// The K slice of a matmul with `k` values in a row of `W`.
pub fn k_slice(k: usize) -> Result<usize, String> {
    KT_TABLE.iter().find(|(known, _)| *known == k).map(|(_, kt)| *kt).ok_or_else(|| {
        let known: Vec<String> = KT_TABLE.iter().map(|(k, kt)| format!("{k} (slice {kt})")).collect();
        format!(
            "the matmul has no K slice for K = {k}. The slice comes from the shape and it is part of the order of a \
             sum, so add it to KT_TABLE in repack.rs. The known K values are {}",
            known.join(", ")
        )
    })
}

/// The cut of one matmul into work items. It comes from `N` and `K` only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulShape {
    /// Output columns: the rows of `W`.
    pub n: usize,
    /// The values in a row of `W`.
    pub k: usize,
    /// The K values of one work item.
    pub kt: usize,
}

impl MatmulShape {
    /// The cut of `W[n, k]`. `n` must be a multiple of the tile, and `k` must have a slice in [`KT_TABLE`].
    pub fn new(n: usize, k: usize) -> Result<Self, String> {
        let kt = k_slice(k)?;
        if n == 0 || !n.is_multiple_of(COLUMN_TILE) {
            return Err(format!(
                "a matmul with N = {n} output columns does not fill the tiles of {COLUMN_TILE} columns. \
                 Pad the weight to a multiple of {COLUMN_TILE} rows"
            ));
        }
        debug_assert!(k.is_multiple_of(kt) && kt.is_multiple_of(K_WARPS * K_STEP));
        Ok(Self { n, k, kt })
    }

    /// Tiles of columns.
    pub fn tiles(&self) -> usize {
        self.n / COLUMN_TILE
    }

    /// K slices in a row. They are the items of one merge group.
    pub fn slices(&self) -> usize {
        self.k / self.kt
    }

    /// Work items: a tile and a slice each.
    pub fn items(&self) -> usize {
        self.tiles() * self.slices()
    }

    /// Bytes of one item.
    pub fn item_bytes(&self) -> usize {
        COLUMN_TILE * self.kt * 2
    }

    /// Bytes of the whole repacked weight: the bytes of `W`.
    pub fn packed_bytes(&self) -> usize {
        self.items() * self.item_bytes()
    }

    /// MMA steps of one item.
    pub fn steps(&self) -> usize {
        self.kt / K_STEP
    }

    /// Floats of scratch (`f` of the descriptor) that a launch with `rows` tokens needs: one partial of
    /// `rows * COLUMN_TILE` floats for each item, then one counter for each merge group. The counters must be 0 at
    /// the first launch. The op leaves them at 0.
    pub fn scratch_floats(&self, rows: usize) -> usize {
        self.items() * rows * COLUMN_TILE + self.tiles()
    }

    /// The byte offset in the repacked weight of the 32-bit word that holds `W[col][kk]` and `W[col][kk + 1]`, with
    /// `kk` even.
    pub fn word_offset(&self, col: usize, kk: usize) -> usize {
        debug_assert!(col < self.n && kk < self.k && kk.is_multiple_of(2));
        let (tile, row) = (col / COLUMN_TILE, col % COLUMN_TILE);
        let (slice, in_slice) = (kk / self.kt, kk % self.kt);
        let (step, in_step) = (in_slice / K_STEP, in_slice % K_STEP);
        let (g, half) = (row % 8, row / 8);
        let (t, part) = ((in_step / 2) % 4, in_step / 8);
        let lane = g * 4 + t;
        let item = tile * self.slices() + slice;
        item * self.item_bytes() + (step * 32 + lane) * 16 + (part * 2 + half) * 4
    }

    /// The descriptor of one `OP_MATMUL_BF16` task. The pointers are device addresses: `x` holds `rows` rows of `k`
    /// floats, `packed` is the repacked weight, `y` receives `rows` rows of `n` floats, and `scratch` holds
    /// `scratch_floats(rows)` floats with the counters at 0. The builder of the ring sets the waits and the signal.
    pub fn descriptor(
        &self,
        rows: usize,
        x: u64,
        packed: u64,
        y: u64,
        scratch: u64,
    ) -> Result<MkTaskDescriptor, String> {
        if rows == 0 || rows > MAX_ROWS {
            return Err(format!(
                "a matmul takes from 1 to {MAX_ROWS} rows, and this one has {rows}. Split the rows over several tasks"
            ));
        }
        for (name, address, align) in [
            ("x", x, 8),
            ("the repacked weight", packed, 16),
            ("y", y, 4),
            ("scratch", scratch, 4),
        ] {
            if !address.is_multiple_of(align) {
                return Err(format!(
                    "the address of {name} is {address:#x}, which is not a multiple of {align}. Align the buffer"
                ));
            }
        }
        let count = |what: &str, value: usize| {
            u32::try_from(value).map_err(|_| format!("the {what} of the matmul, {value}, does not fit in 32 bits"))
        };
        Ok(MkTaskDescriptor {
            opcode: crate::protocol::OP_MATMUL_BF16,
            items: count("count of items", self.items())?,
            rows: rows as u32,
            a: x,
            b: packed,
            c: y,
            f: scratch,
            n: count("N", self.n)?,
            k: count("K", self.k)?,
            kt: count("slice", self.kt)?,
            group_items: count("count of slices", self.slices())?,
            item_bytes: count("item size", self.item_bytes())?,
            ..Default::default()
        })
    }
}

fn check_len(what: &str, bytes: &[u8], shape: &MatmulShape) -> Result<(), String> {
    if bytes.len() == shape.n * shape.k * 2 {
        Ok(())
    } else {
        Err(format!(
            "{what} has {} bytes, and a bf16 matrix of {} x {} needs {} bytes. Check the shape of the tensor",
            bytes.len(),
            shape.n,
            shape.k,
            shape.n * shape.k * 2
        ))
    }
}

/// Repack the row-major `bf16` bytes of `W[n, k]` into the order of the matmul.
pub fn repack_bf16(w: &[u8], shape: &MatmulShape) -> Result<Vec<u8>, String> {
    check_len("the weight", w, shape)?;
    let mut packed = vec![0u8; shape.packed_bytes()];
    for_each_word(shape, |dst, src| packed[dst..dst + 4].copy_from_slice(&w[src..src + 4]));
    Ok(packed)
}

/// The inverse of [`repack_bf16`]: the row-major bytes of `W`.
pub fn unpack_bf16(packed: &[u8], shape: &MatmulShape) -> Result<Vec<u8>, String> {
    if packed.len() != shape.packed_bytes() {
        return Err(format!(
            "the repacked weight has {} bytes, and the shape needs {}. Check the shape",
            packed.len(),
            shape.packed_bytes()
        ));
    }
    let mut w = vec![0u8; shape.n * shape.k * 2];
    for_each_word(shape, |dst, src| w[src..src + 4].copy_from_slice(&packed[dst..dst + 4]));
    Ok(w)
}

/// Call `f(packed_offset, row_major_offset)` for each 32-bit word, in the order of the packed bytes.
fn for_each_word(shape: &MatmulShape, mut f: impl FnMut(usize, usize)) {
    let slices = shape.slices();
    let mut dst = 0;
    for tile in 0..shape.tiles() {
        for slice in 0..slices {
            for step in 0..shape.steps() {
                for lane in 0..32 {
                    let (g, t) = (lane / 4, lane % 4);
                    for word in 0..4 {
                        let (part, half) = (word / 2, word % 2);
                        let col = tile * COLUMN_TILE + g + 8 * half;
                        let kk = slice * shape.kt + step * K_STEP + part * 8 + 2 * t;
                        f(dst, (col * shape.k + kk) * 2);
                        dst += 4;
                    }
                }
            }
        }
    }
    debug_assert_eq!(dst, shape.packed_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix in which each bf16 value is its own index, so that a moved value shows.
    fn matrix(n: usize, k: usize) -> Vec<u8> {
        (0..n * k)
            .flat_map(|i| {
                let h = (i as u32).wrapping_mul(0x9E37_79B1);
                ((h >> 13) as u16 ^ (i as u16)).to_le_bytes()
            })
            .collect()
    }

    fn word(w: &[u8], shape: &MatmulShape, col: usize, kk: usize) -> [u8; 4] {
        let at = (col * shape.k + kk) * 2;
        w[at..at + 4].try_into().unwrap()
    }

    #[test]
    fn the_k_slice_comes_from_the_table_and_an_unknown_k_is_refused() {
        assert_eq!(k_slice(2304).unwrap(), 768);
        assert_eq!(k_slice(3072).unwrap(), 768);
        assert_eq!(k_slice(1152).unwrap(), 1152);
        let error = k_slice(2000).unwrap_err();
        assert!(error.contains("K = 2000") && error.contains("KT_TABLE"), "{error}");
        for (k, kt) in KT_TABLE {
            assert!(k % kt == 0 && kt % (K_WARPS * K_STEP) == 0, "K {k} slice {kt}");
        }
    }

    #[test]
    fn the_shape_gives_the_items_of_the_four_gate_matrices() {
        let q = MatmulShape::new(3072, 2304).unwrap();
        assert_eq!((q.tiles(), q.slices(), q.items(), q.item_bytes()), (192, 3, 576, 24576));
        let o = MatmulShape::new(2304, 3072).unwrap();
        assert_eq!((o.tiles(), o.slices(), o.items()), (144, 4, 576));
        let head = MatmulShape::new(201_024, 2304).unwrap();
        assert_eq!((head.tiles(), head.items()), (12_564, 37_692));
        assert_eq!(head.packed_bytes(), 201_024 * 2304 * 2);
        assert_eq!(head.scratch_floats(1), 37_692 * 16 + 12_564);
        let error = MatmulShape::new(100, 2304).unwrap_err();
        assert!(error.contains("N = 100") && error.contains("Pad"), "{error}");
    }

    #[test]
    fn a_lane_vector_holds_the_words_of_the_a_fragment_of_the_mma() {
        // The fragment as the binding defines it: a0 = row g word t, a1 = row g + 8 word t, a2 = row g word t + 4,
        // a3 = row g + 8 word t + 4. Rows are output columns, and a word is two consecutive K values.
        let shape = MatmulShape::new(32, 2304).unwrap();
        let w = matrix(shape.n, shape.k);
        let packed = repack_bf16(&w, &shape).unwrap();
        for (tile, slice, step, lane) in [
            (0, 0, 0, 0),
            (1, 2, 47, 31),
            (0, 1, 5, 13),
            (1, 0, 20, 4),
            (1, 1, 33, 22),
        ] {
            let (g, t) = (lane / 4, lane % 4);
            let item = tile * shape.slices() + slice;
            let at = item * shape.item_bytes() + (step * 32 + lane) * 16;
            let k0 = slice * shape.kt + step * 16;
            let row = |r: usize| tile * 16 + r;
            let want = [
                word(&w, &shape, row(g), k0 + 2 * t),
                word(&w, &shape, row(g + 8), k0 + 2 * t),
                word(&w, &shape, row(g), k0 + 8 + 2 * t),
                word(&w, &shape, row(g + 8), k0 + 8 + 2 * t),
            ];
            for (i, want) in want.iter().enumerate() {
                assert_eq!(
                    &packed[at + i * 4..at + i * 4 + 4],
                    want,
                    "tile {tile} slice {slice} step {step} lane {lane} a{i}"
                );
            }
        }
    }

    #[test]
    fn an_item_is_one_contiguous_span() {
        let shape = MatmulShape::new(32, 3072).unwrap();
        let w = matrix(shape.n, shape.k);
        let packed = repack_bf16(&w, &shape).unwrap();
        // The values of one item, as a multiset, are those of the item span and no others.
        let (tile, slice) = (1, 2);
        let item = tile * shape.slices() + slice;
        let mut span: Vec<[u8; 2]> = packed[item * shape.item_bytes()..(item + 1) * shape.item_bytes()]
            .chunks(2)
            .map(|c| [c[0], c[1]])
            .collect();
        let mut want: Vec<[u8; 2]> = (0..16)
            .flat_map(|r| {
                let at = ((tile * 16 + r) * shape.k + slice * shape.kt) * 2;
                w[at..at + shape.kt * 2]
                    .chunks(2)
                    .map(|c| [c[0], c[1]])
                    .collect::<Vec<_>>()
            })
            .collect();
        span.sort_unstable();
        want.sort_unstable();
        assert_eq!(span, want);
    }

    #[test]
    fn the_inverse_gives_back_each_byte_of_w() {
        for (n, k) in [(16, 1152), (48, 2304), (32, 3072)] {
            let shape = MatmulShape::new(n, k).unwrap();
            let w = matrix(n, k);
            let packed = repack_bf16(&w, &shape).unwrap();
            assert_ne!(packed, w);
            assert_eq!(packed.len(), w.len());
            assert_eq!(unpack_bf16(&packed, &shape).unwrap(), w, "N {n} K {k}");
        }
    }

    #[test]
    fn word_offset_names_the_word_that_the_repack_wrote() {
        let shape = MatmulShape::new(48, 2304).unwrap();
        let w = matrix(shape.n, shape.k);
        let packed = repack_bf16(&w, &shape).unwrap();
        for col in (0..48).step_by(5) {
            for kk in (0..2304).step_by(58) {
                let at = shape.word_offset(col, kk);
                assert_eq!(packed[at..at + 4], word(&w, &shape, col, kk), "col {col} kk {kk}");
            }
        }
    }

    #[test]
    fn a_wrong_length_is_refused_with_the_sizes() {
        let shape = MatmulShape::new(16, 2304).unwrap();
        let error = repack_bf16(&[0u8; 10], &shape).unwrap_err();
        assert!(error.contains("10 bytes") && error.contains("73728"), "{error}");
        assert!(unpack_bf16(&[0u8; 10], &shape).is_err());
    }

    #[test]
    fn the_descriptor_follows_the_design_table_and_refuses_bad_rows_and_addresses() {
        let shape = MatmulShape::new(1024, 2304).unwrap();
        let d = shape.descriptor(5, 0x1000, 0x2000, 0x3000, 0x4000).unwrap();
        assert_eq!(
            (d.items, d.rows, d.n, d.k, d.kt, d.group_items, d.item_bytes),
            (192, 5, 1024, 2304, 768, 3, 24576)
        );
        assert_eq!((d.a, d.b, d.c, d.f), (0x1000, 0x2000, 0x3000, 0x4000));
        assert!(shape.descriptor(0, 0x1000, 0x2000, 0x3000, 0x4000).is_err());
        assert!(shape.descriptor(17, 0x1000, 0x2000, 0x3000, 0x4000).is_err());
        let error = shape.descriptor(1, 0x1000, 0x2008, 0x3000, 0x4000).unwrap_err();
        assert!(error.contains("repacked weight") && error.contains("16"), "{error}");
    }
}
