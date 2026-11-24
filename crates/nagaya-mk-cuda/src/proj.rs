//! The joined projection of the attention input (`OP_PROJ`, milestone 37 wave B lane D).
//!
//! One matrix `J` holds the six or eight matrices that read the normed input of an attention layer, so that one task
//! reads them. The rows of `J`, in order, with `K = 2304`:
//!
//! | segment | rows | tiles | source |
//! |---|---:|---:|---|
//! | `taps_k` | 32 | 2 | `k_sconv.weight_proj`, a full layer only |
//! | `k` | 1,024 | 64 | `wk_dv` |
//! | `taps_v` | 32 | 2 | `v_sconv.weight_proj`, a full layer only |
//! | `v` | 1,024 | 64 | `wv_dv` |
//! | `q` | 3,072 | 192 | `wq_du` |
//! | `rel` | 384 | 24 | `wr_du` |
//! | `sink` | 32 | 2 | `attn_sink_proj`, 24 rows, then 8 rows of zero |
//! | `gate` | 3,072 | 192 | `gate_attn` |
//!
//! A sliding layer has 8,608 rows (538 tiles), and a full layer has 8,672 rows (542 tiles). The join moves rows and
//! changes no value: [`split`] gives the parts back bit for bit. The tiles run in this order, so the heads of the key and
//! the value, which have a finisher, finish early, and the last tiles are the gate, which has the cheapest epilogue. The
//! taps come before their matrix, so that the merge of the taps does not wait for the last tile of a head.

use crate::attn::CacheFormat;
use crate::protocol::{FLAG_FULL_LAYER, FLAG_NORM, MkTaskDescriptor, OP_PROJ};
use crate::repack::{COLUMN_TILE, MatmulShape};

/// The values in a row of `J`: the width of the residual stream.
pub const K: usize = 2304;

/// The sinks that a layer uses: one for each query head. The tile holds 32 rows.
pub const SINKS: usize = 24;

/// The segments of `J`, in the order of its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    TapsK,
    K,
    TapsV,
    V,
    Q,
    Rel,
    Sink,
    Gate,
}

/// The rows of each segment.
const ROWS: [(Segment, usize); 8] = [
    (Segment::TapsK, 32),
    (Segment::K, 1024),
    (Segment::TapsV, 32),
    (Segment::V, 1024),
    (Segment::Q, 3072),
    (Segment::Rel, 384),
    (Segment::Sink, 32),
    (Segment::Gate, 3072),
];

/// The layout of `J` for a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjLayout {
    /// A full layer has the dynamic taps and the log scale.
    pub full: bool,
}

impl ProjLayout {
    /// The rows of a segment: 0 for the taps of a sliding layer.
    pub fn rows_of(&self, segment: Segment) -> usize {
        match segment {
            Segment::TapsK | Segment::TapsV if !self.full => 0,
            _ => ROWS.iter().find(|(s, _)| *s == segment).expect("a segment of J").1,
        }
    }

    /// The first row of a segment.
    pub fn first_row(&self, segment: Segment) -> usize {
        ROWS.iter()
            .take_while(|(s, _)| *s != segment)
            .map(|(s, _)| self.rows_of(*s))
            .sum()
    }

    /// The rows of `J`.
    pub fn rows(&self) -> usize {
        ROWS.iter().map(|(s, _)| self.rows_of(*s)).sum()
    }

    /// The matmul shape of `J`.
    pub fn shape(&self) -> MatmulShape {
        MatmulShape::new(self.rows(), K).expect("J fills its tiles")
    }

    /// The floats of scratch that a task of `rows` rows needs: the counters, the partials, the raw rows of the key
    /// and the value, and their taps. It is the number that the kernel checks against `scale_bytes`.
    pub fn scratch_floats(&self, rows: usize) -> usize {
        let shape = self.shape();
        shape.tiles() + PROJ_HEAD_COUNTERS + shape.items() * rows * COLUMN_TILE + 2 * rows * 1024 + 2 * rows * 32
    }

    /// The groups of the block form, in the order of the items: a group is up to 8 tiles of one segment (see
    /// `kernels/ops_dense.cuh`, "OP_PROJ in the block form").
    pub fn block_groups(&self) -> usize {
        (if self.full { 2 } else { 0 }) + 8 + 8 + 24 + 3 + 1 + 24
    }

    /// The items of the block form for `rows` rows: the groups times the blocks of 32 rows.
    pub fn block_items(&self, rows: usize) -> usize {
        self.block_groups() * rows.div_ceil(crate::repack::BLOCK_ROWS)
    }

    /// The floats of scratch of the block form: one word for each tile (not used), the counters of the heads, the raw rows
    /// of the key and the value, and their taps. It has no partials.
    pub fn block_scratch_floats(&self, rows: usize) -> usize {
        self.shape().tiles() + PROJ_HEAD_COUNTERS + 2 * rows * 1024 + 2 * rows * 32
    }

    /// The descriptor of one `OP_PROJ` task in the block form, for 1 to 512 rows. The arguments are those of
    /// [`ProjLayout::descriptor`], and `scratch` holds [`ProjLayout::block_scratch_floats`] floats with the counters at 0.
    /// The address of `x`, and of each row of the task table that is an input row, must be a multiple of 16: the form
    /// copies rows with 16-byte pieces.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor_block(
        &self,
        rows: usize,
        pos0: u32,
        plane_rows: u32,
        format: CacheFormat,
        x: u64,
        packed: u64,
        out: u64,
        table: u64,
        rf: u64,
        scratch: u64,
    ) -> Result<MkTaskDescriptor, String> {
        let mut desc = self.descriptor(rows, pos0, plane_rows, format, x, packed, out, table, rf, scratch)?;
        if !x.is_multiple_of(16) {
            return Err(format!(
                "the address of x is {x:#x}, which is not a multiple of 16, and the block form copies rows of x in pieces of 16 bytes. Align the buffer"
            ));
        }
        desc.flags |= crate::protocol::FLAG_BLOCK_FORM;
        desc.items = u32::try_from(self.block_items(rows))
            .map_err(|_| "the count of items of the block form does not fit in 32 bits".to_string())?;
        desc.scale_bytes = u32::try_from(self.block_scratch_floats(rows))
            .map_err(|_| "the scratch of the projection does not fit in 32 bits".to_string())?;
        Ok(desc)
    }

    /// The floats of the output at `c`: q, rel, sink and gate, for `rows` rows.
    pub fn output_floats(rows: usize) -> usize {
        rows * (3072 + 384 + SINKS + 3072)
    }

    /// The descriptor of one `OP_PROJ` task. The pointers are device addresses: `x` is the first input, `packed` is the
    /// repacked `J`, `out` receives q, rel, sink and gate (see [`ProjLayout::output_floats`]), `table` is the task
    /// table, `rf` holds the row factors of the first input, and `scratch` holds `scratch_floats(rows)` floats, the
    /// counters at 0. `pos0` is the position of the first row. `plane_rows` is the rows of a plane of the cache and
    /// `format` is its format (see `attn.rs`): the task table holds the two planes, and the head finisher writes the
    /// rows of the key and the value there.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor(
        &self,
        rows: usize,
        pos0: u32,
        plane_rows: u32,
        format: CacheFormat,
        x: u64,
        packed: u64,
        out: u64,
        table: u64,
        rf: u64,
        scratch: u64,
    ) -> Result<MkTaskDescriptor, String> {
        let shape = self.shape();
        let mut desc = shape.descriptor(rows, x, packed, out, scratch)?;
        desc.opcode = OP_PROJ;
        desc.flags = FLAG_NORM | format.flags() | if self.full { FLAG_FULL_LAYER } else { 0 };
        desc.d = table;
        desc.e = rf;
        desc.pos0 = pos0;
        desc.layer = plane_rows;
        desc.scale_bytes = u32::try_from(self.scratch_floats(rows))
            .map_err(|_| "the scratch of the projection does not fit in 32 bits".to_string())?;
        Ok(desc)
    }
}

/// The counters of the heads that follow the counters of the tiles in the scratch: 24 of q, 8 of k, 8 of v, and spare.
pub const PROJ_HEAD_COUNTERS: usize = 64;

/// Join the matrices into `J`. Each part is the `bf16` bytes of its matrix, row-major, in the order of
/// [`Segment`]. The sink part has 24 rows, and it gets 8 rows of zero. A sliding layer passes `None` for the taps.
pub fn join(layout: &ProjLayout, parts: &[(Segment, &[u8])]) -> Result<Vec<u8>, String> {
    if let Some((segment, _)) = parts.iter().find(|(s, _)| layout.rows_of(*s) == 0) {
        return Err(format!("a sliding layer has no part for {segment:?}. Leave it out"));
    }
    let mut out = Vec::with_capacity(layout.rows() * K * 2);
    for (segment, rows) in ROWS {
        if layout.rows_of(segment) == 0 {
            continue;
        }
        let Some((_, bytes)) = parts.iter().find(|(s, _)| *s == segment) else {
            return Err(format!(
                "the projection of this layer has no part for {segment:?}. Give its matrix"
            ));
        };
        // The sink matrix has 24 rows, and the tile holds 32: the last 8 rows are zero.
        let have = if segment == Segment::Sink { SINKS } else { rows };
        if bytes.len() != have * K * 2 {
            return Err(format!(
                "the part {segment:?} has {} bytes, and J needs {} bytes ({have} rows of {K} values) for it. Check the tensor",
                bytes.len(),
                have * K * 2
            ));
        }
        out.extend_from_slice(bytes);
        out.resize(out.len() + (rows - have) * K * 2, 0);
    }
    Ok(out)
}

/// Give the parts back: the inverse of [`join`]. The sink part has its 24 rows.
pub fn split(layout: &ProjLayout, joined: &[u8]) -> Result<Vec<(Segment, Vec<u8>)>, String> {
    if joined.len() != layout.rows() * K * 2 {
        return Err(format!(
            "J has {} bytes, and this layer needs {}. Check the layer kind",
            joined.len(),
            layout.rows() * K * 2
        ));
    }
    let mut parts = Vec::new();
    for (segment, _) in ROWS {
        let rows = layout.rows_of(segment);
        if rows == 0 {
            continue;
        }
        let first = layout.first_row(segment) * K * 2;
        let used = if segment == Segment::Sink { SINKS } else { rows };
        parts.push((segment, joined[first..first + used * K * 2].to_vec()));
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FLAG_F32;

    fn part(rows: usize, seed: u8) -> Vec<u8> {
        (0..rows * K * 2)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn parts(layout: &ProjLayout) -> Vec<(Segment, Vec<u8>)> {
        ROWS.iter()
            .filter(|(s, _)| layout.rows_of(*s) != 0)
            .enumerate()
            .map(|(i, (s, _))| {
                let rows = if *s == Segment::Sink { SINKS } else { layout.rows_of(*s) };
                (*s, part(rows, i as u8))
            })
            .collect()
    }

    #[test]
    fn the_rows_and_the_tiles_are_those_of_the_design() {
        let sliding = ProjLayout { full: false };
        let full = ProjLayout { full: true };
        assert_eq!(
            (sliding.rows(), sliding.shape().tiles(), sliding.shape().items()),
            (8608, 538, 1614)
        );
        assert_eq!((full.rows(), full.shape().tiles()), (8672, 542));
        assert_eq!(sliding.first_row(Segment::K), 0);
        assert_eq!(sliding.first_row(Segment::V), 1024);
        assert_eq!(sliding.first_row(Segment::Q), 2048);
        assert_eq!(sliding.first_row(Segment::Rel), 5120);
        assert_eq!(sliding.first_row(Segment::Sink), 5504);
        assert_eq!(sliding.first_row(Segment::Gate), 5536);
        assert_eq!(full.first_row(Segment::K), 32);
        assert_eq!(full.first_row(Segment::TapsV), 32 + 1024);
        assert_eq!(full.first_row(Segment::V), 32 + 1024 + 32);
        assert_eq!(full.first_row(Segment::Q), 2112);
        assert_eq!(full.first_row(Segment::Gate), 5600);
    }

    #[test]
    fn the_inverse_of_the_join_gives_each_part_back() {
        for full in [false, true] {
            let layout = ProjLayout { full };
            let given = parts(&layout);
            let refs: Vec<(Segment, &[u8])> = given.iter().map(|(s, b)| (*s, b.as_slice())).collect();
            let joined = join(&layout, &refs).unwrap();
            assert_eq!(joined.len(), layout.rows() * K * 2);
            assert_eq!(split(&layout, &joined).unwrap(), given, "full = {full}");
            // The padded rows of the sink are zero.
            let at = (layout.first_row(Segment::Sink) + SINKS) * K * 2;
            assert!(joined[at..at + 8 * K * 2].iter().all(|b| *b == 0));
        }
    }

    #[test]
    fn a_wrong_part_is_an_error_that_names_it() {
        let layout = ProjLayout { full: false };
        let given = parts(&layout);
        let mut refs: Vec<(Segment, &[u8])> = given.iter().map(|(s, b)| (*s, b.as_slice())).collect();
        let short = vec![0u8; 10];
        let at = refs.iter().position(|(s, _)| *s == Segment::Rel).unwrap();
        refs[at] = (Segment::Rel, &short);
        let error = join(&layout, &refs).unwrap_err();
        assert!(error.contains("Rel") && error.contains("10 bytes"), "{error}");
        let full_parts = parts(&ProjLayout { full: true });
        let refs: Vec<(Segment, &[u8])> = full_parts.iter().map(|(s, b)| (*s, b.as_slice())).collect();
        assert!(join(&layout, &refs).unwrap_err().contains("TapsK"));
        let missing = join(&ProjLayout { full: true }, &refs[2..]).unwrap_err();
        assert!(missing.contains("TapsK") || missing.contains("K"), "{missing}");
    }

    #[test]
    fn the_descriptor_carries_the_shape_the_flags_and_the_scratch_size() {
        let layout = ProjLayout { full: true };
        let d = layout
            .descriptor(
                16,
                100,
                3072,
                CacheFormat::F32,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                0x5000,
                0x6000,
            )
            .unwrap();
        assert_eq!(
            (d.opcode, d.items, d.n, d.k, d.kt, d.group_items),
            (OP_PROJ, 542 * 3, 8672, 2304, 768, 3)
        );
        assert_eq!(d.flags, FLAG_NORM | FLAG_FULL_LAYER | FLAG_F32);
        assert_eq!((d.d, d.e, d.pos0, d.layer), (0x4000, 0x5000, 100, 3072));
        assert_eq!(d.scale_bytes as usize, layout.scratch_floats(16));
        assert_eq!(
            ProjLayout { full: false }
                .descriptor(
                    1,
                    0,
                    64,
                    CacheFormat::E4m3,
                    0x1000,
                    0x2000,
                    0x3000,
                    0x4000,
                    0x5000,
                    0x6000
                )
                .unwrap()
                .flags,
            FLAG_NORM
        );
    }
}
