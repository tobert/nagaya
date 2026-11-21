//! The layout of the expert planes on the card (engine-abi "The first Inkling 2 binding", "The experts on the card").
//!
//! An XMX step reads 16 columns of 16 output rows: each lane of a subgroup holds one row. The load puts the bytes
//! of one step of one row into words, and the words of the 16 lanes side by side. Thus the 16 lanes read 64 bytes
//! in one load, and each lane gets the bytes of its own row.

use std::fmt;

/// The rows of one tile: one row for each lane of a subgroup.
pub const TILE_ROWS: usize = 16;

/// What a repack refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError(String);

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShapeError {}

fn refuse(what: String) -> Result<(), ShapeError> {
    Err(ShapeError(format!(
        "{what}. The expert planes of the checkpoint do not have this shape; check the shape of the tensor."
    )))
}

/// The words of `rows` rows of `row_bytes` bytes, in the order of the card: tile, step, word, lane.
///
/// A step holds `step_bytes` bytes of a row: 8 for NVFP4, 16 for MXFP8. A word is 4 bytes, little-endian.
pub fn repack_codes(src: &[u8], rows: usize, row_bytes: usize, step_bytes: usize) -> Result<Vec<u32>, ShapeError> {
    if rows == 0 || !rows.is_multiple_of(TILE_ROWS) {
        refuse(format!("{rows} rows are not whole tiles of {TILE_ROWS}"))?;
    }
    if step_bytes == 0 || !step_bytes.is_multiple_of(4) {
        refuse(format!("a step of {step_bytes} bytes is not whole words of 4"))?;
    }
    if !row_bytes.is_multiple_of(step_bytes) {
        refuse(format!("a row of {row_bytes} bytes is not whole steps of {step_bytes}"))?;
    }
    if src.len() != rows * row_bytes {
        refuse(format!(
            "the source holds {} bytes, and {rows} rows of {row_bytes} need {}",
            src.len(),
            rows * row_bytes
        ))?;
    }
    let (steps, words) = (row_bytes / step_bytes, step_bytes / 4);
    let mut out = vec![0u32; rows * row_bytes / 4];
    for row in 0..rows {
        let (tile, lane) = (row / TILE_ROWS, row % TILE_ROWS);
        for step in 0..steps {
            for w in 0..words {
                let start = row * row_bytes + step * step_bytes + 4 * w;
                let word = u32::from_le_bytes([src[start], src[start + 1], src[start + 2], src[start + 3]]);
                out[((tile * steps + step) * words + w) * TILE_ROWS + lane] = word;
            }
        }
    }
    Ok(out)
}

/// The scales of `rows` rows of `row_scales` scales, in the order of the card: tile, scale, lane.
pub fn repack_scales(src: &[u8], rows: usize, row_scales: usize) -> Result<Vec<u8>, ShapeError> {
    if rows == 0 || !rows.is_multiple_of(TILE_ROWS) {
        refuse(format!("{rows} rows are not whole tiles of {TILE_ROWS}"))?;
    }
    if src.len() != rows * row_scales {
        refuse(format!(
            "the scales hold {} bytes, and {rows} rows of {row_scales} need {}",
            src.len(),
            rows * row_scales
        ))?;
    }
    let mut out = vec![0u8; rows * row_scales];
    for row in 0..rows {
        let (tile, lane) = (row / TILE_ROWS, row % TILE_ROWS);
        for s in 0..row_scales {
            out[(tile * row_scales + s) * TILE_ROWS + lane] = src[row * row_scales + s];
        }
    }
    Ok(out)
}

/// The checkpoint row of each row of the card for `w13` of `rows` rows, in the order of the card.
///
/// The checkpoint interleaves: row `2i` is gate row `i`, and row `2i + 1` is up row `i`. On the card, tile `2t`
/// holds the gate rows `16t` to `16t + 15`, and tile `2t + 1` holds the up rows of the same `i`. Thus one lane holds
/// the gate and the up row of one `i`.
pub fn gate_up_tile_order(rows: usize) -> Vec<usize> {
    assert!(
        rows.is_multiple_of(2 * TILE_ROWS),
        "gate_up_tile_order: {rows} rows are not whole pairs of tiles of {TILE_ROWS}"
    );
    let mut order = Vec::with_capacity(rows);
    for t in 0..rows / (2 * TILE_ROWS) {
        for part in 0..2 {
            for lane in 0..TILE_ROWS {
                order.push(2 * (t * TILE_ROWS + lane) + part);
            }
        }
    }
    order
}
