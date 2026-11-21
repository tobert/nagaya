//! The image spans of a pass: the records of `meta`, and the rules that tie a span to the tokens and the rows of a
//! segment (engine-abi "The images of Inkling 2", "The choices of the pass"). None of this needs a card.
//!
//! The family writes each record (`nagaya_text_inkling2::media::ImageRows::record`), and this module reads it. A
//! record is little-endian: the kind (`u32`), the count of positions (`u32`), the first position (`u64`), then
//! `grid_t`, `grid_h`, `grid_w` and 0, in patches. The rows of a segment are the patch rows of each span whole, in
//! `f32`, in the order of the records.

use std::ops::Range;

pub use nagaya_text_inkling2::media::{Grid, IMAGE_KIND, IMAGE_TOKEN, PATCH_VALUES, RECORD_BYTES};
use nagaya_text_inkling2::processor::{MAX_PATCH_SIDE, MERGE};

/// One image span of `meta`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The grid of the image in image tokens.
    pub grid: Grid,
    /// The first position of the span.
    pub first: u64,
}

impl Span {
    /// The count of positions: one for each image token.
    pub fn count(&self) -> usize {
        self.grid.tokens()
    }

    /// The positions of the span.
    pub fn positions(&self) -> Range<u64> {
        self.first..self.first + self.count() as u64
    }

    /// The bytes of its patch rows in the rows of the segment.
    pub fn bytes(&self) -> usize {
        self.grid.patches() * PATCH_VALUES * 4
    }
}

/// The spans of `meta`. A record that is not whole, of another kind, or that breaks the shape of an image is an error
/// that names the record.
pub fn decode(meta: &[u8]) -> Result<Vec<Span>, String> {
    if !meta.len().is_multiple_of(RECORD_BYTES) {
        return Err(format!(
            "the meta has {} bytes, not whole records of {RECORD_BYTES} bytes",
            meta.len()
        ));
    }
    let mut spans = Vec::with_capacity(meta.len() / RECORD_BYTES);
    for (i, record) in meta.as_chunks::<RECORD_BYTES>().0.iter().enumerate() {
        let word = |at: usize| u32::from_le_bytes(record[at..at + 4].try_into().expect("4 bytes")) as usize;
        let first = u64::from_le_bytes(record[8..16].try_into().expect("8 bytes"));
        let (kind, count) = (word(0) as u32, word(4));
        let [grid_t, grid_h, grid_w, last] = [word(16), word(20), word(24), word(28)];
        if kind != IMAGE_KIND {
            return Err(format!(
                "media record {i} has the kind {kind}; the Inkling 2 engine reads images only, the kind {IMAGE_KIND}"
            ));
        }
        if grid_t != 1 || last != 0 {
            return Err(format!(
                "media record {i} is an image with grid_t {grid_t} and a fourth word of {last}; an image has \
                 grid_t 1 and a fourth word of 0"
            ));
        }
        let side = 2..=MAX_PATCH_SIDE;
        if !side.contains(&grid_h) || !side.contains(&grid_w) || grid_h % MERGE != 0 || grid_w % MERGE != 0 {
            return Err(format!(
                "media record {i} is an image of {grid_h} by {grid_w} patches; each side is an even count of 2 to \
                 {MAX_PATCH_SIDE} patches, because a token holds 2 by 2 patches and a side has at most 4,096 px"
            ));
        }
        let grid = Grid {
            rows: grid_h / MERGE,
            columns: grid_w / MERGE,
        };
        if count != grid.tokens() {
            return Err(format!(
                "media record {i} is an image of {grid_h} by {grid_w} patches with {count} positions; it has {}",
                grid.tokens()
            ));
        }
        if first.checked_add(count as u64).is_none() {
            return Err(format!(
                "media record {i} starts at position {first}, past the last position"
            ));
        }
        spans.push(Span { grid, first });
    }
    Ok(spans)
}

/// One span that a block reads: the span, where its rows start in the rows of the segment, and the positions of the
/// block that it fills.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanUse {
    pub span: Span,
    /// The first byte of its rows in the rows of the segment.
    pub rows_at: usize,
    /// The positions of the span that the block holds, relative to the first position of the span.
    pub take: Range<usize>,
    /// The row of the block of the first position that it fills.
    pub block_row: usize,
}

/// Check the spans of a block of `tokens` from position `pos0` against its `rows` bytes of rows, and give what the
/// block reads.
///
/// - Each span reads at least one position of the block, and the spans do not overlap.
/// - Each position of a span in the block holds the placeholder. Each placeholder of the block is in a span, so no
///   placeholder enters the model as its embedding row.
/// - The rows are the rows of every span whole, and nothing more.
pub fn check_spans(spans: &[Span], tokens: &[u32], pos0: u64, rows: usize) -> Result<Vec<SpanUse>, String> {
    let end = pos0 + tokens.len() as u64;
    let mut uses = Vec::with_capacity(spans.len());
    let mut covered = vec![false; tokens.len()];
    let mut rows_at = 0usize;
    let mut last_end = 0u64;
    for (i, span) in spans.iter().enumerate() {
        let p = span.positions();
        if i > 0 && p.start < last_end {
            return Err(format!(
                "media record {i} starts at {} inside the span before it, which ends at {last_end}",
                p.start
            ));
        }
        last_end = p.end;
        if p.end <= pos0 || p.start >= end {
            return Err(format!(
                "media record {i} holds the positions {} to {}, and the block holds {pos0} to {end}; a block names \
                 only the spans that it reads",
                p.start, p.end
            ));
        }
        let (lo, hi) = (p.start.max(pos0), p.end.min(end));
        for at in lo..hi {
            let t = (at - pos0) as usize;
            if tokens[t] != IMAGE_TOKEN {
                return Err(format!(
                    "position {at} is in media record {i}, and it holds the token {}, not the image placeholder \
                     {IMAGE_TOKEN}",
                    tokens[t]
                ));
            }
            covered[t] = true;
        }
        uses.push(SpanUse {
            span: *span,
            rows_at,
            take: (lo - p.start) as usize..(hi - p.start) as usize,
            block_row: (lo - pos0) as usize,
        });
        rows_at += span.bytes();
    }
    if rows != rows_at {
        return Err(format!(
            "the segment has {rows} bytes of rows, and its media records need {rows_at}"
        ));
    }
    if let Some(t) = (0..tokens.len()).find(|t| !covered[*t] && tokens[*t] == IMAGE_TOKEN) {
        return Err(format!(
            "position {} holds the image placeholder {IMAGE_TOKEN}, and no media record names it; send the image \
             part, or send the text with no placeholder",
            pos0 + t as u64
        ));
    }
    Ok(uses)
}

/// The image whose projector rows the engine holds, and the count of the runs of the tower.
///
/// An image of more positions than a block is in several passes, and each pass gives the patch rows of the whole
/// image. The tower runs for the first of them. Each later pass copies its rows from the held rows (engine-abi "The
/// choices of the pass", "A span longer than a block").
#[derive(Debug, Default)]
pub struct HeldRows {
    /// The grid and the BLAKE3 hash of the patch rows.
    image: Option<(Grid, blake3::Hash)>,
    runs: u64,
}

impl HeldRows {
    /// True when the held rows are those of the image of `grid` and `rows`. If not, this records the image and counts a
    /// run, and the caller must run the tower before the next call.
    pub fn holds(&mut self, grid: Grid, rows: &[u8]) -> bool {
        let image = Some((grid, blake3::hash(rows)));
        if self.image == image {
            return true;
        }
        self.image = image;
        self.runs += 1;
        false
    }

    /// The held rows are those of no image: another task wrote them.
    pub fn forget(&mut self) {
        self.image = None;
    }

    /// The count of the images that the tower ran for.
    pub fn runs(&self) -> u64 {
        self.runs
    }
}

#[cfg(test)]
mod tests;
