//! The split of a long spectrum into chunks of the model, and the join of their logits: `split_piece` with
//! `avoid_short_end` and `aggregate_prediction` with `keep_first` of upstream `beat_this.inference`.

use crate::mel::MELS;

/// The frames of one chunk: 30 s, the length the model was trained on.
pub const CHUNK: usize = 1500;
/// The frames at each end of a chunk that the join drops, because the model was not trained on its edges.
pub const BORDER: usize = 6;

/// One chunk: `left` zero rows, then `rows` rows of the spectrum from `max(start, 0)`, then `right` zero rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: isize,
    pub left: usize,
    pub rows: usize,
    pub right: usize,
}

impl Chunk {
    /// The frames that the model reads for this chunk.
    pub fn len(&self) -> usize {
        self.left + self.rows + self.right
    }

    /// True when the chunk has no frames.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The chunks of a spectrum of `frames` frames. The starts step by `CHUNK - 2 * BORDER` from `-BORDER`. When the
/// piece is longer than one step, the last start moves so that the last chunk ends `BORDER` frames past the piece.
pub fn plan(frames: usize) -> Vec<Chunk> {
    let (t, b, c) = (frames as isize, BORDER as isize, CHUNK as isize);
    let mut starts: Vec<isize> = (-b..t - b).step_by(CHUNK - 2 * BORDER).collect();
    if frames > CHUNK - 2 * BORDER
        && let Some(last) = starts.last_mut()
    {
        *last = t - (c - b);
    }
    starts
        .into_iter()
        .map(|start| {
            let lo = start.max(0);
            let hi = (start + c).min(t);
            Chunk {
                start,
                left: (-start).max(0) as usize,
                rows: (hi - lo) as usize,
                right: (start + c - t).clamp(0, b) as usize,
            }
        })
        .collect()
}

/// The input rows of one chunk, `chunk.len()` rows of `MELS` values: zero rows at its borders.
pub fn input(spect: &[f32], frames: usize, chunk: &Chunk) -> Vec<f32> {
    assert_eq!(
        spect.len(),
        frames * MELS,
        "the spectrum must have {frames} rows of {MELS} values"
    );
    let lo = chunk.start.max(0) as usize;
    let mut out = vec![0.0f32; chunk.len() * MELS];
    out[chunk.left * MELS..(chunk.left + chunk.rows) * MELS]
        .copy_from_slice(&spect[lo * MELS..(lo + chunk.rows) * MELS]);
    out
}

/// Joins the logits of each chunk into one row of `frames` beat logits and one of downbeat logits. Each chunk gives
/// its frames without its borders. Where two chunks overlap, the earlier chunk wins (`keep_first`). Upstream fills
/// the row with -1000 first, but the plan covers every frame, so here a frame that no chunk covers is a defect: it
/// panics, and no fill value can reach a caller.
pub fn join(frames: usize, plan: &[Chunk], logits: &[(Vec<f32>, Vec<f32>)]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(plan.len(), logits.len(), "one pair of logits for each chunk");
    let mut beat = vec![0.0f32; frames];
    let mut down = vec![0.0f32; frames];
    let mut covered = vec![false; frames];
    for (chunk, (b, d)) in plan.iter().zip(logits).rev() {
        assert!(
            b.len() == chunk.len() && d.len() == chunk.len(),
            "the logits of a chunk must have {} frames, not {} and {}",
            chunk.len(),
            b.len(),
            d.len()
        );
        let lo = (chunk.start + BORDER as isize) as usize;
        let hi = ((chunk.start + (CHUNK - BORDER) as isize) as usize).min(frames);
        let inner = chunk.len() - 2 * BORDER;
        assert_eq!(
            hi - lo,
            inner,
            "a chunk at {} covers {} frames of the join",
            chunk.start,
            hi - lo
        );
        beat[lo..hi].copy_from_slice(&b[BORDER..BORDER + inner]);
        down[lo..hi].copy_from_slice(&d[BORDER..BORDER + inner]);
        covered[lo..hi].fill(true);
    }
    if let Some(gap) = covered.iter().position(|&c| !c) {
        panic!("the plan of {frames} frames does not cover frame {gap}");
    }
    (beat, down)
}
