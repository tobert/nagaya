//! The minimal postprocess of upstream (`Postprocessor(type="minimal")`): the beat and downbeat times of the logits.

use crate::mel::FPS;

/// The beat and downbeat times, in seconds. Each downbeat is also a beat.
#[derive(Clone, Debug, PartialEq)]
pub struct Beats {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
}

/// The frames that are the largest in a window of 7 (3 at each side, the edges padded with minus infinity) and above
/// zero, as `F.max_pool1d(x, 7, 1, 3)` and `> 0` find them.
fn peak_frames(logits: &[f32]) -> Vec<usize> {
    let n = logits.len();
    (0..n)
        .filter(|&i| {
            let lo = i.saturating_sub(3);
            let hi = (i + 4).min(n);
            let max = logits[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            logits[i] == max && logits[i] > 0.0
        })
        .collect()
}

/// The peak frames of one row of logits, after the join of adjacent peaks.
pub fn frames(logits: &[f32]) -> Vec<f64> {
    deduplicate(&peak_frames(logits))
}

/// Upstream's `deduplicate_peaks` with `width=1`: a run of peaks becomes their mean. The next peak joins the run when
/// it is at most 1 frame past the mean so far, not past the last peak.
pub fn deduplicate(peaks: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    let Some((&first, rest)) = peaks.split_first() else {
        return out;
    };
    let (mut p, mut c) = (first as f64, 1.0f64);
    for &q in rest {
        let q = q as f64;
        if q - p <= 1.0 {
            c += 1.0;
            p += (q - p) / c;
        } else {
            out.push(p);
            p = q;
            c = 1.0;
        }
    }
    out.push(p);
    out
}

/// Moves each downbeat time onto its nearest beat time (the first one on a tie, as `np.argmin`), then sorts them and
/// drops repeats (`np.unique`). With no beats, the downbeats are only sorted, as upstream skips the move.
pub fn snap(beats: &[f64], downbeats: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = downbeats
        .iter()
        .map(|&d| {
            let mut best: Option<(f64, f64)> = None;
            for &b in beats {
                let gap = (b - d).abs();
                if best.is_none_or(|(g, _)| gap < g) {
                    best = Some((gap, b));
                }
            }
            best.map_or(d, |(_, b)| b)
        })
        .collect();
    out.sort_by(f64::total_cmp);
    out.dedup();
    out
}

/// The beat and downbeat times of a row of beat logits and a row of downbeat logits.
pub fn beats(beat: &[f32], downbeat: &[f32]) -> Beats {
    assert_eq!(
        beat.len(),
        downbeat.len(),
        "the two rows of logits must have the same frames"
    );
    let fps = f64::from(FPS);
    let beats: Vec<f64> = frames(beat).into_iter().map(|f| f / fps).collect();
    let downs: Vec<f64> = frames(downbeat).into_iter().map(|f| f / fps).collect();
    let downbeats = snap(&beats, &downs);
    Beats { beats, downbeats }
}
