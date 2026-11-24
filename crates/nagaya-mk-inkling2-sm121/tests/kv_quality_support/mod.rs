//! The measures of the quality of a cache format against the f32 cache (milestone 37 step 11, lane E). The f32 logits
//! of a row are the reference. These functions use no GPU, so `cuda_kv_quality_metrics.rs` tests them on the host.

#![allow(dead_code)]

/// The count of the largest f32 logits over which the gap is taken.
pub const TOP_GAP_IDS: usize = 64;

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, x)| if *x > b.1 { (i, *x) } else { b })
        .0
}

/// The logsumexp of a row, in f64.
pub fn logsumexp(v: &[f32]) -> f64 {
    let top = f64::from(v[argmax(v)]);
    top + v.iter().map(|x| (f64::from(*x) - top).exp()).sum::<f64>().ln()
}

/// The measures of one row: the format `got` against the reference `want`.
#[derive(Clone, Copy, Debug)]
pub struct RowMeasure {
    /// The two rows have the same top token.
    pub agrees: bool,
    /// KL(want || got) over the whole vocabulary, in nats.
    pub kl: f64,
    /// The largest `|got - want|` over the `TOP_GAP_IDS` largest logits of `want`.
    pub gap: f64,
    /// The count of tokens that `got` ranks above the top token of `want`. Zero means the top token stays first.
    pub rank: usize,
}

pub fn measure_row(want: &[f32], got: &[f32]) -> RowMeasure {
    assert_eq!(want.len(), got.len(), "the two rows have different lengths");
    let (lw, lg) = (logsumexp(want), logsumexp(got));
    let mut kl = 0.0;
    for (w, g) in want.iter().zip(got) {
        let lp = f64::from(*w) - lw;
        let lq = f64::from(*g) - lg;
        kl += lp.exp() * (lp - lq);
    }
    let mut order: Vec<usize> = (0..want.len()).collect();
    order.select_nth_unstable_by(TOP_GAP_IDS - 1, |a, b| want[*b].total_cmp(&want[*a]));
    let gap = order[..TOP_GAP_IDS]
        .iter()
        .map(|i| (f64::from(got[*i]) - f64::from(want[*i])).abs())
        .fold(0.0, f64::max);
    let top = argmax(want);
    RowMeasure {
        agrees: argmax(got) == top,
        kl,
        gap,
        rank: got.iter().filter(|x| **x > got[top]).count(),
    }
}

/// The measures of a window of rows.
#[derive(Clone, Copy, Debug)]
pub struct WindowMeasure {
    pub rows: usize,
    pub agreement: f64,
    pub kl_mean: f64,
    pub kl_max: f64,
    pub gap_max: f64,
    pub rank_max: usize,
    /// The rows where the top token of the reference is not first in the format.
    pub rank_nonzero: usize,
}

/// `want` and `got` hold `rows` rows of `width` values each. The row `i` of `want` meets the row `(i + shift) % rows`
/// of `got`. The honest shift is 0; the test of the harness uses another.
pub fn measure_window(want: &[f32], got: &[f32], width: usize, shift: usize) -> WindowMeasure {
    assert_eq!(want.len(), got.len());
    assert_eq!(want.len() % width, 0);
    let rows = want.len() / width;
    let all: Vec<RowMeasure> = (0..rows)
        .map(|i| {
            let j = (i + shift) % rows;
            measure_row(&want[i * width..(i + 1) * width], &got[j * width..(j + 1) * width])
        })
        .collect();
    WindowMeasure {
        rows,
        agreement: all.iter().filter(|m| m.agrees).count() as f64 / rows as f64,
        kl_mean: all.iter().map(|m| m.kl).sum::<f64>() / rows as f64,
        kl_max: all.iter().map(|m| m.kl).fold(0.0, f64::max),
        gap_max: all.iter().map(|m| m.gap).fold(0.0, f64::max),
        rank_max: all.iter().map(|m| m.rank).max().unwrap_or(0),
        rank_nonzero: all.iter().filter(|m| m.rank > 0).count(),
    }
}

/// The share of rows of a window whose top token is the real next token.
pub fn accuracy(rows: &[f32], width: usize, next: &[u32]) -> f64 {
    let count = rows.len() / width;
    assert_eq!(count, next.len());
    (0..count)
        .filter(|i| argmax(&rows[i * width..(i + 1) * width]) as u32 == next[*i])
        .count() as f64
        / count as f64
}
