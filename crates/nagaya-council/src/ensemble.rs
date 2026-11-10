//! The rules of joined forks (docs/lab.md, "`ensemble`"): one row from the logit rows of K contexts.

/// The rule that combines the rows.
#[derive(Clone, Debug, PartialEq)]
pub enum Combine {
    /// The weighted sum of the log softmax of each row.
    LogprobSum,
    /// The log of the weighted mean of the probabilities, with the weights scaled to a sum of 1.
    ProbMean,
    /// `(1 + strength) l_0 - strength l_1`, for exactly two rows.
    Contrast { strength: f64 },
}

/// Why the rows do not combine.
#[derive(Clone, Debug, PartialEq)]
pub enum EnsembleError {
    NoRows,
    Length(usize),
    WeightCount,
    Weight(usize),
    NotFinite(usize),
    ContrastCount(usize),
    Strength,
}

impl std::fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRows => write!(f, "an ensemble needs at least one row"),
            Self::Length(i) => write!(f, "row {i} has another length than row 0"),
            Self::WeightCount => write!(f, "give one weight for each row"),
            Self::Weight(i) => write!(f, "weight {i} must be finite and above 0"),
            Self::NotFinite(i) => write!(f, "row {i} holds a value that is not finite"),
            Self::ContrastCount(n) => write!(f, "contrast needs exactly 2 rows, not {n}"),
            Self::Strength => write!(f, "the strength must be finite"),
        }
    }
}

impl std::error::Error for EnsembleError {}

/// The log softmax of a row, in `f64`.
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
    let sum: f64 = row.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    let shift = max + sum.ln();
    row.iter().map(|&x| f64::from(x) - shift).collect()
}

/// Combine the logit rows of K contexts into one row for the sampler of spec 11.2. The sums run in `f64`, and the row
/// goes to `f32`. One row with `LogprobSum` and the weight 1 comes back unchanged, so it keeps the bits of a generation.
pub fn combine(rows: &[&[f32]], rule: &Combine, weights: &[f64]) -> Result<Vec<f32>, EnsembleError> {
    let first = rows.first().ok_or(EnsembleError::NoRows)?;
    if let Some(i) = rows.iter().position(|r| r.len() != first.len()) {
        return Err(EnsembleError::Length(i));
    }
    if weights.len() != rows.len() {
        return Err(EnsembleError::WeightCount);
    }
    if let Some(i) = weights.iter().position(|w| !(w.is_finite() && *w > 0.0)) {
        return Err(EnsembleError::Weight(i));
    }
    if let Some(i) = rows.iter().position(|r| r.iter().any(|x| !x.is_finite())) {
        return Err(EnsembleError::NotFinite(i));
    }
    if let Combine::Contrast { strength } = rule {
        if rows.len() != 2 {
            return Err(EnsembleError::ContrastCount(rows.len()));
        }
        if !strength.is_finite() {
            return Err(EnsembleError::Strength);
        }
    }
    if rows.len() == 1 && *rule == Combine::LogprobSum && weights[0] == 1.0 {
        return Ok(first.to_vec());
    }
    let logs: Vec<Vec<f64>> = rows.iter().map(|r| log_softmax(r)).collect();
    let n = first.len();
    let out: Vec<f64> = match rule {
        Combine::LogprobSum => (0..n)
            .map(|t| logs.iter().zip(weights).map(|(l, w)| w * l[t]).sum())
            .collect(),
        Combine::ProbMean => {
            let total: f64 = weights.iter().sum();
            let lw: Vec<f64> = weights.iter().map(|w| (w / total).ln()).collect();
            (0..n)
                .map(|t| {
                    let m = logs
                        .iter()
                        .zip(&lw)
                        .fold(f64::NEG_INFINITY, |m, (l, w)| m.max(l[t] + w));
                    m + logs
                        .iter()
                        .zip(&lw)
                        .map(|(l, w)| (l[t] + w - m).exp())
                        .sum::<f64>()
                        .ln()
                })
                .collect()
        }
        Combine::Contrast { strength } => (0..n)
            .map(|t| (1.0 + strength) * logs[0][t] - strength * logs[1][t])
            .collect(),
    };
    Ok(out.into_iter().map(|x| x as f32).collect())
}
