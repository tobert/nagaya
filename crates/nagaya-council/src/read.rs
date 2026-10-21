//! The numbers of a read: log probabilities from a logit row (gogi "Raw reads", "Answers").
//!
//! Each number is float64, and each sum runs in token order. Thus the same logit row gives the same bits (gogi rule
//! 1).

use std::fmt;

use gogi_proto::math::log_sum_exp;

/// Why a logit row gives no numbers. Each one is a defect of the engine or of its caller.
#[derive(Clone, Debug, PartialEq)]
pub enum RowError {
    /// The row has the wrong number of bytes for the vocabulary.
    Length {
        /// The bytes of the row.
        bytes: usize,
        /// The tokens of the vocabulary.
        vocab: usize,
    },
    /// A logit is NaN or infinite.
    NotFinite {
        /// The token.
        id: u32,
        /// The logit.
        value: f32,
    },
}

impl fmt::Display for RowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RowError::Length { bytes, vocab } => write!(
                f,
                "a logit row has {bytes} bytes, but a vocabulary of {vocab} tokens needs {} bytes",
                vocab * 4
            ),
            RowError::NotFinite { id, value } => {
                write!(f, "logit {id} is {value}: the engine gave a logit that is not finite")
            }
        }
    }
}

impl std::error::Error for RowError {}

/// Decode a logit row: one little-endian F32 for each token of the vocabulary (engine-abi "Logit outputs").
pub fn logit_row(bytes: &[u8], vocab: usize) -> Result<Vec<f32>, RowError> {
    if bytes.len() != vocab * 4 {
        return Err(RowError::Length {
            bytes: bytes.len(),
            vocab,
        });
    }
    let (chunks, rest) = bytes.as_chunks::<4>();
    debug_assert!(rest.is_empty(), "the length check above leaves no rest");
    Ok(chunks.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect())
}

/// The log probability of each token of a logit row, over the whole vocabulary.
pub fn log_probs(row: &[f32]) -> Result<Vec<f64>, RowError> {
    if let Some((id, &value)) = row.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(RowError::NotFinite { id: id as u32, value });
    }
    let wide: Vec<f64> = row.iter().map(|&value| f64::from(value)).collect();
    let total = log_sum_exp(&wide);
    Ok(wide.into_iter().map(|value| value - total).collect())
}

/// The `count` most likely tokens, with their log probabilities: the most likely first, and a tie in id order.
pub fn top(log_probs: &[f64], count: usize) -> Vec<(u32, f64)> {
    let mut ids: Vec<u32> = (0..log_probs.len() as u32).collect();
    ids.sort_by(|a, b| log_probs[*b as usize].total_cmp(&log_probs[*a as usize]).then(a.cmp(b)));
    ids.into_iter()
        .take(count)
        .map(|id| (id, log_probs[id as usize]))
        .collect()
}
