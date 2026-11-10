//! Latent steps (docs/lab.md, "Latent steps in a request"): the map from the residual after the last layer to the
//! input row of the next position. No I/O.
//!
//! The map is the final norm of the model, then a scale to the mean norm of the embedding rows. LFM3's head is its
//! embedding table, so this is the map of LatentMAS with a ridge weight near 0 (`tools/lfm3_reference.py`,
//! `run_latent`).

/// The most latent steps of one context.
pub const MAX_STEPS: usize = 256;

/// The map of a latent step for one model.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentMap {
    /// The layers of the residual tap: the tap gives one row for each layer, in layer order.
    pub layers: usize,
    /// The width of the residual.
    pub width: usize,
    /// The weight of the final norm.
    pub norm: Vec<f32>,
    /// The epsilon of the final norm.
    pub eps: f64,
    /// The mean L2 norm of the rows of the embedding table.
    pub scale: f32,
}

/// Why the bytes of a tap do not give a row.
#[derive(Clone, Debug, PartialEq)]
pub enum LatentError {
    /// The tap has another count of bytes than `layers` rows of `width` f32 values.
    Size { got: usize, want: usize },
    /// The row after the final norm has a length of 0 or a value that is not finite.
    Degenerate,
}

impl std::fmt::Display for LatentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Size { got, want } => write!(f, "the residual tap has {got} bytes, not {want}"),
            Self::Degenerate => write!(f, "the hidden state has a length of 0 or a value that is not finite"),
        }
    }
}

impl std::error::Error for LatentError {}

impl LatentMap {
    /// The input row of the next position, from the bytes of the tap `residual` of one logit row. The tap holds
    /// `layers` rows of `width` little-endian f32 values, and the last row is the residual after the last layer.
    pub fn row(&self, residual: &[u8]) -> Result<Vec<f32>, LatentError> {
        let want = self.layers * self.width * 4;
        if residual.len() != want || self.width == 0 {
            return Err(LatentError::Size {
                got: residual.len(),
                want,
            });
        }
        let last: Vec<f32> = residual[want - self.width * 4..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        self.scaled(&self.normed(&last))
    }

    /// The final norm of a residual row: each value divides by the root mean square of the row with the epsilon, then
    /// takes the weight. The mean runs in `f64`.
    pub fn normed(&self, x: &[f32]) -> Vec<f32> {
        let mean = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64;
        let inverse = 1.0 / (mean + self.eps).sqrt();
        x.iter()
            .zip(&self.norm)
            .map(|(v, g)| (f64::from(*v) * inverse) as f32 * g)
            .collect()
    }

    /// The hidden state `h` scaled to the length `scale`: `h * (scale / |h|)`, as the reference does.
    pub fn scaled(&self, h: &[f32]) -> Result<Vec<f32>, LatentError> {
        let length = h.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
        if !length.is_finite() || length == 0.0 || h.iter().any(|v| !v.is_finite()) {
            return Err(LatentError::Degenerate);
        }
        let factor = (f64::from(self.scale) / length) as f32;
        Ok(h.iter().map(|v| v * factor).collect())
    }
}

/// The top token of a logit row: the id of the largest value, and the smaller id in a tie.
pub fn top(row: &[f32]) -> Option<u32> {
    row.iter()
        .enumerate()
        .fold(None, |best: Option<(usize, f32)>, (i, v)| match best {
            Some((_, b)) if v.total_cmp(&b).is_le() => best,
            _ => Some((i, *v)),
        })
        .map(|(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> LatentMap {
        LatentMap {
            layers: 2,
            width: 3,
            norm: vec![2.0, 0.5, -1.0],
            eps: 0.25,
            scale: 3.0,
        }
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn a_row_is_the_final_norm_of_the_last_layer_scaled_to_the_mean_norm() {
        let m = map();
        // Layer 0 is noise; the map reads layer 1.
        let tap = bytes(&[9.0, 9.0, 9.0, 3.0, -1.0, 2.0]);
        let row = m.row(&tap).unwrap();
        // The mean of the squares of [3, -1, 2] is 14 / 3, and the epsilon is 0.25.
        let inverse = 1.0 / (14.0f64 / 3.0 + 0.25).sqrt();
        let h: Vec<f32> = [3.0f32, -1.0, 2.0]
            .iter()
            .zip([2.0f32, 0.5, -1.0])
            .map(|(x, g)| (f64::from(*x) * inverse) as f32 * g)
            .collect();
        assert_eq!(m.normed(&[3.0, -1.0, 2.0]), h);
        let length = h.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        let want: Vec<f32> = h.iter().map(|v| v * (3.0 / length) as f32).collect();
        assert_eq!(row, want);
        let norm = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 3.0).abs() < 1e-6, "{norm}");
    }

    #[test]
    fn a_tap_of_another_size_or_a_row_of_length_0_is_an_error() {
        let m = map();
        assert_eq!(
            m.row(&bytes(&[1.0; 3])).unwrap_err(),
            LatentError::Size { got: 12, want: 24 }
        );
        assert_eq!(
            m.row(&bytes(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])).unwrap_err(),
            LatentError::Degenerate
        );
        assert_eq!(m.scaled(&[f32::NAN, 1.0, 1.0]).unwrap_err(), LatentError::Degenerate);
    }

    #[test]
    fn the_top_token_is_the_largest_value_and_the_smaller_id_in_a_tie() {
        assert_eq!(top(&[0.5, 2.0, -1.0, 2.0]), Some(1));
        assert_eq!(top(&[3.0, 2.0]), Some(0));
        assert_eq!(top(&[]), None);
    }
}
