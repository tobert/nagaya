//! Exact comparison of values by integer bit pattern (spec 8.3).

use std::fmt;

/// One place where two values differ.
///
/// `left` and `right` hold the bit patterns of two f32 values, or the values of two bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Difference {
    /// The row of the first difference.
    pub row: usize,
    /// The index of the first difference in the row.
    pub index: usize,
    /// The left bit pattern or byte.
    pub left: u32,
    /// The right bit pattern or byte.
    pub right: u32,
    /// The number of differing values in the row.
    pub differing: usize,
}

/// Why a conformance gate failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// The gate that failed.
    pub gate: &'static str,
    /// What the gate saw.
    pub message: String,
    /// The first difference, when the gate compared values.
    pub difference: Option<Difference>,
}

impl Failure {
    /// Make a failure with no difference detail.
    pub fn new(gate: &'static str, message: impl Into<String>) -> Self {
        Self {
            gate,
            message: message.into(),
            difference: None,
        }
    }

    /// Make a failure that names the first difference.
    pub fn with_difference(gate: &'static str, message: impl Into<String>, difference: Difference) -> Self {
        Self {
            gate,
            message: message.into(),
            difference: Some(difference),
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.gate, self.message)?;
        if let Some(difference) = &self.difference {
            write!(
                formatter,
                " (row {} index {}: {:#010x} and {:#010x}, {} differ in the row)",
                difference.row, difference.index, difference.left, difference.right, difference.differing
            )?;
        }
        Ok(())
    }
}

/// Compare two f32 dumps by integer bit pattern.
///
/// A NaN or an infinity fails, even when both bit patterns are equal. A signed zero differs from a zero.
/// The counts must match, the dump must be non-empty, and each row must be whole.
pub fn compare_f32_rows(gate: &'static str, left: &[f32], right: &[f32], row_len: usize) -> Result<(), Failure> {
    if row_len == 0 {
        return Err(Failure::new(gate, "the row length is zero"));
    }
    if left.is_empty() || right.is_empty() {
        return Err(Failure::new(gate, "a dump is empty"));
    }
    if left.len() != right.len() {
        return Err(Failure::new(
            gate,
            format!("the counts differ: {} values and {} values", left.len(), right.len()),
        ));
    }
    if !left.len().is_multiple_of(row_len) {
        return Err(Failure::new(gate, "a dump ends with a partial row"));
    }
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if !left.is_finite() || !right.is_finite() {
            return Err(Failure::new(
                gate,
                format!(
                    "row {} index {} is not finite: {:#010x} and {:#010x}",
                    index / row_len,
                    index % row_len,
                    left.to_bits(),
                    right.to_bits()
                ),
            ));
        }
    }
    for (row, (left_row, right_row)) in left.chunks(row_len).zip(right.chunks(row_len)).enumerate() {
        let mut differing = 0;
        let mut first = None;
        for (index, (left, right)) in left_row.iter().zip(right_row).enumerate() {
            if left.to_bits() != right.to_bits() {
                differing += 1;
                if first.is_none() {
                    first = Some((index, left.to_bits(), right.to_bits()));
                }
            }
        }
        if let Some((index, left, right)) = first {
            return Err(Failure::with_difference(
                gate,
                format!("row {row} differs: {differing} values in the row"),
                Difference {
                    row,
                    index,
                    left,
                    right,
                    differing,
                },
            ));
        }
    }
    Ok(())
}

/// Compare two byte dumps. The counts must match and the dumps must be non-empty.
pub fn compare_bytes(gate: &'static str, left: &[u8], right: &[u8]) -> Result<(), Failure> {
    if left.is_empty() || right.is_empty() {
        return Err(Failure::new(gate, "a dump is empty"));
    }
    if left.len() != right.len() {
        return Err(Failure::new(
            gate,
            format!("the counts differ: {} bytes and {} bytes", left.len(), right.len()),
        ));
    }
    let differing = left.iter().zip(right).filter(|(left, right)| left != right).count();
    if differing > 0 {
        let index = left
            .iter()
            .zip(right)
            .position(|(left, right)| left != right)
            .unwrap_or(0);
        return Err(Failure::with_difference(
            gate,
            format!("{differing} bytes differ"),
            Difference {
                row: 0,
                index,
                left: left[index] as u32,
                right: right[index] as u32,
                differing,
            },
        ));
    }
    Ok(())
}
