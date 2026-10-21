//! The numbers of a read (gogi "Raw reads", "Answers").

use nagaya_council::read::{RowError, log_probs, logit_row, top};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[test]
fn a_logit_row_is_little_endian_f32_with_one_value_for_each_token() {
    assert_eq!(logit_row(&bytes(&[1.5, -2.0, 0.25]), 3).unwrap(), [1.5, -2.0, 0.25]);
    assert_eq!(
        logit_row(&bytes(&[1.5, -2.0]), 3),
        Err(RowError::Length { bytes: 8, vocab: 3 })
    );
}

#[test]
fn log_probabilities_sum_to_one_and_keep_the_differences_of_the_logits() {
    let lp = log_probs(&[2.0, 0.0, -1.0, 2.0]).unwrap();
    let total: f64 = lp.iter().map(|x| x.exp()).sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
    assert!((lp[0] - lp[1] - 2.0).abs() < 1e-12);
    assert_eq!(lp[0], lp[3]);
    // The value is the logit less log Σ exp, in float64.
    let expected = 2.0 - (2f64.exp() * 2.0 + 1.0 + (-1f64).exp()).ln();
    assert!((lp[0] - expected).abs() < 1e-12, "{} {expected}", lp[0]);
}

#[test]
fn a_logit_that_is_not_finite_is_an_error() {
    assert!(matches!(
        log_probs(&[0.0, f32::NAN]),
        Err(RowError::NotFinite { id: 1, .. })
    ));
    assert!(matches!(
        log_probs(&[f32::INFINITY]),
        Err(RowError::NotFinite { id: 0, .. })
    ));
}

#[test]
fn the_top_tokens_are_the_most_likely_first_and_a_tie_goes_by_id() {
    let lp = [-3.0, -1.0, -2.0, -1.0, -0.5];
    assert_eq!(top(&lp, 3), [(4, -0.5), (1, -1.0), (3, -1.0)]);
    assert_eq!(top(&lp, 10).len(), 5);
    assert!(top(&lp, 0).is_empty());
}
