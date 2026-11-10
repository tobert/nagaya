//! The rules of joined forks (docs/lab.md, "`ensemble`"): each rule against hand sums and against NumPy, the identity
//! of one context, and each refusal.

use nagaya_council::ensemble::{Combine, EnsembleError, combine};

fn ln(x: f64) -> f64 {
    x.ln()
}

fn close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let gap = (f64::from(g) - w).abs();
        assert!(gap <= tol * w.abs().max(1.0), "{what}[{i}]: {g} against {w}");
    }
}

#[test]
fn one_context_with_logprob_sum_and_the_weight_1_gives_its_row_unchanged() {
    let row = [3.25f32, -1.0, 0.5, 7.0, -0.0];
    let got = combine(&[&row], &Combine::LogprobSum, &[1.0]).unwrap();
    let bits = |r: &[f32]| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(&row));
}

#[test]
fn each_rule_gives_the_hand_sums() {
    // Row a is 0.5, 0.5 and row b is 0.9, 0.1, as probabilities.
    let a = [ln(0.5) as f32 + 3.0, ln(0.5) as f32 + 3.0];
    let b = [ln(0.9) as f32 - 2.0, ln(0.1) as f32 - 2.0];
    let sum = combine(&[&a, &b], &Combine::LogprobSum, &[1.0, 1.0]).unwrap();
    close(&sum, &[ln(0.45), ln(0.05)], 1e-6, "logprob_sum");
    let mean = combine(&[&a, &b], &Combine::ProbMean, &[1.0, 1.0]).unwrap();
    close(&mean, &[ln(0.7), ln(0.3)], 1e-6, "prob_mean");
    let weighted = combine(&[&a, &b], &Combine::ProbMean, &[1.0, 3.0]).unwrap();
    close(
        &weighted,
        &[ln(0.25 * 0.5 + 0.75 * 0.9), ln(0.25 * 0.5 + 0.75 * 0.1)],
        1e-6,
        "prob_mean, weights",
    );
    let contrast = combine(&[&b, &a], &Combine::Contrast { strength: 1.0 }, &[1.0, 1.0]).unwrap();
    close(
        &contrast,
        &[2.0 * ln(0.9) - ln(0.5), 2.0 * ln(0.1) - ln(0.5)],
        1e-6,
        "contrast",
    );
    let none = combine(&[&b, &a], &Combine::Contrast { strength: 0.0 }, &[1.0, 1.0]).unwrap();
    close(&none, &[ln(0.9), ln(0.1)], 1e-6, "contrast at strength 0");
}

fn floats(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn each_rule_matches_numpy() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ensemble.json")).unwrap();
    let f: serde_json::Value = serde_json::from_str(&text).unwrap();
    let rows: Vec<Vec<f32>> = f["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| floats(r).into_iter().map(|x| x as f32).collect())
        .collect();
    let rows: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    let weights = floats(&f["weights"]);
    let strength = f["strength"].as_f64().unwrap();
    let got = combine(&rows, &Combine::LogprobSum, &weights).unwrap();
    close(&got, &floats(&f["logprob_sum"]), 1e-6, "logprob_sum");
    let got = combine(&rows, &Combine::ProbMean, &weights).unwrap();
    close(&got, &floats(&f["prob_mean"]), 1e-6, "prob_mean");
    let got = combine(&rows[..2], &Combine::Contrast { strength }, &weights[..2]).unwrap();
    close(&got, &floats(&f["contrast"]), 1e-6, "contrast");
}

#[test]
fn each_fault_is_refused_with_its_place() {
    let a = [0.0f32, 1.0];
    let b = [0.0f32, 1.0, 2.0];
    let n = [0.0f32, f32::NAN];
    assert_eq!(combine(&[], &Combine::LogprobSum, &[]), Err(EnsembleError::NoRows));
    assert_eq!(
        combine(&[&a, &b], &Combine::LogprobSum, &[1.0, 1.0]),
        Err(EnsembleError::Length(1))
    );
    assert_eq!(
        combine(&[&a, &a], &Combine::LogprobSum, &[1.0]),
        Err(EnsembleError::WeightCount)
    );
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            combine(&[&a, &a], &Combine::ProbMean, &[1.0, bad]),
            Err(EnsembleError::Weight(1))
        );
    }
    assert_eq!(
        combine(&[&a, &n], &Combine::LogprobSum, &[1.0, 1.0]),
        Err(EnsembleError::NotFinite(1))
    );
    assert_eq!(
        combine(&[&a, &a, &a], &Combine::Contrast { strength: 1.0 }, &[1.0, 1.0, 1.0]),
        Err(EnsembleError::ContrastCount(3))
    );
    assert_eq!(
        combine(&[&a, &a], &Combine::Contrast { strength: f64::NAN }, &[1.0, 1.0]),
        Err(EnsembleError::Strength)
    );
}
