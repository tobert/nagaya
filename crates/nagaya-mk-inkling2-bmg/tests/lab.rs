//! The head of the lab of Inkling 2 against the checkpoint (`docs/lab.md`, "The rules of `project`", "Inkling 2").
//! A test skips when the model is absent.

mod common;

use common::Which;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, bf16_to_f32};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use nagaya_mk_inkling2_bmg::host::weights::LOGITS;
use nagaya_mk_inkling2_bmg::lab::{Facts, HEAD, Head, NORM, dot};

/// The score of a direction is the row of the head times the weight of the final norm times the direction, with no
/// division and no softcap. The head has 200,058 rows, and the end token is 200,006.
///
/// Falsified by a score with no weight of the norm, and by a head of 201,024 rows.
#[test]
fn a_score_is_the_row_of_the_head_times_the_norm_times_the_direction() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let weights = common::model_weights(Which::Mini);
    let dir = weights.parent().unwrap();
    let facts = Facts::open(dir).unwrap();
    assert_eq!(
        (facts.layers, facts.width, facts.ends.clone()),
        (24, 2304, vec![200_006])
    );
    let head = Head::open(dir).unwrap();
    assert_eq!(
        (head.rows(), head.width(), head.ends()),
        (LOGITS, HIDDEN, &[200_006][..])
    );
    let v: Vec<f32> = (0..HIDDEN).map(|i| ((i * 13 % 17) as f32 - 8.0) / 4.0).collect();
    let scores = head.scores(&[&v]).remove(0);
    assert_eq!(scores.len(), LOGITS);
    let file = MappedSafeTensors::open(&weights).unwrap();
    let view = file.view().unwrap();
    let row = |name: &str, r: usize| -> Vec<f32> {
        view.tensor_bytes(name).unwrap()[r * HIDDEN * 2..(r + 1) * HIDDEN * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
            .collect()
    };
    let norm = row(NORM, 0);
    let unit: Vec<f32> = v.iter().zip(&norm).map(|(x, g)| x * g).collect();
    for id in [0usize, 1, 200_006, LOGITS - 1] {
        assert_eq!(scores[id], dot(&row(HEAD, id), &unit), "the score of {id}");
    }
    // The same request gives the same bits.
    assert_eq!(head.scores(&[&v]).remove(0), scores);
}
