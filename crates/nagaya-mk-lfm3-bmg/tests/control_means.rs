//! The means of the oracle's rows give the oracle's vector bit for bit (spec 11.12): the job and transformers both sum
//! in `f64` in prompt order. No card runs.

mod common;

use common::Which;
use nagaya_council::control::vector_of_means;

fn refs(set: &[Vec<f32>]) -> Vec<&[f32]> {
    set.iter().map(|r| r.as_slice()).collect()
}

#[test]
fn the_means_of_the_oracle_rows_are_the_oracle_vector() {
    if common::absent(Which::Text) || common::golden_absent("lfm3.control.safetensors") {
        return;
    }
    let golden = common::golden_tensors("lfm3.control.safetensors");
    let rows = |set: &str| {
        (0..4)
            .map(|i| golden[&format!("{set}{i}.rows")].f32())
            .collect::<Vec<_>>()
    };
    let (positive, negative) = (rows("positive"), rows("negative"));
    let vector = vector_of_means(&refs(&positive), &refs(&negative)).unwrap();
    let want = golden["vector"].f32();
    assert_eq!(vector.len(), 24 * 2048);
    assert_eq!(
        common::first_other_bits(&vector, &want),
        None,
        "the vector differs from the oracle's bits"
    );
}
