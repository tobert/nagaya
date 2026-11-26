//! The canary content and the check of a record (spec 8.1).

use nagaya_family_beatthis::canary::{self, FRAMES, Refusal};
use nagaya_family_beatthis::mel::MELS;

#[test]
fn the_inputs_are_fixed_and_in_the_range_of_a_log_mel() {
    for (i, &frames) in FRAMES.iter().enumerate() {
        let x = canary::input(i);
        assert_eq!(x.len(), frames * MELS);
        assert!(x.iter().all(|v| (0.0..8.0).contains(v)), "chunk {i} leaves [0, 8)");
        assert_eq!(x, canary::input(i), "chunk {i} is the same at each call");
    }
    assert_ne!(canary::input(0)[..64], canary::input(1)[..64], "the two chunks differ");
    // The first values, pinned, so a change of the generator changes this test, not only every record.
    let first: Vec<u32> = canary::input(0)[..4].iter().map(|v| v.to_bits()).collect();
    assert_eq!(first, PIN_FIRST_BITS);
}

const PIN_FIRST_BITS: [u32; 4] = [1074110894, 1089021440, 1073636388, 1078024294];

#[test]
fn the_hash_covers_each_bit_of_each_output() {
    let outs = vec![vec![0.5f32; FRAMES[0] * 2], vec![-1.0f32; FRAMES[1] * 2]];
    let h = canary::hash(&outs);
    assert!(h.starts_with("sha256:") && h.len() == 7 + 64, "{h}");
    let mut flipped = outs.clone();
    flipped[1][2999] = f32::from_bits((-1.0f32).to_bits() ^ 1);
    assert_ne!(canary::hash(&flipped), h, "the last bit of the last value");
}

const TABLE: &[(&str, &str)] = &[("sha256:aa", "sha256:11")];

#[test]
fn a_record_that_matches_opens() {
    assert_eq!(canary::verify(TABLE, "cpu", "sha256:aa", "sha256:11"), Ok(()));
}

#[test]
fn a_checkpoint_with_no_record_is_refused_with_the_line_to_add() {
    let e = canary::verify(TABLE, "cpu", "sha256:bb", "sha256:22").unwrap_err();
    assert!(matches!(&e, Refusal::NoRecord { .. }));
    let text = e.to_string();
    assert!(
        text.contains("(\"sha256:bb\", \"sha256:22\")") && text.contains("on cpu"),
        "{text}"
    );
}

#[test]
fn other_bits_are_refused() {
    let e = canary::verify(TABLE, "gfx1151", "sha256:aa", "sha256:33").unwrap_err();
    assert!(matches!(&e, Refusal::Mismatch { .. }));
    assert!(e.to_string().contains("Do not serve it"), "{e}");
}
