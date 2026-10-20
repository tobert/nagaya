//! The template id of spec 11.3a.

use nagaya_council::template_id::{council_hash, hash16, template_id};

#[test]
fn hash16_is_the_first_16_hex_digits_of_a_sha256() {
    // SHA-256 of the empty input.
    assert_eq!(hash16(b""), "e3b0c44298fc1c14");
    // SHA-256 of "abc".
    assert_eq!(hash16(b"abc"), "ba7816bf8f01cfea");
}

#[test]
fn the_council_hash_has_a_fixed_value_for_fixed_renders() {
    let renders = vec![vec![1, 2, 3], vec![], vec![4_000_000_000]];
    assert_eq!(council_hash(&renders), "baee569b63e41276");
}

#[test]
fn one_changed_token_changes_the_council_hash() {
    let base = vec![vec![1, 2, 3], vec![4, 5]];
    let mut changed = base.clone();
    changed[1][0] = 6;
    assert_ne!(council_hash(&base), council_hash(&changed));
}

#[test]
fn the_borders_between_renders_are_part_of_the_hash() {
    assert_ne!(
        council_hash(&[vec![1, 2], vec![3]]),
        council_hash(&[vec![1], vec![2, 3]])
    );
    assert_ne!(council_hash(&[vec![1, 2]]), council_hash(&[vec![1, 2], vec![]]));
}

#[test]
fn the_template_id_has_the_form_of_spec_11_3a() {
    assert_eq!(
        template_id("0123456789abcdef", "toy", "fedcba9876543210"),
        "nagaya-council:0123456789abcdef+toy-chat:fedcba9876543210"
    );
}

#[test]
#[should_panic(expected = "16 hexadecimal digits")]
fn a_hash_of_the_wrong_form_is_a_panic() {
    template_id("0123", "toy", "fedcba9876543210");
}
