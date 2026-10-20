//! The draw function of spec 11.2: fixed vectors, determinism and the effect of each input.

use nagaya_council::sampler::{Purpose, child_seed, draw_uniform};

const SEED: u64 = 0x0123_4567_89ab_cdef;

/// The input bytes, written out by hand: the domain, the seed, the position and the purpose.
fn draw_input(position_le: [u8; 8], purpose: u8) -> Vec<u8> {
    let mut bytes = b"nagaya draw v1".to_vec();
    bytes.extend_from_slice(&[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]); // SEED, little-endian
    bytes.extend_from_slice(&position_le);
    bytes.push(purpose);
    bytes
}

#[test]
fn the_draw_has_a_fixed_test_vector() {
    // Position 5, purpose sample. The hash was computed once, from the bytes of `draw_input`.
    let hash = blake3::hash(&draw_input([5, 0, 0, 0, 0, 0, 0, 0], 0));
    assert_eq!(
        hash.to_hex().as_str(),
        "edfd667c279d4e281301ae8eb2b2178609fb33c79123ef9cadc0a82467b2a728"
    );
    // The uniform value is the first 8 bytes, little-endian, shifted right by 11, times 2^-53.
    let u = draw_uniform(SEED, 5, Purpose::Sample);
    assert_eq!(u.to_bits(), 0x3fc4_274e_93be_337c_u64);
    // The other purposes have their own vectors.
    assert_eq!(
        draw_uniform(SEED, 5, Purpose::Resample).to_bits(),
        0x3fdf_769f_301f_3344_u64
    );
    assert_eq!(
        draw_uniform(SEED, 5, Purpose::Draft).to_bits(),
        0x3fe6_b0e1_8564_d31c_u64
    );
}

#[test]
fn the_draw_follows_the_rule_in_the_spec() {
    let hash = blake3::hash(&draw_input([5, 0, 0, 0, 0, 0, 0, 0], 0));
    let first = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    assert_eq!(
        draw_uniform(SEED, 5, Purpose::Sample),
        (first >> 11) as f64 / 9007199254740992.0
    );
}

#[test]
fn the_child_seed_has_a_fixed_test_vector() {
    let mut bytes = b"nagaya fork v1".to_vec();
    bytes.extend_from_slice(&[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
    bytes.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0]);
    let hash = blake3::hash(&bytes);
    assert_eq!(
        hash.to_hex().as_str(),
        "d1d64a37440c3b0f8fc1d2b260e7e3d1048ab1f83a3c3de91dcf9aba7fd4d22c"
    );
    assert_eq!(child_seed(SEED, 3), 0x0f3b_0c44_374a_d6d1_u64);
}

#[test]
fn the_same_inputs_give_the_same_value() {
    for _ in 0..3 {
        assert_eq!(
            draw_uniform(SEED, 17, Purpose::Sample),
            draw_uniform(SEED, 17, Purpose::Sample)
        );
    }
    assert_eq!(child_seed(SEED, 1), child_seed(SEED, 1));
}

#[test]
fn a_different_input_changes_the_value() {
    let base = draw_uniform(SEED, 17, Purpose::Sample);
    assert_ne!(base, draw_uniform(SEED, 18, Purpose::Sample), "position");
    assert_ne!(base, draw_uniform(SEED, 17, Purpose::Resample), "purpose");
    assert_ne!(base, draw_uniform(SEED, 17, Purpose::Draft), "purpose");
    assert_ne!(base, draw_uniform(SEED + 1, 17, Purpose::Sample), "seed");
    assert_ne!(child_seed(SEED, 1), child_seed(SEED, 2), "branch");
    assert_ne!(child_seed(SEED, 1), child_seed(SEED + 1, 1), "parent");
}

#[test]
fn the_value_is_in_the_half_open_unit_interval() {
    for position in 0..2000 {
        let u = draw_uniform(SEED, position, Purpose::Sample);
        assert!((0.0..1.0).contains(&u), "position {position}: {u}");
    }
}

#[test]
fn the_values_spread_over_the_interval() {
    let n = 4000;
    let mean = (0..n).map(|p| draw_uniform(SEED, p, Purpose::Sample)).sum::<f64>() / n as f64;
    assert!((mean - 0.5).abs() < 0.03, "mean {mean}");
}
