//! The random stream of the engine (spec 11.9): deterministic, standard normal, and independent across images.

use nagaya_mk_sdxl_bmg::noise::{Purpose, normal};

#[test]
fn the_same_key_gives_the_same_values_and_another_key_others() {
    let a = normal(42, 0, 0, Purpose::Start, 1000);
    assert_eq!(a, normal(42, 0, 0, Purpose::Start, 1000));
    for other in [
        normal(43, 0, 0, Purpose::Start, 1000),
        normal(42, 1, 0, Purpose::Start, 1000),
        normal(42, 0, 1, Purpose::Start, 1000),
        normal(42, 0, 0, Purpose::Step, 1000),
        normal(42, 0, 0, Purpose::Encode, 1000),
        normal(42, 0, 0, Purpose::EncodeMasked, 1000),
    ] {
        let same = a.iter().zip(&other).filter(|(x, y)| x == y).count();
        assert!(same < 5, "{same} of 1000 values are equal");
    }
}

/// The draws of the image latent and of the masked image latent differ from each other (spec 11.9b).
#[test]
fn the_two_encode_purposes_give_other_values() {
    let encode = normal(42, 0, 0, Purpose::Encode, 1000);
    let masked = normal(42, 0, 0, Purpose::EncodeMasked, 1000);
    let same = encode.iter().zip(&masked).filter(|(x, y)| x == y).count();
    assert!(same < 5, "{same} of 1000 values are equal");
}

#[test]
fn a_prefix_does_not_depend_on_the_count() {
    assert_eq!(
        normal(7, 2, 3, Purpose::Step, 10),
        normal(7, 2, 3, Purpose::Step, 100)[..10]
    );
}

#[test]
fn the_values_have_mean_0_and_variance_1() {
    let v = normal(9, 0, 0, Purpose::Start, 200_000);
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((var - 1.0).abs() < 0.01, "variance {var}");
    let tail = v.iter().filter(|x| x.abs() > 3.0).count() as f64 / v.len() as f64;
    assert!((tail - 0.0027).abs() < 0.0007, "the share past 3 sigma is {tail}");
}
