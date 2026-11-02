//! The host work of the engine: the features of a timestep, the token weights of A1111, the empty chunk, and the
//! bytes of an image.

use nagaya_mk_sdxl_bmg::host::*;

#[test]
fn the_features_of_timestep_0_are_cosines_of_1_then_sines_of_0() {
    let f = timestep_features(0.0, 320);
    assert_eq!(f.len(), 320);
    assert!(f[..160].iter().all(|v| *v == 1.0) && f[160..].iter().all(|v| *v == 0.0));
    // The first frequency is 1, so feature 0 is cos(t) and feature 160 is sin(t).
    let f = timestep_features(751.0, 320);
    assert!((f[0] - 751f32.cos()).abs() < 1e-6 && (f[160] - 751f32.sin()).abs() < 1e-6);
    // The last frequency is 10000^(-159/160).
    let last = (-(10000f32.ln()) * 159.0 / 160.0).exp();
    assert!((f[159] - (751.0 * last).cos()).abs() < 1e-6);
}

#[test]
fn the_add_features_are_the_pooled_text_then_six_time_ids() {
    let pooled: Vec<f32> = (0..1280).map(|i| i as f32).collect();
    let add = add_features(&pooled, &time_ids(256, 320));
    assert_eq!(add.len(), 1280 + 6 * 256);
    assert_eq!(&add[..1280], &pooled[..]);
    assert_eq!(
        &add[1280..1280 + 256],
        &timestep_features(320.0, 256)[..],
        "the original height first"
    );
    assert_eq!(&add[1280 + 256..1280 + 512], &timestep_features(256.0, 256)[..]);
}

#[test]
fn token_weights_scale_their_rows_and_keep_the_mean() {
    let mut rows: Vec<f32> = (0..3 * 4).map(|i| 1.0 + i as f32).collect();
    let before: f32 = rows.iter().sum::<f32>() / rows.len() as f32;
    let original = rows.clone();
    weight_chunk(&mut rows, &[1.0, 2.0, 0.5]);
    let after: f32 = rows.iter().sum::<f32>() / rows.len() as f32;
    assert!((after - before).abs() < 1e-5, "the mean {before} became {after}");
    // Row 1 has twice the weight of row 0, after the common factor.
    let ratio = rows[4] / original[4] / (rows[0] / original[0]);
    assert!((ratio - 2.0).abs() < 1e-5, "{ratio}");
    let mut plain = original.clone();
    weight_chunk(&mut plain, &[1.0, 1.0, 1.0]);
    assert_eq!(plain, original);
}

#[test]
fn the_empty_chunk_pads_each_encoder_as_the_family_pads() {
    let c = empty_chunk();
    assert_eq!(&c.ids1[..3], &[49406, 49407, 49407]);
    assert_eq!(&c.ids2[..3], &[49406, 49407, 0]);
    assert_eq!(eos_position(&c.ids2), 1);
    assert!(c.weights.iter().all(|w| *w == 1.0));
}

/// The image latent: the first 4 channels of the moments are the mean, the last 4 the log variance. The latent is
/// `(mean + exp(0.5 * clamp(logvar, -30, 20)) * z) * 0.13025` (spec 11.9b, "The image latent").
#[test]
fn the_image_latent_draws_from_the_mean_and_the_clamped_log_variance() {
    let moments = [1.0, -1.0, 0.5, 0.0, 2.0, 50.0, -100.0, 0.0];
    let z = [0.5, 1.0, 2.0, -3.0];
    let latent = image_latent(&moments, &z);
    let want = [
        (1.0 + 1f32.exp() * 0.5) * 0.13025,
        (-1.0 + 10f32.exp() * 1.0) * 0.13025,
        (0.5 + (-15f32).exp() * 2.0) * 0.13025,
        (0.0 + 1.0 * -3.0) * 0.13025,
    ];
    assert_eq!(latent, want);
}

#[test]
#[should_panic(expected = "8 moments for each latent pixel")]
fn the_image_latent_refuses_moments_that_do_not_match_the_draw() {
    image_latent(&[0.0; 8], &[0.0; 8]);
}

#[test]
fn image_bytes_round_the_clamped_values() {
    assert_eq!(
        rgb_bytes(&[-1.0, 1.0, 0.0, -2.0, 2.0, 0.003]),
        vec![0, 255, 128, 0, 255, 128]
    );
}
