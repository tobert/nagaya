//! The host parts of the VAE encode against diffusers (`tools/sdxl_vae_encode_golden.sh`): the input values and the
//! draw of the image latent. They need no card. An absent reference file skips each test.

mod common;

use common::{golden, rel};
use nagaya_mk_sdxl_bmg::host::{image_latent, image_values};

/// NCHW to pixel-major for one image: [c][h][w] to [h][w][c].
fn pixels(x: &[f32], c: usize) -> Vec<f32> {
    let p = x.len() / c;
    (0..p * c).map(|i| x[(i % c) * p + i / c]).collect()
}

#[test]
fn the_input_values_are_the_image_of_diffusers_bit_for_bit() {
    let Some(oracle) = golden("vae_encode.safetensors") else {
        return;
    };
    let rgb: Vec<u8> = oracle.values("rgb").iter().map(|v| *v as u8).collect();
    let want = pixels(oracle.values("image"), 3);
    let got = image_values(&rgb);
    assert_eq!(got.len(), want.len());
    let off = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    assert_eq!(off, 0, "{off} of {} values differ from diffusers", want.len());
}

#[test]
fn the_draw_of_the_image_latent_gives_the_latent_of_diffusers() {
    let Some(oracle) = golden("vae_encode.safetensors") else {
        return;
    };
    let moments = pixels(oracle.values("moments"), 8);
    let z = pixels(oracle.values("z"), 4);
    let want = pixels(oracle.values("latent"), 4);
    // Only exp may differ from torch's, by an ulp or so.
    let err = rel(&image_latent(&moments, &z), &want);
    assert!(err < 1e-6, "relative error {err}");
}
