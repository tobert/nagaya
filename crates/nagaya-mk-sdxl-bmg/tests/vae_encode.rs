//! The draw of the image latent against diffusers (`tools/sdxl_inpaint_golden.sh`): the moments and the `z` of the
//! reference give its latent. It needs no card. An absent reference file skips each test.

mod common;

use common::{golden, rel};
use nagaya_mk_sdxl_bmg::host::image_latent;

/// NCHW to pixel-major for one image: [c][h][w] to [h][w][c].
fn pixels(x: &[f32], c: usize) -> Vec<f32> {
    let p = x.len() / c;
    (0..p * c).map(|i| x[(i % c) * p + i / c]).collect()
}

/// The pixel-major moments of the engine from the mean and the raw log variance of the reference: 8 values for each
/// latent pixel.
fn moments(mean: &[f32], logvar: &[f32]) -> Vec<f32> {
    let (mean, logvar) = (pixels(mean, 4), pixels(logvar, 4));
    mean.chunks(4)
        .zip(logvar.chunks(4))
        .flat_map(|(m, l)| [m, l].concat())
        .collect()
}

/// The image latent of img2img and the masked image latent of inpainting, each with its own draw.
#[test]
fn the_draw_of_the_image_latent_gives_the_latents_of_diffusers() {
    let (Some(img2img), Some(inpaint)) = (golden("img2img.safetensors"), golden("inpaint.safetensors")) else {
        return;
    };
    let cases = [
        (&img2img, "euler.s50", "encode", "image_latent"),
        (&inpaint, "euler.s99", "encode", "image_latent"),
        (&inpaint, "euler.s99", "encode_masked", "masked_latent"),
    ];
    for (file, run, purpose, latent) in cases {
        let m = moments(
            file.values(&format!("{run}.{purpose}.mean")),
            file.values(&format!("{run}.{purpose}.logvar")),
        );
        let z = pixels(file.values(&format!("{run}.noise.{purpose}")), 4);
        let want = pixels(file.values(&format!("{run}.{latent}")), 4);
        // Only exp may differ from torch's, by an ulp or so.
        let err = rel(&image_latent(&m, &z), &want);
        assert!(err < 1e-6, "{run} {latent}: relative error {err}");
    }
}
