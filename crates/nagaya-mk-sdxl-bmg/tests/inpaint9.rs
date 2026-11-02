//! The host work of the 9-channel UNet against the oracle (`tools/sdxl_inpaint_golden.sh`, spec 11.9b, "The 9-channel
//! UNet"), with no card: the masked image, the input of each step, and the start latent. It reads the reference files,
//! and an absent file skips.

mod common;

use common::golden;
use nagaya_abi::{ImageSampler, ImageSpacing};
use nagaya_mk_sdxl_bmg::host::inpaint_input;
use nagaya_mk_sdxl_bmg::sampler::{Schedule, begin_index};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_text_sdxl::inputs::{binary, masked};

const WIDTH: usize = 256;
const HEIGHT: usize = 320;
const STEPS: u32 = 4;
/// The latent pixels of the reference: 40 by 32.
const LATENT_PIXELS: usize = (HEIGHT / VAE_SCALE) * (WIDTH / VAE_SCALE);

/// The runs of `inpaint9.safetensors`: the sampler and the strength in percent.
const RUNS: [(&str, ImageSampler, &str); 6] = [
    ("euler", ImageSampler::Euler, "s99"),
    ("euler_a", ImageSampler::EulerAncestral, "s99"),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s99"),
    ("dpmpp_2m_sde_karras", ImageSampler::DpmPp2mSdeKarras, "s99"),
    ("euler", ImageSampler::Euler, "s100"),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s100"),
];

/// NCHW to pixel-major for `planes` pixels in each plane.
fn pixels(x: &[f32], c: usize, planes: usize) -> Vec<f32> {
    let b = x.len() / (c * planes);
    assert_eq!(b * c * planes, x.len(), "a tensor of {c} channels has whole planes");
    let mut y = vec![0.0f32; x.len()];
    for img in 0..b {
        for ch in 0..c {
            for q in 0..planes {
                y[(img * planes + q) * c + ch] = x[(img * c + ch) * planes + q];
            }
        }
    }
    y
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().map(|v| *v as u8).collect()
}

/// The masked image is the image with 0 at each repainted pixel, after the map to -1 to 1 (inpaint.py:1455).
#[test]
fn the_masked_image_has_0_at_each_repainted_pixel() {
    let Some(inputs) = golden("image_inputs.safetensors") else {
        return;
    };
    let rgb = bytes(inputs.values("init.rgb"));
    let mask = binary(&bytes(inputs.values("mask.blurred")));
    assert!(mask.contains(&0) && mask.contains(&1), "the mask keeps and repaints");
    let got = masked(&rgb, &mask);
    let want = pixels(inputs.values("masked.tensor"), 3, HEIGHT * WIDTH);
    // torch gives -0.0 for a negative value times 0; the spec says 0, and the encoder sums both alike.
    assert!(got.iter().zip(&want).all(|(g, w)| g == w), "the masked image differs");
}

/// The input of each step is the scaled latent, the latent mask, and the masked latent, in this order, in both halves
/// of CFG (inpaint.py:1612-1615). The mask and the masked latent are exact. The scaled latent differs from torch's
/// division by at most a rounding.
#[test]
fn each_step_takes_the_scaled_latent_the_mask_and_the_masked_latent() {
    let (Some(inputs), Some(t)) = (golden("image_inputs.safetensors"), golden("inpaint9.safetensors")) else {
        return;
    };
    let mask = bytes(inputs.values("mask.latent"));
    for (name, sampler, percent) in RUNS {
        let key = format!("{name}.{percent}");
        let v = |k: &str| t.values(&format!("{key}.{k}"));
        let schedule = Schedule::new(sampler, ImageSpacing::Leading, STEPS as usize);
        let begin = begin_index(STEPS, t.f64s(&format!("{key}.strength"))[0]);
        let size = LATENT * LATENT_PIXELS;
        let wide = INPAINT_CHANNELS * LATENT_PIXELS;
        let masked_latent = pixels(v("masked_latent"), LATENT, LATENT_PIXELS);
        let (latents, unet_in) = (v("latents"), v("unet_in"));
        for k in 0..schedule.timesteps.len() - begin {
            let scale = schedule.input_scale(begin + k);
            let latent = pixels(&latents[k * size..(k + 1) * size], LATENT, LATENT_PIXELS);
            let scaled: Vec<f32> = latent.iter().map(|x| x * scale).collect();
            let got = inpaint_input(&scaled, &mask, &masked_latent);
            assert_eq!(got.len(), wide);
            for half in 0..2 {
                let at = (2 * k + half) * wide;
                let want = pixels(&unet_in[at..at + wide], INPAINT_CHANNELS, LATENT_PIXELS);
                for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                    if j % INPAINT_CHANNELS < LATENT {
                        assert!(
                            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                            "{key} step {k}: value {j}: {g} and {w}"
                        );
                    } else {
                        assert!(g == w, "{key} step {k} half {half}: value {j} is {g}, not {w}");
                    }
                }
            }
        }
    }
}

/// At a strength of 1.0 a 9-channel UNet encodes no image: the run has no draw `encode` and no image latent, and the
/// start latent is the start noise times `init_noise_sigma` (inpaint.py:805 and :813-815).
#[test]
fn at_a_strength_of_1_the_start_is_the_noise_and_no_image_is_encoded() {
    let Some(t) = golden("inpaint9.safetensors") else {
        return;
    };
    for (name, sampler) in [
        ("euler", ImageSampler::Euler),
        ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras),
    ] {
        let key = format!("{name}.s100");
        assert!(!t.has(&format!("{key}.noise.encode")) && !t.has(&format!("{key}.image_latent")));
        let schedule = Schedule::new(sampler, ImageSpacing::Leading, STEPS as usize);
        let sigma = schedule.init_noise_sigma();
        let start: Vec<f32> = t
            .values(&format!("{key}.noise.start"))
            .iter()
            .map(|v| v * sigma)
            .collect();
        let latents = t.values(&format!("{key}.latents"));
        let want = &latents[..start.len()];
        assert!(
            start
                .iter()
                .zip(want)
                .all(|(g, w)| (g - w).abs() <= 1e-6 * w.abs().max(1.0)),
            "{key}: the start latent"
        );
    }
    assert!(t.has("euler.s99.noise.encode"), "below 1.0 the image is encoded");
}
