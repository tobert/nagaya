//! The VAE encoder on a B70 card against diffusers (`tools/sdxl_vae_encode_golden.sh`): the moments, their mean and
//! log variance, and the image latent with the `z` of the reference. Then the encode quantum through the image
//! capability: the first `image_step` of a job with an image (spec 11.9b, engine-abi "Image inputs").
//!
//! It needs a card, `SDXL_MODEL_DIR` and the reference file. Stop `nagaya.service` first (spec 3.1). Run it alone:
//! `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_vae_encode -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use common::{golden, rel};
use nagaya_abi::{Image, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_mk_sdxl_bmg::host::{empty_chunk, image_latent, image_values};
use nagaya_mk_sdxl_bmg::noise::{Purpose, normal};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};

/// The size of the reference image (`tools/sdxl_vae_encode_reference.py`).
const WIDTH: usize = 128;
const HEIGHT: usize = 192;

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("SDXL_MODEL_DIR").map(PathBuf::from);
    if dir.as_ref().is_some_and(|d| d.join("unet").is_dir()) {
        return dir;
    }
    let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
    assert!(
        !require,
        "NAGAYA_REQUIRE_MODELS is set: set SDXL_MODEL_DIR to the SDXL base 1.0 checkpoint"
    );
    eprintln!("skipped: SDXL_MODEL_DIR does not name an SDXL checkpoint");
    None
}

/// The PCI address of the first B70 card.
fn card_address() -> String {
    std::env::var("SDXL_CARD").unwrap_or_else(|_| "0000:03:00.0".to_owned())
}

/// NCHW to pixel-major for one image: [c][h][w] to [h][w][c].
fn pixels(x: &[f32], c: usize) -> Vec<f32> {
    let p = x.len() / c;
    (0..p * c).map(|i| x[(i % c) * p + i / c]).collect()
}

/// The channels `from..from + 4` of each pixel of pixel-major moments.
fn channels(moments: &[f32], from: usize) -> Vec<f32> {
    moments
        .as_chunks::<8>()
        .0
        .iter()
        .flat_map(|m| m[from..from + LATENT].to_vec())
        .collect()
}

fn check(what: &str, got: &[f32], want: &[f32], bound: f32) {
    let err = rel(got, want);
    eprintln!("{what}: relative error {err:.3e}");
    assert!(err < bound, "{what}: relative error {err} is past {bound}");
}

#[test]
fn the_encoder_follows_diffusers_and_the_first_step_of_an_image_job_encodes() {
    let Some(dir) = model_dir() else { return };
    let Some(oracle) = golden("vae_encode.safetensors") else {
        return;
    };
    let rgb: Vec<u8> = oracle.values("rgb").iter().map(|v| *v as u8).collect();
    assert_eq!(rgb.len(), WIDTH * HEIGHT * 3);
    let started = Instant::now();
    let mut engine = SdxlEngine::open(Config::folder(dir), &card_address()).unwrap();
    eprintln!("open: {:.1} s", started.elapsed().as_secs_f32());

    // The encoder alone, from the input values of the reference.
    let values = image_values(&rgb);
    let t = Instant::now();
    let moments = engine.encode_image(&values, HEIGHT, WIDTH);
    eprintln!("VAE encode of {WIDTH} by {HEIGHT}: {} ms", t.elapsed().as_millis());
    // The bound of the VAE decode in xe_oracle.rs: f32 sums in another order over the whole encoder.
    check("the moments", &moments, &pixels(oracle.values("moments"), 8), 1e-3);
    check(
        "the mean",
        &channels(&moments, 0),
        &pixels(oracle.values("mean"), 4),
        1e-3,
    );
    let logvar: Vec<f32> = channels(&moments, LATENT)
        .iter()
        .map(|v| v.clamp(-30.0, 20.0))
        .collect();
    check("the log variance", &logvar, &pixels(oracle.values("logvar"), 4), 1e-3);
    let z = pixels(oracle.values("z"), 4);
    check(
        "the image latent with the z of the reference",
        &image_latent(&moments, &z),
        &pixels(oracle.values("latent"), 4),
        1e-3,
    );
    assert_eq!(
        engine.encode_image(&values, HEIGHT, WIDTH),
        moments,
        "the same image gives the same bits"
    );

    // The encode quantum: the first step of a job with an image. Image i draws z from the seed plus i.
    let job = ImageJob {
        prompt: ImagePrompt::Chunks {
            prompt: vec![empty_chunk()],
            negative: Vec::new(),
            clip_skip: 2,
        },
        width: WIDTH as u32,
        height: HEIGHT as u32,
        steps: 20,
        cfg: 7.0,
        sampler: ImageSampler::Euler,
        spacing: ImageSpacing::Leading,
        count: 2,
        seed: 1234,
        previews: true,
        loras: Vec::new(),
        conditioning: None,
        init: Some(Arc::new(rgb.clone())),
        strength: 0.75,
        mask: None,
    };
    engine.image_begin(0, &job).unwrap();
    assert!(engine.image_latents().is_none(), "image_begin does not encode");
    let t = Instant::now();
    let out = engine.image_step(0).unwrap();
    eprintln!("the encode quantum of 2 images: {} ms", t.elapsed().as_millis());
    assert_eq!(
        (out.step, out.previews.len()),
        (0, 0),
        "the encode is step 0 with no previews"
    );
    let latents = engine
        .image_latents()
        .expect("the encode quantum keeps the image latents")
        .to_vec();
    assert_eq!(latents.len(), 2);
    let n = LATENT * (HEIGHT / VAE_SCALE) * (WIDTH / VAE_SCALE);
    for (i, latent) in latents.iter().enumerate() {
        let z = normal(1234 + i as u64, 0, 0, Purpose::Encode, n);
        assert_eq!(
            latent,
            &image_latent(&moments, &z),
            "image {i} draws from the seed plus {i}"
        );
    }
    assert_ne!(latents[0], latents[1], "each image has its own draw");
    // img2img and inpainting steps are milestone 20 steps 4 and 5, so the next step refuses.
    assert!(engine.image_step(0).is_err());
    engine.image_clear(0);
    assert!(engine.image_latents().is_none());
}
