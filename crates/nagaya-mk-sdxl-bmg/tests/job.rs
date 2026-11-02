//! The checks of an image job that the engine runs before any quantum (engine-abi "Image", "Image inputs"). They need
//! no card.

use std::sync::Arc;

use nagaya_abi::{ImageInit, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_mk_sdxl_bmg::host::empty_chunk;
use nagaya_mk_sdxl_bmg::job::job_problem;
use nagaya_mk_sdxl_bmg::spec::{INPAINT_CHANNELS, LATENT};

/// A text to image job of 512 by 768 that the engine runs.
fn job() -> ImageJob {
    ImageJob {
        prompt: ImagePrompt::Chunks {
            prompt: vec![empty_chunk()],
            negative: Vec::new(),
            clip_skip: 2,
        },
        width: 512,
        height: 768,
        steps: 20,
        cfg: 7.0,
        sampler: ImageSampler::Euler,
        spacing: ImageSpacing::Leading,
        count: 1,
        seed: 1,
        previews: false,
        loras: Vec::new(),
        conditioning: None,
        init: None,
    }
}

/// A job with an image of `bytes` bytes, the strength `strength` and a mask of `mask` bytes, if any.
fn with_image(bytes: usize, strength: f64, mask: Option<usize>) -> ImageJob {
    ImageJob {
        init: Some(ImageInit {
            pixels: Arc::new(vec![0; bytes]),
            strength,
            mask: mask.map(|n| Arc::new(vec![255; n])),
        }),
        ..job()
    }
}

fn problem(job: &ImageJob) -> String {
    job_problem4(job).expect("the engine refuses the job")
}

const PIXELS: usize = 512 * 768;

/// The reason an engine with a 4-channel UNet refuses a job.
fn job_problem4(job: &ImageJob) -> Option<String> {
    job_problem(job, LATENT)
}

#[test]
fn a_text_to_image_job_and_an_image_job_of_the_right_sizes_pass_the_checks() {
    assert_eq!(job_problem4(&job()), None);
    assert_eq!(job_problem4(&with_image(3 * PIXELS, 0.75, None)), None);
    assert_eq!(job_problem4(&with_image(3 * PIXELS, 1.0, Some(PIXELS))), None);
}

/// The image is 3 bytes for each pixel of the job, and the mask 1 byte (engine-abi "Image inputs").
#[test]
fn an_image_or_a_mask_of_another_size_than_the_job_is_refused() {
    let short = with_image(3 * PIXELS - 1, 0.5, None);
    assert!(problem(&short).contains("1179647 bytes"), "{}", problem(&short));
    let mask = with_image(3 * PIXELS, 0.5, Some(512 * 512));
    assert!(problem(&mask).contains("mask"), "{}", problem(&mask));
}

/// `strength` runs `floor(steps * strength)` steps (spec 11.9b). A strength out of range, or one that runs no step,
/// is refused.
#[test]
fn a_strength_out_of_range_or_one_that_runs_no_step_is_refused() {
    for strength in [0.0, -0.5, 1.5, f64::NAN, 0.04] {
        let bad = with_image(3 * PIXELS, strength, None);
        assert!(problem(&bad).contains("strength"), "{strength}: {}", problem(&bad));
    }
    assert_eq!(
        job_problem4(&with_image(3 * PIXELS, 0.05, None)),
        None,
        "20 * 0.05 runs 1 step"
    );
}

/// A 9-channel UNet needs an image and a mask (spec 11.9b, engine-abi "Image inputs"). It refuses text to image and
/// img2img, and runs inpainting. A 4-channel UNet runs all three.
#[test]
fn a_9_channel_unet_refuses_a_job_without_an_image_and_a_mask() {
    for job in [job(), with_image(3 * PIXELS, 0.75, None)] {
        let problem = job_problem(&job, INPAINT_CHANNELS).expect("a 9-channel UNet refuses the job");
        assert!(problem.contains("an image and a mask"), "{problem}");
        assert_eq!(job_problem(&job, LATENT), None);
    }
    assert_eq!(
        job_problem(&with_image(3 * PIXELS, 1.0, Some(PIXELS)), INPAINT_CHANNELS),
        None
    );
    assert_eq!(
        job_problem(&with_image(3 * PIXELS, 0.99, Some(PIXELS)), INPAINT_CHANNELS),
        None
    );
}
