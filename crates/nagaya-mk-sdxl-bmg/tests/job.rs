//! The checks of an image job that the engine runs before any quantum (engine-abi "Image", "Image inputs"). They need
//! no card.

use std::sync::Arc;

use nagaya_abi::{ImageInit, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_mk_sdxl_bmg::host::empty_chunk;
use nagaya_mk_sdxl_bmg::job::job_problem;

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
    job_problem(job).expect("the engine refuses the job")
}

const PIXELS: usize = 512 * 768;

#[test]
fn a_text_to_image_job_and_an_image_job_of_the_right_sizes_pass_the_checks() {
    assert_eq!(job_problem(&job()), None);
    assert_eq!(job_problem(&with_image(3 * PIXELS, 0.75, None)), None);
    assert_eq!(job_problem(&with_image(3 * PIXELS, 1.0, Some(PIXELS))), None);
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
        job_problem(&with_image(3 * PIXELS, 0.05, None)),
        None,
        "20 * 0.05 runs 1 step"
    );
}
