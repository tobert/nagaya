//! The checks of an image job that the engine runs before any quantum (engine-abi "Image", "Image inputs"). They need
//! no card.

use std::sync::Arc;

use nagaya_abi::{ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
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
        strength: 1.0,
        mask: None,
    }
}

fn problem(job: &ImageJob) -> String {
    job_problem(job).expect("the engine refuses the job")
}

#[test]
fn a_text_to_image_job_and_an_image_job_of_the_right_sizes_run() {
    assert_eq!(job_problem(&job()), None);
    let img2img = ImageJob {
        init: Some(Arc::new(vec![0; 512 * 768 * 3])),
        strength: 0.75,
        ..job()
    };
    assert_eq!(job_problem(&img2img), None);
    let inpaint = ImageJob {
        mask: Some(Arc::new(vec![255; 512 * 768])),
        strength: 1.0,
        ..img2img
    };
    assert_eq!(job_problem(&inpaint), None);
}

/// The image is 3 bytes for each pixel of the job, and the mask 1 byte (engine-abi "Image inputs"). The route checks
/// the sizes first (spec 11.9b), so a mismatch here is a defect of the caller.
#[test]
fn an_image_or_a_mask_of_another_size_than_the_job_is_refused() {
    let short = ImageJob {
        init: Some(Arc::new(vec![0; 512 * 768 * 3 - 1])),
        strength: 0.5,
        ..job()
    };
    assert!(problem(&short).contains("1179647 bytes"), "{}", problem(&short));
    let mask = ImageJob {
        init: Some(Arc::new(vec![0; 512 * 768 * 3])),
        mask: Some(Arc::new(vec![0; 512 * 512])),
        strength: 0.5,
        ..job()
    };
    assert!(problem(&mask).contains("mask"), "{}", problem(&mask));
}

/// `strength` runs `floor(steps * strength)` steps of an image job (spec 11.9b). With no image it has no effect, so
/// it must be 1.0.
#[test]
fn a_strength_out_of_range_or_without_an_image_is_refused() {
    let image = Some(Arc::new(vec![0u8; 512 * 768 * 3]));
    for strength in [0.0, -0.5, 1.5, f32::NAN] {
        let bad = ImageJob {
            init: image.clone(),
            strength,
            ..job()
        };
        assert!(problem(&bad).contains("strength"), "{strength}: {}", problem(&bad));
    }
    let alone = ImageJob {
        strength: 0.75,
        ..job()
    };
    assert!(problem(&alone).contains("strength"), "{}", problem(&alone));
}

#[test]
fn a_mask_without_an_image_is_refused() {
    let alone = ImageJob {
        mask: Some(Arc::new(vec![0; 512 * 768])),
        ..job()
    };
    assert!(problem(&alone).contains("mask"), "{}", problem(&alone));
}
