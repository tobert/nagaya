//! The checks of an image job that the engine runs before any quantum (engine-abi "Image", "Image inputs"). The
//! council checks the ranges of spec 11.9a and 11.9b first, so a refusal here is a defect of the caller.

use nagaya_abi::{ImageChunk, ImageJob, ImagePrompt, ImageSampler, ImageSpacing, steps_that_run};

use crate::spec::*;

/// The reason an image job is refused, or `None` when the engine can run it.
pub fn job_problem(job: &ImageJob) -> Option<String> {
    let pixels = job.width as usize * job.height as usize;
    let chunk_ok =
        |c: &ImageChunk| c.ids1.len() == CHUNK_IDS && c.ids2.len() == CHUNK_IDS && c.weights.len() == CHUNK_IDS;
    let ImagePrompt::Chunks {
        prompt,
        negative,
        clip_skip,
    } = &job.prompt
    else {
        return Some("the SDXL engine reads the chunks form of a prompt, not tokens".to_owned());
    };
    if job.sampler == ImageSampler::FlowEuler {
        return Some("the SDXL engine has no flow sampler; give euler, euler a or a DPM++ sampler".to_owned());
    }
    if job.conditioning.is_some() {
        return Some("the SDXL engine keeps no conditioning, so it takes no rows".to_owned());
    }
    if let Some(problem) = inputs_problem(job, pixels) {
        return Some(problem);
    }
    if job.count == 0 || job.count as usize > MAX_IMAGES {
        Some(format!(
            "the job asks for {} images; the engine makes 1 to {MAX_IMAGES}",
            job.count
        ))
    } else if !job.width.is_multiple_of(64) || !job.height.is_multiple_of(64) || job.width == 0 || job.height == 0 {
        Some(format!(
            "the size {} by {} is not a multiple of 64",
            job.width, job.height
        ))
    } else if pixels > MAX_IMAGE_PIXELS {
        Some(format!(
            "an image of {pixels} pixels is more than the {MAX_IMAGE_PIXELS} of the engine"
        ))
    } else if job.steps == 0 {
        Some("the job has 0 steps".to_owned())
    } else if !(2..=3).contains(clip_skip) {
        Some(format!("clip_skip {clip_skip} is not 2 or 3"))
    } else if !job.cfg.is_finite() {
        Some("the CFG scale is not a finite number".to_owned())
    } else if job.spacing == ImageSpacing::Trailing
        && !matches!(job.sampler, ImageSampler::Euler | ImageSampler::EulerAncestral)
    {
        Some("trailing spacing needs euler or euler a; the DPM++ samplers take their sigmas".to_owned())
    } else if prompt.is_empty()
        || prompt.len() > MAX_CHUNKS
        || negative.len() > MAX_CHUNKS
        || !prompt.iter().chain(negative).all(chunk_ok)
    {
        Some(format!(
            "a prompt has 1 to {MAX_CHUNKS} chunks of {CHUNK_IDS} ids and weights"
        ))
    } else {
        None
    }
}

/// The reason the image inputs of a job are refused (engine-abi "Image inputs"), or `None`. `pixels` is the pixel
/// count of one image of the job. The route checks these first (spec 11.9b), so a refusal is a defect of the caller.
fn inputs_problem(job: &ImageJob, pixels: usize) -> Option<String> {
    let init = job.init.as_ref()?;
    if init.pixels.len() != 3 * pixels {
        return Some(format!(
            "the image has {} bytes; an RGB image of {} by {} has {}",
            init.pixels.len(),
            job.width,
            job.height,
            3 * pixels
        ));
    }
    if !(init.strength > 0.0 && init.strength <= 1.0) {
        return Some(format!(
            "the strength {} is not above 0.0 and at most 1.0",
            init.strength
        ));
    }
    if steps_that_run(job.steps, init.strength) == 0 {
        return Some(format!(
            "the strength {} runs no step of {} steps",
            init.strength, job.steps
        ));
    }
    match &init.mask {
        Some(mask) if mask.len() != pixels => Some(format!(
            "the mask has {} bytes; a gray mask of {} by {} has {pixels}",
            mask.len(),
            job.width,
            job.height
        )),
        _ => None,
    }
}
