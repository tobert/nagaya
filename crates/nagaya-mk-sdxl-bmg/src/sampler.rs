//! The samplers of spec 11.9a, as diffusers 0.42 runs them for SDXL base 1.0: `EulerDiscreteScheduler`,
//! `EulerAncestralDiscreteScheduler`, and `DPMSolverMultistepScheduler` with Karras sigmas, as `dpmsolver++` and as
//! `sde-dpmsolver++`. The config is the checkpoint's: 1000 training steps, betas from 0.00085 to 0.012 on the scaled
//! linear schedule, leading spacing, and a steps offset of 1.
//!
//! The schedule is computed in f64 and kept in f32, as numpy and torch keep it. Each step runs in f32 in the order of
//! diffusers' operations, so a latent follows the reference within rounding.

use nagaya_abi::{ImageSampler, ImageSpacing};

const TRAIN_STEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;
const STEPS_OFFSET: f64 = 1.0;

/// The schedule of a job: the timestep that the UNet takes at each step, and the sigmas.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub sampler: ImageSampler,
    /// The timestep of each step.
    pub timesteps: Vec<f32>,
    /// The sigma of each step, then the final sigma 0.
    pub sigmas: Vec<f32>,
}

/// The alphas_cumprod of the training schedule, in f32 as torch makes them.
fn alphas_cumprod() -> Vec<f32> {
    let (lo, hi) = (BETA_START.sqrt() as f32, BETA_END.sqrt() as f32);
    let step = (hi - lo) / (TRAIN_STEPS - 1) as f32;
    let mut product = 1.0f32;
    (0..TRAIN_STEPS)
        .map(|i| {
            let root = if i < TRAIN_STEPS / 2 {
                lo + step * i as f32
            } else {
                hi - step * (TRAIN_STEPS - 1 - i) as f32
            };
            product *= 1.0 - root * root;
            product
        })
        .collect()
}

/// The sigma of each training timestep: sqrt((1 - a) / a), in f32.
fn train_sigmas() -> Vec<f32> {
    alphas_cumprod().iter().map(|a| ((1.0 - a) / a).sqrt()).collect()
}

/// numpy's `interp` of `x` over the points (i, ys[i]).
fn interp(x: f64, ys: &[f32]) -> f64 {
    if x <= 0.0 {
        return ys[0] as f64;
    }
    let last = ys.len() - 1;
    if x >= last as f64 {
        return ys[last] as f64;
    }
    let i = x.floor() as usize;
    let w = x - i as f64;
    ys[i] as f64 + (ys[i + 1] as f64 - ys[i] as f64) * w
}

/// diffusers' `_sigma_to_t`: the fractional timestep of a sigma, by the log sigmas of training.
fn sigma_to_t(sigma: f64, log_sigmas: &[f32]) -> f64 {
    let log_sigma = sigma.max(1e-10).ln();
    let count = log_sigmas.iter().filter(|l| log_sigma - **l as f64 >= 0.0).count();
    let low = count.saturating_sub(1).min(log_sigmas.len() - 2);
    let (lo, hi) = (log_sigmas[low] as f64, log_sigmas[low + 1] as f64);
    let w = ((lo - log_sigma) / (lo - hi)).clamp(0.0, 1.0);
    (1.0 - w) * low as f64 + w * (low + 1) as f64
}

impl Schedule {
    /// The schedule of `steps` steps of a sampler, with the spacing of the job.
    pub fn new(sampler: ImageSampler, spacing: ImageSpacing, steps: usize) -> Schedule {
        assert!(steps >= 1, "a job has at least one step");
        let sigmas = train_sigmas();
        if spacing == ImageSpacing::Trailing {
            assert!(
                matches!(sampler, ImageSampler::Euler | ImageSampler::EulerAncestral),
                "the DPM++ samplers take their timesteps from their sigmas; the engine refuses trailing spacing"
            );
        }
        match sampler {
            ImageSampler::Euler | ImageSampler::EulerAncestral => {
                let timesteps: Vec<f64> = match spacing {
                    ImageSpacing::Leading => {
                        let ratio = TRAIN_STEPS / steps;
                        (0..steps).rev().map(|i| (i * ratio) as f64 + STEPS_OFFSET).collect()
                    }
                    // diffusers' trailing: the steps count down from the last training step, rounded, less 1.
                    ImageSpacing::Trailing => {
                        let ratio = TRAIN_STEPS as f64 / steps as f64;
                        (0..steps)
                            .map(|i| (TRAIN_STEPS as f64 - i as f64 * ratio).round_ties_even() - 1.0)
                            .collect()
                    }
                };
                let mut out: Vec<f32> = timesteps.iter().map(|t| interp(*t, &sigmas) as f32).collect();
                out.push(0.0);
                Schedule {
                    sampler,
                    timesteps: timesteps.iter().map(|t| *t as f32).collect(),
                    sigmas: out,
                }
            }
            ImageSampler::DpmPp2mKarras | ImageSampler::DpmPp2mSdeKarras => {
                let log_sigmas: Vec<f32> = sigmas.iter().map(|s| s.ln()).collect();
                let (min, max) = (sigmas[0] as f64, sigmas[TRAIN_STEPS - 1] as f64);
                let rho = 7.0;
                let (min_inv, max_inv) = (min.powf(1.0 / rho), max.powf(1.0 / rho));
                let karras: Vec<f64> = (0..steps)
                    .map(|i| {
                        let ramp = if steps == 1 { 0.0 } else { i as f64 / (steps - 1) as f64 };
                        (max_inv + ramp * (min_inv - max_inv)).powf(rho)
                    })
                    .collect();
                let timesteps = karras
                    .iter()
                    .map(|s| sigma_to_t(*s, &log_sigmas).round() as f32)
                    .collect();
                let mut out: Vec<f32> = karras.iter().map(|s| *s as f32).collect();
                out.push(0.0);
                Schedule {
                    sampler,
                    timesteps,
                    sigmas: out,
                }
            }
        }
    }

    /// The factor of the start noise: `init_noise_sigma`.
    pub fn init_noise_sigma(&self) -> f32 {
        match self.sampler {
            ImageSampler::Euler | ImageSampler::EulerAncestral => {
                let max = self.sigmas.iter().fold(0.0f32, |m, s| m.max(*s));
                (max * max + 1.0).sqrt()
            }
            ImageSampler::DpmPp2mKarras | ImageSampler::DpmPp2mSdeKarras => 1.0,
        }
    }

    /// The factor of the UNet input at step `i`: `scale_model_input`.
    pub fn input_scale(&self, i: usize) -> f32 {
        match self.sampler {
            ImageSampler::Euler | ImageSampler::EulerAncestral => {
                let s = self.sigmas[i];
                1.0 / (s * s + 1.0).sqrt()
            }
            ImageSampler::DpmPp2mKarras | ImageSampler::DpmPp2mSdeKarras => 1.0,
        }
    }

    /// True when step `i` draws noise. The engine draws it only then.
    pub fn draws(&self, i: usize) -> bool {
        let _ = i;
        matches!(
            self.sampler,
            ImageSampler::EulerAncestral | ImageSampler::DpmPp2mSdeKarras
        )
    }
}

/// The state of the sampler of one image across the steps: the last model output of DPM++.
#[derive(Clone, Debug, Default)]
pub struct History {
    last: Option<Vec<f32>>,
}

/// One step `i` of the sampler for one image: `x` becomes the latent of step `i + 1`. `eps` is the guided noise
/// prediction, and `noise` the draw of the step when the sampler draws.
pub fn step(schedule: &Schedule, i: usize, x: &mut [f32], eps: &[f32], noise: Option<&[f32]>, history: &mut History) {
    let sigma = schedule.sigmas[i];
    let next = schedule.sigmas[i + 1];
    match schedule.sampler {
        ImageSampler::Euler => {
            for (x, e) in x.iter_mut().zip(eps) {
                let pred = *x - sigma * e;
                let derivative = (*x - pred) / sigma;
                *x += derivative * (next - sigma);
            }
        }
        ImageSampler::EulerAncestral => {
            let noise = noise.expect("euler a draws noise at each step");
            let up = (next * next * (sigma * sigma - next * next) / (sigma * sigma)).sqrt();
            let down = (next * next - up * up).sqrt();
            for ((x, e), n) in x.iter_mut().zip(eps).zip(noise) {
                let pred = *x - sigma * e;
                let derivative = (*x - pred) / sigma;
                *x = *x + derivative * (down - sigma) + n * up;
            }
        }
        ImageSampler::DpmPp2mKarras | ImageSampler::DpmPp2mSdeKarras => {
            let sde = schedule.sampler == ImageSampler::DpmPp2mSdeKarras;
            let ab = |s: f32| {
                let alpha = 1.0 / (s * s + 1.0).sqrt();
                (alpha, s * alpha)
            };
            let (alpha_s, sigma_s) = ab(sigma);
            let (alpha_t, sigma_t) = ab(next);
            let lambda = |a: f32, s: f32| a.ln() - s.ln();
            let h = lambda(alpha_t, sigma_t) - lambda(alpha_s, sigma_s);
            let model: Vec<f32> = x.iter().zip(eps).map(|(x, e)| (x - sigma_s * e) / alpha_s).collect();
            let last_step = i == schedule.timesteps.len() - 1;
            let first_order = history.last.is_none() || last_step;
            let d1 = if first_order {
                None
            } else {
                let (alpha_p, sigma_p) = ab(schedule.sigmas[i - 1]);
                let h0 = lambda(alpha_s, sigma_s) - lambda(alpha_p, sigma_p);
                let r0 = h0 / h;
                let prev = history.last.as_ref().expect("a history");
                Some(
                    model
                        .iter()
                        .zip(prev)
                        .map(|(m0, m1)| (1.0 / r0) * (m0 - m1))
                        .collect::<Vec<f32>>(),
                )
            };
            for (j, x) in x.iter_mut().enumerate() {
                let d0 = model[j];
                *x = if sde {
                    let n = noise.expect("dpm++ 2m sde draws noise at each step")[j];
                    let mut v = (sigma_t / sigma_s * (-h).exp()) * *x + (alpha_t * (1.0 - (-2.0 * h).exp())) * d0;
                    if let Some(d1) = &d1 {
                        v += 0.5 * (alpha_t * (1.0 - (-2.0 * h).exp())) * d1[j];
                    }
                    v + sigma_t * (1.0 - (-2.0 * h).exp()).sqrt() * n
                } else {
                    let mut v = (sigma_t / sigma_s) * *x - (alpha_t * ((-h).exp() - 1.0)) * d0;
                    if let Some(d1) = &d1 {
                        v -= 0.5 * (alpha_t * ((-h).exp() - 1.0)) * d1[j];
                    }
                    v
                };
            }
            history.last = Some(model);
        }
    }
}
