//! The samplers against diffusers (`tools/sdxl_golden.sh`, samplers.safetensors). The test steps each sampler with the
//! guided outputs and the noise of the oracle, and compares the schedule, each UNet input and each latent.

mod common;

use common::{golden, rel};
use nagaya_abi::ImageSampler;
use nagaya_mk_sdxl_bmg::sampler::{History, Schedule, step};

const SAMPLERS: [(&str, ImageSampler); 4] = [
    ("euler", ImageSampler::Euler),
    ("euler_a", ImageSampler::EulerAncestral),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras),
    ("dpmpp_2m_sde_karras", ImageSampler::DpmPp2mSdeKarras),
];

#[test]
fn each_sampler_follows_diffusers_step_by_step() {
    let Some(t) = golden("samplers.safetensors") else {
        return;
    };
    let start = t.values("start");
    let n = start.len();
    for (name, sampler) in SAMPLERS {
        let want_t = t.values(&format!("{name}.timesteps"));
        let schedule = Schedule::new(sampler, want_t.len());
        assert_eq!(schedule.timesteps, want_t, "{name}: timesteps");
        let err = rel(&schedule.sigmas, t.values(&format!("{name}.sigmas")));
        assert!(err < 1e-6, "{name}: sigmas, relative error {err}");
        let latents = t.values(&format!("{name}.latents"));
        let unet_in = t.values(&format!("{name}.unet_in"));
        let guided = t.values(&format!("{name}.guided"));
        let noise = t.values(&format!("{name}.step_noise"));
        let mut x: Vec<f32> = start.iter().map(|v| v * schedule.init_noise_sigma()).collect();
        let err = rel(&x, &latents[..n]);
        assert!(err < 1e-6, "{name}: start latent, relative error {err}");
        let mut history = History::default();
        for i in 0..want_t.len() {
            // The UNet takes the latent of each image twice: for the negative prompt, then for the prompt.
            let scaled: Vec<f32> = x.iter().map(|v| v * schedule.input_scale(i)).collect();
            // The input is the latent times a scale, so it has the bound of the latent.
            let err = rel(&scaled, &unet_in[i * 2 * n..i * 2 * n + n]);
            assert!(err < 1e-5, "{name} step {i}: UNet input, relative error {err}");
            let draw = schedule.draws(i).then(|| &noise[i * n..(i + 1) * n]);
            step(&schedule, i, &mut x, &guided[i * n..(i + 1) * n], draw, &mut history);
            let err = rel(&x, &latents[(i + 1) * n..(i + 2) * n]);
            assert!(err < 1e-5, "{name} step {i}: latent, relative error {err}");
        }
    }
}

#[test]
fn a_schedule_of_30_steps_has_31_sigmas_ending_at_zero() {
    for (name, sampler) in SAMPLERS {
        let s = Schedule::new(sampler, 30);
        assert_eq!(s.timesteps.len(), 30, "{name}");
        assert_eq!(s.sigmas.len(), 31, "{name}");
        assert_eq!(s.sigmas[30], 0.0, "{name}");
        assert!(
            s.sigmas.windows(2).all(|w| w[0] > w[1]),
            "{name}: the sigmas fall: {:?}",
            s.sigmas
        );
    }
}
