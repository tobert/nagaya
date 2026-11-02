//! The samplers against diffusers (`tools/sdxl_golden.sh`, samplers.safetensors). The test steps each sampler with the
//! guided outputs and the noise of the oracle, and compares the schedule, each UNet input and each latent.

mod common;

use common::{golden, rel};
use nagaya_abi::{ImageSampler, ImageSpacing, steps_that_run};
use nagaya_mk_sdxl_bmg::sampler::{History, Schedule, begin_index, step};

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
        let schedule = Schedule::new(sampler, ImageSpacing::Leading, want_t.len());
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
        let s = Schedule::new(sampler, ImageSpacing::Leading, 30);
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

/// The trailing spacing of the Euler samplers gives diffusers' timesteps and sigmas (tests/fixtures/trailing_schedule.json,
/// one CPU run of diffusers 0.42.1). The timesteps are exact, and the sigmas are within 1e-6.
#[test]
fn trailing_spacing_follows_diffusers() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/trailing_schedule.json")).expect("the fixture is JSON");
    for steps in ["3", "4", "8"] {
        let want = &fixture[steps];
        let want_t: Vec<f32> = want["timesteps"]
            .as_array()
            .expect("timesteps")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let want_s: Vec<f32> = want["sigmas"]
            .as_array()
            .expect("sigmas")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for sampler in [ImageSampler::Euler, ImageSampler::EulerAncestral] {
            let s = Schedule::new(sampler, ImageSpacing::Trailing, steps.parse().unwrap());
            assert_eq!(s.timesteps, want_t, "{steps} steps of {sampler:?}: timesteps");
            let err = rel(&s.sigmas, &want_s);
            assert!(err < 1e-6, "{steps} steps of {sampler:?}: sigmas, relative error {err}");
        }
    }
}

/// The cut of the schedule of a job with an image (spec 11.9b): `n = floor(steps * strength)` in f64, and the first
/// step that runs is `steps - n`. The cases of `image_inputs.safetensors` come from diffusers' `get_timesteps`.
#[test]
fn a_job_with_an_image_runs_the_last_n_steps_as_diffusers_does() {
    let Some(t) = golden("image_inputs.safetensors") else {
        return;
    };
    let cases = t.f64s("schedule.cases");
    let (want_n, want_begin) = (t.values("schedule.n"), t.values("schedule.begin"));
    assert_eq!(cases.len(), 2 * want_n.len());
    for (k, case) in cases.chunks(2).enumerate() {
        let (steps, strength) = (case[0] as u32, case[1]);
        assert_eq!(
            steps_that_run(steps, strength) as f32,
            want_n[k],
            "{steps} steps at strength {strength}: n"
        );
        assert_eq!(
            begin_index(steps, strength) as f32,
            want_begin[k],
            "{steps} steps at strength {strength}: the begin index"
        );
    }
}

/// The runs of img2img in `img2img.safetensors`: the sampler, and the strength in percent.
const IMG2IMG_RUNS: [(&str, ImageSampler, &str); 5] = [
    ("euler", ImageSampler::Euler, "s50"),
    ("euler_a", ImageSampler::EulerAncestral, "s50"),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s50"),
    ("dpmpp_2m_sde_karras", ImageSampler::DpmPp2mSdeKarras, "s50"),
    ("euler", ImageSampler::Euler, "s100"),
];

/// img2img against diffusers (spec 11.9b): the whole schedule, the begin index, the start latent of `add_noise`, and
/// each step from the guided outputs and the step noise of the reference. The DPM++ history starts empty.
#[test]
fn img2img_follows_diffusers_from_the_start_latent_to_the_last_step() {
    let Some(t) = golden("img2img.safetensors") else {
        return;
    };
    for (name, sampler, strength) in IMG2IMG_RUNS {
        let run = format!("{name}.{strength}");
        let v = |key: &str| t.values(&format!("{run}.{key}"));
        let steps = v("timesteps").len();
        let schedule = Schedule::new(sampler, ImageSpacing::Leading, steps);
        assert_eq!(schedule.timesteps, v("timesteps"), "{run}: timesteps");
        let err = rel(&schedule.sigmas, v("sigmas"));
        assert!(err < 1e-6, "{run}: sigmas, relative error {err}");
        let begin = begin_index(steps as u32, t.f64s(&format!("{run}.strength"))[0]);
        assert_eq!(begin as f32, v("begin")[0], "{run}: the begin index");
        let n = steps - begin;
        assert_eq!(n as f32, v("n")[0], "{run}: the count of steps that run");
        assert_eq!(
            &schedule.timesteps[begin..],
            v("run_timesteps"),
            "{run}: the timesteps that run"
        );
        let err = rel(&schedule.sigmas[begin..], v("run_sigmas"));
        assert!(err < 1e-6, "{run}: the sigmas that run, relative error {err}");

        let latents = v("latents");
        let size = v("image_latent").len();
        let mut x = schedule.add_noise(begin, v("image_latent"), v("noise.start"));
        let err = rel(&x, &latents[..size]);
        eprintln!("{run}: the start latent, relative error {err:.3e}");
        assert!(err < 1e-6, "{run}: the start latent, relative error {err}");
        let (unet_in, guided, noise) = (v("unet_in"), v("guided"), v("noise.step"));
        let mut history = History::default();
        for k in 0..n {
            let i = begin + k;
            let scaled: Vec<f32> = x.iter().map(|v| v * schedule.input_scale(i)).collect();
            let err = rel(&scaled, &unet_in[k * 2 * size..k * 2 * size + size]);
            assert!(err < 1e-5, "{run} step {k}: UNet input, relative error {err}");
            let draw = schedule.draws(i).then(|| &noise[k * size..(k + 1) * size]);
            step(
                &schedule,
                i,
                &mut x,
                &guided[k * size..(k + 1) * size],
                draw,
                &mut history,
            );
            let err = rel(&x, &latents[(k + 1) * size..(k + 2) * size]);
            assert!(err < 1e-5, "{run} step {k}: latent, relative error {err}");
        }
        let err = rel(&x, v("final_latent"));
        assert!(err < 1e-5, "{run}: final latent, relative error {err}");
    }
}
