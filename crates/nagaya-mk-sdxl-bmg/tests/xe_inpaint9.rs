//! Inpainting with the 9-channel UNet on a B70 card against diffusers (`tools/sdxl_inpaint_golden.sh`, spec 11.9b,
//! "The 9-channel UNet"). Each run of `inpaint9.safetensors` goes block by block with the draws of the reference: the
//! image latent, the masked image latent, the start latent, each 9-channel UNet input, the UNet on the input of the
//! reference, each sampler step, the final latent, and the VAE decode with the VAE of the inpainting folder. Then the
//! same run goes through the image capability with the draws of the engine, and the test replays it with the same
//! calls. The two must give the same bits and the same previews, so the capability runs the math that the oracle checks.
//!
//! It needs a card, `SDXL_INPAINT_MODEL_DIR` and the reference files. Stop `nagaya.service` first (spec 3.1). Run it
//! alone: `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_inpaint9 -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use common::{Tensors, golden, rel};
use nagaya_abi::{DecodedImage, Image, ImageChunk, ImageInit, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_mk_sdxl_bmg::host;
use nagaya_mk_sdxl_bmg::model::UnetSize;
use nagaya_mk_sdxl_bmg::noise::{Purpose, normal};
use nagaya_mk_sdxl_bmg::sampler::{History, Schedule, begin_index, step};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};
use nagaya_text_sdxl::SdxlFamily;
use nagaya_text_sdxl::inputs::{binary, latent_mask, masked, unit};

/// The request of the reference (`tools/sdxl_inpaint_golden.sh`).
const PROMPT: &str = "a red fox in a snowy forest, watercolor";
const NEGATIVE: &str = "blurry, low quality";
const WIDTH: usize = 256;
const HEIGHT: usize = 320;
const STEPS: u32 = 4;
const SEED: u64 = 1234;
const CFG: f32 = 7.0;

/// The runs of `inpaint9.safetensors`: the sampler, the strength in percent, and the strength.
const RUNS: [(&str, ImageSampler, &str, f64); 6] = [
    ("euler", ImageSampler::Euler, "s99", 0.99),
    ("euler_a", ImageSampler::EulerAncestral, "s99", 0.99),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s99", 0.99),
    ("dpmpp_2m_sde_karras", ImageSampler::DpmPp2mSdeKarras, "s99", 0.99),
    ("euler", ImageSampler::Euler, "s100", 1.0),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s100", 1.0),
];

/// The folder of `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, from `SDXL_INPAINT_MODEL_DIR`.
fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("SDXL_INPAINT_MODEL_DIR").map(PathBuf::from);
    if dir.as_ref().is_some_and(|d| d.join("unet").is_dir()) {
        return dir;
    }
    let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
    assert!(
        !require,
        "NAGAYA_REQUIRE_MODELS is set: set SDXL_INPAINT_MODEL_DIR to the SDXL inpainting 0.1 checkpoint"
    );
    eprintln!("skipped: SDXL_INPAINT_MODEL_DIR does not name the SDXL inpainting checkpoint");
    None
}

/// The PCI address of the first B70 card.
fn card_address() -> String {
    std::env::var("SDXL_CARD").unwrap_or_else(|_| "0000:03:00.0".to_owned())
}

/// NCHW to pixel-major: [b][c][h][w] to [b][h][w][c]. A tensor of 3 channels is the decoded image, and any other is
/// a latent or a UNet input.
fn pixels(x: &[f32], c: usize) -> Vec<f32> {
    let p = if c == 3 {
        HEIGHT * WIDTH
    } else {
        HEIGHT / VAE_SCALE * (WIDTH / VAE_SCALE)
    };
    let b = x.len() / (c * p);
    assert_eq!(b * c * p, x.len(), "a tensor of {c} channels has whole planes");
    let mut y = vec![0.0f32; x.len()];
    for img in 0..b {
        for ch in 0..c {
            for q in 0..p {
                y[(img * p + q) * c + ch] = x[(img * c + ch) * p + q];
            }
        }
    }
    y
}

/// The moments of the encoder, pixel-major, from the mean and the raw log variance of the reference.
fn moments(mean: &[f32], logvar: &[f32]) -> Vec<f32> {
    let (mean, logvar) = (pixels(mean, LATENT), pixels(logvar, LATENT));
    mean.chunks(LATENT)
        .zip(logvar.chunks(LATENT))
        .flat_map(|(m, l)| m.iter().chain(l).copied())
        .collect()
}

/// The chunks of a prompt, as the family makes them.
fn chunks(family: &SdxlFamily, text: &str) -> Vec<ImageChunk> {
    family
        .prompt(text)
        .unwrap()
        .chunks
        .into_iter()
        .map(|c| ImageChunk {
            ids1: c.ids1.to_vec(),
            ids2: c.ids2.to_vec(),
            weights: c.weights.to_vec(),
        })
        .collect()
}

fn check(what: &str, got: &[f32], want: &[f32], bound: f32) {
    let err = rel(got, want);
    eprintln!("{what}: relative error {err:.3e}");
    assert!(err <= bound, "{what}: relative error {err} is past {bound}");
}

/// The add features of the batch of one image: the negative prompt, then the prompt. `pooled` is `[2][1280]`.
fn add_of(pooled: &[f32]) -> Vec<f32> {
    let ids = host::time_ids(WIDTH, HEIGHT);
    (0..2)
        .flat_map(|b| host::add_features(&pooled[b * POOLED..(b + 1) * POOLED], &ids))
        .collect()
}

/// The start latent of a run (spec 11.9b): the noise times `init_noise_sigma` at a strength of 1.0, and else the image
/// latent with the noise of the first step that runs.
fn start_of(schedule: &Schedule, begin: usize, image_latent: Option<&[f32]>, noise: &[f32]) -> Vec<f32> {
    match image_latent {
        None => noise.iter().map(|v| v * schedule.init_noise_sigma()).collect(),
        Some(latent) => schedule.add_noise(begin, latent, noise),
    }
}

/// What one step that runs gives: its 9-channel UNet input of the batch, the noise prediction, and the latent after
/// the sampler.
struct Seen<'a> {
    batch: &'a [f32],
    eps: &'a [f32],
    latent: &'a [f32],
}

/// The steps that run, from the start latent `x`: the UNet on the card, then the sampler on the host. Each step takes
/// the scaled latent, the latent mask and the masked latent (spec 11.9b), and no blend follows. `draw(i)` gives the
/// step noise of step `i` of the schedule. `each(k, seen)` sees step `k` of the steps that run.
#[allow(clippy::too_many_arguments)]
fn denoise(
    engine: &mut SdxlEngine,
    schedule: &Schedule,
    begin: usize,
    mut x: Vec<f32>,
    context: &[f32],
    add: &[f32],
    mask: &[u8],
    masked_latent: &[f32],
    draw: &dyn Fn(usize) -> Vec<f32>,
    each: &mut dyn FnMut(usize, Seen),
) -> Vec<f32> {
    let size = UnetSize {
        images: 1,
        h: HEIGHT / VAE_SCALE,
        w: WIDTH / VAE_SCALE,
        ctx: context.len() / (2 * CONTEXT),
    };
    let mut history = History::default();
    for i in begin..schedule.timesteps.len() {
        let scaled: Vec<f32> = x.iter().map(|v| v * schedule.input_scale(i)).collect();
        let one = host::inpaint_input(&scaled, mask, masked_latent);
        let batch: Vec<f32> = one.iter().chain(&one).copied().collect();
        let (eps, _) = engine.unet_forward(size, &batch, schedule.timesteps[i], context, add, false);
        let n = x.len();
        let guided: Vec<f32> = (0..n).map(|j| eps[j] + CFG * (eps[n + j] - eps[j])).collect();
        let noise = schedule.draws(i).then(|| draw(i));
        step(schedule, i, &mut x, &guided, noise.as_deref(), &mut history);
        each(
            i - begin,
            Seen {
                batch: &batch,
                eps: &eps,
                latent: &x,
            },
        );
    }
    x
}

/// The moments of the engine's encoder for the image and for the masked image.
struct Encoded<'a> {
    image: &'a [f32],
    masked: &'a [f32],
}

/// One run of the reference, block by block, with its draws.
fn oracle_run(
    engine: &mut SdxlEngine,
    inputs: &Tensors,
    t: &Tensors,
    encoded: &Encoded,
    mask: &[u8],
    run: (&str, ImageSampler, &str),
) {
    let (name, sampler, percent) = run;
    let key = format!("{name}.{percent}");
    let v = |k: &str| t.values(&format!("{key}.{k}"));
    let schedule = Schedule::new(sampler, ImageSpacing::Leading, STEPS as usize);
    assert_eq!(&schedule.timesteps[..], v("timesteps"), "{key}: the timesteps");
    let strength = t.f64s(&format!("{key}.strength"))[0];
    let begin = begin_index(STEPS, strength);
    assert_eq!(begin as f32, v("begin")[0], "{key}: the begin index");
    // The bound of the moments in xe_vae_encode.rs.
    let masked_latent = host::image_latent(encoded.masked, &pixels(v("noise.encode_masked"), LATENT));
    check(
        &format!("{key}: the masked image latent"),
        &masked_latent,
        &pixels(v("masked_latent"), LATENT),
        1e-3,
    );
    // At a strength of 1.0, diffusers does not encode the image (inpaint.py:805), so the run has no image latent.
    let image_latent = if strength == 1.0 {
        assert!(
            !t.has(&format!("{key}.noise.encode")) && !t.has(&format!("{key}.image_latent")),
            "{key}: diffusers encodes no image at a strength of 1.0"
        );
        None
    } else {
        let latent = host::image_latent(encoded.image, &pixels(v("noise.encode"), LATENT));
        check(
            &format!("{key}: the image latent"),
            &latent,
            &pixels(v("image_latent"), LATENT),
            1e-3,
        );
        Some(latent)
    };
    let start = pixels(v("noise.start"), LATENT);
    let x = start_of(&schedule, begin, image_latent.as_deref(), &start);
    let latents = v("latents");
    let size = start.len();
    check(
        &format!("{key}: the start latent"),
        &x,
        &pixels(&latents[..size], LATENT),
        1e-3,
    );
    let context = inputs.values("prompt_embeds");
    let add = add_of(inputs.values("pooled"));
    let steps = v("noise.step");
    // The step noise of the reference has one row for each step that runs.
    let draw = |i: usize| pixels(&steps[(i - begin) * size..(i - begin + 1) * size], LATENT);
    let (unet_in, unet_out, stepped) = (v("unet_in"), v("unet_out"), v("stepped"));
    let wide = size / LATENT * INPAINT_CHANNELS;
    let row = |all: &[f32], r: usize| pixels(&all[r * size..(r + 1) * size], LATENT);
    // A chained value carries the error of the encoder through each step, so the test prints the chained values. It
    // checks each block on the input of the reference below, as xe_oracle.rs checks its UNet.
    let mut each = |k: usize, seen: Seen| {
        let rows = &unet_in[k * 2 * wide..(k + 1) * 2 * wide];
        eprintln!(
            "{key} step {k}: the chained 9-channel UNet input: relative error {:.3e}",
            rel(seen.batch, &pixels(rows, INPAINT_CHANNELS))
        );
        let rows = &unet_out[k * 2 * size..(k + 1) * 2 * size];
        eprintln!(
            "{key} step {k}: the chained noise prediction: relative error {:.3e}",
            rel(seen.eps, &pixels(rows, LATENT))
        );
        eprintln!(
            "{key} step {k}: the chained latent after the step: relative error {:.3e}",
            rel(seen.latent, &row(latents, k + 1))
        );
    };
    let chained = denoise(
        engine,
        &schedule,
        begin,
        x,
        context,
        &add,
        mask,
        &masked_latent,
        &draw,
        &mut each,
    );
    // Each step on the input of the reference: the 9-channel input from the reference's latent and masked latent, the
    // UNet on the reference's input, then the sampler from the reference's latent before the step.
    let size_of = UnetSize {
        images: 1,
        h: HEIGHT / VAE_SCALE,
        w: WIDTH / VAE_SCALE,
        ctx: CHUNK_IDS,
    };
    let want_masked = pixels(v("masked_latent"), LATENT);
    let mut history = History::default();
    for k in 0..schedule.timesteps.len() - begin {
        let i = begin + k;
        let rows = pixels(&unet_in[k * 2 * wide..(k + 1) * 2 * wide], INPAINT_CHANNELS);
        let scaled: Vec<f32> = row(latents, k).iter().map(|v| v * schedule.input_scale(i)).collect();
        let one = host::inpaint_input(&scaled, mask, &want_masked);
        check(
            &format!("{key} step {k}: the 9-channel UNet input"),
            &one.iter().chain(&one).copied().collect::<Vec<f32>>(),
            &rows,
            1e-3,
        );
        let (eps, _) = engine.unet_forward(size_of, &rows, schedule.timesteps[i], context, &add, false);
        let want = &unet_out[k * 2 * size..(k + 1) * 2 * size];
        check(
            &format!("{key} step {k}: the noise prediction"),
            &eps,
            &pixels(want, LATENT),
            1e-3,
        );
        let guided: Vec<f32> = (0..size).map(|j| eps[j] + CFG * (eps[size + j] - eps[j])).collect();
        let mut x = row(latents, k);
        let noise = schedule.draws(i).then(|| draw(i));
        step(&schedule, i, &mut x, &guided, noise.as_deref(), &mut history);
        check(
            &format!("{key} step {k}: the sampler output"),
            &x,
            &row(stepped, k),
            1e-3,
        );
        // A 9-channel UNet does no blend (inpaint.py:1648), so the next latent is the sampler output.
        check(
            &format!("{key} step {k}: the latent after the step"),
            &x,
            &row(latents, k + 1),
            1e-3,
        );
        if i == schedule.timesteps.len() - 1 {
            check(
                &format!("{key}: the final latent"),
                &x,
                &pixels(v("final_latent"), LATENT),
                1e-3,
            );
        }
    }
    eprintln!(
        "{key}: the chained final latent: relative error {:.3e}",
        rel(&chained, &pixels(v("final_latent"), LATENT))
    );
    let (h, w) = (HEIGHT / VAE_SCALE, WIDTH / VAE_SCALE);
    // The VAE of the folder on the final latent of the reference. The decode is before the paste.
    let want_image = pixels(v("image"), 3);
    let image = engine.decode_latent(&pixels(v("final_latent"), LATENT), h, w);
    check(&format!("{key}: the VAE image"), &image, &want_image, 1e-3);
    // The decode of the chained latent carries the error of the encoder too, so the test checks its RGB bytes, as
    // xe_inpaint.rs does.
    let image = engine.decode_latent(&chained, h, w);
    eprintln!(
        "{key}: the chained VAE image: relative error {:.3e}",
        rel(&image, &want_image)
    );
    let rgb = host::rgb_bytes(&image);
    let want: Vec<u8> = v("rgb").iter().map(|v| *v as u8).collect();
    let off = rgb.iter().zip(&want).filter(|(a, b)| a.abs_diff(**b) > 1).count();
    eprintln!("{key}: {off} of {} RGB bytes differ by more than 1", rgb.len());
    assert!(
        off * 1000 < rgb.len(),
        "{key}: {off} of {} RGB bytes differ by more than 1",
        rgb.len()
    );
}

/// The job of a run through the image capability.
fn job(
    prompt: &[ImageChunk],
    negative: &[ImageChunk],
    rgb: &[u8],
    blurred: Option<&[u8]>,
    sampler: ImageSampler,
    strength: f64,
) -> ImageJob {
    ImageJob {
        prompt: ImagePrompt::Chunks {
            prompt: prompt.to_vec(),
            negative: negative.to_vec(),
            clip_skip: 2,
        },
        width: WIDTH as u32,
        height: HEIGHT as u32,
        steps: STEPS,
        cfg: CFG,
        sampler,
        spacing: ImageSpacing::Leading,
        count: 1,
        seed: SEED,
        previews: true,
        loras: Vec::new(),
        conditioning: None,
        init: Some(ImageInit {
            pixels: Arc::new(rgb.to_vec()),
            strength,
            mask: blurred.map(|b| Arc::new(b.to_vec())),
        }),
    }
}

#[test]
fn the_9_channel_unet_follows_diffusers_and_the_capability_runs_the_same_math() {
    let Some(dir) = model_dir() else { return };
    let (Some(inputs), Some(t)) = (golden("image_inputs.safetensors"), golden("inpaint9.safetensors")) else {
        return;
    };
    // The inpainting folder holds the tokenizer of SDXL base.
    let family = SdxlFamily::from_dir(&dir).unwrap();
    let rgb: Vec<u8> = inputs.values("init.rgb").iter().map(|v| *v as u8).collect();
    let blurred: Vec<u8> = inputs.values("mask.blurred").iter().map(|v| *v as u8).collect();
    let mask = latent_mask(&binary(&blurred), WIDTH as u32, HEIGHT as u32);
    let started = Instant::now();
    let mut engine = SdxlEngine::open(Config::folder(dir), &card_address()).unwrap();
    eprintln!("open: {:.1} s", started.elapsed().as_secs_f32());
    assert_eq!(engine.in_channels(), INPAINT_CHANNELS);
    assert!(engine.image_needs_mask());
    let (h, w) = (HEIGHT / VAE_SCALE, WIDTH / VAE_SCALE);
    // The VAE of the folder, madebyollin's sdxl-vae-fp16-fix, in f32 (spec 11.9). The bound of xe_vae_encode.rs.
    let image_moments = engine.encode_image(&unit(&rgb), HEIGHT, WIDTH);
    check(
        "the moments of the image",
        &image_moments,
        &moments(t.values("euler.s99.encode.mean"), t.values("euler.s99.encode.logvar")),
        1e-3,
    );
    let masked_values = masked(&rgb, &binary(&blurred));
    check(
        "the masked input",
        &masked_values,
        &pixels(inputs.values("masked.tensor"), 3),
        0.0,
    );
    let masked_moments = engine.encode_image(&masked_values, HEIGHT, WIDTH);
    check(
        "the moments of the masked image",
        &masked_moments,
        &moments(
            t.values("euler.s99.encode_masked.mean"),
            t.values("euler.s99.encode_masked.logvar"),
        ),
        1e-3,
    );
    let encoded = Encoded {
        image: &image_moments,
        masked: &masked_moments,
    };

    // Block by block, with the draws of the reference.
    for (name, sampler, percent, _) in RUNS {
        oracle_run(&mut engine, &inputs, &t, &encoded, &mask, (name, sampler, percent));
    }

    // A job without a mask is refused before any quantum (spec 11.9b).
    let (prompt, negative) = (chunks(&family, PROMPT), chunks(&family, NEGATIVE));
    let img2img = job(&prompt, &negative, &rgb, None, ImageSampler::Euler, 0.5);
    let error = engine.image_begin(0, &img2img).unwrap_err();
    assert!(error.message.contains("image and a mask"), "{}", error.message);
    let text = ImageJob { init: None, ..img2img };
    let error = engine.image_begin(0, &text).unwrap_err();
    assert!(error.message.contains("image and a mask"), "{}", error.message);

    // Through the capability, with the draws of the engine: the encode, then n steps of n, each with a preview.
    let (context, pooled) = engine.encode(&prompt, &negative, 2);
    let add = add_of(&pooled);
    for (name, sampler, percent, strength) in RUNS {
        let key = format!("{name}.{percent}");
        let n = STEPS - begin_index(STEPS, strength) as u32;
        let tm = Instant::now();
        engine
            .image_begin(0, &job(&prompt, &negative, &rgb, Some(&blurred), sampler, strength))
            .unwrap();
        let out = engine.image_step(0).unwrap();
        assert_eq!(
            (out.step, out.steps, out.previews.len()),
            (0, n, 0),
            "{key}: the encode"
        );
        // At a strength of 1.0 the encode quantum encodes only the masked image (inpaint.py:805).
        assert_eq!(
            engine.image_latents().is_some(),
            strength < 1.0,
            "{key}: the image is encoded only below a strength of 1.0"
        );
        let got_masked = engine.masked_latents().expect("a 9-channel job has a masked latent")[0].clone();
        let mut previews: Vec<DecodedImage> = Vec::new();
        for k in 1..=n {
            let mut out = engine.image_step(0).unwrap();
            assert_eq!((out.step, out.steps, out.previews.len()), (k, n, 1), "{key}: step {k}");
            previews.push(out.previews.remove(0));
        }
        assert!(
            engine.image_step(0).is_err(),
            "{key}: a step after the last one refuses"
        );
        let images = engine.image_decode(0).unwrap();
        let got = engine.latents().unwrap()[0].clone();
        engine.image_clear(0);
        eprintln!("{key}: the job took {} ms", tm.elapsed().as_millis());

        // The replay: the same calls, with the draws of the engine for image 0.
        let schedule = Schedule::new(sampler, ImageSpacing::Leading, STEPS as usize);
        let begin = begin_index(STEPS, strength);
        let count = image_moments.len() / 2;
        let masked_latent = host::image_latent(&masked_moments, &normal(SEED, 0, 0, Purpose::EncodeMasked, count));
        assert!(
            got_masked == masked_latent,
            "{key}: the masked latent of the capability differs from the replay"
        );
        let image_latent =
            (strength < 1.0).then(|| host::image_latent(&image_moments, &normal(SEED, 0, 0, Purpose::Encode, count)));
        let start = normal(SEED, 0, 0, Purpose::Start, count);
        let x = start_of(&schedule, begin, image_latent.as_deref(), &start);
        let draw = |i: usize| normal(SEED, 0, i as u32, Purpose::Step, count);
        let mut each = |k: usize, seen: Seen| {
            let want = host::preview(seen.latent, w, h);
            assert!(
                previews[k].pixels == want.pixels,
                "{key} step {k}: the preview is not that of the latent after the step"
            );
        };
        let want = denoise(
            &mut engine,
            &schedule,
            begin,
            x,
            &context,
            &add,
            &mask,
            &masked_latent,
            &draw,
            &mut each,
        );
        assert!(
            got == want,
            "{key}: the capability gives other bits than the replay, relative error {:e}",
            rel(&got, &want)
        );
        let rgb_want = host::rgb_bytes(&engine.decode_latent(&want, h, w));
        assert!(
            images[0].pixels == rgb_want,
            "{key}: the decode of the capability differs from the replay"
        );
        eprintln!("{key}: the capability gives the bits and the previews of the replay");
    }
}
