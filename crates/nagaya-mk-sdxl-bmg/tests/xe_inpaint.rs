//! Inpainting with the latent blend on a B70 card against diffusers (`tools/sdxl_inpaint_golden.sh`, spec 11.9b).
//! Each run of `inpaint.safetensors` goes block by block with the draws of the reference: the image latent, the start
//! latent, each UNet step, each sampler step before the blend, each `known`, each blend, and the VAE decode. Then the
//! same run goes through the image capability with the draws of the engine, and the test replays it with the same
//! calls. The two must give the same bits and the same previews, so the capability runs the math that the oracle checks.
//!
//! It needs a card, `SDXL_MODEL_DIR` and the reference files. Stop `nagaya.service` first (spec 3.1). Run it alone:
//! `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_inpaint -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use common::{Tensors, golden, rel};
use nagaya_abi::{DecodedImage, Image, ImageChunk, ImageInit, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_family_sdxl::SdxlFamily;
use nagaya_family_sdxl::inputs::{binary, latent_mask, unit};
use nagaya_mk_sdxl_bmg::host;
use nagaya_mk_sdxl_bmg::model::UnetSize;
use nagaya_mk_sdxl_bmg::noise::{Purpose, normal};
use nagaya_mk_sdxl_bmg::sampler::{History, Schedule, begin_index, blend, image_start, step};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};

/// The request of the reference (`tools/sdxl_inpaint_golden.sh`).
const PROMPT: &str = "a red fox in a snowy forest, watercolor";
const NEGATIVE: &str = "blurry, low quality";
const WIDTH: usize = 256;
const HEIGHT: usize = 320;
const STEPS: u32 = 4;
const SEED: u64 = 1234;
const CFG: f32 = 7.0;

/// The runs of `inpaint.safetensors`: the sampler, the strength in percent, and the strength.
const RUNS: [(&str, ImageSampler, &str, f64); 6] = [
    ("euler", ImageSampler::Euler, "s99", 0.99),
    ("euler_a", ImageSampler::EulerAncestral, "s99", 0.99),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s99", 0.99),
    ("dpmpp_2m_sde_karras", ImageSampler::DpmPp2mSdeKarras, "s99", 0.99),
    ("euler", ImageSampler::Euler, "s100", 1.0),
    ("dpmpp_2m_karras", ImageSampler::DpmPp2mKarras, "s100", 1.0),
];

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
    nagaya_mk_bmg::test_cards::card()
}

/// NCHW to pixel-major: [b][c][h][w] to [b][h][w][c]. A tensor of 3 channels is the decoded image, and any other is
/// a latent.
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

/// What one step that runs gives: its UNet input of the batch, the noise prediction, the sampler output before the
/// blend, `known`, and the latent after the blend.
struct Seen<'a> {
    batch: &'a [f32],
    eps: &'a [f32],
    stepped: &'a [f32],
    known: &'a [f32],
    latent: &'a [f32],
}

/// The steps that run, from the start latent `x`: the UNet on the card, then the sampler and the blend on the host.
/// `draw(i)` gives the step noise of step `i` of the schedule, and `known(i)` the `known` after it. `each(k, seen)`
/// sees step `k` of the steps that run.
#[allow(clippy::too_many_arguments)]
fn denoise(
    engine: &mut SdxlEngine,
    schedule: &Schedule,
    begin: usize,
    mut x: Vec<f32>,
    context: &[f32],
    add: &[f32],
    mask: &[u8],
    draw: &dyn Fn(usize) -> Vec<f32>,
    known: &dyn Fn(usize) -> Vec<f32>,
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
        let batch: Vec<f32> = scaled.iter().chain(&scaled).copied().collect();
        let (eps, _) = engine.unet_forward(size, &batch, schedule.timesteps[i], context, add, false);
        let n = x.len();
        let guided: Vec<f32> = (0..n).map(|j| eps[j] + CFG * (eps[n + j] - eps[j])).collect();
        let noise = schedule.draws(i).then(|| draw(i));
        step(schedule, i, &mut x, &guided, noise.as_deref(), &mut history);
        let stepped = x.clone();
        let known = known(i);
        blend(&mut x, &known, mask);
        each(
            i - begin,
            Seen {
                batch: &batch,
                eps: &eps,
                stepped: &stepped,
                known: &known,
                latent: &x,
            },
        );
    }
    x
}

/// The relative error of `got` where the latent mask repaints, and where it keeps. Both are relative to the largest
/// absolute value of `want`.
fn split(got: &[f32], want: &[f32], mask: &[u8]) -> (f32, f32) {
    let scale = want.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
    let mut out = [0.0f32; 2];
    for (j, (g, w)) in got.iter().zip(want).enumerate() {
        let side = usize::from(mask[j / LATENT] == 0);
        out[side] = out[side].max((g - w).abs() / scale);
    }
    (out[0], out[1])
}

/// One run of the reference, block by block, with its draws.
fn oracle_run(
    engine: &mut SdxlEngine,
    inputs: &Tensors,
    t: &Tensors,
    moments: &[f32],
    mask: &[u8],
    run: (&str, ImageSampler, &str),
) {
    let (name, sampler, percent) = run;
    let key = format!("{name}.{percent}");
    let v = |k: &str| t.values(&format!("{key}.{k}"));
    let schedule = Schedule::new(sampler, ImageSpacing::Leading, STEPS as usize);
    let strength = t.f64s(&format!("{key}.strength"))[0];
    let begin = begin_index(STEPS, strength);
    assert_eq!(begin as f32, v("begin")[0], "{key}: the begin index");
    let latent = host::image_latent(moments, &pixels(v("noise.encode"), LATENT));
    // The bound of the moments in xe_vae_encode.rs.
    check(
        &format!("{key}: the image latent"),
        &latent,
        &pixels(v("image_latent"), LATENT),
        1e-3,
    );
    let start = pixels(v("noise.start"), LATENT);
    let x = image_start(&schedule, begin, &latent, &start, true, strength);
    let latents = v("latents");
    let size = latent.len();
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
    let known_of = |i: usize| schedule.known(i, &latent, &start);
    let (unet_in, unet_out, stepped, known) = (v("unet_in"), v("unet_out"), v("stepped"), v("known"));
    let row = |all: &[f32], r: usize| pixels(&all[r * size..(r + 1) * size], LATENT);
    // The bounds of xe_oracle.rs and xe_img2img.rs: f32 sums in another order over the whole UNet, and the latent
    // after each step.
    let mut each = |k: usize, seen: Seen| {
        // A chained value carries the error of the image latent through each step. Over the 3 steps of a strength of
        // 0.99 it can pass 1e-3, so the test prints the chained values. It checks each block on the input of the
        // reference below, as xe_oracle.rs checks its UNet.
        let rows = &unet_in[k * 2 * size..(k + 1) * 2 * size];
        eprintln!(
            "{key} step {k}: the chained UNet input: relative error {:.3e}",
            rel(seen.batch, &pixels(rows, LATENT))
        );
        let rows = &unet_out[k * 2 * size..(k + 1) * 2 * size];
        eprintln!(
            "{key} step {k}: the chained noise prediction: relative error {:.3e}",
            rel(seen.eps, &pixels(rows, LATENT))
        );
        let want = row(stepped, k);
        let (repaint, keep) = split(seen.stepped, &want, mask);
        eprintln!(
            "{key} step {k}: the chained sampler output before the blend: relative error {:.3e} ({repaint:.3e} where \
             the mask repaints, {keep:.3e} where it keeps)",
            rel(seen.stepped, &want)
        );
        check(&format!("{key} step {k}: known"), seen.known, &row(known, k), 1e-3);
        let want = row(latents, k + 1);
        let (repaint, keep) = split(seen.latent, &want, mask);
        eprintln!(
            "{key} step {k}: the chained latent after the blend: relative error {:.3e} ({repaint:.3e} where the mask \
             repaints, {keep:.3e} where it keeps)",
            rel(seen.latent, &want)
        );
    };
    let chained = denoise(
        engine, &schedule, begin, x, context, &add, mask, &draw, &known_of, &mut each,
    );
    // Each step on the input of the reference: the UNet, as xe_oracle.rs checks step 0, then the sampler from the
    // reference's latent before the step, then the blend.
    let size_of = UnetSize {
        images: 1,
        h: HEIGHT / VAE_SCALE,
        w: WIDTH / VAE_SCALE,
        ctx: CHUNK_IDS,
    };
    let mut history = History::default();
    for k in 0..schedule.timesteps.len() - begin {
        let i = begin + k;
        let rows = &unet_in[k * 2 * size..(k + 1) * 2 * size];
        let (eps, _) = engine.unet_forward(
            size_of,
            &pixels(rows, LATENT),
            schedule.timesteps[i],
            context,
            &add,
            false,
        );
        let rows = &unet_out[k * 2 * size..(k + 1) * 2 * size];
        check(
            &format!("{key} step {k}: the noise prediction"),
            &eps,
            &pixels(rows, LATENT),
            1e-3,
        );
        let guided: Vec<f32> = (0..size).map(|j| eps[j] + CFG * (eps[size + j] - eps[j])).collect();
        let mut x = row(latents, k);
        let noise = schedule.draws(i).then(|| draw(i));
        step(&schedule, i, &mut x, &guided, noise.as_deref(), &mut history);
        check(
            &format!("{key} step {k}: the sampler output before the blend"),
            &x,
            &row(stepped, k),
            1e-3,
        );
        blend(&mut x, &known_of(i), mask);
        check(&format!("{key} step {k}: the blend"), &x, &row(latents, k + 1), 1e-3);
        // The blend of the last step on the input of the reference is the final latent.
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
    // The VAE on the final latent of the reference, as xe_oracle.rs checks it. The decode is before the paste.
    let want_image = pixels(v("image"), 3);
    let image = engine.decode_latent(&pixels(v("final_latent"), LATENT), h, w);
    check(&format!("{key}: the VAE image"), &image, &want_image, 1e-3);
    // The decode of the chained latent carries the error of the encoder too, so the test checks its RGB bytes, as
    // xe_img2img.rs does.
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
    blurred: &[u8],
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
            mask: Some(Arc::new(blurred.to_vec())),
        }),
    }
}

#[test]
fn inpainting_follows_diffusers_and_the_capability_runs_the_same_math() {
    let Some(dir) = model_dir() else { return };
    let (Some(inputs), Some(t)) = (golden("image_inputs.safetensors"), golden("inpaint.safetensors")) else {
        return;
    };
    let family = SdxlFamily::from_dir(&dir).unwrap();
    let rgb: Vec<u8> = inputs.values("init.rgb").iter().map(|v| *v as u8).collect();
    let blurred: Vec<u8> = inputs.values("mask.blurred").iter().map(|v| *v as u8).collect();
    let mask = latent_mask(&binary(&blurred), WIDTH as u32, HEIGHT as u32);
    let want_mask: Vec<u8> = inputs.values("mask.latent").iter().map(|v| *v as u8).collect();
    assert_eq!(mask, want_mask, "the latent mask");
    let started = Instant::now();
    let mut engine = SdxlEngine::open(Config::folder(dir), &card_address()).unwrap();
    eprintln!("open: {:.1} s", started.elapsed().as_secs_f32());
    // SDXL base has a UNet of 4 input channels, so it takes a job without a mask (spec 11.9b).
    assert_eq!((engine.in_channels(), engine.image_needs_mask()), (LATENT, false));
    let (h, w) = (HEIGHT / VAE_SCALE, WIDTH / VAE_SCALE);
    let moments = engine.encode_image(&unit(&rgb), HEIGHT, WIDTH);

    // Block by block, with the draws of the reference.
    for (name, sampler, percent, _) in RUNS {
        oracle_run(&mut engine, &inputs, &t, &moments, &mask, (name, sampler, percent));
    }

    // Through the capability, with the draws of the engine: the encode, then n steps of n, each with a preview.
    let (prompt, negative) = (chunks(&family, PROMPT), chunks(&family, NEGATIVE));
    let (context, pooled) = engine.encode(&prompt, &negative, 2);
    let add = add_of(&pooled);
    for (name, sampler, percent, strength) in RUNS {
        let key = format!("{name}.{percent}");
        let n = STEPS - begin_index(STEPS, strength) as u32;
        let tm = Instant::now();
        engine
            .image_begin(0, &job(&prompt, &negative, &rgb, &blurred, sampler, strength))
            .unwrap();
        let out = engine.image_step(0).unwrap();
        assert_eq!(
            (out.step, out.steps, out.previews.len()),
            (0, n, 0),
            "{key}: the encode"
        );
        // A 4-channel UNet blends, so the encode quantum does not encode the masked image (spec 11.9b).
        assert!(
            engine.masked_latents().is_none(),
            "{key}: a 4-channel UNet has no masked latent"
        );
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
        let count = moments.len() / 2;
        let latent = host::image_latent(&moments, &normal(SEED, 0, 0, Purpose::Encode, count));
        let start = normal(SEED, 0, 0, Purpose::Start, count);
        let x = image_start(&schedule, begin, &latent, &start, true, strength);
        let draw = |i: usize| normal(SEED, 0, i as u32, Purpose::Step, count);
        let known_of = |i: usize| schedule.known(i, &latent, &start);
        // A preview shows the latent after the blend (spec 11.9b, "The blend").
        let mut each = |k: usize, seen: Seen| {
            let want = host::preview(seen.latent, w, h);
            assert!(
                previews[k].pixels == want.pixels,
                "{key} step {k}: the preview is not that of the latent after the blend"
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
            &draw,
            &known_of,
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
