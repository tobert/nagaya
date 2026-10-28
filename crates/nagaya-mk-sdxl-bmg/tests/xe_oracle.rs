//! The engine on a B70 card against the oracle of diffusers (`tools/sdxl_golden.sh`), block by block: the text
//! encoders, each run of a UNet step, the VAE decode, and four Euler steps from the start latent of the oracle. Then
//! one real image through the image capability.
//!
//! It needs a card, `SDXL_MODEL_DIR` and the oracle. Stop `nagaya.service` first (spec 3.1). Run it alone:
//! `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_oracle -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{Tensors, golden, rel};
use nagaya_abi::{Image, ImageChunk, ImageJob, ImageSampler};
use nagaya_mk_sdxl_bmg::host;
use nagaya_mk_sdxl_bmg::model::UnetSize;
use nagaya_mk_sdxl_bmg::sampler::{History, Schedule, step};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};
use nagaya_text_sdxl::SdxlFamily;

const PROMPT: &str = "a red fox in a snowy forest, watercolor";
const NEGATIVE: &str = "blurry, low quality";

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

/// NCHW to pixel-major: [b][c][h][w] to [b][h][w][c].
fn pixels(x: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for img in 0..b {
        for ch in 0..c {
            for p in 0..h * w {
                y[(img * h * w + p) * c + ch] = x[(img * c + ch) * h * w + p];
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
    assert!(err < bound, "{what}: relative error {err} is past {bound}");
}

struct Oracle {
    encode: Tensors,
    samplers: Tensors,
    parts: Tensors,
}

#[test]
fn the_engine_follows_the_oracle_block_by_block_and_makes_an_image() {
    let Some(dir) = model_dir() else { return };
    let (Some(encode), Some(samplers), Some(parts)) = (
        golden("encode.safetensors"),
        golden("samplers.safetensors"),
        golden("parts.safetensors"),
    ) else {
        return;
    };
    let oracle = Oracle {
        encode,
        samplers,
        parts,
    };
    let family = SdxlFamily::from_dir(&dir).unwrap();
    let started = Instant::now();
    let mut engine = SdxlEngine::open(Config::folder(dir.clone()), &card_address()).unwrap();
    eprintln!("open: {:.1} s", started.elapsed().as_secs_f32());

    // The text encoders.
    let (prompt, negative) = (chunks(&family, PROMPT), chunks(&family, NEGATIVE));
    for skip in [1usize, 2] {
        let t = Instant::now();
        let (context, pooled) = engine.encode(&prompt, &negative, skip);
        eprintln!("encode with clip_skip {skip}: {} ms", t.elapsed().as_millis());
        check(
            &format!("context, clip_skip {skip}"),
            &context,
            oracle.encode.values(&format!("clip_skip{skip}.prompt_embeds")),
            1e-3,
        );
        check(
            &format!("pooled, clip_skip {skip}"),
            &pooled,
            oracle.encode.values(&format!("clip_skip{skip}.pooled")),
            1e-3,
        );
    }

    // Step 0 of Euler, run by run.
    let (h, w) = (40, 32);
    let size = UnetSize {
        images: 1,
        h,
        w,
        ctx: CHUNK_IDS,
    };
    let x_in = pixels(oracle.parts.values("unet.in"), 2, LATENT, h, w);
    let context = oracle.parts.values("unet.encoder_hidden_states").to_vec();
    let pooled = oracle.encode.values("clip_skip1.pooled");
    let ids = host::time_ids(w * 8, h * 8);
    let add: Vec<f32> = (0..2)
        .flat_map(|b| host::add_features(&pooled[b * POOLED..(b + 1) * POOLED], &ids))
        .collect();
    let t0 = oracle.samplers.values("euler.timesteps")[0];
    let t = Instant::now();
    let (eps, probes) = engine.unet_forward(size, &x_in, t0, &context, &add, true);
    eprintln!("one UNet step of 2 rows at 256 by 320: {} ms", t.elapsed().as_millis());
    let stages = [
        ("unet.conv_in", 320, h, w),
        ("unet.down0", 320, h / 2, w / 2),
        ("unet.down1", 640, h / 4, w / 4),
        ("unet.down2", 1280, h / 4, w / 4),
        ("unet.mid", 1280, h / 4, w / 4),
        ("unet.up0", 1280, h / 2, w / 2),
        ("unet.up1", 640, h, w),
        ("unet.up2", 320, h, w),
        ("unet.conv_out", 4, h, w),
    ];
    for (k, (name, c, hh, ww)) in stages.iter().enumerate() {
        let want = pixels(oracle.parts.values(name), 2, *c, *hh, *ww);
        check(name, &probes[k], &want, 1e-3);
    }
    check(
        "the noise prediction of step 0",
        &eps,
        &pixels(&oracle.samplers.values("euler.unet_out")[..2 * 4 * h * w], 2, 4, h, w),
        1e-3,
    );

    // The VAE decode of the final latent of Euler.
    let latent = pixels(oracle.samplers.values("euler.final_latent"), 1, LATENT, h, w);
    let t = Instant::now();
    let image = engine.decode_latent(&latent, h, w);
    eprintln!("VAE decode of 256 by 320: {} ms", t.elapsed().as_millis());
    check(
        "the VAE image",
        &image,
        &pixels(oracle.samplers.values("euler.image"), 1, 3, 8 * h, 8 * w),
        1e-3,
    );

    // Four Euler steps from the start latent of the oracle, then the decode.
    let schedule = Schedule::new(ImageSampler::Euler, 4);
    let mut x: Vec<f32> = pixels(oracle.samplers.values("start"), 1, LATENT, h, w)
        .iter()
        .map(|v| v * schedule.init_noise_sigma())
        .collect();
    let mut history = History::default();
    for i in 0..4 {
        let scaled: Vec<f32> = x.iter().map(|v| v * schedule.input_scale(i)).collect();
        let batch: Vec<f32> = scaled.iter().chain(&scaled).copied().collect();
        let (eps, _) = engine.unet_forward(size, &batch, schedule.timesteps[i], &context, &add, false);
        let n = x.len();
        let guided: Vec<f32> = (0..n).map(|j| eps[j] + 7.0 * (eps[n + j] - eps[j])).collect();
        step(&schedule, i, &mut x, &guided, None, &mut history);
    }
    check(
        "the final latent of four Euler steps",
        &x,
        &pixels(oracle.samplers.values("euler.final_latent"), 1, LATENT, h, w),
        1e-3,
    );
    let rgb = host::rgb_bytes(&engine.decode_latent(&x, h, w));
    let want: Vec<u8> = oracle.samplers.values("euler.rgb").iter().map(|v| *v as u8).collect();
    let off = rgb.iter().zip(&want).filter(|(a, b)| a.abs_diff(**b) > 1).count();
    eprintln!(
        "RGB bytes of four Euler steps: {off} of {} differ by more than 1",
        rgb.len()
    );
    assert!(
        off * 1000 < rgb.len(),
        "{off} of {} RGB bytes differ by more than 1",
        rgb.len()
    );

    // A real image through the capability: 512 by 512, 20 steps. The same seed gives the same bytes.
    let job = ImageJob {
        prompt: prompt.clone(),
        negative: negative.clone(),
        clip_skip: 1,
        width: 512,
        height: 512,
        steps: 20,
        cfg: 7.0,
        sampler: ImageSampler::DpmPp2mKarras,
        count: 1,
        seed: 1234,
        previews: true,
    };
    let mut made = Vec::new();
    for _ in 0..2 {
        let t = Instant::now();
        engine.image_begin(0, &job).unwrap();
        let begin = t.elapsed().as_millis();
        let t = Instant::now();
        let mut last = None;
        for _ in 0..job.steps {
            last = Some(engine.image_step(0).unwrap());
        }
        let steps = t.elapsed().as_millis();
        let t = Instant::now();
        let images = engine.image_decode(0).unwrap();
        eprintln!(
            "512 by 512, 20 steps: begin {begin} ms, steps {steps} ms ({} ms each), decode {} ms",
            steps / 20,
            t.elapsed().as_millis()
        );
        let last = last.unwrap();
        assert_eq!((last.step, last.steps, last.previews.len()), (20, 20, 1));
        assert_eq!((last.previews[0].width, last.previews[0].height), (64, 64));
        engine.image_clear(0);
        made.push(images);
    }
    assert_eq!(made[0], made[1], "the same job and seed give the same bytes");
    let image = &made[0][0];
    assert_eq!((image.width, image.height, image.rgb.len()), (512, 512, 512 * 512 * 3));
    if let Some(out) = std::env::var_os("SDXL_IMAGE_OUT") {
        let ppm = [b"P6 512 512 255\n".to_vec(), image.rgb.clone()].concat();
        std::fs::write(&out, ppm).unwrap();
        eprintln!("wrote {}", PathBuf::from(out).display());
    }
}
