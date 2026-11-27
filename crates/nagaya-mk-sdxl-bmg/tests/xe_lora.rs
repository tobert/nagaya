//! The LoRA deltas on a B70 card against the LoRA oracle of diffusers (`tools/sdxl_lora_reference.py`, spec 11.9a). The
//! test checks the text encodings, one UNet step, and that clearing the slot gives back the base encodings.
//!
//! It needs a card, `SDXL_MODEL_DIR` and the LoRA oracle in `SDXL_LORA_GOLDEN_DIR` or `target/sdxl-lora-golden`. Stop
//! `nagaya.service` first (spec 3.1). Run it alone:
//! `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_lora -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::{golden_in, rel};
use nagaya_abi::{Image, ImageChunk, ImageJob, ImagePrompt, ImageSampler, ImageSpacing, LoraInput};
use nagaya_family_sdxl::SdxlFamily;
use nagaya_mk_sdxl_bmg::host;
use nagaya_mk_sdxl_bmg::model::UnetSize;
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};

const PROMPT: &str = "a red fox in a snowy forest, watercolor";
const NEGATIVE: &str = "blurry, low quality";
/// The weight of the oracle's LoRA: `sdxl_lora_reference.py` loads it at 0.8.
const WEIGHT: f32 = 0.8;

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

/// The directory of the LoRA oracle: `SDXL_LORA_GOLDEN_DIR`, or `target/sdxl-lora-golden` of the workspace.
fn oracle_dir() -> PathBuf {
    std::env::var_os("SDXL_LORA_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/sdxl-lora-golden"))
}

/// The PCI address of the first B70 card.
fn card_address() -> String {
    nagaya_mk_bmg::test_cards::card()
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

#[test]
fn a_lora_on_the_card_moves_the_encodings_and_a_unet_step_as_diffusers_does() {
    let Some(dir) = model_dir() else { return };
    let oracle = oracle_dir();
    let (Some(encode), Some(unet)) = (
        golden_in(&oracle, "lora.encode.safetensors"),
        golden_in(&oracle, "lora.unet.safetensors"),
    ) else {
        return;
    };
    let Ok(file) = std::fs::read(oracle.join("lora.safetensors")) else {
        eprintln!(
            "skipped: {} has no lora.safetensors; run tools/sdxl_lora_reference.py",
            oracle.display()
        );
        return;
    };
    let family = SdxlFamily::from_dir(&dir).unwrap();
    let mut engine = SdxlEngine::open(Config::folder(dir.clone()), &card_address()).unwrap();
    let (prompt, negative) = (chunks(&family, PROMPT), chunks(&family, NEGATIVE));
    let id = format!("blake3:{}", blake3::hash(&file).to_hex());
    let (h, w) = (40, 32);
    let job = ImageJob {
        prompt: ImagePrompt::Chunks {
            prompt: prompt.clone(),
            negative: negative.clone(),
            clip_skip: 2,
        },
        width: 256,
        height: 320,
        steps: 1,
        cfg: 7.0,
        sampler: ImageSampler::Euler,
        spacing: ImageSpacing::Leading,
        count: 1,
        seed: 1234,
        previews: false,
        loras: vec![LoraInput {
            id,
            weight: WEIGHT,
            file: Arc::new(file),
        }],
        conditioning: None,
        init: None,
    };
    // The job attaches the LoRA, then encodes with it.
    engine.image_begin(0, &job).unwrap();

    // The text encoders, with the deltas of both encoders and of the UNet's cross attention.
    let (context, pooled) = engine.encode(&prompt, &negative, 2);
    let want_context = [encode.values("neg.prompt_embeds"), encode.values("pos.prompt_embeds")].concat();
    let want_pooled = [encode.values("neg.pooled"), encode.values("pos.pooled")].concat();
    check("the context with the LoRA", &context, &want_context, 1e-3);
    check("the pooled outputs with the LoRA", &pooled, &want_pooled, 1e-3);

    // One UNet step with the LoRA. The input is the latent of the oracle for both rows, in pixel-major order.
    let latent = unet.values("latent");
    let x_in = pixels(&[latent, latent].concat(), 2, LATENT, h, w);
    let size = UnetSize {
        images: 1,
        h,
        w,
        ctx: CHUNK_IDS,
    };
    let ids = host::time_ids(w * 8, h * 8);
    let add: Vec<f32> = (0..2)
        .flat_map(|b| host::add_features(&pooled[b * POOLED..(b + 1) * POOLED], &ids))
        .collect();
    let (eps, _) = engine.unet_forward(size, &x_in, 999.0, &context, &add, false);
    let want = pixels(unet.values("unet_out"), 2, LATENT, h, w);
    check("the UNet step with the LoRA", &eps, &want, 1e-3);
    let base = pixels(unet.values("base_out"), 2, LATENT, h, w);
    let moved = rel(&eps, &base);
    assert!(
        moved > 1e-3,
        "the LoRA changed the UNet step by {moved}, which is no change"
    );

    // Clearing the slot releases the deltas: the encodings are the base encodings again.
    engine.image_clear(0);
    let (base_context, _) = engine.encode(&prompt, &negative, 2);
    let want_base = [
        encode.values("neg.base_prompt_embeds"),
        encode.values("pos.base_prompt_embeds"),
    ]
    .concat();
    check("the context after the slot is cleared", &base_context, &want_base, 1e-3);
}
