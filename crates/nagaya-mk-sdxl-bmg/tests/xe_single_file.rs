//! The engine opened from the single file of SDXL base 1.0 on a B70 card: its identity names the file, and its UNet
//! gives the same bits as the engine opened from the folder, because both hold the same F16 weights. The VAEs differ:
//! the single file holds the VAE of `vae_1_0/`.
//!
//! It needs a card, `SDXL_MODEL_DIR` and `SDXL_SINGLE_FILE`. Stop `nagaya.service` first (spec 3.1). Run it alone:
//! `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_single_file -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::path::PathBuf;

use nagaya_abi::Engine;
use nagaya_mk_sdxl_bmg::model::UnetSize;
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};

fn paths() -> Option<(PathBuf, PathBuf)> {
    let dir = std::env::var_os("SDXL_MODEL_DIR").map(PathBuf::from);
    let file = std::env::var_os("SDXL_SINGLE_FILE").map(PathBuf::from);
    if let (Some(dir), Some(file)) = (dir, file)
        && dir.join("unet").is_dir()
        && file.is_file()
    {
        return Some((dir, file));
    }
    let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
    assert!(
        !require,
        "NAGAYA_REQUIRE_MODELS is set: set SDXL_MODEL_DIR and SDXL_SINGLE_FILE"
    );
    eprintln!("skipped: SDXL_MODEL_DIR or SDXL_SINGLE_FILE does not name the SDXL base 1.0 checkpoint");
    None
}

/// One UNet step of 2 rows at 256 by 320, from a fixed input.
fn step(engine: &mut SdxlEngine) -> Vec<f32> {
    let (h, w) = (40, 32);
    let size = UnetSize {
        images: 1,
        h,
        w,
        ctx: CHUNK_IDS,
    };
    let x: Vec<f32> = (0..2 * h * w * LATENT)
        .map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let context: Vec<f32> = (0..2 * CHUNK_IDS * CONTEXT)
        .map(|i| ((i * 104_729 % 997) as f32 / 997.0) - 0.5)
        .collect();
    let add: Vec<f32> = (0..2 * ADD_IN).map(|i| ((i * 31 % 89) as f32 / 89.0) - 0.5).collect();
    engine.unet_forward(size, &x, 500.0, &context, &add, false).0
}

#[test]
fn the_single_file_names_the_model_and_gives_the_unet_bits_of_the_folder() {
    let Some((dir, file)) = paths() else { return };
    let card = std::env::var("SDXL_CARD").unwrap_or_else(|_| "0000:03:00.0".to_owned());
    let folder = {
        let mut engine = SdxlEngine::open(Config::folder(dir.clone()), &card).unwrap();
        step(&mut engine)
        // The drop frees the card for the next engine.
    };
    let mut engine = SdxlEngine::open(
        Config {
            model_dir: dir,
            checkpoint: Some(file),
        },
        &card,
    )
    .unwrap();
    assert_eq!(engine.identity().model, "sd_xl_base_1.0");
    let single = step(&mut engine);
    let same = folder
        .iter()
        .zip(&single)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    eprintln!("{same} of {} values with the same bits", folder.len());
    assert_eq!(
        same,
        folder.len(),
        "the UNet of the single file gives the bits of the folder"
    );
}
