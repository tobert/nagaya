//! The time of each op of a UNet step of one 1024 by 1024 image on a B70 card, for the work on speed. It checks no
//! number. Stop `nagaya.service` first (spec 3.1). Run it alone:
//! `cargo test --release -p nagaya-mk-sdxl-bmg --features xe --test xe_profile -- --nocapture`.
#![cfg(feature = "xe")]

use std::path::PathBuf;
use std::time::Instant;

use nagaya_mk_bmg::protocol::*;
use nagaya_mk_sdxl_bmg::model::UnetSize;
use nagaya_mk_sdxl_bmg::{Config, SdxlEngine};

#[test]
fn the_time_of_each_op_of_a_unet_step_at_1024() {
    let Some(dir) = std::env::var_os("SDXL_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("skipped: set SDXL_MODEL_DIR");
        return;
    };
    let card = std::env::var("SDXL_CARD").unwrap_or_else(|_| "0000:03:00.0".to_owned());
    let mut engine = SdxlEngine::open(Config::folder(dir), &card).unwrap();
    let size = UnetSize {
        images: 1,
        h: 128,
        w: 128,
        ctx: 77,
    };
    let x = vec![0.1f32; size.rows(0) * 4];
    let ctx = vec![0.01f32; size.batch() * 77 * 2048];
    let add = vec![0.0f32; size.batch() * 2816];
    let t = Instant::now();
    engine.unet_forward(size, &x, 500.0, &ctx, &add, false);
    eprintln!(
        "one UNet step at 1024 by 1024, batch 2, in runs: {} ms",
        t.elapsed().as_millis()
    );
    let names = [
        (OP_CONV2D, "conv2d (and linear)"),
        (OP_GROUPNORM, "groupnorm"),
        (OP_SDPA, "sdpa"),
        (OP_UPSAMPLE2X, "upsample2x"),
        (OP_GEGLU_HALF, "geglu_half"),
        (OP_COPY_COLS, "copy_cols"),
        (OP_ADD_IMAGE_ROW, "add_image_row"),
        (OP_LAYERNORM, "layernorm"),
        (OP_COPY, "copy"),
        (OP_SILU_MUL, "silu"),
    ];
    let profile = engine.profile_unet(size);
    let total: f64 = profile.values().map(|(_, ms)| ms).sum();
    for (op, (count, ms)) in &profile {
        let name = names.iter().find(|(o, _)| o == op).map_or("other", |(_, n)| n);
        eprintln!("{name:>20}: {count:5} tasks, {ms:9.1} ms, {:5.1} %", 100.0 * ms / total);
    }
    eprintln!("sum of single tasks: {total:.1} ms");
    let latent = vec![0.1f32; 128 * 128 * 4];
    let t = Instant::now();
    engine.decode_latent(&latent, 128, 128);
    eprintln!("VAE decode of 1024 by 1024: {} ms", t.elapsed().as_millis());
}
