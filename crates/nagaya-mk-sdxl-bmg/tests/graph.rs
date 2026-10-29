//! The graphs against the tensor headers of the checkpoint, with no card: each tensor that a graph names exists with
//! its shape, and each run fits in a ring. The test reads `SDXL_MODEL_DIR`, and skips when it is absent.

use std::path::PathBuf;

use nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD;
use nagaya_mk_sdxl_bmg::model::{Carve, TextBufs, UnetBufs, UnetSize, VaeBufs, text_encoder, unet, vae};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::weights::{Count, Weights};

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

#[test]
fn every_graph_names_tensors_of_the_checkpoint_and_fits_its_rings() {
    let Some(dir) = model_dir() else { return };
    let mut count = Count::default();
    let weights = Weights::load(&nagaya_mk_sdxl_bmg::Config::folder(dir), &mut count).unwrap();
    assert!(
        count.bytes > 6 << 30,
        "the weights of SDXL are about 7 GB, and the count is {}",
        count.bytes
    );
    let mut carve = Carve::new(0, u64::MAX / 2);
    let text = TextBufs::carve(&mut carve, 2 * MAX_CHUNKS);
    let mut sizes = Vec::new();
    for second in [false, true] {
        for skip in [2, 3] {
            for run in text_encoder(&weights, second, &text, 2 * MAX_CHUNKS, skip, [1, 80]) {
                sizes.push(run.tasks.len());
            }
        }
    }
    let max_rows = MAX_IMAGES * MAX_IMAGE_PIXELS / 64;
    let bufs = UnetBufs::carve(&mut carve, MAX_IMAGES, max_rows, MAX_CHUNKS * CHUNK_IDS);
    // The largest image of the spec is 1.5 MP: 1536 by 1024, or 192 by 128 latent rows.
    let size = UnetSize {
        images: MAX_IMAGES,
        h: 192,
        w: 128,
        ctx: MAX_CHUNKS * CHUNK_IDS,
    };
    let runs = unet(&weights, &bufs, &size);
    assert_eq!(runs.len(), 9);
    for (run, _) in &runs {
        sizes.push(run.tasks.len());
    }
    let vb = VaeBufs::carve(&mut carve, MAX_IMAGE_PIXELS);
    sizes.push(vae(&weights, &vb, 192, 128).tasks.len());
    for (i, n) in sizes.iter().enumerate() {
        assert!(*n > 0 && *n < MAX_TASKS_PER_CARD, "run {i} has {n} tasks");
    }
    eprintln!("tasks of each run: {sizes:?}");
}
