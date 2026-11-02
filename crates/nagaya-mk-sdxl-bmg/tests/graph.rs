//! The graphs against the tensor headers of the checkpoint, with no card: each tensor that a graph names exists with
//! its shape, and each run fits in a ring. The test reads `SDXL_MODEL_DIR`, and skips when it is absent.

use std::collections::HashMap;
use std::path::PathBuf;

use nagaya_mk_bmg::protocol::{FLAG_PAD_END, MAX_TASKS_PER_CARD, OP_GEMM};
use nagaya_mk_sdxl_bmg::model::{Carve, TextBufs, UnetBufs, UnetSize, VaeBufs, text_encoder, unet, vae, vae_encoder};
use nagaya_mk_sdxl_bmg::spec::*;
use nagaya_mk_sdxl_bmg::weights::{Adapter, Count, Weights};

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
    let bufs = UnetBufs::carve(&mut carve, MAX_IMAGES, max_rows, MAX_CHUNKS * CHUNK_IDS, LATENT);
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
    sizes.push(vae_encoder(&weights, &vb, 1536, 1024).tasks.len());
    for (i, n) in sizes.iter().enumerate() {
        assert!(*n > 0 && *n < MAX_TASKS_PER_CARD, "run {i} has {n} tasks");
    }
    eprintln!("tasks of each run: {sizes:?}");
}

/// The VAE encoder reads the image and writes the 8 moments of each latent pixel. Its 3 down samplers pad the bottom
/// and the right only, with stride 2 (diffusers' `downsampling.py:139-145`), and no other task pads that way.
#[test]
fn the_vae_encoder_reads_the_image_writes_the_moments_and_pads_its_down_samplers_at_the_end() {
    let Some(dir) = model_dir() else { return };
    let mut count = Count::default();
    let weights = Weights::load(&nagaya_mk_sdxl_bmg::Config::folder(dir), &mut count).unwrap();
    let mut carve = Carve::new(0, u64::MAX / 2);
    let vb = VaeBufs::carve(&mut carve, MAX_IMAGE_PIXELS);
    let run = vae_encoder(&weights, &vb, 192, 128);
    assert_eq!(run.tasks.first().unwrap().a, vb.image);
    assert_eq!(run.tasks.last().unwrap().c, vb.moments);
    assert_eq!(run.tasks.last().unwrap().m, 8, "the last task gives 8 channels");
    let ends: Vec<_> = run.tasks.iter().filter(|t| t.flags & FLAG_PAD_END != 0).collect();
    assert_eq!(ends.len(), 3);
    for t in ends {
        assert_eq!(t.opcode, OP_GEMM);
        assert_eq!(
            (t.layer & 0xFF, (t.layer >> 8) & 0xFF, (t.layer >> 16) & 0xFF),
            (3, 2, 1)
        );
    }
    // The pixels of each down sampler halve: 192 by 128, then 96 by 64, then 48 by 32.
    let inputs: Vec<u32> = run
        .tasks
        .iter()
        .filter(|t| t.flags & FLAG_PAD_END != 0)
        .map(|t| t.n)
        .collect();
    assert_eq!(inputs, [192 | 128 << 16, 96 | 64 << 16, 48 | 32 << 16]);
}

/// The plan counts the weights of the VAE encoder: the open uploads each tensor of `vae/`, and the count of the plan is
/// the count of the upload (`weights.rs`). The encoder's F32 tensors hold 136,654,656 bytes in the file. On the card
/// its convolutions take 3 BF16 planes each, so it holds 204,845,312 bytes (read from the header, 2026-11-02).
#[test]
fn the_plan_counts_the_weights_of_the_vae_encoder() {
    use nagaya_mk_bmg::safetensors::MappedSafeTensors;
    use nagaya_mk_sdxl_bmg::weights::{items, sources};
    let Some(dir) = model_dir() else { return };
    let config = nagaya_mk_sdxl_bmg::Config::folder(dir);
    let maps: Vec<_> = sources(&config)
        .unwrap()
        .into_iter()
        .map(|(prefix, path)| (prefix, MappedSafeTensors::open(&path).unwrap()))
        .collect();
    let all = items(&maps, false).unwrap();
    let encoder = |name: &str| name.starts_with("vae.encoder.") || name.starts_with("vae.quant_conv.");
    let (mut with, mut without) = (Count::default(), Count::default());
    Weights::from_items(&all, &mut with).unwrap();
    let rest: Vec<_> = all
        .iter()
        .filter(|item| !encoder(&item.name))
        .map(|item| nagaya_mk_sdxl_bmg::weights::Item {
            name: item.name.clone(),
            dtype: item.dtype,
            shape: item.shape.clone(),
            bytes: item.bytes.clone(),
        })
        .collect();
    assert!(rest.len() < all.len(), "the checkpoint holds the encoder");
    Weights::from_items(&rest, &mut without).unwrap();
    assert_eq!(with.bytes - without.bytes, 204_845_312);
    assert_eq!(with.bytes, nagaya_mk_sdxl_bmg::plan::weights_bytes(&config).unwrap());
}

/// A LoRA on every module of the checkpoint adds its delta to each graph, and each run still fits in its ring (spec
/// 11.9a). The addresses are fake: the graphs are built here, not run. Each matrix of each module has its own address,
/// and each address must be the weight of some task.
#[test]
fn a_lora_on_every_module_reaches_its_graph_and_fits_the_rings() {
    let Some(dir) = model_dir() else { return };
    let mut count = Count::default();
    let weights = Weights::load(&nagaya_mk_sdxl_bmg::Config::folder(dir), &mut count).unwrap();
    let mut carve = Carve::new(0, u64::MAX / 2);
    let text = TextBufs::carve(&mut carve, 2 * MAX_CHUNKS);
    let max_rows = MAX_IMAGES * MAX_IMAGE_PIXELS / 64;
    let bufs = UnetBufs::carve(&mut carve, MAX_IMAGES, max_rows, MAX_CHUNKS * CHUNK_IDS, LATENT);
    let size = UnetSize {
        images: MAX_IMAGES,
        h: 192,
        w: 128,
        ctx: MAX_CHUNKS * CHUNK_IDS,
    };
    // The sizes of the runs, and the weight address of each task.
    let runs_of = |weights: &Weights| {
        let mut sizes = Vec::new();
        let mut weight_at = std::collections::HashSet::new();
        let mut note = |run: &nagaya_mk_sdxl_bmg::model::Run| {
            sizes.push(run.tasks.len());
            weight_at.extend(run.tasks.iter().map(|t| t.b));
        };
        for second in [false, true] {
            for run in text_encoder(weights, second, &text, 2 * MAX_CHUNKS, 2, [1, 80]) {
                note(&run);
            }
        }
        for (run, _) in unet(weights, &bufs, &size) {
            note(&run);
        }
        (sizes, weight_at)
    };
    let (plain, _) = runs_of(&weights);
    let modules = weights.modules();
    let mut adapters = HashMap::new();
    let mut matrices = Vec::new();
    for (i, (name, shape)) in modules.iter().enumerate() {
        let kernel = if shape.len() == 4 { shape[2] } else { 1 };
        let (down, up) = ((1u64 << 32) + 2 * i as u64, (1u64 << 32) + 2 * i as u64 + 1);
        matrices.extend([down, up]);
        let adapter = Adapter {
            down,
            up,
            rank: 16,
            kernel,
            scratch: 4 << 20,
        };
        adapters.insert(name.clone(), vec![adapter]);
    }
    weights.set_adapters(adapters);
    let (with, weight_at) = runs_of(&weights);
    assert_eq!(with.len(), plain.len());
    assert!(
        with.iter().zip(&plain).all(|(a, b)| a > b),
        "each run gets its LoRA tasks: {plain:?} and {with:?}"
    );
    assert!(with.iter().all(|n| *n < MAX_TASKS_PER_CARD), "a run has {with:?} tasks");
    let missing: Vec<&String> = modules
        .iter()
        .enumerate()
        .filter(|(i, _)| !weight_at.contains(&((1u64 << 32) + 2 * *i as u64)))
        .map(|(_, (name, _))| name)
        .collect();
    assert!(
        missing.is_empty(),
        "these modules have no down matrix in any task: {missing:?}"
    );
    assert!(
        matrices.iter().all(|m| weight_at.contains(m)),
        "each up matrix is a weight of a task"
    );
    eprintln!("tasks of each run with a LoRA on every module: {with:?}");
}

/// The inpainting checkpoint of spec 11.9b, `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, from
/// `SDXL_INPAINT_MODEL_DIR`. Its UNet has 9 input channels, each graph names tensors of the checkpoint, and the plan
/// counts the bytes that the open uploads. The test prints the plan for the profile.
#[test]
fn the_inpainting_checkpoint_has_9_input_channels_and_plans_its_bytes() {
    use nagaya_mk_sdxl_bmg::model::{check, unet_in_channels};
    use nagaya_mk_sdxl_bmg::plan::{arena_bytes, plan};
    let dir = std::env::var_os("SDXL_INPAINT_MODEL_DIR").map(PathBuf::from);
    let Some(dir) = dir.filter(|d| d.join("unet").is_dir()) else {
        let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
        assert!(
            !require,
            "NAGAYA_REQUIRE_MODELS is set: set SDXL_INPAINT_MODEL_DIR to the SDXL inpainting 0.1 checkpoint"
        );
        eprintln!("skipped: SDXL_INPAINT_MODEL_DIR does not name the SDXL inpainting checkpoint");
        return;
    };
    let config = nagaya_mk_sdxl_bmg::Config::folder(dir);
    let mut count = Count::default();
    let weights = Weights::load(&config, &mut count).unwrap();
    assert_eq!(unet_in_channels(&weights), Ok(INPAINT_CHANNELS));
    check(&weights).unwrap();
    let limits = plan(&config).unwrap();
    let term = |name: &str| limits.memory.mandatory.iter().find(|t| t.name == name).unwrap().bytes;
    assert_eq!(term("weights"), count.bytes);
    assert!(term("activations") > arena_bytes(INPAINT_CHANNELS));
    eprintln!(
        "the plan of the inpainting checkpoint: weights {} bytes, activations {} bytes (arena {}), pinned {} bytes",
        term("weights"),
        term("activations"),
        arena_bytes(INPAINT_CHANNELS),
        term("pinned")
    );
}
