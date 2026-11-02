//! The Qwen-Image engine through the `Engine` trait (milestone 22 step 7): the conformance suite of spec 8.3, the whole
//! oracle through the image calls, and the cache gate of spec 8.3. Stop `nagaya.service` first (spec 3.1).
//!
//! The tests read the base folder from `QWENIMAGE21_MODEL_DIR`, FUSAL's folder from `QWENIMAGE21_FUSAL_DIR`, and the
//! oracle from `QWENIMAGE21_GOLDEN_DIR` or `target/qwenimage21-golden`. Run:
//! `cargo test -p nagaya-mk-qwenimage21-gfx1151 --features hip --test hip_engine -- --test-threads=1 --nocapture`.
#![cfg(feature = "hip")]

mod common;

use std::path::Path;
use std::sync::Arc;

use nagaya_abi::{Engine, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_conformance::run_all;
use nagaya_mk_hip::bf16;
use nagaya_mk_qwenimage21_gfx1151::engine::{Config, QwenImageEngine};
use nagaya_mk_qwenimage21_gfx1151::safetensors::{Dtype, Safetensors};

/// The cap of the image gate of spec 8.3: 2 on average in each byte. The oracle has no bf16 run of the whole job.
const IMAGE_MEAN_CAP: f64 = 2.0;

struct Inputs {
    config: Config,
    ids: Vec<u32>,
    cut: u32,
    start: Vec<u16>,
    rgba: Vec<u8>,
    size: u32,
}

fn inputs() -> Option<Inputs> {
    let dir = common::oracle()?;
    let model = common::model_dir()?;
    let fusal = common::fusal_dir()?;
    let encode = Safetensors::open(&dir.join("encode.safetensors")).unwrap();
    let steps = Safetensors::open(&dir.join("steps.safetensors")).unwrap();
    let shape = encode.info("ids").unwrap().shape.clone();
    let ids: Vec<u32> = encode
        .read("ids", Dtype::I64, &shape)
        .unwrap()
        .chunks(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as u32)
        .collect();
    let cut = i64::from_le_bytes(encode.read("drop", Dtype::I64, &[1]).unwrap()[..8].try_into().unwrap()) as u32;
    let request: serde_json::Value = serde_json::from_str(&encode.metadata()["request"]).unwrap();
    let packed_shape = steps.info("packed").unwrap().shape.clone();
    let start: Vec<u16> = steps
        .read("packed", Dtype::F32, &packed_shape)
        .unwrap()
        .chunks(4)
        .map(|b| bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    let size = request["width"].as_u64().unwrap() as u32;
    let rgba = steps
        .read("rgba", Dtype::U8, &[size as usize, size as usize, 4])
        .unwrap();
    Some(Inputs {
        config: Config {
            model: model.clone(),
            encoder: fusal.join(request["encoder"].as_str().unwrap()),
            lora: Some((fusal.join(request["lora"].as_str().unwrap()), 1.0)),
            plan_dir: None,
            flash: true,
        },
        ids,
        cut,
        start,
        rgba,
        size,
    })
}

fn job(inputs: &Inputs, count: u32, conditioning: Option<Arc<Vec<u8>>>) -> ImageJob {
    ImageJob {
        prompt: ImagePrompt::Tokens {
            ids: inputs.ids.clone(),
            cut: inputs.cut,
        },
        width: inputs.size,
        height: inputs.size,
        steps: 4,
        cfg: 1.0,
        sampler: ImageSampler::FlowEuler,
        spacing: ImageSpacing::Leading,
        count,
        seed: 1234,
        previews: false,
        loras: Vec::new(),
        conditioning,
        init: None,
    }
}

/// Run a job of the oracle's size to its images. The start latent of image 0 is the oracle's when `oracle_start`.
fn run(
    engine: &mut QwenImageEngine,
    inputs: &Inputs,
    job: &ImageJob,
    oracle_start: bool,
) -> (Option<Vec<u8>>, Vec<Vec<u8>>) {
    let made = engine.begin_any_size(job).unwrap();
    if oracle_start {
        engine.set_latent(0, &inputs.start);
    }
    let image = engine.image().unwrap();
    for k in 1..=4 {
        assert_eq!(image.image_step(0).unwrap().step, k);
    }
    let images = image.image_decode(0).unwrap();
    image.image_clear(0);
    (
        made,
        images
            .into_iter()
            .map(|i| {
                assert_eq!((i.width, i.height, i.channels), (inputs.size, inputs.size, 4));
                i.pixels
            })
            .collect(),
    )
}

#[test]
fn the_conformance_suite_passes() {
    let Some(inputs) = inputs() else { return };
    let suite = run_all(&mut || QwenImageEngine::open(&inputs.config).expect("open the engine"));
    println!("{suite}");
    assert!(suite.passed(), "{suite}");
}

#[test]
fn the_whole_oracle_and_the_cache_gate() {
    let Some(inputs) = inputs() else { return };
    let mut engine = QwenImageEngine::open(&inputs.config).unwrap();
    let conditioning = engine
        .image()
        .unwrap()
        .image_conditioning()
        .expect("the engine caches conditionings");
    assert_eq!((conditioning.row_bytes, conditioning.max_rows), (8192, 1010));
    let started = std::time::Instant::now();
    let (made, images) = run(&mut engine, &inputs, &job(&inputs, 1, None), true);
    eprintln!("the job of the oracle in {:.1} s", started.elapsed().as_secs_f64());
    let rows = made.expect("a miss gives the rows that the encoder made");
    assert_eq!(rows.len(), (inputs.ids.len() - inputs.cut as usize) * 8192);
    let image = &images[0];
    let gaps: Vec<u8> = image.iter().zip(&inputs.rgba).map(|(a, b)| a.abs_diff(*b)).collect();
    let mean = gaps.iter().map(|&g| f64::from(g)).sum::<f64>() / gaps.len() as f64;
    let worst = *gaps.iter().max().unwrap();
    eprintln!("image gate: mean byte gap {mean:.3}, worst {worst}; the cap is {IMAGE_MEAN_CAP} on average");
    assert!(mean <= IMAGE_MEAN_CAP, "the image gate: mean byte gap {mean}");
    // The cache gate: a job on a hit gives the bits of the job on a miss, and takes no rows back.
    let (again, hit) = run(
        &mut engine,
        &inputs,
        &job(&inputs, 1, Some(Arc::new(rows.clone()))),
        true,
    );
    assert!(again.is_none(), "a hit gives no rows");
    assert_eq!(hit, images, "a job on a hit has the bits of the job on a miss");
    // Two runs give the same bits, and image 0 of a job of 2 is the image of the job of 1.
    let (_, twice) = run(&mut engine, &inputs, &job(&inputs, 1, None), true);
    assert_eq!(twice, images, "two runs give the same bits");
    let (_, one) = run(&mut engine, &inputs, &job(&inputs, 1, None), false);
    let (_, two) = run(&mut engine, &inputs, &job(&inputs, 2, None), false);
    assert_eq!(two.len(), 2);
    assert_eq!(two[0], one[0], "image 0 does not depend on the other images of its job");
    assert_ne!(two[1], two[0], "image 1 has its own seed");
    // Rows of another count are refused.
    let short = Some(Arc::new(rows[..rows.len() - 8192].to_vec()));
    assert!(
        engine.begin_any_size(&job(&inputs, 1, short)).is_err(),
        "a conditioning of the wrong count"
    );
    if let Some(out) = std::env::var_os("QWENIMAGE21_IMAGE_OUT") {
        let ppm: Vec<u8> = image.chunks(4).flat_map(|p| p[..3].to_vec()).collect();
        let header = format!("P6 {} {} 255\n", inputs.size, inputs.size);
        std::fs::write(Path::new(&out), [header.into_bytes(), ppm].concat()).unwrap();
    }
}
