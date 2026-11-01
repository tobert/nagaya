//! The decode gate of Qwen-Image 2.1 (spec 8.3, with the bf16 baselines): the decoder on the oracle's final latent
//! against `rgba` of diffusers in f32. The gap is the mean of the absolute difference of each byte. The gate passes
//! when the engine's gap is at most 4 times the gap of the VAE in bf16 under torch (`rgba_bf16`), with a floor of
//! 0.25. Stop `nagaya.service` first (spec 3.1).
//!
//! The test reads the base folder from `QWENIMAGE21_MODEL_DIR`, and the oracle from `QWENIMAGE21_GOLDEN_DIR` or
//! `target/qwenimage21-golden`. Run:
//! `cargo test -p nagaya-mk-qwenimage21-gfx1151 --features hip --test hip_vae -- --nocapture`.
#![cfg(feature = "hip")]

mod common;

use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::PlanTable;
use nagaya_mk_qwenimage21_gfx1151::safetensors::{Dtype, Safetensors};
use nagaya_mk_qwenimage21_gfx1151::vae::{CHANNELS, SCALE, VaeDecoder, Z};

/// The floor of the bound, in byte steps.
const FLOOR: f64 = 0.25;

fn byte_gap(got: &[u8], want: &[u8]) -> (f64, u8) {
    assert_eq!(got.len(), want.len());
    let mut sum = 0u64;
    let mut worst = 0u8;
    for (g, w) in got.iter().zip(want) {
        let d = g.abs_diff(*w);
        sum += u64::from(d);
        worst = worst.max(d);
    }
    (sum as f64 / got.len() as f64, worst)
}

#[test]
fn the_decode_gate() {
    let Some(dir) = common::oracle() else { return };
    let Some(model) = common::model_dir() else { return };
    let steps = Safetensors::open(&dir.join("steps.safetensors")).unwrap();
    let shape = steps.info("final_latent").unwrap().shape.clone();
    let tokens = shape[1];
    let side = (tokens as f64).sqrt() as usize;
    assert_eq!(side * side, tokens);
    let latent: Vec<u16> = steps
        .read("final_latent", Dtype::F32, &shape)
        .unwrap()
        .chunks(4)
        .map(|b| bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    let size = side * SCALE;
    let want = steps.read("rgba", Dtype::U8, &[size, size, CHANNELS]).unwrap();
    let baseline = steps.read("rgba_bf16", Dtype::U8, &[size, size, CHANNELS]).unwrap();
    let started = std::time::Instant::now();
    let decoder = VaeDecoder::load(&model.join("vae")).unwrap();
    eprintln!("loaded the decoder in {:.1} s", started.elapsed().as_secs_f64());
    let stream = Stream::new().unwrap();
    let mut gemms = Gemms::new(PlanTable::in_memory(&lineage().unwrap()), 32 << 20).unwrap();
    let x = DeviceBuffer::from_slice(&latent).unwrap();
    assert_eq!(latent.len(), tokens * Z);
    let started = std::time::Instant::now();
    let got = decoder.decode(&mut gemms, &x, side, side, &stream).unwrap();
    eprintln!(
        "first decode of {size} by {size} in {:.0} ms, with the picks",
        started.elapsed().as_secs_f64() * 1e3
    );
    let started = std::time::Instant::now();
    let again = decoder.decode(&mut gemms, &x, side, side, &stream).unwrap();
    eprintln!("second decode in {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
    assert_eq!(got, again, "two decodes give the same bytes");
    let (gap, worst) = byte_gap(&got, &want);
    let (torch_gap, torch_worst) = byte_gap(&baseline, &want);
    let bound = (4.0 * torch_gap).max(FLOOR);
    eprintln!(
        "decode: mean byte gap {gap:.3} (worst {worst}), torch bf16 {torch_gap:.3} (worst {torch_worst}), bound {bound:.3}"
    );
    let alpha: Vec<u8> = got.iter().skip(3).step_by(4).copied().collect();
    eprintln!(
        "alpha: min {}, max {}",
        alpha.iter().min().unwrap(),
        alpha.iter().max().unwrap()
    );
    assert!(
        gap <= bound,
        "the decode gate: mean byte gap {gap} over the bound {bound}"
    );
}
