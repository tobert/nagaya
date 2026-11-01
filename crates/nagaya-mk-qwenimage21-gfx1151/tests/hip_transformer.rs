//! The step, run and LoRA gates of Qwen-Image 2.1 (spec 8.3, "The Qwen-Image oracle", with the bf16 baselines). The
//! gap of a tensor is 1 minus its cosine with the f32 reference. A gate passes when the engine's gap is at most 4
//! times the gap of diffusers in bf16, with a floor of 1e-5. Stop `nagaya.service` first.
//!
//! The tests read the base folder from `QWENIMAGE21_MODEL_DIR`, FUSAL's folder from `QWENIMAGE21_FUSAL_DIR`, and the
//! oracle from `QWENIMAGE21_GOLDEN_DIR` or `target/qwenimage21-golden`. Run:
//! `cargo test -p nagaya-mk-qwenimage21-gfx1151 --features hip --test hip_transformer -- --test-threads=1 --nocapture`.
#![cfg(feature = "hip")]

mod common;

use std::path::Path;

use nagaya_mk_hip::attention::Attention;
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::PlanTable;
use nagaya_mk_qwenimage21_gfx1151::names::HIDDEN;
use nagaya_mk_qwenimage21_gfx1151::safetensors::{Dtype, Safetensors};
use nagaya_mk_qwenimage21_gfx1151::sampler;
use nagaya_mk_qwenimage21_gfx1151::transformer::{LATENT, Transformer};

/// The floor of a bound (spec 8.3, "The bf16 baseline of each gate").
const FLOOR: f64 = 1e-5;

struct Oracle {
    encode: Safetensors,
    steps: Safetensors,
    parts: Safetensors,
    rows: usize,
    tokens: usize,
    side: usize,
}

impl Oracle {
    fn open(dir: &Path) -> Oracle {
        let encode = Safetensors::open(&dir.join("encode.safetensors")).unwrap();
        let steps = Safetensors::open(&dir.join("steps.safetensors")).unwrap();
        let parts = Safetensors::open(&dir.join("parts.safetensors")).unwrap();
        let rows = encode.info("prompt_embeds").unwrap().shape[1];
        let tokens = steps.info("packed").unwrap().shape[1];
        let side = (tokens as f64).sqrt() as usize;
        assert_eq!(side * side, tokens, "the oracle image is square");
        Oracle {
            encode,
            steps,
            parts,
            rows,
            tokens,
            side,
        }
    }

    fn floats(st: &Safetensors, name: &str) -> Vec<f32> {
        let shape = st.info(name).unwrap().shape.clone();
        st.read(name, Dtype::F32, &shape)
            .unwrap()
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn steps(&self, name: &str) -> Vec<f32> {
        Oracle::floats(&self.steps, name)
    }
}

fn to_bf16(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn gap(got: &[u16], want: &[f32]) -> f64 {
    let g: Vec<f64> = got.iter().map(|&h| f64::from(bf16::to_f32(h))).collect();
    let w: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
    gap_f64(&g, &w)
}

fn gap_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    1.0 - dot / (a.iter().map(|x| x * x).sum::<f64>().sqrt() * b.iter().map(|x| x * x).sum::<f64>().sqrt())
}

fn baseline_gap(base: &[f32], want: &[f32]) -> f64 {
    let a: Vec<f64> = base.iter().map(|&x| f64::from(x)).collect();
    let b: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
    gap_f64(&a, &b)
}

struct Rig {
    gemms: Gemms,
    stream: Stream,
    transformer: Transformer,
}

fn rig(model: &Path, fusal: Option<&Path>, oracle: &Oracle) -> Rig {
    let stream = Stream::new().unwrap();
    let mut gemms = Gemms::new(PlanTable::in_memory(&lineage().unwrap()), 32 << 20).unwrap();
    let lora_file = fusal.map(|dir| {
        let request: serde_json::Value = serde_json::from_str(&oracle.encode.metadata()["request"]).unwrap();
        Safetensors::open(&dir.join(request["lora"].as_str().unwrap())).unwrap()
    });
    let started = std::time::Instant::now();
    let transformer = Transformer::load(
        &model.join("transformer"),
        lora_file.as_ref().map(|f| (f, 1.0)),
        &mut gemms,
        &stream,
    )
    .unwrap();
    eprintln!(
        "loaded {:.2} GiB in {:.1} s, LoRA {}",
        transformer.bytes() as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
        lora_file.is_some()
    );
    Rig {
        gemms,
        stream,
        transformer,
    }
}

#[test]
fn the_step_and_run_gates() {
    let Some(dir) = common::oracle() else { return };
    let Some(model) = common::model_dir() else { return };
    let Some(fusal) = common::fusal_dir() else { return };
    let oracle = Oracle::open(&dir);
    let mut r = rig(&model, Some(&fusal), &oracle);
    let (rows, n, side) = (oracle.rows, oracle.tokens, oracle.side);
    let attention = Attention::new(Transformer::attention_capacity(rows, n) * 8).unwrap();
    let embeds = DeviceBuffer::from_slice(&to_bf16(&Oracle::floats(&oracle.encode, "prompt_embeds"))).unwrap();
    let started = std::time::Instant::now();
    let conditioned = r
        .transformer
        .begin(&mut r.gemms, &attention, &embeds, 0, rows, &r.stream)
        .unwrap();
    eprintln!(
        "begin of {rows} rows in {:.0} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    let sigmas = sampler::sigmas(4, n);
    assert_eq!(sigmas, oracle.steps("sigmas"), "the schedule of the oracle");
    let latents = oracle.steps("latents");
    let outs = oracle.steps("transformer_out");
    let outs16 = oracle.steps("transformer_out_bf16");
    let one = n * LATENT;
    let mut failed = Vec::new();
    for k in 0..4 {
        let x = DeviceBuffer::from_slice(&to_bf16(&latents[k * one..(k + 1) * one])).unwrap();
        let started = std::time::Instant::now();
        let v = r
            .transformer
            .step(
                &mut r.gemms,
                &attention,
                &conditioned,
                &x,
                side,
                side,
                sampler::timestep(sigmas[k]),
                &r.stream,
            )
            .unwrap();
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let got = v.to_vec::<u16>().unwrap();
        let want = &outs[k * one..(k + 1) * one];
        let (g, b) = (gap(&got, want), baseline_gap(&outs16[k * one..(k + 1) * one], want));
        let bound = (4.0 * b).max(FLOOR);
        eprintln!("step {k}: gap {g:.2e}, torch bf16 gap {b:.2e}, bound {bound:.2e}, {ms:.0} ms");
        if g > bound {
            failed.push(format!("step {k}"));
        }
        if k == 0 {
            let again = r
                .transformer
                .step(
                    &mut r.gemms,
                    &attention,
                    &conditioned,
                    &x,
                    side,
                    side,
                    sampler::timestep(sigmas[0]),
                    &r.stream,
                )
                .unwrap();
            assert_eq!(got, again.to_vec::<u16>().unwrap(), "two steps give the same bits");
        }
    }
    // The run: 4 steps from the start latent, with the Euler steps on the host.
    let mut x = to_bf16(&oracle.steps("packed"));
    for k in 0..4 {
        let xd = DeviceBuffer::from_slice(&x).unwrap();
        let v = r
            .transformer
            .step(
                &mut r.gemms,
                &attention,
                &conditioned,
                &xd,
                side,
                side,
                sampler::timestep(sigmas[k]),
                &r.stream,
            )
            .unwrap();
        sampler::euler(&mut x, &v.to_vec::<u16>().unwrap(), sigmas[k], sigmas[k + 1]);
    }
    let want = oracle.steps("final_latent");
    let base = &oracle.steps("latents_bf16")[4 * one..5 * one];
    let (g, b) = (gap(&x, &want), baseline_gap(base, &want));
    let bound = (4.0 * b).max(FLOOR);
    eprintln!("run: gap {g:.2e}, torch bf16 gap {b:.2e}, bound {bound:.2e}");
    if g > bound {
        failed.push("run".to_owned());
    }
    eprintln!(
        "conditioned keys and values: {:.1} MiB",
        conditioned.bytes() as f64 / (1 << 20) as f64
    );
    assert!(failed.is_empty(), "the gates failed: {failed:?}");
}

#[test]
fn the_lora_gate() {
    let Some(dir) = common::oracle() else { return };
    let Some(model) = common::model_dir() else { return };
    let oracle = Oracle::open(&dir);
    let mut r = rig(&model, None, &oracle);
    let (rows, n, side) = (oracle.rows, oracle.tokens, oracle.side);
    let attention = Attention::new(Transformer::attention_capacity(rows, n) * 8).unwrap();
    let embeds = DeviceBuffer::from_slice(&to_bf16(&Oracle::floats(&oracle.encode, "prompt_embeds"))).unwrap();
    let conditioned = r
        .transformer
        .begin(&mut r.gemms, &attention, &embeds, 0, rows, &r.stream)
        .unwrap();
    let one = n * LATENT;
    let latents = oracle.steps("latents");
    let x = DeviceBuffer::from_slice(&to_bf16(&latents[..one])).unwrap();
    let sigmas = sampler::sigmas(4, n);
    let v = r
        .transformer
        .step(
            &mut r.gemms,
            &attention,
            &conditioned,
            &x,
            side,
            side,
            sampler::timestep(sigmas[0]),
            &r.stream,
        )
        .unwrap()
        .to_vec::<u16>()
        .unwrap();
    let no_lora = Oracle::floats(&oracle.parts, "tr.step0_no_lora");
    let with_lora = &oracle.steps("transformer_out")[..one];
    let base = baseline_gap(&oracle.steps("transformer_out_bf16")[..one], with_lora);
    let bound = (4.0 * base).max(FLOOR);
    let (own, other) = (gap(&v, &no_lora), gap(&v, with_lora));
    eprintln!("no LoRA: gap {own:.2e} to step 0 with no LoRA, {other:.2e} to step 0 with the LoRA; bound {bound:.2e}");
    assert!(
        own <= bound,
        "the engine with no LoRA matches the oracle's step 0 with no LoRA"
    );
    assert!(other > bound, "the step gate fails when the engine drops the LoRA");
    let _ = HIDDEN;
}
