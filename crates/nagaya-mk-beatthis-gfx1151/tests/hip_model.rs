//! Beat This! on a gfx1151 card against the oracle, with the gates of the CPU engine: each mark within 4 times torch's
//! own f32 gap to the f64 rows, the logits of each case within 4 times torch's gap, and upstream's beat times.
//!
//! A card test. Stop `nagaya.service` first (spec 3.1), and run with one thread:
//! `cargo test -p nagaya-mk-beatthis-gfx1151 --features hip --test hip_model -- --test-threads=1`.
#![cfg(feature = "hip")]

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod common;

use std::time::Instant;

use nagaya_family_beatthis::plan::{self, Config, Plan};
use nagaya_family_beatthis::weights::{self, Tensors};
use nagaya_family_beatthis::{chunks, mel, peaks};
use nagaya_mk_beatthis_gfx1151::gpu::Gpu;

struct Model {
    config: Config,
    gpu: Gpu,
}

fn model(name: &str) -> Option<Model> {
    let bytes = common::weights(name)?;
    let config = Config::from_json(&Tensors::hyper_parameters(&bytes).unwrap()).unwrap();
    let t = Tensors::parse(&bytes).unwrap();
    let prepared = weights::prepare(&plan::build(&config, chunks::CHUNK).weights, &t).unwrap();
    Some(Model {
        config,
        gpu: Gpu::new(&prepared).unwrap(),
    })
}

fn channels_last(v: &[f64], c: usize, f: usize, t: usize) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for ci in 0..c {
        for fi in 0..f {
            for ti in 0..t {
                out[(ti * f + fi) * c + ci] = v[(ci * f + fi) * t + ti];
            }
        }
    }
    out
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let peak = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    common::max_abs(a, b) / peak
}

fn piece(m: &Model, wave: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let spect = mel::log_mel(wave).unwrap();
    let frames = mel::frames(wave.len());
    let plan_of = chunks::plan(frames);
    let logits: Vec<(Vec<f32>, Vec<f32>)> = plan_of
        .iter()
        .map(|c| {
            let p: Plan = plan::build(&m.config, c.len());
            let out = m
                .gpu
                .run(&p, &chunks::input(&spect, frames, c), false, |_, _| {})
                .unwrap();
            plan::logits(&out)
        })
        .collect();
    chunks::join(frames, &plan_of, &logits)
}

#[test]
fn each_mark_of_final0_is_within_four_times_the_gap_of_torch() {
    let Some(g) = common::golden() else { return };
    let Some(m) = model("final0") else { return };
    let (wave, _) = g.f32("click120_44.wave");
    let spect = mel::log_mel(&wave).unwrap();
    let frames = mel::frames(wave.len());
    let c = chunks::plan(frames)[0];
    let p = plan::build(&m.config, c.len());
    let mut lines = Vec::new();
    let mut failed = Vec::new();
    m.gpu
        .run(&p, &chunks::input(&spect, frames, &c), true, |mark, rows| {
            let (want64, shape) = g.f64(&format!("parts64.{}", mark.name));
            let (want32, _) = g.f32(&format!("parts.{}", mark.name));
            let want32: Vec<f64> = want32.into_iter().map(f64::from).collect();
            let (w64, w32) = if shape.len() == 3 {
                (
                    channels_last(&want64, shape[0], shape[1], shape[2]),
                    channels_last(&want32, shape[0], shape[1], shape[2]),
                )
            } else {
                (want64, want32)
            };
            let got: Vec<f64> = rows.iter().map(|&v| f64::from(v)).collect();
            let (torch, ours) = (rel(&w32, &w64), rel(&got, &w64));
            let gate = (4.0 * torch).max(1e-6);
            lines.push(format!(
                "{:13} gpu {ours:.2e} torch {torch:.2e} gate {gate:.2e}",
                mark.name
            ));
            if ours > gate {
                failed.push(mark.name.clone());
            }
        })
        .unwrap();
    for l in &lines {
        eprintln!("{l}");
    }
    assert_eq!(lines.len(), p.marks.len(), "every mark was checked");
    assert!(failed.is_empty(), "marks over the gate: {failed:?}");
}

#[test]
fn the_logits_and_beats_of_each_case_match_the_oracle() {
    let Some(g) = common::golden() else { return };
    for name in ["small0", "final0"] {
        let Some(m) = model(name) else { return };
        for case in g.case_names() {
            let (wave, _) = g.f32(&format!("{case}.wave"));
            let start = Instant::now();
            let (beat, down) = piece(&m, &wave);
            let ms = start.elapsed().as_millis();
            let to64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
            let (b64, _) = g.f64(&format!("{name}.{case}.beat64"));
            let (d64, _) = g.f64(&format!("{name}.{case}.downbeat64"));
            let (b32, _) = g.f32(&format!("{name}.{case}.beat"));
            let (d32, _) = g.f32(&format!("{name}.{case}.downbeat"));
            let torch = common::max_abs(&to64(&b32), &b64).max(common::max_abs(&to64(&d32), &d64));
            let ours = common::max_abs(&to64(&beat), &b64).max(common::max_abs(&to64(&down), &d64));
            let gate = (4.0 * torch).max(1e-4);
            eprintln!("{name} {case}: {ms} ms, logit gap {ours:.2e}, torch {torch:.2e}, gate {gate:.2e}");
            assert!(ours <= gate, "{name} {case}: logit gap {ours:e} over the gate {gate:e}");
            let got = peaks::beats(&beat, &down);
            let (want_b, _) = g.f64(&format!("{name}.{case}.beats_s"));
            let (want_d, _) = g.f64(&format!("{name}.{case}.downbeats_s"));
            assert_eq!(got.beats, want_b, "{name} {case}: beats");
            assert_eq!(got.downbeats, want_d, "{name} {case}: downbeats");
        }
    }
}

/// The same input gives the same bits on each run, and the bits of gfx1151 are pinned (spec 8.5): a change that moves
/// them must change the engine identity. The pin holds for one ROCm build of the kernels.
#[test]
fn two_runs_give_the_same_bits() {
    let Some(g) = common::golden() else { return };
    let Some(m) = model("small0") else { return };
    let (wave, _) = g.f32("click97_34.wave");
    let one = piece(&m, &wave);
    let two = piece(&m, &wave);
    let same = one
        .0
        .iter()
        .chain(&one.1)
        .zip(two.0.iter().chain(&two.1))
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "two runs gave other bits");
    let mut h = blake3::Hasher::new();
    for v in one.0.iter().chain(&one.1) {
        h.update(&v.to_le_bytes());
    }
    assert_eq!(
        h.finalize().to_hex().to_string(),
        PIN_GFX1151_SMALL0,
        "the logits of small0 on gfx1151 moved"
    );
}

const PIN_GFX1151_SMALL0: &str = "7b1ce38828152a73534aff17c364896c1cec7cd4410513d3f02ada2eab4c2b93";
