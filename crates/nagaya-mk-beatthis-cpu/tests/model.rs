//! The whole model on the CPU against the oracle: each mark of a chunk of final0 against the f64 rows, and the
//! logits and beat times of each case. The input is this family's own log-mel of the case's wave, so the test runs
//! the path of a request end to end.
//!
//! The gate of a row is 4 times torch's own f32 gap at that row (`parts` against `parts64`), with a floor of 1e-6 of
//! the row's peak. The interpreter sums in another order than torch, so its gap is near torch's, not equal.

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod common;

use std::time::Instant;

use nagaya_family_beatthis::plan::{self, Config, Plan};
use nagaya_family_beatthis::weights::{self, Tensors};
use nagaya_family_beatthis::{chunks, mel, peaks};
use nagaya_mk_beatthis_cpu as cpu;

struct Model {
    config: Config,
    weights: Vec<Vec<f32>>,
}

fn model(name: &str) -> Option<Model> {
    let bytes = common::weights(name)?;
    let config = Config::from_json(&Tensors::hyper_parameters(&bytes).unwrap()).unwrap();
    let t = Tensors::parse(&bytes).unwrap();
    let weights = weights::prepare(&plan::build(&config, chunks::CHUNK).weights, &t).unwrap();
    Some(Model { config, weights })
}

/// The rows of `[channels, freqs, frames]` (the oracle's layout) in the plan's `[frames, freqs, channels]`.
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

/// The logits of a whole piece: its chunks, each through the plan, joined keep_first.
fn piece(m: &Model, wave: &[f32], threads: usize) -> (Vec<f32>, Vec<f32>) {
    let spect = mel::log_mel(wave).unwrap();
    let frames = mel::frames(wave.len());
    let plan_of = chunks::plan(frames);
    let logits: Vec<(Vec<f32>, Vec<f32>)> = plan_of
        .iter()
        .map(|c| {
            let p: Plan = plan::build(&m.config, c.len());
            let out = cpu::run(&p, &m.weights, &chunks::input(&spect, frames, c), threads, |_, _| {});
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
    let mut worst = Vec::new();
    let start = Instant::now();
    cpu::run(&p, &m.weights, &chunks::input(&spect, frames, &c), 16, |mark, rows| {
        let (want64, shape) = g.f64(&format!("parts64.{}", mark.name));
        let (want32, _) = g.f32(&format!("parts.{}", mark.name));
        let want32: Vec<f64> = want32.into_iter().map(f64::from).collect();
        let (w64, w32) = if shape.len() == 3 {
            let (cc, ff, tt) = (shape[0], shape[1], shape[2]);
            assert_eq!((cc, ff, tt), (mark.channels, mark.freqs, p.frames), "{}", mark.name);
            (channels_last(&want64, cc, ff, tt), channels_last(&want32, cc, ff, tt))
        } else {
            assert_eq!(shape, [p.frames, mark.channels], "{}", mark.name);
            (want64, want32)
        };
        let got: Vec<f64> = rows.iter().map(|&v| f64::from(v)).collect();
        let torch = rel(&w32, &w64);
        let ours = rel(&got, &w64);
        let gate = (4.0 * torch).max(1e-6);
        worst.push(format!(
            "{:13} ours {ours:.2e} torch {torch:.2e} gate {gate:.2e}",
            mark.name
        ));
        assert!(
            ours <= gate,
            "{}: gap {ours:e} over the gate {gate:e} (torch {torch:e})",
            mark.name
        );
    });
    eprintln!(
        "final0, one chunk of {} frames, 16 threads: {:?}",
        p.frames,
        start.elapsed()
    );
    for w in &worst {
        eprintln!("{w}");
    }
    assert_eq!(worst.len(), p.marks.len(), "every mark was checked");
}

/// The logits of each case and model against the f64 run, and the beat times against upstream's.
#[test]
fn the_logits_and_beats_of_each_case_match_the_oracle() {
    let Some(g) = common::golden() else { return };
    for name in ["small0", "final0"] {
        let Some(m) = model(name) else { return };
        for case in g.case_names() {
            let (wave, _) = g.f32(&format!("{case}.wave"));
            let start = Instant::now();
            let (beat, down) = piece(&m, &wave, 16);
            let ms = start.elapsed().as_millis();
            let (b64, _) = g.f64(&format!("{name}.{case}.beat64"));
            let (d64, _) = g.f64(&format!("{name}.{case}.downbeat64"));
            let (b32, _) = g.f32(&format!("{name}.{case}.beat"));
            let (d32, _) = g.f32(&format!("{name}.{case}.downbeat"));
            let to64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
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

#[test]
fn the_bits_of_a_piece_do_not_change_with_the_thread_count() {
    let Some(g) = common::golden() else { return };
    let Some(m) = model("small0") else { return };
    let (wave, _) = g.f32("click97_34.wave");
    let one = piece(&m, &wave, 1);
    for threads in [3, 16] {
        let got = piece(&m, &wave, threads);
        let same = one
            .0
            .iter()
            .chain(&one.1)
            .zip(got.0.iter().chain(&got.1))
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "{threads} threads moved bits");
    }
}

/// The bits of the engine are its own (spec 8.5): a change that moves them must change the engine identity. This pin
/// holds the logits of small0 on the 10 s case. The kernels use no libm of the machine and no fused multiply-add, so
/// the pin holds on each machine for one build.
#[test]
fn the_bits_of_small0_are_pinned() {
    let Some(g) = common::golden() else { return };
    let Some(m) = model("small0") else { return };
    let (wave, _) = g.f32("click97_34.wave");
    let (beat, down) = piece(&m, &wave, 4);
    let mut h = blake3::Hasher::new();
    for v in beat.iter().chain(&down) {
        h.update(&v.to_le_bytes());
    }
    assert_eq!(
        h.finalize().to_hex().to_string(),
        PIN_SMALL0,
        "the logits of small0 moved"
    );
}

const PIN_SMALL0: &str = "258ef4d80827179d3ec83cec4dc4626af06347721ca7a537da618dd7daac7a57";
