//! The log-mel spectrum against the reference files, and the pins of its own bits.

mod common;

use nagaya_family_beatthis::Error;
use nagaya_family_beatthis::mel::{self, BINS, MELS, N_FFT};

/// A test input with no libm in it: a sum of two square waves and the noise of a linear congruential generator.
fn synthetic(samples: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..samples)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let a = if (i / 25) % 2 == 0 { 0.3 } else { -0.3 };
            let b = if (i / 11_025) % 2 == 0 { 0.1 } else { -0.1 };
            a + b + 0.05 * noise
        })
        .collect()
}

fn hash_f64(v: &[f64]) -> String {
    let mut h = blake3::Hasher::new();
    for x in v {
        h.update(&x.to_le_bytes());
    }
    h.finalize().to_hex().to_string()
}

#[test]
fn frames_follow_the_hop() {
    assert_eq!(mel::frames(30 * 22_050), 1501);
    assert_eq!(mel::frames(11_025), 26);
    assert_eq!(mel::frames(513), 2);
    assert_eq!(mel::frames(882), 3);
}

#[test]
fn too_short_input_is_refused() {
    assert_eq!(
        mel::log_mel(&[0.0; 512]).unwrap_err(),
        Error::TooShort {
            samples: 512,
            minimum: 513
        }
    );
    assert!(mel::log_mel(&[0.0; 513]).is_ok());
}

#[test]
fn silence_gives_zeros() {
    let rows = mel::log_mel(&[0.0; 5000]).unwrap();
    assert_eq!(rows.len(), mel::frames(5000) * MELS);
    assert!(rows.iter().all(|v| v.to_bits() == 0), "log1p(0) must be +0");
}

#[test]
fn the_rows_have_the_shape_of_the_frames() {
    let x = synthetic(22_050);
    assert_eq!(mel::magnitudes(&x).unwrap().len(), mel::frames(x.len()) * BINS);
    assert_eq!(mel::log_mel(&x).unwrap().len(), mel::frames(x.len()) * MELS);
}

/// The tables come from the libm crate's cos, log and exp. A change of that crate that moves them fails here.
#[test]
fn the_tables_are_pinned() {
    assert_eq!(hash_f64(mel::window()), PIN_WINDOW, "the window moved");
    assert_eq!(hash_f64(mel::filterbank()), PIN_FILTERS, "the mel filters moved");
}

/// The f64 magnitudes hold the bits of the FFT and its twiddles, which the rounding to f32 can hide.
#[test]
fn the_magnitude_bits_are_pinned() {
    let mags = mel::magnitudes(&synthetic(22_050)).unwrap();
    assert_eq!(hash_f64(&mags), PIN_MAGNITUDES, "the f64 magnitudes moved");
}

/// The bits of the spectrum are this library's own. A change that moves them must update the engine identity.
#[test]
fn the_log_mel_bits_are_pinned() {
    let rows = mel::log_mel(&synthetic(3 * 22_050)).unwrap();
    let mut h = blake3::Hasher::new();
    for v in &rows {
        h.update(&v.to_le_bytes());
    }
    assert_eq!(h.finalize().to_hex().to_string(), PIN_LOG_MEL, "the log-mel bits moved");
}

/// Near silence, `1000 * mel` is below 1e-8. There `ln(1 + x)` loses more digits than f32 keeps, and `log1p` does not.
#[test]
fn the_log_mel_bits_of_quiet_audio_are_pinned() {
    let quiet: Vec<f32> = synthetic(22_050).into_iter().map(|v| v * 1e-12).collect();
    let rows = mel::log_mel(&quiet).unwrap();
    assert!(
        rows.iter().all(|&v| v > 0.0 && v < 1e-8),
        "the input must be quiet enough to test log1p"
    );
    let mut h = blake3::Hasher::new();
    for v in &rows {
        h.update(&v.to_le_bytes());
    }
    assert_eq!(
        h.finalize().to_hex().to_string(),
        PIN_LOG_MEL_QUIET,
        "the log-mel bits of quiet audio moved"
    );
}

const PIN_WINDOW: &str = "c963bdf4dbde52f25eba776f0a2e95d74ad299736650104f527da19d7c83b207";
const PIN_LOG_MEL_QUIET: &str = "1c83496f4681b8fc6d4001cde102dff39c3270391252a4a00fcb03f21affc0c6";
const PIN_MAGNITUDES: &str = "5a1c2aa63b7a5643c9b6f8891681db9fd9ebf473bbeadec667261c3052f9ddda";
const PIN_FILTERS: &str = "f5f4bcf02295d4fb73966e7340d28d3d8bea2368afb848d8ded10b6e71bf583f";
const PIN_LOG_MEL: &str = "8f0a131f5054cf9abe2659266c9da8d221dbb41e185bcd966ab1e2f13aaf07e5";

// The reference of these gates is torchaudio with f64 as the default dtype (`tools/beatthis_golden.sh`). This module
// computes in f64 too, so the gaps are a few steps of f64, and a wrong formula shows as a gap of 1e-6 or more.

#[test]
fn the_window_matches_the_reference() {
    let Some(g) = common::golden() else { return };
    let (want, shape) = g.f64("mel.window64");
    assert_eq!(shape, [N_FFT]);
    let gap = common::max_abs(mel::window(), &want);
    assert!(gap <= 1e-15, "window gap {gap:e}");
}

#[test]
fn the_filters_match_the_reference() {
    let Some(g) = common::golden() else { return };
    let (want, shape) = g.f64("mel.filterbank64");
    assert_eq!(shape, [BINS, MELS]);
    let gap = common::max_abs(mel::filterbank(), &want);
    assert!(gap <= 1e-12, "filter gap {gap:e}");
}

#[test]
fn the_magnitudes_match_the_reference() {
    let Some(g) = common::golden() else { return };
    let (wave, _) = g.f32("click120_44.wave");
    let (want, shape) = g.f64("parts.stft_mag64");
    assert_eq!(shape, [mel::frames(wave.len()), BINS]);
    let got = mel::magnitudes(&wave).unwrap();
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let gap = common::max_abs(&got, &want);
    assert!(gap <= 1e-12 * peak, "magnitude gap {gap:e}, peak {peak:e}");
}

/// Each value is the f64 log-mel rounded once to f32. Thus it is within one f32 step of the f64 reference.
#[test]
fn the_log_mel_matches_the_reference_in_each_case() {
    let Some(g) = common::golden() else { return };
    for case in g.case_names() {
        let (wave, _) = g.f32(&format!("{case}.wave"));
        let (want, shape) = g.f64(&format!("{case}.spect64"));
        assert_eq!(shape, [mel::frames(wave.len()), MELS], "{case}");
        let got = mel::log_mel(&wave).unwrap();
        let mut worst = 0.0f64;
        for (i, (&v, &w)) in got.iter().zip(&want).enumerate() {
            let step = f64::from((w as f32).abs().max(f32::MIN_POSITIVE)) * f64::from(f32::EPSILON);
            let steps = (f64::from(v) - w).abs() / step;
            worst = worst.max(steps);
            assert!(
                steps <= 1.0,
                "{case}: value {i} is {v}, the reference {w}: {steps} f32 steps"
            );
        }
        eprintln!("{case}: at most {worst:.3} f32 steps from the f64 reference");
    }
}
