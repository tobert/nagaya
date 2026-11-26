//! The log-mel spectrum of `beat_this.preprocessing.LogMelSpect`.
//!
//! The reference is torchaudio's `MelSpectrogram` in f32: a periodic Hann window of 1024, a hop of 441, reflect
//! padding of 512 at each end, a magnitude spectrum divided by sqrt(1024), 128 Slaney mel filters from 30 Hz to
//! 11 kHz with no area normalization, then `log1p(1000 * mel)`. This module computes in f64 and rounds once to f32,
//! as the LFM3 family does. The gate against the reference is a tolerance; the bits of this module are its own.
//! cos, sin, log, exp and log1p come from the `libm` crate (a pure Rust port of musl), not the machine's libm, so the
//! bits do not change with the machine or an update of its C library.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::Error;
use crate::audio::SAMPLE_RATE;

/// The size of a frame of the spectrum.
pub const N_FFT: usize = 1024;
/// The step between frames: 22,050 / 441 = 50 frames each second.
pub const HOP: usize = 441;
/// The bins of the one-sided spectrum.
pub const BINS: usize = N_FFT / 2 + 1;
/// The mel bands.
pub const MELS: usize = 128;
/// The frames each second.
pub const FPS: u32 = 50;
/// The reflect padding at each end.
pub const PAD: usize = N_FFT / 2;

/// The fewest samples that the reflect padding takes: torch refuses a pad as long as the input.
pub const MIN_SAMPLES: usize = PAD + 1;

/// The frames of an input of `samples` samples: `1 + samples / HOP`, as torch's centered STFT gives.
pub fn frames(samples: usize) -> usize {
    1 + samples / HOP
}

/// The periodic Hann window: `0.5 - 0.5 cos(2 pi n / N_FFT)`.
pub fn window() -> &'static [f64; N_FFT] {
    static TABLE: OnceLock<[f64; N_FFT]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|n| 0.5 - 0.5 * libm::cos(2.0 * PI * n as f64 / N_FFT as f64)))
}

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const F_MIN: f64 = 30.0;
const F_MAX: f64 = 11_000.0;

fn logstep() -> f64 {
    libm::log(6.4) / 27.0
}

/// `torchaudio.functional._hz_to_mel` with `mel_scale="slaney"`.
fn hz_to_mel(hz: f64) -> f64 {
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + libm::log(hz / MIN_LOG_HZ) / logstep()
    } else {
        hz / F_SP
    }
}

/// `torchaudio.functional._mel_to_hz` with `mel_scale="slaney"`.
fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * libm::exp(logstep() * (mel - MIN_LOG_MEL))
    } else {
        F_SP * mel
    }
}

/// `n` values from `start` to `stop`. For the two uses here, the last value is `stop` exactly (the table pins hold it).
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// The mel filters, `BINS` rows of `MELS` values: `torchaudio.functional.melscale_fbanks` with `norm=None`.
pub fn filterbank() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        // torchaudio spaces the bins up to `sample_rate // 2`, an integer.
        let freqs = linspace(0.0, f64::from(SAMPLE_RATE / 2), BINS);
        let points: Vec<f64> = linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), MELS + 2)
            .into_iter()
            .map(mel_to_hz)
            .collect();
        let mut out = Vec::with_capacity(BINS * MELS);
        for &f in &freqs {
            for m in 0..MELS {
                let down = (f - points[m]) / (points[m + 1] - points[m]);
                let up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
                out.push(0.0f64.max(down.min(up)));
            }
        }
        out
    })
}

/// For each mel band, the bins where its filter is above zero, as a range.
fn bands() -> &'static [(usize, usize); MELS] {
    static TABLE: OnceLock<[(usize, usize); MELS]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let fb = filterbank();
        std::array::from_fn(|m| {
            let live: Vec<usize> = (0..BINS).filter(|&f| fb[f * MELS + m] > 0.0).collect();
            match (live.first(), live.last()) {
                (Some(&lo), Some(&hi)) => (lo, hi + 1),
                _ => (0, 0),
            }
        })
    })
}

/// The twiddles `exp(-2 pi i k / N_FFT)` for `k < N_FFT / 2`.
fn twiddles() -> &'static [(f64, f64); N_FFT / 2] {
    static TABLE: OnceLock<[(f64, f64); N_FFT / 2]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|k| {
            let a = 2.0 * PI * k as f64 / N_FFT as f64;
            (libm::cos(a), -libm::sin(a))
        })
    })
}

/// An in-place radix-2 FFT of `N_FFT` points, decimation in time. The order of each sum is fixed by the loops.
fn fft(re: &mut [f64; N_FFT], im: &mut [f64; N_FFT]) {
    let bits = N_FFT.trailing_zeros();
    for i in 0..N_FFT {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let tw = twiddles();
    let mut size = 2;
    while size <= N_FFT {
        let half = size / 2;
        let stride = N_FFT / size;
        for start in (0..N_FFT).step_by(size) {
            for k in 0..half {
                let (wr, wi) = tw[k * stride];
                let (a, b) = (start + k, start + k + half);
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        size *= 2;
    }
}

/// The sample at `index` of the input with reflect padding of `PAD` at each end.
fn reflected(samples: &[f32], index: usize) -> f32 {
    let n = samples.len();
    let i = index as isize - PAD as isize;
    let i = if i < 0 {
        -i
    } else if i >= n as isize {
        2 * (n as isize - 1) - i
    } else {
        i
    };
    samples[i as usize]
}

/// The magnitudes of one frame, divided by sqrt(`N_FFT`) = 32, into `mag`. `re` and `im` are scratch.
fn frame_magnitudes(samples: &[f32], frame: usize, re: &mut [f64; N_FFT], im: &mut [f64; N_FFT], mag: &mut [f64]) {
    let w = window();
    for n in 0..N_FFT {
        re[n] = f64::from(reflected(samples, frame * HOP + n)) * w[n];
        im[n] = 0.0;
    }
    fft(re, im);
    for f in 0..BINS {
        // sqrt is exact to the last bit (IEEE 754); 1/32 is a power of two, so the scale adds no error.
        mag[f] = (re[f] * re[f] + im[f] * im[f]).sqrt() / 32.0;
    }
}

/// The magnitude spectrum, `frames(samples.len())` rows of `BINS` values, divided by sqrt(`N_FFT`) = 32.
pub fn magnitudes(samples: &[f32]) -> Result<Vec<f64>, Error> {
    if samples.len() < MIN_SAMPLES {
        return Err(Error::TooShort {
            samples: samples.len(),
            minimum: MIN_SAMPLES,
        });
    }
    let t = frames(samples.len());
    let mut out = vec![0.0f64; t * BINS];
    let (mut re, mut im) = ([0.0f64; N_FFT], [0.0f64; N_FFT]);
    for (frame, mag) in out.as_chunks_mut::<BINS>().0.iter_mut().enumerate() {
        frame_magnitudes(samples, frame, &mut re, &mut im, mag);
    }
    Ok(out)
}

/// The most threads of one log-mel. Each frame is computed whole by one thread, so the count changes no bit.
const MEL_THREADS: usize = 4;
/// The fewest frames for each thread; a short input runs on the caller's thread alone.
const MEL_FRAMES_PER_THREAD: usize = 128;

/// The log-mel spectrum, `frames(samples.len())` rows of `MELS` values: `log1p(1000 * mel)`, rounded once to f32. The
/// frames are split over up to 4 threads; each frame is computed whole, in the same order, by one thread.
pub fn log_mel(samples: &[f32]) -> Result<Vec<f32>, Error> {
    if samples.len() < MIN_SAMPLES {
        return Err(Error::TooShort {
            samples: samples.len(),
            minimum: MIN_SAMPLES,
        });
    }
    let t = frames(samples.len());
    let mut out = vec![0.0f32; t * MELS];
    let threads = (t / MEL_FRAMES_PER_THREAD).clamp(1, MEL_THREADS);
    let per = t.div_ceil(threads);
    let run = |first: usize, rows: &mut [f32]| {
        let fb = filterbank();
        let bands = bands();
        let (mut re, mut im) = ([0.0f64; N_FFT], [0.0f64; N_FFT]);
        let mut mag = [0.0f64; BINS];
        for (k, row) in rows.as_chunks_mut::<MELS>().0.iter_mut().enumerate() {
            frame_magnitudes(samples, first + k, &mut re, &mut im, &mut mag);
            for (m, &(lo, hi)) in bands.iter().enumerate() {
                // The bins outside the band have a filter of +0, and the sum is never below 0, so they add nothing.
                let mut sum = 0.0f64;
                for f in lo..hi {
                    sum += mag[f] * fb[f * MELS + m];
                }
                row[m] = libm::log1p(1000.0 * sum) as f32;
            }
        }
    };
    if threads == 1 {
        run(0, &mut out);
    } else {
        std::thread::scope(|s| {
            for (n, rows) in out.chunks_mut(per * MELS).enumerate() {
                let run = &run;
                s.spawn(move || run(n * per, rows));
            }
        });
    }
    Ok(out)
}
