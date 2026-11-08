//! The clip of a media part (spec 11.11): the WAV and the mel features of the feature extractor of `LFM3-Omni-2B`
//! (`feature_extraction_lfm3_omni.py`, transformers 5.20.2).
//!
//! The reference computes in f32 with torch: a periodic Hann window, `torch.stft` with no centering, the power, the
//! slaney mel filters, `log10`, a floor of -12, and `(x + 4) / 4`. Its FFT and its matrix product are not exact, so
//! this module computes the spectrum and the mel sums in f64 and rounds once to f32. The window and the filters are
//! the f32 values of the reference.

use std::f64::consts::PI;
use std::sync::OnceLock;

use super::MediaError;

/// The sample rate of a clip.
pub const RATE: u32 = 16_000;
/// The window of one frame in samples.
pub const N_FFT: usize = 400;
/// The step between frames in samples.
pub const HOP: usize = 160;
/// The frequency bins of one frame's spectrum.
pub const FREQUENCIES: usize = N_FFT / 2 + 1;
/// The mel bins of one frame.
pub const BINS: usize = 128;
/// The frames of one position: the audio encoder reduces the frames by 8.
pub const STRIDE: usize = 8;
/// The floor of the log mel values.
const LOG_FLOOR: f64 = -12.0;
/// The offset and the divisor of the normalization.
const OFFSET: f64 = 4.0;
const DIVISOR: f64 = 4.0;

/// The prepared rows of a clip: one row of [`BINS`] mel values for each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioRows {
    pub frames: usize,
    /// The mel rows, `frames` rows of [`BINS`] values. The reference holds the transpose, bins first.
    pub values: Vec<f32>,
}

impl AudioRows {
    /// The positions of the clip.
    pub fn positions(&self) -> usize {
        positions(self.frames)
    }
}

/// The samples of a clip after the padding: at least one window, and whole steps after it (`pad_length`).
pub fn padded(samples: usize) -> usize {
    if samples < N_FFT {
        return N_FFT;
    }
    samples + (HOP - (samples - N_FFT) % HOP) % HOP
}

/// The mel frames of a clip of `samples` samples (`num_frames`).
pub fn frames(samples: usize) -> usize {
    1 + (padded(samples) - N_FFT) / HOP
}

/// The positions of `frames` mel frames: `ceil(frames / 8)` (`num_encoder_tokens`).
pub fn positions(frames: usize) -> usize {
    frames.div_ceil(STRIDE)
}

/// Read the little-endian integer of `n` bytes at `at`.
fn le(bytes: &[u8], at: usize, n: usize) -> u32 {
    bytes[at..at + n].iter().rev().fold(0, |v, &b| (v << 8) | u32::from(b))
}

/// The sub-format GUID of PCM in a `WAVE_FORMAT_EXTENSIBLE` header, after its first two bytes.
const PCM_GUID_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// The PCM samples of a WAV that is PCM in 16 bits, mono, at 16 kHz. Any other WAV is an error that says what the
/// file holds. This is the rule of Python's `wave` module, which the reference reads the clip with.
pub fn read_wav(bytes: &[u8]) -> Result<Vec<i16>, MediaError> {
    let bad = |why: &str| MediaError::Wav(why.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("the file does not start with a RIFF WAVE header"));
    }
    let mut at = 12;
    let mut format = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = le(bytes, at + 4, 4) as usize;
        let body = at + 8;
        let end = body.checked_add(size).filter(|&e| e <= bytes.len());
        match id {
            b"fmt " => {
                end.ok_or_else(|| bad("the fmt chunk is longer than the file"))?;
                if size < 16 {
                    return Err(bad("the fmt chunk is shorter than 16 bytes"));
                }
                let tag = le(bytes, body, 2);
                let channels = le(bytes, body + 2, 2);
                let rate = le(bytes, body + 4, 4);
                let bits = le(bytes, body + 14, 2);
                let pcm = match tag {
                    1 => true,
                    0xFFFE => {
                        size >= 40 && le(bytes, body + 24, 2) == 1 && bytes[body + 26..body + 40] == PCM_GUID_TAIL
                    }
                    _ => false,
                };
                if !pcm {
                    return Err(MediaError::Wav(format!(
                        "the WAV has the format tag {tag:#06x}, not PCM"
                    )));
                }
                if channels != 1 || bits != 16 || rate != RATE {
                    return Err(MediaError::Wav(format!(
                        "the WAV has {channels} channels of {bits} bits at {rate} Hz"
                    )));
                }
                format = Some(());
            }
            b"data" => {
                if format.is_none() {
                    return Err(bad("the data chunk comes before the fmt chunk"));
                }
                // Python's `wave` reads the frames that the file holds when the size is longer than the file.
                let end = end.unwrap_or(bytes.len());
                let data = &bytes[body..end];
                if !data.len().is_multiple_of(2) {
                    return Err(bad("the data chunk holds an odd count of bytes"));
                }
                return Ok(data.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect());
            }
            _ => {}
        }
        // A chunk of an odd size has a pad byte.
        at = body.saturating_add(size).saturating_add(size % 2);
    }
    Err(if format.is_none() {
        bad("the WAV has no fmt chunk")
    } else {
        bad("the WAV has no data chunk")
    })
}

/// The samples of a WAV at 16 kHz mono, and true when the resampler made them. A WAV of PCM in 16 bits, mono, at 16
/// kHz reads as before, so its rows do not change. In a build with the feature `video`, any other WAV that ffmpeg
/// reads resamples to 16 kHz mono (spec 11.11, "Video, and audio at any rate"). Without it, such a WAV is refused.
pub fn clip_samples(bytes: &[u8]) -> Result<(Vec<f32>, bool), MediaError> {
    match read_wav(bytes) {
        Ok(pcm) => Ok((samples(&pcm), false)),
        #[cfg(feature = "video")]
        Err(strict) => super::decode::audio(bytes).map(|s| (s, true)).map_err(|_| strict),
        #[cfg(not(feature = "video"))]
        Err(strict) => Err(strict),
    }
}

/// The samples of a clip in f32: each PCM sample divided by 32768, as the reference does.
pub fn samples(pcm: &[i16]) -> Vec<f32> {
    pcm.iter().map(|&v| f32::from(v) / 32768.0).collect()
}

/// The periodic Hann window of 400 samples in f32, as `torch.hann_window` computes it: `n * (2 pi / 400)`, the
/// cosine, times -0.5, plus 0.5, each step in f32.
pub fn window() -> &'static [f32; N_FFT] {
    static WINDOW: OnceLock<[f32; N_FFT]> = OnceLock::new();
    WINDOW.get_or_init(|| {
        let step = (2.0 * PI / N_FFT as f64) as f32;
        std::array::from_fn(|n| (n as f32 * step).cos() * -0.5 + 0.5)
    })
}

/// The slaney mel scale of a frequency in Hz (`hertz_to_mel`).
fn hertz_to_mel(freq: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * logstep
    } else {
        3.0 * freq / 200.0
    }
}

/// The frequency in Hz of a value of the slaney mel scale (`mel_to_hertz`).
fn mel_to_hertz(mel: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// numpy's `linspace(start, stop, n)`: `i * step + start`, with the last value set to `stop`.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n - 1) as f64;
    let mut out: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    out[n - 1] = stop;
    out
}

/// The slaney mel filters with slaney norms, [`FREQUENCIES`] rows of [`BINS`] values, in f32. This is
/// `mel_filter_bank` of transformers in f64, cast to f32 as the reference casts it.
pub fn mel_filters() -> &'static [f32] {
    static FILTERS: OnceLock<Vec<f32>> = OnceLock::new();
    FILTERS.get_or_init(|| {
        let mels = linspace(hertz_to_mel(0.0), hertz_to_mel(f64::from(RATE) / 2.0), BINS + 2);
        let edges: Vec<f64> = mels.iter().map(|&m| mel_to_hertz(m)).collect();
        let fft = linspace(0.0, f64::from(RATE / 2), FREQUENCIES);
        let diff: Vec<f64> = edges.windows(2).map(|p| p[1] - p[0]).collect();
        let mut out = Vec::with_capacity(FREQUENCIES * BINS);
        for &f in &fft {
            for m in 0..BINS {
                let down = -(edges[m] - f) / diff[m];
                let up = (edges[m + 2] - f) / diff[m + 1];
                let enorm = 2.0 / (edges[m + 2] - edges[m]);
                out.push((0.0f64.max(down.min(up)) * enorm) as f32);
            }
        }
        out
    })
}

/// The cosine and the sine of `2 pi m / 400` for each `m` below 400.
fn twiddles() -> &'static [(f64, f64); N_FFT] {
    static TABLE: OnceLock<[(f64, f64); N_FFT]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|m| {
            let a = 2.0 * PI * m as f64 / N_FFT as f64;
            (a.cos(), a.sin())
        })
    })
}

/// The mel rows of a clip of f32 samples: the padding with zeros, then one row of [`BINS`] values for each frame.
pub fn mel(samples: &[f32]) -> AudioRows {
    let count = frames(samples.len());
    let mut clip = samples.to_vec();
    clip.resize(padded(samples.len()), 0.0);
    let window = window();
    let filters = mel_filters();
    let table = twiddles();
    let mut values = Vec::with_capacity(count * BINS);
    let mut frame = [0.0f64; N_FFT];
    let mut power = [0.0f64; FREQUENCIES];
    for t in 0..count {
        let start = t * HOP;
        for (n, v) in frame.iter_mut().enumerate() {
            // torch multiplies the frame by the window in f32.
            *v = f64::from(clip[start + n] * window[n]);
        }
        for (k, p) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, &x) in frame.iter().enumerate() {
                let (c, s) = table[(k * n) % N_FFT];
                re += x * c;
                im -= x * s;
            }
            *p = re * re + im * im;
        }
        for m in 0..BINS {
            let sum: f64 = (0..FREQUENCIES)
                .map(|f| f64::from(filters[f * BINS + m]) * power[f])
                .sum();
            let log = sum.log10().max(LOG_FLOOR);
            values.push(((log + OFFSET) / DIVISOR) as f32);
        }
    }
    AudioRows { frames: count, values }
}

/// Read a WAV and compute its mel rows.
pub fn prepare(bytes: &[u8]) -> Result<AudioRows, MediaError> {
    Ok(mel(&clip_samples(bytes)?.0))
}
