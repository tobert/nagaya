//! The WAV reader. Beat This! takes 22,050 Hz mono. The family refuses other audio and does not resample, so the
//! client owns the resampler (Amy, 2026-11-26: kaijutsu does its own transcoding for now).

use crate::Error;

/// The sample rate of the model.
pub const SAMPLE_RATE: u32 = 22_050;

/// Reads a WAV file of 22,050 Hz mono into samples in [-1, 1). A 16-bit sample is divided by 32,768, as torchaudio
/// loads it. A 32-bit float sample is kept as it is.
pub fn read_wav(bytes: &[u8]) -> Result<Vec<f32>, Error> {
    let reader = hound::WavReader::new(bytes).map_err(|e| Error::Wav(e.to_string()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        return Err(Error::Layout {
            rate: spec.sample_rate,
            channels: spec.channels,
        });
    }
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .into_samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / 32_768.0))
            .collect::<Result<_, _>>()
            .map_err(|e| Error::Wav(e.to_string())),
        (hound::SampleFormat::Float, 32) => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| Error::Wav(e.to_string())),
        (format, bits) => Err(Error::Format {
            bits,
            float: format == hound::SampleFormat::Float,
        }),
    }
}
