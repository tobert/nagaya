use std::fmt;

/// What can fail in this library. Each message says what failed and what the caller can do.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The bytes are not a WAV file that `hound` can read.
    Wav(String),
    /// The WAV file is not 22,050 Hz mono. The error names both values, so one reply tells the client what to fix.
    Layout { rate: u32, channels: u16 },
    /// The WAV file has a sample format other than 16-bit integers or 32-bit floats.
    Format { bits: u16, float: bool },
    /// The input has too few samples for the reflect padding of the spectrum.
    TooShort { samples: usize, minimum: usize },
    /// A tensor of the checkpoint is missing, not f32, or not of the shape that the plan reads.
    Tensor { name: String, why: String },
    /// The hyper_parameters of a checkpoint are missing, not JSON, or name a model that this family does not build.
    Config(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Wav(m) => write!(f, "the audio is not a readable WAV file: {m}. Send a RIFF WAV file."),
            Error::Layout { rate, channels } => write!(
                f,
                "the audio is {rate} Hz with {channels} channels, and Beat This! takes 22050 Hz mono only. \
                 Resample and mix it down on the client."
            ),
            Error::Format { bits, float } => write!(
                f,
                "the audio has {bits}-bit {} samples, and Beat This! takes 16-bit integer or 32-bit float samples. \
                 Convert it on the client.",
                if *float { "float" } else { "integer" }
            ),
            Error::Tensor { name, why } => write!(
                f,
                "the checkpoint's tensor {name} is not usable: {why}. Convert a Beat This! checkpoint with \
                 tools/beatthis_golden.sh."
            ),
            Error::Config(m) => write!(
                f,
                "the checkpoint's hyper_parameters are not usable: {m}. Convert a Beat This! checkpoint with \
                 tools/beatthis_golden.sh."
            ),
            Error::TooShort { samples, minimum } => write!(
                f,
                "the audio has {samples} samples, and the spectrum needs at least {minimum} (about 23 ms). Send a \
                 longer clip."
            ),
        }
    }
}

impl std::error::Error for Error {}
