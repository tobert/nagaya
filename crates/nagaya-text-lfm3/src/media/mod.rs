//! The media preprocessing of `LFM3-Omni-2B` on the CPU (spec 11.11, "The media of a request").
//!
//! An image becomes patch rows, and a clip becomes mel rows, as the reference processor gives them. The engine reads
//! the rows in f32 and runs the encoder of each media part on the card (engine-abi "LFM3", Inputs). The tests compare
//! each stage with the reference (`tools/lfm3_media_fixture.py`, `tools/lfm3_reference.py`).

pub mod audio;
#[cfg(feature = "video")]
pub mod decode;
pub mod image;
pub mod video;

use std::fmt;

pub use audio::AudioRows;
pub use image::ImageRows;

/// Why the bytes of a media part are not media that the family reads. The text completes a sentence that starts
/// with the name of the part, and it says what the client can do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MediaError {
    /// The bytes are not a PNG or a JPEG that the decoder reads. The text says what the decoder found.
    Image(String),
    /// The PNG has this count of bits in a sample, more than 8.
    Depth(u8),
    /// The long side of the image is more than 200 times the short side.
    Ratio { height: usize, width: usize },
    /// The bytes are not a WAV of PCM in 16 bits, mono, at 16 kHz. The text says what the file holds.
    Wav(String),
    /// The media type is not one that the family reads. This is a defect of the caller.
    Mime(String),
    /// The bytes are not a video that the server can read. The text says what the decoder found.
    Video(String),
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaError::Image(why) => write!(
                f,
                "is not an image that the server can read ({why}). Send a PNG or a JPEG"
            ),
            MediaError::Depth(bits) => write!(
                f,
                "is a PNG with {bits} bits in a sample. Send a PNG with 8 bits or fewer in each channel"
            ),
            MediaError::Ratio { height, width } => write!(
                f,
                "is an image of {height} by {width} pixels. Send an image whose long side is at most 200 times its short side"
            ),
            MediaError::Wav(why) => write!(
                f,
                "is not a WAV that the server can read: {why}. Send a WAV of PCM in 16 bits, mono, at 16 kHz"
            ),
            MediaError::Mime(mime) => write!(f, "has the media type {mime}, which the LFM3 family does not read"),
            MediaError::Video(why) => write!(
                f,
                "is not a video that the server can read: {why}. Send an MP4 or a WebM with a video stream"
            ),
        }
    }
}

impl std::error::Error for MediaError {}

/// The prepared rows of one media part.
#[derive(Clone, Debug, PartialEq)]
pub enum MediaRows {
    Image(ImageRows),
    Audio(AudioRows),
}

impl MediaRows {
    /// The positions of the part.
    pub fn positions(&self) -> usize {
        match self {
            MediaRows::Image(rows) => rows.positions(),
            MediaRows::Audio(rows) => rows.positions(),
        }
    }

    /// The values of one row: [`image::ROW`] for an image, [`audio::BINS`] for a clip.
    pub fn width(&self) -> usize {
        match self {
            MediaRows::Image(_) => image::ROW,
            MediaRows::Audio(_) => audio::BINS,
        }
    }

    /// The rows, one after the other, in f32.
    pub fn values(&self) -> &[f32] {
        match self {
            MediaRows::Image(rows) => &rows.values,
            MediaRows::Audio(rows) => &rows.values,
        }
    }
}

/// The positions of a media part of the media type `mime`. An image needs only its header. A clip needs its WAV
/// header and the size of its data.
pub fn positions(bytes: &[u8], mime: &str) -> Result<usize, MediaError> {
    match mime {
        "image/png" | "image/jpeg" => {
            let (height, width) = image::size(bytes, mime)?;
            image::positions(height, width)
        }
        "audio/wav" => Ok(audio::positions(audio::frames(audio::read_wav(bytes)?.len()))),
        other => Err(MediaError::Mime(other.to_string())),
    }
}

/// The rows of a media part of the media type `mime`.
pub fn prepare(bytes: &[u8], mime: &str) -> Result<MediaRows, MediaError> {
    match mime {
        "image/png" | "image/jpeg" => Ok(MediaRows::Image(image::prepare(bytes, mime)?)),
        "audio/wav" => Ok(MediaRows::Audio(audio::prepare(bytes)?)),
        other => Err(MediaError::Mime(other.to_string())),
    }
}
