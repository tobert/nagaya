//! The images of a media part on the CPU (engine-abi "The images of Inkling 2"): the decode, the convert of an
//! alpha channel, and the patch rows that the engine reads.
//!
//! The rules of the processor are those of the engine crate's host form (`nagaya_mk_inkling2_bmg::host::vision`),
//! which the oracle checks. This module adds only the decode of the bytes. The tests compare each stage with the
//! reference processor (`tools/inkling2_media_fixture.py`) and with the oracle of the tower
//! (`tools/inkling2_golden.sh`).

use std::fmt;
use std::io::Cursor;

pub use nagaya_mk_inkling2_bmg::host::vision::{self, Grid, IMAGE_TOKEN, PATCH_VALUES};

/// The media types that the family reads.
pub const MEDIA_INPUTS: [&str; 2] = ["image/png", "image/jpeg"];

/// The bytes of one record of `meta`, as in "The LFM3 vision binding": the kind, the count of positions, the first
/// position as a `u64`, and four words of the shape.
pub const RECORD_BYTES: usize = 32;
/// The kind of an image in a record.
pub const IMAGE_KIND: u32 = 1;

/// Why the bytes of a media part are not an image that the family reads. The text completes a sentence that starts
/// with the name of the part, and it says what the client can do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MediaError {
    /// The bytes are not a PNG or a JPEG that the decoder reads. The text says what the decoder found.
    Image(String),
    /// The size of the image is outside the rule of the model card. The text gives the size and the rule.
    Size(String),
    /// The media type is not one that the family reads. This is a defect of the caller.
    Mime(String),
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaError::Image(why) => write!(
                f,
                "is not an image that the server can read ({why}). Send a PNG or a JPEG"
            ),
            MediaError::Size(why) => write!(f, "is refused: {why}"),
            MediaError::Mime(mime) => write!(f, "has the media type {mime}, which the Inkling 2 family does not read"),
        }
    }
}

impl std::error::Error for MediaError {}

/// An image as RGB bytes after the convert step of the processor, 3 for each pixel, line by line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub bytes: Vec<u8>,
}

/// The prepared rows of an image: one row of [`PATCH_VALUES`] values for each patch, in the order of the merge.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRows {
    pub grid: Grid,
    pub values: Vec<f32>,
}

impl ImageRows {
    /// The positions of the image: one for each image token.
    pub fn positions(&self) -> usize {
        self.grid.tokens()
    }

    /// The rows for the engine, little-endian `f32`.
    pub fn bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The record of the span in `meta`, from position `first` (engine-abi "The meta of a span").
    pub fn record(&self, first: usize) -> Vec<u8> {
        let word = |value: usize| u32::try_from(value).expect("a span of an image fits in 32 bits");
        let [t, h, w] = self.grid.meta_grid();
        let mut record = Vec::with_capacity(RECORD_BYTES);
        record.extend(IMAGE_KIND.to_le_bytes());
        record.extend(word(self.positions()).to_le_bytes());
        record.extend((first as u64).to_le_bytes());
        for value in [t, h, w, 0] {
            record.extend(value.to_le_bytes());
        }
        record
    }
}

fn png_reader(bytes: &[u8]) -> Result<png::Reader<Cursor<&[u8]>>, MediaError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    // EXPAND maps a palette to RGB, adds an alpha channel for `tRNS`, and scales a gray of 1, 2 or 4 bits to 8 bits,
    // as Pillow's convert to RGBA does. STRIP_16 keeps the high byte of a sample of 16 bits (spec 11.11, "Depth").
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.read_info().map_err(|e| MediaError::Image(e.to_string()))
}

fn jpeg_decoder(bytes: &[u8]) -> zune_jpeg::JpegDecoder<zune_jpeg::zune_core::bytestream::ZCursor<&[u8]>> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    zune_jpeg::JpegDecoder::new_with_options(zune_jpeg::zune_core::bytestream::ZCursor::new(bytes), options)
}

/// The size of a PNG or a JPEG from its header, as `(width, height)`. The pixels are not decoded.
pub fn size(bytes: &[u8], mime: &str) -> Result<(usize, usize), MediaError> {
    match mime {
        "image/png" => {
            let reader = png_reader(bytes)?;
            let info = reader.info();
            Ok((info.width as usize, info.height as usize))
        }
        "image/jpeg" => {
            let mut decoder = jpeg_decoder(bytes);
            decoder.decode_headers().map_err(|e| MediaError::Image(e.to_string()))?;
            let (width, height) = decoder
                .dimensions()
                .ok_or_else(|| MediaError::Image("the JPEG has no frame header".to_string()))?;
            Ok((width, height))
        }
        other => Err(MediaError::Mime(other.to_string())),
    }
}

/// The grid of an image from its header. A size outside the rule is refused before any pixel is decoded.
pub fn grid(bytes: &[u8], mime: &str) -> Result<Grid, MediaError> {
    let (width, height) = size(bytes, mime)?;
    vision::grid(width, height).map_err(MediaError::Size)
}

/// Decode a PNG or a JPEG, and convert it to RGB as the processor does: an image with alpha goes over the
/// checkerboard (`rgba_to_rgb`), and one with no alpha keeps its RGB.
pub fn decode(bytes: &[u8], mime: &str) -> Result<Rgb, MediaError> {
    match mime {
        "image/png" => decode_png(bytes),
        "image/jpeg" => decode_jpeg(bytes),
        other => Err(MediaError::Mime(other.to_string())),
    }
}

fn decode_png(bytes: &[u8]) -> Result<Rgb, MediaError> {
    let mut reader = png_reader(bytes)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| MediaError::Image("the image is too large to decode".to_string()))?;
    let mut samples = vec![0; size];
    let frame = reader
        .next_frame(&mut samples)
        .map_err(|e| MediaError::Image(e.to_string()))?;
    samples.truncate(frame.buffer_size());
    let (width, height) = (frame.width as usize, frame.height as usize);
    let channels = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        // EXPAND maps each palette entry to RGB or RGBA.
        png::ColorType::Indexed => panic!("the PNG decoder gave palette indices after EXPAND"),
    };
    assert_eq!(
        samples.len(),
        width * height * channels,
        "the PNG decoder gave {} samples for {width} by {height} pixels of {channels} channels",
        samples.len()
    );
    let bytes = match channels {
        1 => samples.iter().flat_map(|&v| [v, v, v]).collect(),
        3 => samples,
        2 => {
            let rgba: Vec<u8> = samples
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|[g, a]| [*g, *g, *g, *a])
                .collect();
            vision::rgba_to_rgb(&rgba, width, height)
        }
        _ => vision::rgba_to_rgb(&samples, width, height),
    };
    Ok(Rgb { width, height, bytes })
}

fn decode_jpeg(bytes: &[u8]) -> Result<Rgb, MediaError> {
    let mut decoder = jpeg_decoder(bytes);
    let pixels = decoder.decode().map_err(|e| MediaError::Image(e.to_string()))?;
    let (width, height) = decoder.dimensions().expect("a decoded JPEG has its dimensions");
    assert_eq!(
        pixels.len(),
        height * width * 3,
        "the JPEG decoder gave {} bytes for {width} by {height} RGB pixels",
        pixels.len()
    );
    Ok(Rgb {
        width,
        height,
        bytes: pixels,
    })
}

/// The rows of RGB bytes: the pixel values of the processor, then the patch rows.
pub fn rows_of(rgb: &Rgb) -> Result<ImageRows, MediaError> {
    let (grid, pixels) = vision::pixel_values(&rgb.bytes, rgb.width, rgb.height).map_err(MediaError::Size)?;
    Ok(ImageRows {
        grid,
        values: vision::patch_rows(&pixels, grid),
    })
}

/// Decode an image and cut it into its patch rows, as the processor does. A render checks the size rule on the
/// header first (`grid`), so it decodes no image of a refused size.
pub fn prepare(bytes: &[u8], mime: &str) -> Result<ImageRows, MediaError> {
    rows_of(&decode(bytes, mime)?)
}
