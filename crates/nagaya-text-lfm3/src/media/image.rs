//! The image of a media part (spec 11.11): the decode, the resize and the patches of the processor of
//! `LFM3-Omni-2B` (`image_processing_lfm3_vl.py`, transformers 5.20.2, torchvision 0.29.2).
//!
//! The processor resizes the RGB bytes with torchvision's bicubic filter with antialias, from u8 to u8. On the CPU,
//! torch does this in integer arithmetic with weights in 16 bits (`_compute_index_ranges_int16_weights` of
//! `UpSampleKernel.cpp`). This module does the same arithmetic, so the resized bytes are the same bits.

use std::io::Cursor;

use super::MediaError;

/// The side of a patch in pixels.
pub const PATCH: usize = 16;
/// The side of a merge block in patches. One position holds a block of 2 by 2 patches.
pub const MERGE: usize = 2;
/// The copies of an image in time. The vision tower reads an image as 2 equal frames.
pub const TEMPORAL: usize = 2;
/// The resized sides are multiples of this.
pub const FACTOR: usize = PATCH * MERGE;
/// The bounds of the resized pixel count (`processor_config.json`, `size`).
pub const MIN_PIXELS: usize = 65_536;
pub const MAX_PIXELS: usize = 2_097_152;
/// The values of one patch row: 3 channels, 2 copies in time, 16 by 16 pixels.
pub const ROW: usize = 3 * TEMPORAL * PATCH * PATCH;
/// The largest ratio of the long side to the short side that `smart_resize` takes.
const MAX_RATIO: f64 = 200.0;

/// An image as RGB bytes, 3 for each pixel, row by row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub height: usize,
    pub width: usize,
    pub bytes: Vec<u8>,
}

/// The prepared rows of an image: one row of [`ROW`] values for each patch, in the order of the merge blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRows {
    /// `grid_t`, `grid_h` and `grid_w`: the patches in time, down and across.
    pub grid: [usize; 3],
    /// The patch rows, `grid_t * grid_h * grid_w` rows of [`ROW`] values.
    pub values: Vec<f32>,
}

impl ImageRows {
    /// The positions of the image: one for each merge block.
    pub fn positions(&self) -> usize {
        self.grid.iter().product::<usize>() / (MERGE * MERGE)
    }
}

/// The size of a PNG or a JPEG from its header, as `(height, width)`. The pixels are not decoded.
pub fn size(bytes: &[u8], mime: &str) -> Result<(usize, usize), MediaError> {
    match mime {
        "image/png" => {
            let reader = png_reader(bytes)?;
            let info = reader.info();
            Ok((info.height as usize, info.width as usize))
        }
        "image/jpeg" => {
            let mut decoder = jpeg_decoder(bytes);
            decoder.decode_headers().map_err(|e| MediaError::Image(e.to_string()))?;
            let (width, height) = decoder
                .dimensions()
                .ok_or_else(|| MediaError::Image("the JPEG has no frame header".to_string()))?;
            Ok((height, width))
        }
        other => Err(MediaError::Mime(other.to_string())),
    }
}

/// The positions of an image of `height` by `width` pixels.
pub fn positions(height: usize, width: usize) -> Result<usize, MediaError> {
    let (h, w) = smart_resize(height, width)?;
    Ok((h / PATCH) * (w / PATCH) / (MERGE * MERGE))
}

/// Decode, resize and cut an image into its patch rows, as the processor does.
pub fn prepare(bytes: &[u8], mime: &str) -> Result<ImageRows, MediaError> {
    let rgb = decode(bytes, mime)?;
    let (height, width) = smart_resize(rgb.height, rgb.width)?;
    let resized = resize(&rgb, height, width);
    Ok(ImageRows {
        grid: [1, height / PATCH, width / PATCH],
        values: patches(&resized),
    })
}

fn png_reader(bytes: &[u8]) -> Result<png::Reader<Cursor<&[u8]>>, MediaError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    // EXPAND maps a palette to RGB and scales a gray of 1, 2 or 4 bits to 8 bits, as Pillow does. It also adds an
    // alpha channel for `tRNS`, which the convert to RGB then drops, as Pillow's `convert("RGB")` does. STRIP_16
    // keeps the high byte of a sample of 16 bits (spec 11.11, "Depth").
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.read_info().map_err(|e| MediaError::Image(e.to_string()))
}

fn jpeg_decoder(bytes: &[u8]) -> zune_jpeg::JpegDecoder<zune_jpeg::zune_core::bytestream::ZCursor<&[u8]>> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    zune_jpeg::JpegDecoder::new_with_options(zune_jpeg::zune_core::bytestream::ZCursor::new(bytes), options)
}

/// Decode a PNG or a JPEG to RGB in 8 bits. An alpha channel has no effect, as in Pillow's `convert("RGB")`.
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
    let (height, width) = (frame.height as usize, frame.width as usize);
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
        height * width * channels,
        "the PNG decoder gave {} samples for {height} by {width} pixels of {channels} channels",
        samples.len()
    );
    let bytes = match channels {
        1 => samples.iter().flat_map(|&v| [v, v, v]).collect(),
        3 => samples,
        n => samples
            .chunks_exact(n)
            .flat_map(|p| if n == 2 { [p[0], p[0], p[0]] } else { [p[0], p[1], p[2]] })
            .collect(),
    };
    Ok(Rgb { height, width, bytes })
}

fn decode_jpeg(bytes: &[u8]) -> Result<Rgb, MediaError> {
    let mut decoder = jpeg_decoder(bytes);
    let pixels = decoder.decode().map_err(|e| MediaError::Image(e.to_string()))?;
    let (width, height) = decoder.dimensions().expect("a decoded JPEG has its dimensions");
    assert_eq!(
        pixels.len(),
        height * width * 3,
        "the JPEG decoder gave {} bytes for {height} by {width} RGB pixels",
        pixels.len()
    );
    Ok(Rgb {
        height,
        width,
        bytes: pixels,
    })
}

/// Python's `round` of a float: the nearest integer, with a half to the even integer.
fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// The resized size of an image, as `smart_resize` of the processor (`image_processing_lfm3_vl.py:50`). Both sides
/// become multiples of [`FACTOR`], and the pixel count stays in the bounds. The float steps are Python's.
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize), MediaError> {
    if height == 0 || width == 0 {
        return Err(MediaError::Image(format!("the image has {height} by {width} pixels")));
    }
    let (h, w, f) = (height as f64, width as f64, FACTOR as f64);
    if h.max(w) / h.min(w) > MAX_RATIO {
        return Err(MediaError::Ratio { height, width });
    }
    let mut h_bar = round_half_even(h / f) * f;
    let mut w_bar = round_half_even(w / f) * f;
    if h_bar * w_bar > MAX_PIXELS as f64 {
        let beta = ((h * w) / MAX_PIXELS as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// The bicubic filter with `a = -0.5`, which torch uses with antialias (`HelperInterpCubic::aa_filter`).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// The taps of a resize along one side: for each output pixel, the first input pixel and the integer weights.
struct Taps {
    first: Vec<usize>,
    weights: Vec<Vec<i32>>,
    /// The weights hold this many bits after the binary point.
    precision: u32,
}

/// The taps of a resize from `input` to `output` pixels. This is torch's `_compute_indices_min_size_weights_aa`
/// and `_compute_index_ranges_int16_weights`, with the same f64 steps and the same rounding.
fn taps(input: usize, output: usize) -> Taps {
    let scale = input as f64 / output as f64;
    // The bicubic filter has 4 taps at scale 1.
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_size = (support.ceil() as i64) * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut first = Vec::with_capacity(output);
    let mut real = Vec::with_capacity(output);
    let mut max = 0.0f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        // C casts a double to an integer toward zero, as `as` does.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = (((center + support + 0.5) as i64).min(input as i64) - xmin).clamp(0, max_size);
        let mut w: Vec<f64> = (0..xsize)
            .map(|j| cubic(((j + xmin) as f64 - center + 0.5) * invscale))
            .collect();
        let total: f64 = w.iter().fold(0.0, |s, &v| s + v);
        if total != 0.0 {
            for v in &mut w {
                *v /= total;
                max = max.max(*v);
            }
        }
        first.push(xmin as usize);
        real.push(w);
    }
    let mut precision = 0u32;
    while precision < 22 {
        if (0.5 + max * f64::from(1u32 << (precision + 1))) as i32 >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    assert!(precision >= 1, "the resize weights have no bits of precision");
    let unit = f64::from(1u32 << precision);
    let weights = real
        .into_iter()
        .map(|w| {
            w.into_iter()
                .map(|v| {
                    let v = v * unit;
                    if v < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 }
                })
                .collect()
        })
        .collect();
    Taps {
        first,
        weights,
        precision,
    }
}

/// One weighted sum in torch's integer form: a half for the rounding, the shift, and the clamp to a byte.
fn tap_sum(precision: u32, terms: impl Iterator<Item = (u8, i32)>) -> u8 {
    let sum = terms.fold(1i32 << (precision - 1), |s, (v, w)| s + i32::from(v) * w);
    (sum >> precision).clamp(0, 255) as u8
}

/// Resize an image with torchvision's bicubic filter with antialias, from u8 to u8. The resize goes across first,
/// then down, and it skips a side whose size does not change, as torch does.
pub fn resize(rgb: &Rgb, height: usize, width: usize) -> Rgb {
    let mut image = rgb.clone();
    if width != image.width {
        let t = taps(image.width, width);
        let mut out = Vec::with_capacity(image.height * width * 3);
        for row in image.bytes.chunks_exact(image.width * 3) {
            for (x, w) in t.weights.iter().enumerate() {
                let x0 = t.first[x];
                for c in 0..3 {
                    out.push(tap_sum(
                        t.precision,
                        w.iter().enumerate().map(|(k, &wk)| (row[(x0 + k) * 3 + c], wk)),
                    ));
                }
            }
        }
        image = Rgb {
            height: image.height,
            width,
            bytes: out,
        };
    }
    if height != image.height {
        let t = taps(image.height, height);
        let stride = image.width * 3;
        let mut out = Vec::with_capacity(height * stride);
        for (y, w) in t.weights.iter().enumerate() {
            let y0 = t.first[y];
            for i in 0..stride {
                out.push(tap_sum(
                    t.precision,
                    w.iter()
                        .enumerate()
                        .map(|(k, &wk)| (image.bytes[(y0 + k) * stride + i], wk)),
                ));
            }
        }
        image = Rgb {
            height,
            width: image.width,
            bytes: out,
        };
    }
    image
}

/// The patch rows of a resized image (`patchify` of the processor). Each value is `(x - 127.5) / 127.5` in f32: the
/// processor folds the rescale by 1/255 into the mean and the std of 0.5. A row is the 3 channels, then the 2 copies
/// in time, then 16 rows and 16 columns of pixels. The 4 patches of a merge block are 4 rows in raster order, and
/// the blocks go in raster order too.
pub fn patches(rgb: &Rgb) -> Vec<f32> {
    assert!(
        rgb.height.is_multiple_of(FACTOR) && rgb.width.is_multiple_of(FACTOR),
        "the patches need sides that are multiples of {FACTOR}, not {} by {}",
        rgb.height,
        rgb.width
    );
    let (gh, gw) = (rgb.height / PATCH, rgb.width / PATCH);
    let mut out = Vec::with_capacity(gh * gw * ROW);
    for by in 0..gh / MERGE {
        for bx in 0..gw / MERGE {
            for my in 0..MERGE {
                for mx in 0..MERGE {
                    let (py, px) = ((by * MERGE + my) * PATCH, (bx * MERGE + mx) * PATCH);
                    for c in 0..3 {
                        for _ in 0..TEMPORAL {
                            for iy in 0..PATCH {
                                for ix in 0..PATCH {
                                    let v = rgb.bytes[((py + iy) * rgb.width + px + ix) * 3 + c];
                                    out.push((f32::from(v) - 127.5) / 127.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
